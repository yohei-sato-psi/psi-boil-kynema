#include "src/equation_systems/vof/vof_spurious_current.H"

#include <cmath>
#include <limits>
#include "src/core/Field.H"
#include "src/core/FieldRepo.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParReduce.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Reduce.H"

using namespace amrex::literals;

namespace kynema_sgf::multiphase {
namespace spurious_current_impl {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool liquid_cell(
    const amrex::Real phi, const amrex::Real c_liq_cell)
{
    bool flag = false;
    if (phi >= c_liq_cell) {
        flag = true;
    }
    return flag;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool near_interface_cell(
    const amrex::Array4<amrex::Real const>& color,
    const int i,
    const int j,
    const int k,
    const amrex::Real c_liq_cell)
{
    bool near_interface = false;

    if (color(i - 1, j, k) < c_liq_cell) {
        near_interface = true;
    }
    if (color(i + 1, j, k) < c_liq_cell) {
        near_interface = true;
    }
    if (color(i, j - 1, k) < c_liq_cell) {
        near_interface = true;
    }
    if (color(i, j + 1, k) < c_liq_cell) {
        near_interface = true;
    }
    if (color(i, j, k - 1) < c_liq_cell) {
        near_interface = true;
    }
    if (color(i, j, k + 1) < c_liq_cell) {
        near_interface = true;
    }

    return near_interface;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool cell_centered_velocity(
    const amrex::Array4<amrex::Real const>& color,
    const amrex::Array4<amrex::Real const>& u,
    const amrex::Array4<amrex::Real const>& v,
    const amrex::Array4<amrex::Real const>& w,
    const int i,
    const int j,
    const int k,
    const amrex::Real c_liq_cell,
    amrex::Real& uc,
    amrex::Real& vc,
    amrex::Real& wc)
{
    bool flag = true;

    uc = 0.0_rt;
    vc = 0.0_rt;
    wc = 0.0_rt;

    int nuface = 0;
    int nvface = 0;
    int nwface = 0;

    if (color(i - 1, j, k) >= c_liq_cell &&
        color(i, j, k) >= c_liq_cell) {
        uc += u(i, j, k);
        nuface++;
    }
    if (color(i, j, k) >= c_liq_cell &&
        color(i + 1, j, k) >= c_liq_cell) {
        uc += u(i + 1, j, k);
        nuface++;
    }

    if (color(i, j - 1, k) >= c_liq_cell &&
        color(i, j, k) >= c_liq_cell) {
        vc += v(i, j, k);
        nvface++;
    }
    if (color(i, j, k) >= c_liq_cell &&
        color(i, j + 1, k) >= c_liq_cell) {
        vc += v(i, j + 1, k);
        nvface++;
    }

    if (color(i, j, k - 1) >= c_liq_cell &&
        color(i, j, k) >= c_liq_cell) {
        wc += w(i, j, k);
        nwface++;
    }
    if (color(i, j, k) >= c_liq_cell &&
        color(i, j, k + 1) >= c_liq_cell) {
        wc += w(i, j, k + 1);
        nwface++;
    }

    if (nuface == 0 || nvface == 0 || nwface == 0) {
        flag = false;
    }

    if (flag) {
        uc /= static_cast<amrex::Real>(nuface);
        vc /= static_cast<amrex::Real>(nvface);
        wc /= static_cast<amrex::Real>(nwface);
    }

    return flag;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real velocity_mag(
    const amrex::Real u, const amrex::Real v, const amrex::Real w)
{
    return std::sqrt(u * u + v * v + w * w);
}

} // namespace spurious_current_impl

SpuriousCurrentStats spurious_current(
    const Field& phi,
    const Field& u_mac,
    const Field& v_mac,
    const Field& w_mac,
    const SpuriousCurrentParams& params)
{
    BL_PROFILE("kynema-sgf::multiphase::spurious_current");

    SpuriousCurrentStats stats;
    stats.radius = params.radius;

    const int nlevels = phi.repo().num_active_levels();
    const auto& mesh = phi.repo().mesh();

    amrex::Real vol_gamma = 0.0_rt;
    amrex::Real sum_abs_gamma = 0.0_rt;
    amrex::Real sum_sq_gamma = 0.0_rt;
    amrex::Real mean_abs_gamma = 0.0_rt;
    amrex::Real linf_gamma = 0.0_rt;
    amrex::Real count_gamma = 0.0_rt;
    amrex::Real count_skip_no_liq_face = 0.0_rt;

    stats.dx_min = std::numeric_limits<amrex::Real>::max();

    for (int lev = 0; lev < nlevels; ++lev) {
        amrex::iMultiFab level_mask;
        if (lev < nlevels - 1) {
            level_mask = makeFineMask(
                mesh.boxArray(lev), mesh.DistributionMap(lev),
                mesh.boxArray(lev + 1), mesh.refRatio(lev), 1, 0);
        } else {
            level_mask.define(
                mesh.boxArray(lev), mesh.DistributionMap(lev), 1, 0,
                amrex::MFInfo());
            level_mask.setVal(1);
        }

        const auto dx = mesh.Geom(lev).CellSizeArray();
        const amrex::Real dv = dx[0] * dx[1] * dx[2];
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (dx[dir] < stats.dx_min) {
                stats.dx_min = dx[dir];
            }
        }

        const auto& color_fab = phi(lev);
        auto const& color_arr = color_fab.const_arrays();
        auto const& u_arr = u_mac(lev).const_arrays();
        auto const& v_arr = v_mac(lev).const_arrays();
        auto const& w_arr = w_mac(lev).const_arrays();
        auto const& mask_arr = level_mask.const_arrays();

        amrex::GpuTuple<
            amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
            amrex::Real, amrex::Real>
            vals = amrex::ParReduce(
                amrex::TypeList<
                    amrex::ReduceOpSum, amrex::ReduceOpSum,
                    amrex::ReduceOpSum, amrex::ReduceOpSum,
                    amrex::ReduceOpSum, amrex::ReduceOpSum,
                    amrex::ReduceOpMax>{},
                amrex::TypeList<
                    amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                    amrex::Real, amrex::Real, amrex::Real>{},
                color_fab, amrex::IntVect(0),
                [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)
                    -> amrex::GpuTuple<
                        amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                        amrex::Real, amrex::Real, amrex::Real> {
                    auto const& color = color_arr[box_no];
                    auto const& u = u_arr[box_no];
                    auto const& v = v_arr[box_no];
                    auto const& w = w_arr[box_no];
                    auto const& mask = mask_arr[box_no];

                    amrex::Real vol = 0.0_rt;
                    amrex::Real sum_abs = 0.0_rt;
                    amrex::Real sum_sq = 0.0_rt;
                    amrex::Real mean_abs = 0.0_rt;
                    amrex::Real count = 0.0_rt;
                    amrex::Real count_skip = 0.0_rt;
                    amrex::Real linf = 0.0_rt;

                    amrex::Real uc = 0.0_rt;
                    amrex::Real vc = 0.0_rt;
                    amrex::Real wc = 0.0_rt;

                    if (mask(i, j, k) > 0 &&
                        spurious_current_impl::liquid_cell(
                            color(i, j, k), params.c_liq_min) &&
                        spurious_current_impl::near_interface_cell(
                            color, i, j, k, params.c_liq_min)) {
                        if (spurious_current_impl::cell_centered_velocity(
                                color, u, v, w, i, j, k, params.c_liq_min, uc,
                                vc, wc)) {
                            const amrex::Real umag =
                                spurious_current_impl::velocity_mag(uc, vc, wc);

                            vol = dv;
                            sum_abs = umag * dv;
                            sum_sq = umag * umag * dv;
                            mean_abs = umag;
                            count = 1.0_rt;
                            linf = umag;
                        } else {
                            count_skip = 1.0_rt;
                        }
                    }

                    return {
                        vol, sum_abs, sum_sq, mean_abs, count, count_skip,
                        linf};
                });

        vol_gamma += amrex::get<0>(vals);
        sum_abs_gamma += amrex::get<1>(vals);
        sum_sq_gamma += amrex::get<2>(vals);
        mean_abs_gamma += amrex::get<3>(vals);
        count_gamma += amrex::get<4>(vals);
        count_skip_no_liq_face += amrex::get<5>(vals);
        if (amrex::get<6>(vals) > linf_gamma) {
            linf_gamma = amrex::get<6>(vals);
        }
    }

    amrex::ParallelDescriptor::ReduceRealSum(vol_gamma);
    amrex::ParallelDescriptor::ReduceRealSum(sum_abs_gamma);
    amrex::ParallelDescriptor::ReduceRealSum(sum_sq_gamma);
    amrex::ParallelDescriptor::ReduceRealSum(mean_abs_gamma);
    amrex::ParallelDescriptor::ReduceRealSum(count_gamma);
    amrex::ParallelDescriptor::ReduceRealSum(count_skip_no_liq_face);
    amrex::ParallelDescriptor::ReduceRealMax(linf_gamma);

    stats.count = count_gamma;
    stats.skipped_no_liq_face = count_skip_no_liq_face;
    stats.vol = vol_gamma;
    stats.Linf = linf_gamma;

    if (count_gamma > 0.0_rt) {
        stats.mean_abs = mean_abs_gamma / count_gamma;
    }

    if (vol_gamma > 0.0_rt) {
        stats.L1_avg = sum_abs_gamma / vol_gamma;
        stats.L2_rms = std::sqrt(sum_sq_gamma / vol_gamma);
        stats.L1_int = sum_abs_gamma;
        stats.L2_int = std::sqrt(sum_sq_gamma);
    }

    if (stats.dx_min > 0.0_rt && params.radius > 0.0_rt) {
        stats.R_over_dx = params.radius / stats.dx_min;
        stats.dx_over_R = stats.dx_min / params.radius;
    }

    if (params.sigma > 0.0_rt && params.rho_ref > 0.0_rt &&
        params.radius > 0.0_rt) {
        stats.U_sigma = std::sqrt(params.sigma / (params.rho_ref * params.radius));
    }

    if (stats.U_sigma > 0.0_rt) {
        stats.mean_abs_nd = stats.mean_abs / stats.U_sigma;
        stats.L1_avg_nd = stats.L1_avg / stats.U_sigma;
        stats.L2_rms_nd = stats.L2_rms / stats.U_sigma;
        stats.Linf_nd = stats.Linf / stats.U_sigma;
    }

    return stats;
}

} // namespace kynema_sgf::multiphase
