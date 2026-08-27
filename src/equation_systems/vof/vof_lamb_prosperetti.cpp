#include "src/equation_systems/vof/vof_lamb_prosperetti.H"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <numbers>
#include "src/core/Field.H"
#include "src/core/FieldRepo.H"
#include "src/equation_systems/vof/vof_front_minmax.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_Reduce.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf::multiphase {
namespace lamb_prosperetti_impl {

void set_axes(LambProsperettiParams& params)
{
    const amrex::Real vol_scale = std::pow(
        (1.0_rt + params.A0_init) *
            std::pow(1.0_rt - 0.5_rt * params.A0_init, 2.0_rt),
        -1.0_rt / 3.0_rt);

    params.Ra = params.R0 * vol_scale *
                (1.0_rt - 0.5_rt * params.A0_init);
    params.Rb = params.R0 * vol_scale *
                (1.0_rt - 0.5_rt * params.A0_init);
    params.Rc = params.R0 * vol_scale * (1.0_rt + params.A0_init);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real ellipsoid_distance(
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real z,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& center,
    const amrex::Real Ra,
    const amrex::Real Rb,
    const amrex::Real Rc)
{
    const amrex::Real xx = (x - center[0]) / Ra;
    const amrex::Real yy = (y - center[1]) / Rb;
    const amrex::Real zz = (z - center[2]) / Rc;
    return std::sqrt(xx * xx + yy * yy + zz * zz);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real cell_inside_fraction(
    const int i,
    const int j,
    const int k,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& problo,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& center,
    const amrex::Real Ra,
    const amrex::Real Rb,
    const amrex::Real Rc,
    const int mmx,
    const int mmy,
    const int mmz)
{
    const amrex::Real x = problo[0] + (i + 0.5_rt) * dx[0];
    const amrex::Real y = problo[1] + (j + 0.5_rt) * dx[1];
    const amrex::Real z = problo[2] + (k + 0.5_rt) * dx[2];

    amrex::Real dist = ellipsoid_distance(x, y, z, center, Ra, Rb, Rc);
    const amrex::Real dist_margin =
        0.5_rt *
        std::sqrt(
            (dx[0] * dx[0]) / (Ra * Ra) +
            (dx[1] * dx[1]) / (Rb * Rb) +
            (dx[2] * dx[2]) / (Rc * Rc));
    amrex::Real inside = 0.0_rt;

    if (dist + dist_margin < 1.0_rt) {
        inside = 1.0_rt;
    } else if (dist - dist_margin > 1.0_rt) {
        inside = 0.0_rt;
    } else {
        const amrex::Real x0 = problo[0] + i * dx[0];
        const amrex::Real y0 = problo[1] + j * dx[1];
        const amrex::Real z0 = problo[2] + k * dx[2];

        const amrex::Real ddx = dx[0] / static_cast<amrex::Real>(mmx);
        const amrex::Real ddy = dx[1] / static_cast<amrex::Real>(mmy);
        const amrex::Real ddz = dx[2] / static_cast<amrex::Real>(mmz);

        int itmp = 0;
        for (int ii = 0; ii < mmx; ++ii) {
            for (int jj = 0; jj < mmy; ++jj) {
                for (int kk = 0; kk < mmz; ++kk) {
                    const amrex::Real xxc =
                        x0 + 0.5_rt * ddx +
                        static_cast<amrex::Real>(ii) * ddx;
                    const amrex::Real yyc =
                        y0 + 0.5_rt * ddy +
                        static_cast<amrex::Real>(jj) * ddy;
                    const amrex::Real zzc =
                        z0 + 0.5_rt * ddz +
                        static_cast<amrex::Real>(kk) * ddz;

                    const amrex::Real sub_dist =
                        ellipsoid_distance(xxc, yyc, zzc, center, Ra, Rb, Rc);
                    if (sub_dist < 1.0_rt) {
                        itmp += 1;
                    }
                }
            }
        }

        inside = static_cast<amrex::Real>(itmp) /
                 static_cast<amrex::Real>(mmx * mmy * mmz);
    }

    return inside;
}

amrex::Real volume_inside(
    const Field& vof, const LambProsperettiParams& params)
{
    const int nlevels = vof.repo().num_active_levels();
    const auto& mesh = vof.repo().mesh();

    amrex::Real volume = 0.0_rt;
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
        const amrex::Real cell_vol = dx[0] * dx[1] * dx[2];
        const bool fluid1_inside = params.fluid1_inside;

        volume += amrex::ReduceSum(
            vof(lev), level_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& color,
                const amrex::Array4<int const>& mask) -> amrex::Real {
                amrex::Real vol_fab = 0.0_rt;
                amrex::Loop(bx, [=, &vol_fab](int i, int j, int k) {
                    if (mask(i, j, k) > 0) {
                        amrex::Real inside = color(i, j, k);
                        if (!fluid1_inside) {
                            inside = 1.0_rt - color(i, j, k);
                        }
                        vol_fab += inside * cell_vol;
                    }
                });
                return vol_fab;
            });
    }

    amrex::ParallelDescriptor::ReduceRealSum(volume);
    return volume;
}

} // namespace lamb_prosperetti_impl

LambProsperettiParams lamb_prosperetti_params(
    const amrex::Vector<amrex::Real>& center,
    const amrex::Real radius,
    const bool fluid1_inside)
{
    LambProsperettiParams params;
    params.R0 = radius;
    params.fluid1_inside = fluid1_inside;
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        params.center[n] = center[n];
    }

    amrex::ParmParse pp_vof("VOF");
    pp_vof.query("lamb_prosperetti", params.enabled);
    pp_vof.query("lamb_prosperetti_radius", params.R0);
    pp_vof.query("lamb_prosperetti_A0", params.A0_init);
    pp_vof.query("lamb_prosperetti_amplitude", params.A0_init);
    pp_vof.query("lamb_prosperetti_mode", params.mode_l);
    pp_vof.query("lamb_prosperetti_output", params.output_file);
    pp_vof.query("lamb_prosperetti_write_front_minmax", params.write_front_minmax);

    amrex::Vector<amrex::Real> center_input(AMREX_SPACEDIM);
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        center_input[n] = params.center[n];
    }
    pp_vof.queryarr("lamb_prosperetti_center", center_input, 0, AMREX_SPACEDIM);
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        params.center[n] = center_input[n];
    }

    int subcell = params.subcell_x;
    pp_vof.query("lamb_prosperetti_subcell", subcell);
    pp_vof.query("lamb_prosperetti_subcells", subcell);
    params.subcell_x = subcell;
    params.subcell_y = subcell;
    params.subcell_z = subcell;

    int aspect_ratio = 1;
    pp_vof.query("lamb_prosperetti_aspect_ratio", aspect_ratio);
    if (aspect_ratio > 1) {
        params.subcell_z = subcell / aspect_ratio;
        if (params.subcell_z < 1) {
            params.subcell_z = 1;
        }
    }

    amrex::Vector<int> subcell_xyz(AMREX_SPACEDIM);
    subcell_xyz[0] = params.subcell_x;
    subcell_xyz[1] = params.subcell_y;
    subcell_xyz[2] = params.subcell_z;
    pp_vof.queryarr(
        "lamb_prosperetti_subcells_xyz", subcell_xyz, 0, AMREX_SPACEDIM);
    params.subcell_x = subcell_xyz[0];
    params.subcell_y = subcell_xyz[1];
    params.subcell_z = subcell_xyz[2];

    if (params.subcell_x < 1) {
        params.subcell_x = 1;
    }
    if (params.subcell_y < 1) {
        params.subcell_y = 1;
    }
    if (params.subcell_z < 1) {
        params.subcell_z = 1;
    }

    amrex::ParmParse pp_transport("transport");
    pp_transport.query("viscosity_fluid1", params.mu_l);
    pp_transport.query("viscosity_fluid2", params.mu_g);

    lamb_prosperetti_impl::set_axes(params);
    pp_vof.query("lamb_prosperetti_Ra", params.Ra);
    pp_vof.query("lamb_prosperetti_Rb", params.Rb);
    pp_vof.query("lamb_prosperetti_Rc", params.Rc);

    return params;
}

void initialize_lamb_prosperetti(
    Field& vof,
    Field& levelset,
    const int level,
    const amrex::Geometry& geom,
    const LambProsperettiParams& params)
{
    BL_PROFILE("kynema-sgf::multiphase::initialize_lamb_prosperetti");

    auto& volfrac = vof(level);
    auto& phi = levelset(level);
    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();

    const auto& volfrac_arrs = volfrac.arrays();
    const auto& phi_arrs = phi.arrays();

    const auto center = params.center;
    const bool fluid1_inside = params.fluid1_inside;
    const amrex::Real R0 = params.R0;
    const amrex::Real Ra = params.Ra;
    const amrex::Real Rb = params.Rb;
    const amrex::Real Rc = params.Rc;
    const int mmx = params.subcell_x;
    const int mmy = params.subcell_y;
    const int mmz = params.subcell_z;

    amrex::ParallelFor(
        volfrac, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
            const amrex::Real inside =
                lamb_prosperetti_impl::cell_inside_fraction(
                    i, j, k, problo, dx, center, Ra, Rb, Rc, mmx, mmy, mmz);

            amrex::Real color = inside;
            if (!fluid1_inside) {
                color = 1.0_rt - inside;
            }
            volfrac_arrs[nbx](i, j, k) = color;

            const amrex::Real x = problo[0] + (i + 0.5_rt) * dx[0];
            const amrex::Real y = problo[1] + (j + 0.5_rt) * dx[1];
            const amrex::Real z = problo[2] + (k + 0.5_rt) * dx[2];
            const amrex::Real dist =
                lamb_prosperetti_impl::ellipsoid_distance(
                    x, y, z, center, Ra, Rb, Rc);

            if (fluid1_inside) {
                phi_arrs[nbx](i, j, k) = R0 * (1.0_rt - dist);
            } else {
                phi_arrs[nbx](i, j, k) = R0 * (dist - 1.0_rt);
            }
        });
    amrex::Gpu::streamSynchronize();

    volfrac.FillBoundary(geom.periodicity());
    phi.FillBoundary(geom.periodicity());
}

void lamb_prosperetti(
    const Field& vof,
    const amrex::Real time,
    const int step,
    const amrex::Real sigma,
    const amrex::Real rho_l,
    const amrex::Real rho_g,
    const LambProsperettiParams& params,
    LambProsperettiState& state)
{
    BL_PROFILE("kynema-sgf::multiphase::lamb_prosperetti");

    if (!state.initialized) {
        state.volume = lamb_prosperetti_impl::volume_inside(vof, params);
        state.R_eq = std::pow(
            3.0_rt * state.volume /
                (4.0_rt * std::numbers::pi_v<amrex::Real>),
            1.0_rt / 3.0_rt);

        const amrex::Real l = static_cast<amrex::Real>(params.mode_l);
        const amrex::Real M = (l + 1.0_rt) * rho_g + l * rho_l;
        state.omega0 = std::sqrt(
            (l * (l - 1.0_rt) * (l + 1.0_rt) * (l + 2.0_rt) * sigma) /
            (M * std::pow(state.R_eq, 3.0_rt)));
        state.beta =
            ((2.0_rt * l + 1.0_rt) *
             ((l - 1.0_rt) * params.mu_g + (l + 2.0_rt) * params.mu_l)) /
            (M * std::pow(state.R_eq, 2.0_rt));

        amrex::Real omega_d_sq = state.omega0 * state.omega0 -
                                 state.beta * state.beta;
        if (omega_d_sq < 0.0_rt) {
            omega_d_sq = 0.0_rt;
        }
        state.omegad = std::sqrt(omega_d_sq);
        state.A0 = 2.0_rt / 3.0_rt *
                   (params.Rc - 0.5_rt * (params.Ra + params.Rb)) /
                   state.R_eq;

        amrex::Print() << "Pre_Lamb-Prosperetti"
                       << " V " << state.volume
                       << " R_eq " << state.R_eq
                       << " A0 " << state.A0
                       << " omega0 " << state.omega0
                       << " beta " << state.beta
                       << " omegad " << state.omegad;
        if (state.omegad > 0.0_rt) {
            amrex::Print() << " T "
                           << 2.0_rt * std::numbers::pi_v<amrex::Real> /
                                  state.omegad;
        } else {
            amrex::Print() << " overdamped_T_undefined";
        }
        amrex::Print() << "\n";

        state.initialized = true;
    }

    const VOFFrontMinMax front =
        front_minmax(vof, time, params.write_front_minmax);
    if (!front.front_exist) {
        amrex::Print() << "LP:"
                       << " step " << step
                       << " time " << time
                       << " no_front\n";
        return;
    }

    amrex::Real rx = front.xmaxft - params.center[0];
    const amrex::Real rxm = params.center[0] - front.xminft;
    if (rxm > rx) {
        rx = rxm;
    }

    amrex::Real ry = front.ymaxft - params.center[1];
    const amrex::Real rym = params.center[1] - front.yminft;
    if (rym > ry) {
        ry = rym;
    }

    amrex::Real rz = front.zmaxft - params.center[2];
    const amrex::Real rzm = params.center[2] - front.zminft;
    if (rzm > rz) {
        rz = rzm;
    }

    const amrex::Real r_e = 0.5_rt * (rx + ry);
    const amrex::Real At = 2.0_rt / 3.0_rt * (rz - r_e) / state.R_eq;
    const amrex::Real At_th = state.A0 * std::exp(-state.beta * time) *
                              std::cos(state.omegad * time);
    const amrex::Real xy_th = state.R_eq * (1.0_rt - 0.5_rt * At_th);
    const amrex::Real z_th = state.R_eq * (1.0_rt + At_th);
    const amrex::Real errA = At - At_th;

    amrex::Print() << "LP:"
                   << " step " << step
                   << " time " << time
                   << " x " << rx
                   << " y " << ry
                   << " z " << rz
                   << " xy_th " << xy_th
                   << " z_th " << z_th
                   << " At " << At
                   << " At_th " << At_th
                   << " errA " << errA << "\n";

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ios_base::openmode mode = std::ios::out;
        if (state.file_initialized) {
            mode = std::ios::out | std::ios::app;
        }
        std::ofstream output(params.output_file, mode);
        output << std::setprecision(17);
        if (!state.file_initialized) {
            output << "step,time,x,y,z,xy_th,z_th,At,At_th,errA,"
                      "V,R_eq,A0,omega0,beta,omegad\n";
        }
        output << step << ',' << time << ','
               << rx << ',' << ry << ',' << rz << ','
               << xy_th << ',' << z_th << ','
               << At << ',' << At_th << ',' << errA << ','
               << state.volume << ',' << state.R_eq << ','
               << state.A0 << ',' << state.omega0 << ','
               << state.beta << ',' << state.omegad << '\n';
    }
    state.file_initialized = true;
}

} // namespace kynema_sgf::multiphase
