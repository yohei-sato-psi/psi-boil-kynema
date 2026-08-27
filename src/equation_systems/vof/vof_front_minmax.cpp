#include "src/equation_systems/vof/vof_front_minmax.H"

#include <cmath>
#include <limits>
#include "src/core/Field.H"
#include "src/core/FieldRepo.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Reduce.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf::multiphase {
namespace front_minmax_impl {

static constexpr amrex::Real phisurf = 0.5;
static constexpr amrex::Real exa = 1.0e300;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool crossed_front(
    const amrex::Real phim, const amrex::Real phip)
{
    bool crossed = false;
    if ((phim - phisurf) * (phip - phisurf) <= 0.0_rt) {
        crossed = true;
    }
    return crossed;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real front_position(
    const amrex::Real xm,
    const amrex::Real xp,
    const amrex::Real phim,
    const amrex::Real phip)
{
    amrex::Real xyzfront = 0.5_rt * (xm + xp);
    const amrex::Real dphi = phip - phim;
    if (std::abs(dphi) > std::numeric_limits<amrex::Real>::epsilon()) {
        xyzfront = xm + (phisurf - phim) * (xp - xm) / dphi;
    }
    return xyzfront;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool in_range(
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real z,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& range_min,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& range_max)
{
    bool inside = true;
    if (x < range_min[0] || x > range_max[0]) {
        inside = false;
    }
    if (y < range_min[1] || y > range_max[1]) {
        inside = false;
    }
    if (z < range_min[2] || z > range_max[2]) {
        inside = false;
    }
    return inside;
}

} // namespace front_minmax_impl

VOFFrontMinMax front_minmax(
    const Field& phi,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& range_min,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& range_max,
    const amrex::Real time,
    const bool write_log)
{
    BL_PROFILE("kynema-sgf::multiphase::front_minmax");

    VOFFrontMinMax front;
    front.xminft = front_minmax_impl::exa;
    front.xmaxft = -front_minmax_impl::exa;
    front.yminft = front_minmax_impl::exa;
    front.ymaxft = -front_minmax_impl::exa;
    front.zminft = front_minmax_impl::exa;
    front.zmaxft = -front_minmax_impl::exa;
    front.front_exist = false;

    const int nlevels = phi.repo().num_active_levels();
    const auto& mesh = phi.repo().mesh();

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
        const auto problo = mesh.Geom(lev).ProbLoArray();

        const amrex::Real xmin_lev = amrex::ReduceMin(
            phi(lev), level_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& color,
                const amrex::Array4<int const>& mask) -> amrex::Real {
                amrex::Real xmin = front_minmax_impl::exa;
                amrex::Loop(bx, [=, &xmin](int i, int j, int k) {
                    if (mask(i, j, k) <= 0) {
                        return;
                    }

                    const amrex::Real x = problo[0] + (i + 0.5_rt) * dx[0];
                    const amrex::Real y = problo[1] + (j + 0.5_rt) * dx[1];
                    const amrex::Real z = problo[2] + (k + 0.5_rt) * dx[2];

                    const amrex::Real xp = x + dx[0];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            xp, y, z, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i + 1, j, k))) {
                        const amrex::Real xfront =
                            front_minmax_impl::front_position(
                                x, xp, color(i, j, k), color(i + 1, j, k));
                        xmin = amrex::min<amrex::Real>(xmin, xfront);
                    }

                    const amrex::Real yp = y + dx[1];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            x, yp, z, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i, j + 1, k))) {
                        xmin = amrex::min<amrex::Real>(xmin, x);
                    }

                    const amrex::Real zp = z + dx[2];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            x, y, zp, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i, j, k + 1))) {
                        xmin = amrex::min<amrex::Real>(xmin, x);
                    }
                });
                return xmin;
            });

        const amrex::Real xmax_lev = amrex::ReduceMax(
            phi(lev), level_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& color,
                const amrex::Array4<int const>& mask) -> amrex::Real {
                amrex::Real xmax = -front_minmax_impl::exa;
                amrex::Loop(bx, [=, &xmax](int i, int j, int k) {
                    if (mask(i, j, k) <= 0) {
                        return;
                    }

                    const amrex::Real x = problo[0] + (i + 0.5_rt) * dx[0];
                    const amrex::Real y = problo[1] + (j + 0.5_rt) * dx[1];
                    const amrex::Real z = problo[2] + (k + 0.5_rt) * dx[2];

                    const amrex::Real xp = x + dx[0];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            xp, y, z, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i + 1, j, k))) {
                        const amrex::Real xfront =
                            front_minmax_impl::front_position(
                                x, xp, color(i, j, k), color(i + 1, j, k));
                        xmax = amrex::max<amrex::Real>(xmax, xfront);
                    }

                    const amrex::Real yp = y + dx[1];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            x, yp, z, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i, j + 1, k))) {
                        xmax = amrex::max<amrex::Real>(xmax, x);
                    }

                    const amrex::Real zp = z + dx[2];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            x, y, zp, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i, j, k + 1))) {
                        xmax = amrex::max<amrex::Real>(xmax, x);
                    }
                });
                return xmax;
            });

        const amrex::Real ymin_lev = amrex::ReduceMin(
            phi(lev), level_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& color,
                const amrex::Array4<int const>& mask) -> amrex::Real {
                amrex::Real ymin = front_minmax_impl::exa;
                amrex::Loop(bx, [=, &ymin](int i, int j, int k) {
                    if (mask(i, j, k) <= 0) {
                        return;
                    }

                    const amrex::Real x = problo[0] + (i + 0.5_rt) * dx[0];
                    const amrex::Real y = problo[1] + (j + 0.5_rt) * dx[1];
                    const amrex::Real z = problo[2] + (k + 0.5_rt) * dx[2];

                    const amrex::Real xp = x + dx[0];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            xp, y, z, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i + 1, j, k))) {
                        ymin = amrex::min<amrex::Real>(ymin, y);
                    }

                    const amrex::Real yp = y + dx[1];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            x, yp, z, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i, j + 1, k))) {
                        const amrex::Real yfront =
                            front_minmax_impl::front_position(
                                y, yp, color(i, j, k), color(i, j + 1, k));
                        ymin = amrex::min<amrex::Real>(ymin, yfront);
                    }

                    const amrex::Real zp = z + dx[2];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            x, y, zp, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i, j, k + 1))) {
                        ymin = amrex::min<amrex::Real>(ymin, y);
                    }
                });
                return ymin;
            });

        const amrex::Real ymax_lev = amrex::ReduceMax(
            phi(lev), level_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& color,
                const amrex::Array4<int const>& mask) -> amrex::Real {
                amrex::Real ymax = -front_minmax_impl::exa;
                amrex::Loop(bx, [=, &ymax](int i, int j, int k) {
                    if (mask(i, j, k) <= 0) {
                        return;
                    }

                    const amrex::Real x = problo[0] + (i + 0.5_rt) * dx[0];
                    const amrex::Real y = problo[1] + (j + 0.5_rt) * dx[1];
                    const amrex::Real z = problo[2] + (k + 0.5_rt) * dx[2];

                    const amrex::Real xp = x + dx[0];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            xp, y, z, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i + 1, j, k))) {
                        ymax = amrex::max<amrex::Real>(ymax, y);
                    }

                    const amrex::Real yp = y + dx[1];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            x, yp, z, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i, j + 1, k))) {
                        const amrex::Real yfront =
                            front_minmax_impl::front_position(
                                y, yp, color(i, j, k), color(i, j + 1, k));
                        ymax = amrex::max<amrex::Real>(ymax, yfront);
                    }

                    const amrex::Real zp = z + dx[2];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            x, y, zp, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i, j, k + 1))) {
                        ymax = amrex::max<amrex::Real>(ymax, y);
                    }
                });
                return ymax;
            });

        const amrex::Real zmin_lev = amrex::ReduceMin(
            phi(lev), level_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& color,
                const amrex::Array4<int const>& mask) -> amrex::Real {
                amrex::Real zmin = front_minmax_impl::exa;
                amrex::Loop(bx, [=, &zmin](int i, int j, int k) {
                    if (mask(i, j, k) <= 0) {
                        return;
                    }

                    const amrex::Real x = problo[0] + (i + 0.5_rt) * dx[0];
                    const amrex::Real y = problo[1] + (j + 0.5_rt) * dx[1];
                    const amrex::Real z = problo[2] + (k + 0.5_rt) * dx[2];

                    const amrex::Real xp = x + dx[0];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            xp, y, z, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i + 1, j, k))) {
                        zmin = amrex::min<amrex::Real>(zmin, z);
                    }

                    const amrex::Real yp = y + dx[1];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            x, yp, z, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i, j + 1, k))) {
                        zmin = amrex::min<amrex::Real>(zmin, z);
                    }

                    const amrex::Real zp = z + dx[2];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            x, y, zp, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i, j, k + 1))) {
                        const amrex::Real zfront =
                            front_minmax_impl::front_position(
                                z, zp, color(i, j, k), color(i, j, k + 1));
                        zmin = amrex::min<amrex::Real>(zmin, zfront);
                    }
                });
                return zmin;
            });

        const amrex::Real zmax_lev = amrex::ReduceMax(
            phi(lev), level_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& color,
                const amrex::Array4<int const>& mask) -> amrex::Real {
                amrex::Real zmax = -front_minmax_impl::exa;
                amrex::Loop(bx, [=, &zmax](int i, int j, int k) {
                    if (mask(i, j, k) <= 0) {
                        return;
                    }

                    const amrex::Real x = problo[0] + (i + 0.5_rt) * dx[0];
                    const amrex::Real y = problo[1] + (j + 0.5_rt) * dx[1];
                    const amrex::Real z = problo[2] + (k + 0.5_rt) * dx[2];

                    const amrex::Real xp = x + dx[0];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            xp, y, z, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i + 1, j, k))) {
                        zmax = amrex::max<amrex::Real>(zmax, z);
                    }

                    const amrex::Real yp = y + dx[1];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            x, yp, z, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i, j + 1, k))) {
                        zmax = amrex::max<amrex::Real>(zmax, z);
                    }

                    const amrex::Real zp = z + dx[2];
                    if (front_minmax_impl::in_range(
                            x, y, z, range_min, range_max) &&
                        front_minmax_impl::in_range(
                            x, y, zp, range_min, range_max) &&
                        front_minmax_impl::crossed_front(
                            color(i, j, k), color(i, j, k + 1))) {
                        const amrex::Real zfront =
                            front_minmax_impl::front_position(
                                z, zp, color(i, j, k), color(i, j, k + 1));
                        zmax = amrex::max<amrex::Real>(zmax, zfront);
                    }
                });
                return zmax;
            });

        front.xminft = amrex::min<amrex::Real>(front.xminft, xmin_lev);
        front.xmaxft = amrex::max<amrex::Real>(front.xmaxft, xmax_lev);
        front.yminft = amrex::min<amrex::Real>(front.yminft, ymin_lev);
        front.ymaxft = amrex::max<amrex::Real>(front.ymaxft, ymax_lev);
        front.zminft = amrex::min<amrex::Real>(front.zminft, zmin_lev);
        front.zmaxft = amrex::max<amrex::Real>(front.zmaxft, zmax_lev);
    }

    amrex::ParallelDescriptor::ReduceRealMin(front.xminft);
    amrex::ParallelDescriptor::ReduceRealMax(front.xmaxft);
    amrex::ParallelDescriptor::ReduceRealMin(front.yminft);
    amrex::ParallelDescriptor::ReduceRealMax(front.ymaxft);
    amrex::ParallelDescriptor::ReduceRealMin(front.zminft);
    amrex::ParallelDescriptor::ReduceRealMax(front.zmaxft);

    if (front.xminft < 0.5_rt * front_minmax_impl::exa &&
        front.xmaxft > -0.5_rt * front_minmax_impl::exa &&
        front.yminft < 0.5_rt * front_minmax_impl::exa &&
        front.ymaxft > -0.5_rt * front_minmax_impl::exa &&
        front.zminft < 0.5_rt * front_minmax_impl::exa &&
        front.zmaxft > -0.5_rt * front_minmax_impl::exa) {
        front.front_exist = true;
    }

    if (write_log) {
        amrex::Print() << std::scientific
                       << "front_minmax:time,xmax= " << time << " "
                       << front.xmaxft << " " << front.zmaxft << "\n"
                       << std::defaultfloat;
    }

    return front;
}

VOFFrontMinMax front_minmax(
    const Field& phi, const amrex::Real time, const bool write_log)
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> range_min{
        -front_minmax_impl::exa,
        -front_minmax_impl::exa,
        -front_minmax_impl::exa};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> range_max{
        front_minmax_impl::exa,
        front_minmax_impl::exa,
        front_minmax_impl::exa};
    return front_minmax(phi, range_min, range_max, time, write_log);
}

} // namespace kynema_sgf::multiphase
