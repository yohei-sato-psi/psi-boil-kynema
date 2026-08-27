#ifndef VOF_FRONT_MINMAX_H
#define VOF_FRONT_MINMAX_H

#include "AMReX_GpuContainers.H"
#include "AMReX_REAL.H"

namespace kynema_sgf {
class Field;
}

namespace kynema_sgf::multiphase {

struct VOFFrontMinMax
{
    amrex::Real xminft{0.0};
    amrex::Real xmaxft{0.0};
    amrex::Real yminft{0.0};
    amrex::Real ymaxft{0.0};
    amrex::Real zminft{0.0};
    amrex::Real zmaxft{0.0};
    bool front_exist{false};
};

VOFFrontMinMax front_minmax(
    const Field& phi,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& range_min,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& range_max,
    amrex::Real time,
    bool write_log);

VOFFrontMinMax front_minmax(
    const Field& phi, amrex::Real time, bool write_log);

} // namespace kynema_sgf::multiphase

#endif /* VOF_FRONT_MINMAX_H */
