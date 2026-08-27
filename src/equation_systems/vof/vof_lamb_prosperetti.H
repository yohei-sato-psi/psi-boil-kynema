#ifndef VOF_LAMB_PROSPERETTI_H
#define VOF_LAMB_PROSPERETTI_H

#include <string>
#include "AMReX_Geometry.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_REAL.H"
#include "AMReX_Vector.H"

namespace kynema_sgf {
class Field;
}

namespace kynema_sgf::multiphase {

struct LambProsperettiParams
{
    bool enabled{false};
    bool fluid1_inside{true};
    bool write_front_minmax{true};
    int mode_l{2};
    int subcell_x{200};
    int subcell_y{200};
    int subcell_z{200};
    amrex::Real R0{0.5e-3};
    amrex::Real A0_init{0.05};
    amrex::Real Ra{0.0};
    amrex::Real Rb{0.0};
    amrex::Real Rc{0.0};
    amrex::Real mu_l{2.80e-4};
    amrex::Real mu_g{1.26e-5};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center{
        0.0, 0.0, 0.0};
    std::string output_file{"lamb_prosperetti.csv"};
};

struct LambProsperettiState
{
    bool initialized{false};
    bool file_initialized{false};
    amrex::Real volume{0.0};
    amrex::Real R_eq{0.0};
    amrex::Real A0{0.0};
    amrex::Real omega0{0.0};
    amrex::Real beta{0.0};
    amrex::Real omegad{0.0};
};

LambProsperettiParams lamb_prosperetti_params(
    const amrex::Vector<amrex::Real>& center,
    amrex::Real radius,
    bool fluid1_inside);

void initialize_lamb_prosperetti(
    Field& vof,
    Field& levelset,
    int level,
    const amrex::Geometry& geom,
    const LambProsperettiParams& params);

void lamb_prosperetti(
    const Field& vof,
    amrex::Real time,
    int step,
    amrex::Real sigma,
    amrex::Real rho_l,
    amrex::Real rho_g,
    const LambProsperettiParams& params,
    LambProsperettiState& state);

} // namespace kynema_sgf::multiphase

#endif /* VOF_LAMB_PROSPERETTI_H */
