#ifndef VOF_SPURIOUS_CURRENT_H
#define VOF_SPURIOUS_CURRENT_H

#include "AMReX_REAL.H"

namespace kynema_sgf {
class Field;
}

namespace kynema_sgf::multiphase {

struct SpuriousCurrentParams
{
    amrex::Real eps_c{1.0e-6};
    amrex::Real c_liq_min{0.5};
    amrex::Real radius{0.0};
    amrex::Real rho_ref{0.0};
    amrex::Real sigma{0.0};
};

struct SpuriousCurrentStats
{
    amrex::Real dx_min{0.0};
    amrex::Real radius{0.0};
    amrex::Real R_over_dx{0.0};
    amrex::Real dx_over_R{0.0};
    amrex::Real U_sigma{0.0};

    amrex::Real count{0.0};
    amrex::Real skipped_no_liq_face{0.0};
    amrex::Real vol{0.0};

    amrex::Real mean_abs{0.0};
    amrex::Real L1_avg{0.0};
    amrex::Real L2_rms{0.0};
    amrex::Real Linf{0.0};
    amrex::Real L1_int{0.0};
    amrex::Real L2_int{0.0};

    amrex::Real mean_abs_nd{0.0};
    amrex::Real L1_avg_nd{0.0};
    amrex::Real L2_rms_nd{0.0};
    amrex::Real Linf_nd{0.0};
};

SpuriousCurrentStats spurious_current(
    const Field& phi,
    const Field& u_mac,
    const Field& v_mac,
    const Field& w_mac,
    const SpuriousCurrentParams& params);

} // namespace kynema_sgf::multiphase

#endif /* VOF_SPURIOUS_CURRENT_H */
