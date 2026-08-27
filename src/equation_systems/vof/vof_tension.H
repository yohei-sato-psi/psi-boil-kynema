#ifndef VOF_TENSION_H
#define VOF_TENSION_H

#include "AMReX_REAL.H"

namespace kynema_sgf {
class Field;
}

namespace kynema_sgf::multiphase {

void tension(
    Field& phi,
    const Field& rho,
    Field& kappa,
    Field& iflag,
    Field& stx,
    Field& sty,
    Field& stz,
    Field& st,
    amrex::Real sigma,
    amrex::Real rho1,
    amrex::Real rho2,
    amrex::Real time);

} // namespace kynema_sgf::multiphase

#endif /* VOF_TENSION_H */
