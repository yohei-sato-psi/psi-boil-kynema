#ifndef SURFACETENSIONFORCING_H
#define SURFACETENSIONFORCING_H

#include "src/equation_systems/icns/MomentumSource.H"

namespace kynema_sgf::pde::icns {

/**
 * Add PSI-BOIL-style CSF surface-tension force to the Navier-Stokes equation.
 *
 * The VOF post-solve operation stores the CSF volume force on staggered faces
 * as stx, sty, and stz.  ICNS source terms are accumulated as acceleration and
 * multiplied by density after all sources have been applied, so this source
 * interpolates the face force to cell centers and divides by the local density.
 */
class SurfaceTensionForcing
    : public MomentumSource::Register<SurfaceTensionForcing>
{
public:
    static std::string identifier() { return "SurfaceTensionForcing"; }

    explicit SurfaceTensionForcing(const CFDSim& sim);

    ~SurfaceTensionForcing() override;

    void operator()(
        int lev, FieldState fstate, amrex::MultiFab& src_term) const override;

private:
    Field& m_stx;
    Field& m_sty;
    Field& m_stz;
    Field& m_rho;
};

} // namespace kynema_sgf::pde::icns

#endif /* SURFACETENSIONFORCING_H */
