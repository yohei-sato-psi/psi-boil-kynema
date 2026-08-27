#include "src/equation_systems/icns/source_terms/SurfaceTensionForcing.H"

#include "src/CFDSim.H"
#include "src/core/FieldUtils.H"
#include "AMReX_Gpu.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf::pde::icns {

SurfaceTensionForcing::SurfaceTensionForcing(const CFDSim& sim)
    : m_stx(sim.repo().get_field("stx"))
    , m_sty(sim.repo().get_field("sty"))
    , m_stz(sim.repo().get_field("stz"))
    , m_rho(sim.repo().get_field("density"))
{
    m_stx.setVal(0.0_rt);
    m_sty.setVal(0.0_rt);
    m_stz.setVal(0.0_rt);
}

SurfaceTensionForcing::~SurfaceTensionForcing() = default;

void SurfaceTensionForcing::operator()(
    const int lev, const FieldState fstate, amrex::MultiFab& src_term) const
{
    auto const& src_arrs = src_term.arrays();
    auto const& rho_arrs =
        m_rho.state(field_impl::phi_state(fstate))(lev).const_arrays();
    auto const& stx_arrs = m_stx(lev).const_arrays();
    auto const& sty_arrs = m_sty(lev).const_arrays();
    auto const& stz_arrs = m_stz(lev).const_arrays();

    amrex::ParallelFor(
        src_term, amrex::IntVect(0), AMREX_SPACEDIM,
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int n) {
            amrex::Real stmp = 0.0_rt;
            if (n == 0) {
                stmp = 0.5_rt *
                       (stx_arrs[nbx](i, j, k) +
                        stx_arrs[nbx](i + 1, j, k));
            } else if (n == 1) {
                stmp = 0.5_rt *
                       (sty_arrs[nbx](i, j, k) +
                        sty_arrs[nbx](i, j + 1, k));
            } else {
                stmp = 0.5_rt *
                       (stz_arrs[nbx](i, j, k) +
                        stz_arrs[nbx](i, j, k + 1));
            }

            src_arrs[nbx](i, j, k, n) +=
                stmp / rho_arrs[nbx](i, j, k);
        });
}

} // namespace kynema_sgf::pde::icns
