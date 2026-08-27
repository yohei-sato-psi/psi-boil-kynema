#include "src/equation_systems/vof/vof_tension.H"

#include "src/core/Field.H"
#include "src/core/FieldRepo.H"
#include "src/equation_systems/vof/vof_curv_HF.H"
#include "AMReX_Gpu.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf::multiphase {
namespace tension_impl {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
kappa_ave(const amrex::Real r1, const amrex::Real r2)
{
    amrex::Real x = 0.0_rt;
    if (r1 * r2 > 0.0_rt) {
        x = 2.0_rt * r1 * r2 / (r1 + r2);
    } else {
        x = 0.5_rt * (r1 + r2);
    }
    return x;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool iflag_tension(
    const int i1, const int i2)
{
    bool flag = false;
    if (i1 == 1 || i1 == 2 || i2 == 1 || i2 == 2) {
        flag = true;
    }
    return flag;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real kappa_ave(
    const amrex::Real r1,
    const amrex::Real r2,
    const int i1,
    const int i2)
{
    amrex::Real x = 0.0_rt;

    if (i1 == 1 && i2 == 1) {
        x = kappa_ave(r1, r2);
    } else if (i1 == 1 && i2 == 2) {
        x = r1;
    } else if (i1 == 2 && i2 == 1) {
        x = r2;
    } else if (i1 == 2 && i2 == 0) {
        x = r1;
    } else if (i1 == 0 && i2 == 2) {
        x = r2;
    } else if (i1 == 2 && i2 == 2) {
        x = kappa_ave(r1, r2);
    } else if (i1 == 1 && i2 == 0) {
        x = r1;
    } else if (i1 == 0 && i2 == 1) {
        x = r2;
    }

    return x;
}

} // namespace tension_impl

void tension(
    Field& phi,
    const Field& rho,
    Field& kappa,
    Field& iflag,
    Field& stx,
    Field& sty,
    Field& stz,
    Field& st,
    const amrex::Real sigma,
    const amrex::Real rho1,
    const amrex::Real rho2,
    const amrex::Real time)
{
    BL_PROFILE("kynema-sgf::multiphase::tension");

    /*----------------------------------+
    |  1st step: curvature calculation  |
    +----------------------------------*/
    curv_HF(phi, kappa, iflag, time);

    /*-----------------------+
    |  2nd step: body force  |
    +-----------------------*/
    stx.setVal(0.0_rt);
    sty.setVal(0.0_rt);
    stz.setVal(0.0_rt);
    st.setVal(0.0_rt);

    const amrex::Real rho_diff = rho1 - rho2;
    const amrex::Real rho_ave = 0.5_rt * (rho1 + rho2);

    const int nlevels = phi.repo().num_active_levels();
    const int lev = nlevels - 1;
    const auto dxyz = phi.repo().mesh().Geom(lev).CellSizeArray();

    const auto& phi_arrs = phi(lev).const_arrays();
    const auto& rho_arrs = rho(lev).const_arrays();
    const auto& kappa_arrs = kappa(lev).const_arrays();
    const auto& iflag_arrs = iflag(lev).const_arrays();
    const auto& stx_arrs = stx(lev).arrays();
    const auto& sty_arrs = sty(lev).arrays();
    const auto& stz_arrs = stz(lev).arrays();
    const auto& st_arrs = st(lev).arrays();

    if (rho_diff == 0.0_rt) {
        amrex::ParallelFor(
            stx(lev), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                amrex::Real flag = 1.0_rt;
                const int iflagm =
                    static_cast<int>(iflag_arrs[nbx](i - 1, j, k));
                const int iflagp =
                    static_cast<int>(iflag_arrs[nbx](i, j, k));
                if (!tension_impl::iflag_tension(iflagm, iflagp)) {
                    flag = 0.0_rt;
                }
                stx_arrs[nbx](i, j, k) +=
                    sigma *
                    tension_impl::kappa_ave(
                        kappa_arrs[nbx](i - 1, j, k),
                        kappa_arrs[nbx](i, j, k), iflagm, iflagp) *
                    (phi_arrs[nbx](i, j, k) -
                     phi_arrs[nbx](i - 1, j, k)) /
                    dxyz[0] * flag;
            });

        amrex::ParallelFor(
            sty(lev), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                amrex::Real flag = 1.0_rt;
                const int iflagm =
                    static_cast<int>(iflag_arrs[nbx](i, j - 1, k));
                const int iflagp =
                    static_cast<int>(iflag_arrs[nbx](i, j, k));
                if (!tension_impl::iflag_tension(iflagm, iflagp)) {
                    flag = 0.0_rt;
                }
                sty_arrs[nbx](i, j, k) +=
                    sigma *
                    tension_impl::kappa_ave(
                        kappa_arrs[nbx](i, j - 1, k),
                        kappa_arrs[nbx](i, j, k), iflagm, iflagp) *
                    (phi_arrs[nbx](i, j, k) -
                     phi_arrs[nbx](i, j - 1, k)) /
                    dxyz[1] * flag;
            });

        amrex::ParallelFor(
            stz(lev), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                amrex::Real flag = 1.0_rt;
                const int iflagm =
                    static_cast<int>(iflag_arrs[nbx](i, j, k - 1));
                const int iflagp =
                    static_cast<int>(iflag_arrs[nbx](i, j, k));
                if (!tension_impl::iflag_tension(iflagm, iflagp)) {
                    flag = 0.0_rt;
                }
                stz_arrs[nbx](i, j, k) +=
                    sigma *
                    tension_impl::kappa_ave(
                        kappa_arrs[nbx](i, j, k - 1),
                        kappa_arrs[nbx](i, j, k), iflagm, iflagp) *
                    (phi_arrs[nbx](i, j, k) -
                     phi_arrs[nbx](i, j, k - 1)) /
                    dxyz[2] * flag;
            });
    } else {
        amrex::ParallelFor(
            stx(lev), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                amrex::Real flag = 1.0_rt;
                const int iflagm =
                    static_cast<int>(iflag_arrs[nbx](i - 1, j, k));
                const int iflagp =
                    static_cast<int>(iflag_arrs[nbx](i, j, k));
                if (!tension_impl::iflag_tension(iflagm, iflagp)) {
                    flag = 0.0_rt;
                }
                stx_arrs[nbx](i, j, k) +=
                    sigma *
                    tension_impl::kappa_ave(
                        kappa_arrs[nbx](i - 1, j, k),
                        kappa_arrs[nbx](i, j, k), iflagm, iflagp) *
                    (rho_arrs[nbx](i, j, k) -
                     rho_arrs[nbx](i - 1, j, k)) /
                    dxyz[0] / rho_diff *
                    0.5_rt *
                    (rho_arrs[nbx](i, j, k) +
                     rho_arrs[nbx](i - 1, j, k)) /
                    rho_ave * flag;
            });

        amrex::ParallelFor(
            sty(lev), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                amrex::Real flag = 1.0_rt;
                const int iflagm =
                    static_cast<int>(iflag_arrs[nbx](i, j - 1, k));
                const int iflagp =
                    static_cast<int>(iflag_arrs[nbx](i, j, k));
                if (!tension_impl::iflag_tension(iflagm, iflagp)) {
                    flag = 0.0_rt;
                }
                sty_arrs[nbx](i, j, k) +=
                    sigma *
                    tension_impl::kappa_ave(
                        kappa_arrs[nbx](i, j - 1, k),
                        kappa_arrs[nbx](i, j, k), iflagm, iflagp) *
                    (rho_arrs[nbx](i, j, k) -
                     rho_arrs[nbx](i, j - 1, k)) /
                    dxyz[1] / rho_diff *
                    0.5_rt *
                    (rho_arrs[nbx](i, j, k) +
                     rho_arrs[nbx](i, j - 1, k)) /
                    rho_ave * flag;
            });

        amrex::ParallelFor(
            stz(lev), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                amrex::Real flag = 1.0_rt;
                const int iflagm =
                    static_cast<int>(iflag_arrs[nbx](i, j, k - 1));
                const int iflagp =
                    static_cast<int>(iflag_arrs[nbx](i, j, k));
                if (!tension_impl::iflag_tension(iflagm, iflagp)) {
                    flag = 0.0_rt;
                }
                stz_arrs[nbx](i, j, k) +=
                    sigma *
                    tension_impl::kappa_ave(
                        kappa_arrs[nbx](i, j, k - 1),
                        kappa_arrs[nbx](i, j, k), iflagm, iflagp) *
                    (rho_arrs[nbx](i, j, k) -
                     rho_arrs[nbx](i, j, k - 1)) /
                    dxyz[2] / rho_diff *
                    0.5_rt *
                    (rho_arrs[nbx](i, j, k) +
                     rho_arrs[nbx](i, j, k - 1)) /
                    rho_ave * flag;
            });
    }

    amrex::ParallelFor(
        st(lev), amrex::IntVect(0), AMREX_SPACEDIM,
        [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int n) {
            if (n == 0) {
                st_arrs[nbx](i, j, k, n) =
                    0.5_rt *
                    (stx_arrs[nbx](i, j, k) +
                     stx_arrs[nbx](i + 1, j, k));
            } else if (n == 1) {
                st_arrs[nbx](i, j, k, n) =
                    0.5_rt *
                    (sty_arrs[nbx](i, j, k) +
                     sty_arrs[nbx](i, j + 1, k));
            } else {
                st_arrs[nbx](i, j, k, n) =
                    0.5_rt *
                    (stz_arrs[nbx](i, j, k) +
                     stz_arrs[nbx](i, j, k + 1));
            }
        });
    amrex::Gpu::streamSynchronize();
}

} // namespace kynema_sgf::multiphase
