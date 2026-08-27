#include "src/equation_systems/vof/vof_curv_HF.H"

#include <cmath>
#include "src/core/Field.H"
#include "src/core/FieldRepo.H"
#include "src/core/ScratchField.H"
#include "src/utilities/constants.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Reduce.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf::multiphase {

void curv_HF(
    Field& vof, Field& kappa, Field& iflag, const amrex::Real time)
{
    BL_PROFILE("kynema-sgf::multiphase::curv_HF");

    const amrex::IntVect ng_HF(height_function_impl::mof);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        vof.num_grow().allGE(ng_HF),
        "PSI-BOIL height functions require three VOF ghost cells");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        iflag.num_grow().allGE(amrex::IntVect(1)),
        "PSI-BOIL curvature extrapolation requires one iflag ghost cell");

    kappa.setVal(0.0_rt);
    iflag.setVal(0.0_rt);

    const int nlevels = vof.repo().num_active_levels();
    const int lev_HF = nlevels - 1;

    // PSI-BOIL first marks cells adjacent to the C=0.5 isosurface.  The
    // present AMR HF test assumes all interface cells are contained inside the
    // finest level; therefore kappa and the PSI-BOIL extrapolation flag are
    // created only on that level.  For the current three-level setup, this is
    // Level 2.
    for (int lev = lev_HF; lev < nlevels; ++lev) {
        const auto& phi_arrs = vof(lev).const_arrays();
        const auto& iflag_arrs = iflag(lev).arrays();
        amrex::ParallelFor(
            iflag(lev), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                if (height_function_impl::iflag_cell(
                        i, j, k, phi_arrs[nbx])) {
                    iflag_arrs[nbx](i, j, k) = 1.0_rt;
                }
            });
    }

    // A marked cell remains iflag=1 only when its height stencil is valid.
    for (int lev = lev_HF; lev < nlevels; ++lev) {
        const auto dxyz = vof.repo().mesh().Geom(lev).CellSizeArray();
        const auto& phi_arrs = vof(lev).const_arrays();
        const auto& kappa_arrs = kappa(lev).arrays();
        const auto& iflag_arrs = iflag(lev).arrays();

        amrex::ParallelFor(
            kappa(lev), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                if (iflag_arrs[nbx](i, j, k) == 1.0_rt) {
                    const auto hf = height_function_curvature(
                        i, j, k, phi_arrs[nbx], dxyz);
                    if (hf.iflag == 1) {
                        kappa_arrs[nbx](i, j, k) = hf.kappa;
                    } else {
                        iflag_arrs[nbx](i, j, k) = 0.0_rt;
                    }
                }
            });
    }
    amrex::Gpu::streamSynchronize();

    kappa.fillpatch(time);
    iflag.fillpatch(time);

    auto stmp = vof.repo().create_scratch_field("HF_stmp", 1, 0);
    auto iflagx = vof.repo().create_scratch_field("HF_iflagx", 1, 0);

    // PSI-BOIL performs four synchronous face-neighbor extrapolation sweeps.
    for (int iloop = 1; iloop < 5; ++iloop) {
        for (int lev = lev_HF; lev < nlevels; ++lev) {
            amrex::MultiFab::Copy((*stmp)(lev), kappa(lev), 0, 0, 1, 0);
            amrex::MultiFab::Copy((*iflagx)(lev), iflag(lev), 0, 0, 1, 0);

            const auto& kappa_arrs = kappa(lev).const_arrays();
            const auto& iflag_arrs = iflag(lev).const_arrays();
            const auto& stmp_arrs = (*stmp)(lev).arrays();
            const auto& iflagx_arrs = (*iflagx)(lev).arrays();

            amrex::ParallelFor(
                (*stmp)(lev),
                [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                    if (iflag_arrs[nbx](i, j, k) == 0.0_rt) {
                        const amrex::Real im = amrex::min<amrex::Real>(
                            1.0_rt, iflag_arrs[nbx](i - 1, j, k));
                        const amrex::Real ip = amrex::min<amrex::Real>(
                            1.0_rt, iflag_arrs[nbx](i + 1, j, k));
                        const amrex::Real jm = amrex::min<amrex::Real>(
                            1.0_rt, iflag_arrs[nbx](i, j - 1, k));
                        const amrex::Real jp = amrex::min<amrex::Real>(
                            1.0_rt, iflag_arrs[nbx](i, j + 1, k));
                        const amrex::Real km = amrex::min<amrex::Real>(
                            1.0_rt, iflag_arrs[nbx](i, j, k - 1));
                        const amrex::Real kp = amrex::min<amrex::Real>(
                            1.0_rt, iflag_arrs[nbx](i, j, k + 1));
                        const amrex::Real inb = im + ip + jm + jp + km + kp;

                        if (inb >= 1.0_rt) {
                            stmp_arrs[nbx](i, j, k) =
                                (im * kappa_arrs[nbx](i - 1, j, k) +
                                 ip * kappa_arrs[nbx](i + 1, j, k) +
                                 jm * kappa_arrs[nbx](i, j - 1, k) +
                                 jp * kappa_arrs[nbx](i, j + 1, k) +
                                 km * kappa_arrs[nbx](i, j, k - 1) +
                                 kp * kappa_arrs[nbx](i, j, k + 1)) /
                                inb;
                            iflagx_arrs[nbx](i, j, k) = 2.0_rt;
                        }
                    }
                });
        }
        amrex::Gpu::streamSynchronize();

        for (int lev = lev_HF; lev < nlevels; ++lev) {
            amrex::MultiFab::Copy(kappa(lev), (*stmp)(lev), 0, 0, 1, 0);
            amrex::MultiFab::Copy(iflag(lev), (*iflagx)(lev), 0, 0, 1, 0);
        }
        kappa.fillpatch(time);
        iflag.fillpatch(time);
    }
}

void calc_HF_kappa_error(
    const Field& kappa,
    Field& kappa_exact,
    Field& kappa_error,
    const Field& iflag,
    const amrex::Real kappa_ref)
{
    BL_PROFILE("kynema-sgf::multiphase::calc_HF_kappa_error");

    kappa_exact.setVal(0.0_rt);
    kappa_error.setVal(0.0_rt);

    const int nlevels = kappa.repo().num_active_levels();
    for (int lev = 0; lev < nlevels; ++lev) {
        const auto& kappa_arrs = kappa(lev).const_arrays();
        const auto& exact_arrs = kappa_exact(lev).arrays();
        const auto& error_arrs = kappa_error(lev).arrays();
        const auto& iflag_arrs = iflag(lev).const_arrays();

        amrex::ParallelFor(
            kappa_exact(lev),
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                if (iflag_arrs[nbx](i, j, k) == 1.0_rt ||
                    iflag_arrs[nbx](i, j, k) == 2.0_rt) {
                    exact_arrs[nbx](i, j, k) = kappa_ref;
                    error_arrs[nbx](i, j, k) =
                        std::abs(kappa_arrs[nbx](i, j, k) - kappa_ref);
                }
            });
    }
    amrex::Gpu::streamSynchronize();
}

HeightFunctionStats height_function_statistics(
    const Field& vof,
    const Field& kappa,
    const Field& kappa_exact,
    const Field& kappa_error,
    const Field& iflag)
{
    HeightFunctionStats stats;
    amrex::Real kappa_sum = 0.0_rt;
    amrex::Real abs_error_sum = 0.0_rt;
    amrex::Real abs_exact_sum = 0.0_rt;
    amrex::Real squared_error_sum = 0.0_rt;
    amrex::Real exact_scale = 0.0_rt;

    const int nlevels = kappa.repo().num_active_levels();
    for (int lev = 0; lev < nlevels; ++lev) {
        amrex::iMultiFab level_mask;
        if (lev < nlevels - 1) {
            level_mask = amrex::makeFineMask(
                kappa.repo().mesh().boxArray(lev),
                kappa.repo().mesh().DistributionMap(lev),
                kappa.repo().mesh().boxArray(lev + 1),
                kappa.repo().mesh().refRatio(lev), 1, 0);
        } else {
            level_mask.define(
                kappa.repo().mesh().boxArray(lev),
                kappa.repo().mesh().DistributionMap(lev), 1, 0,
                amrex::MFInfo());
            level_mask.setVal(1);
        }

        amrex::MultiFab eval_mask(
            kappa(lev).boxArray(), kappa(lev).DistributionMap(), 1, 0,
            amrex::MFInfo(), kappa(lev).Factory());
        const auto& iflag_arrs = iflag(lev).const_arrays();
        const auto& level_mask_arrs = level_mask.const_arrays();
        const auto& eval_mask_arrs = eval_mask.arrays();
        amrex::ParallelFor(
            eval_mask,
            [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
                if (level_mask_arrs[nbx](i, j, k) > 0 &&
                    (iflag_arrs[nbx](i, j, k) == 1.0_rt ||
                     iflag_arrs[nbx](i, j, k) == 2.0_rt)) {
                    eval_mask_arrs[nbx](i, j, k) = 1.0_rt;
                } else {
                    eval_mask_arrs[nbx](i, j, k) = 0.0_rt;
                }
            });

        stats.interface_cells += amrex::ReduceSum(
            vof(lev), level_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& phi,
                const amrex::Array4<int const>& mask) -> amrex::Real {
                amrex::Real sum = 0.0_rt;
                amrex::Loop(bx, [=, &sum](int i, int j, int k) {
                    if (mask(i, j, k) > 0 &&
                        phi(i, j, k) > constants::TIGHT_TOL &&
                        phi(i, j, k) < 1.0_rt - constants::TIGHT_TOL) {
                        sum += 1.0_rt;
                    }
                });
                return sum;
            });

        stats.iflag_cells += amrex::ReduceSum(
            eval_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& mask) -> amrex::Real {
                amrex::Real sum = 0.0_rt;
                amrex::Loop(bx, [=, &sum](int i, int j, int k) {
                    sum += mask(i, j, k);
                });
                return sum;
            });

        stats.direct_cells += amrex::ReduceSum(
            vof(lev), iflag(lev), level_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& /*phi*/,
                const amrex::Array4<amrex::Real const>& flag,
                const amrex::Array4<int const>& mask) -> amrex::Real {
                amrex::Real sum = 0.0_rt;
                amrex::Loop(bx, [=, &sum](int i, int j, int k) {
                    if (mask(i, j, k) > 0 && flag(i, j, k) == 1.0_rt) {
                        sum += 1.0_rt;
                    }
                });
                return sum;
            });

        stats.extrapolated_cells += amrex::ReduceSum(
            vof(lev), iflag(lev), level_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& /*phi*/,
                const amrex::Array4<amrex::Real const>& flag,
                const amrex::Array4<int const>& mask) -> amrex::Real {
                amrex::Real sum = 0.0_rt;
                amrex::Loop(bx, [=, &sum](int i, int j, int k) {
                    if (mask(i, j, k) > 0 && flag(i, j, k) == 2.0_rt) {
                        sum += 1.0_rt;
                    }
                });
                return sum;
            });

        kappa_sum += amrex::ReduceSum(
            kappa(lev), eval_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& kappa_arr,
                const amrex::Array4<amrex::Real const>& mask) -> amrex::Real {
                amrex::Real sum = 0.0_rt;
                amrex::Loop(bx, [=, &sum](int i, int j, int k) {
                    sum += mask(i, j, k) * kappa_arr(i, j, k);
                });
                return sum;
            });

        abs_error_sum += amrex::ReduceSum(
            kappa_error(lev), eval_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& error,
                const amrex::Array4<amrex::Real const>& mask) -> amrex::Real {
                amrex::Real sum = 0.0_rt;
                amrex::Loop(bx, [=, &sum](int i, int j, int k) {
                    sum += mask(i, j, k) * std::abs(error(i, j, k));
                });
                return sum;
            });

        abs_exact_sum += amrex::ReduceSum(
            kappa_exact(lev), eval_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& exact,
                const amrex::Array4<amrex::Real const>& mask) -> amrex::Real {
                amrex::Real sum = 0.0_rt;
                amrex::Loop(bx, [=, &sum](int i, int j, int k) {
                    sum += mask(i, j, k) * std::abs(exact(i, j, k));
                });
                return sum;
            });

        squared_error_sum += amrex::ReduceSum(
            kappa_error(lev), eval_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& error,
                const amrex::Array4<amrex::Real const>& mask) -> amrex::Real {
                amrex::Real sum = 0.0_rt;
                amrex::Loop(bx, [=, &sum](int i, int j, int k) {
                    const amrex::Real value = error(i, j, k);
                    sum += mask(i, j, k) * value * value;
                });
                return sum;
            });

        const amrex::Real level_max = amrex::ReduceMax(
            kappa_error(lev), eval_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                const amrex::Box& bx,
                const amrex::Array4<amrex::Real const>& error,
                const amrex::Array4<amrex::Real const>& mask) -> amrex::Real {
                amrex::Real maximum = 0.0_rt;
                amrex::Loop(bx, [=, &maximum](int i, int j, int k) {
                    if (mask(i, j, k) > 0.0_rt) {
                        maximum = amrex::max<amrex::Real>(
                            maximum, std::abs(error(i, j, k)));
                    }
                });
                return maximum;
            });
        stats.linf_error =
            amrex::max<amrex::Real>(stats.linf_error, level_max);
    }

    amrex::ParallelDescriptor::ReduceRealSum(stats.interface_cells);
    amrex::ParallelDescriptor::ReduceRealSum(stats.iflag_cells);
    amrex::ParallelDescriptor::ReduceRealSum(stats.direct_cells);
    amrex::ParallelDescriptor::ReduceRealSum(stats.extrapolated_cells);
    amrex::ParallelDescriptor::ReduceRealSum(kappa_sum);
    amrex::ParallelDescriptor::ReduceRealSum(abs_error_sum);
    amrex::ParallelDescriptor::ReduceRealSum(abs_exact_sum);
    amrex::ParallelDescriptor::ReduceRealSum(squared_error_sum);
    amrex::ParallelDescriptor::ReduceRealMax(stats.linf_error);

    stats.missing_interface_cells =
        stats.interface_cells - stats.iflag_cells;
    if (stats.iflag_cells > 0.0_rt) {
        stats.kappa_mean = kappa_sum / stats.iflag_cells;
        if (abs_exact_sum > std::numeric_limits<amrex::Real>::epsilon()) {
            stats.l1_error = abs_error_sum / abs_exact_sum;
            exact_scale = abs_exact_sum / stats.iflag_cells;
            stats.l2_error =
                std::sqrt(squared_error_sum / stats.iflag_cells) /
                exact_scale;
            stats.linf_error /= exact_scale;
        }
    }
    return stats;
}

} // namespace kynema_sgf::multiphase
