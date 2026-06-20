#ifndef HEIGHT_FUNCTIONS_H
#define HEIGHT_FUNCTIONS_H

#include <cmath>
#include <limits>
#include "src/equation_systems/vof/volume_fractions.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_REAL.H"

namespace kynema_sgf {
class Field;
}

namespace kynema_sgf::multiphase {

/** Result of the PSI-BOIL height-function curvature calculation. */
struct HeightFunctionResult
{
    amrex::Real kappa{0.0};
    int dirMax{0};
    int iflag{0};
};

/** Error statistics over cells where a height function is valid. */
struct HeightFunctionStats
{
    amrex::Real interface_cells{0.0};
    amrex::Real iflag_cells{0.0};
    amrex::Real missing_interface_cells{0.0};
    amrex::Real direct_cells{0.0};
    amrex::Real extrapolated_cells{0.0};
    amrex::Real kappa_mean{0.0};
    amrex::Real l1_error{0.0};
    amrex::Real l2_error{0.0};
    amrex::Real linf_error{0.0};
};

namespace height_function_impl {

static constexpr int mof = 3;
static constexpr int nof = 1;
static constexpr amrex::Real phisurf = 0.5;
static constexpr amrex::Real theta_crit = 0.8;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool iflag_cell(
    const int i,
    const int j,
    const int k,
    const amrex::Array4<amrex::Real const>& phi)
{
    const amrex::Real phi_c = phi(i, j, k) - phisurf;
    return ((phi(i - 1, j, k) - phisurf) * phi_c <= 0.0) ||
           ((phi(i + 1, j, k) - phisurf) * phi_c <= 0.0) ||
           ((phi(i, j - 1, k) - phisurf) * phi_c <= 0.0) ||
           ((phi(i, j + 1, k) - phisurf) * phi_c <= 0.0) ||
           ((phi(i, j, k - 1) - phisurf) * phi_c <= 0.0) ||
           ((phi(i, j, k + 1) - phisurf) * phi_c <= 0.0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real sample(
    const amrex::Array4<amrex::Real const>& phi,
    const int i,
    const int j,
    const int k,
    const int dirMax,
    const int ii,
    const int jj,
    const int kk)
{
    amrex::Real phi_stencil = 0.0;
    if (dirMax == 1) {
        phi_stencil = phi(i + ii, j + jj, k + kk);
    } else if (dirMax == 2) {
        phi_stencil = phi(i + jj, j + ii, k + kk);
    } else {
        phi_stencil = phi(i + jj, j + kk, k + ii);
    }
    return amrex::max<amrex::Real>(
        0.0, amrex::min<amrex::Real>(1.0, phi_stencil));
}

/**
 * Build one seven-cell height column using the coloring correction from the
 * PSI-BOIL implementation of the Lopez height-function method.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real column_height(
    const amrex::Array4<amrex::Real const>& phi,
    const int i,
    const int j,
    const int k,
    const int dirMax,
    const int jj,
    const int kk,
    const amrex::Real max_n,
    const amrex::Real dmax,
    amrex::Real& hc)
{
    amrex::GpuArray<amrex::Real, 2 * mof + 1> stmp{};
    for (int ii = -mof; ii <= mof; ++ii) {
        stmp[ii + mof] = sample(phi, i, j, k, dirMax, ii, jj, kk);
    }

    amrex::Real mult = 1.0;
    if (max_n < 0.0) {
        mult = -1.0;
    }
    for (int ii = -1; ii >= -mof; --ii) {
        auto& current = stmp[ii + mof];
        const auto neighbor = stmp[ii + mof + 1];
        if (mult * (current - neighbor) > 0.0) {
            if ((current - phisurf) * (neighbor - phisurf) > 0.0 &&
                (current - phisurf) * (stmp[mof] - phisurf) > 0.0) {
                current = neighbor;
            } else {
                current = 0.5 * (1.0 - mult);
            }
        }
    }

    for (int ii = 1; ii <= mof; ++ii) {
        auto& current = stmp[ii + mof];
        const auto neighbor = stmp[ii + mof - 1];
        if (mult * (current - neighbor) < 0.0) {
            if ((current - phisurf) * (neighbor - phisurf) > 0.0 &&
                (current - phisurf) * (stmp[mof] - phisurf) > 0.0) {
                current = neighbor;
            } else {
                current = 0.5 * (1.0 + mult);
            }
        }
    }

    amrex::Real h = 0.0;
    hc = 0.0;
    for (const auto phi_value : stmp) {
        h += phi_value * dmax;
        hc += phi_value;
    }
    return h;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
height_index(const int jj, const int kk)
{
    return (kk + nof) * (2 * nof + 1) + jj + nof;
}

} // namespace height_function_impl

/**
 * Compute curvature from a 7 x 3 x 3 VOF stencil using the PSI-BOIL
 * height-function algorithm.
 *
 * The implementation is for Cartesian grids and requires three valid VOF ghost
 * cells in every direction. Curvature is evaluated only where the C=0.5
 * isosurface crosses the target cell or one of its face neighbors. The sign
 * convention follows PSI-BOIL: a liquid drop represented by C=1 has positive
 * curvature, while a C=0 bubble has negative curvature.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE HeightFunctionResult
height_function_curvature(
    const int i,
    const int j,
    const int k,
    const amrex::Array4<amrex::Real const>& phi,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxyz)
{
    using namespace height_function_impl;

    HeightFunctionResult result;
    if (!iflag_cell(i, j, k, phi)) {
        return result;
    }

    // Kynema's Young normal is -grad(C). PSI-BOIL selects the height direction
    // using +grad(C), so reverse it and account for anisotropic grid spacing.
    amrex::Real nx = 0.0;
    amrex::Real ny = 0.0;
    amrex::Real nz = 0.0;
    youngs_finite_difference_normal(i, j, k, phi, nx, ny, nz);
    nx = -nx / dxyz[0];
    ny = -ny / dxyz[1];
    nz = -nz / dxyz[2];

    const amrex::Real nnorm = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (nnorm <= std::numeric_limits<amrex::Real>::epsilon()) {
        return result;
    }
    nx /= nnorm;
    ny /= nnorm;
    nz /= nnorm;

    const amrex::Real abs_nx = std::abs(nx);
    const amrex::Real abs_ny = std::abs(ny);
    const amrex::Real abs_nz = std::abs(nz);
    int dirMax = 1;
    amrex::Real max_abs_n = abs_nx;
    amrex::Real max_n = nx;
    if (abs_ny > max_abs_n) {
        dirMax = 2;
        max_abs_n = abs_ny;
        max_n = ny;
    }
    if (abs_nz > max_abs_n) {
        dirMax = 3;
        max_abs_n = abs_nz;
        max_n = nz;
    }

    amrex::GpuArray<amrex::Real, 9> heights{};
    amrex::Real hc = 0.0;
    for (int kk = -nof; kk <= nof; ++kk) {
        for (int jj = -nof; jj <= nof; ++jj) {
            amrex::Real hc_column = 0.0;
            heights[height_index(jj, kk)] = column_height(
                phi, i, j, k, dirMax, jj, kk, max_n,
                dxyz[dirMax - 1], hc_column);
            if (jj == 0 && kk == 0) {
                hc = hc_column;
            }
        }
    }

    // With a complete seven-cell stencil, PSI-BOIL accepts heights whose
    // central column contains three full cells and one interfacial cell.
    const amrex::Real hc_limit = static_cast<amrex::Real>(mof);
    if (!(hc_limit < hc && hc < hc_limit + 1.0)) {
        return result;
    }

    const auto hmm = heights[height_index(-1, -1)];
    const auto hcm = heights[height_index(0, -1)];
    const auto hpm = heights[height_index(1, -1)];
    const auto hmc = heights[height_index(-1, 0)];
    const auto hcc = heights[height_index(0, 0)];
    const auto hpc = heights[height_index(1, 0)];
    const auto hmp = heights[height_index(-1, 1)];
    const auto hcp = heights[height_index(0, 1)];
    const auto hpp = heights[height_index(1, 1)];

    int dir1 = 0;
    if (dirMax == 1) {
        dir1 = 1;
    }
    int dir2 = 2;
    if (dirMax == 3) {
        dir2 = 1;
    }
    const amrex::Real d1 = dxyz[dir1];
    const amrex::Real d2 = dxyz[dir2];

    const amrex::Real theta =
        std::acos(amrex::min<amrex::Real>(1.0, max_abs_n));
    amrex::Real g = 0.0;
    if (theta < theta_crit) {
        g = 0.2;
    }
    const amrex::Real lopez_denom = 1.0 + 2.0 * g;

    const amrex::Real h_1 =
        (g * (hpm - hmm) + (hpc - hmc) + g * (hpp - hmp)) /
        (2.0 * d1 * lopez_denom);
    const amrex::Real h_2 =
        (g * (hmp - hmm) + (hcp - hcm) + g * (hpp - hpm)) /
        (2.0 * d2 * lopez_denom);
    const amrex::Real h_11 =
        (g * (hpp - 2.0 * hcp + hmp) +
         (hpc - 2.0 * hcc + hmc) +
         g * (hpm - 2.0 * hcm + hmm)) /
        (d1 * d1 * lopez_denom);
    const amrex::Real h_22 =
        (g * (hpp - 2.0 * hpc + hpm) +
         (hcp - 2.0 * hcc + hcm) +
         g * (hmp - 2.0 * hmc + hmm)) /
        (d2 * d2 * lopez_denom);
    const amrex::Real h_12 =
        (hpp - hpm - hmp + hmm) / (4.0 * d1 * d2);

    const amrex::Real hmetric = 1.0 + h_1 * h_1 + h_2 * h_2;
    result.kappa =
        -(h_11 + h_22 + h_11 * h_2 * h_2 + h_22 * h_1 * h_1 -
          2.0 * h_12 * h_1 * h_2) /
        (hmetric * std::sqrt(hmetric));
    result.dirMax = dirMax;
    result.iflag = 1;
    return result;
}

/**
 * Compute height-function curvature on the finest active AMR level. The caller
 * must first fill all three VOF ghost layers, including coarse-fine
 * interpolation. The AMR HF validation case assumes all interface cells are
 * contained inside the finest level; extrapolated curvature is therefore kept
 * on that level only.
 */
void compute_height_function_fields(
    Field& vof,
    Field& kappa,
    Field& kappa_exact,
    Field& kappa_error,
    Field& iflag,
    amrex::Real kappa_ref,
    amrex::Real time);

/**
 * Compute composite-grid error norms, excluding coarse cells covered by a
 * finer AMR level.
 */
HeightFunctionStats height_function_statistics(
    const Field& vof,
    const Field& kappa,
    const Field& kappa_error,
    const Field& iflag);

} // namespace kynema_sgf::multiphase
#endif
