#ifndef VOF_OPS_H
#define VOF_OPS_H

#include <fstream>
#include <iomanip>
#include "src/equation_systems/vof/height_functions.H"
#include "src/equation_systems/vof/vof.H"
#include "src/physics/multiphase/MultiPhase.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"

namespace kynema_sgf::pde {

/** Field registration operator for VOF equation system
 *  \ingroup vof
 */
template <typename Scheme>
struct FieldRegOp<VOF, Scheme>
{
    explicit FieldRegOp(CFDSim& sim_in) : sim(sim_in) {}

    PDEFields operator()(const SimTime& time)
    {
        auto& repo = sim.repo();
        auto fields = create_fields_instance<VOF, Scheme>(time, repo);

        auto& levelset = repo.declare_cc_field("levelset", 1, 3, 1);
        auto& kappa = repo.declare_cc_field("kappa", 1, 1, 1);
        auto& kappa_exact = repo.declare_cc_field("kappa_exact", 1, 0, 1);
        auto& kappa_error = repo.declare_cc_field("kappa_error", 1, 0, 1);
        auto& iflag = repo.declare_cc_field("iflag", 1, 1, 1);

        levelset.set_default_fillpatch_bc(sim.time());
        kappa.set_default_fillpatch_bc(sim.time());
        kappa_exact.set_default_fillpatch_bc(sim.time());
        kappa_error.set_default_fillpatch_bc(sim.time());
        iflag.set_default_fillpatch_bc(sim.time());

        // Register fields for output/restart
        auto& iomgr = sim.io_manager();
        iomgr.register_io_var(fields.field.name());
        iomgr.register_output_var(kappa.name());

        return fields;
    }

    CFDSim& sim;
};

/** Right-hand side (RHS) evaluator for VOF transport equation
 */
template <typename Scheme>
struct ComputeRHSOp<VOF, Scheme>
{
    explicit ComputeRHSOp(PDEFields& fields_in) : fields(fields_in) {}

    void predictor_rhs(
        const DiffusionType /*unused*/,
        const amrex::Real /*unused*/,
        bool /*unused*/)
    {}

    void corrector_rhs(
        const DiffusionType /*unused*/,
        const amrex::Real /*unused*/,
        bool /*unused*/)
    {}

    void improve_explicit_diff(const amrex::Real /*unused*/) {}

    // data members
    PDEFields& fields;
};

/** Implementation of post-processing actions for the VOF equation
 *  \ingroup vof
 */

template <>
struct PostSolveOp<VOF>
{
    PostSolveOp(CFDSim& sim, PDEFields& fields) : m_sim(sim), m_fields(fields)
    {
        amrex::ParmParse pp("VOF");
        pp.query("height_function", m_use_HF);
        pp.query("height_function_file", m_HF_file);
        if (pp.contains("reference_curvature")) {
            pp.get("reference_curvature", m_kappa_ref);
            m_compare_kappa = true;
        }
    }

    void operator()(const amrex::Real time)
    {
        // Fill three VOF ghost layers on every level. At a coarse-fine
        // boundary, FieldFillPatchOps uses AMReX FillPatchTwoLevels.
        m_fields.field.fillpatch(m_sim.time().new_time());
        auto& multiphase = m_sim.physics_manager().get<MultiPhase>();
        multiphase.set_density_via_vof();

        // Calculate n+1/2 density
        multiphase.set_nph_density();
        // Convert advalpha to advrho
        multiphase.calculate_advected_facedensity();

        if (m_use_HF) {
            auto& repo = m_sim.repo();
            auto& kappa = repo.get_field("kappa");
            auto& kappa_exact = repo.get_field("kappa_exact");
            auto& kappa_error = repo.get_field("kappa_error");
            auto& iflag = repo.get_field("iflag");
            multiphase::compute_height_function_fields(
                m_fields.field, kappa, kappa_exact, kappa_error, iflag,
                m_kappa_ref, time);

            if (m_compare_kappa) {
                const auto stats = multiphase::height_function_statistics(
                    m_fields.field, kappa, kappa_error, iflag);
                write_HF_diagnostics(time, stats);
            }
        }
    }

    void write_HF_diagnostics(
        const amrex::Real time, const multiphase::HeightFunctionStats& stats)
    {
        const int step = m_sim.time().time_index();
        amrex::Print() << std::setprecision(17) << "HF_CURVATURE " << step << " "
                       << time << " " << stats.interface_cells << " "
                       << stats.iflag_cells << " "
                       << stats.missing_interface_cells << " "
                       << stats.direct_cells << " "
                       << stats.extrapolated_cells << " " << m_kappa_ref << " "
                       << stats.kappa_mean
                       << " " << stats.l1_error << " " << stats.l2_error << " "
                       << stats.linf_error << '\n';

        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::ios_base::openmode mode = std::ios::out;
            if (m_diagnostics_initialized) {
                mode = std::ios::out | std::ios::app;
            }
            std::ofstream output(m_HF_file, mode);
            output << std::setprecision(17);
            if (!m_diagnostics_initialized) {
                output << "step,time,interface_cells,iflag_cells,"
                          "missing_interface_cells,direct_cells,"
                          "extrapolated_cells,kappa_exact,kappa_mean,"
                          "l1_error,l2_error,linf_error\n";
            }
            output << step << ',' << time << ',' << stats.interface_cells << ','
                   << stats.iflag_cells << ','
                   << stats.missing_interface_cells << ',' << stats.direct_cells
                   << ',' << stats.extrapolated_cells << ',' << m_kappa_ref << ','
                   << stats.kappa_mean << ','
                   << stats.l1_error << ',' << stats.l2_error << ','
                   << stats.linf_error << '\n';
        }
        m_diagnostics_initialized = true;
    }

    CFDSim& m_sim;
    PDEFields& m_fields;
    bool m_use_HF{false};
    bool m_compare_kappa{false};
    bool m_diagnostics_initialized{false};
    amrex::Real m_kappa_ref{0.0};
    std::string m_HF_file{"curvature_error.csv"};
};

} // namespace kynema_sgf::pde

#endif /* VOF_OPS_H */
