#ifndef VOF_OPS_H
#define VOF_OPS_H

#include <fstream>
#include <iomanip>
#include "src/equation_systems/vof/vof_curv_HF.H"
#include "src/equation_systems/vof/vof_lamb_prosperetti.H"
#include "src/equation_systems/vof/vof_spurious_current.H"
#include "src/equation_systems/vof/vof_tension.H"
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
        amrex::ParmParse pp("VOF");
        bool surface_tension = false;
        pp.query("surface_tension", surface_tension);
        if (surface_tension) {
            auto st = repo.declare_face_normal_field({"stx", "sty", "stz"});
            for (auto* st_dir : st) {
                st_dir->set_default_fillpatch_bc(sim.time());
            }
            auto& surface_tension_force =
                repo.declare_cc_field("surface_tension", AMREX_SPACEDIM, 0, 1);
            surface_tension_force.set_default_fillpatch_bc(sim.time());
        }
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
        pp.query("surface_tension", m_use_ST);
        pp.query("lamb_prosperetti", m_use_LP);
        pp.query("spurious_current_validation", m_use_spurious_current);
        pp.query("spurious_current", m_use_spurious_current);
        if (pp.contains("sigma")) {
            pp.get("sigma", m_sigma);
            m_has_sigma = true;
        }
        if (pp.contains("surface_tension_coefficient")) {
            pp.get("surface_tension_coefficient", m_sigma);
            m_has_sigma = true;
        }
        pp.query("height_function_file", m_HF_file);
        if (pp.contains("reference_curvature")) {
            pp.get("reference_curvature", m_kappa_ref);
            m_compare_kappa = true;
        }
        if (m_use_ST) {
            m_use_HF = true;
        }
        if (m_use_ST && !m_has_sigma) {
            amrex::Abort(
                "VOF.surface_tension requires VOF.sigma or "
                "VOF.surface_tension_coefficient");
        }
        if (m_use_spurious_current && !m_has_sigma) {
            amrex::Abort(
                "VOF.spurious_current_validation requires VOF.sigma or "
                "VOF.surface_tension_coefficient");
        }
        if (m_use_spurious_current) {
            amrex::ParmParse pp_rain("RainDrop");
            pp_rain.query("radius", m_spurious_params.radius);
            pp.query("spurious_current_eps", m_spurious_params.eps_c);
            pp.query(
                "spurious_current_liquid_min", m_spurious_params.c_liq_min);
            pp.query("spurious_current_radius", m_spurious_params.radius);
            pp.query("spurious_current_rho_ref", m_spurious_params.rho_ref);
        }
        if (m_use_LP) {
            if (!m_has_sigma) {
                amrex::Abort(
                    "VOF.lamb_prosperetti requires VOF.sigma or "
                    "VOF.surface_tension_coefficient");
            }

            amrex::Vector<amrex::Real> center(AMREX_SPACEDIM);
            center[0] = 0.0;
            center[1] = 0.0;
            center[2] = 0.0;
            amrex::Real radius = 0.5e-3;
            bool fluid1_inside = true;
            amrex::ParmParse pp_rain("RainDrop");
            pp_rain.queryarr("location", center, 0, AMREX_SPACEDIM);
            pp_rain.query("radius", radius);
            pp_rain.query("fluid1_inside", fluid1_inside);
            m_LP_params = multiphase::lamb_prosperetti_params(
                center, radius, fluid1_inside);
        }
    }

    void operator()(const amrex::Real time)
    {
        auto& repo = m_sim.repo();
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
            auto& kappa = repo.get_field("kappa");
            auto& kappa_exact = repo.get_field("kappa_exact");
            auto& kappa_error = repo.get_field("kappa_error");
            auto& iflag = repo.get_field("iflag");
            if (m_use_ST) {
                auto& density = repo.get_field("density");
                auto& stx = repo.get_field("stx");
                auto& sty = repo.get_field("sty");
                auto& stz = repo.get_field("stz");
                auto& st = repo.get_field("surface_tension");
                multiphase::tension(
                    m_fields.field, density, kappa, iflag, stx, sty, stz, st,
                    m_sigma, multiphase.rho1(), multiphase.rho2(), time);
            } else {
                multiphase::curv_HF(
                    m_fields.field, kappa, iflag, time);
            }

            if (m_compare_kappa) {
                multiphase::calc_HF_kappa_error(
                    kappa, kappa_exact, kappa_error, iflag, m_kappa_ref);
                const auto stats = multiphase::height_function_statistics(
                    m_fields.field, kappa, kappa_exact, kappa_error, iflag);
                write_HF_diagnostics(time, stats);
            }
        }

        if (m_use_LP) {
            multiphase::lamb_prosperetti(
                m_fields.field, time, m_sim.time().time_index(), m_sigma,
                multiphase.rho1(), multiphase.rho2(), m_LP_params, m_LP_state);
        }

        if (m_use_spurious_current) {
            auto& u_mac = m_sim.repo().get_field("u_mac");
            auto& v_mac = m_sim.repo().get_field("v_mac");
            auto& w_mac = m_sim.repo().get_field("w_mac");
            m_spurious_params.sigma = m_sigma;
            if (m_spurious_params.rho_ref <= 0.0) {
                m_spurious_params.rho_ref = multiphase.rho1();
            }
            const auto stats = multiphase::spurious_current(
                m_fields.field, u_mac, v_mac, w_mac, m_spurious_params);
            write_spurious_current(time, stats);
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
                          "relative_l1_error,relative_l2_error,"
                          "relative_linf_error\n";
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

    void write_spurious_current(
        const amrex::Real time,
        const multiphase::SpuriousCurrentStats& stats)
    {
        const int step = m_sim.time().time_index();
        amrex::Print() << std::setprecision(17)
                       << "SPURIOUS_CURRENT:"
                       << " region "
                       << "liquid_side_interface_cells_cell_centered_velocity"
                       << " c_liq_cell " << m_spurious_params.c_liq_min
                       << " time " << time
                       << " step " << step
                       << " dx_min " << stats.dx_min
                       << " radius " << stats.radius
                       << " R_over_dx " << stats.R_over_dx
                       << " dx_over_R " << stats.dx_over_R
                       << " U_sigma " << stats.U_sigma
                       << " count " << stats.count
                       << " skipped_no_liq_face "
                       << stats.skipped_no_liq_face
                       << " vol " << stats.vol
                       << " mean_abs " << stats.mean_abs
                       << " L1_avg " << stats.L1_avg
                       << " L2_rms " << stats.L2_rms
                       << " Linf " << stats.Linf
                       << " L1_int " << stats.L1_int
                       << " L2_int " << stats.L2_int
                       << " mean_abs_nd " << stats.mean_abs_nd
                       << " L1_avg_nd " << stats.L1_avg_nd
                       << " L2_rms_nd " << stats.L2_rms_nd
                       << " Linf_nd " << stats.Linf_nd << '\n';
    }

    CFDSim& m_sim;
    PDEFields& m_fields;
    bool m_use_HF{false};
    bool m_use_ST{false};
    bool m_use_LP{false};
    bool m_use_spurious_current{false};
    bool m_has_sigma{false};
    bool m_compare_kappa{false};
    bool m_diagnostics_initialized{false};
    amrex::Real m_kappa_ref{0.0};
    amrex::Real m_sigma{0.0};
    std::string m_HF_file{"curvature_error.csv"};
    multiphase::LambProsperettiParams m_LP_params;
    multiphase::LambProsperettiState m_LP_state;
    multiphase::SpuriousCurrentParams m_spurious_params;
};

} // namespace kynema_sgf::pde

#endif /* VOF_OPS_H */
