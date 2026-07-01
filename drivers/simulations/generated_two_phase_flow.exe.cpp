#include "kernel_diagnostics.hpp"
#include "sfem_API.hpp"
#include "sfem_DirichletConditions.hpp"
#include "sfem_GeneratedTwoPhaseFlow.hpp"
#include "sfem_TwoPhaseFlowTimeIntegration.hpp"
#include "smesh_env.hpp"
#include "smesh_sideset.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>

extern "C" {
#define DECLARE_DIAGNOSTICS(element)                                                                                     \
    const sfem::codegen::KernelDiagnostics *two_phase_flow_##element##_residual_element_soa_diagnostics(void); \
    const sfem::codegen::KernelDiagnostics *two_phase_flow_##element##_jacobian_action_element_soa_diagnostics(void)

DECLARE_DIAGNOSTICS(tri3);
DECLARE_DIAGNOSTICS(tet4);
DECLARE_DIAGNOSTICS(quad4);
DECLARE_DIAGNOSTICS(hex8);
#undef DECLARE_DIAGNOSTICS
}

namespace {
    using Diagnostics = sfem::codegen::KernelDiagnostics;

    struct DiagnosticTotals {
        double flops{0};
        size_t bytes{0};
    };

    struct OpStats {
        double    residual_seconds{0};
        double    jacobian_seconds{0};
        ptrdiff_t residual_calls{0};
        ptrdiff_t jacobian_calls{0};

        void reset() { *this = {}; }
    };

    struct PhaseBalance {
        real_t left[2]{0, 0};
        real_t right[2]{0, 0};
        real_t interior[2]{0, 0};
        real_t total[2]{0, 0};
    };

    PhaseBalance phase_balance(const real_t *const              residual,
                               const ptrdiff_t                  nnodes,
                               const sfem::SharedBuffer<idx_t> &left,
                               const sfem::SharedBuffer<idx_t> &right) {
        PhaseBalance result;
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            result.total[0] += residual[2 * node + 0];
            result.total[1] += residual[2 * node + 1];
        }
        for (size_t i = 0; i < left->size(); ++i) {
            const idx_t node = left->data()[i];
            result.left[0] += residual[2 * node + 0];
            result.left[1] += residual[2 * node + 1];
        }
        for (size_t i = 0; i < right->size(); ++i) {
            const idx_t node = right->data()[i];
            result.right[0] += residual[2 * node + 0];
            result.right[1] += residual[2 * node + 1];
        }
        result.interior[0] = result.total[0] - result.left[0] - result.right[0];
        result.interior[1] = result.total[1] - result.left[1] - result.right[1];
        return result;
    }

    bool admissible(const real_t *const state, const ptrdiff_t nnodes) {
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            const real_t pw = state[2 * node + 0];
            const real_t pc = state[2 * node + 1];
            if (!std::isfinite(pw) || !std::isfinite(pc) || pw <= 0 || pc <= pw) {
                return false;
            }
        }
        return true;
    }

    const Diagnostics *kernel_diagnostics(const smesh::ElemType type, const bool jacobian) {
        switch (type) {
            case smesh::TRI3:
                return jacobian ? two_phase_flow_tri3_jacobian_action_element_soa_diagnostics()
                                : two_phase_flow_tri3_residual_element_soa_diagnostics();
            case smesh::TET4:
                return jacobian ? two_phase_flow_tet4_jacobian_action_element_soa_diagnostics()
                                : two_phase_flow_tet4_residual_element_soa_diagnostics();
            case smesh::QUAD4:
                return jacobian ? two_phase_flow_quad4_jacobian_action_element_soa_diagnostics()
                                : two_phase_flow_quad4_residual_element_soa_diagnostics();
            case smesh::HEX8:
                return jacobian ? two_phase_flow_hex8_jacobian_action_element_soa_diagnostics()
                                : two_phase_flow_hex8_residual_element_soa_diagnostics();
            default:
                return nullptr;
        }
    }

    DiagnosticTotals diagnostic_totals(const std::shared_ptr<sfem::Mesh> &mesh, const bool jacobian) {
        DiagnosticTotals totals;
        for (const auto &block : mesh->blocks()) {
            const Diagnostics *const diagnostics = kernel_diagnostics(block->element_type(), jacobian);
            if (!diagnostics) {
                continue;
            }
            totals.flops += sfem::codegen::KernelDiagnostics_total_flops(diagnostics, block->n_elements());
            totals.bytes += sfem::codegen::KernelDiagnostics_total_bytes(
                    diagnostics, block->n_elements(), sizeof(real_t), sizeof(real_t), sizeof(real_t));
        }
        return totals;
    }

    void compute_capillary_and_water_saturation(const real_t *const state,
                                                const ptrdiff_t     nnodes,
                                                const real_t        s_res,
                                                const real_t        p_r,
                                                const real_t        m,
                                                real_t *const       suction,
                                                real_t *const       water_saturation) {
        const real_t one_minus_s_res = 1.0 - s_res;
        const real_t exponent        = 1.0 / m - 1.0;
        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            const real_t pw        = state[2 * node + 0];
            const real_t pc        = state[2 * node + 1];
            const real_t s         = pc - pw;
            const real_t x         = s / p_r;
            suction[node]          = s;
            water_saturation[node] = s_res + one_minus_s_res * std::pow(1.0 + std::pow(x, m), exponent);
        }
    }

    void write_performance(FILE *const            output,
                           const char *const      sample,
                           const ptrdiff_t        step,
                           const real_t           time,
                           const char *const      name,
                           const double           seconds,
                           const ptrdiff_t        calls,
                           const ptrdiff_t        nelements,
                           const ptrdiff_t        ndofs,
                           const DiagnosticTotals totals) {
        const double evaluations  = static_cast<double>(calls);
        const double element_rate = seconds > 0 ? evaluations * nelements / seconds : 0;
        const double dof_rate     = seconds > 0 ? evaluations * ndofs / seconds : 0;
        const double flop_rate    = seconds > 0 ? evaluations * totals.flops / seconds : 0;
        const double ai           = totals.bytes ? totals.flops / static_cast<double>(totals.bytes) : 0;
        if (output) {
            fprintf(output,
                    "%s,%td,%.17g,%s,%td,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    sample,
                    step,
                    static_cast<double>(time),
                    name,
                    calls,
                    seconds,
                    element_rate,
                    dof_rate,
                    flop_rate,
                    ai);
        }
    }
}  // namespace

int main(int argc, char *argv[]) {
    auto context = sfem::initialize_serial(argc, argv);
    auto comm    = context->communicator();
    if (argc != 3) {
        fprintf(stderr, "usage: %s <mesh|GENERATE> <output>\n", argv[0]);
        return SFEM_FAILURE;
    }

    std::shared_ptr<sfem::Mesh> mesh;
    if (strcmp(argv[1], "GENERATE") == 0) {
        const ptrdiff_t nx = smesh::Env::read("SFEM_NX", 400);
        const ptrdiff_t ny = smesh::Env::read("SFEM_NY", 5);
        const ptrdiff_t nz = smesh::Env::read("SFEM_NZ", 5);
        mesh               = sfem::Mesh::create_hex8_cube(comm, nx, ny, nz, 0, 0, 0, 4000, 50, 50);
    } else {
        mesh = sfem::Mesh::create_from_file(comm, smesh::Path(argv[1]));
    }
    if (!mesh) {
        return SFEM_FAILURE;
    }

    const smesh::Path output(argv[2]);
    smesh::create_directory(output);
    mesh->write(output / "mesh");
    FILE *const nonlinear_output   = fopen((output / "nonlinear_history.csv").c_str(), "w");
    FILE *const balance_output     = fopen((output / "mass_balance.csv").c_str(), "w");
    FILE *const performance_output = fopen((output / "performance.csv").c_str(), "w");
    if (!nonlinear_output || !balance_output || !performance_output) {
        fprintf(stderr, "unable to create run output files in %s\n", output.c_str());
        return SFEM_FAILURE;
    }
    fprintf(nonlinear_output,
            "event,step,time,dt,nonlinear_iteration,linear_iterations,residual,"
            "merit,damping,rejected_updates,rejected_steps\n");
    fprintf(balance_output, "step,time,phase,left_flux,right_flux,interior_error,total\n");
    fprintf(performance_output,
            "sample,step,time,kernel,calls,seconds,elements_per_second,dofs_per_second,"
            "flops_per_second,arithmetic_intensity\n");

    const real_t initial_water_pressure = smesh::Env::read("SFEM_INITIAL_WATER_PRESSURE", 15);
    const real_t initial_co2_pressure   = smesh::Env::read("SFEM_INITIAL_CO2_PRESSURE", 15.1);
    const real_t injection_co2_pressure = smesh::Env::read("SFEM_INJECTION_CO2_PRESSURE", 20);
    const real_t ramp_duration          = smesh::Env::read("SFEM_RAMP_DURATION", 1.0);
    const std::string codegen_geometry  = smesh::Env::read_string("SFEM_CODEGEN_GEOMETRY", "isoparametric");
    if (codegen_geometry != "affine" && codegen_geometry != "isoparametric") {
        SFEM_ERROR("SFEM_CODEGEN_GEOMETRY must be affine or isoparametric\n");
    }
    const bool assume_affine = codegen_geometry == "affine";
    auto         space                  = sfem::FunctionSpace::create(mesh, 2);
    auto         initial_state          = sfem::create_host_buffer<real_t>(space->n_dofs());
    for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
        initial_state->data()[2 * node + 0] = initial_water_pressure;
        initial_state->data()[2 * node + 1] = initial_co2_pressure;
    }
    const auto   bounds             = mesh->compute_bounding_box();
    const geom_t xmin               = bounds.first->data()[0];
    const geom_t xmax               = bounds.second->data()[0];
    const geom_t selector_tolerance = std::max<geom_t>(1, std::abs(xmax - xmin)) * 64 * std::numeric_limits<geom_t>::epsilon();

    auto left_sidesets = sfem::Sideset::create_from_selector(
            mesh, [=](const geom_t x, const geom_t, const geom_t) { return std::abs(x - xmin) <= selector_tolerance; });

    auto right_sidesets = sfem::Sideset::create_from_selector(
            mesh, [=](const geom_t x, const geom_t, const geom_t) { return std::abs(x - xmax) <= selector_tolerance; });

    std::vector<sfem::DirichletConditions::Condition> conditions(4);
    conditions[0]    = {.sidesets = left_sidesets, .value = initial_water_pressure, .component = 0};
    conditions[1]    = {.sidesets = left_sidesets, .value = initial_co2_pressure, .component = 1};
    conditions[2]    = {.sidesets = right_sidesets, .value = initial_water_pressure, .component = 0};
    conditions[3]    = {.sidesets = right_sidesets, .value = initial_co2_pressure, .component = 1};
    auto dirichlet   = sfem::DirichletConditions::create(space, conditions);
    auto left_nodes  = dirichlet->conditions()[0].nodeset;
    auto right_nodes = dirichlet->conditions()[2].nodeset;

    auto update_boundary = [=](const real_t time, sfem::DirichletConditions &boundary) {
        const real_t ramp              = ramp_duration > 0 ? std::min<real_t>(1, std::max<real_t>(0, time / ramp_duration)) : 1;
        boundary.conditions()[1].value = initial_co2_pressure + ramp * (injection_co2_pressure - initial_co2_pressure);
    };
    sfem::TwoPhaseFlowTimeIntegration time_integration(mesh, initial_state, dirichlet, update_boundary);
    const char                       *SFEM_RESTART = nullptr;
    SFEM_READ_ENV(SFEM_RESTART, );
    if (SFEM_RESTART) {
        if (time_integration.load_restart(smesh::Path(SFEM_RESTART)) != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to load restart from %s\n", SFEM_RESTART);
        }
    } else {
        time_integration.initialize();
    }

    auto op = std::make_shared<sfem::GeneratedTwoPhaseFlow>(space);
    op->set_option("assume_affine", assume_affine);
    op->set_option("ASSUME_AFFINE", assume_affine);
    if (op->initialize() != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }

    const real_t    initial_dt                   = smesh::Env::read("SFEM_DT", 0.05);
    const real_t    minimum_dt                   = smesh::Env::read("SFEM_MIN_DT", initial_dt / 2048);
    const real_t    end_time                     = smesh::Env::read("SFEM_T_END", 600);
    const int       nonlinear_max_it             = smesh::Env::read("SFEM_NL_MAX_IT", 20);
    const real_t    nonlinear_atol               = smesh::Env::read("SFEM_NL_ATOL", 1e-12);
    const real_t    nonlinear_rtol               = smesh::Env::read("SFEM_NL_RTOL", 1e-10);
    const int       linear_max_it                = smesh::Env::read("SFEM_LS_MAX_IT", 14000);
    const real_t    linear_atol                  = smesh::Env::read("SFEM_LS_ATOL", 1e-12);
    const real_t    linear_rtol                  = smesh::Env::read("SFEM_LS_RTOL", 1e-8);
    const real_t    armijo                       = smesh::Env::read("SFEM_ARMIJO", 1e-4);
    const real_t    minimum_damping              = smesh::Env::read("SFEM_MIN_DAMPING", 1.0 / 1024);
    const ptrdiff_t checkpoint_frequency         = smesh::Env::read("SFEM_CHECKPOINT_FREQUENCY", 1);
    const int       benchmark_repeats            = smesh::Env::read("SFEM_BENCHMARK_REPEATS", 10);
    const ptrdiff_t performance_report_frequency = smesh::Env::read("SFEM_PERFORMANCE_REPORT_FREQUENCY", 1);
    printf("geometry %s\n", codegen_geometry.c_str());
    printf("output nonlinear=%s balance=%s performance=%s\n",
           (output / "nonlinear_history.csv").c_str(),
           (output / "mass_balance.csv").c_str(),
           (output / "performance.csv").c_str());
    for (const auto &entry : mesh->blocks()) {
        op->set_value_in_block(entry->name(), "dt", initial_dt);
    }

    auto         blas               = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);
    auto         residual           = sfem::create_host_buffer<real_t>(space->n_dofs());
    auto         candidate_residual = sfem::create_host_buffer<real_t>(space->n_dofs());
    auto         increment          = sfem::create_host_buffer<real_t>(space->n_dofs());
    auto         candidate          = sfem::create_host_buffer<real_t>(space->n_dofs());
    const real_t retention_s_res    = smesh::Env::read("SFEM_S_RES", 0.39);
    const real_t retention_p_r      = smesh::Env::read("SFEM_P_R", 9.5e4 / 1.0e6);
    const real_t retention_m        = smesh::Env::read("SFEM_M", 4.2);

    auto scalar_space    = sfem::FunctionSpace::create(mesh, 1);
    auto output_function = sfem::Function::create(space);
    output_function->output()->set_output_dir(output / "out");
    output_function->output()->enable_AoS_to_SoA(true);
    auto scalar_output_function = sfem::Function::create(scalar_space);
    scalar_output_function->output()->set_output_dir(output / "out");

    auto       suction                    = sfem::create_host_buffer<real_t>(mesh->n_nodes());
    auto       water_saturation           = sfem::create_host_buffer<real_t>(mesh->n_nodes());
    const auto write_postprocessed_fields = [&](const real_t time, const real_t *const state) {
        compute_capillary_and_water_saturation(
                state, mesh->n_nodes(), retention_s_res, retention_p_r, retention_m, suction->data(), water_saturation->data());
        scalar_output_function->output()->write_time_step("s", time, suction->data());
        scalar_output_function->output()->write_time_step("S_w", time, water_saturation->data());
    };

    output_function->output()->write_time_step("pressure", time_integration.time(), time_integration.accepted()->data());
    write_postprocessed_fields(time_integration.time(), time_integration.accepted()->data());
    OpStats    op_stats;
    const auto timed_gradient = [&](const real_t *const state, real_t *const out) {
        const auto begin  = std::chrono::steady_clock::now();
        const int  status = op->gradient(state, out);
        op_stats.residual_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        ++op_stats.residual_calls;
        return status;
    };
    const auto residual_diagnostics = diagnostic_totals(mesh, false);
    const auto jacobian_diagnostics = diagnostic_totals(mesh, true);
    const auto solve_begin          = std::chrono::steady_clock::now();
    int        rejected_steps       = 0;

    const auto newton_step = [&](const real_t *previous, real_t *trial, const real_t time, const real_t step_dt) {
        for (const auto &entry : mesh->blocks()) {
            op->set_value_in_block(entry->name(), "dt", step_dt);
        }
        op->update(previous, trial);
        real_t initial_norm = -1;

        for (int nonlinear_it = 0; nonlinear_it < nonlinear_max_it; ++nonlinear_it) {
            blas->zeros(residual->size(), residual->data());
            if (timed_gradient(trial, residual->data()) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }
            time_integration.constrain_residual(trial, residual->data());
            const real_t residual_norm = blas->norm2(residual->size(), residual->data());
            const real_t merit         = 0.5 * residual_norm * residual_norm;
            if (initial_norm < 0) {
                initial_norm = residual_norm;
            }
            const real_t tolerance = std::max(nonlinear_atol, nonlinear_rtol * initial_norm);
            if (residual_norm <= tolerance) {
                fprintf(nonlinear_output,
                        "converged,%td,%.17g,%.17g,%d,0,%.17g,%.17g,0,0,%d\n",
                        time_integration.step() + 1,
                        static_cast<double>(time),
                        static_cast<double>(step_dt),
                        nonlinear_it,
                        static_cast<double>(residual_norm),
                        static_cast<double>(merit),
                        rejected_steps);
                return SFEM_SUCCESS;
            }

            const ptrdiff_t ndofs     = space->n_dofs();
            auto            linear_op = sfem::make_op<real_t>(
                    ndofs,
                    ndofs,
                    [&](const real_t *const direction, real_t *const output) {
                        std::fill(output, output + ndofs, static_cast<real_t>(0));
                        const auto begin = std::chrono::steady_clock::now();
                        const int  err   = op->apply(nullptr, direction, output);
                        op_stats.jacobian_seconds +=
                                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
                        ++op_stats.jacobian_calls;
                        if (err != SFEM_SUCCESS) {
                            SFEM_ERROR("GeneratedTwoPhaseFlow Jacobian action failed\n");
                        }
                        time_integration.constrain_linear(direction, output);
                    },
                    sfem::EXECUTION_SPACE_HOST);
            auto bcgs     = sfem::create_bcgs<real_t>(linear_op, sfem::EXECUTION_SPACE_HOST);
            bcgs->verbose = smesh::Env::read("SFEM_VERBOSE", true);
            bcgs->set_max_it(linear_max_it);
            bcgs->set_atol(std::max(linear_atol, linear_rtol * residual_norm));
            blas->zeros(increment->size(), increment->data());
            if (bcgs->apply(residual->data(), increment->data()) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }
            time_integration.constrain_direction(increment->data());

            real_t damping          = 1;
            int    rejected_updates = 0;
            bool   accepted_update  = false;
            real_t candidate_norm   = residual_norm;
            real_t candidate_merit  = merit;
            while (damping >= minimum_damping) {
                blas->copy(candidate->size(), trial, candidate->data());
                blas->axpy(candidate->size(), -damping, increment->data(), candidate->data());
                time_integration.apply_boundary(time, candidate->data());
                if (admissible(candidate->data(), mesh->n_nodes())) {
                    blas->zeros(candidate_residual->size(), candidate_residual->data());
                    op->update(previous, candidate->data());
                    if (timed_gradient(candidate->data(), candidate_residual->data()) == SFEM_SUCCESS) {
                        time_integration.constrain_residual(candidate->data(), candidate_residual->data());
                        candidate_norm  = blas->norm2(candidate_residual->size(), candidate_residual->data());
                        candidate_merit = 0.5 * candidate_norm * candidate_norm;
                        if (candidate_merit <= (1 - armijo * damping) * merit) {
                            accepted_update = true;
                            break;
                        }
                    }
                }
                damping *= 0.5;
                ++rejected_updates;
            }
            fprintf(nonlinear_output,
                    "update,%td,%.17g,%.17g,%d,%d,%.17g,%.17g,%.17g,%d,%d\n",
                    time_integration.step() + 1,
                    static_cast<double>(time),
                    static_cast<double>(step_dt),
                    nonlinear_it,
                    bcgs->iterations(),
                    static_cast<double>(candidate_norm),
                    static_cast<double>(candidate_merit),
                    static_cast<double>(damping),
                    rejected_updates,
                    rejected_steps);
            if (!accepted_update) {
                return SFEM_FAILURE;
            }

            printf("%d) residual norm: %.17g\n", nonlinear_it, static_cast<double>(candidate_norm));
            printf("%d) merit: %.17g\n", nonlinear_it, static_cast<double>(candidate_merit));

            blas->copy(candidate->size(), candidate->data(), trial);
            op->update(previous, trial);
        }
        return SFEM_FAILURE;
    };

    real_t       dt                 = initial_dt;
    const real_t end_time_tolerance = 64 * std::numeric_limits<real_t>::epsilon() * std::max<real_t>(1, std::abs(end_time));
    while (time_integration.time() + end_time_tolerance < end_time) {
        const real_t step_dt = std::min(dt, end_time - time_integration.time());
        if (time_integration.advance(step_dt, newton_step) != SFEM_SUCCESS) {
            dt = step_dt * 0.5;
            ++rejected_steps;
            fprintf(nonlinear_output,
                    "reject,%td,%.17g,%.17g,-1,-1,nan,nan,nan,0,%d\n",
                    time_integration.step() + 1,
                    static_cast<double>(time_integration.time()),
                    static_cast<double>(step_dt),
                    rejected_steps);
            fflush(nonlinear_output);
            if (dt < minimum_dt) {
                fprintf(stderr, "time step fell below SFEM_MIN_DT\n");
                return SFEM_FAILURE;
            }
            continue;
        }
        fprintf(nonlinear_output,
                "accept,%td,%.17g,%.17g,-1,-1,nan,nan,nan,0,%d\n",
                time_integration.step(),
                static_cast<double>(time_integration.time()),
                static_cast<double>(step_dt),
                rejected_steps);
        fflush(nonlinear_output);
        blas->zeros(residual->size(), residual->data());
        op->update(time_integration.trial()->data(), time_integration.accepted()->data());
        if (timed_gradient(time_integration.accepted()->data(), residual->data()) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        const auto balance = phase_balance(residual->data(), mesh->n_nodes(), left_nodes, right_nodes);
        for (int phase = 0; phase < 2; ++phase) {
            const char *const phase_name = phase == 0 ? "water" : "co2";
            fprintf(balance_output,
                    "%td,%.17g,%s,%.17g,%.17g,%.17g,%.17g\n",
                    time_integration.step(),
                    static_cast<double>(time_integration.time()),
                    phase_name,
                    static_cast<double>(balance.left[phase]),
                    static_cast<double>(balance.right[phase]),
                    static_cast<double>(balance.interior[phase]),
                    static_cast<double>(balance.total[phase]));
        }
        fflush(balance_output);
        if (performance_report_frequency > 0 && time_integration.step() % performance_report_frequency == 0) {
            const auto step_stats = op_stats;
            write_performance(performance_output,
                              "accepted_step",
                              time_integration.step(),
                              time_integration.time(),
                              "residual",
                              step_stats.residual_seconds,
                              step_stats.residual_calls,
                              mesh->n_elements(),
                              space->n_dofs(),
                              residual_diagnostics);
            write_performance(performance_output,
                              "accepted_step",
                              time_integration.step(),
                              time_integration.time(),
                              "jacobian",
                              step_stats.jacobian_seconds,
                              step_stats.jacobian_calls,
                              mesh->n_elements(),
                              space->n_dofs(),
                              jacobian_diagnostics);
            fflush(performance_output);
        }
        dt = std::min(initial_dt, step_dt * 2);
        output_function->output()->write_time_step("pressure", time_integration.time(), time_integration.accepted()->data());
        write_postprocessed_fields(time_integration.time(), time_integration.accepted()->data());
        output_function->output()->log_time(time_integration.time());
        if (checkpoint_frequency > 0 && time_integration.step() % checkpoint_frequency == 0) {
            time_integration.save_restart(output / "restart");
        }
    }

    const auto   solve_end     = std::chrono::steady_clock::now();
    const double solve_seconds = std::chrono::duration<double>(solve_end - solve_begin).count();
    const auto   solve_stats   = op_stats;
    fprintf(performance_output,
            "final,%td,%.17g,solve_wall_time,%td,%.17g,0,0,0,0\n",
            time_integration.step(),
            static_cast<double>(time_integration.time()),
            static_cast<ptrdiff_t>(1),
            solve_seconds);
    write_performance(performance_output,
                      "final",
                      time_integration.step(),
                      time_integration.time(),
                      "residual",
                      solve_stats.residual_seconds,
                      solve_stats.residual_calls,
                      mesh->n_elements(),
                      space->n_dofs(),
                      residual_diagnostics);
    write_performance(performance_output,
                      "final",
                      time_integration.step(),
                      time_integration.time(),
                      "jacobian",
                      solve_stats.jacobian_seconds,
                      solve_stats.jacobian_calls,
                      mesh->n_elements(),
                      space->n_dofs(),
                      jacobian_diagnostics);

    if (benchmark_repeats > 0) {
        auto direction = sfem::create_host_buffer<real_t>(space->n_dofs());
        auto action    = sfem::create_host_buffer<real_t>(space->n_dofs());
        for (ptrdiff_t i = 0; i < direction->size(); ++i) {
            direction->data()[i] = static_cast<real_t>((i % 17) - 8) * 1e-4;
        }
        op->update(time_integration.trial()->data(), time_integration.accepted()->data());
        OpStats benchmark_stats;
        for (int repeat = 0; repeat < benchmark_repeats; ++repeat) {
            blas->zeros(residual->size(), residual->data());
            const auto begin = std::chrono::steady_clock::now();
            op->gradient(time_integration.accepted()->data(), residual->data());
            benchmark_stats.residual_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            ++benchmark_stats.residual_calls;
        }
        for (int repeat = 0; repeat < benchmark_repeats; ++repeat) {
            blas->zeros(action->size(), action->data());
            const auto begin = std::chrono::steady_clock::now();
            op->apply(time_integration.accepted()->data(), direction->data(), action->data());
            benchmark_stats.jacobian_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            ++benchmark_stats.jacobian_calls;
        }
        write_performance(performance_output,
                          "benchmark",
                          time_integration.step(),
                          time_integration.time(),
                          "residual",
                          benchmark_stats.residual_seconds,
                          benchmark_stats.residual_calls,
                          mesh->n_elements(),
                          space->n_dofs(),
                          residual_diagnostics);
        write_performance(performance_output,
                          "benchmark",
                          time_integration.step(),
                          time_integration.time(),
                          "jacobian",
                          benchmark_stats.jacobian_seconds,
                          benchmark_stats.jacobian_calls,
                          mesh->n_elements(),
                          space->n_dofs(),
                          jacobian_diagnostics);
    }

    fclose(nonlinear_output);
    fclose(balance_output);
    fclose(performance_output);
    return time_integration.save_restart(output / "restart");
}
