#include <math.h>
#include <stdio.h>
#include <chrono>
#include <memory>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_MooneyRivlinVisco.hpp"
#include "sfem_bsr_SpMV.hpp"
#include "sfem_test.h"

std::shared_ptr<sfem::Output> create_output(const std::shared_ptr<sfem::Function> &f, const std::string &output_dir) {
    auto fs = f->space();

    sfem::create_directory(output_dir.c_str());
    auto output = f->output();
    output->enable_AoS_to_SoA(fs->block_size() > 1);
    output->set_output_dir(output_dir.c_str());

    if (fs->has_semi_structured_mesh()) {
        fs->semi_structured_mesh().export_as_standard(output_dir.c_str());
    } else {
        fs->mesh_ptr()->write(output_dir.c_str());
    }
    return output;
}

int test_mooney_rivlin_visco_relaxation() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    // Read environment variables
    int SFEM_BASE_RESOLUTION = 10;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    // 1. Create Mesh (4:1:1 ratio)
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm),
                                             SFEM_BASE_RESOLUTION,
                                             SFEM_BASE_RESOLUTION,
                                             SFEM_BASE_RESOLUTION,  // Grid
                                             0,
                                             0,
                                             0,  // Origin
                                             2,
                                             1,
                                             1  // Dimensions (cube)
    );

    auto fs = sfem::FunctionSpace::create(mesh, 3);  // 3D displacement
    auto f  = sfem::Function::create(fs);

    // Operator
    auto op = std::make_shared<sfem::MooneyRivlinVisco>(fs);

    // LumpedMass
    auto mass_op = sfem::create_op(fs, "LumpedMass", es);
    mass_op->initialize();

    // Material Parameters from environment
    real_t SFEM_C10          = 1.0;
    real_t SFEM_C01          = 0.5;
    real_t SFEM_BULK_MODULUS = 100.0;
    real_t SFEM_DT           = 0.1;
    SFEM_READ_ENV(SFEM_C10, atof);
    SFEM_READ_ENV(SFEM_C01, atof);
    SFEM_READ_ENV(SFEM_BULK_MODULUS, atof);
    SFEM_READ_ENV(SFEM_DT, atof);

    op->set_C10(SFEM_C10);
    op->set_C01(SFEM_C01);
    op->set_K(SFEM_BULK_MODULUS);

    real_t dt = SFEM_DT;
    op->set_dt(dt);

    int SFEM_USE_FLEXIBLE_HESSIAN = 0;
    SFEM_READ_ENV(SFEM_USE_FLEXIBLE_HESSIAN, atoi);
    op->set_use_flexible(SFEM_USE_FLEXIBLE_HESSIAN != 0);

    int SFEM_ENABLE_CONTACT = false;
    SFEM_READ_ENV(SFEM_ENABLE_CONTACT, atoi);

    if (SFEM_USE_FLEXIBLE_HESSIAN) {
        printf("Using FLEXIBLE hessian (loop-based)\n");
    } else {
        printf("Using FIXED hessian (unrolled, 10 Prony terms)\n");
    }

    // 10 Prony terms (for fixed version)
    real_t g_prony[]   = {0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03};
    real_t tau_prony[] = {0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0};
    op->set_prony_terms(10, g_prony, tau_prony);

    op->initialize();
    op->initialize_history();
    f->add_operator(op);

    // BC
    auto left_sideset = sfem::Sideset::create_from_selector(
            mesh, [](const geom_t x, const geom_t, const geom_t) -> bool { return x < 1e-5; });

    auto right_sideset = sfem::Sideset::create_from_selector(
            mesh, [](const geom_t x, const geom_t, const geom_t) -> bool { return x > 1.0 - 1e-5; });

    if (!SFEM_ENABLE_CONTACT) {
        sfem::DirichletConditions::Condition left_bc_x{.sidesets = left_sideset, .value = 0, .component = 0};
        sfem::DirichletConditions::Condition left_bc_y{.sidesets = left_sideset, .value = 0, .component = 1};
        sfem::DirichletConditions::Condition left_bc_z{.sidesets = left_sideset, .value = 0, .component = 2};
        auto                                 conds = sfem::create_dirichlet_conditions(fs, {left_bc_x, left_bc_y, left_bc_z}, es);
        f->add_constraint(conds);
    }

    real_t SFEM_NEUMANN_FORCE = -0.5;
    SFEM_READ_ENV(SFEM_NEUMANN_FORCE, atof);
    sfem::NeumannConditions::Condition right_bc_force{.sidesets = right_sideset, .value = SFEM_NEUMANN_FORCE, .component = 0};
    auto                               neumann_op = sfem::create_neumann_conditions(fs, {right_bc_force}, es);
    f->add_operator(neumann_op);

    const ptrdiff_t ndofs   = fs->n_dofs();
    auto            x       = sfem::create_buffer<real_t>(ndofs, es);
    auto            rhs     = sfem::create_buffer<real_t>(ndofs, es);
    auto            delta_x = sfem::create_buffer<real_t>(ndofs, es);
    auto            diag    = sfem::create_buffer<real_t>(ndofs, es);

    // Lumped mass vector (diagonal of M)
    real_t SFEM_DENSITY = 1000.0;
    SFEM_READ_ENV(SFEM_DENSITY, atof);
    real_t density   = SFEM_DENSITY;
    auto   mass_diag = sfem::create_buffer<real_t>(ndofs, es);
    mass_op->hessian_diag(nullptr, mass_diag->data());
    // Scale mass by density
    auto blas = sfem::blas<real_t>(es);
    blas->scal(ndofs, density, mass_diag->data());
    f->set_value_to_constrained_dofs(1.0, mass_diag->data());  // Set 1 for BC nodes

    // Output setup
    bool SFEM_ENABLE_OUTPUT = false;
    SFEM_READ_ENV(SFEM_ENABLE_OUTPUT, atoi);
    auto output = create_output(f, "test_mooney_rivlin_visco");

    // Newmark state
    auto v      = sfem::create_buffer<real_t>(ndofs, es);
    auto a      = sfem::create_buffer<real_t>(ndofs, es);
    auto u_pred = sfem::create_buffer<real_t>(ndofs, es);  // Predicted displacement
    auto v_pred = sfem::create_buffer<real_t>(ndofs, es);  // Predicted velocity

    blas->zeros(ndofs, x->data());
    blas->zeros(ndofs, v->data());
    blas->zeros(ndofs, a->data());

    // Newmark parameters (implicit, unconditionally stable)
    real_t beta_nm  = 0.25;
    real_t gamma_nm = 0.5;
    real_t c0       = 1.0 / (beta_nm * dt * dt);  // Coefficient for M in effective stiffness

    // Matrix assembly buffers (BSR format: 3x3 blocks)
    auto            graph      = fs->node_to_node_graph();
    const int       block_size = 3;
    const ptrdiff_t n_nodes    = fs->mesh_ptr()->n_nodes();
    auto            values     = sfem::create_buffer<real_t>(graph->nnz() * block_size * block_size, es);

    // Linear Solver Wrapper (BSR SpMV)
    auto linear_op_apply = sfem::make_op<real_t>(
            ndofs,
            ndofs,
            [=](const real_t *const in, real_t *const out) {
                sfem::bsr_spmv<count_t, idx_t, real_t>(n_nodes,
                                                       n_nodes,
                                                       block_size,
                                                       graph->rowptr()->data(),
                                                       graph->colidx()->data(),
                                                       values->data(),
                                                       0.0,
                                                       in,
                                                       out);
            },
            es);

    std::shared_ptr<sfem::MatrixFreeLinearSolver<real_t>> solver;
    auto                                                  jacobi = sfem::create_shiftable_jacobi(diag, es);
    sfem::SharedBuffer<real_t>                            lower_bound;

    if (!SFEM_ENABLE_CONTACT) {
        auto cg = sfem::create_cg<real_t>(linear_op_apply, es);
        cg->set_n_dofs(ndofs);
        cg->set_max_it(2000);
        cg->set_rtol(1e-5);
        cg->verbose = false;

        // Preconditioner (Jacobi)

        cg->set_preconditioner_op(jacobi);

        solver = cg;

    } else {
        auto mprgp = sfem::create_mprgp(linear_op_apply, es);
        lower_bound = sfem::create_buffer<real_t>(ndofs, es);

        {  // Fill default upper-bound value
            auto lb = lower_bound->data();
            for (ptrdiff_t i = 0; i < ndofs; i++) {
                lb[i] = -1000;
            }

            auto      lnodes  = sfem::create_nodeset_from_sideset(fs, left_sideset[0]);
            const ptrdiff_t nbnodes = lnodes->size();
            for (ptrdiff_t i = 0; i < nbnodes; i++) {
                const ptrdiff_t idx  = lnodes->data()[i];
                lb[idx * block_size] = 0;
            }
        }

        mprgp->verbose = false;
        mprgp->set_lower_bound(lower_bound);

        solver = mprgp;
    }

    // FD check removed for cleaner test output

    // 5. Time Loop with Full Newmark Integration
    real_t SFEM_T = 8.0;
    SFEM_READ_ENV(SFEM_T, atof);
    real_t t           = 0;
    size_t steps       = 0;
    size_t export_freq = 1;

    auto f_int        = sfem::create_buffer<real_t>(ndofs, es);
    auto f_neumann    = sfem::create_buffer<real_t>(ndofs, es);
    auto inertia_term = sfem::create_buffer<real_t>(ndofs, es);

    printf("===== Mooney-Rivlin Viscoelastic Test =====\n");
    printf("Mesh: %ld nodes, %ld DOFs\n", (long)n_nodes, (long)ndofs);
    printf("Newmark parameters: beta=%.2f, gamma=%.2f, c0=%.2e, dt=%.3f, density=%.1f\n", beta_nm, gamma_nm, c0, dt, density);
    printf("Time: T=%.2f, dt=%.3f\n", SFEM_T, dt);

    // Output
    if (SFEM_ENABLE_OUTPUT) {
        output->write_time_step("disp", t, x->data());
        output->write_time_step("velocity", t, v->data());
        output->write_time_step("acceleration", t, a->data());
        output->log_time(t);
    }

    // Time counting
    double total_hessian_time = 0;

    while (t < SFEM_T) {
        printf("Step %zu: t=%.3f\n", steps, t);

        // Newmark Prediction Step
        // u_pred = u_n + dt*v_n + (0.5-beta)*dt^2*a_n
        // v_pred = v_n + (1-gamma)*dt*a_n
        blas->copy(ndofs, x->data(), u_pred->data());
        blas->axpy(ndofs, dt, v->data(), u_pred->data());
        blas->axpy(ndofs, (0.5 - beta_nm) * dt * dt, a->data(), u_pred->data());

        blas->copy(ndofs, v->data(), v_pred->data());
        blas->axpy(ndofs, (1.0 - gamma_nm) * dt, a->data(), v_pred->data());

        if (SFEM_ENABLE_CONTACT) {
            blas->axpy(ndofs, 1, x->data(), lower_bound->data());
            blas->axpy(ndofs, -1, u_pred->data(), lower_bound->data());
        }

        // Use prediction as initial guess for Newton
        blas->copy(ndofs, u_pred->data(), x->data());

        // Newton Loop with Inertia
        for (int iter = 0; iter < 20; ++iter) {
            // Residual: R = M*a + F_int(u) - F_ext
            // where a = c0*(u - u_pred) from Newmark relation
            blas->zeros(ndofs, rhs->data());

            // Inertia term: c0 * M * (u - u_pred)
            blas->zeros(ndofs, inertia_term->data());
            for (ptrdiff_t i = 0; i < ndofs; ++i) {
                inertia_term->data()[i] = c0 * mass_diag->data()[i] * (x->data()[i] - u_pred->data()[i]);
            }
            blas->axpy(ndofs, 1.0, inertia_term->data(), rhs->data());

            // F_int(x)
            blas->zeros(ndofs, f_int->data());
            op->gradient(x->data(), f_int->data());
            real_t f_int_norm = blas->norm2(ndofs, f_int->data());
            blas->axpy(ndofs, 1.0, f_int->data(), rhs->data());

            // F_ext (Neumann) - note: gradient gives -F_ext
            blas->zeros(ndofs, f_neumann->data());
            neumann_op->gradient(x->data(), f_neumann->data());
            real_t f_ext_norm = blas->norm2(ndofs, f_neumann->data());
            blas->axpy(ndofs, 1.0, f_neumann->data(), rhs->data());

            if (iter == 0) {
                real_t x_norm       = blas->norm2(ndofs, x->data());
                real_t inertia_norm = blas->norm2(ndofs, inertia_term->data());
                printf("    |u|=%e, |M*a|=%e, |F_int|=%e, |F_ext|=%e\n", x_norm, inertia_norm, f_int_norm, f_ext_norm);
            }

            // Apply BC to residual
            f->set_value_to_constrained_dofs(0.0, rhs->data());
            real_t r_norm = blas->norm2(ndofs, rhs->data());

            printf("  Iter %d: |R| = %e\n", iter, r_norm);
            if (r_norm < 1e-8) break;

            // ===== Tangent Stiffness: K_eff = K_tan + c0*M =====
            blas->zeros(values->size(), values->data());
            auto t_start = std::chrono::high_resolution_clock::now();
            op->hessian_bsr(x->data(), graph->rowptr()->data(), graph->colidx()->data(), values->data());
            auto t_end = std::chrono::high_resolution_clock::now();
            total_hessian_time += std::chrono::duration<double>(t_end - t_start).count();

            auto      rowptr = graph->rowptr()->data();
            auto      colidx = graph->colidx()->data();
            auto      vals   = values->data();
            const int bs2    = block_size * block_size;

            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                for (count_t k = rowptr[node]; k < rowptr[node + 1]; ++k) {
                    if (colidx[k] == (idx_t)node) {
                        for (int d = 0; d < block_size; ++d) {
                            ptrdiff_t dof_idx = node * block_size + d;
                            vals[k * bs2 + d * block_size + d] += c0 * mass_diag->data()[dof_idx];
                        }
                        break;
                    }
                }
            }

            // Apply Dirichlet BC
            // conds->hessian_bsr(x->data(), rowptr, colidx, vals);

            // Extract diagonal for Jacobi preconditioner
            blas->zeros(ndofs, diag->data());
            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                for (count_t k = rowptr[node]; k < rowptr[node + 1]; ++k) {
                    if (colidx[k] == (idx_t)node) {
                        for (int d = 0; d < block_size; ++d) {
                            ptrdiff_t dof_idx     = node * block_size + d;
                            diag->data()[dof_idx] = vals[k * bs2 + d * block_size + d];
                        }
                        break;
                    }
                }
            }
            jacobi->set_diag(diag);

            blas->scal(ndofs, -1.0, rhs->data());
            blas->zeros(ndofs, delta_x->data());
            solver->apply(rhs->data(), delta_x->data());

            // u = u + dx
            real_t dx_norm = blas->norm2(ndofs, delta_x->data());
            if (iter == 0) {
                printf("    |dx|=%e\n", dx_norm);
            }
            blas->axpy(ndofs, 1.0, delta_x->data(), x->data());

            if (SFEM_ENABLE_CONTACT) {
                blas->axpy(ndofs, -1, delta_x->data(), lower_bound->data());
            }
        }

        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            a->data()[i] = c0 * (x->data()[i] - u_pred->data()[i]);
            v->data()[i] = v_pred->data()[i] + gamma_nm * dt * a->data()[i];
        }

        // Update history
        op->update_history(x->data());

        t += dt;
        steps++;

        // // Print velocity magnitude for dynamics check
        // real_t v_norm = blas->norm2(ndofs, v->data());
        // real_t a_norm = blas->norm2(ndofs, a->data());
        // printf("  After step: |v|=%e, |a|=%e\n\n", v_norm, a_norm);

        // Output
        if (SFEM_ENABLE_OUTPUT && steps % export_freq == 0) {
            output->write_time_step("disp", t, x->data());
            output->write_time_step("velocity", t, v->data());
            output->write_time_step("acceleration", t, a->data());
            output->log_time(t);
        }
    }

    printf("===== Test Completed =====\n");
    printf("Total Hessian assembly time: %.3f s\n", total_hessian_time);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_mooney_rivlin_visco_relaxation);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
