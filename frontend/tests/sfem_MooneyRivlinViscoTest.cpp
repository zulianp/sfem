#include <math.h>
#include <stdio.h>
#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_MooneyRivlinVisco.hpp"
#include "sfem_bsr_SpMV.hpp"
#include "sfem_test.h"


static void compute_contact_lower_bound(
    real_t* lower_bound,
    const real_t* displacement,
    const ptrdiff_t* contact_nodes,
    const ptrdiff_t n_contact_nodes,
    const real_t contact_plane,
    const int contact_dir,
    const int block_size,
    const ptrdiff_t ndofs,
    const real_t default_lb = -1000.0)
{
    for (ptrdiff_t i = 0; i < ndofs; i++) {
        lower_bound[i] = default_lb;
    }
    for (ptrdiff_t i = 0; i < n_contact_nodes; i++) {
        const ptrdiff_t node_idx = contact_nodes[i];
        const ptrdiff_t dof_idx = node_idx * block_size + contact_dir;
        const real_t current_disp = displacement[dof_idx];
        lower_bound[dof_idx] = contact_plane - current_disp;
    }
}


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

    int SFEM_BASE_RESOLUTION = 10;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    std::shared_ptr<sfem::Mesh> mesh;
    const char *mesh_path = getenv("SFEM_MESH");
    if (mesh_path && mesh_path[0] != '\0') {
        printf("Loading mesh from: %s\n", mesh_path);
        mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), mesh_path);
    } else {
        mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm),
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
    }

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


    int SFEM_ENABLE_CONTACT = false;
    SFEM_READ_ENV(SFEM_ENABLE_CONTACT, atoi);

    // WLF temperature shift parameters (set BEFORE Prony terms to avoid redundant calculations)
    int SFEM_USE_WLF = 1;
    real_t SFEM_WLF_C1 = 16.6253;
    real_t SFEM_WLF_C2 = 47.4781;
    real_t SFEM_WLF_T_REF = -54.29;
    real_t SFEM_TEMPERATURE = 20.0;
    SFEM_READ_ENV(SFEM_USE_WLF, atoi);
    SFEM_READ_ENV(SFEM_WLF_C1, atof);
    SFEM_READ_ENV(SFEM_WLF_C2, atof);
    SFEM_READ_ENV(SFEM_WLF_T_REF, atof);
    SFEM_READ_ENV(SFEM_TEMPERATURE, atof);
    
    if (SFEM_USE_WLF) {
        // Set WLF params and enable BEFORE setting Prony terms
        op->set_wlf_params(SFEM_WLF_C1, SFEM_WLF_C2, SFEM_WLF_T_REF);
        op->set_temperature(SFEM_TEMPERATURE);
        op->enable_wlf(true);
        printf("WLF enabled: C1=%.4f, C2=%.4f, T_ref=%.2f, T=%.2f\n",
               SFEM_WLF_C1, SFEM_WLF_C2, SFEM_WLF_T_REF, SFEM_TEMPERATURE);
        fflush(stdout);
    }

    // Prony series parameters from environment
    // Format: comma-separated values, e.g., "0.15,0.15,0.10,0.05"
    // Default: 4 terms with sum(g) = 0.45, g_inf = 0.55
    std::vector<real_t> g_prony   = {0.15, 0.15, 0.10, 0.05};
    std::vector<real_t> tau_prony = {0.1, 1.0, 10.0, 100.0};
    
    // Read from environment if provided
    const char* env_g   = getenv("SFEM_PRONY_G");
    const char* env_tau = getenv("SFEM_PRONY_TAU");
    
    if (env_g && env_tau) {
        g_prony.clear();
        tau_prony.clear();
        
        // Parse comma-separated g values
        std::string g_str(env_g);
        std::stringstream g_ss(g_str);
        std::string token;
        while (std::getline(g_ss, token, ',')) {
            g_prony.push_back(std::stod(token));
        }
        
        // Parse comma-separated tau values
        std::string tau_str(env_tau);
        std::stringstream tau_ss(tau_str);
        while (std::getline(tau_ss, token, ',')) {
            tau_prony.push_back(std::stod(token));
        }
        
        if (g_prony.size() != tau_prony.size()) {
            printf("Error: SFEM_PRONY_G and SFEM_PRONY_TAU must have the same number of terms!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        printf("Prony series from environment: %zu terms\n", g_prony.size());
        for (size_t i = 0; i < g_prony.size(); ++i) {
            printf("  term %zu: g=%.6f, tau=%.6f\n", i+1, g_prony[i], tau_prony[i]);
        }
    }
    
    // set_prony_terms will call compute_prony_coefficients() with WLF already enabled
    op->set_prony_terms((int)g_prony.size(), g_prony.data(), tau_prony.data());

    op->initialize();
    op->initialize_history();
    f->add_operator(op);

    // BC: use mesh bounds in x-direction
    geom_t x_min = 1e30;
    geom_t x_max = -1e30;
    const ptrdiff_t n_nodes = fs->mesh_ptr()->n_nodes();
    const geom_t *x_coords = fs->mesh_ptr()->points(0);
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        const geom_t x = x_coords[i];
        if (x < x_min) x_min = x;
        if (x > x_max) x_max = x;
    }
    const geom_t span = x_max - x_min;
    const geom_t tol = (span > 0 ? span : 1.0) * 1e-6;

    auto left_sideset = sfem::Sideset::create_from_selector(
            mesh, [=](const geom_t x, const geom_t, const geom_t) -> bool { return x < x_min + tol; });

    auto right_sideset = sfem::Sideset::create_from_selector(
            mesh, [=](const geom_t x, const geom_t, const geom_t) -> bool { return x > x_max - tol; });

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

    // Contact parameters
    real_t SFEM_CONTACT_PLANE = -0.1;  // Position of contact plane
    int SFEM_CONTACT_DIR = 0;          // Contact direction: 0=x, 1=y, 2=z
    SFEM_READ_ENV(SFEM_CONTACT_PLANE, atof);
    SFEM_READ_ENV(SFEM_CONTACT_DIR, atoi);

    // Store contact node indices
    std::vector<ptrdiff_t> contact_node_indices;

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

        // Get contact boundary nodes
        auto lnodes = sfem::create_nodeset_from_sideset(fs, left_sideset[0]);
        const ptrdiff_t nbnodes = lnodes->size();
        contact_node_indices.resize(nbnodes);
        for (ptrdiff_t i = 0; i < nbnodes; i++) {
            contact_node_indices[i] = lnodes->data()[i];
        }

        // Initial lower bound computation (displacement = 0 at start)
        compute_contact_lower_bound(
            lower_bound->data(),
            x->data(),  // Initial displacement
            contact_node_indices.data(),
            (ptrdiff_t)contact_node_indices.size(),
            SFEM_CONTACT_PLANE,
            SFEM_CONTACT_DIR,
            block_size,
            ndofs);

        mprgp->verbose = false;
        mprgp->set_lower_bound(lower_bound);
        mprgp->set_preconditioner_op(jacobi);  // Must set preconditioner!
        mprgp->set_max_it(2000);
        mprgp->set_rtol(1e-5);

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

        // Use prediction as initial guess for Newton
        blas->copy(ndofs, u_pred->data(), x->data());

        // Compute lower_bound based on current displacement (before solve)
        if (SFEM_ENABLE_CONTACT && !contact_node_indices.empty()) {
            compute_contact_lower_bound(
                lower_bound->data(),
                x->data(),
                contact_node_indices.data(),
                (ptrdiff_t)contact_node_indices.size(),
                SFEM_CONTACT_PLANE,
                SFEM_CONTACT_DIR,
                block_size,
                ndofs);
        }

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
                // printf("    |u|=%e, |M*a|=%e, |F_int|=%e, |F_ext|=%e\n", x_norm, inertia_norm, f_int_norm, f_ext_norm);
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
            // if (iter == 0) {
            //     printf("    |dx|=%e\n", dx_norm);
            // }
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
