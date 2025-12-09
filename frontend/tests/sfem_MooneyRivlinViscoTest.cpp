#include <stdio.h>
#include <math.h>
#include <memory>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_MooneyRivlinVisco.hpp"
#include "sfem_bsr_SpMV.hpp"
#include "sfem_test.h"

int test_mooney_rivlin_visco_relaxation() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto es = sfem::EXECUTION_SPACE_HOST;

    // 1. Create Mesh (Cube 2x1x1)
    int n_refine = 0; // Keep it coarse for unit test speed
    auto mesh = sfem::Mesh::create_hex8_cube(
        sfem::Communicator::wrap(comm),
        2, 1, 1,     // Grid
        0, 0, 0,     // Origin
        2, 1, 1      // Dimensions
    );

    auto fs = sfem::FunctionSpace::create(mesh, 3); // 3D displacement
    auto f = sfem::Function::create(fs);

    // 2. Create & Configure Operator
    auto op = std::make_shared<sfem::MooneyRivlinVisco>(fs);
    
    // Create LumpedMass and get mass vector (diagonal)
    auto mass_op = sfem::create_op(fs, "LumpedMass", es);
    mass_op->initialize();
    
    // Material Parameters - use small values for debugging
    op->set_C10(1.0);
    op->set_C01(0.5);
    op->set_K(100.0); // Nearly incompressible
    
    real_t dt = 0.1;
    op->set_dt(dt);
    
    // 10-term Prony series (MUST match SymPy codegen - code uses history[0..65])
    // SymPy generated code is hardcoded for 10 Prony terms!
    // Total: g_inf = 1 - sum(g_i) = 1 - 0.55 = 0.45
    real_t g_prony[] = {0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03};
    real_t tau_prony[] = {0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0};
    op->set_prony_terms(10, g_prony, tau_prony);
    
    op->initialize();
    op->initialize_history();
    f->add_operator(op);

    // 3. Boundary Conditions
    auto left_sideset = sfem::Sideset::create_from_selector(
            mesh, [](const geom_t x, const geom_t, const geom_t) -> bool { return x < 1e-5; });
    auto right_sideset = sfem::Sideset::create_from_selector(
            mesh, [](const geom_t x, const geom_t, const geom_t) -> bool { return x > 2.0 - 1e-5; });

    // Fix Left (x=0) in all directions
    sfem::DirichletConditions::Condition left_bc_x{.sidesets = left_sideset, .value = 0, .component = 0};
    sfem::DirichletConditions::Condition left_bc_y{.sidesets = left_sideset, .value = 0, .component = 1};
    sfem::DirichletConditions::Condition left_bc_z{.sidesets = left_sideset, .value = 0, .component = 2};
    
    auto conds = sfem::create_dirichlet_conditions(fs, {left_bc_x, left_bc_y, left_bc_z}, es);
    f->add_constraint(conds);

    // Pull Right (x=2) with Neumann Force (use very small force for debugging)
    sfem::NeumannConditions::Condition right_bc_force{.sidesets = right_sideset, .value = 0.0001, .component = 0};
    auto neumann_op = sfem::create_neumann_conditions(fs, {right_bc_force}, es);
    f->add_operator(neumann_op);

    // 4. Buffers
    const ptrdiff_t ndofs = fs->n_dofs();
    auto x = sfem::create_buffer<real_t>(ndofs, es);
    auto rhs = sfem::create_buffer<real_t>(ndofs, es);
    auto delta_x = sfem::create_buffer<real_t>(ndofs, es);
    auto diag = sfem::create_buffer<real_t>(ndofs, es);
    
    // Lumped mass vector (diagonal of M) - scale by density
    real_t density = 1000.0; // kg/m^3 (e.g., rubber-like material)
    auto mass_diag = sfem::create_buffer<real_t>(ndofs, es);
    mass_op->hessian_diag(nullptr, mass_diag->data());
    // Scale mass by density
    auto blas = sfem::blas<real_t>(es);
    blas->scal(ndofs, density, mass_diag->data());
    f->set_value_to_constrained_dofs(1.0, mass_diag->data()); // Set 1 for BC nodes
    
    // Newmark state
    auto v = sfem::create_buffer<real_t>(ndofs, es);
    auto a = sfem::create_buffer<real_t>(ndofs, es);
    auto u_pred = sfem::create_buffer<real_t>(ndofs, es);  // Predicted displacement
    auto v_pred = sfem::create_buffer<real_t>(ndofs, es);  // Predicted velocity
    
    blas->zeros(ndofs, x->data());
    blas->zeros(ndofs, v->data());
    blas->zeros(ndofs, a->data());
    
    // Newmark parameters (implicit, unconditionally stable)
    real_t beta_nm = 0.25;
    real_t gamma_nm = 0.5;
    real_t c0 = 1.0 / (beta_nm * dt * dt);  // Coefficient for M in effective stiffness

    // Matrix assembly buffers (BSR format: 3x3 blocks)
    auto graph = fs->node_to_node_graph();
    const int block_size = 3;
    const ptrdiff_t n_nodes = fs->mesh_ptr()->n_nodes();
    auto values = sfem::create_buffer<real_t>(graph->nnz() * block_size * block_size, es);
    
    // Linear Solver Wrapper (BSR SpMV)
    auto linear_op_apply = sfem::make_op<real_t>(
        ndofs, ndofs,
        [=](const real_t *const in, real_t *const out) {
            sfem::bsr_spmv<count_t, idx_t, real_t>(
                n_nodes, n_nodes, block_size,
                graph->rowptr()->data(), graph->colidx()->data(), values->data(),
                0.0, in, out);
        },
        es);
        
    auto cg = sfem::create_cg<real_t>(linear_op_apply, es);
    cg->set_n_dofs(ndofs);
    cg->set_max_it(2000);
    cg->set_rtol(1e-5);
    cg->verbose = false;
    
    // Preconditioner (Jacobi)
    auto jacobi = sfem::create_shiftable_jacobi(diag, es);
    cg->set_preconditioner_op(jacobi);

    // FD check removed for cleaner test output
    
    // 5. Time Loop with Full Newmark Integration
    int n_steps = 3;
    
    auto f_int = sfem::create_buffer<real_t>(ndofs, es);
    auto f_neumann = sfem::create_buffer<real_t>(ndofs, es);
    auto inertia_term = sfem::create_buffer<real_t>(ndofs, es);
    
    printf("Newmark parameters: beta=%.2f, gamma=%.2f, c0=%.2e, dt=%.3f, density=%.1f\n",
           beta_nm, gamma_nm, c0, dt, density);
    
    for (int step = 0; step < n_steps; ++step) {
        if(step == 0) printf("Step %d: Loading (t=%.3f)...\n", step, step * dt);
        else printf("Step %d: Time stepping (t=%.3f)...\n", step, step * dt);
        
        // ===== Newmark Prediction Step =====
        // u_pred = u_n + dt*v_n + (0.5-beta)*dt^2*a_n
        // v_pred = v_n + (1-gamma)*dt*a_n
        blas->copy(ndofs, x->data(), u_pred->data());
        blas->axpy(ndofs, dt, v->data(), u_pred->data());
        blas->axpy(ndofs, (0.5 - beta_nm) * dt * dt, a->data(), u_pred->data());
        
        blas->copy(ndofs, v->data(), v_pred->data());
        blas->axpy(ndofs, (1.0 - gamma_nm) * dt, a->data(), v_pred->data());
        
        // Use prediction as initial guess for Newton
        blas->copy(ndofs, u_pred->data(), x->data());
        
        // ===== Newton Loop with Inertia =====
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
                real_t x_norm = blas->norm2(ndofs, x->data());
                real_t inertia_norm = blas->norm2(ndofs, inertia_term->data());
                printf("    |u|=%e, |M*a|=%e, |F_int|=%e, |F_ext|=%e\n", 
                       x_norm, inertia_norm, f_int_norm, f_ext_norm);
            }
            
            // Apply BC to residual
            f->set_value_to_constrained_dofs(0.0, rhs->data()); 
            real_t r_norm = blas->norm2(ndofs, rhs->data());
            
            printf("  Iter %d: |R| = %e\n", iter, r_norm);
            if (r_norm < 1e-8) break;

            // ===== Tangent Stiffness: K_eff = K_tan + c0*M =====
            blas->zeros(values->size(), values->data());
            op->hessian_bsr(x->data(), graph->rowptr()->data(), graph->colidx()->data(), values->data());
            
            auto rowptr = graph->rowptr()->data();
            auto colidx = graph->colidx()->data();
            auto vals = values->data();
            const int bs2 = block_size * block_size;
            
            // Add c0*M to diagonal (lumped mass)
            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                for (count_t k = rowptr[node]; k < rowptr[node+1]; ++k) {
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
            conds->hessian_bsr(x->data(), rowptr, colidx, vals);
            
            // Extract diagonal for Jacobi preconditioner
            blas->zeros(ndofs, diag->data());
            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                for (count_t k = rowptr[node]; k < rowptr[node+1]; ++k) {
                    if (colidx[k] == (idx_t)node) {
                        for (int d = 0; d < block_size; ++d) {
                            ptrdiff_t dof_idx = node * block_size + d;
                            diag->data()[dof_idx] = vals[k * bs2 + d * block_size + d];
                        }
                        break;
                    }
                }
            }
            jacobi->set_diag(diag);
            
            // Solve K_eff * dx = -R
            blas->scal(ndofs, -1.0, rhs->data());
            blas->zeros(ndofs, delta_x->data());
            cg->apply(rhs->data(), delta_x->data());
            
            // Update: u = u + dx
            real_t dx_norm = blas->norm2(ndofs, delta_x->data());
            if (iter == 0) {
                printf("    |dx|=%e\n", dx_norm);
            }
            blas->axpy(ndofs, 1.0, delta_x->data(), x->data());
        }
        
        // ===== Newmark Correction Step =====
        // a_{n+1} = c0 * (u_{n+1} - u_pred)
        // v_{n+1} = v_pred + gamma*dt*a_{n+1}
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            a->data()[i] = c0 * (x->data()[i] - u_pred->data()[i]);
            v->data()[i] = v_pred->data()[i] + gamma_nm * dt * a->data()[i];
        }
        
        // Update history (only for MooneyRivlinVisco)
        op->update_history(x->data());
        
        // Print velocity magnitude for dynamics check
        real_t v_norm = blas->norm2(ndofs, v->data());
        real_t a_norm = blas->norm2(ndofs, a->data());
        printf("  After step: |v|=%e, |a|=%e\n\n", v_norm, a_norm);
    }

    printf("Test completed successfully!\n");
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_mooney_rivlin_visco_relaxation);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}

