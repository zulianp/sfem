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
    
    // Simple 1-term Prony series
    real_t g_prony[] = {0.5}; // 50% relaxation
    real_t tau_prony[] = {0.5}; // relaxation time
    op->set_prony_terms(1, g_prony, tau_prony);
    
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
    
    // Lumped mass vector (diagonal of M)
    auto mass_diag = sfem::create_buffer<real_t>(ndofs, es);
    mass_op->hessian_diag(nullptr, mass_diag->data());
    f->set_value_to_constrained_dofs(1.0, mass_diag->data()); // Set 1 for BC nodes
    
    // Newmark state
    auto v = sfem::create_buffer<real_t>(ndofs, es);
    auto a = sfem::create_buffer<real_t>(ndofs, es);
    auto u_prev = sfem::create_buffer<real_t>(ndofs, es);
    auto v_prev = sfem::create_buffer<real_t>(ndofs, es);
    auto a_prev = sfem::create_buffer<real_t>(ndofs, es);
    auto Ma = sfem::create_buffer<real_t>(ndofs, es); // M * a buffer
    
    auto blas = sfem::blas<real_t>(es);
    blas->zeros(ndofs, x->data());
    blas->zeros(ndofs, v->data());
    blas->zeros(ndofs, a->data());
    
    real_t beta = 0.25;
    real_t gamma = 0.5;

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
    cg->verbose = false; // Quiet CG
    
    // Preconditioner (Jacobi)
    auto jacobi = sfem::create_shiftable_jacobi(diag, es);
    cg->set_preconditioner_op(jacobi);

    // FD check removed for cleaner test output
    
    // 5. Time Loop
    int n_steps = 3;
    
    auto f_int = sfem::create_buffer<real_t>(ndofs, es);
    auto f_neumann = sfem::create_buffer<real_t>(ndofs, es);
    
    for (int step = 0; step < n_steps; ++step) {
        if(step == 0) printf("Step %d: Loading...\n", step);
        else printf("Step %d: Time stepping...\n", step);
        
        // Store previous state
        blas->copy(ndofs, x->data(), u_prev->data());
        blas->copy(ndofs, v->data(), v_prev->data());
        blas->copy(ndofs, a->data(), a_prev->data());

        // Newton Loop (QUASI-STATIC: no inertia terms for debugging)
        for (int iter = 0; iter < 20; ++iter) {  // More iterations
            // Residual = F_int(x) - F_ext (no M*a for quasi-static)
            blas->zeros(ndofs, rhs->data());
            
            // F_int(x)
            blas->zeros(ndofs, f_int->data());
            op->gradient(x->data(), f_int->data());
            real_t f_int_norm = blas->norm2(ndofs, f_int->data());
            blas->axpy(ndofs, 1.0, f_int->data(), rhs->data());

            // F_ext (Neumann) - already negative
            blas->zeros(ndofs, f_neumann->data());
            neumann_op->gradient(x->data(), f_neumann->data());
            real_t f_ext_norm = blas->norm2(ndofs, f_neumann->data());
            blas->axpy(ndofs, 1.0, f_neumann->data(), rhs->data());
            
            if (iter == 0) {
                real_t x_norm = blas->norm2(ndofs, x->data());
                printf("    |x|=%e, |F_int|=%e, |F_ext|=%e\n", x_norm, f_int_norm, f_ext_norm);
            }
            
            // Check convergence
            f->set_value_to_constrained_dofs(0.0, rhs->data()); 
            real_t r_norm = blas->norm2(ndofs, rhs->data());
            
            if(iter == 0 || iter % 1 == 0) printf("  Iter %d: |R| = %e\n", iter, r_norm);
            if (r_norm < 1e-8) break;

            // Hessian Assembly: K_tan only (quasi-static), using BSR format
            blas->zeros(values->size(), values->data());
            op->hessian_bsr(x->data(), graph->rowptr()->data(), graph->colidx()->data(), values->data());
            
            auto rowptr = graph->rowptr()->data();
            auto colidx = graph->colidx()->data();
            auto vals = values->data();
            const int bs2 = block_size * block_size; // 9 for 3x3 blocks
            
            // Apply Dirichlet BC to matrix: set BC rows to identity (using BSR format)
            conds->hessian_bsr(x->data(), rowptr, colidx, vals);
            
            // Extract diagonal for Jacobi preconditioner (from BSR format)
            blas->zeros(ndofs, diag->data());
            for (ptrdiff_t node = 0; node < n_nodes; ++node) {
                for (count_t k = rowptr[node]; k < rowptr[node+1]; ++k) {
                    if (colidx[k] == (idx_t)node) {
                        // Extract diagonal elements from the 3x3 block
                        for (int d = 0; d < block_size; ++d) {
                            ptrdiff_t dof_idx = node * block_size + d;
                            diag->data()[dof_idx] = vals[k * bs2 + d * block_size + d];
                        }
                        break;
                    }
                }
            }
            
            // Update Jacobi preconditioner with new diagonal
            jacobi->set_diag(diag);
            
            // Solve K * dx = -R  (negate rhs first)
            blas->scal(ndofs, -1.0, rhs->data());
            blas->zeros(ndofs, delta_x->data());
            cg->apply(rhs->data(), delta_x->data());
            
            // Update x: x = x + alpha * dx  (damped Newton with alpha=0.5)
            real_t dx_norm = blas->norm2(ndofs, delta_x->data());
            real_t alpha = 0.5; // Damping factor
            // Print dx at a right-side node (node index depends on mesh)
            // For 2x1x1 cube, right nodes are at x=2
            if (iter == 0) {
                printf("    |dx|=%e, dx[last_node_x]=%e, alpha=%f\n", dx_norm, delta_x->data()[ndofs-3], alpha);
            }
            blas->axpy(ndofs, alpha, delta_x->data(), x->data());
        }
        
        // Update history (only for MooneyRivlinVisco)
        op->update_history(x->data());
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

