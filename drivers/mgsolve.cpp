#include "sfem_Function.hpp"

#include "sfem_Multigrid.hpp"
#include "sfem_PointJacobi.hpp"

#include "sfem_base.h"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

#include <vector>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 3) {
        fprintf(stderr, "usage: %s <folder> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_path = argv[2];

    double tick = MPI_Wtime();

    bool SFEM_USE_GPU = true;
    SFEM_READ_ENV(SFEM_USE_GPU, atoi);

    // -------------------------------
    // Read inputs
    // -------------------------------

    const char *folder = argv[1];
    auto m = sfem::Mesh::create_from_file(comm, folder);

    const char *SFEM_OPERATOR = "Laplacian";
    int SFEM_BLOCK_SIZE = 1;
    int SFEM_USE_PRECONDITIONER = 0;

    SFEM_READ_ENV(SFEM_OPERATOR, );
    SFEM_READ_ENV(SFEM_BLOCK_SIZE, atoi);
    SFEM_READ_ENV(SFEM_USE_PRECONDITIONER, atoi);

    auto fs = sfem::FunctionSpace::create(m, SFEM_BLOCK_SIZE);
    auto conds = sfem::DirichletConditions::create_from_env(fs);
    auto f = sfem::Function::create(fs);

    auto diag = sfem::h_buffer<real_t>(fs->n_dofs());
    auto x = sfem::h_buffer<real_t>(fs->n_dofs());
    auto rhs = sfem::h_buffer<real_t>(fs->n_dofs());

    auto op = sfem::Factory::create_op(fs, SFEM_OPERATOR);
    op->initialize();
    f->add_constraint(conds);
    f->add_operator(op);

    real_t tol = 1e-8;

    double solve_tick = MPI_Wtime();

    // Fine level
    auto linear_op = sfem::make_op<real_t>(
            fs->n_dofs(), fs->n_dofs(), [=](const real_t *const x, real_t *const y) {
                f->apply(nullptr, x, y);
            });

    f->hessian_diag(nullptr, diag->data());

#if 0  // MG
    bool cascadic_mg = false;
    auto c = sfem::h_buffer<real_t>(fs->n_dofs());
    auto r = sfem::h_buffer<real_t>(fs->n_dofs());
    
    // auto smoother = sfem::h_pjacobi(fs->n_dofs(), diag->data(), 0.5);
    // smoother->set_max_it(10);
    // smoother->verbose = false;

    auto smoother = sfem::h_cg<real_t>();
    smoother->set_n_dofs(fs->n_dofs());
    smoother->set_max_it(cascadic_mg ? 1000 : 2);

    //  Coarse level
    auto fs_coarse = fs->derefine();
    auto f_coarse = f->derefine(fs_coarse, !cascadic_mg);
    auto linear_op_coarse = sfem::make_op<real_t>(
            fs_coarse->n_dofs(), fs_coarse->n_dofs(), [=](const real_t *const x, real_t *const y) {
                f_coarse->apply(nullptr, x, y);
            });

    auto c_coarse = sfem::h_buffer<real_t>(fs_coarse->n_dofs());
    auto r_coarse = sfem::h_buffer<real_t>(fs_coarse->n_dofs());
    auto diag_coarse = sfem::h_buffer<real_t>(fs_coarse->n_dofs());
    auto solver_coarse = sfem::h_cg<real_t>();

    {
        f_coarse->hessian_diag(nullptr, diag_coarse->data());
        auto preconditioner =
                sfem::make_op<real_t>(diag_coarse->size(),
                                      diag_coarse->size(),
                                      [=](const real_t *const x, real_t *const y) {
                                          auto d = diag_coarse->data();
#pragma omp parallel for
                                          for (ptrdiff_t i = 0; i < diag_coarse->size(); ++i) {
                                              y[i] = x[i] / d[i];
                                          }
                                      });

        solver_coarse->set_n_dofs(fs_coarse->n_dofs());
        solver_coarse->set_preconditioner_op(preconditioner);
        solver_coarse->set_op(linear_op_coarse);
        solver_coarse->verbose = true;
        solver_coarse->set_max_it(1000);
        // solver_coarse->tol = 1e-8
    }

    // Multigrid
    auto restriction = f->hierarchical_restriction();
    auto prolongation = f->hierarchical_prolongation();

    smoother->set_op(linear_op);

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    real_t rtr = 0;
    for (int k = 0; k < (cascadic_mg ? 1 : 80); k++) {
        // Coarse grid
        smoother->apply(rhs->data(), x->data());

        { //Residual
            smoother->zeros(fs->n_dofs(), r->data());
            linear_op->apply(x->data(), r->data());
            smoother->axpby(fs->n_dofs(), 1, rhs->data(), -1, r->data());
        }

        // Compute residual
        // rtr = smoother->dot(fs->n_dofs(), r->data(), r->data());
        // printf("%d residual norm (before): %g\n", k, rtr);

        if (!cascadic_mg) {
            f->apply_zero_constraints(r->data());
            
            // Restriction
            solver_coarse->zeros(fs_coarse->n_dofs(), r_coarse->data());
            restriction->apply(r->data(), r_coarse->data());

            // Set guess to zero
            solver_coarse->zeros(fs_coarse->n_dofs(), c_coarse->data());
        } else {
            f_coarse->apply_constraints(c_coarse->data());
            f_coarse->apply_constraints(r_coarse->data());
        }

        solver_coarse->apply(r_coarse->data(), c_coarse->data());

        // Prolongation
        prolongation->apply(c_coarse->data(), c->data());

        // Apply correction
        smoother->axpby(fs->n_dofs(), 1, c->data(), 1, x->data());
        f->apply_constraints(x->data());

        smoother->apply(rhs->data(), x->data());

        { //Residual
            smoother->zeros(fs->n_dofs(), r->data());
            linear_op->apply(x->data(), r->data());
            smoother->axpby(fs->n_dofs(), 1, rhs->data(), -1, r->data());
            rtr = smoother->dot(fs->n_dofs(), r->data(), r->data());
        }

        printf("%d) residual norm: %g\n",k, rtr);

        if(rtr < tol) {
            break;
        }        
    }

#else  // Basic solvers
    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());
#if 0
    // Point Jacobi solver (relaxed)
    auto solver = sfem::h_pjacobi(fs->n_dofs(), diag->data(), 0.6);
    solver->set_op(linear_op);
    solver->verbose = true;
#else
    // CG solver
    auto preconditioner = sfem::make_op<real_t>(
            diag->size(), diag->size(), [=](const real_t *const x, real_t *const y) {
                auto d = diag->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < diag->size(); ++i) {
                    y[i] = x[i] / d[i];
                }
            });

    auto solver = sfem::h_cg<real_t>();
    solver->set_n_dofs(fs->n_dofs());
    solver->set_preconditioner_op(preconditioner);
    solver->verbose = true;
    solver->tol = tol;

#endif

    solver->set_op(linear_op);
    solver->apply(rhs->data(), x->data());
#endif  // MG

    double solve_tock = MPI_Wtime();

    // -------------------------------
    // Write output
    // -------------------------------

    f->set_output_dir(output_path);
    auto output = f->output();

    output->write("x", x->data());
    output->write("rhs", rhs->data());


    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #dofs %ld\n",
               (long)m->n_elements(),
               (long)m->n_nodes(),
               (long)fs->n_dofs());
        printf("TTS:\t\t\t%g seconds (solve: %g)\n", tock - tick, solve_tock - solve_tick);
    }

    return MPI_Finalize();
}
