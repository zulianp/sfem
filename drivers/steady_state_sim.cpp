#include "sfem_Function.hpp"

#include "sfem_Multigrid.hpp"
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

    std::shared_ptr<sfem::MatrixFreeLinearSolver<real_t>> solver;
    std::shared_ptr<sfem::Buffer<real_t>> b_x;
    std::shared_ptr<sfem::Buffer<real_t>> b_b;

#ifdef SFEM_ENABLE_CUDA
    if (SFEM_USE_GPU) {
        printf("Using GPU...\n");
        solver = sfem::d_cg<real_t>();

        // Register CUDA kernels
        sfem::register_device_ops();
        auto le = sfem::Factory::create_op_gpu(fs, SFEM_OPERATOR);

        le->initialize();

        // Transfer Boundary conditions to device
        auto d_conds = sfem::to_device(conds);

        f->add_constraint(d_conds);
        f->add_operator(le);

        // Create device buffers
        b_x = sfem::d_buffer<real_t>(fs->n_dofs());
        b_b = sfem::d_buffer<real_t>(fs->n_dofs());

        if (SFEM_USE_PRECONDITIONER) {
            auto b_d = sfem::d_buffer<real_t>(fs->n_dofs());
            if (f->hessian_diag(b_x->data(), b_d->data()) == 0) {
                // auto h_d = sfem::to_host(b_d);
                // h_d->print(std::cout);

                solver->set_preconditioner_op(sfem::make_op<real_t>(
                    b_x->size(), b_x->size(), [=](const real_t *const x, real_t *const y) {
                        d_ediv(b_d->size(), x, b_d->data(), y);
                    }));
            } else {
                fprintf(stderr, "[Warning] Preconditioner unavailable for mesh %s\n", folder);
            }
        }
    } else
#else
    SFEM_USE_GPU = false;
#endif
    {
        auto cg = sfem::h_cg<real_t>();
        cg->verbose = true;
        cg->set_max_it(10);
        solver = cg;

        auto op = sfem::Factory::create_op(fs, SFEM_OPERATOR);
        op->initialize();

        f->add_constraint(conds);
        f->add_operator(op);

        b_x = sfem::h_buffer<real_t>(fs->n_dofs());
        b_b = sfem::h_buffer<real_t>(fs->n_dofs());

        if (SFEM_USE_PRECONDITIONER) {
            if (true) {
                // if (false) {
                auto mg = sfem::h_mg<real_t>();

                auto lor_fs = fs->lor();
                auto lor_f = f->lor(lor_fs);

                auto coarse_fs = lor_fs->derefine();
                auto coarse_f = lor_f->derefine(coarse_fs, true);

                auto restriction = lor_f->hierarchical_restriction();
                auto prolongation = lor_f->hierarchical_prolongation();

                if (false) {
                    for (ptrdiff_t i = 0; i < b_x->size(); i++) {
                        b_x->data()[i] = 1;
                    }

                    restriction->apply(b_x->data(), b_b->data());

                    for (ptrdiff_t i = 0; i < coarse_fs->n_dofs(); i++) {
                        printf("%g ", b_b->data()[i]);
                        b_b->data()[i] = 0;
                    }
                    printf("\n");
                }

                // Fine level
                {
                    std::shared_ptr<sfem::Operator<real_t>> fine_op =
                        sfem::make_op<real_t>(lor_fs->n_dofs(),
                                              lor_fs->n_dofs(),
                                              [=](const real_t *const x, real_t *const y) {
                                                  lor_f->apply(nullptr, x, y);
                                                  // lor_f->apply_zero_constraints(y);
                                              });

                    std::shared_ptr<sfem::Operator<real_t>> fine_residual_op =
                        sfem::make_op<real_t>(lor_fs->n_dofs(),
                                              lor_fs->n_dofs(),
                                              [=](const real_t *const x, real_t *const y) {
                                                  lor_f->gradient(x, y);
                                                  // lor_f->apply_zero_constraints(y);
                                              });

                    auto fine_smoother = sfem::h_cg<real_t>();
                    fine_smoother->set_n_dofs(fs->n_dofs());
                    fine_smoother->set_op(fine_op);
                    fine_smoother->verbose = false;
                    fine_smoother->set_max_it(4);

                    mg->add_level(fine_residual_op, fine_smoother, nullptr, restriction);
                }

                // Coarse level
                {
                    std::shared_ptr<sfem::Operator<real_t>> coarse_op =
                        sfem::make_op<real_t>(coarse_fs->n_dofs(),
                                              coarse_fs->n_dofs(),
                                              [=](const real_t *const x, real_t *const y) {
                                                  coarse_f->apply(nullptr, x, y);
                                                  // coarse_f->apply_zero_constraints(y);
                                              });

                    std::shared_ptr<sfem::Operator<real_t>> coarse_residual_op =
                        sfem::make_op<real_t>(coarse_fs->n_dofs(),
                                              coarse_fs->n_dofs(),
                                              [=](const real_t *const x, real_t *const y) {
                                                  coarse_f->gradient(x, y);
                                                  // coarse_f->apply_zero_constraints(y);
                                              });

                    auto coarse_grid_solver = sfem::h_cg<real_t>();
                    coarse_grid_solver->set_n_dofs(coarse_fs->n_dofs());
                    coarse_grid_solver->set_op(coarse_op);
                    coarse_grid_solver->verbose = true;
                    mg->add_level(coarse_residual_op, coarse_grid_solver, prolongation, nullptr);
                }

                solver->set_preconditioner_op(mg);

            } else {
                auto b_d = sfem::h_buffer<real_t>(fs->n_dofs());

                if (f->hessian_diag(b_x->data(), b_d->data()) == 0) {
                    solver->set_preconditioner_op(sfem::make_op<real_t>(
                        b_x->size(), b_x->size(), [=](const real_t *const x, real_t *const y) {
                            auto d = b_d->data();

#pragma omp parallel for
                            for (ptrdiff_t i = 0; i < b_d->size(); ++i) {
                                y[i] = x[i] / d[i];
                            }
                        }));
                } else {
                    fprintf(stderr, "[Warning] Preconditioner unavailable for mesh %s\n", folder);
                }
            }
        }
    }

    // -------------------------------
    // Solver set-up
    // -------------------------------
    solver->set_op(sfem::make_op<real_t>(
        fs->n_dofs(), fs->n_dofs(), [=](const real_t *const v, real_t *const w) {
            f->apply(nullptr, v, w);
        }));

    // -------------------------------
    // Solve
    // -------------------------------
    double solve_tick = MPI_Wtime();

    f->apply_constraints(b_x->data());
    f->apply_constraints(b_b->data());

    // solver->set_max_it(40000);
    solver->set_n_dofs(fs->n_dofs());
    solver->apply(b_b->data(), b_x->data());

    double solve_tock = MPI_Wtime();

    // -------------------------------
    // Write output
    // -------------------------------

    f->set_output_dir(output_path);
    auto output = f->output();

#ifdef SFEM_ENABLE_CUDA
    auto h_x = sfem::to_host(b_x);
#else
    auto h_x = b_x;
#endif
    output->write("x", h_x->data());

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
