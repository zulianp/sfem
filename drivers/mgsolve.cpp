#include <memory>
#include "sfem_Function.hpp"

#include "sfem_Chebyshev3.hpp"
#include "sfem_GaussSeidel.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_base.h"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"
#include "sfem_crs_SpMV.hpp"
#include "spmv.h"

#include "matrixio_array.h"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

#include <vector>

template <typename T>
inline void zeros(std::size_t n, T *arr) {
    memset(arr, 0, n * sizeof(T));
}

template <typename T>
void axpby(const ptrdiff_t n, const T alpha, const T *const x, const T beta, T *const y) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; i++) {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

template <typename T>
T dot(const ptrdiff_t n, const T *const l, const T *const r) {
    T ret = 0;

#pragma omp parallel for reduction(+ : ret)
    for (ptrdiff_t i = 0; i < n; i++) {
        ret += l[i] * r[i];
    }

    return ret;
}

auto crs_hessian(sfem::Function &f) {
    ptrdiff_t nlocal;
    ptrdiff_t nglobal;
    ptrdiff_t nnz;
    isolver_idx_t *rowptr;
    isolver_idx_t *colidx;
    f.space()->create_crs_graph(&nlocal, &nglobal, &nnz, &rowptr, &colidx);
    real_t *values = (real_t *)calloc(rowptr[nlocal], sizeof(real_t));
    f.hessian_crs(nullptr, rowptr, colidx, values);

    // Owns the pointers
    return sfem::h_crs_spmv(nlocal, nlocal, rowptr, colidx, values, (real_t)1);
}

real_t residual(sfem::Operator<real_t> &op,
                const real_t *const rhs,
                const real_t *const x,
                real_t *const r) {
    zeros(op.rows(), r);
    op.apply(x, r);
    axpby<real_t>(op.rows(), 1, rhs, -1, r);
    return sqrt(dot(op.rows(), r, r));
}

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
    int SFEM_MATRIX_FREE = 0;
    int SFEM_USE_CHEB = 0;
    int SFEM_DEBUG = 0;
    int SFEM_MG = 0;
    float SFEM_CHEB_EIG_MAX_SCALE = 1;

    SFEM_READ_ENV(SFEM_MATRIX_FREE, atoi);
    SFEM_READ_ENV(SFEM_OPERATOR, );
    SFEM_READ_ENV(SFEM_BLOCK_SIZE, atoi);
    SFEM_READ_ENV(SFEM_USE_PRECONDITIONER, atoi);
    SFEM_READ_ENV(SFEM_USE_CHEB, atoi);
    SFEM_READ_ENV(SFEM_DEBUG, atoi);
    SFEM_READ_ENV(SFEM_MG, atoi);
    SFEM_READ_ENV(SFEM_CHEB_EIG_MAX_SCALE, atof);

    printf("SFEM_MATRIX_FREE: %d\n"
           "SFEM_OPERATOR: %s\n"
           "SFEM_BLOCK_SIZE: %d\n"
           "SFEM_USE_PRECONDITIONER: %d\n"
           "SFEM_USE_CHEB: %d\n"
           "SFEM_DEBUG: %d\n"
           "SFEM_MG: %d\n"
           "SFEM_CHEB_EIG_MAX_SCALE: %f\n",
           SFEM_MATRIX_FREE,
           SFEM_OPERATOR,
           SFEM_BLOCK_SIZE,
           SFEM_USE_PRECONDITIONER,
           SFEM_USE_CHEB,
           SFEM_DEBUG,
           SFEM_MG,
           SFEM_CHEB_EIG_MAX_SCALE);

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

    real_t tol = 1e-12;

    double solve_tick = MPI_Wtime();

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    std::shared_ptr<sfem::Operator<real_t>> linear_op;
    std::shared_ptr<sfem::MatrixFreeLinearSolver<real_t>> smoother;

    if (SFEM_MATRIX_FREE) {
        linear_op = sfem::make_op<real_t>(
                fs->n_dofs(), fs->n_dofs(), [=](const real_t *const x, real_t *const y) {
                    f->apply(nullptr, x, y);
                });

        if (SFEM_USE_CHEB) {
            auto cheb = sfem::h_cheb3<real_t>(linear_op);
            cheb->init(rhs->data());
            cheb->scale_eig_max = SFEM_CHEB_EIG_MAX_SCALE;
            cheb->set_max_it(3);
            smoother = cheb;
        } else {
            f->hessian_diag(nullptr, diag->data());
        }

    } else {
        auto crs = crs_hessian(*f);
        linear_op = crs;

        if (SFEM_USE_CHEB) {
            auto cheb = sfem::h_cheb3<real_t>(linear_op);
            cheb->init(rhs->data());
            cheb->scale_eig_max = SFEM_CHEB_EIG_MAX_SCALE;
            cheb->set_max_it(3);
            smoother = cheb;
        } else {
            f->hessian_diag(nullptr, diag->data());
            auto gs = sfem::h_gauss_seidel(crs, diag->data());
            gs->set_max_it(5);
            // gs->verbose = true;
            smoother = gs;

            if (SFEM_DEBUG) {
                array_write(comm,
                            "./rhs.raw",
                            SFEM_MPI_REAL_T,
                            rhs->data(),
                            fs->n_dofs(),
                            fs->n_dofs());
                array_write(comm,
                            "./diag.raw",
                            SFEM_MPI_REAL_T,
                            diag->data(),
                            fs->n_dofs(),
                            fs->n_dofs());
                array_write(comm,
                            "./rowptr.raw",
                            SFEM_MPI_COUNT_T,
                            crs->row_ptr->data(),
                            fs->n_dofs() + 1,
                            fs->n_dofs() + 1);
                array_write(comm,
                            "./colidx.raw",
                            SFEM_MPI_IDX_T,
                            crs->col_idx->data(),
                            crs->row_ptr->data()[fs->n_dofs()],
                            crs->row_ptr->data()[fs->n_dofs()]);
                array_write(comm,
                            "./values.raw",
                            SFEM_MPI_REAL_T,
                            crs->values->data(),
                            crs->row_ptr->data()[fs->n_dofs()],
                            crs->row_ptr->data()[fs->n_dofs()]);
            }
        }
    }

    f->set_output_dir(output_path);
    auto output = f->output();

    if (SFEM_MG) {
        auto c = sfem::h_buffer<real_t>(fs->n_dofs());
        auto r = sfem::h_buffer<real_t>(fs->n_dofs());

        //  Coarse level
        auto fs_coarse = fs->derefine();
        auto f_coarse = f->derefine(fs_coarse, true);

        std::shared_ptr<sfem::Operator<real_t>> linear_op_coarse;
        if (SFEM_MATRIX_FREE) {
            linear_op_coarse = sfem::make_op<real_t>(fs_coarse->n_dofs(),
                                                     fs_coarse->n_dofs(),
                                                     [=](const real_t *const x, real_t *const y) {
                                                         f_coarse->apply(nullptr, x, y);
                                                     });
        } else {
            linear_op_coarse = crs_hessian(*f_coarse);
        }

        auto c_coarse = sfem::h_buffer<real_t>(fs_coarse->n_dofs());
        auto r_coarse = sfem::h_buffer<real_t>(fs_coarse->n_dofs());
        auto diag_coarse = sfem::h_buffer<real_t>(fs_coarse->n_dofs());
        auto solver_coarse = sfem::h_cg<real_t>();

        {
            solver_coarse->set_n_dofs(fs_coarse->n_dofs());
            solver_coarse->set_op(linear_op_coarse);
            solver_coarse->verbose = false;
            solver_coarse->set_max_it(1000);
            solver_coarse->tol = 1e-12;

            if (SFEM_USE_PRECONDITIONER) {
                f_coarse->hessian_diag(nullptr, diag_coarse->data());
                auto preconditioner = sfem::make_op<real_t>(
                        diag_coarse->size(),
                        diag_coarse->size(),
                        [=](const real_t *const x, real_t *const y) {
                            auto d = diag_coarse->data();
#pragma omp parallel for
                            for (ptrdiff_t i = 0; i < diag_coarse->size(); ++i) {
                                y[i] = x[i] / d[i];
                            }
                        });

                solver_coarse->set_preconditioner_op(preconditioner);
                solver_coarse->set_initial_guess_zero(true);
            }
        }

        smoother->set_initial_guess_zero(false);

        // Multigrid
        auto restriction = f->hierarchical_restriction();
        auto prolongation = f->hierarchical_prolongation();

        f->apply_constraints(x->data());
        f->apply_constraints(rhs->data());

        real_t rtr = residual(*linear_op, rhs->data(), x->data(), r->data());
        for (int k = 0; k < 200; k++) {
            smoother->apply(rhs->data(), x->data());

            {  // Residual
                real_t rtr_new = residual(*linear_op, rhs->data(), x->data(), r->data());
                real_t rate = rtr_new / rtr;
                rtr = rtr_new;
                printf("MG: %d)\tresidual: %g,\trate: %g\n", k, rtr, rate);
                if (rtr < tol || rate > 0.999) {
                    break;
                }
            }

            // Restriction
            zeros(solver_coarse->rows(), r_coarse->data());
            restriction->apply(r->data(), r_coarse->data());

            // Set guess to zero
            zeros(solver_coarse->rows(), c_coarse->data());

            solver_coarse->apply(r_coarse->data(), c_coarse->data());

            // Prolongation
            zeros(smoother->rows(), c->data());
            prolongation->apply(c_coarse->data(), c->data());

            // Check do we need this?
            f->apply_zero_constraints(c->data());

            // Apply correction
            axpby<real_t>(smoother->rows(), 1, c->data(), 1, x->data());

            smoother->apply(rhs->data(), x->data());
        }

    } else {
#if 0
    auto solver = smoother;
    solver->set_max_it(100);
    solver->set_op(linear_op);

#else  // CG solver

        auto solver = sfem::h_cg<real_t>();
        solver->set_n_dofs(fs->n_dofs());
        solver->set_op(linear_op);

        if (smoother) {
            auto preconditioner = smoother;
            smoother->set_initial_guess_zero(true);
            solver->set_preconditioner_op(preconditioner);
        } else {
            auto preconditioner = sfem::make_op<real_t>(
                    diag->size(), diag->size(), [=](const real_t *const x, real_t *const y) {
                        auto d = diag->data();

#pragma omp parallel for
                        for (ptrdiff_t i = 0; i < diag->size(); ++i) {
                            y[i] = x[i] / d[i];
                        }
                    });

            solver->set_preconditioner_op(preconditioner);
        }

        solver->verbose = true;
        solver->tol = tol;
        solver->set_max_it(800);

#endif

        solver->set_op(linear_op);
        solver->apply(rhs->data(), x->data());
    }

    double solve_tock = MPI_Wtime();

    auto r = sfem::h_buffer<real_t>(fs->n_dofs());
    real_t rtr = residual(*linear_op, rhs->data(), x->data(), r->data());

    // -------------------------------
    // Write output
    // -------------------------------

    output->write("x", x->data());
    output->write("rhs", rhs->data());
    output->write("residual", r->data());

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #dofs %ld\n",
               (long)m->n_elements(),
               (long)m->n_nodes(),
               (long)fs->n_dofs());
        printf("TTS:\t\t\t%g seconds (solve: %g)\n", tock - tick, solve_tock - solve_tick);
        printf("residual: %g\n", rtr);
    }

    return MPI_Finalize();
}
