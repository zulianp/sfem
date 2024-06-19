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

std::shared_ptr<sfem::Operator<real_t>> crs_hessian(sfem::Function &f) {
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

    SFEM_READ_ENV(SFEM_MATRIX_FREE, atoi);
    SFEM_READ_ENV(SFEM_OPERATOR, );
    SFEM_READ_ENV(SFEM_BLOCK_SIZE, atoi);
    SFEM_READ_ENV(SFEM_USE_PRECONDITIONER, atoi);

    printf("SFEM_MATRIX_FREE: %d\n"
           "SFEM_OPERATOR: %s\n"
           "SFEM_BLOCK_SIZE: %d\n"
           "SFEM_USE_PRECONDITIONER: %d\n",
           SFEM_MATRIX_FREE,
           SFEM_OPERATOR,
           SFEM_BLOCK_SIZE,
           SFEM_USE_PRECONDITIONER);

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

    bool use_diag_preconditioner = false;
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

        auto cheb = sfem::h_cheb3<real_t>(linear_op);
        cheb->init(rhs->data());
        cheb->set_max_it(3);
        smoother = cheb;
    } else {
        linear_op = crs_hessian(*f);
        auto cheb = sfem::h_cheb3<real_t>(linear_op);
        cheb->init(rhs->data());
        smoother = cheb;
    }

#if 1  // MG
    auto c = sfem::h_buffer<real_t>(fs->n_dofs());
    auto r = sfem::h_buffer<real_t>(fs->n_dofs());

    //  Coarse level
    auto fs_coarse = fs->derefine();
    auto f_coarse = f->derefine(fs_coarse, true);

    // auto linear_op_coarse = sfem::make_op<real_t>(
    //         fs_coarse->n_dofs(), fs_coarse->n_dofs(), [=](const real_t *const x, real_t *const y)
    //         {
    //             f_coarse->apply(nullptr, x, y);
    //             f_coarse->apply_zero_constraints(y);
    //         });

    auto linear_op_coarse = crs_hessian(*f_coarse);

    auto c_coarse = sfem::h_buffer<real_t>(fs_coarse->n_dofs());
    auto r_coarse = sfem::h_buffer<real_t>(fs_coarse->n_dofs());
    auto diag_coarse = sfem::h_buffer<real_t>(fs_coarse->n_dofs());
    auto solver_coarse = sfem::h_cg<real_t>();

    {
        solver_coarse->set_n_dofs(fs_coarse->n_dofs());
        solver_coarse->set_op(linear_op_coarse);
        solver_coarse->verbose = true;
        solver_coarse->set_max_it(1000);
        solver_coarse->tol = 1e-12;

        if (use_diag_preconditioner) {
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

    real_t rtr = 0;
    for (int k = 0; k < 20; k++) {
        // Coarse grid
        smoother->apply(rhs->data(), x->data());

        const real_t xtx = dot(fs->n_dofs(), x->data(), x->data());

        {  // Residual
            zeros(fs->n_dofs(), r->data());
            linear_op->apply(x->data(), r->data());
            axpby<real_t>(fs->n_dofs(), 1, rhs->data(), -1, r->data());

            rtr = dot(fs->n_dofs(), r->data(), r->data());

            printf("MG: %d) residual norm: %g, x norm: %g\n", k, rtr, xtx);

            if (rtr < tol) {
                break;
            }
        }

        f->apply_zero_constraints(r->data());

        // Restriction
        zeros(fs_coarse->n_dofs(), r_coarse->data());
        restriction->apply(r->data(), r_coarse->data());

        // f_coarse->apply_zero_constraints(r_coarse->data());

        // Set guess to zero
        zeros(fs_coarse->n_dofs(), c_coarse->data());

        solver_coarse->apply(r_coarse->data(), c_coarse->data());

        // Prolongation
        prolongation->apply(c_coarse->data(), c->data());

        // Apply correction
        axpby<real_t>(fs->n_dofs(), 1, c->data(), 1, x->data());
        f->apply_constraints(x->data());

        smoother->apply(rhs->data(), x->data());
    }

#else  // Basic solvers
#if 0
    // Point Jacobi solver (relaxed)
    auto solver = smoother;
    solver->set_op(linear_op);
    solver->verbose = true;
#else  // CG solver

    //     auto preconditioner = sfem::make_op<real_t>(
    //             diag->size(), diag->size(), [=](const real_t *const x, real_t *const y) {
    //                 auto d = diag->data();

    // #pragma omp parallel for
    //                 for (ptrdiff_t i = 0; i < diag->size(); ++i) {
    //                     y[i] = x[i] / d[i];
    //                 }
    //             });

    // smoother->set_max_it(3);
    auto preconditioner = smoother;
    smoother->set_initial_guess_zero(true);

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
