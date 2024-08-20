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

#include "sfem_API.hpp"

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

auto crs_hessian(sfem::Function &f, const sfem::ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
        if (es == sfem::EXECUTION_SPACE_DEVICE) {
            assert(false && "IMPLEMENT ME");
        }
#endif
    ptrdiff_t nlocal;
    ptrdiff_t nglobal;
    ptrdiff_t nnz;
    count_t *rowptr;
    idx_t *colidx;
    f.space()->create_crs_graph(&nlocal, &nglobal, &nnz, &rowptr, &colidx);
    real_t *values = (real_t *)calloc(rowptr[nlocal], sizeof(real_t));
    f.hessian_crs(nullptr, rowptr, colidx, values);

    // Owns the pointers
    return sfem::h_crs_spmv(nlocal, nlocal, rowptr, colidx, values, (real_t)1);
}

real_t residual(sfem::Operator<real_t> &op,
                const real_t *const rhs,
                const real_t *const x,
                real_t *const r,
                const sfem::ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
    if (es == sfem::EXECUTION_SPACE_DEVICE) {
        assert(false);
        // TODO
        return -1;
    } else
#endif
    {
        zeros(op.rows(), r);
        op.apply(x, r);
        axpby<real_t>(op.rows(), 1, rhs, -1, r);
        return sqrt(dot(op.rows(), r, r));
    }
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

    sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;

    // -------------------------------
    // Read inputs
    // -------------------------------

    const char *folder = argv[1];
    auto m = sfem::Mesh::create_from_file(comm, folder);

    const char *SFEM_OPERATOR = "Laplacian";
    int SFEM_BLOCK_SIZE = 1;
    int SFEM_USE_PRECONDITIONER = 0;
    int SFEM_MATRIX_FREE = 0;
    int SFEM_COARSE_MATRIX_FREE = 0;
    int SFEM_USE_CHEB = 0;
    int SFEM_DEBUG = 0;
    int SFEM_MG = 0;
    int SFEM_MAX_IT = 4000;
    int SFEM_SMOOTHER_SWEEPS = 3;
    float SFEM_CHEB_EIG_MAX_SCALE = 1.02;
    float SFEM_TOL = 1e-9;
    float SFEM_CHEB_EIG_TOL = 1e-5;

    SFEM_READ_ENV(SFEM_MATRIX_FREE, atoi);
    SFEM_READ_ENV(SFEM_COARSE_MATRIX_FREE, atoi);
    SFEM_READ_ENV(SFEM_OPERATOR, );
    SFEM_READ_ENV(SFEM_BLOCK_SIZE, atoi);
    SFEM_READ_ENV(SFEM_USE_PRECONDITIONER, atoi);
    SFEM_READ_ENV(SFEM_USE_CHEB, atoi);
    SFEM_READ_ENV(SFEM_DEBUG, atoi);
    SFEM_READ_ENV(SFEM_MG, atoi);
    SFEM_READ_ENV(SFEM_CHEB_EIG_MAX_SCALE, atof);
    SFEM_READ_ENV(SFEM_MAX_IT, atoi);
    SFEM_READ_ENV(SFEM_SMOOTHER_SWEEPS, atoi);
    SFEM_READ_ENV(SFEM_CHEB_EIG_TOL, atof);

    printf("SFEM_MATRIX_FREE: %d\n"
           "SFEM_COARSE_MATRIX_FREE: %d\n"
           "SFEM_OPERATOR: %s\n"
           "SFEM_BLOCK_SIZE: %d\n"
           "SFEM_USE_PRECONDITIONER: %d\n"
           "SFEM_USE_CHEB: %d\n"
           "SFEM_DEBUG: %d\n"
           "SFEM_MG: %d\n"
           "SFEM_CHEB_EIG_MAX_SCALE: %f\n"
           "SFEM_TOL: %f\n"
           "SFEM_SMOOTHER_SWEEPS: %d\n"
           "SFEM_CHEB_EIG_TOL: %f\n",
           SFEM_MATRIX_FREE,
           SFEM_COARSE_MATRIX_FREE,
           SFEM_OPERATOR,
           SFEM_BLOCK_SIZE,
           SFEM_USE_PRECONDITIONER,
           SFEM_USE_CHEB,
           SFEM_DEBUG,
           SFEM_MG,
           SFEM_CHEB_EIG_MAX_SCALE,
           SFEM_TOL,
           SFEM_SMOOTHER_SWEEPS,
           SFEM_CHEB_EIG_TOL);

    auto fs = sfem::FunctionSpace::create(m, SFEM_BLOCK_SIZE);
    auto conds = sfem::create_dirichlet_conditions_from_env(fs, es);
    auto f = sfem::Function::create(fs);

    auto diag = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto x = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto rhs = sfem::create_buffer<real_t>(fs->n_dofs(), es);

    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);
    op->initialize();
    f->add_constraint(conds);
    f->add_operator(op);

    double compute_tick = MPI_Wtime();
    double init_tick = MPI_Wtime();
    double init_tock;
    double solve_tick, solve_tock;

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    std::shared_ptr<sfem::Operator<real_t>> linear_op;
    std::shared_ptr<sfem::MatrixFreeLinearSolver<real_t>> smoother;

    if (SFEM_MATRIX_FREE) {
        linear_op = sfem::make_linear_op(f);

        if (SFEM_USE_CHEB) {
            auto cheb = sfem::create_cheb3<real_t>(linear_op, es);

            // Power-method
            auto r = sfem::create_buffer<real_t>(fs->n_dofs(), es);
            residual(*linear_op, rhs->data(), x->data(), r->data(), es);
            cheb->eigen_solver_tol = SFEM_CHEB_EIG_TOL;
            cheb->init(r->data());

            cheb->scale_eig_max = SFEM_CHEB_EIG_MAX_SCALE;
            cheb->set_max_it(SFEM_SMOOTHER_SWEEPS);
            cheb->set_initial_guess_zero(false);
            smoother = cheb;
        } else if (SFEM_USE_PRECONDITIONER) {
            f->hessian_diag(nullptr, diag->data());
        }

    } else {
        auto crs = crs_hessian(*f, es);
        linear_op = crs;
        f->hessian_diag(nullptr, diag->data());

        if (SFEM_USE_CHEB) {
            auto cheb = sfem::create_cheb3<real_t>(linear_op, es);

            // Power-method
            auto r = sfem::create_buffer<real_t>(fs->n_dofs(), es);
            residual(*linear_op, rhs->data(), x->data(), r->data(), es);
            cheb->init(r->data());

            cheb->scale_eig_max = SFEM_CHEB_EIG_MAX_SCALE;
            cheb->set_max_it(SFEM_SMOOTHER_SWEEPS);
            cheb->set_initial_guess_zero(false);
            smoother = cheb;
        } else {
            if ((!SFEM_USE_PRECONDITIONER || SFEM_MG) && fs->block_size() == 1) {
                auto gs = sfem::h_gauss_seidel(crs, diag->data());
                gs->set_max_it(SFEM_SMOOTHER_SWEEPS);
                // gs->verbose = true;
                smoother = gs;
            }
        }

        if (SFEM_DEBUG) {
            array_write(
                    comm, "./rhs.raw", SFEM_MPI_REAL_T, rhs->data(), fs->n_dofs(), fs->n_dofs());
            array_write(
                    comm, "./diag.raw", SFEM_MPI_REAL_T, diag->data(), fs->n_dofs(), fs->n_dofs());
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

    f->set_output_dir(output_path);
    auto output = f->output();

    if (SFEM_MG) {
        auto c = sfem::create_buffer<real_t>(fs->n_dofs(), es);
        auto r = sfem::create_buffer<real_t>(fs->n_dofs(), es);

        //  Coarse level
        auto fs_coarse = fs->derefine();
        auto f_coarse = f->derefine(fs_coarse, true);

        std::shared_ptr<sfem::Operator<real_t>> linear_op_coarse;
        if (SFEM_COARSE_MATRIX_FREE) {
            linear_op_coarse = sfem::make_linear_op(f_coarse);
        } else {
            linear_op_coarse = crs_hessian(*f_coarse, es);
        }

        auto c_coarse = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
        auto r_coarse = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
        auto diag_coarse = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
        auto solver_coarse = sfem::create_cg<real_t>(linear_op_coarse, es);

        {
            solver_coarse->verbose = false;
            solver_coarse->set_max_it(10000);
            solver_coarse->set_atol(1e-8);
            solver_coarse->set_rtol(1e-8);

            if (SFEM_USE_PRECONDITIONER) {
                f_coarse->hessian_diag(nullptr, diag_coarse->data());
                auto preconditioner = sfem::create_inverse_diagonal_scaling(diag_coarse, es);
                solver_coarse->set_preconditioner_op(preconditioner);
                solver_coarse->set_initial_guess_zero(true);
            }
        }

        smoother->set_initial_guess_zero(false);

        auto restriction = create_hierarchical_restriction(f, es);
        auto prolongation = create_hierarchical_prolongation(f, es);

        f->apply_constraints(x->data());
        f->apply_constraints(rhs->data());

        // Multigrid
        auto mg = sfem::create_mg<real_t>(es);
        mg->add_level(linear_op, smoother, nullptr, restriction);
        mg->add_level(nullptr, solver_coarse, prolongation, nullptr);
        mg->set_max_it(SFEM_MAX_IT);
        mg->set_atol(SFEM_TOL);

        init_tock = MPI_Wtime();

        solve_tick = MPI_Wtime();
        mg->apply(rhs->data(), x->data());
        solve_tock = MPI_Wtime();

    } else {
#if 0
    auto solver = smoother;
    solver->set_max_it(100);
    solver->set_op(linear_op);

#else  // CG solver

        auto solver = sfem::create_cg<real_t>(linear_op, es);

        if (smoother) {
            auto preconditioner = smoother;
            smoother->set_initial_guess_zero(true);
            solver->set_preconditioner_op(preconditioner);
        } else if (SFEM_USE_PRECONDITIONER) {
            auto preconditioner = sfem::create_inverse_diagonal_scaling(diag, es);
            solver->set_preconditioner_op(preconditioner);
        }

        solver->verbose = true;
        solver->set_atol(SFEM_TOL);
        solver->set_max_it(SFEM_MAX_IT);

#endif
        solver->set_op(linear_op);

        init_tock = MPI_Wtime();
        solve_tick = MPI_Wtime();
        solver->apply(rhs->data(), x->data());
        solve_tock = MPI_Wtime();
    }

    double compute_tock = MPI_Wtime();

    auto r = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    real_t rtr = residual(*linear_op, rhs->data(), x->data(), r->data(), es);

    // -------------------------------
    // Write output
    // -------------------------------

#ifdef SFEM_ENABLE_CUDA
    auto h_x = sfem::to_host(x);
    auto h_rhs = sfem::to_host(rhs);
    auto h_r = sfem::to_host(r);
#else
    auto h_x = x;
    auto h_rhs = rhs;
    auto h_r = r;
#endif

    output->write("x", h_x->data());
    output->write("rhs", h_rhs->data());
    output->write("residual", h_r->data());

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("%s (%s):\n", argv[0], type_to_string(fs->element_type()));
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #dofs %ld\n",
               (long)m->n_elements(),
               (long)m->n_nodes(),
               (long)fs->n_dofs());
        printf("TTS:\t\t%g [s], compute %g [s] (solve: %g [s], init: %g [s])\n",
               tock - tick,
               compute_tock - compute_tick,
               solve_tock - solve_tick,
               init_tock - init_tick);
        printf("residual:\t%g\n", rtr);
        printf("----------------------------------------\n");
    }

    return MPI_Finalize();
}
