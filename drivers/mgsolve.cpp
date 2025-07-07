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

    if (SFEM_USE_GPU) {
        es = sfem::EXECUTION_SPACE_DEVICE;
    }

    // -------------------------------
    // Read inputs
    // -------------------------------

    const char *folder = argv[1];
    auto m = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), folder);

    const char *SFEM_OPERATOR = "Laplacian";
    const char *SFEM_FINE_OP_TYPE = "MF";
    const char *SFEM_COARSE_OP_TYPE = "MF";

    int SFEM_BLOCK_SIZE = 1;
    int SFEM_USE_PRECONDITIONER = 0;
    int SFEM_USE_CHEB = 0;
    int SFEM_DEBUG = 0;
    int SFEM_MG = 0;
    int SFEM_MAX_IT = 4000;
    int SFEM_SMOOTHER_SWEEPS = 3;
    int SFEM_USE_MG_PRECONDITIONER = 0;
    int SFEM_WRITE_OUTPUT = 1;
    float SFEM_CHEB_EIG_MAX_SCALE = 1.02;
    float SFEM_TOL = 1e-9;
    double SFEM_COARSE_TOL = 1e-10;
    double SFEM_CHEB_EIG_TOL = 1e-5;
    int SFEM_ELEMENT_REFINE_LEVEL = 0;
    int SFEM_VERBOSITY_LEVEL = 1;

    SFEM_READ_ENV(SFEM_OPERATOR, );
    SFEM_READ_ENV(SFEM_FINE_OP_TYPE, );
    SFEM_READ_ENV(SFEM_COARSE_OP_TYPE, );

    SFEM_READ_ENV(SFEM_BLOCK_SIZE, atoi);
    SFEM_READ_ENV(SFEM_USE_PRECONDITIONER, atoi);
    SFEM_READ_ENV(SFEM_USE_CHEB, atoi);
    SFEM_READ_ENV(SFEM_DEBUG, atoi);
    SFEM_READ_ENV(SFEM_MG, atoi);
    SFEM_READ_ENV(SFEM_MAX_IT, atoi);
    SFEM_READ_ENV(SFEM_USE_MG_PRECONDITIONER, atoi);
    SFEM_READ_ENV(SFEM_WRITE_OUTPUT, atoi);
    SFEM_READ_ENV(SFEM_CHEB_EIG_MAX_SCALE, atof);
    SFEM_READ_ENV(SFEM_TOL, atof);
    SFEM_READ_ENV(SFEM_VERBOSITY_LEVEL, atoi);
    SFEM_READ_ENV(SFEM_COARSE_TOL, atof);

    SFEM_READ_ENV(SFEM_SMOOTHER_SWEEPS, atoi);
    SFEM_READ_ENV(SFEM_CHEB_EIG_TOL, atof);
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);
    

    printf("SFEM_OPERATOR: %s\n"
           "SFEM_FINE_OP_TYPE: %s\n"
           "SFEM_COARSE_OP_TYPE: %s\n"
           "SFEM_BLOCK_SIZE: %d\n"
           "SFEM_USE_PRECONDITIONER: %d\n"
           "SFEM_USE_CHEB: %d\n"
           "SFEM_DEBUG: %d\n"
           "SFEM_MG: %d\n"
           "SFEM_USE_MG_PRECONDITIONER: %d\n"
           "SFEM_WRITE_OUTPUT: %d\n"
           "SFEM_CHEB_EIG_MAX_SCALE: %f\n"
           "SFEM_TOL: %f\n"
           "SFEM_SMOOTHER_SWEEPS: %d\n"
           "SFEM_CHEB_EIG_TOL: %g\n"
           "SFEM_ELEMENT_REFINE_LEVEL: %d\n",
           SFEM_OPERATOR,
           SFEM_FINE_OP_TYPE,
           SFEM_COARSE_OP_TYPE,
           SFEM_BLOCK_SIZE,
           SFEM_USE_PRECONDITIONER,
           SFEM_USE_CHEB,
           SFEM_DEBUG,
           SFEM_MG,
           SFEM_USE_MG_PRECONDITIONER,
           SFEM_WRITE_OUTPUT,
           SFEM_CHEB_EIG_MAX_SCALE,
           SFEM_TOL,
           SFEM_SMOOTHER_SWEEPS,
           SFEM_CHEB_EIG_TOL,
           SFEM_ELEMENT_REFINE_LEVEL);

#ifdef SFEM_ENABLE_CUDA
    sfem::register_device_ops();
#endif

    auto fs = sfem::FunctionSpace::create(m, SFEM_BLOCK_SIZE);

    if (SFEM_ELEMENT_REFINE_LEVEL > 0) {
        fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    }

#ifdef SFEM_ENABLE_CUDA
    {
        auto elements = fs->device_elements();
        if (!elements) {
            elements = create_device_elements(fs, fs->element_type());
            fs->set_device_elements(elements);
        }
    }
#endif

    auto conds = sfem::create_dirichlet_conditions_from_env(fs, es);
    auto f = sfem::Function::create(fs);

    printf("Running %s\n", argv[0]);
    printf("#elements %ld #nodes %ld #dofs %ld\n",
           (long)m->n_elements(),
           (long)m->n_nodes(),
           (long)fs->n_dofs());

    fflush(stderr);
    fflush(stdout);

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

    linear_op = sfem::create_linear_operator(SFEM_FINE_OP_TYPE, f, x, es);

    if (SFEM_USE_CHEB) {
        auto cheb = sfem::create_cheb3<real_t>(linear_op, es);

        // Power-method
        cheb->eigen_solver_tol = SFEM_CHEB_EIG_TOL;
        cheb->init_with_ones();
        // cheb->init_with_random();

        cheb->scale_eig_max = SFEM_CHEB_EIG_MAX_SCALE;
        cheb->set_max_it(SFEM_SMOOTHER_SWEEPS);
        cheb->set_initial_guess_zero(false);
        smoother = cheb;
    } else if (SFEM_USE_PRECONDITIONER) {
        int err = f->hessian_diag(nullptr, diag->data());
        assert(!err);
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
        linear_op_coarse = sfem::create_linear_operator(SFEM_COARSE_OP_TYPE, f_coarse, nullptr, es);

        auto c_coarse = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
        auto r_coarse = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
        auto diag_coarse = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
        auto solver_coarse = sfem::create_cg<real_t>(linear_op_coarse, es);

        {
            solver_coarse->verbose = SFEM_VERBOSITY_LEVEL >= 2;
            solver_coarse->set_max_it(40000);
            solver_coarse->set_atol(SFEM_COARSE_TOL);
            solver_coarse->set_rtol(1e-8);

            if (SFEM_USE_PRECONDITIONER) {
                f_coarse->hessian_diag(nullptr, diag_coarse->data());
                auto preconditioner = sfem::create_inverse_diagonal_scaling(diag_coarse, es);
                solver_coarse->set_preconditioner_op(preconditioner);
                solver_coarse->set_initial_guess_zero(true);
            }
        }

        smoother->set_initial_guess_zero(false);

        auto restriction = sfem::create_hierarchical_restriction(fs, fs_coarse, es);
        auto prolong_unconstr = sfem::create_hierarchical_prolongation(fs_coarse, fs, es);
        auto prolongation = sfem::make_op<real_t>(
                prolong_unconstr->rows(),
                prolong_unconstr->cols(),
                [=](const real_t *const from, real_t *const to) {
                    prolong_unconstr->apply(from, to);
                    f->apply_zero_constraints(to);
                },
                es);

        f->apply_constraints(x->data());
        f->apply_constraints(rhs->data());

        // Multigrid
        auto mg = sfem::create_mg<real_t>(es);
        mg->debug = SFEM_DEBUG;
        mg->add_level(linear_op, smoother, nullptr, restriction);
        mg->add_level(nullptr, solver_coarse, prolongation, nullptr);
        mg->set_max_it(SFEM_MAX_IT);
        mg->set_atol(SFEM_TOL);

        fflush(stderr);
        fflush(stdout);

        init_tock = MPI_Wtime();

        solve_tick = MPI_Wtime();

        if (SFEM_USE_MG_PRECONDITIONER) {
            auto linear_op = sfem::make_linear_op(f);
            auto ksp = sfem::create_cg<real_t>(linear_op, es);
            ksp->check_each = 1;
            ksp->verbose = true;
            mg->set_max_it(1);
            mg->set_atol(0);
            mg->verbose = false;
            ksp->set_preconditioner_op(mg);
            ksp->set_max_it(SFEM_MAX_IT);
            ksp->apply(rhs->data(), x->data());
        } else {
            mg->apply(rhs->data(), x->data());
        }

        solve_tock = MPI_Wtime();

    } else {
        auto solver = sfem::create_cg<real_t>(linear_op, es);
        // auto solver = sfem::create_bcgs<real_t>(linear_op, es);

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

        solver->set_op(linear_op);

        init_tock = MPI_Wtime();
        solve_tick = MPI_Wtime();
        solver->apply(rhs->data(), x->data());
        solve_tock = MPI_Wtime();
    }

    double compute_tock = MPI_Wtime();

    auto r = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    real_t rtr = residual(*linear_op, rhs->data(), x->data(), r->data());

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

    if (SFEM_WRITE_OUTPUT) {
        output->write("x", h_x->data());
        output->write("rhs", h_rhs->data());
        output->write("residual", h_r->data());
    }

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
