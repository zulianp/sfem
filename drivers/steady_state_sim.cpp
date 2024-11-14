#include "sfem_Function.hpp"

#include "sfem_API.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif
#include "sfem_Stationary.hpp"

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

#ifdef SFEM_ENABLE_CUDA
    sfem::register_device_ops();
#endif

    double tick = MPI_Wtime();

    // -------------------------------
    // Read inputs
    // -------------------------------

    const char *folder = argv[1];
    const char *output_path = argv[2];
    const char *SFEM_OPERATOR = "Laplacian";
    int SFEM_BLOCK_SIZE = 1;
    int SFEM_USE_PRECONDITIONER = 0;
    int SFEM_ELEMENT_REFINE_LEVEL = 0;
    int SFEM_MAX_IT = 1000;
    bool SFEM_USE_GPU = true;
    bool SFEM_USE_AMG = false;

    SFEM_READ_ENV(SFEM_OPERATOR, );
    SFEM_READ_ENV(SFEM_BLOCK_SIZE, atoi);
    SFEM_READ_ENV(SFEM_USE_PRECONDITIONER, atoi);
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);
    SFEM_READ_ENV(SFEM_MAX_IT, atoi);
    SFEM_READ_ENV(SFEM_USE_GPU, atoi);
    SFEM_READ_ENV(SFEM_USE_AMG, atoi);

    sfem::ExecutionSpace es =
            SFEM_USE_GPU ? sfem::EXECUTION_SPACE_DEVICE : sfem::EXECUTION_SPACE_HOST;

    // -------------------------------
    // Create discretization
    // -------------------------------

    auto m = sfem::Mesh::create_from_file(comm, folder);
    auto fs = sfem::FunctionSpace::create(m, SFEM_BLOCK_SIZE);

    if (SFEM_ELEMENT_REFINE_LEVEL > 0) {
        fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    }

    // -------------------------------
    // Create problem
    // -------------------------------

    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);
    op->initialize();
    auto conds = sfem::create_dirichlet_conditions_from_env(fs, es);
    auto f = sfem::Function::create(fs);
    f->add_constraint(conds);
    f->add_operator(op);

    auto x = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto rhs = sfem::create_buffer<real_t>(fs->n_dofs(), es);

    // -------------------------------
    // Create linear solver
    // -------------------------------

    std::shared_ptr<sfem::Operator<real_t>> solver;

    if (SFEM_USE_AMG) {
        auto stat_iter = sfem::h_stationary<real_t>();
        // stat_iter->set_op(linear_op);

        // auto smoother = sfem::h_lpsmoother<real_t>(diag);
        // stat_iter->set_preconditioner_op(smoother);
        solver =  stat_iter;
    } else {
        auto linear_op = sfem::make_linear_op(f);
        auto cg = sfem::create_cg<real_t>(linear_op, es);
        cg->verbose = true;
        cg->set_max_it(SFEM_MAX_IT);
        cg->set_op(linear_op);
        solver = cg;
    }

    

    // -------------------------------
    // Solve
    // -------------------------------

    double solve_tick = MPI_Wtime();

    f->apply_constraints(x->data());
    f->apply_constraints(rhs->data());

    solver->apply(rhs->data(), x->data());

    double solve_tock = MPI_Wtime();

    // -------------------------------
    // Write output
    // -------------------------------

    f->set_output_dir(output_path);
    auto output = f->output();

#ifdef SFEM_ENABLE_CUDA
    auto h_x = sfem::to_host(x);
#else
    auto h_x = x;
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
