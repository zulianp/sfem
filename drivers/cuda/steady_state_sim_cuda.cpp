#include "mpi.h"
#include "sfem_Function.hpp"
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_base.h"
#include "sfem_cuda_blas.h"

#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"


#include <vector>


template <typename T>
void sfem_cuda_init_solver(sfem::ConjugateGradient<T> &cg) {
    cg.allocate = d_allocate;
    cg.destroy = d_destroy;
    cg.copy = d_copy;
    cg.dot = d_dot;
    cg.axpby = d_axpby;
}

template <typename T>
void sfem_cuda_init_solver(sfem::BiCGStab<T> &cg) {
    cg.allocate = d_allocate;
    cg.destroy = d_destroy;
    cg.copy = d_copy;
    cg.dot = d_dot;
    cg.axpby = d_axpby;
    cg.zaxpby = d_zaxpby;
}

using Solver_t = sfem::ConjugateGradient<real_t>;
// using Solver_t = sfem::BiCGStab<real_t>;

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

    SFEM_READ_ENV(SFEM_OPERATOR, );
    SFEM_READ_ENV(SFEM_BLOCK_SIZE, atoi);

    auto fs = sfem::FunctionSpace::create(m, SFEM_BLOCK_SIZE);
    auto conds = sfem::DirichletConditions::create_from_env(fs);
    auto f = sfem::Function::create(fs);

    Solver_t solver;
    solver.max_it = 9000;
    solver.tol = 1e-10;

    real_t *d_x;
    real_t *d_b;

    if(SFEM_USE_GPU) {
        printf("Using GPU...\n");
        sfem_cuda_init_solver(solver);
        
        sfem::register_device_ops();
        auto le = sfem::Factory::create_op(fs, (std::string("gpu:") + SFEM_OPERATOR).c_str());
        le->initialize();

        auto d_conds = sfem::to_device(conds);
        
        f->add_constraint(d_conds);
        f->add_operator(le);

        d_x = d_allocate(fs->n_dofs());
        d_b = d_allocate(fs->n_dofs());
        
    } else {
        solver.default_init();

        auto le = sfem::Factory::create_op(fs, SFEM_OPERATOR);
        le->initialize();

        f->add_constraint(conds);
        f->add_operator(le);

        d_x = (real_t*)calloc(fs->n_dofs(), sizeof(real_t));
        d_b = (real_t*)calloc(fs->n_dofs(), sizeof(real_t));
    }

    // -------------------------------
    // Solver set-up
    // -------------------------------

    solver.apply_op = [=](const real_t *const x, real_t *const y) {
        if(SFEM_USE_GPU) {
            d_memset(y, 0, fs->n_dofs() * sizeof(real_t));
        } else {
            memset(y, 0, fs->n_dofs() * sizeof(real_t));
        }

        f->apply(nullptr, x, y);
    };

    // -------------------------------
    // Solve
    // -------------------------------
    double solve_tick = MPI_Wtime();

    f->apply_constraints(d_x);
    f->apply_constraints(d_b);

    solver.max_it = 40000;
    solver.apply(fs->n_dofs(), d_b, d_x);

    double solve_tock = MPI_Wtime();

    // -------------------------------
    // Write output
    // -------------------------------

    f->set_output_dir(output_path);
    auto output = f->output();

    if(SFEM_USE_GPU) {
        std::vector<real_t> x(fs->n_dofs(), 0);
        device_to_host(fs->n_dofs(), d_x, x.data());
        output->write("x", x.data());
    } else {
        output->write("x", d_x);
    }

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)m->n_elements(), (long)m->n_nodes());
        printf("TTS:\t\t\t%g seconds (solve: %g)\n", tock - tick, solve_tock - solve_tick);
    }

    return MPI_Finalize();
}
