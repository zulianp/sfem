#include "mpi.h"
#include "sfem_Function.hpp"
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_base.h"
#include "sfem_cuda_blas.h"

#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"
#include "sfem_cuda_solver.hpp"

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

    SFEM_READ_ENV(SFEM_OPERATOR, );
    SFEM_READ_ENV(SFEM_BLOCK_SIZE, atoi);

    auto fs    = sfem::FunctionSpace::create(m, SFEM_BLOCK_SIZE);
    auto conds = sfem::DirichletConditions::create_from_env(fs);
    auto f     = sfem::Function::create(fs);

    std::shared_ptr<sfem::MatrixFreeLinearSolver<real_t>> solver;
    std::shared_ptr<sfem::Buffer<real_t>> b_x;
    std::shared_ptr<sfem::Buffer<real_t>> b_b;

    if (SFEM_USE_GPU) {
        printf("Using GPU...\n");
        solver = sfem::d_cg<real_t>();

        // Register CUDA kernels
        sfem::register_device_ops();
        auto le = sfem::Factory::create_op(
            fs, 
            sfem::d_op_str(SFEM_OPERATOR).c_str()
        );

        le->initialize();

        // Transfer Boundary conditions to device
        auto d_conds = sfem::to_device(conds);

        f->add_constraint(d_conds);
        f->add_operator(le);

        // Create device buffers
        b_x = sfem::d_buffer<real_t>(fs->n_dofs());
        b_b = sfem::d_buffer<real_t>(fs->n_dofs());
    } else {
        solver = sfem::h_cg<real_t>();

        auto le = sfem::Factory::create_op(fs, SFEM_OPERATOR);
        le->initialize();

        f->add_constraint(conds);
        f->add_operator(le);

        b_x = sfem::h_buffer<real_t>(fs->n_dofs());
        b_b = sfem::h_buffer<real_t>(fs->n_dofs());
    }

    // -------------------------------
    // Solver set-up
    // -------------------------------
    solver->set_op(sfem::make_op<real_t>([=](const real_t *const x, real_t *const y) {
        if (SFEM_USE_GPU) {
            d_memset(y, 0, fs->n_dofs() * sizeof(real_t));
        } else {
            memset(y, 0, fs->n_dofs() * sizeof(real_t));
        }

        f->apply(nullptr, x, y);
    }));

    // -------------------------------
    // Solve
    // -------------------------------
    double solve_tick = MPI_Wtime();

    f->apply_constraints(b_x->data());
    f->apply_constraints(b_b->data());

    solver->set_max_it(40000);
    solver->set_n_dofs(fs->n_dofs());
    solver->apply(b_b->data(), b_x->data());

    double solve_tock = MPI_Wtime();

    // -------------------------------
    // Write output
    // -------------------------------

    f->set_output_dir(output_path);
    auto output = f->output();

    if (SFEM_USE_GPU) {
        std::vector<real_t> x(fs->n_dofs(), 0);
        device_to_host(fs->n_dofs(), b_x->data(), x.data());
        output->write("x", x.data());

    } else {
        output->write("x", b_x->data());
    }

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)m->n_elements(), (long)m->n_nodes());
        printf("TTS:\t\t\t%g seconds (solve: %g)\n", tock - tick, solve_tock - solve_tick);
    }

    return MPI_Finalize();
}
