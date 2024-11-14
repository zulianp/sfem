#include <memory>
#include "sfem_Function.hpp"

#include "sfem_base.h"
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

#define OP_HEADERS()                                                            \
    do {                                                                        \
        printf("Op,\t\tTTS [s],\tRTP [MDOF/s],\tBW [MDOF/s],\t(rows, cols)\n"); \
    } while (0)

#define OP_TIME(op, x, y)                                      \
    do {                                                       \
        sfem::device_synchronize();                            \
        double start = MPI_Wtime();                            \
        op->apply(x, y);                                       \
        sfem::device_synchronize();                            \
        double stop = MPI_Wtime();                             \
        double elapsed = stop - start;                         \
        printf("%s,\t%.5f,\t%.1f,\t\t%.1f,\t\t(%ld, %ld)\n",   \
               #op,                                            \
               elapsed,                                        \
               1e-6 * (op)->rows() / elapsed,                  \
               1e-6 * ((op)->rows() + (op)->cols()) / elapsed, \
               (op)->rows(),                                   \
               (op)->cols());                                  \
        fflush(stdout);                                        \
    } while (0)

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

    const char *SFEM_OPERATOR = "Laplacian";
    bool SFEM_USE_GPU = true;
    int SFEM_BLOCK_SIZE = 1;
    int SFEM_ELEMENT_REFINE_LEVEL = 0;
    int SFEM_PRINT_VECTORS = 0;
    int SFEM_SKIP_VERIFICATION = 0;
    int SFEM_MATRIX_FREE = 1;
    int SFEM_COARSE_MATRIX_FREE = 1;
    int SFEM_USE_BSR = 1;
    int SFEM_COARSE_USE_BSR = 1;

    SFEM_READ_ENV(SFEM_OPERATOR, );
    SFEM_READ_ENV(SFEM_USE_GPU, atoi);
    SFEM_READ_ENV(SFEM_BLOCK_SIZE, atoi);
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);
    SFEM_READ_ENV(SFEM_PRINT_VECTORS, atoi);
    SFEM_READ_ENV(SFEM_SKIP_VERIFICATION, atoi);
    SFEM_READ_ENV(SFEM_MATRIX_FREE, atoi);
    SFEM_READ_ENV(SFEM_COARSE_MATRIX_FREE, atoi);
    SFEM_READ_ENV(SFEM_COARSE_USE_BSR, atoi);

    sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;

    if (SFEM_USE_GPU) {
        es = sfem::EXECUTION_SPACE_DEVICE;
    }

    const char *folder = argv[1];
    const char *output_path = argv[2];

    auto m = sfem::Mesh::create_from_file(comm, folder);
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
    auto x = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);

    op->initialize();
    // f->add_constraint(conds);
    f->add_operator(op);

    std::shared_ptr<sfem::Operator<real_t>> fine_op, coarse_op;

    if (SFEM_MATRIX_FREE) {
        fine_op = sfem::make_linear_op(f);
    } else {
        if (fs->block_size() == 1) {
            fine_op = sfem::hessian_crs(f, nullptr, es);
        } else {
            if (SFEM_USE_BSR) {
                fine_op = sfem::hessian_bsr(f, nullptr, es);
            } else {
                fine_op = sfem::hessian_bcrs_sym(f, nullptr, es);
            }
        }
    }

    auto fs_coarse = fs->derefine();
    auto f_coarse = f->derefine(fs_coarse, true);

    if (SFEM_COARSE_MATRIX_FREE) {
        coarse_op = sfem::make_linear_op(f_coarse);
    } else {
        if (fs->block_size() == 1) {
            coarse_op = sfem::hessian_crs(f_coarse, nullptr, es);
        } else {
            if (SFEM_COARSE_USE_BSR) {
                coarse_op = sfem::hessian_bsr(f_coarse, nullptr, es);
            } else {
                coarse_op = sfem::hessian_bcrs_sym(f_coarse, nullptr, es);
            }
        }
    }

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

    auto h_input = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), sfem::MEMORY_SPACE_HOST);
    {
        ptrdiff_t n = fs_coarse->n_dofs();
        auto data = h_input->data();
        for (ptrdiff_t i = 0; i < n; i++) {
            data[i] = i;
            // data[i] = i % 2;
        }
    }

    std::shared_ptr<sfem::Buffer<real_t>> input;

#ifdef SFEM_ENABLE_CUDA
    if (es == sfem::EXECUTION_SPACE_DEVICE) {
        input = sfem::to_device(h_input);
    } else
#endif
    {
        input = h_input;
    }

    auto prolongated = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto Ax_fine = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto restricted = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
    auto Ax_coarse = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);

    double tick = MPI_Wtime();

    OP_HEADERS();
    OP_TIME(coarse_op, input->data(), Ax_coarse->data());
    OP_TIME(prolongation, input->data(), prolongated->data());
    OP_TIME(fine_op, prolongated->data(), Ax_fine->data());
    OP_TIME(restriction, Ax_fine->data(), restricted->data());

    double tock = MPI_Wtime();

    printf("#elements %ld #ndofs fine %ld coarse %ld\nTTS: %g [s]\n",
           m->n_elements(),
           fs->n_dofs(),
           fs_coarse->n_dofs(),
           tock - tick);

    if (SFEM_PRINT_VECTORS) {
#ifdef SFEM_ENABLE_CUDA
        sfem::to_host(input)->print(std::cout);
        sfem::to_host(prolongated)->print(std::cout);
        sfem::to_host(Ax_fine)->print(std::cout);
        sfem::to_host(Ax_coarse)->print(std::cout);
        sfem::to_host(restricted)->print(std::cout);
#else
        input->print(std::cout);
        prolongated->print(std::cout);
        Ax_fine->print(std::cout);
        Ax_coarse->print(std::cout);
        restricted->print(std::cout);
#endif
    }

    if (0)  //
    {
        auto upanddown = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
        restriction->apply(prolongated->data(), upanddown->data());
        upanddown->print(std::cout);
    }

    auto error = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), sfem::MEMORY_SPACE_HOST);

    if (!SFEM_SKIP_VERIFICATION) {
        // Compare two results
#ifdef SFEM_ENABLE_CUDA
        auto h_actual = sfem::to_host(restricted);
        auto h_expected = sfem::to_host(Ax_coarse);
#else
        auto h_actual = restricted;
        auto h_expected = Ax_coarse;
#endif
        {
            // printf("dof actual != expected, (diff, actual/expected)\n");
            auto err = error->data();
            ptrdiff_t n = fs_coarse->n_dofs();
            auto actual = h_actual->data();
            auto expected = h_expected->data();

            real_t largest_diff = 0;
            real_t largest_diff_factor = 0;
            ptrdiff_t arg_largest_diff = -1;
            for (ptrdiff_t i = 0; i < n; i++) {
                // actual: is composition of operators
                // expected: is application of coarse operator
                real_t diff = fabs(actual[i] - expected[i]);
                err[i] = diff;
                if (diff > 1e-8) {
                    printf("%ld) %g != %g (%g, %g)\n",
                           i,
                           (double)actual[i],
                           (double)expected[i],
                           (double)diff,
                           (double)actual[i] / expected[i]);
                }

                if (diff > largest_diff) {
                    largest_diff = diff;
                    arg_largest_diff = i;
                    largest_diff_factor = actual[i] / expected[i];
                }
            }

            if (arg_largest_diff != -1) {
                printf("largest_diff(%ld) = %g, %g\n",
                       arg_largest_diff,
                       largest_diff,
                       largest_diff_factor);
            }
        }

        f_coarse->set_output_dir(output_path);
        auto output = f_coarse->output();
        output->write("error", error->data());
    }

    return MPI_Finalize();
}
