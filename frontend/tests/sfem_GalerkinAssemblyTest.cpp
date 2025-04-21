#include <memory>

#include "sfem_test.h"

#include "sfem_Function.hpp"

#include "sfem_Buffer.hpp"
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
        double stop    = MPI_Wtime();                          \
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

int test_cube() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    const char *SFEM_OPERATOR       = "Laplacian";
    const char *SFEM_FINE_OP_TYPE   = "MF";
    const char *SFEM_COARSE_OP_TYPE = "MF";

    int SFEM_ELEMENT_REFINE_LEVEL = 4;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int SFEM_BASE_RESOLUTION = 1;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    int SFEM_ELEMENT_DEREFINE = 1;
    SFEM_READ_ENV(SFEM_ELEMENT_DEREFINE, atoi);

    int SFEM_DEBUG_EXPORT = 0;
    SFEM_READ_ENV(SFEM_DEBUG_EXPORT, atoi);

    int  block_size = 1;
    auto m          = sfem::Mesh::create_hex8_cube(
            comm, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, 0, 0, 0, 1, 1, 1);

    auto fs = sfem::FunctionSpace::create(m, block_size);

    fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    fs->semi_structured_mesh().apply_hierarchical_renumbering();

#ifdef SFEM_ENABLE_CUDA
    {
        auto elements = fs->device_elements();
        if (!elements) {
            elements = create_device_elements(fs, fs->element_type());
            fs->set_device_elements(elements);
        }
    }
#endif

    auto f  = sfem::Function::create(fs);
    auto x  = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);

    op->initialize();
    f->add_operator(op);

    std::shared_ptr<sfem::Operator<real_t>> fine_op, coarse_op;

    printf("Fine op (%d):\t%s\n", SFEM_ELEMENT_REFINE_LEVEL, SFEM_FINE_OP_TYPE);
    fine_op = sfem::create_linear_operator(SFEM_FINE_OP_TYPE, f, nullptr, es);

    auto levels    = fs->semi_structured_mesh().derefinement_levels();
    auto fs_coarse = fs->derefine(levels[SFEM_ELEMENT_DEREFINE]);
    auto f_coarse  = f->derefine(fs_coarse, true);

    printf("Coarse op (%d):\t%s\n", levels[SFEM_ELEMENT_DEREFINE], SFEM_COARSE_OP_TYPE);
    coarse_op = sfem::create_linear_operator(SFEM_COARSE_OP_TYPE, f_coarse, nullptr, es);

    auto restriction      = sfem::create_hierarchical_restriction(fs, fs_coarse, es);
    auto prolong_unconstr = sfem::create_hierarchical_prolongation(fs_coarse, fs, es);
    auto prolongation     = sfem::make_op<real_t>(
            prolong_unconstr->rows(),
            prolong_unconstr->cols(),
            [=](const real_t *const from, real_t *const to) {
                prolong_unconstr->apply(from, to);
                f->apply_zero_constraints(to);
            },
            es);

    auto h_input = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), sfem::MEMORY_SPACE_HOST);
    {
        ptrdiff_t n      = fs_coarse->n_dofs();
        auto      data   = h_input->data();
        auto      points = fs_coarse->semi_structured_mesh().points()->data();
        for (ptrdiff_t i = 0; i < n; i++) {
            data[i] = points[0][i] * points[0][i];
            // data[i] = points[0][i];
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
    auto Ax_fine     = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    auto restricted  = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
    auto Ax_coarse   = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);

    double tick = MPI_Wtime();

    OP_HEADERS();
    OP_TIME(coarse_op, input->data(), Ax_coarse->data());
    OP_TIME(prolongation, input->data(), prolongated->data());
    OP_TIME(fine_op, prolongated->data(), Ax_fine->data());

    // sfem::blas<real_t>(es)->values(Ax_fine->size(), 1.0, Ax_fine->data());

    OP_TIME(restriction, Ax_fine->data(), restricted->data());

    double tock = MPI_Wtime();

    printf("#elements %ld #ndofs fine %ld coarse %ld\nTTS: %g [s]\n",
           m->n_elements(),
           fs->n_dofs(),
           fs_coarse->n_dofs(),
           tock - tick);

    auto error = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), sfem::MEMORY_SPACE_HOST);

    // Compare two results
#ifdef SFEM_ENABLE_CUDA
    auto h_restricted = sfem::to_host(restricted);
    auto h_Ax_coarse  = sfem::to_host(Ax_coarse);
#else
    auto h_restricted = restricted;
    auto h_Ax_coarse  = Ax_coarse;
#endif
    {
        auto      err      = error->data();
        ptrdiff_t n        = fs_coarse->n_dofs();
        auto      actual   = h_restricted->data();
        auto      expected = h_Ax_coarse->data();

        real_t    largest_diff        = 0;
        real_t    largest_diff_factor = 0;
        ptrdiff_t arg_largest_diff    = SFEM_PTRDIFF_INVALID;
        for (ptrdiff_t i = 0; i < n; i++) {
            // actual: is composition of operators
            // expected: is application of coarse operator
            real_t diff = fabs(actual[i] - expected[i]);
            err[i]      = diff;
            if (diff > 1e-8 || diff != diff) {
                printf("%ld) %g != %g (%g, %g)\n",
                       i,
                       (double)actual[i],
                       (double)expected[i],
                       (double)diff,
                       (double)actual[i] / expected[i]);
            }

            if (diff > largest_diff) {
                largest_diff        = diff;
                arg_largest_diff    = i;
                largest_diff_factor = actual[i] / expected[i];
            }
        }

        if (SFEM_DEBUG_EXPORT) {
            sfem::create_directory("galerkin");
            sfem::create_directory("galerkin/fields");

            {  // COARSE
                SFEM_TEST_ASSERT(fs_coarse->semi_structured_mesh().export_as_standard("galerkin") == SFEM_SUCCESS);

                sfem::Output out(fs_coarse);

                out.set_output_dir("galerkin/fields");
                SFEM_TEST_ASSERT(out.write("R", h_restricted->data()) == SFEM_SUCCESS);
                SFEM_TEST_ASSERT(out.write("u", h_input->data()) == SFEM_SUCCESS);
                SFEM_TEST_ASSERT(out.write("Ax_coarse", Ax_coarse->data()) == SFEM_SUCCESS);
                SFEM_TEST_ASSERT(out.write("err", error->data()) == SFEM_SUCCESS);
            }

            {  // FINE
#ifdef SFEM_ENABLE_CUDA
                auto h_prolongated = sfem::to_host(prolongated);
#else
                auto h_prolongated = prolongated;
#endif

                sfem::create_directory("galerkin_fine");
                sfem::create_directory("galerkin_fine/fields");
                SFEM_TEST_ASSERT(fs->semi_structured_mesh().export_as_standard("galerkin_fine") == SFEM_SUCCESS);

                sfem::Output out(fs);
                out.set_output_dir("galerkin_fine/fields");

                SFEM_TEST_ASSERT(out.write("P", h_prolongated->data()) == SFEM_SUCCESS);
            }
        }

        if (arg_largest_diff != -1) {
            fflush(stdout);
            printf("largest_diff(%ld) = %g, %g\n", arg_largest_diff, largest_diff, largest_diff_factor);
            SFEM_TEST_ASSERT(largest_diff < 1e-7);
        }
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

#ifdef SFEM_ENABLE_CUDA
    sfem::register_device_ops();
#endif

    SFEM_RUN_TEST(test_cube);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
