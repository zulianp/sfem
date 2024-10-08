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

#ifdef SFEM_ENABLE_CUDA
    sfem::register_device_ops();
#endif

    const char *SFEM_OPERATOR = "Laplacian";
    bool SFEM_USE_GPU = true;
    int SFEM_BLOCK_SIZE = 1;
    int SFEM_ELEMENT_REFINE_LEVEL = 0;

    SFEM_READ_ENV(SFEM_OPERATOR, );
    SFEM_READ_ENV(SFEM_USE_GPU, atoi);
    SFEM_READ_ENV(SFEM_BLOCK_SIZE, atoi);
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

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

    auto fine_op = sfem::make_linear_op(f);
    auto fs_coarse = fs->derefine();
    auto f_coarse = f->derefine(fs_coarse, true);
    auto coarse_op = sfem::make_linear_op(f_coarse);

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
            // data[i] = 1;
        }
    }

    std::shared_ptr<sfem::Buffer<real_t>> input;

#if SFEM_ENABLE_CUDA
    if (es == sfem::MEMORY_SPACE_DEVICE) {
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

    prolongation->apply(input->data(), prolongated->data());
    fine_op->apply(prolongated->data(), Ax_fine->data());
    restriction->apply(Ax_fine->data(), restricted->data());
    coarse_op->apply(input->data(), Ax_coarse->data());

    printf("#elements %ld #ndofs fine %ld coarse %ld\n",
           m->n_elements(),
           fs->n_dofs(),
           fs_coarse->n_dofs());

    input->print(std::cout);
    prolongated->print(std::cout);
    Ax_fine->print(std::cout);

    if (0) {
        auto upanddown = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
        restriction->apply(prolongated->data(), upanddown->data());
        upanddown->print(std::cout);
    }

    // Compare two results
#if SFEM_ENABLE_CUDA
    auto h_actual = sfem::to_host(restricted);
    auto h_expected = sfem::to_host(Ax_coarse);
#else
    auto h_actual = restricted;
    auto h_expected = Ax_coarse;
#endif
    {
        ptrdiff_t n = fs_coarse->n_dofs();
        auto actual = h_actual->data();
        auto expected = h_expected->data();
        for (ptrdiff_t i = 0; i < n; i++) {
            // actual: is composition of operators
            // expected: is application of coarse operator
            real_t diff = fabs(actual[i] - expected[i]);
            if (diff > 1e-12) {
                printf("%ld) %g != %g (%g, %g)\n",
                       i,
                       actual[i],
                       expected[i],
                       diff,
                       actual[i] / expected[i]);
            }
        }
    }

    f->set_output_dir(output_path);
    auto output = f->output();

    return MPI_Finalize();
}
