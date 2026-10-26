#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "smesh_semistructured.hpp"
#include "sshex8_linear_elasticity.hpp"
#include "sshex8_stencil_element_matrix_apply.hpp"

static void zero_values(const ptrdiff_t n, real_t *const values) { memset(values, 0, n * sizeof(real_t)); }

static real_t largest_abs_diff(const ptrdiff_t             n,
                               const real_t *const         a,
                               const real_t *const         b,
                               ptrdiff_t *const            argmax) {
    real_t    ret = 0;
    ptrdiff_t arg = SFEM_PTRDIFF_INVALID;

    for (ptrdiff_t i = 0; i < n; ++i) {
        const real_t diff = fabs(a[i] - b[i]);
        if (diff > ret || diff != diff) {
            ret = diff;
            arg = i;
        }
    }

    *argmax = arg;
    return ret;
}

static int test_sshex8_linear_elasticity_hyteg_stencil() {
    auto comm = sfem::Communicator::world();

    int SFEM_BASE_RESOLUTION = 4;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    int SFEM_ELEMENT_REFINE_LEVEL = 4;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int SFEM_VALIDATE = 1;
    SFEM_READ_ENV(SFEM_VALIDATE, atoi);

    int SFEM_THROUGHPUT_REPEAT = 10;
    SFEM_READ_ENV(SFEM_THROUGHPUT_REPEAT, atoi);

    const real_t mu     = 2.25;
    const real_t lambda = 1.75;

    auto macro_mesh = sfem::Mesh::create_hex8_cube(comm,
                                                   SFEM_BASE_RESOLUTION,
                                                   SFEM_BASE_RESOLUTION,
                                                   SFEM_BASE_RESOLUTION,
                                                   0,
                                                   0,
                                                   0,
                                                   1,
                                                   1,
                                                   1);
    auto mesh        = smesh::to_semistructured(SFEM_ELEMENT_REFINE_LEVEL, macro_mesh, true, false);
    auto matrix_mesh = smesh::derefine(mesh, 1);
    if (matrix_mesh && matrix_mesh->element_type(0) == smesh::PROTEUS_HEX8) {
        matrix_mesh = smesh::sshex_to_hex8(matrix_mesh);
    }

    const int       level   = smesh::semistructured_level(*mesh);
    const int       nxe     = smesh::sshex8_nxe(level);
    const ptrdiff_t n_nodes = mesh->n_nodes();
    const auto      es      = sfem::EXECUTION_SPACE_HOST;

    auto input       = sfem::create_buffer<real_t>(n_nodes * 3, es);
    auto direct_out  = sfem::create_buffer<real_t>(n_nodes * 3, es);
    auto current_out = sfem::create_buffer<real_t>(n_nodes * 3, es);
    auto hyteg_out   = sfem::create_buffer<real_t>(n_nodes * 3, es);
    auto op_out      = sfem::create_buffer<real_t>(n_nodes * 3, es);
    auto bench_out   = sfem::create_buffer<real_t>(n_nodes * 3, es);

    auto points = mesh->points()->data();
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        const real_t x = points[0][i];
        const real_t y = points[1][i];
        const real_t z = points[2][i];

        input->data()[i * 3 + 0] = 0.11 + x + 0.5 * y * z + 0.0078125 * ((i * 13 + 5) % 17);
        input->data()[i * 3 + 1] = -0.07 + y * y + 0.25 * z + 0.00390625 * ((i * 7 + 3) % 19);
        input->data()[i * 3 + 2] = 0.19 + z * x - 0.125 * y + 0.001953125 * ((i * 11 + 1) % 23);
    }

    zero_values(n_nodes * 3, direct_out->data());
    zero_values(n_nodes * 3, current_out->data());
    zero_values(n_nodes * 3, hyteg_out->data());
    zero_values(n_nodes * 3, op_out->data());
    zero_values(n_nodes * 3, bench_out->data());

    std::vector<std::shared_ptr<sfem::Buffer<scalar_t>>> block_matrices;
    std::vector<std::shared_ptr<sfem::Buffer<scalar_t>>> block_category_stencils;

    ptrdiff_t total_macro_elements = 0;
    ptrdiff_t total_local_dofs     = 0;
    ptrdiff_t total_microhexes     = 0;

    for (size_t b = 0; b < mesh->n_blocks(); ++b) {
        auto ss_block     = mesh->block(b);
        auto matrix_block = matrix_mesh->block(b);
        const ptrdiff_t ne = ss_block->n_elements();

        SFEM_TEST_ASSERT(ss_block->n_nodes_per_element() == nxe);
        SFEM_TEST_ASSERT(ne == matrix_block->n_elements());

        auto matrix = sfem::create_host_buffer<scalar_t>(ne * 24 * 24);
        SFEM_TEST_ASSERT(sshex8_linear_elasticity_element_matrix_cartesian(level,
                                                                           ne,
                                                                           matrix_mesh->n_nodes(),
                                                                           matrix_block->elements()->data(),
                                                                           matrix_mesh->points()->data(),
                                                                           mu,
                                                                           lambda,
                                                                           matrix->data()) == SFEM_SUCCESS);

        auto category_stencils = sfem::create_host_buffer<scalar_t>(ne * 27 * 27 * 9);
        SFEM_TEST_ASSERT(sshex8_linear_elasticity_element_matrix_to_category_stencils(ne,
                                                                                     matrix->data(),
                                                                                     category_stencils->data()) ==
                         SFEM_SUCCESS);

        block_matrices.push_back(matrix);
        block_category_stencils.push_back(category_stencils);

        total_macro_elements += ne;
        total_local_dofs += ne * nxe * 3;
        total_microhexes += ne * level * level * level;
    }

    if (SFEM_VALIDATE) {
        for (size_t b = 0; b < mesh->n_blocks(); ++b) {
            auto ss_block = mesh->block(b);
            const ptrdiff_t ne = ss_block->n_elements();

            SFEM_TEST_ASSERT(affine_sshex8_linear_elasticity_apply(level,
                                                                   ne,
                                                                   mesh->n_nodes(),
                                                                   ss_block->elements()->data(),
                                                                   mesh->points()->data(),
                                                                   mu,
                                                                   lambda,
                                                                   3,
                                                                   &input->data()[0],
                                                                   &input->data()[1],
                                                                   &input->data()[2],
                                                                   3,
                                                                   &direct_out->data()[0],
                                                                   &direct_out->data()[1],
                                                                   &direct_out->data()[2]) == SFEM_SUCCESS);

            SFEM_TEST_ASSERT(sshex8_stencil_element_matrix_apply3(level,
                                                                  ne,
                                                                  ss_block->elements()->data(),
                                                                  block_matrices[b]->data(),
                                                                  3,
                                                                  &input->data()[0],
                                                                  &input->data()[1],
                                                                  &input->data()[2],
                                                                  3,
                                                                  &current_out->data()[0],
                                                                  &current_out->data()[1],
                                                                  &current_out->data()[2]) == SFEM_SUCCESS);

            SFEM_TEST_ASSERT(sshex8_stencil_element_matrix_apply3_hyteg_stencil(level,
                                                                                ne,
                                                                                ss_block->elements()->data(),
                                                                                block_category_stencils[b]->data(),
                                                                                3,
                                                                                &input->data()[0],
                                                                                &input->data()[1],
                                                                                &input->data()[2],
                                                                                3,
                                                                                &hyteg_out->data()[0],
                                                                                &hyteg_out->data()[1],
                                                                                &hyteg_out->data()[2]) ==
                             SFEM_SUCCESS);
        }

        ptrdiff_t current_arg = SFEM_PTRDIFF_INVALID;
        ptrdiff_t hyteg_arg   = SFEM_PTRDIFF_INVALID;
        const real_t current_diff =
                largest_abs_diff(n_nodes * 3, current_out->data(), direct_out->data(), &current_arg);
        const real_t hyteg_diff = largest_abs_diff(n_nodes * 3, hyteg_out->data(), direct_out->data(), &hyteg_arg);

        auto space = sfem::FunctionSpace::create(mesh, 3);
        auto op    = sfem::create_op(space, "LinearElasticityHyTeG", es);
        SFEM_TEST_ASSERT(op != nullptr);
        SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);
        op->set_value_in_block("", "mu", mu);
        op->set_value_in_block("", "lambda", lambda);
        SFEM_TEST_ASSERT(op->apply(nullptr, input->data(), op_out->data()) == SFEM_SUCCESS);

        ptrdiff_t op_arg  = SFEM_PTRDIFF_INVALID;
        const real_t op_diff = largest_abs_diff(n_nodes * 3, op_out->data(), direct_out->data(), &op_arg);

        printf("SSHEX8 linear elasticity HyTeG validation level=%d base=%d current_packed_vs_direct(%ld)=%g "
               "category_stencil_vs_direct(%ld)=%g op_vs_direct(%ld)=%g\n",
               level,
               SFEM_BASE_RESOLUTION,
               current_arg,
               (double)current_diff,
               hyteg_arg,
               (double)hyteg_diff,
               op_arg,
               (double)op_diff);

        const real_t tolerance = sizeof(real_t) == 4 ? 5e-4 : 1e-8;
        SFEM_TEST_ASSERT(hyteg_diff < tolerance);
        SFEM_TEST_ASSERT(op_diff < tolerance);
    }

    if (SFEM_THROUGHPUT_REPEAT > 0) {
        zero_values(n_nodes * 3, bench_out->data());
        double tick = smesh::time_seconds();
        for (int r = 0; r < SFEM_THROUGHPUT_REPEAT; ++r) {
            for (size_t b = 0; b < mesh->n_blocks(); ++b) {
                auto ss_block = mesh->block(b);
                SFEM_TEST_ASSERT(affine_sshex8_linear_elasticity_apply(level,
                                                                       ss_block->n_elements(),
                                                                       mesh->n_nodes(),
                                                                       ss_block->elements()->data(),
                                                                       mesh->points()->data(),
                                                                       mu,
                                                                       lambda,
                                                                       3,
                                                                       &input->data()[0],
                                                                       &input->data()[1],
                                                                       &input->data()[2],
                                                                       3,
                                                                       &bench_out->data()[0],
                                                                       &bench_out->data()[1],
                                                                       &bench_out->data()[2]) == SFEM_SUCCESS);
            }
        }
        const double direct_elapsed = (smesh::time_seconds() - tick) / SFEM_THROUGHPUT_REPEAT;

        zero_values(n_nodes * 3, bench_out->data());
        tick = smesh::time_seconds();
        for (int r = 0; r < SFEM_THROUGHPUT_REPEAT; ++r) {
            for (size_t b = 0; b < mesh->n_blocks(); ++b) {
                auto ss_block = mesh->block(b);
                SFEM_TEST_ASSERT(sshex8_stencil_element_matrix_apply3(level,
                                                                      ss_block->n_elements(),
                                                                      ss_block->elements()->data(),
                                                                      block_matrices[b]->data(),
                                                                      3,
                                                                      &input->data()[0],
                                                                      &input->data()[1],
                                                                      &input->data()[2],
                                                                      3,
                                                                      &bench_out->data()[0],
                                                                      &bench_out->data()[1],
                                                                      &bench_out->data()[2]) == SFEM_SUCCESS);
            }
        }
        const double current_elapsed = (smesh::time_seconds() - tick) / SFEM_THROUGHPUT_REPEAT;

        zero_values(n_nodes * 3, bench_out->data());
        tick = smesh::time_seconds();
        for (int r = 0; r < SFEM_THROUGHPUT_REPEAT; ++r) {
            for (size_t b = 0; b < mesh->n_blocks(); ++b) {
                auto ss_block = mesh->block(b);
                SFEM_TEST_ASSERT(sshex8_stencil_element_matrix_apply3_hyteg_stencil(level,
                                                                                    ss_block->n_elements(),
                                                                                    ss_block->elements()->data(),
                                                                                    block_category_stencils[b]->data(),
                                                                                    3,
                                                                                    &input->data()[0],
                                                                                    &input->data()[1],
                                                                                    &input->data()[2],
                                                                                    3,
                                                                                    &bench_out->data()[0],
                                                                                    &bench_out->data()[1],
                                                                                    &bench_out->data()[2]) ==
                                 SFEM_SUCCESS);
            }
        }
        const double hyteg_elapsed = (smesh::time_seconds() - tick) / SFEM_THROUGHPUT_REPEAT;

        printf("SSHEX8 linear elasticity direct affine throughput level=%d base=%d macro_elements=%ld global_nodes=%ld "
               "local_dofs=%ld microhexes=%ld repeat=%d elapsed=%g MDOF/s=%g Mmicrohex/s=%g\n",
               level,
               SFEM_BASE_RESOLUTION,
               total_macro_elements,
               n_nodes,
               total_local_dofs,
               total_microhexes,
               SFEM_THROUGHPUT_REPEAT,
               direct_elapsed,
               1e-6 * total_local_dofs / direct_elapsed,
               1e-6 * total_microhexes / direct_elapsed);

        printf("SSHEX8 linear elasticity current packed HyTeG throughput level=%d base=%d macro_elements=%ld "
               "global_nodes=%ld local_dofs=%ld microhexes=%ld repeat=%d elapsed=%g MDOF/s=%g Mmicrohex/s=%g "
               "speedup_vs_direct=%g\n",
               level,
               SFEM_BASE_RESOLUTION,
               total_macro_elements,
               n_nodes,
               total_local_dofs,
               total_microhexes,
               SFEM_THROUGHPUT_REPEAT,
               current_elapsed,
               1e-6 * total_local_dofs / current_elapsed,
               1e-6 * total_microhexes / current_elapsed,
               direct_elapsed / current_elapsed);

        printf("SSHEX8 linear elasticity category HyTeG throughput level=%d base=%d macro_elements=%ld "
               "global_nodes=%ld local_dofs=%ld microhexes=%ld repeat=%d elapsed=%g MDOF/s=%g Mmicrohex/s=%g "
               "speedup_vs_current=%g speedup_vs_direct=%g\n",
               level,
               SFEM_BASE_RESOLUTION,
               total_macro_elements,
               n_nodes,
               total_local_dofs,
               total_microhexes,
               SFEM_THROUGHPUT_REPEAT,
               hyteg_elapsed,
               1e-6 * total_local_dofs / hyteg_elapsed,
               1e-6 * total_microhexes / hyteg_elapsed,
               current_elapsed / hyteg_elapsed,
               direct_elapsed / hyteg_elapsed);
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_sshex8_linear_elasticity_hyteg_stencil);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
