#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "smesh_semistructured.hpp"
#include "sshex8_laplacian.hpp"
#include "sshex8_stencil_element_matrix_apply.hpp"
#include "stencil3.hpp"

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

static int test_sshex8_hyteg_stencil() {
    auto comm = sfem::Communicator::world();

    int SFEM_BASE_RESOLUTION = 4;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    int SFEM_ELEMENT_REFINE_LEVEL = 4;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int SFEM_VALIDATE = 1;
    SFEM_READ_ENV(SFEM_VALIDATE, atoi);

    int SFEM_THROUGHPUT_REPEAT = 10;
    SFEM_READ_ENV(SFEM_THROUGHPUT_REPEAT, atoi);

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
    auto mesh       = smesh::to_semistructured(SFEM_ELEMENT_REFINE_LEVEL, macro_mesh, true, false);
    auto matrix_mesh = smesh::derefine(mesh, 1);
    if (matrix_mesh && matrix_mesh->element_type(0) == smesh::PROTEUS_HEX8) {
        matrix_mesh = smesh::sshex_to_hex8(matrix_mesh);
    }

    const int       level   = smesh::semistructured_level(*mesh);
    const int       nxe     = smesh::sshex8_nxe(level);
    const ptrdiff_t n_nodes = mesh->n_nodes();
    const auto      es      = sfem::EXECUTION_SPACE_HOST;

    auto input          = sfem::create_buffer<real_t>(n_nodes, es);
    auto direct_out     = sfem::create_buffer<real_t>(n_nodes, es);
    auto current_out    = sfem::create_buffer<real_t>(n_nodes, es);
    auto hyteg_out      = sfem::create_buffer<real_t>(n_nodes, es);
    auto vectorized_out = sfem::create_buffer<real_t>(n_nodes, es);
    auto bench_out      = sfem::create_buffer<real_t>(n_nodes, es);

    auto points = mesh->points()->data();
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        const real_t x = points[0][i];
        const real_t y = points[1][i];
        const real_t z = points[2][i];
        input->data()[i] = 0.17 + x * x + 0.5 * y * z - 0.125 * z + 0.00390625 * ((i * 17 + 3) % 29);
    }

    zero_values(n_nodes, direct_out->data());
    zero_values(n_nodes, current_out->data());
    zero_values(n_nodes, hyteg_out->data());
    zero_values(n_nodes, vectorized_out->data());
    zero_values(n_nodes, bench_out->data());

    std::vector<std::shared_ptr<sfem::Buffer<scalar_t>>> block_matrices;
    std::vector<std::shared_ptr<sfem::Buffer<scalar_t>>> block_stencils;

    ptrdiff_t total_macro_elements = 0;
    ptrdiff_t total_local_dofs     = 0;
    ptrdiff_t total_microhexes     = 0;

    for (size_t b = 0; b < mesh->n_blocks(); ++b) {
        auto ss_block     = mesh->block(b);
        auto matrix_block = matrix_mesh->block(b);
        const ptrdiff_t ne = ss_block->n_elements();

        SFEM_TEST_ASSERT(ss_block->n_nodes_per_element() == nxe);
        SFEM_TEST_ASSERT(ne == matrix_block->n_elements());

        auto matrix = sfem::create_host_buffer<scalar_t>(ne * 64);
        SFEM_TEST_ASSERT(sshex8_laplacian_element_matrix_cartesian(level,
                                                                   ne,
                                                                   matrix_mesh->n_nodes(),
                                                                   matrix_block->elements()->data(),
                                                                   matrix_mesh->points()->data(),
                                                                   matrix->data()) == SFEM_SUCCESS);

        auto stencil = sfem::create_host_buffer<scalar_t>(ne * 27);
#pragma omp parallel for
        for (ptrdiff_t e = 0; e < ne; ++e) {
            hex8_matrix_to_stencil(&matrix->data()[e * 64], &stencil->data()[e * 27]);
        }

        block_matrices.push_back(matrix);
        block_stencils.push_back(stencil);

        total_macro_elements += ne;
        total_local_dofs += ne * nxe;
        total_microhexes += ne * level * level * level;
    }

    if (SFEM_VALIDATE) {
        for (size_t b = 0; b < mesh->n_blocks(); ++b) {
            auto ss_block = mesh->block(b);
            const ptrdiff_t ne = ss_block->n_elements();

            SFEM_TEST_ASSERT(affine_sshex8_laplacian_apply(level,
                                                           ne,
                                                           ss_block->elements()->data(),
                                                           mesh->points()->data(),
                                                           input->data(),
                                                           direct_out->data()) == SFEM_SUCCESS);

            SFEM_TEST_ASSERT(sshex8_stencil_element_matrix_apply(level,
                                                                 ne,
                                                                 ss_block->elements()->data(),
                                                                 block_matrices[b]->data(),
                                                                 input->data(),
                                                                 current_out->data()) == SFEM_SUCCESS);

            SFEM_TEST_ASSERT(sshex8_stencil_element_matrix_apply_hyteg(level,
                                                                       ne,
                                                                       ss_block->elements()->data(),
                                                                       block_matrices[b]->data(),
                                                                       block_stencils[b]->data(),
                                                                       input->data(),
                                                                       hyteg_out->data()) == SFEM_SUCCESS);

            SFEM_TEST_ASSERT(sshex8_stencil_element_matrix_apply_hyteg_vectorized(level,
                                                                                  ne,
                                                                                  ss_block->elements()->data(),
                                                                                  block_matrices[b]->data(),
                                                                                  block_stencils[b]->data(),
                                                                                  input->data(),
                                                                                  vectorized_out->data()) == SFEM_SUCCESS);
        }

        ptrdiff_t current_arg       = SFEM_PTRDIFF_INVALID;
        ptrdiff_t hyteg_arg         = SFEM_PTRDIFF_INVALID;
        ptrdiff_t current_hyteg_arg = SFEM_PTRDIFF_INVALID;
        ptrdiff_t vectorized_arg    = SFEM_PTRDIFF_INVALID;
        const real_t current_diff       = largest_abs_diff(n_nodes, current_out->data(), direct_out->data(), &current_arg);
        const real_t hyteg_diff         = largest_abs_diff(n_nodes, hyteg_out->data(), direct_out->data(), &hyteg_arg);
        const real_t current_hyteg_diff = largest_abs_diff(n_nodes, hyteg_out->data(), current_out->data(), &current_hyteg_arg);
        const real_t vectorized_diff = largest_abs_diff(n_nodes, vectorized_out->data(), current_out->data(), &vectorized_arg);

        printf("SSHEX8 HyTeG stencil validation level=%d base=%d current_vs_direct(%ld)=%g hyteg_vs_direct(%ld)=%g "
               "hyteg_vs_current(%ld)=%g vectorized_vs_current(%ld)=%g\n",
               level,
               SFEM_BASE_RESOLUTION,
               current_arg,
               (double)current_diff,
               hyteg_arg,
               (double)hyteg_diff,
               current_hyteg_arg,
               (double)current_hyteg_diff,
               vectorized_arg,
               (double)vectorized_diff);

        const real_t tolerance = sizeof(real_t) == 4 ? 5e-4 : 1e-8;
        SFEM_TEST_ASSERT(current_hyteg_diff < tolerance);
        SFEM_TEST_ASSERT(vectorized_diff < tolerance);
    }

    double direct_elapsed = 0;
    double current_elapsed = 0;
    double hyteg_elapsed = 0;
    double vectorized_elapsed = 0;

    if (SFEM_THROUGHPUT_REPEAT > 0) {
        zero_values(n_nodes, bench_out->data());
        double tick = smesh::time_seconds();
        for (int r = 0; r < SFEM_THROUGHPUT_REPEAT; ++r) {
            for (size_t b = 0; b < mesh->n_blocks(); ++b) {
                auto ss_block = mesh->block(b);
                SFEM_TEST_ASSERT(affine_sshex8_laplacian_apply(level,
                                                               ss_block->n_elements(),
                                                               ss_block->elements()->data(),
                                                               mesh->points()->data(),
                                                               input->data(),
                                                               bench_out->data()) == SFEM_SUCCESS);
            }
        }
        direct_elapsed = (smesh::time_seconds() - tick) / SFEM_THROUGHPUT_REPEAT;

        zero_values(n_nodes, bench_out->data());
        tick = smesh::time_seconds();
        for (int r = 0; r < SFEM_THROUGHPUT_REPEAT; ++r) {
            for (size_t b = 0; b < mesh->n_blocks(); ++b) {
                auto ss_block = mesh->block(b);
                SFEM_TEST_ASSERT(sshex8_stencil_element_matrix_apply(level,
                                                                     ss_block->n_elements(),
                                                                     ss_block->elements()->data(),
                                                                     block_matrices[b]->data(),
                                                                     input->data(),
                                                                     bench_out->data()) == SFEM_SUCCESS);
            }
        }
        current_elapsed = (smesh::time_seconds() - tick) / SFEM_THROUGHPUT_REPEAT;

        zero_values(n_nodes, bench_out->data());
        tick = smesh::time_seconds();
        for (int r = 0; r < SFEM_THROUGHPUT_REPEAT; ++r) {
            for (size_t b = 0; b < mesh->n_blocks(); ++b) {
                auto ss_block = mesh->block(b);
                SFEM_TEST_ASSERT(sshex8_stencil_element_matrix_apply_hyteg(level,
                                                                           ss_block->n_elements(),
                                                                           ss_block->elements()->data(),
                                                                           block_matrices[b]->data(),
                                                                           block_stencils[b]->data(),
                                                                           input->data(),
                                                                           bench_out->data()) == SFEM_SUCCESS);
            }
        }
        hyteg_elapsed = (smesh::time_seconds() - tick) / SFEM_THROUGHPUT_REPEAT;

        zero_values(n_nodes, bench_out->data());
        tick = smesh::time_seconds();
        for (int r = 0; r < SFEM_THROUGHPUT_REPEAT; ++r) {
            for (size_t b = 0; b < mesh->n_blocks(); ++b) {
                auto ss_block = mesh->block(b);
                SFEM_TEST_ASSERT(sshex8_stencil_element_matrix_apply_hyteg_vectorized(level,
                                                                                      ss_block->n_elements(),
                                                                                      ss_block->elements()->data(),
                                                                                      block_matrices[b]->data(),
                                                                                      block_stencils[b]->data(),
                                                                                      input->data(),
                                                                                      bench_out->data()) == SFEM_SUCCESS);
            }
        }
        vectorized_elapsed = (smesh::time_seconds() - tick) / SFEM_THROUGHPUT_REPEAT;

        printf("SSHEX8 direct affine throughput level=%d base=%d macro_elements=%ld global_nodes=%ld local_dofs=%ld "
               "microhexes=%ld repeat=%d elapsed=%g MDOF/s=%g Mmicrohex/s=%g\n",
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

        printf("SSHEX8 current element-matrix stencil throughput level=%d base=%d macro_elements=%ld global_nodes=%ld "
               "local_dofs=%ld microhexes=%ld repeat=%d elapsed=%g MDOF/s=%g Mmicrohex/s=%g speedup_vs_direct=%g\n",
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

        printf("SSHEX8 HyTeG prebuilt stencil throughput level=%d base=%d macro_elements=%ld global_nodes=%ld "
               "local_dofs=%ld microhexes=%ld repeat=%d elapsed=%g MDOF/s=%g Mmicrohex/s=%g speedup_vs_current=%g "
               "speedup_vs_direct=%g\n",
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

        printf("SSHEX8 HyTeG vectorized fused stencil throughput level=%d base=%d macro_elements=%ld global_nodes=%ld "
               "local_dofs=%ld microhexes=%ld repeat=%d elapsed=%g MDOF/s=%g Mmicrohex/s=%g speedup_vs_hyteg=%g "
               "speedup_vs_current=%g speedup_vs_direct=%g\n",
               level,
               SFEM_BASE_RESOLUTION,
               total_macro_elements,
               n_nodes,
               total_local_dofs,
               total_microhexes,
               SFEM_THROUGHPUT_REPEAT,
               vectorized_elapsed,
               1e-6 * total_local_dofs / vectorized_elapsed,
               1e-6 * total_microhexes / vectorized_elapsed,
               hyteg_elapsed / vectorized_elapsed,
               current_elapsed / vectorized_elapsed,
               direct_elapsed / vectorized_elapsed);
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_sshex8_hyteg_stencil);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
