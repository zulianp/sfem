#include <math.h>
#include <stdlib.h>

#include <vector>

#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "smesh_sstet4.hpp"
#include "sstet4_linear_elasticity.hpp"

static int test_sstet4_linear_elasticity_stencil() {
    auto comm = sfem::Communicator::world();

    int SFEM_BASE_RESOLUTION = 2;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    int SFEM_ELEMENT_REFINE_LEVEL = 4;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int SFEM_VALIDATE = 1;
    SFEM_READ_ENV(SFEM_VALIDATE, atoi);

    int SFEM_THROUGHPUT_REPEAT = 10;
    SFEM_READ_ENV(SFEM_THROUGHPUT_REPEAT, atoi);

    const real_t mu     = 2.25;
    const real_t lambda = 1.75;

    auto mesh = sfem::Mesh::create_tet4_cube(
            comm, SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, 0, 0, 0, 1, 1, 1);
    mesh = smesh::to_semistructured(SFEM_ELEMENT_REFINE_LEVEL, mesh, true, false);

    const int       level   = smesh::semistructured_level(*mesh);
    const int       nxe     = smesh::sstet4_nxe(level);
    const ptrdiff_t n_nodes = mesh->n_nodes();
    const auto      es      = sfem::EXECUTION_SPACE_HOST;

    auto input      = sfem::create_buffer<real_t>(n_nodes * 3, es);
    auto point_out  = sfem::create_buffer<real_t>(n_nodes * 3, es);
    auto local_out  = sfem::create_buffer<real_t>(n_nodes * 3, es);
    auto global_out = sfem::create_buffer<real_t>(n_nodes * 3, es);

    auto points = mesh->points()->data();
    for (ptrdiff_t i = 0; i < n_nodes; ++i) {
        const real_t x = points[0][i];
        const real_t y = points[1][i];
        const real_t z = points[2][i];

        input->data()[i * 3 + 0] = 0.11 + x + 0.5 * y * z + 0.0078125 * ((i * 13 + 5) % 17);
        input->data()[i * 3 + 1] = -0.07 + y * y + 0.25 * z + 0.00390625 * ((i * 7 + 3) % 19);
        input->data()[i * 3 + 2] = 0.19 + z * x - 0.125 * y + 0.001953125 * ((i * 11 + 1) % 23);

        point_out->data()[i * 3 + 0]  = 0;
        point_out->data()[i * 3 + 1]  = 0;
        point_out->data()[i * 3 + 2]  = 0;
        local_out->data()[i * 3 + 0]  = 0;
        local_out->data()[i * 3 + 1]  = 0;
        local_out->data()[i * 3 + 2]  = 0;
        global_out->data()[i * 3 + 0] = 0;
        global_out->data()[i * 3 + 1] = 0;
        global_out->data()[i * 3 + 2] = 0;
    }

    std::vector<ptrdiff_t>                                      block_ne;
    std::vector<sfem::SharedBuffer<real_t>>                     block_u;
    std::vector<sfem::SharedBuffer<real_t>>                     block_stencil_out;
    std::vector<sstet4_linear_elasticity_stencil_t *>           block_stencils;

    ptrdiff_t total_macro_elements = 0;
    ptrdiff_t total_local_dofs     = 0;
    ptrdiff_t total_microtets      = 0;
    ptrdiff_t total_unique_stencils = 0;

    if (SFEM_VALIDATE) {
        for (size_t b = 0; b < mesh->n_blocks(); ++b) {
            auto block = mesh->block(b);
            SFEM_TEST_ASSERT(sstet4_linear_elasticity_apply_points(level,
                                                                   block->n_elements(),
                                                                   block->elements()->data(),
                                                                   points,
                                                                   mu,
                                                                   lambda,
                                                                   input->data(),
                                                                   point_out->data()) == SFEM_SUCCESS);
        }
    }

    for (size_t b = 0; b < mesh->n_blocks(); ++b) {
        auto block = mesh->block(b);
        const ptrdiff_t ne = block->n_elements();
        auto elements = block->elements()->data();

        SFEM_TEST_ASSERT(block->n_nodes_per_element() == nxe);

        auto local_u = sfem::create_buffer<real_t>(ne * nxe * 3, es);
        auto stencil_out = sfem::create_buffer<real_t>(ne * nxe * 3, es);

        for (ptrdiff_t e = 0; e < ne; ++e) {
            for (int v = 0; v < nxe; ++v) {
                const idx_t node = elements[v][e];
                local_u->data()[(e * nxe + v) * 3 + 0] = input->data()[node * 3 + 0];
                local_u->data()[(e * nxe + v) * 3 + 1] = input->data()[node * 3 + 1];
                local_u->data()[(e * nxe + v) * 3 + 2] = input->data()[node * 3 + 2];
                stencil_out->data()[(e * nxe + v) * 3 + 0] = 0;
                stencil_out->data()[(e * nxe + v) * 3 + 1] = 0;
                stencil_out->data()[(e * nxe + v) * 3 + 2] = 0;
            }
        }

        sstet4_linear_elasticity_stencil_t *stencil = nullptr;
        SFEM_TEST_ASSERT(sstet4_linear_elasticity_stencil_create_from_points(
                                 level, ne, elements, points, mu, lambda, &stencil) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(stencil != nullptr);
        total_unique_stencils += sstet4_linear_elasticity_stencil_n_unique_stencils(stencil);

        if (block_stencils.empty()) {
            printf("SSTET4 linear elasticity stencil topology level=%d rows=%d max_row_len=%d\n",
                   level,
                   sstet4_linear_elasticity_stencil_nrows(stencil),
                   sstet4_linear_elasticity_stencil_max_row_len(stencil));
        }

        SFEM_TEST_ASSERT(sstet4_linear_elasticity_apply_stencil(
                                 stencil, ne, local_u->data(), stencil_out->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(sstet4_linear_elasticity_apply_stencil_global_vectorized(
                                 stencil, ne, elements, input->data(), global_out->data()) == SFEM_SUCCESS);

        if (SFEM_VALIDATE) {
            for (ptrdiff_t e = 0; e < ne; ++e) {
                for (int v = 0; v < nxe; ++v) {
                    const idx_t node = elements[v][e];
                    local_out->data()[node * 3 + 0] += stencil_out->data()[(e * nxe + v) * 3 + 0];
                    local_out->data()[node * 3 + 1] += stencil_out->data()[(e * nxe + v) * 3 + 1];
                    local_out->data()[node * 3 + 2] += stencil_out->data()[(e * nxe + v) * 3 + 2];
                }
            }
        }

        block_ne.push_back(ne);
        block_u.push_back(local_u);
        block_stencil_out.push_back(stencil_out);
        block_stencils.push_back(stencil);

        total_macro_elements += ne;
        total_local_dofs += ne * nxe * 3;
        total_microtets += ne * level * level * level;
    }

    if (SFEM_VALIDATE) {
        real_t local_largest_diff = 0;
        real_t global_largest_diff = 0;
        ptrdiff_t local_arg = SFEM_PTRDIFF_INVALID;
        ptrdiff_t global_arg = SFEM_PTRDIFF_INVALID;

        for (ptrdiff_t i = 0; i < n_nodes * 3; ++i) {
            const real_t local_diff = fabs(local_out->data()[i] - point_out->data()[i]);
            if (local_diff > local_largest_diff || local_diff != local_diff) {
                local_largest_diff = local_diff;
                local_arg          = i;
            }

            const real_t global_diff = fabs(global_out->data()[i] - point_out->data()[i]);
            if (global_diff > global_largest_diff || global_diff != global_diff) {
                global_largest_diff = global_diff;
                global_arg          = i;
            }
        }

        printf("SSTET4 linear elasticity local stencil check level=%d largest_diff(%ld) = %g\n",
               level,
               local_arg,
               (double)local_largest_diff);
        printf("SSTET4 linear elasticity global vectorized stencil check level=%d largest_diff(%ld) = %g\n",
               level,
               global_arg,
               (double)global_largest_diff);
        SFEM_TEST_ASSERT(local_largest_diff < 1e-7);
        SFEM_TEST_ASSERT(global_largest_diff < 1e-7);
        printf("SSTET4 linear elasticity stencil variants level=%d total_unique_stencils=%ld\n",
               level,
               total_unique_stencils);
    }

    if (SFEM_THROUGHPUT_REPEAT > 0) {
        const double local_tick = smesh::time_seconds();
        for (int r = 0; r < SFEM_THROUGHPUT_REPEAT; ++r) {
            for (size_t b = 0; b < block_ne.size(); ++b) {
                SFEM_TEST_ASSERT(sstet4_linear_elasticity_apply_stencil(
                                         block_stencils[b],
                                         block_ne[b],
                                         block_u[b]->data(),
                                         block_stencil_out[b]->data()) == SFEM_SUCCESS);
            }
        }
        const double local_elapsed = (smesh::time_seconds() - local_tick) / SFEM_THROUGHPUT_REPEAT;

        printf("SSTET4 linear elasticity local stencil throughput level=%d base=%d macro_elements=%ld global_nodes=%ld "
               "local_dofs=%ld microtets=%ld repeat=%d elapsed=%g MDOF/s=%g Mmicrotet/s=%g\n",
               level,
               SFEM_BASE_RESOLUTION,
               total_macro_elements,
               n_nodes,
               total_local_dofs,
               total_microtets,
               SFEM_THROUGHPUT_REPEAT,
               local_elapsed,
               1e-6 * total_local_dofs / local_elapsed,
               1e-6 * total_microtets / local_elapsed);

        const double global_tick = smesh::time_seconds();
        for (int r = 0; r < SFEM_THROUGHPUT_REPEAT; ++r) {
            for (size_t b = 0; b < block_ne.size(); ++b) {
                auto block = mesh->block(b);
                SFEM_TEST_ASSERT(sstet4_linear_elasticity_apply_stencil_global_vectorized(
                                         block_stencils[b],
                                         block_ne[b],
                                         block->elements()->data(),
                                         input->data(),
                                         global_out->data()) == SFEM_SUCCESS);
            }
        }
        const double global_elapsed = (smesh::time_seconds() - global_tick) / SFEM_THROUGHPUT_REPEAT;

        printf("SSTET4 linear elasticity global vectorized stencil throughput level=%d base=%d macro_elements=%ld "
               "global_nodes=%ld local_dofs=%ld microtets=%ld repeat=%d elapsed=%g MDOF/s=%g Mmicrotet/s=%g "
               "speedup_vs_global=%g\n",
               level,
               SFEM_BASE_RESOLUTION,
               total_macro_elements,
               n_nodes,
               total_local_dofs,
               total_microtets,
               SFEM_THROUGHPUT_REPEAT,
               global_elapsed,
               1e-6 * total_local_dofs / global_elapsed,
               1e-6 * total_microtets / global_elapsed,
               local_elapsed / global_elapsed);
    }

    for (size_t b = 0; b < block_stencils.size(); ++b) {
        sstet4_linear_elasticity_stencil_destroy(block_stencils[b]);
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_sstet4_linear_elasticity_stencil);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
