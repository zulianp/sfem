#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_BSR.hpp"
#include "sfem_CRS.hpp"
#include "sfem_DIA.hpp"
#include "sfem_Function.hpp"
#include "sfem_GeneratedLaplace.hpp"
#include "sfem_GeneratedLaplace_c_abi.hpp"
#include "sfem_GeneratedLinearElasticity.hpp"
#include "sfem_GeneratedLinearElasticity_c_abi.hpp"
#include "sfem_GeneratedNeoHookeanOgden.hpp"
#include "sfem_GeneratedNeoHookeanOgden_c_abi.hpp"
#include "sfem_Laplacian.hpp"
#include "sfem_OpFactory.hpp"
#include "smesh_mesh_reorder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace {
    constexpr int    BLOCK_SIZE = 3;

    void fill_state_and_direction(const std::shared_ptr<sfem::FunctionSpace> &space,
                                  std::vector<real_t>                        &state,
                                  std::vector<real_t>                        &direction) {
        geom_t **const  points = space->mesh_ptr()->points()->data();
        const ptrdiff_t nnodes = space->mesh_ptr()->n_nodes();

        for (ptrdiff_t node = 0; node < nnodes; ++node) {
            const real_t x = points[0][node];
            const real_t y = points[1][node];
            const real_t z = points[2][node];

            state[node * BLOCK_SIZE + 0] = 1e-3 * (1 + x + 0.25 * y);
            state[node * BLOCK_SIZE + 1] = 1e-3 * (2 + y + 0.50 * z);
            state[node * BLOCK_SIZE + 2] = 1e-3 * (3 + z + 0.75 * x);

            direction[node * BLOCK_SIZE + 0] = 2e-4 * (1 + 0.5 * x + z);
            direction[node * BLOCK_SIZE + 1] = -3e-4 * (1 + y + 0.25 * x);
            direction[node * BLOCK_SIZE + 2] = 4e-4 * (1 + z + 0.5 * y);
        }
    }

    int generated_hessian_action(sfem::Function            &function,
                                 const std::vector<real_t> &state,
                                 const real_t *const        direction,
                                 real_t *const              output) {
        std::fill(output, output + state.size(), static_cast<real_t>(0));
        return function.apply(state.data(), direction, output);
    }

    std::vector<int> full_node_diagonal_offsets(const ptrdiff_t nnodes) {
        std::vector<int> offsets;
        offsets.reserve(2 * nnodes - 1);
        for (ptrdiff_t offset = 1 - nnodes; offset < nnodes; ++offset) {
            offsets.push_back(static_cast<int>(offset));
        }
        return offsets;
    }

    void apply_dia_blocks(const std::vector<int> &diag_offsets,
                          const real_t *const     values,
                          const real_t *const     x,
                          real_t *const           y,
                          const ptrdiff_t         nnodes) {
        std::fill(y, y + nnodes * BLOCK_SIZE, static_cast<real_t>(0));

        for (ptrdiff_t diagonal = 0; diagonal < static_cast<ptrdiff_t>(diag_offsets.size()); ++diagonal) {
            const ptrdiff_t offset = diag_offsets[diagonal];
            for (ptrdiff_t node_i = 0; node_i < nnodes; ++node_i) {
                const ptrdiff_t node_j = node_i + offset;
                if (node_j < 0 || node_j >= nnodes) {
                    continue;
                }

                const real_t *const block = &values[(diagonal * nnodes + node_i) * BLOCK_SIZE * BLOCK_SIZE];
                for (int bi = 0; bi < BLOCK_SIZE; ++bi) {
                    real_t acc = 0;
                    for (int bj = 0; bj < BLOCK_SIZE; ++bj) {
                        acc += block[bi * BLOCK_SIZE + bj] * x[node_j * BLOCK_SIZE + bj];
                    }
                    y[node_i * BLOCK_SIZE + bi] += acc;
                }
            }
        }
    }

    void apply_scalar_dia(const std::vector<int> &diag_offsets,
                          const real_t *const     values,
                          const real_t *const     x,
                          real_t *const           y,
                          const ptrdiff_t         nnodes) {
        std::fill(y, y + nnodes, static_cast<real_t>(0));

        for (ptrdiff_t diagonal = 0; diagonal < static_cast<ptrdiff_t>(diag_offsets.size()); ++diagonal) {
            const ptrdiff_t offset = diag_offsets[diagonal];
            for (ptrdiff_t node_i = 0; node_i < nnodes; ++node_i) {
                const ptrdiff_t node_j = node_i + offset;
                if (node_j < 0 || node_j >= nnodes) {
                    continue;
                }

                y[node_i] += values[diagonal * nnodes + node_i] * x[node_j];
            }
        }
    }

    void build_coo_graph(const count_t *const rowptr,
                         const idx_t *const   colidx,
                         const ptrdiff_t      nnodes,
                         std::vector<idx_t>  &rows,
                         std::vector<idx_t>  &cols) {
        rows.clear();
        cols.clear();
        rows.reserve(rowptr[nnodes]);
        cols.reserve(rowptr[nnodes]);

        for (ptrdiff_t node_i = 0; node_i < nnodes; ++node_i) {
            for (count_t k = rowptr[node_i]; k < rowptr[node_i + 1]; ++k) {
                rows.push_back(static_cast<idx_t>(node_i));
                cols.push_back(colidx[k]);
            }
        }
    }

    void apply_coo_blocks(const std::vector<idx_t> &rows,
                          const std::vector<idx_t> &cols,
                          const real_t *const       values,
                          const real_t *const       x,
                          real_t *const             y,
                          const ptrdiff_t           nnodes) {
        std::fill(y, y + nnodes * BLOCK_SIZE, static_cast<real_t>(0));

        for (ptrdiff_t entry = 0; entry < static_cast<ptrdiff_t>(rows.size()); ++entry) {
            const idx_t         node_i = rows[entry];
            const idx_t         node_j = cols[entry];
            const real_t *const block  = &values[entry * BLOCK_SIZE * BLOCK_SIZE];

            for (int bi = 0; bi < BLOCK_SIZE; ++bi) {
                real_t acc = 0;
                for (int bj = 0; bj < BLOCK_SIZE; ++bj) {
                    acc += block[bi * BLOCK_SIZE + bj] * x[node_j * BLOCK_SIZE + bj];
                }
                y[node_i * BLOCK_SIZE + bi] += acc;
            }
        }
    }

    void apply_coo_triplets(const std::vector<idx_t> &rows,
                            const std::vector<idx_t> &cols,
                            const real_t *const       values,
                            const real_t *const       x,
                            real_t *const             y,
                            const ptrdiff_t           ndofs) {
        std::fill(y, y + ndofs, static_cast<real_t>(0));

        for (ptrdiff_t entry = 0; entry < static_cast<ptrdiff_t>(rows.size()); ++entry) {
            y[rows[entry]] += values[entry] * x[cols[entry]];
        }
    }

    int assert_equal_indices(const char *const         name,
                             const std::vector<idx_t> &expected,
                             const std::vector<idx_t> &actual) {
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(expected.size()); ++i) {
            if (actual[i] != expected[i]) {
                std::fprintf(stderr,
                             "%s index mismatch at %td: actual=%ld expected=%ld\n",
                             name,
                             i,
                             static_cast<long>(actual[i]),
                             static_cast<long>(expected[i]));
                return SFEM_TEST_FAILURE;
            }
        }

        return SFEM_SUCCESS;
    }

    int assert_close_values(const char *const          name,
                            const std::vector<real_t> &expected,
                            const std::vector<real_t> &actual,
                            const real_t               atol,
                            const real_t               rtol) {
        real_t    max_abs = 0;
        real_t    max_ref = 0;
        ptrdiff_t argmax  = -1;

        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(expected.size()); ++i) {
            const real_t diff = std::abs(actual[i] - expected[i]);
            const real_t ref  = std::abs(expected[i]);
            if (diff > max_abs) {
                max_abs = diff;
                argmax  = i;
            }
            max_ref = std::max(max_ref, ref);
        }

        const real_t tol = atol + rtol * max_ref;
        if (!(max_abs <= tol)) {
            std::fprintf(stderr,
                         "%s values mismatch at %td: max_abs=%g tol=%g ref=%g actual=%g expected=%g\n",
                         name,
                         argmax,
                         static_cast<double>(max_abs),
                         static_cast<double>(tol),
                         static_cast<double>(max_ref),
                         static_cast<double>(actual[argmax]),
                         static_cast<double>(expected[argmax]));
            return SFEM_TEST_FAILURE;
        }

        return SFEM_SUCCESS;
    }

    int assert_close_crs_coo_blocks(const char *const         name,
                                    const count_t *const      rowptr,
                                    const idx_t *const        colidx,
                                    const ptrdiff_t           nnodes,
                                    const std::vector<real_t> &crs_values,
                                    const std::vector<real_t> &coo_values,
                                    const real_t              atol,
                                    const real_t              rtol) {
        real_t    max_abs = 0;
        real_t    max_ref = 0;
        ptrdiff_t argmax  = -1;

        for (ptrdiff_t node_i = 0; node_i < nnodes; ++node_i) {
            const count_t  row_begin = rowptr[node_i];
            const count_t  row_end   = rowptr[node_i + 1];
            const ptrdiff_t lenrow    = row_end - row_begin;

            for (count_t k = row_begin; k < row_end; ++k) {
                const ptrdiff_t local_col = k - row_begin;
                for (int bi = 0; bi < BLOCK_SIZE; ++bi) {
                    for (int bj = 0; bj < BLOCK_SIZE; ++bj) {
                        const ptrdiff_t crs_index =
                                row_begin * BLOCK_SIZE * BLOCK_SIZE + bi * lenrow * BLOCK_SIZE + local_col * BLOCK_SIZE + bj;
                        const ptrdiff_t coo_index = k * BLOCK_SIZE * BLOCK_SIZE + bi * BLOCK_SIZE + bj;
                        const real_t    diff      = std::abs(coo_values[coo_index] - crs_values[crs_index]);
                        const real_t    ref       = std::abs(crs_values[crs_index]);
                        if (diff > max_abs) {
                            max_abs = diff;
                            argmax  = coo_index;
                        }
                        max_ref = std::max(max_ref, ref);
                    }
                }
            }
        }

        const real_t tol = atol + rtol * max_ref;
        if (!(max_abs <= tol)) {
            std::fprintf(stderr,
                         "%s CRS/COO values mismatch at COO block scalar %td: max_abs=%g tol=%g ref=%g\n",
                         name,
                         argmax,
                         static_cast<double>(max_abs),
                         static_cast<double>(tol),
                         static_cast<double>(max_ref));
            return SFEM_TEST_FAILURE;
        }

        return SFEM_SUCCESS;
    }

    int assert_close_action(const char *const          name,
                            const std::vector<real_t> &expected,
                            const std::vector<real_t> &actual,
                            const real_t               atol,
                            const real_t               rtol) {
        real_t    max_abs = 0;
        real_t    max_ref = 0;
        ptrdiff_t argmax  = -1;

        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(expected.size()); ++i) {
            const real_t diff = std::abs(actual[i] - expected[i]);
            const real_t ref  = std::abs(expected[i]);
            if (diff > max_abs) {
                max_abs = diff;
                argmax  = i;
            }
            max_ref = std::max(max_ref, ref);
        }

        const real_t tol = atol + rtol * max_ref;
        if (!(max_abs <= tol)) {
            std::fprintf(stderr,
                         "%s hessian action mismatch at %td: max_abs=%g tol=%g ref=%g actual=%g expected=%g\n",
                         name,
                         argmax,
                         static_cast<double>(max_abs),
                         static_cast<double>(tol),
                         static_cast<double>(max_ref),
                         static_cast<double>(actual[argmax]),
                         static_cast<double>(expected[argmax]));
            return SFEM_TEST_FAILURE;
        }

        return SFEM_SUCCESS;
    }

    int build_single_pack_elements(idx_t **const                       elements,
                                   const ptrdiff_t                     nelements,
                                   const int                           n_nodes_per_element,
                                   std::vector<std::vector<uint16_t>> &storage,
                                   std::vector<uint16_t *>            &packed_elements) {
        storage.assign(n_nodes_per_element, std::vector<uint16_t>(nelements, 0));
        packed_elements.resize(n_nodes_per_element);

        for (int shape = 0; shape < n_nodes_per_element; ++shape) {
            packed_elements[shape] = storage[shape].data();
            for (ptrdiff_t element = 0; element < nelements; ++element) {
                const idx_t node = elements[shape][element];
                if (node < 0 || node > static_cast<idx_t>(UINT16_MAX)) {
                    std::fprintf(stderr,
                                 "single-pack packed test node id %ld is outside uint16_t range\n",
                                 static_cast<long>(node));
                    return SFEM_TEST_FAILURE;
                }
                storage[shape][element] = static_cast<uint16_t>(node);
            }
        }

        return SFEM_SUCCESS;
    }
}  // namespace

int test_generated_neohookean_hessian_action_matrix_formats() {
    auto mesh     = sfem::Mesh::create_cube(sfem::Communicator::self(), smesh::HEX8, 4, 4, 4, 0, 0, 0, 1, 1, 1);
    auto space    = sfem::FunctionSpace::create(mesh, BLOCK_SIZE);
    auto function = sfem::Function::create(space);
    auto op       = sfem::create_op(space, "GeneratedNeoHookeanOgden", sfem::EXECUTION_SPACE_HOST);

    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_ASSERT(std::strcmp(op->name(), "GeneratedNeoHookeanOgden") == 0);
    auto *const generated_op = dynamic_cast<sfem::GeneratedNeoHookeanOgden *>(op.get());
    SFEM_TEST_ASSERT(generated_op != nullptr);
    generated_op->set_value_in_block("default", "mu", 1.0);
    generated_op->set_value_in_block("default", "lmbda", 1.0);
    function->add_operator(op);

    constexpr smesh::block_idx_t block_id            = 0;
    const ptrdiff_t              nnodes              = mesh->n_nodes();
    const ptrdiff_t              ndofs               = space->n_dofs();
    const int                    n_nodes_per_element = mesh->n_nodes_per_element(block_id);
    const ptrdiff_t              nelements           = mesh->n_elements(block_id);
    const geom_t *const *const   points              = const_cast<const geom_t *const *>(mesh->points()->data());
    SFEM_TEST_ASSERT(n_nodes_per_element == 8);

    std::vector<real_t> state(ndofs, 0);
    std::vector<real_t> direction(ndofs, 0);
    fill_state_and_direction(space, state, direction);

    std::vector<std::vector<uint16_t>> packed_element_storage;
    std::vector<uint16_t *>            packed_elements;
    SFEM_TEST_ASSERT(build_single_pack_elements(mesh->elements(block_id)->data(),
                                                nelements,
                                                n_nodes_per_element,
                                                packed_element_storage,
                                                packed_elements) == SFEM_SUCCESS);
    const ptrdiff_t owned_nodes_ptr[2] = {0, nnodes};
    const ptrdiff_t n_shared_nodes[1]  = {0};
    const ptrdiff_t ghost_ptr[2]       = {0, 0};
    const idx_t     ghost_idx[1]       = {0};

    std::vector<real_t> expected_apply(ndofs, 0);
    std::vector<real_t> packed_apply(ndofs, 0);
    SFEM_TEST_ASSERT(generated_hessian_action(*function, state, direction.data(), expected_apply.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(neohookean_ogden_apply_packed_3d_isoparametric_mesh_soa(smesh::HEX8,
                                                                                    1,
                                                                                    nelements,
                                                                                    nelements,
                                                                                    nnodes,
                                                                                    nnodes,
                                                                                    packed_elements.data(),
                                                                                    owned_nodes_ptr,
                                                                                    n_shared_nodes,
                                                                                    ghost_ptr,
                                                                                    ghost_idx,
                                                                                    points,
                                                                                    1.0,
                                                                                    1.0,
                                                                                    BLOCK_SIZE,
                                                                                    state.data() + 0,
                                                                                    state.data() + 1,
                                                                                    state.data() + 2,
                                                                                    BLOCK_SIZE,
                                                                                    direction.data() + 0,
                                                                                    direction.data() + 1,
                                                                                    direction.data() + 2,
                                                                                    BLOCK_SIZE,
                                                                                    packed_apply.data() + 0,
                                                                                    packed_apply.data() + 1,
                                                                                    packed_apply.data() + 2) ==
                     SFEM_SUCCESS);
    SFEM_TEST_ASSERT(assert_close_action("generated NeoHookean packed hessian action",
                                         expected_apply,
                                         packed_apply,
                                         1e-12,
                                         1e-10) == SFEM_SUCCESS);

    std::vector<real_t> gradient_reference(ndofs, 0);
    std::vector<real_t> gradient_packed(ndofs, 0);
    SFEM_TEST_ASSERT(function->gradient(state.data(), gradient_reference.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(neohookean_ogden_gradient_packed_3d_isoparametric_mesh_soa(smesh::HEX8,
                                                                                       1,
                                                                                       nelements,
                                                                                       nelements,
                                                                                       nnodes,
                                                                                       nnodes,
                                                                                       packed_elements.data(),
                                                                                       owned_nodes_ptr,
                                                                                       n_shared_nodes,
                                                                                       ghost_ptr,
                                                                                       ghost_idx,
                                                                                       points,
                                                                                       1.0,
                                                                                       1.0,
                                                                                       BLOCK_SIZE,
                                                                                       state.data() + 0,
                                                                                       state.data() + 1,
                                                                                       state.data() + 2,
                                                                                       BLOCK_SIZE,
                                                                                       gradient_packed.data() + 0,
                                                                                       gradient_packed.data() + 1,
                                                                                       gradient_packed.data() + 2) ==
                     SFEM_SUCCESS);
    SFEM_TEST_ASSERT(assert_close_action("generated NeoHookean packed gradient",
                                         gradient_reference,
                                         gradient_packed,
                                         1e-12,
                                         1e-10) == SFEM_SUCCESS);

    constexpr int       n_value_steps       = 4;
    const real_t        steps[n_value_steps] = {-1.0, -0.5, 0.25, 1.0};
    std::vector<real_t> value_steps_reference(n_value_steps, 0);
    std::vector<real_t> value_steps_packed(n_value_steps, 0);
    std::vector<real_t> packed_step_element_values(n_value_steps * nelements, 0);
    SFEM_TEST_ASSERT(function->value_steps(state.data(),
                                           direction.data(),
                                           n_value_steps,
                                           steps,
                                           value_steps_reference.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(neohookean_ogden_objective_steps_packed_3d_isoparametric_mesh_soa(smesh::HEX8,
                                                                                              1,
                                                                                              nelements,
                                                                                              nelements,
                                                                                              nnodes,
                                                                                              nnodes,
                                                                                              packed_elements.data(),
                                                                                              owned_nodes_ptr,
                                                                                              n_shared_nodes,
                                                                                              ghost_ptr,
                                                                                              ghost_idx,
                                                                                              points,
                                                                                              1.0,
                                                                                              1.0,
                                                                                              BLOCK_SIZE,
                                                                                              state.data() + 0,
                                                                                              state.data() + 1,
                                                                                              state.data() + 2,
                                                                                              BLOCK_SIZE,
                                                                                              direction.data() + 0,
                                                                                              direction.data() + 1,
                                                                                              direction.data() + 2,
                                                                                              n_value_steps,
                                                                                              steps,
                                                                                              packed_step_element_values.data()) ==
                     SFEM_SUCCESS);
    for (int step = 0; step < n_value_steps; ++step) {
        real_t sum = 0;
#pragma omp simd reduction(+ : sum)
        for (ptrdiff_t element = 0; element < nelements; ++element) {
            sum += packed_step_element_values[step * nelements + element];
        }
        value_steps_packed[step] = sum;
    }
    SFEM_TEST_ASSERT(assert_close_values("generated NeoHookean packed value_steps",
                                         value_steps_reference,
                                         value_steps_packed,
                                         1e-12,
                                         1e-10) == SFEM_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int test_generated_linear_elasticity_packed_gradient_value_steps() {
    auto mesh     = sfem::Mesh::create_cube(sfem::Communicator::self(), smesh::HEX8, 4, 4, 4, 0, 0, 0, 1, 1, 1);
    auto space    = sfem::FunctionSpace::create(mesh, BLOCK_SIZE);
    auto function = sfem::Function::create(space);
    auto op       = sfem::create_op(space, "GeneratedLinearElasticity", sfem::EXECUTION_SPACE_HOST);

    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_ASSERT(std::strcmp(op->name(), "GeneratedLinearElasticity") == 0);
    auto *const generated_op = dynamic_cast<sfem::GeneratedLinearElasticity *>(op.get());
    SFEM_TEST_ASSERT(generated_op != nullptr);
    function->add_operator(op);

    constexpr real_t mu    = 0.75;
    constexpr real_t lmbda = 1.35;
    generated_op->set_value_in_block("default", "mu", mu);
    generated_op->set_value_in_block("default", "lmbda", lmbda);

    constexpr smesh::block_idx_t block_id            = 0;
    const ptrdiff_t              nnodes              = mesh->n_nodes();
    const ptrdiff_t              ndofs               = space->n_dofs();
    const int                    n_nodes_per_element = mesh->n_nodes_per_element(block_id);
    const ptrdiff_t              nelements           = mesh->n_elements(block_id);
    SFEM_TEST_ASSERT(n_nodes_per_element == 8);

    std::vector<real_t> state(ndofs, 0);
    std::vector<real_t> direction(ndofs, 0);
    fill_state_and_direction(space, state, direction);

    std::vector<std::vector<uint16_t>> packed_element_storage;
    std::vector<uint16_t *>            packed_elements;
    SFEM_TEST_ASSERT(build_single_pack_elements(mesh->elements(block_id)->data(),
                                                nelements,
                                                n_nodes_per_element,
                                                packed_element_storage,
                                                packed_elements) == SFEM_SUCCESS);
    const ptrdiff_t owned_nodes_ptr[2] = {0, nnodes};
    const ptrdiff_t n_shared_nodes[1]  = {0};
    const ptrdiff_t ghost_ptr[2]       = {0, 0};
    const idx_t     ghost_idx[1]       = {0};
    const geom_t *const *const points = const_cast<const geom_t *const *>(mesh->points()->data());

    std::vector<real_t> gradient_reference(ndofs, 0);
    std::vector<real_t> gradient_packed(ndofs, 0);
    SFEM_TEST_ASSERT(function->gradient(state.data(), gradient_reference.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(linear_elasticity_gradient_packed_3d_isoparametric_mesh_soa(smesh::HEX8,
                                                                                         1,
                                                                                         nelements,
                                                                                         nelements,
                                                                                         nnodes,
                                                                                         nnodes,
                                                                                         packed_elements.data(),
                                                                                         owned_nodes_ptr,
                                                                                         n_shared_nodes,
                                                                                         ghost_ptr,
                                                                                         ghost_idx,
                                                                                         points,
                                                                                         lmbda,
                                                                                         mu,
                                                                                         BLOCK_SIZE,
                                                                                         state.data() + 0,
                                                                                         state.data() + 1,
                                                                                         state.data() + 2,
                                                                                         BLOCK_SIZE,
                                                                                         gradient_packed.data() + 0,
                                                                                         gradient_packed.data() + 1,
                                                                                         gradient_packed.data() + 2) ==
                     SFEM_SUCCESS);
    SFEM_TEST_ASSERT(assert_close_action("generated LinearElasticity packed gradient",
                                         gradient_reference,
                                         gradient_packed,
                                         1e-12,
                                         1e-10) == SFEM_SUCCESS);

    constexpr int n_value_steps = 3;
    const real_t  steps[n_value_steps] = {-0.25, 0.5, 1.25};
    std::vector<real_t> value_steps_reference(n_value_steps, 0);
    std::vector<real_t> value_steps_packed(n_value_steps, 0);
    std::vector<real_t> packed_step_element_values(n_value_steps * nelements, 0);
    SFEM_TEST_ASSERT(function->value_steps(state.data(),
                                           direction.data(),
                                           n_value_steps,
                                           steps,
                                           value_steps_reference.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(linear_elasticity_objective_steps_packed_3d_isoparametric_mesh_soa(smesh::HEX8,
                                                                                               1,
                                                                                               nelements,
                                                                                               nelements,
                                                                                               nnodes,
                                                                                               nnodes,
                                                                                               packed_elements.data(),
                                                                                               owned_nodes_ptr,
                                                                                               n_shared_nodes,
                                                                                               ghost_ptr,
                                                                                               ghost_idx,
                                                                                               points,
                                                                                               lmbda,
                                                                                               mu,
                                                                                               BLOCK_SIZE,
                                                                                               state.data() + 0,
                                                                                               state.data() + 1,
                                                                                               state.data() + 2,
                                                                                               BLOCK_SIZE,
                                                                                               direction.data() + 0,
                                                                                               direction.data() + 1,
                                                                                               direction.data() + 2,
                                                                                               n_value_steps,
                                                                                               steps,
                                                                                               packed_step_element_values.data()) ==
                     SFEM_SUCCESS);
    for (int step = 0; step < n_value_steps; ++step) {
        real_t sum = 0;
#pragma omp simd reduction(+ : sum)
        for (ptrdiff_t element = 0; element < nelements; ++element) {
            sum += packed_step_element_values[step * nelements + element];
        }
        value_steps_packed[step] = sum;
    }
    SFEM_TEST_ASSERT(assert_close_values("generated LinearElasticity packed value_steps",
                                         value_steps_reference,
                                         value_steps_packed,
                                         1e-12,
                                         1e-10) == SFEM_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int test_generated_laplace_crs_bsr_matches_existing_laplacian() {
    auto mesh  = sfem::Mesh::create_cube(sfem::Communicator::self(), smesh::TET4, 2, 2, 2, 0, 0, 0, 1, 1, 1);
    auto sfc   = smesh::SFC::create_from_env();
    sfc->reorder(*mesh);
    auto space = sfem::FunctionSpace::create(mesh, 1);

    auto existing_function = sfem::Function::create(space);
    auto generated_function = sfem::Function::create(space);

    auto existing_op = sfem::create_op(space, "Laplacian", sfem::EXECUTION_SPACE_HOST);
    auto generated_op = sfem::create_op(space, "GeneratedLaplace", sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(existing_op != nullptr);
    SFEM_TEST_ASSERT(generated_op != nullptr);
    auto *const generated_laplace = dynamic_cast<sfem::GeneratedLaplace *>(generated_op.get());
    SFEM_TEST_ASSERT(generated_laplace != nullptr);
    SFEM_TEST_ASSERT(existing_op->initialize() == SFEM_SUCCESS);
    generated_op->set_option("ASSUME_AFFINE", true);
    SFEM_TEST_ASSERT(generated_op->initialize() == SFEM_SUCCESS);

    existing_function->add_operator(existing_op);
    generated_function->add_operator(generated_op);

    const ptrdiff_t ndofs = space->n_dofs();
    std::vector<real_t> direction(ndofs, 0);
    std::vector<real_t> state(ndofs, 0);
    std::vector<real_t> expected_gradient(ndofs, 0);
    std::vector<real_t> generated_gradient(ndofs, 0);
    std::vector<real_t> expected_action(ndofs, 0);
    std::vector<real_t> generated_action(ndofs, 0);
    std::vector<real_t> packed_action(ndofs, 0);
    geom_t **const mesh_points = mesh->points()->data();
    for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
        state[node]     = 0.25 * (1 + mesh_points[0][node] - mesh_points[1][node] + mesh_points[2][node]);
        direction[node] = 0.125 * (1 + mesh_points[0][node] + 2 * mesh_points[1][node] + 3 * mesh_points[2][node]);
    }

    SFEM_TEST_ASSERT(existing_function->gradient(state.data(), expected_gradient.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(generated_function->gradient(state.data(), generated_gradient.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(assert_close_action("generated Laplace gradient",
                                         expected_gradient,
                                         generated_gradient,
                                         1e-14,
                                         1e-12) == SFEM_SUCCESS);

    SFEM_TEST_ASSERT(generated_function->apply(nullptr, direction.data(), expected_action.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(existing_function->apply(nullptr, direction.data(), generated_action.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(assert_close_action("generated Laplace apply",
                                         expected_action,
                                         generated_action,
                                         1e-14,
                                         1e-12) == SFEM_SUCCESS);

    auto packed_mesh = sfem::FunctionSpace::PackedMesh::create(mesh, {}, true);
    auto packed_space = sfem::FunctionSpace::create(packed_mesh, 1);
    auto packed_generated_op = sfem::create_op(packed_space, "GeneratedLaplace", sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(packed_generated_op != nullptr);
    packed_generated_op->set_option("ASSUME_AFFINE", true);
    SFEM_TEST_ASSERT(packed_generated_op->initialize() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(packed_generated_op->apply(nullptr, direction.data(), packed_action.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(assert_close_action("generated packed Laplace apply",
                                         expected_action,
                                         packed_action,
                                         1e-14,
                                         1e-12) == SFEM_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int test_generated_laplace_hex8_dia_matches_apply() {
    auto mesh = sfem::Mesh::create_cube(sfem::Communicator::self(), smesh::HEX8, 2, 2, 2, 0, 0, 0, 1, 1, 1);
    auto sfc  = smesh::SFC::create_from_env();
    sfc->reorder(*mesh);
    auto space = sfem::FunctionSpace::create(mesh, 1);
    auto function = sfem::Function::create(space);
    auto generated_op = sfem::create_op(space, "GeneratedLaplace", sfem::EXECUTION_SPACE_HOST);

    SFEM_TEST_ASSERT(generated_op != nullptr);
    auto *const generated_laplace = dynamic_cast<sfem::GeneratedLaplace *>(generated_op.get());
    SFEM_TEST_ASSERT(generated_laplace != nullptr);
    SFEM_TEST_ASSERT(generated_op->initialize() == SFEM_SUCCESS);
    function->add_operator(generated_op);

    const ptrdiff_t nnodes = mesh->n_nodes();
    const ptrdiff_t ndofs = space->n_dofs();
    std::vector<real_t> direction(ndofs, 0);
    std::vector<real_t> expected_action(ndofs, 0);
    std::vector<real_t> packed_action(ndofs, 0);

    geom_t **const points = mesh->points()->data();
    for (ptrdiff_t node = 0; node < nnodes; ++node) {
        direction[node] = 0.0625 * (1 + 3 * points[0][node] - points[1][node] + 2 * points[2][node]);
    }

    SFEM_TEST_ASSERT(function->apply(nullptr, direction.data(), expected_action.data()) == SFEM_SUCCESS);

    auto packed_mesh = sfem::FunctionSpace::PackedMesh::create(mesh, {}, true);
    auto packed_space = sfem::FunctionSpace::create(packed_mesh, 1);
    auto packed_generated_op = sfem::create_op(packed_space, "GeneratedLaplace", sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(packed_generated_op != nullptr);
    packed_generated_op->set_option("ASSUME_AFFINE", true);
    SFEM_TEST_ASSERT(packed_generated_op->initialize() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(packed_generated_op->apply(nullptr, direction.data(), packed_action.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(assert_close_action("generated HEX8 packed Laplace apply",
                                         expected_action,
                                         packed_action,
                                         1e-14,
                                         1e-12) == SFEM_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int test_generated_linear_elasticity_packed_one_pass_matches_two_pass() {
    setenv("SMESH_ELEMENTS_PER_PACK", "64", 1);
    setenv("SFEM_PACKED_TWO_PASS", "0", 1);

    constexpr int N = 8;
    auto          mesh =
            sfem::Mesh::create_cube(sfem::Communicator::self(), smesh::HEX8, N, N, N, 0, 0, 0, 1, 1, 1);
    auto packed_mesh  = sfem::FunctionSpace::PackedMesh::create(mesh, {}, true, 64);
    auto packed_space = sfem::FunctionSpace::create(packed_mesh, BLOCK_SIZE);
    auto packed       = packed_space->packed_mesh();
    SFEM_TEST_ASSERT(packed != nullptr);
    SFEM_TEST_ASSERT(packed->n_packs(0) > 1);
    SFEM_TEST_ASSERT(packed->n_ghost_entries(0) > 0);
    SFEM_TEST_ASSERT(packed->n_ghost_reduce_rows(0) > 0);

    constexpr real_t mu    = 0.75;
    constexpr real_t lmbda = 1.35;

    auto one_pass = sfem::create_op(packed_space, "GeneratedLinearElasticity", sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(one_pass != nullptr);
    one_pass->set_option("ASSUME_AFFINE", true);
    one_pass->set_option("PACKED_TWO_PASS", false);
    auto *const one_pass_generated = dynamic_cast<sfem::GeneratedLinearElasticity *>(one_pass.get());
    SFEM_TEST_ASSERT(one_pass_generated != nullptr);
    one_pass_generated->set_value_in_block("default", "mu", mu);
    one_pass_generated->set_value_in_block("default", "lmbda", lmbda);

    auto two_pass = sfem::create_op(packed_space, "GeneratedLinearElasticity", sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(two_pass != nullptr);
    two_pass->set_option("ASSUME_AFFINE", true);
    two_pass->set_option("PACKED_TWO_PASS", true);
    auto *const two_pass_generated = dynamic_cast<sfem::GeneratedLinearElasticity *>(two_pass.get());
    SFEM_TEST_ASSERT(two_pass_generated != nullptr);
    two_pass_generated->set_value_in_block("default", "mu", mu);
    two_pass_generated->set_value_in_block("default", "lmbda", lmbda);

    auto ref_space = sfem::FunctionSpace::create(mesh, BLOCK_SIZE);
    auto ref_op    = sfem::create_op(ref_space, "GeneratedLinearElasticity", sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(ref_op != nullptr);
    ref_op->set_option("ASSUME_AFFINE", true);
    auto *const ref_generated = dynamic_cast<sfem::GeneratedLinearElasticity *>(ref_op.get());
    SFEM_TEST_ASSERT(ref_generated != nullptr);
    ref_generated->set_value_in_block("default", "mu", mu);
    ref_generated->set_value_in_block("default", "lmbda", lmbda);

    const ptrdiff_t     ndofs = packed_space->n_dofs();
    std::vector<real_t> state(ndofs, 0);
    std::vector<real_t> direction(ndofs, 0);
    fill_state_and_direction(packed_space, state, direction);

    std::vector<real_t> grad_ref(ndofs, 0);
    std::vector<real_t> grad_one(ndofs, 0);
    std::vector<real_t> grad_two(ndofs, 0);
    std::vector<real_t> apply_ref(ndofs, 0);
    std::vector<real_t> apply_one(ndofs, 0);
    std::vector<real_t> apply_two(ndofs, 0);

    SFEM_TEST_ASSERT(ref_op->gradient(state.data(), grad_ref.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(one_pass->gradient(state.data(), grad_one.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(two_pass->gradient(state.data(), grad_two.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(ref_op->apply(state.data(), direction.data(), apply_ref.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(one_pass->apply(state.data(), direction.data(), apply_one.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(two_pass->apply(state.data(), direction.data(), apply_two.data()) == SFEM_SUCCESS);

    real_t grad_norm  = 0;
    real_t apply_norm = 0;
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        grad_norm  = std::max(grad_norm, std::abs(grad_ref[i]));
        apply_norm = std::max(apply_norm, std::abs(apply_ref[i]));
    }
    SFEM_TEST_ASSERT(grad_norm > 1e-8);
    SFEM_TEST_ASSERT(apply_norm > 1e-8);

    SFEM_TEST_ASSERT(assert_close_action("generated LE packed one-pass gradient vs reference",
                                         grad_ref,
                                         grad_one,
                                         1e-12,
                                         1e-10) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(assert_close_action("generated LE packed two-pass gradient vs reference",
                                         grad_ref,
                                         grad_two,
                                         1e-12,
                                         1e-10) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(assert_close_action("generated LE packed one-pass vs two-pass gradient",
                                         grad_one,
                                         grad_two,
                                         1e-14,
                                         1e-12) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(assert_close_action("generated LE packed one-pass apply vs reference",
                                         apply_ref,
                                         apply_one,
                                         1e-12,
                                         1e-10) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(assert_close_action("generated LE packed two-pass apply vs reference",
                                         apply_ref,
                                         apply_two,
                                         1e-12,
                                         1e-10) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(assert_close_action("generated LE packed one-pass vs two-pass apply",
                                         apply_one,
                                         apply_two,
                                         1e-14,
                                         1e-12) == SFEM_SUCCESS);

    unsetenv("SFEM_PACKED_TWO_PASS");
    unsetenv("SMESH_ELEMENTS_PER_PACK");
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_generated_neohookean_hessian_action_matrix_formats);
    SFEM_RUN_TEST(test_generated_linear_elasticity_packed_gradient_value_steps);
    SFEM_RUN_TEST(test_generated_linear_elasticity_packed_one_pass_matches_two_pass);
    SFEM_RUN_TEST(test_generated_laplace_crs_bsr_matches_existing_laplacian);
    SFEM_RUN_TEST(test_generated_laplace_hex8_dia_matches_apply);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}

