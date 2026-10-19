// The generated device BSR assembly of energy materials against the host one: the same material
// from the same YAML on both, the same deformed state, the tangent assembled on the node graph.

#include <algorithm>
#include <cmath>
#include <string>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_HistoryDependent.hpp"
#include "sfem_test.hpp"

#include <ryml.hpp>
#include <ryml_std.hpp>

using namespace sfem;

namespace {
    std::shared_ptr<Mesh> cube() { return Mesh::create_hex8_cube(Communicator::world(), 4, 3, 3, 0, 0, 0, 1, 1, 1); }

    // max |device - host| / max |host| over the BSR values, on `mesh` (a semi-structured one through
    // its micro elements).
    real_t assemble_and_compare(const std::shared_ptr<Mesh> &mesh, const char *const yaml) {
        auto fs = FunctionSpace::create(mesh, 3);

        ryml::Tree tree = ryml::parse_in_arena(ryml::to_csubstr(yaml));
        auto       host = create_op_from_yaml(fs, tree.crootref(), EXECUTION_SPACE_HOST);
        auto       dev  = create_op_from_yaml(fs, tree.crootref(), EXECUTION_SPACE_DEVICE);
        if (!host || !dev) return 1;

        for (auto &op : {host, dev}) {
            if (auto history = std::dynamic_pointer_cast<HistoryDependent>(op)) history->begin_step(real_t(0.1), real_t(0.1));
        }

        const ptrdiff_t      ndofs = fs->n_dofs();
        auto                 x     = create_host_buffer<real_t>(ndofs);
        const geom_t *const *xyz   = mesh->points()->data();
        for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
            x->data()[i * 3 + 0] = real_t(0.05) * xyz[0][i] * xyz[1][i];
            x->data()[i * 3 + 1] = real_t(-0.03) * xyz[0][i] + real_t(0.02) * xyz[2][i];
            x->data()[i * 3 + 2] = real_t(0.04) * xyz[1][i] * xyz[2][i];
        }

        auto       graph  = fs->node_to_node_graph();
        const auto nvals  = graph->nnz() * 9;
        auto       values = create_host_buffer<real_t>(nvals);
        if (host->hessian_bsr(x->data(), graph->rowptr()->data(), graph->colidx()->data(), values->data()) != SFEM_SUCCESS) return 1;

        auto d_x      = smesh::to_device(x);
        auto d_rowptr = smesh::to_device(graph->rowptr());
        auto d_colidx = smesh::to_device(graph->colidx());
        auto d_values = create_buffer<real_t>(nvals, EXECUTION_SPACE_DEVICE);
        if (dev->hessian_bsr(d_x->data(), d_rowptr->data(), d_colidx->data(), d_values->data()) != SFEM_SUCCESS) return 1;
        auto from_device = smesh::to_host(d_values);

        real_t worst = 0, scale = 0;
        for (ptrdiff_t i = 0; i < nvals; ++i) {
            worst = std::max(worst, std::abs(from_device->data()[i] - values->data()[i]));
            scale = std::max(scale, std::abs(values->data()[i]));
        }
        printf("  %s: max |K| %g, worst relative difference %.3g\n", dev->name(), (double)scale, (double)(worst / scale));
        return scale > 0 ? worst / scale : 1;
    }

    // The symmetric block diagonal (hessian_block_diag_sym), what the block-Jacobi smoothers of
    // SSMGC read: max |device - host| / max |host| on `mesh` (a semi-structured one through its
    // micro elements).
    real_t block_diag_and_compare(const std::shared_ptr<Mesh> &mesh, const char *const yaml) {
        auto fs = FunctionSpace::create(mesh, 3);

        ryml::Tree tree = ryml::parse_in_arena(ryml::to_csubstr(yaml));
        auto       host = create_op_from_yaml(fs, tree.crootref(), EXECUTION_SPACE_HOST);
        auto       dev  = create_op_from_yaml(fs, tree.crootref(), EXECUTION_SPACE_DEVICE);
        if (!host || !dev) return 1;

        const ptrdiff_t nvals  = fs->n_dofs() / 3 * 6;
        auto            values = create_host_buffer<real_t>(nvals);
        if (host->hessian_block_diag_sym(nullptr, values->data()) != SFEM_SUCCESS) return 1;
        auto d_values = create_buffer<real_t>(nvals, EXECUTION_SPACE_DEVICE);
        if (dev->hessian_block_diag_sym(nullptr, d_values->data()) != SFEM_SUCCESS) return 1;
        auto from_device = smesh::to_host(d_values);

        real_t worst = 0, scale = 0;
        for (ptrdiff_t i = 0; i < nvals; ++i) {
            worst = std::max(worst, std::abs(from_device->data()[i] - values->data()[i]));
            scale = std::max(scale, std::abs(values->data()[i]));
        }
        printf("  %s block_diag_sym: max |D| %g, worst relative difference %.3g\n", dev->name(), (double)scale, (double)(worst / scale));
        return scale > 0 ? worst / scale : 1;
    }
}  // namespace

int test_linear_elasticity_block_diag_on_device() {
    const char *const yaml = "{type: GeneratedLinearElasticity, mu: 1, lmbda: 2}";
    auto              hex  = Mesh::create_hex8_cube(Communicator::world(), 4, 3, 3, 0, 0, 0, 1, 1, 1);
    SFEM_TEST_ASSERT(block_diag_and_compare(hex, yaml) < 1e-12);
    SFEM_TEST_ASSERT(block_diag_and_compare(smesh::to_semistructured(3, hex, true, false), yaml) < 1e-12);
    return SFEM_TEST_SUCCESS;
}

// The node_sdf obstacle assembles this tangent on the device every Newton iteration.
int test_linear_elasticity_bsr_on_device() {
    const char *const yaml = "{type: GeneratedLinearElasticity, mu: 1, lmbda: 2}";
    SFEM_TEST_ASSERT(assemble_and_compare(cube(), yaml) < 1e-12);
    SFEM_TEST_ASSERT(assemble_and_compare(smesh::to_semistructured(3, cube(), true, false), yaml) < 1e-12);
    return SFEM_TEST_SUCCESS;
}

int test_neohookean_bsr_on_device() {
    SFEM_TEST_ASSERT(assemble_and_compare(cube(), "{type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}") < 1e-12);
    return SFEM_TEST_SUCCESS;
}

int test_prony_bsr_on_device() {
    SFEM_TEST_ASSERT(assemble_and_compare(cube(), "{type: GeneratedMooneyRivlinProny, C10: 0.3, C01: 0.05, K: 50, "
                                          "relaxation: {iso: {prony: [{g: 0.4, tau: 1}, {g: 0.3, tau: 10}]}}}") < 1e-12);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_linear_elasticity_bsr_on_device);
    SFEM_RUN_TEST(test_neohookean_bsr_on_device);
    SFEM_RUN_TEST(test_prony_bsr_on_device);
    SFEM_RUN_TEST(test_linear_elasticity_block_diag_on_device);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
