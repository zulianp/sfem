// The generated device BSR assembly of energy materials against the host one: the same material
// from the same YAML on both, the same deformed state, the tangent assembled on the node graph.

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "sfem_API.hpp"
#include "sfem_BSRTangent.hpp"
#include "sfem_Function.hpp"
#include "sfem_HistoryDependent.hpp"
#include "sfem_test.hpp"

#include <ryml.hpp>
#include <ryml_std.hpp>

using namespace sfem;

namespace {
    std::shared_ptr<Mesh> cube() { return Mesh::create_hex8_cube(Communicator::world(), 4, 3, 3, 0, 0, 0, 1, 1, 1); }
    // TET4 takes the affine kernels, with the geometry the op caches on the device.
    std::shared_ptr<Mesh> tet_cube() { return Mesh::create_tet4_cube(Communicator::world(), 4, 3, 3, 0, 0, 0, 1, 1, 1); }

    // The deformed state both sides evaluate.
    SharedBuffer<real_t> deformed(const std::shared_ptr<Mesh> &mesh, const ptrdiff_t ndofs) {
        auto                 x   = create_host_buffer<real_t>(ndofs);
        const geom_t *const *xyz = mesh->points()->data();
        for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
            x->data()[i * 3 + 0] = real_t(0.05) * xyz[0][i] * xyz[1][i];
            x->data()[i * 3 + 1] = real_t(-0.03) * xyz[0][i] + real_t(0.02) * xyz[2][i];
            x->data()[i * 3 + 2] = real_t(0.04) * xyz[1][i] * xyz[2][i];
        }
        return x;
    }

    // max |device - host| / max |host| over the gradient at the deformed state.
    real_t gradient_and_compare(const std::shared_ptr<Mesh> &mesh, const char *const yaml) {
        auto       fs   = FunctionSpace::create(mesh, 3);
        ryml::Tree tree = ryml::parse_in_arena(ryml::to_csubstr(yaml));
        auto       host = create_op_from_yaml(fs, tree.crootref(), EXECUTION_SPACE_HOST);
        auto       dev  = create_op_from_yaml(fs, tree.crootref(), EXECUTION_SPACE_DEVICE);
        if (!host || !dev) return 1;

        const ptrdiff_t ndofs = fs->n_dofs();
        auto            x     = deformed(mesh, ndofs);
        auto            g     = create_host_buffer<real_t>(ndofs);
        if (host->gradient(x->data(), g->data()) != SFEM_SUCCESS) return 1;
        auto d_x = smesh::to_device(x);
        auto d_g = create_buffer<real_t>(ndofs, EXECUTION_SPACE_DEVICE);
        if (dev->gradient(d_x->data(), d_g->data()) != SFEM_SUCCESS) return 1;
        auto from_device = smesh::to_host(d_g);

        real_t worst = 0, scale = 0;
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            worst = std::max(worst, std::abs(from_device->data()[i] - g->data()[i]));
            scale = std::max(scale, std::abs(g->data()[i]));
        }
        printf("  %s gradient: max |g| %g, worst relative difference %.3g\n", dev->name(), (double)scale, (double)(worst / scale));
        return scale > 0 ? worst / scale : 1;
    }

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

        const ptrdiff_t ndofs = fs->n_dofs();
        auto            x     = deformed(mesh, ndofs);

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

// HEX8 through the isoparametric kernels, TET4 through the affine ones and their cached geometry,
// whose components the launch reads on the host.
int test_neohookean_bsr_on_device() {
    const char *const yaml = "{type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}";
    SFEM_TEST_ASSERT(assemble_and_compare(cube(), yaml) < 1e-12);
    SFEM_TEST_ASSERT(assemble_and_compare(tet_cube(), yaml) < 1e-12);
    SFEM_TEST_ASSERT(gradient_and_compare(tet_cube(), yaml) < 1e-12);
    return SFEM_TEST_SUCCESS;
}

int test_prony_bsr_on_device() {
    SFEM_TEST_ASSERT(assemble_and_compare(cube(), "{type: GeneratedMooneyRivlinProny, C10: 0.3, C01: 0.05, K: 50, "
                                          "relaxation: {iso: {prony: [{g: 0.4, tau: 1}, {g: 0.3, tau: 10}]}}}") < 1e-12);
    return SFEM_TEST_SUCCESS;
}

// A tangent stored in single precision (BSRTangent, storage_bytes 4), assembled on the device from
// element matrices in double: its values, product, diagonal, bound and shift against the host's
// double tangent, to float's rounding.
int test_single_precision_tangent_on_device() {
#if SFEM_REAL_T_IS_FLOAT32
    return SFEM_TEST_SUCCESS;
#else
    const char *const yaml = "{type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}";
    auto              mesh = tet_cube();
    auto              fs   = FunctionSpace::create(mesh, 3);
    ryml::Tree        tree = ryml::parse_in_arena(ryml::to_csubstr(yaml));
    auto              fh   = Function::create(fs);
    auto              fd   = Function::create(fs);
    fh->add_operator(create_op_from_yaml(fs, tree.crootref(), EXECUTION_SPACE_HOST));
    fd->add_operator(create_op_from_yaml(fs, tree.crootref(), EXECUTION_SPACE_DEVICE));

    auto full   = BSRTangent::create(fh, EXECUTION_SPACE_HOST);
    auto single = BSRTangent::create(fd, EXECUTION_SPACE_DEVICE, 4);
    SFEM_TEST_ASSERT(single->values_type() == smesh::SMESH_FLOAT32);
    const ptrdiff_t n = fs->n_dofs();
    auto            x = deformed(mesh, n);
    SFEM_TEST_ASSERT(full->assemble(x->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(single->assemble(smesh::to_device(x)->data()) == SFEM_SUCCESS);

    const real_t        eps     = std::numeric_limits<float>::epsilon();
    const ptrdiff_t     nvalues = full->rowptr()[full->block_rows()] * 9;
    const real_t *const v64     = static_cast<const real_t *>(full->values());
    std::vector<float>  v32(nvalues);
    SFEM_TEST_ASSERT(cudaMemcpy(v32.data(), single->values(), nvalues * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    real_t vmax = 0, verr = 0;
    for (ptrdiff_t k = 0; k < nvalues; ++k) {
        vmax = std::max(vmax, std::abs(v64[k]));
        verr = std::max(verr, std::abs(real_t(v32[k]) - v64[k]));
    }
    printf("  single-precision values: |K32 - K64| / |K64| = %.3g\n", (double)(verr / vmax));
    SFEM_TEST_ASSERT(verr > 0 && verr <= 32 * eps * vmax);

    auto h = create_host_buffer<real_t>(n), y64 = create_host_buffer<real_t>(n);
    for (ptrdiff_t i = 0; i < n; ++i) h->data()[i] = std::sin(real_t(0.37) * (i + 1));
    SFEM_TEST_ASSERT(full->op()->apply(h->data(), y64->data()) == SFEM_SUCCESS);
    auto d_y = create_buffer<real_t>(n, EXECUTION_SPACE_DEVICE);
    blas<real_t>(EXECUTION_SPACE_DEVICE)->zeros(n, d_y->data());
    SFEM_TEST_ASSERT(single->op()->apply(smesh::to_device(h)->data(), d_y->data()) == SFEM_SUCCESS);
    auto   y32  = smesh::to_host(d_y);
    real_t ymax = 0, yerr = 0;
    for (ptrdiff_t i = 0; i < n; ++i) {
        ymax = std::max(ymax, std::abs(y64->data()[i]));
        yerr = std::max(yerr, std::abs(y32->data()[i] - y64->data()[i]));
    }
    printf("  single-precision product: %.3g\n", (double)(yerr / ymax));
    SFEM_TEST_ASSERT(yerr <= 32 * eps * ymax);

    auto d64 = create_host_buffer<real_t>(n);
    auto d_d = create_buffer<real_t>(n, EXECUTION_SPACE_DEVICE);
    SFEM_TEST_ASSERT(full->diagonal(d64->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(single->diagonal(d_d->data()) == SFEM_SUCCESS);
    auto   d32  = smesh::to_host(d_d);
    real_t dmax = 0;
    for (ptrdiff_t i = 0; i < n; ++i) dmax = std::max(dmax, std::abs(d64->data()[i]));
    for (ptrdiff_t i = 0; i < n; ++i) SFEM_TEST_ASSERT(std::abs(d32->data()[i] - d64->data()[i]) <= 32 * eps * dmax);
    SFEM_TEST_ASSERT(std::abs(single->max_abs_row_sum() - full->max_abs_row_sum()) <= 32 * eps * full->max_abs_row_sum());

    SFEM_TEST_ASSERT(single->shift_diagonal(real_t(0.5), smesh::to_device(d64)->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(single->diagonal(d_d->data()) == SFEM_SUCCESS);
    auto shifted = smesh::to_host(d_d);
    for (ptrdiff_t i = 0; i < n; ++i) {
        SFEM_TEST_ASSERT(std::abs(shifted->data()[i] - (d32->data()[i] + real_t(0.5) * std::abs(d64->data()[i]))) <= 2 * eps * dmax);
    }
    return SFEM_TEST_SUCCESS;
#endif
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_linear_elasticity_bsr_on_device);
    SFEM_RUN_TEST(test_neohookean_bsr_on_device);
    SFEM_RUN_TEST(test_prony_bsr_on_device);
    SFEM_RUN_TEST(test_linear_elasticity_block_diag_on_device);
    SFEM_RUN_TEST(test_single_precision_tangent_on_device);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
