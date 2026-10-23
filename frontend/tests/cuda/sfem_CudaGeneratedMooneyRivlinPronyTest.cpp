// The generated Prony-series Mooney-Rivlin operator on the device, against the host.
//
// The device operator holds its relaxation state in device memory: the stored H_i, the frozen
// history the kernels read, the previous displacement, and the stress workspaces the
// quadrature_output kernel writes.  A device test that only exits 0 proves nothing about that,
// so both operators are driven through the same committed history and compared at every step:
// the objective, the gradient and the Hessian action.  The history feeds all three, so a step
// that went wrong on the device -- a wrong S_hist, a lost commit, a host pointer handed to a
// kernel -- shows up as a mismatch in the step after it.
//
// Run for full storage and for two reduced formats of H_i.  The device codecs write the same
// bits as the host ones -- one rounding to the stored type, as on the host -- so even the
// quantised stores agree up to the rare code that a different rounding of the same arithmetic
// puts on the other side of a tie.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_GeneratedMooneyRivlinProny.hpp"
#include "sfem_GeneratedMooneyRivlinProny_cuda.hpp"
#include "sfem_HistoryDependent.hpp"
#include "sfem_cuda_blas.hpp"

using namespace sfem;

namespace {

    const real_t test_C10 = 0.3;
    const real_t test_C01 = 0.05;
    const real_t test_K   = 50.0;
    const real_t test_dt  = 0.1;

    void fill_pseudo_random(const ptrdiff_t n, const real_t scale, real_t *const x, uint64_t state) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            x[i] = scale * (real_t)((double)(state >> 11) / (double)(1ull << 53) - 0.5);
        }
    }

    void distort_interior(const std::shared_ptr<Mesh> &mesh, const int n, const real_t amplitude) {
        const ptrdiff_t     nnodes = mesh->n_nodes();
        auto                points = mesh->points()->data();
        std::vector<real_t> offset(nnodes * 3);
        fill_pseudo_random((ptrdiff_t)offset.size(), amplitude / n, offset.data(), 0x9e3779b97f4a7c15ull);
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            bool interior = true;
            for (int d = 0; d < 3; ++d) {
                const geom_t c = points[d][i];
                if (c <= geom_t(1e-6) || c >= geom_t(1 - 1e-6)) interior = false;
            }
            if (interior) {
                for (int d = 0; d < 3; ++d) points[d][i] += (geom_t)offset[i * 3 + d];
            }
        }
    }

    real_t relative_difference(const std::shared_ptr<Buffer<real_t>> &host, const std::shared_ptr<Buffer<real_t>> &device) {
        auto   hd   = smesh::to_host(device);
        real_t diff = 0, norm = 0;
        for (ptrdiff_t i = 0; i < host->size(); i++) {
            if (!std::isfinite(hd->data()[i])) return 1e30;
            diff = std::max(diff, std::abs(host->data()[i] - hd->data()[i]));
            norm = std::max(norm, std::abs(host->data()[i]));
        }
        return norm > 0 ? diff / norm : diff;
    }

    template <class OpType>
    void configure(OpType &op, const std::shared_ptr<Mesh> &mesh, const HistoryStorage storage) {
        for (const auto &block : mesh->blocks()) {
            op.set_value_in_block(block->name(), "C10", test_C10);
            op.set_value_in_block(block->name(), "C01", test_C01);
            op.set_value_in_block(block->name(), "K", test_K);
        }
        PronyRelaxation::Options options;
        options.history_storage = storage;
        op.relaxation_groups().set_series("", "iso", PronySeries({0.40, 0.30}, {1.0, 10.0}), options);
    }

    int host_vs_device(const HistoryStorage storage, const real_t tolerance, const smesh::ElemType element = smesh::HEX8) {
        const int n = 6;
        std::shared_ptr<Mesh> mesh;
        if (element == smesh::HEX8) {
            mesh = Mesh::create_hex8_cube(Communicator::self(), n, n, n, 0, 0, 0, 1, 1, 1);
        } else {
            mesh = Mesh::create_tet4_cube(Communicator::self(), n, n, n, 0, 0, 0, 1, 1, 1);
        }
        // On the grid's vertices, so before TET10's midside nodes exist: promoted after, its
        // elements stay straight.
        distort_interior(mesh, n, real_t(0.15));
        if (element == smesh::TET10) mesh = smesh::promote_to(smesh::TET10, mesh);
        auto fs = FunctionSpace::create(mesh, 3);

        auto host_op   = std::dynamic_pointer_cast<GeneratedMooneyRivlinProny>(create_op(fs, "GeneratedMooneyRivlinProny", EXECUTION_SPACE_HOST));
        auto device_op = std::dynamic_pointer_cast<GPUGeneratedMooneyRivlinProny>(
                create_op(fs, "GeneratedMooneyRivlinProny", EXECUTION_SPACE_DEVICE));
        SFEM_TEST_ASSERT(host_op != nullptr);
        SFEM_TEST_ASSERT(device_op != nullptr);
        configure(*host_op, mesh, storage);
        configure(*device_op, mesh, storage);

        const ptrdiff_t ndofs = fs->n_dofs();
        auto            u_h   = create_host_buffer<real_t>(ndofs);
        auto            d_h   = create_host_buffer<real_t>(ndofs);
        // Random nodal noise grown by 1.15 a step: the six tetrahedra of each cube, and TET10's
        // midside nodes, invert where a hexahedron does not, so the simplices take a fifth of it.
        fill_pseudo_random(ndofs, element == smesh::HEX8 ? real_t(0.05) : real_t(0.01), u_h->data(), 12345);
        fill_pseudo_random(ndofs, real_t(1), d_h->data(), 777);
        auto d_d = smesh::to_device(d_h);

        auto g_h = create_host_buffer<real_t>(ndofs);
        auto g_d = create_buffer<real_t>(ndofs, EXECUTION_SPACE_DEVICE);
        auto a_h = create_host_buffer<real_t>(ndofs);
        auto a_d = create_buffer<real_t>(ndofs, EXECUTION_SPACE_DEVICE);

        real_t worst = 0;
        real_t t     = 0;
        for (int step = 1; step <= 6; step++) {
            for (ptrdiff_t i = 0; i < ndofs; i++) u_h->data()[i] *= real_t(1.15);
            auto u_d = smesh::to_device(u_h);
            t += test_dt;

            host_op->begin_step(t, test_dt);
            device_op->begin_step(t, test_dt);

            real_t v_h = 0, v_d = 0;
            SFEM_TEST_ASSERT(host_op->value(u_h->data(), &v_h) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(device_op->value(u_d->data(), &v_d) == SFEM_SUCCESS);

            std::fill(g_h->data(), g_h->data() + ndofs, real_t(0));
            d_memset(g_d->data(), 0, ndofs * sizeof(real_t));
            SFEM_TEST_ASSERT(host_op->gradient(u_h->data(), g_h->data()) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(device_op->gradient(u_d->data(), g_d->data()) == SFEM_SUCCESS);

            std::fill(a_h->data(), a_h->data() + ndofs, real_t(0));
            d_memset(a_d->data(), 0, ndofs * sizeof(real_t));
            SFEM_TEST_ASSERT(host_op->apply(u_h->data(), d_h->data(), a_h->data()) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(device_op->apply(u_d->data(), d_d->data(), a_d->data()) == SFEM_SUCCESS);

            const real_t value_error    = std::abs(v_h - v_d) / std::max(std::abs(v_h), real_t(1e-30));
            const real_t gradient_error = relative_difference(g_h, g_d);
            const real_t apply_error    = relative_difference(a_h, a_d);
            printf("  %-6s %-15s step %d ndof=%td: value %.3e (host %.6g) gradient %.3e apply %.3e\n",
                   smesh::type_to_string(element),
                   to_string(storage),
                   step,
                   ndofs,
                   (double)value_error,
                   (double)v_h,
                   (double)gradient_error,
                   (double)apply_error);
            worst = std::max({worst, value_error, gradient_error, apply_error});

            host_op->advance(u_h->data());
            device_op->advance(u_d->data());
        }

        SFEM_TEST_ASSERT(worst < tolerance);
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

// Full storage: the two sides differ only in the order of floating-point operations.
int test_device_matches_host_full_storage() { return host_vs_device(HistoryStorage::FULL, real_t(1e-10)); }

// The simplices: TET4's one-point kernels and TET10's eleven-point ones.
int test_device_matches_host_tet4() { return host_vs_device(HistoryStorage::FULL, real_t(1e-10), smesh::TET4); }
int test_device_matches_host_tet10() { return host_vs_device(HistoryStorage::FULL, real_t(1e-10), smesh::TET10); }

// Quantised H_i: identical codes except where a rounding tie falls differently; one code of
// int16 is 1.5e-5 of an element's largest entry.
int test_device_matches_host_int16_storage() { return host_vs_device(HistoryStorage::INT16_SCALED, real_t(1e-6)); }

// float16 under a power-of-two scale: one code is 2^-11 of the entry.
int test_device_matches_host_float16_storage() {
    return host_vs_device(HistoryStorage::FLOAT16_SCALED, real_t(1e-5));
}

// Any generated energy material on HEX8 goes through the same HEX8 -> PROTEUS_HEX8 forwarding
// as this one, so the existing modified Mooney-Rivlin is checked on the device too: its gradient
// against the host's.  If this fails as well, a failure above is the forwarding's, not Prony's.
int test_existing_hex8_material_runs_on_the_device() {
    auto mesh = Mesh::create_hex8_cube(Communicator::self(), 4, 4, 4, 0, 0, 0, 1, 1, 1);
    auto fs   = FunctionSpace::create(mesh, 3);
    auto op_h = create_op(fs, "GeneratedModifiedMooneyRivlin", EXECUTION_SPACE_HOST);
    auto op_d = create_op(fs, "GeneratedModifiedMooneyRivlin", EXECUTION_SPACE_DEVICE);
    SFEM_TEST_ASSERT(op_h && op_d);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            u_h   = create_host_buffer<real_t>(ndofs);
    fill_pseudo_random(ndofs, real_t(0.05), u_h->data(), 4242);
    auto u_d = smesh::to_device(u_h);
    auto g_h = create_host_buffer<real_t>(ndofs);
    std::fill(g_h->data(), g_h->data() + ndofs, real_t(0));
    auto g_d = create_buffer<real_t>(ndofs, EXECUTION_SPACE_DEVICE);
    d_memset(g_d->data(), 0, ndofs * sizeof(real_t));

    SFEM_TEST_ASSERT(op_h->gradient(u_h->data(), g_h->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op_d->gradient(u_d->data(), g_d->data()) == SFEM_SUCCESS);
    const real_t error = relative_difference(g_h, g_d);
    printf("  modified Mooney-Rivlin HEX8 device vs host gradient: %.3e\n", (double)error);
    SFEM_TEST_ASSERT(error < 1e-10);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_existing_hex8_material_runs_on_the_device);
    SFEM_RUN_TEST(test_device_matches_host_full_storage);
    SFEM_RUN_TEST(test_device_matches_host_int16_storage);
    SFEM_RUN_TEST(test_device_matches_host_float16_storage);
    SFEM_RUN_TEST(test_device_matches_host_tet4);
    SFEM_RUN_TEST(test_device_matches_host_tet10);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
