// The continued Mooney-Rivlin-Prony operator on the device, against the host.
//
// Both operators are driven through the same committed history and compared at every step on the
// objective, the gradient and the Hessian action -- once where nothing is compressed below Jc,
// and once on a field that inverts elements, where only the continuation keeps the three finite
// and the device has to take the same pieces of h_m as the host.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_BLAS_cuda.hpp"
#include "sfem_GeneratedContinuedMooneyRivlinProny.hpp"
#include "sfem_GeneratedContinuedMooneyRivlinProny_cuda.hpp"
#include "sfem_HistoryDependent.hpp"
#include "sfem_PronyRelaxation.hpp"

using namespace sfem;

namespace {

    const real_t test_dt = 0.1;

    void fill_pseudo_random(const ptrdiff_t n, const real_t scale, real_t *const x, uint64_t state) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            x[i] = scale * (real_t)((double)(state >> 11) / (double)(1ull << 53) - 0.5);
        }
    }

    real_t relative_difference(const std::shared_ptr<Buffer<real_t>> &host, const std::shared_ptr<Buffer<real_t>> &device) {
        auto   hd   = smesh::to_host(device);
        real_t diff = 0, norm = 0;
        for (ptrdiff_t i = 0; i < (ptrdiff_t)host->size(); i++) {
            if (!std::isfinite(hd->data()[i]) || !std::isfinite(host->data()[i])) return 1e30;
            diff = std::max(diff, std::abs(host->data()[i] - hd->data()[i]));
            norm = std::max(norm, std::abs(host->data()[i]));
        }
        return norm > 0 ? diff / norm : diff;
    }

    template <class OpType>
    void configure(OpType &op, const std::shared_ptr<Mesh> &mesh) {
        for (const auto &block : mesh->blocks()) {
            op.set_value_in_block(block->name(), "C10", real_t(0.3));
            op.set_value_in_block(block->name(), "C01", real_t(0.05));
            op.set_value_in_block(block->name(), "K", real_t(50));
            op.set_value_in_block(block->name(), "Jc", real_t(0.2));
            op.set_value_in_block(block->name(), "Jmin", real_t(-1));
        }
        op.relaxation_groups().set_series("", "iso", PronySeries({0.40, 0.30}, {1.0, 10.0}));
    }

    /// `noise` sets how far the elements are deformed: 0.01 keeps every J near 1, 0.3 inverts some.
    int host_vs_device(const smesh::ElemType element, const real_t noise) {
        const int n    = 4;
        auto      mesh = Mesh::create_tet4_cube(Communicator::self(), n, n, n, 0, 0, 0, 1, 1, 1);
        if (element == smesh::TET10) mesh = smesh::promote_to(smesh::TET10, mesh);
        auto fs = FunctionSpace::create(mesh, 3);

        auto host_op = std::dynamic_pointer_cast<GeneratedContinuedMooneyRivlinProny>(
                create_op(fs, "GeneratedContinuedMooneyRivlinProny", EXECUTION_SPACE_HOST));
        auto device_op = std::dynamic_pointer_cast<GPUGeneratedContinuedMooneyRivlinProny>(
                create_op(fs, "GeneratedContinuedMooneyRivlinProny", EXECUTION_SPACE_DEVICE));
        SFEM_TEST_ASSERT(host_op != nullptr);
        SFEM_TEST_ASSERT(device_op != nullptr);
        configure(*host_op, mesh);
        configure(*device_op, mesh);

        const ptrdiff_t ndofs = fs->n_dofs();
        auto            u_h   = create_host_buffer<real_t>(ndofs);
        auto            d_h   = create_host_buffer<real_t>(ndofs);
        fill_pseudo_random(ndofs, noise, u_h->data(), 12345);
        fill_pseudo_random(ndofs, real_t(1), d_h->data(), 777);
        auto d_d = smesh::to_device(d_h);

        auto g_h = create_host_buffer<real_t>(ndofs);
        auto g_d = create_buffer<real_t>(ndofs, EXECUTION_SPACE_DEVICE);
        auto a_h = create_host_buffer<real_t>(ndofs);
        auto a_d = create_buffer<real_t>(ndofs, EXECUTION_SPACE_DEVICE);

        real_t worst = 0;
        real_t t     = 0;
        for (int step = 1; step <= 4; step++) {
            for (ptrdiff_t i = 0; i < ndofs; i++) u_h->data()[i] *= real_t(1.1);
            auto u_d = smesh::to_device(u_h);
            t += test_dt;
            host_op->begin_step(t, test_dt);
            device_op->begin_step(t, test_dt);

            real_t v_h = 0, v_d = 0;
            SFEM_TEST_ASSERT(host_op->value(u_h->data(), &v_h) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(device_op->value(u_d->data(), &v_d) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(std::isfinite(v_h));

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
            printf("  %-6s noise %.2f step %d ndof=%td: value %.3e (host %.6g) gradient %.3e apply %.3e\n",
                   smesh::type_to_string(element),
                   (double)noise,
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

        SFEM_TEST_ASSERT(worst < 1e-10);
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int test_device_matches_host_tet4() { return host_vs_device(smesh::TET4, real_t(0.01)); }
int test_device_matches_host_tet10() { return host_vs_device(smesh::TET10, real_t(0.005)); }
// The history is advanced at inverted states here too, which a run never does (a committed state
// has J >= Jc); it is the same arithmetic on both sides all the same.
int test_device_matches_host_through_inversion_tet4() { return host_vs_device(smesh::TET4, real_t(0.3)); }
int test_device_matches_host_through_inversion_tet10() { return host_vs_device(smesh::TET10, real_t(0.15)); }

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_device_matches_host_tet4);
    SFEM_RUN_TEST(test_device_matches_host_tet10);
    SFEM_RUN_TEST(test_device_matches_host_through_inversion_tet4);
    SFEM_RUN_TEST(test_device_matches_host_through_inversion_tet10);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
