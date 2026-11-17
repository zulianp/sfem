// The generated high-order applies on the device against the host's, for the operators and orders
// the matrix-core apply takes (sfem_cuda_TensorProductMatrixCoreApply.cuh: Q2 and Q3, a point of at
// most 200 common subexpressions), affine and isoparametric. ctest also runs it with
// SFEM_MATRIX_CORE_APPLY=0, where the generated kernels answer instead.

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_test.hpp"
#include "smesh_semistructured.hpp"

#include <ryml.hpp>
#include <ryml_std.hpp>

namespace {

    std::vector<real_t> pseudo_random(const ptrdiff_t n, const real_t amplitude, unsigned seed) {
        std::vector<real_t> v(n);
        for (ptrdiff_t i = 0; i < n; ++i) {
            seed = seed * 1664525u + 1013904223u;
            v[i] = amplitude * ((real_t)(seed >> 8) / (real_t)(1u << 24) - real_t(0.5));
        }
        return v;
    }

    sfem::SharedBuffer<real_t> host_buffer(const std::vector<real_t> &v) {
        auto h = sfem::create_host_buffer<real_t>((ptrdiff_t)v.size());
        std::copy(v.begin(), v.end(), h->data());
        return h;
    }

    // A smooth displacement of the base mesh: its hexahedra become trilinear, not affine.
    void warp(const std::shared_ptr<sfem::Mesh> &mesh) {
        geom_t **const  x = mesh->points()->data();
        const ptrdiff_t n = mesh->n_nodes();
        for (ptrdiff_t i = 0; i < n; ++i) {
            const geom_t a = x[0][i], b = x[1][i], c = x[2][i];
            x[0][i] += geom_t(0.04) * std::sin(geom_t(3) * b + c);
            x[1][i] += geom_t(0.04) * std::sin(geom_t(2) * c + a);
            x[2][i] += geom_t(0.04) * std::sin(geom_t(3) * a + b);
        }
    }

    // ||y_device - y_host|| / ||y_host|| for the apply at u (amplitude 1e-3) along h.
    real_t device_against_host(const int level, const bool curved, const char *const yaml, const bool affine) {
        auto hex = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 5, 4, 3, 0, 0, 0, 1, 0.8, 0.7);
        if (curved) warp(hex);
        auto fs = sfem::FunctionSpace::create(smesh::to_semistructured(level, hex, true, false), 3);
        fs->set_semi_structured_basis(sfem::FunctionSpace::SemiStructuredBasis::HIGH_ORDER);

        ryml::Tree tree = ryml::parse_in_arena(ryml::to_csubstr(yaml));
        auto       op_h = sfem::create_op_from_yaml(fs, tree.crootref(), sfem::EXECUTION_SPACE_HOST);
        auto       op_d = sfem::create_op_from_yaml(fs, tree.crootref(), sfem::EXECUTION_SPACE_DEVICE);
        if (!op_h || !op_d) return 1e30;
        op_h->set_option("apply_assume_affine", affine);
        op_d->set_option("apply_assume_affine", affine);
        if (op_h->initialize() != SFEM_SUCCESS || op_d->initialize() != SFEM_SUCCESS) return 1e30;

        const ptrdiff_t n   = fs->n_dofs();
        auto            u_h = host_buffer(pseudo_random(n, real_t(1e-3), 3));
        auto            x_h = host_buffer(pseudo_random(n, 1, 5));
        auto            y_h = host_buffer(std::vector<real_t>(n, 0));
        auto            u_d = smesh::to_device(u_h);
        auto            x_d = smesh::to_device(x_h);
        auto            y_d = smesh::to_device(y_h);
        if (op_h->apply(u_h->data(), x_h->data(), y_h->data()) != SFEM_SUCCESS) return 1e30;
        if (op_d->apply(u_d->data(), x_d->data(), y_d->data()) != SFEM_SUCCESS) return 1e30;
        auto y = smesh::to_host(y_d);

        real_t num = 0, den = 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            if (!std::isfinite(y->data()[i])) return 1e30;
            const real_t d = y->data()[i] - y_h->data()[i];
            num += d * d;
            den += y_h->data()[i] * y_h->data()[i];
        }
        return std::sqrt(num / den);
    }

    int check(const int level, const char *const yaml) {
        const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-12) : real_t(1e-5);
        for (const bool curved : {false, true}) {
            for (const bool affine : {true, false}) {
                // The affine apply on a curved mesh is an approximation, but the same one on both.
                const real_t error = device_against_host(level, curved, yaml, affine);
                printf("level %d %-60s %-7s %-14s %.3e\n", level, yaml, curved ? "curved" : "straight",
                       affine ? "affine" : "isoparametric", (double)error);
                SFEM_TEST_ASSERT(error < tol);
            }
        }
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int test_hex27_linear_elasticity() { return check(2, "{type: GeneratedLinearElasticity, mu: 1, lmbda: 2}"); }

int test_hex27_saint_venant_kirchhoff() { return check(2, "{type: GeneratedSaintVenantKirchhoff, mu: 1, lmbda: 2}"); }

int test_hex27_neohookean() { return check(2, "{type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}"); }

int test_hex64_saint_venant_kirchhoff() { return check(3, "{type: GeneratedSaintVenantKirchhoff, mu: 1, lmbda: 2}"); }

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_hex27_linear_elasticity);
    SFEM_RUN_TEST(test_hex27_saint_venant_kirchhoff);
    SFEM_RUN_TEST(test_hex27_neohookean);
    SFEM_RUN_TEST(test_hex64_saint_venant_kirchhoff);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
