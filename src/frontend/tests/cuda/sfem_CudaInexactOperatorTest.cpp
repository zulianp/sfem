// The stored-tangent (INEXACT) operator on the device, against the device's exact apply and the
// host's stored one, element by element.
//
// The device stores keep the host's encodings in the device's layouts and kernel shapes, so their
// operator is the host's: with f64 codes both are the exact apply to rounding; with the others the
// codes agree up to the rare one that a different rounding of the same arithmetic puts on the other
// side of a tie, so the two stored operators differ by far less than either differs from the exact
// one.  The diagonal (Jacobi on the stored operator) is checked against the assembled one.

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_test.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

namespace {
    struct Case {
        const char *store;
        const char *encoding;
        double      tolerance;  // against the exact apply
        double      agreement;  // against the host's stored apply
    };

    const Case CASES[] = {
            {"points", "f64", 1e-12, 1e-12},
            {"points", "f32", 1e-5, 1e-5},
            {"points", "f16", 5e-3, 1e-3},
            {"points", "i8md", 5e-2, 1e-2},
            {"element_matrix", "f64", 1e-12, 1e-12},
            {"element_matrix", "f32", 1e-5, 1e-5},
            {"element_matrix", "f16", 5e-3, 1e-3},
            {"element_matrix", "i8pair", 5e-2, 1e-2},
            {"bsr", "f16", 5e-3, 1e-3},
            {"bsr", "i8", 5e-2, 1e-2},
    };

    std::vector<real_t> pseudo_random(const ptrdiff_t n, const real_t amplitude, unsigned seed) {
        std::vector<real_t> v(n);
        for (ptrdiff_t i = 0; i < n; ++i) {
            seed = seed * 1664525u + 1013904223u;
            v[i] = amplitude * ((real_t)(seed >> 8) / (real_t)(1u << 24) - real_t(0.5));
        }
        return v;
    }

    real_t relative_difference(const std::vector<real_t> &a, const std::vector<real_t> &b) {
        real_t num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(a[i])) return 1e30;
            num += (a[i] - b[i]) * (a[i] - b[i]);
            den += b[i] * b[i];
        }
        return std::sqrt(num / den);
    }

    sfem::SharedBuffer<real_t> on_device(const std::vector<real_t> &v) {
        auto host = sfem::create_host_buffer<real_t>((ptrdiff_t)v.size());
        std::copy(v.begin(), v.end(), host->data());
        return smesh::to_device(host);
    }

    std::vector<real_t> from_device(const sfem::SharedBuffer<real_t> &d) {
        auto host = smesh::to_host(d);
        return std::vector<real_t>(host->data(), host->data() + host->size());
    }

    /// The stored apply (`diagonal`: the stored diagonal) at u, in h, on either side.
    std::vector<real_t> stored(const std::shared_ptr<sfem::Op> &op,
                               const bool                       device,
                               const sfem::InexactOptions      &options,
                               const std::vector<real_t>        &u,
                               const std::vector<real_t>        &h,
                               const bool                       diagonal = false) {
        op->set_inexact_options(options);
        if (!device) {
            std::vector<real_t> out(h.size(), 0);
            if (op->inexact_update(u.data()) != SFEM_SUCCESS) return {};
            const int rc = diagonal ? op->inexact_diag(out.data()) : op->inexact_apply(h.data(), out.data());
            return rc == SFEM_SUCCESS ? out : std::vector<real_t>{};
        }
        auto du  = on_device(u);
        auto dh  = on_device(h);
        auto out = on_device(std::vector<real_t>(h.size(), 0));
        if (op->inexact_update(du->data()) != SFEM_SUCCESS) return {};
        const int rc = diagonal ? op->inexact_diag(out->data()) : op->inexact_apply(dh->data(), out->data());
        return rc == SFEM_SUCCESS ? from_device(out) : std::vector<real_t>{};
    }

    int check(const std::shared_ptr<sfem::FunctionSpace> &fs, const char *const material, const char *const label) {
        auto host   = sfem::create_op(fs, material, sfem::EXECUTION_SPACE_HOST);
        auto device = sfem::create_op(fs, material, sfem::EXECUTION_SPACE_DEVICE);
        SFEM_TEST_ASSERT(host && device);
        SFEM_TEST_ASSERT(host->initialize() == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(device->initialize() == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(device->inexact_supported());

        const ptrdiff_t           ndofs = fs->n_dofs();
        // Small enough that no element of the finer meshes inverts (0.05 does on TET10 4x4x3,
        // where the exact apply is not finite either).
        const std::vector<real_t> u     = pseudo_random(ndofs, 0.01, 7);
        const std::vector<real_t> h     = pseudo_random(ndofs, 1, 11);
        std::vector<real_t>       exact;
        {
            auto du  = on_device(u);
            auto dh  = on_device(h);
            auto out = on_device(std::vector<real_t>(ndofs, 0));
            SFEM_TEST_ASSERT(device->apply(du->data(), dh->data(), out->data()) == SFEM_SUCCESS);
            exact = from_device(out);
        }

        for (const Case &c : CASES) {
            const auto options = sfem::InexactOptions::from_strings(c.store, c.encoding);
            const auto d       = stored(device, true, options, u, h);
            const auto r       = stored(host, false, options, u, h);
            SFEM_TEST_ASSERT(!d.empty() && !r.empty());
            const real_t error = relative_difference(d, exact), agreement = relative_difference(d, r);
            printf("%-14s %-15s %-7s rel diff %.3e (bound %.0e), host %.3e (bound %.0e), %zu bytes\n",
                   label,
                   c.store,
                   c.encoding,
                   (double)error,
                   c.tolerance,
                   (double)agreement,
                   c.agreement,
                   device->inexact_bytes());
            SFEM_TEST_ASSERT(error < c.tolerance);
            SFEM_TEST_ASSERT(agreement < c.agreement);

            const auto dd = stored(device, true, options, u, h, true);
            const auto rd = stored(host, false, options, u, h, true);
            SFEM_TEST_ASSERT(!dd.empty() && !rd.empty());
            const real_t diagonal = relative_difference(dd, rd);
            printf("%-14s %-15s %-7s diagonal against the host's %.3e\n", label, c.store, c.encoding, (double)diagonal);
            SFEM_TEST_ASSERT(diagonal < c.agreement);
        }
        return SFEM_TEST_SUCCESS;
    }

    std::shared_ptr<sfem::FunctionSpace> space(const std::shared_ptr<sfem::Mesh> &mesh) {
        return sfem::FunctionSpace::create(mesh, 3);
    }

    // Curved cells: every node moved by a smooth field, so the Jacobian varies within each element.
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
}  // namespace

// Several blocks of the thread-per-element kernels (128 elements) and a partial last one.
int test_hex8() {
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 9, 7, 5, 0, 0, 0, 1, 0.8, 0.7);
    return check(space(mesh), "GeneratedNeoHookeanOgden", "HEX8");
}

int test_tet4() {
    auto mesh = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 6, 5, 5, 0, 0, 0, 1, 1, 1);
    return check(space(mesh), "GeneratedNeoHookeanOgden", "TET4");
}

int test_tet10() {
    auto mesh = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 4, 4, 3, 0, 0, 0, 1, 1, 1);
    return check(space(smesh::promote_to(smesh::TET10, mesh)), "GeneratedNeoHookeanOgden", "TET10");
}

int test_proteus_hex27() {
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 3, 3, 2, 0, 0, 0, 1, 1, 1);
    auto fs   = space(smesh::to_semistructured(2, mesh, true, false));
    SFEM_TEST_ASSERT(fs != nullptr);
    fs->set_semi_structured_basis(sfem::FunctionSpace::SemiStructuredBasis::HIGH_ORDER);
    return check(fs, "GeneratedNeoHookeanOgden", "PROTEUS_HEX27");
}

int test_curved_hex8() {
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 6, 6, 5, 0, 0, 0, 1, 1, 1);
    warp(mesh);
    return check(space(mesh), "GeneratedNeoHookeanOgden", "curved HEX8");
}

int test_curved_tet10() {
    auto mesh = smesh::promote_to(smesh::TET10, sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 3, 3, 3, 0, 0, 0, 1, 1, 1));
    warp(mesh);
    return check(space(mesh), "GeneratedNeoHookeanOgden", "curved TET10");
}

int test_linear_elasticity() {
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 6, 6, 5, 0, 0, 0, 1, 1, 1);
    return check(space(mesh), "GeneratedLinearElasticity", "HEX8 linear");
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_hex8);
    SFEM_RUN_TEST(test_tet4);
    SFEM_RUN_TEST(test_tet10);
    SFEM_RUN_TEST(test_proteus_hex27);
    SFEM_RUN_TEST(test_curved_hex8);
    SFEM_RUN_TEST(test_curved_tet10);
    SFEM_RUN_TEST(test_linear_elasticity);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
