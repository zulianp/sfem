// The stored-tangent (INEXACT) operator against the exact apply, element by element.
//
// The tangent is stored at the points of the rule the exact apply integrates with, so with f64
// codes the stored apply is the exact one to rounding; every other encoding is the exact one up to
// its own rounding, which bounds the difference here.  The packed and the atomic traversals, and
// the folded and the unfolded point store, compute the same operator.

#include <cmath>
#include <cstdio>
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
        double      tolerance;
    };

    // f64 is the exact operator; the others carry their encoding's rounding: f32 about 1e-7 per
    // number, f16 about 1e-3, int8 mean-plus-deviation about 1e-2 of the deviations.
    const Case CASES[] = {
            {"points", "f64", 1e-12},
            {"points", "f32", 1e-5},
            {"points", "f16", 5e-3},
            {"points", "i8md", 5e-2},
            {"element_matrix", "f64", 1e-12},
            {"element_matrix", "f32", 1e-5},
            {"element_matrix", "f16", 5e-3},
            {"element_matrix", "i8pair", 5e-2},
            {"bsr", "f16", 5e-3},
            {"bsr", "i8", 5e-2},
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
            num += (a[i] - b[i]) * (a[i] - b[i]);
            den += b[i] * b[i];
        }
        return std::sqrt(num / den);
    }

    std::vector<real_t> stored_apply(const std::shared_ptr<sfem::Op> &op,
                                     const sfem::InexactOptions     &options,
                                     const std::vector<real_t>       &u,
                                     const std::vector<real_t>       &h) {
        std::vector<real_t> out(h.size(), 0);
        op->set_inexact_options(options);
        if (op->inexact_update(u.data()) != SFEM_SUCCESS) return {};
        if (op->inexact_apply(h.data(), out.data()) != SFEM_SUCCESS) return {};
        return out;
    }

    int check(const std::shared_ptr<sfem::FunctionSpace> &fs, const char *const material, const char *const label) {
        auto op = sfem::create_op(fs, material, sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(op != nullptr);
        SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(op->inexact_supported());

        const ptrdiff_t           ndofs = fs->n_dofs();
        const std::vector<real_t> u     = pseudo_random(ndofs, 0.05, 7);
        const std::vector<real_t> h     = pseudo_random(ndofs, 1, 11);
        std::vector<real_t>       exact(ndofs, 0);
        SFEM_TEST_ASSERT(op->apply(u.data(), h.data(), exact.data()) == SFEM_SUCCESS);

        for (const Case &c : CASES) {
            const auto options = sfem::InexactOptions::from_strings(c.store, c.encoding);
            const auto stored  = stored_apply(op, options, u, h);
            SFEM_TEST_ASSERT(!stored.empty());
            const real_t error = relative_difference(stored, exact);
            printf("%-14s %-15s %-7s rel diff %.3e (bound %.0e), %zu bytes\n",
                   label,
                   c.store,
                   c.encoding,
                   (double)error,
                   c.tolerance,
                   op->inexact_bytes());
            SFEM_TEST_ASSERT(error < c.tolerance);
        }

        // The stored diagonal against the assembled one: the diagonal of the operator applied.
        {
            auto              graph  = fs->element_node_graph();
            const count_t    *rowptr = graph->rowptr()->data();
            const idx_t      *colidx = graph->colidx()->data();
            const ptrdiff_t   rows   = graph->n_nodes();
            std::vector<real_t> values((size_t)graph->nnz() * 9, 0);
            SFEM_TEST_ASSERT(op->hessian_bsr(u.data(), rowptr, colidx, values.data()) == SFEM_SUCCESS);
            std::vector<real_t> assembled(ndofs, 0);
            for (ptrdiff_t i = 0; i < rows; ++i)
                for (count_t k = rowptr[i]; k < rowptr[i + 1]; ++k)
                    if (colidx[k] == i)
                        for (int d = 0; d < 3; ++d) assembled[i * 3 + d] = values[(size_t)k * 9 + d * 3 + d];
            for (const Case &c : CASES) {
                op->set_inexact_options(sfem::InexactOptions::from_strings(c.store, c.encoding));
                SFEM_TEST_ASSERT(op->inexact_update(u.data()) == SFEM_SUCCESS);
                std::vector<real_t> diag(ndofs, 0);
                SFEM_TEST_ASSERT(op->inexact_diag(diag.data()) == SFEM_SUCCESS);
                const real_t error = relative_difference(diag, assembled);
                printf("%-14s %-15s %-7s diagonal rel diff %.3e\n", label, c.store, c.encoding, (double)error);
                SFEM_TEST_ASSERT(error < c.tolerance);
            }
        }

        // The same operator by both traversals, and with and without the fold.
        const auto packed = stored_apply(op, sfem::InexactOptions::from_strings("points", "f32", "auto", "auto"), u, h);
        const auto atomic = stored_apply(op, sfem::InexactOptions::from_strings("points", "f32", "auto", "atomic"), u, h);
        SFEM_TEST_ASSERT(relative_difference(atomic, packed) < 1e-12);
        const auto unfolded = stored_apply(op, sfem::InexactOptions::from_strings("points", "f64", "off"), u, h);
        const auto folded   = stored_apply(op, sfem::InexactOptions::from_strings("points", "f64", "auto"), u, h);
        SFEM_TEST_ASSERT(relative_difference(folded, unfolded) < 1e-12);
        return SFEM_TEST_SUCCESS;
    }

    std::shared_ptr<sfem::FunctionSpace> space(const std::shared_ptr<sfem::Mesh> &mesh) {
        auto fs = sfem::FunctionSpace::create(mesh, 3);
        if (fs->initialize_packed_mesh() != SFEM_SUCCESS) return nullptr;
        return fs;
    }
}  // namespace

int test_hex8() {
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 4, 3, 3, 0, 0, 0, 1, 0.8, 0.7);
    return check(space(mesh), "GeneratedNeoHookeanOgden", "HEX8");
}

int test_tet4() {
    auto mesh = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 3, 3, 3, 0, 0, 0, 1, 1, 1);
    return check(space(mesh), "GeneratedNeoHookeanOgden", "TET4");
}

int test_tet10() {
    auto mesh = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 2, 2, 2, 0, 0, 0, 1, 1, 1);
    return check(space(smesh::promote_to(smesh::TET10, mesh)), "GeneratedNeoHookeanOgden", "TET10");
}

int test_proteus_hex27() {
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 2, 0, 0, 0, 1, 1, 1);
    auto fs   = space(smesh::to_semistructured(2, mesh, true, false));
    SFEM_TEST_ASSERT(fs != nullptr);
    fs->set_semi_structured_basis(sfem::FunctionSpace::SemiStructuredBasis::HIGH_ORDER);
    return check(fs, "GeneratedNeoHookeanOgden", "PROTEUS_HEX27");
}

// Curved cells: every node moved by a smooth field, so the Jacobian varies within each element.
// The tangent is pulled back per point, so the stored apply stays the exact one.
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

int test_curved_hex8() {
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 3, 3, 3, 0, 0, 0, 1, 1, 1);
    warp(mesh);
    return check(space(mesh), "GeneratedNeoHookeanOgden", "curved HEX8");
}

int test_curved_tet10() {
    auto mesh = smesh::promote_to(smesh::TET10, sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 2, 2, 2, 0, 0, 0, 1, 1, 1));
    warp(mesh);
    return check(space(mesh), "GeneratedNeoHookeanOgden", "curved TET10");
}

int test_linear_elasticity() {
    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 3, 3, 3, 0, 0, 0, 1, 1, 1);
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
}
