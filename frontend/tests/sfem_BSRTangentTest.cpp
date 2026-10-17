// BSRTangent against the arrays it assembles: its diagonal, its row-sum bound, its diagonal shift,
// and the operator `sfem::hessian_bsr` returns, which it now builds.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_BSRTangent.hpp"
#include "sfem_elasticity_test_problem.hpp"
#include "sfem_test.hpp"

namespace {
    // A displacement that makes the Neo-Hookean tangent depend on the state.
    void deform(sfem::test::CubeProblem &p) {
        const geom_t *const *const xyz = p.mesh->points()->data();
        real_t *const              x   = p.x->data();
        for (ptrdiff_t i = 0; i < p.mesh->n_nodes(); ++i) {
            x[i * 3 + 0] += real_t(0.05) * xyz[1][i] * xyz[2][i];
            x[i * 3 + 1] += real_t(0.03) * xyz[0][i];
            x[i * 3 + 2] -= real_t(0.02) * xyz[0][i] * xyz[1][i];
        }
        p.f->apply_constraints(x);
    }
}  // namespace

int test_diagonal_and_row_sums() {
    auto p = sfem::test::make_cube_problem("GeneratedNeoHookeanOgden", 3, real_t(0.1));
    deform(p);

    auto tangent = sfem::BSRTangent::create(p.f, sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(tangent->assemble(p.x->data()) == SFEM_SUCCESS);

    const ptrdiff_t      nrows = tangent->block_rows();
    const int            bs    = tangent->block_size();
    const count_t *const rowptr = tangent->rowptr();
    const idx_t *const   colidx = tangent->colidx();
    real_t *const        vals   = tangent->values();

    std::vector<real_t> diag(nrows * bs);
    SFEM_TEST_ASSERT(tangent->diagonal(diag.data()) == SFEM_SUCCESS);

    real_t bound = 0;
    for (ptrdiff_t row = 0; row < nrows; ++row) {
        const real_t *const B = sfem::bsr_diagonal_block(row, rowptr, colidx, bs, vals);
        SFEM_TEST_ASSERT(B != nullptr);
        for (int d = 0; d < bs; ++d) {
            SFEM_TEST_ASSERT(diag[row * bs + d] == B[d * bs + d]);

            real_t row_sum = 0;
            for (count_t k = rowptr[row]; k < rowptr[row + 1]; ++k) {
                for (int c = 0; c < bs; ++c) row_sum += std::abs(vals[k * bs * bs + d * bs + c]);
            }
            bound = std::max(bound, row_sum);
        }
    }
    SFEM_TEST_ASSERT(bound > 0);
    SFEM_TEST_ASSERT(std::abs(tangent->max_abs_row_sum() - bound) <= 1e-12 * bound);

    std::vector<real_t> before(diag);
    SFEM_TEST_ASSERT(tangent->shift_diagonal(real_t(0.5), before.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(tangent->diagonal(diag.data()) == SFEM_SUCCESS);
    for (size_t i = 0; i < diag.size(); ++i) {
        SFEM_TEST_ASSERT(std::abs(diag[i] - (before[i] + real_t(0.5) * std::abs(before[i]))) <= 1e-14 * std::abs(before[i]));
    }
    return SFEM_TEST_SUCCESS;
}

// The operator is K on the free dofs and the identity on the constrained ones, and is the one
// `sfem::hessian_bsr` hands out.
int test_operator() {
    auto p = sfem::test::make_cube_problem("GeneratedNeoHookeanOgden", 3, real_t(0.1));
    deform(p);

    const ptrdiff_t n       = p.fs->n_dofs();
    auto            tangent = sfem::BSRTangent::create(p.f, sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(tangent->assemble(p.x->data()) == SFEM_SUCCESS);
    auto op  = tangent->op();
    auto api = sfem::hessian_bsr(p.f, p.x, sfem::EXECUTION_SPACE_HOST);

    std::vector<real_t> h(n), y(n, 0), z(n, 0), mf(n, 0);
    for (ptrdiff_t i = 0; i < n; ++i) h[i] = std::sin(real_t(0.37) * (i + 1));
    op->apply(h.data(), y.data());
    api->apply(h.data(), z.data());
    // Two assemblies, each summed with atomics in whatever order the threads take: equal to
    // rounding, not bitwise.
    real_t ymax = 0;
    for (ptrdiff_t i = 0; i < n; ++i) ymax = std::max(ymax, std::abs(y[i]));
    for (ptrdiff_t i = 0; i < n; ++i) SFEM_TEST_ASSERT(std::abs(y[i] - z[i]) <= 1e-13 * ymax);

    // Against the matrix-free tangent, on the free dofs.
    p.f->update(p.x->data());
    p.f->apply(p.x->data(), h.data(), mf.data());
    std::vector<mask_t> mask(mask_count(n), 0);
    p.f->constraints_mask(mask.data());
    real_t scale = 0, error = 0;
    for (ptrdiff_t i = 0; i < n; ++i) {
        if (mask_get(i, mask.data())) {
            SFEM_TEST_ASSERT(y[i] == h[i]);
            continue;
        }
        scale = std::max(scale, std::abs(mf[i]));
        error = std::max(error, std::abs(mf[i] - y[i]));
    }
    SFEM_TEST_ASSERT(error <= 1e-10 * scale);
    return SFEM_TEST_SUCCESS;
}

// On a level-2 semi-structured mesh, in both bases. Micro: the generated operator takes each macro
// element as its eight trilinear micro elements, as the hand-written one does, and the two assemble
// the same matrix on the micro-element graph. High-order: the generated element spans the macro
// element and needs every macro node coupled -- on the micro-element graph it lost those couplings
// and the product was 9% off the action.
int test_semi_structured_mesh() {
    auto box  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 1, 0, 0, 0, 1, real_t(0.7), real_t(0.5));
    auto mesh = smesh::to_semistructured(2, box, true, false);

    std::vector<real_t> h;
    auto assemble = [&](const std::shared_ptr<sfem::FunctionSpace> &fs, const char *name, std::vector<real_t> &y) -> int {
        const ptrdiff_t n  = fs->n_dofs();
        auto            op = sfem::create_op(fs, name, sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(op != nullptr && op->initialize() == SFEM_SUCCESS);
        auto f = sfem::Function::create(fs);
        f->add_operator(op);
        auto tangent = sfem::BSRTangent::create(f, sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(tangent->assemble(nullptr) == SFEM_SUCCESS);
        h.resize(n);
        for (ptrdiff_t i = 0; i < n; ++i) h[i] = std::sin(real_t(0.37) * (i + 1));
        y.assign(n, 0);
        std::vector<real_t> mf(n, 0);
        SFEM_TEST_ASSERT(tangent->op()->apply(h.data(), y.data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(f->apply(nullptr, h.data(), mf.data()) == SFEM_SUCCESS);
        real_t scale = 0, error = 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            scale = std::max(scale, std::abs(mf[i]));
            error = std::max(error, std::abs(mf[i] - y[i]));
        }
        printf("  %s, %s basis: %td nonzero blocks, |K h - J h| / |J h| = %.3g\n",
               name, fs->micro_elements() ? "micro" : "high-order", (ptrdiff_t)fs->element_node_graph()->nnz(),
               (double)(error / scale));
        // The hand-written operator's element Jacobians are single precision (jacobian_t).
        SFEM_TEST_ASSERT(error <= (std::string(name) == "LinearElasticity" ? 1e-6 : 1e-12) * scale);
        return SFEM_TEST_SUCCESS;
    };

    auto micro = sfem::FunctionSpace::create(mesh, 3);
    SFEM_TEST_ASSERT(micro->micro_elements() && micro->element_node_graph() == micro->node_to_node_graph());
    std::vector<real_t> generated, hand;
    SFEM_TEST_ASSERT(assemble(micro, "GeneratedLinearElasticity", generated) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(assemble(micro, "LinearElasticity", hand) == SFEM_TEST_SUCCESS);
    real_t scale = 0, error = 0;
    for (size_t i = 0; i < hand.size(); ++i) {
        scale = std::max(scale, std::abs(hand[i]));
        error = std::max(error, std::abs(hand[i] - generated[i]));
    }
    printf("  micro: generated against hand-written %.3g\n", (double)(error / scale));
    SFEM_TEST_ASSERT(error <= 1e-6 * scale);

    auto high = sfem::FunctionSpace::create(mesh, 3);
    high->set_semi_structured_basis(sfem::FunctionSpace::SemiStructuredBasis::HIGH_ORDER);
    SFEM_TEST_ASSERT(high->node_to_node_graph()->nnz() < high->element_node_graph()->nnz());
    std::vector<real_t> quadratic;
    SFEM_TEST_ASSERT(assemble(high, "GeneratedLinearElasticity", quadratic) == SFEM_TEST_SUCCESS);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_diagonal_and_row_sums);
    SFEM_RUN_TEST(test_operator);
    SFEM_RUN_TEST(test_semi_structured_mesh);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
