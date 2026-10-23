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

// The micro basis on a distorted mesh, where a macro element is not a parallelepiped and its micro
// elements differ, against the same micro elements as a plain HEX8 mesh: the generated operator,
// and the hand-written one on its isoparametric route. Its default takes one macro Jacobian per
// element (`ASSUME_AFFINE`), exact only on the box above.
int test_semi_structured_mesh_non_affine() {
    auto box = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 2, 0, 0, 0, 1, 1, 1);
    {
        geom_t *const *const xyz = box->points()->data();
        for (ptrdiff_t i = 0; i < box->n_nodes(); ++i) {
            const geom_t x = xyz[0][i], y = xyz[1][i], z = xyz[2][i];
            xyz[0][i] += geom_t(0.15) * y * z;
            xyz[1][i] += geom_t(0.1) * x * x;
            xyz[2][i] -= geom_t(0.12) * x * y;
        }
    }
    auto ss = smesh::to_semistructured(2, box, true, false);
    // Written out and read back as HEX8, with the same node numbering.
    SFEM_TEST_ASSERT(smesh::semistructured_export_as_standard(ss, smesh::Path("bsr_tangent_non_affine_hex8")) == SFEM_SUCCESS);
    auto hex8 = sfem::Mesh::create_from_file(sfem::Communicator::self(), smesh::Path("bsr_tangent_non_affine_hex8"));
    SFEM_TEST_ASSERT(hex8 && hex8->n_nodes() == ss->n_nodes());

    auto            fs      = sfem::FunctionSpace::create(ss, 3);
    auto            fs_hex8 = sfem::FunctionSpace::create(hex8, 3);
    const ptrdiff_t n       = fs->n_dofs();

    std::vector<real_t> h(n);
    for (ptrdiff_t i = 0; i < n; ++i) h[i] = std::sin(real_t(0.37) * (i + 1));
    std::vector<real_t> blocks, diagonal;
    auto action = [&](const std::shared_ptr<sfem::FunctionSpace> &space, const char *name, std::vector<real_t> &y) -> int {
        auto op = sfem::create_op(space, name, sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(op != nullptr);
        // The hand-written operator on its isoparametric route, set before initialize, which builds
        // the per-element Jacobians only for the affine one.
        op->set_option("ASSUME_AFFINE", false);
        SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);
        y.assign(n, 0);
        SFEM_TEST_ASSERT(op->apply(nullptr, h.data(), y.data()) == SFEM_SUCCESS);
        // And its diagonal blocks, packed 00 01 02 11 12 22 per node, and its diagonal.
        blocks.assign(n / 3 * 6, 0);
        SFEM_TEST_ASSERT(op->hessian_block_diag_sym(nullptr, blocks.data()) == SFEM_SUCCESS);
        diagonal.assign(n, 0);
        if (std::string(name) == "LinearElasticity") {
            SFEM_TEST_ASSERT(op->hessian_diag(nullptr, diagonal.data()) == SFEM_SUCCESS);
        }
        return SFEM_TEST_SUCCESS;
    };
    std::vector<real_t> micro, hand_micro, generated_hex8, hand_hex8;
    SFEM_TEST_ASSERT(action(fs, "GeneratedLinearElasticity", micro) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(action(fs, "LinearElasticity", hand_micro) == SFEM_TEST_SUCCESS);
    const std::vector<real_t> hand_micro_blocks = blocks, hand_micro_diagonal = diagonal;
    SFEM_TEST_ASSERT(action(fs_hex8, "GeneratedLinearElasticity", generated_hex8) == SFEM_TEST_SUCCESS);
    const std::vector<real_t> reference_blocks = blocks;
    SFEM_TEST_ASSERT(action(fs_hex8, "LinearElasticity", hand_hex8) == SFEM_TEST_SUCCESS);

    real_t block_scale = 0, e_blocks = 0, e_diagonal = 0;
    for (ptrdiff_t i = 0; i < n / 3; ++i) {
        for (int k = 0; k < 6; ++k) {
            block_scale = std::max(block_scale, std::abs(reference_blocks[i * 6 + k]));
            e_blocks    = std::max(e_blocks, std::abs(hand_micro_blocks[i * 6 + k] - reference_blocks[i * 6 + k]));
        }
        const int packed_diagonal[3] = {0, 3, 5};
        for (int d = 0; d < 3; ++d) {
            e_diagonal = std::max(e_diagonal, std::abs(hand_micro_diagonal[i * 3 + d] - reference_blocks[i * 6 + packed_diagonal[d]]));
        }
    }
    printf("  distorted: hand-written micro diagonal blocks against generated HEX8 %.3g, diagonal %.3g\n",
           (double)(e_blocks / block_scale), (double)(e_diagonal / block_scale));
    SFEM_TEST_ASSERT(e_blocks <= 1e-6 * block_scale);
    SFEM_TEST_ASSERT(e_diagonal <= 1e-6 * block_scale);

    real_t scale = 0, e_micro = 0, e_hand_micro = 0, e_hand = 0;
    for (ptrdiff_t i = 0; i < n; ++i) {
        scale        = std::max(scale, std::abs(generated_hex8[i]));
        e_micro      = std::max(e_micro, std::abs(micro[i] - generated_hex8[i]));
        e_hand_micro = std::max(e_hand_micro, std::abs(hand_micro[i] - generated_hex8[i]));
        e_hand       = std::max(e_hand, std::abs(hand_hex8[i] - generated_hex8[i]));
    }
    printf("  distorted, %td dofs, against generated HEX8: generated micro %.3g, hand-written micro %.3g, "
           "hand-written HEX8 %.3g\n",
           n, (double)(e_micro / scale), (double)(e_hand_micro / scale), (double)(e_hand / scale));
    SFEM_TEST_ASSERT(e_micro <= 1e-12 * scale);
    SFEM_TEST_ASSERT(e_hand_micro <= 1e-6 * scale);
    SFEM_TEST_ASSERT(e_hand <= 1e-6 * scale);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_diagonal_and_row_sums);
    SFEM_RUN_TEST(test_operator);
    SFEM_RUN_TEST(test_semi_structured_mesh);
    SFEM_RUN_TEST(test_semi_structured_mesh_non_affine);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
