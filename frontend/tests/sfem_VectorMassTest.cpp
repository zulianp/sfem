// The generated consistent vector mass: its row sums are the lumped mass, its total is the mass
// of the body, and its assembled matrix is its action -- on the elements a time scheme uses it on.

#include <algorithm>
#include <cmath>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_LumpedMass.hpp"
#include "sfem_test.hpp"

namespace {
    constexpr real_t rho = real_t(2.5);
    // The box's volume as the mesh stores it: its coordinates are geom_t.
    const real_t volume_of_box = real_t(geom_t(0.7)) * real_t(geom_t(0.5));

    int check(const char *const label, const std::shared_ptr<sfem::Mesh> &mesh, const real_t volume, const bool row_sums,
              const bool high_order = false) {
        auto fs = sfem::FunctionSpace::create(mesh, 3);
        if (high_order) fs->set_semi_structured_basis(sfem::FunctionSpace::SemiStructuredBasis::HIGH_ORDER);
        const ptrdiff_t n  = fs->n_dofs();
        auto            op = sfem::create_op(fs, "GeneratedVectorMass", sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(op != nullptr && op->initialize() == SFEM_SUCCESS);
        for (const auto &block : mesh->blocks()) op->set_value_in_block(block->name(), "density", rho);

        std::vector<real_t> ones(n, 1), m1(n, 0);
        SFEM_TEST_ASSERT(op->apply(nullptr, ones.data(), m1.data()) == SFEM_SUCCESS);
        real_t total = 0;
        for (auto v : m1) total += v;
        // Three components, each carrying the body's mass.
        SFEM_TEST_ASSERT(std::abs(total - 3 * rho * volume) <= 1e-12 * 3 * rho * volume);

        if (row_sums) {
            // The lumped mass is the row-summed consistent one.
            sfem::LumpedMass lumped(fs);
            SFEM_TEST_ASSERT(lumped.initialize() == SFEM_SUCCESS);
            std::vector<real_t> diag(n, 0);
            SFEM_TEST_ASSERT(lumped.hessian_diag(nullptr, diag.data()) == SFEM_SUCCESS);
            real_t worst = 0, scale = 0;
            for (ptrdiff_t i = 0; i < n; ++i) {
                worst = std::max(worst, std::abs(m1[i] - rho * diag[i]));
                scale = std::max(scale, std::abs(m1[i]));
            }
            // The semi-structured LumpedMass works from single-precision Jacobians (jacobian_t).
            SFEM_TEST_ASSERT(worst <= (fs->has_semi_structured_mesh() ? 1e-6 : 1e-12) * scale);
        }

        auto f = sfem::Function::create(fs);
        f->add_operator(op);
        auto                graph = fs->element_node_graph();
        std::vector<real_t> values(graph->nnz() * 9, 0), h(n), kh(n, 0), mh(n, 0);
        SFEM_TEST_ASSERT(op->hessian_bsr(nullptr, graph->rowptr()->data(), graph->colidx()->data(), values.data()) == SFEM_SUCCESS);
        for (ptrdiff_t i = 0; i < n; ++i) h[i] = std::sin(real_t(0.37) * (i + 1));
        SFEM_TEST_ASSERT(op->apply(nullptr, h.data(), mh.data()) == SFEM_SUCCESS);
        const count_t *rowptr = graph->rowptr()->data();
        const idx_t   *colidx = graph->colidx()->data();
        for (ptrdiff_t i = 0; i < n / 3; ++i) {
            for (count_t k = rowptr[i]; k < rowptr[i + 1]; ++k) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) kh[i * 3 + r] += values[(k * 3 + r) * 3 + c] * h[colidx[k] * 3 + c];
                }
            }
        }
        real_t worst = 0, scale = 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            worst = std::max(worst, std::abs(kh[i] - mh[i]));
            scale = std::max(scale, std::abs(mh[i]));
        }
        printf("  %s, %td dofs: total %.12g (expected %.12g), |K h - M h| / |M h| = %.3g\n",
               label, n, (double)total, (double)(3 * rho * volume), (double)(worst / scale));
        SFEM_TEST_ASSERT(worst <= 1e-12 * scale);
        return SFEM_TEST_SUCCESS;
    }
}  // namespace

int test_hex8() {
    return check("HEX8", sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 3, 2, 2, 0, 0, 0, 1, real_t(0.7), real_t(0.5)), volume_of_box, true);
}

int test_tet4() {
    return check("TET4", sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), 2, 2, 2, 0, 0, 0, 1, real_t(0.7), real_t(0.5)), volume_of_box, true);
}

// The micro elements, whose row sums are the (micro) lumped mass; and the quadratic element on the
// macro element, whose row sums are not a lumped mass but whose total still is the body's.
int test_semi_structured() {
    auto box  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 1, 0, 0, 0, 1, real_t(0.7), real_t(0.5));
    auto mesh = smesh::to_semistructured(2, box, true, false);
    SFEM_TEST_ASSERT(check("SSHEX8 level 2, micro", mesh, volume_of_box, true) == SFEM_TEST_SUCCESS);
    return check("SSHEX8 level 2, high-order", mesh, volume_of_box, false, true);
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_hex8);
    SFEM_RUN_TEST(test_tet4);
    SFEM_RUN_TEST(test_semi_structured);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
