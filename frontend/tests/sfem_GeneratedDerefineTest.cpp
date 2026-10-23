// A generated material on the coarser levels of a semi-structured hierarchy (what SSMGC and SSGMG
// build with Function::derefine): the same material on the coarse micro-element space, and a held
// time scheme seen from there -- the fine level's shift and alpha, the mass rediscretised.

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_InertiaPotential.hpp"
#include "sfem_NewmarkScheme.hpp"
#include "sfem_test.hpp"

namespace {
    real_t worst_relative(const std::vector<real_t> &a, const std::vector<real_t> &b) {
        real_t worst = 0, scale = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            worst = std::max(worst, std::abs(a[i] - b[i]));
            scale = std::max(scale, std::abs(b[i]));
        }
        return scale > 0 ? worst / scale : 1;
    }

    std::shared_ptr<sfem::Op> material(const std::shared_ptr<sfem::FunctionSpace> &fs) {
        auto op = sfem::create_op(fs, "GeneratedLinearElasticity", sfem::EXECUTION_SPACE_HOST);
        if (!op || op->initialize() != SFEM_SUCCESS) return nullptr;
        for (const auto &block : fs->mesh_ptr()->blocks()) {
            op->set_value_in_block(block->name(), "mu", 3);
            op->set_value_in_block(block->name(), "lmbda", 2);
        }
        return op;
    }
}  // namespace

// Level 4 to 2 to 1 (HEX8): every coarse level's action against the material built there directly,
// plus the coarse inertia at the fine level's alpha.
int test_derefined_material_holds_the_coarse_scheme() {
    auto box   = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 1, 0, 0, 0, 1, real_t(0.7), real_t(0.5));
    auto fine  = sfem::FunctionSpace::create(smesh::to_semistructured(4, box, true, false), 3);
    SFEM_TEST_ASSERT(fine->micro_elements());

    auto scheme = std::make_shared<sfem::NewmarkScheme>(fine);
    scheme->set_density(2);
    SFEM_TEST_ASSERT(scheme->initialize() == SFEM_SUCCESS);
    auto op = material(fine);
    SFEM_TEST_ASSERT(op != nullptr);
    std::dynamic_pointer_cast<sfem::TimeSteppable>(op)->set_time_scheme(scheme);
    scheme->begin_step(0, real_t(0.01));
    const real_t alpha = scheme->weight("alpha_a");

    auto f = sfem::Function::create(fine);
    f->add_operator(op);

    for (const int level : {2, 1}) {
        auto coarse   = fine->derefine(level);
        auto derefined = f->derefine(coarse, true);
        SFEM_TEST_ASSERT(derefined != nullptr);

        auto reference = sfem::Function::create(coarse);
        reference->add_operator(material(coarse));
        auto inertia = std::make_shared<sfem::InertiaPotential>(coarse);
        inertia->set_density(2);
        inertia->set_alpha(alpha);
        SFEM_TEST_ASSERT(inertia->initialize() == SFEM_SUCCESS);
        reference->add_operator(inertia);

        const ptrdiff_t     n = coarse->n_dofs();
        std::vector<real_t> h(n), a(n, 0), b(n, 0);
        for (ptrdiff_t i = 0; i < n; ++i) h[i] = std::sin(real_t(0.37) * (i + 1));
        SFEM_TEST_ASSERT(derefined->apply(nullptr, h.data(), a.data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(reference->apply(nullptr, h.data(), b.data()) == SFEM_SUCCESS);
        const real_t e = worst_relative(a, b);
        printf("  level %d, %td dofs: derefined against built there %.3g\n", level, n, (double)e);
        SFEM_TEST_ASSERT(e < 1e-12);

        // The coarse scheme follows the fine one's alpha.
        scheme->begin_step(real_t(0.01), real_t(0.02));
        inertia->set_alpha(scheme->weight("alpha_a"));
        std::fill(a.begin(), a.end(), 0);
        std::fill(b.begin(), b.end(), 0);
        SFEM_TEST_ASSERT(derefined->apply(nullptr, h.data(), a.data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(reference->apply(nullptr, h.data(), b.data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(worst_relative(a, b) < 1e-12);
        scheme->begin_step(0, real_t(0.01));
    }
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_derefined_material_holds_the_coarse_scheme);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
