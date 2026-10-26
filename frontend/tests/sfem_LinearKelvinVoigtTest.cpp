// The generated linear Kelvin-Voigt material against the hand-written linear elasticity, and its
// merit against its residual.
//
// The law is sigma = K tr(eps) I + k dev(eps) + eta dev(eps_dot): in the generated material's
// parameters mu = k/2, lmbda = K - k/3, eta_s = eta/2, eta_b = 0. Both stresses are linear
// elasticity: the elastic one with (k/2, K - k/3) in the state, the viscous one with
// (eta/2, -eta/3) in the velocity the scheme implies, whose derivative in the state is its shift.
//
// On a semi-structured mesh the micro-element basis is the hand-written discretisation -- each
// level-2 macro element as eight trilinear micro elements -- and the high-order basis a different
// one, where the generated operator is one quadratic element on the same 27 nodes.

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_NewmarkScheme.hpp"
#include "sfem_test.hpp"

namespace {
    constexpr real_t k = 4, K = 8, eta = real_t(0.05), dt = real_t(0.01);

    void seed(const ptrdiff_t n, const int salt, real_t *const values) {
        for (ptrdiff_t i = 0; i < n; ++i) values[i] = real_t(1e-3) * std::sin(real_t(0.37 * (i + 1) + salt));
    }

    real_t worst_relative(const ptrdiff_t n, const real_t *const a, const real_t *const b) {
        real_t worst = 0, scale = 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            worst = std::max(worst, std::abs(a[i] - b[i]));
            scale = std::max(scale, std::abs(b[i]));
        }
        return scale > 0 ? worst / scale : 1;
    }

    std::shared_ptr<sfem::Op> generated(const std::shared_ptr<sfem::FunctionSpace> &fs,
                                        const std::shared_ptr<sfem::NewmarkScheme> &scheme) {
        auto op = sfem::create_op(fs, "GeneratedLinearKelvinVoigt", sfem::EXECUTION_SPACE_HOST);
        std::dynamic_pointer_cast<sfem::TimeSteppable>(op)->set_time_scheme(scheme);
        // After initialize, which sets every block to the defaults.
        if (op->initialize() != SFEM_SUCCESS) return nullptr;
        for (const auto &block : fs->mesh()->blocks()) {
            op->set_value_in_block(block->name(), "mu", k / 2);
            op->set_value_in_block(block->name(), "lmbda", K - k / 3);
            op->set_value_in_block(block->name(), "eta_s", eta / 2);
            op->set_value_in_block(block->name(), "eta_b", 0);
        }
        return op;
    }

    std::shared_ptr<sfem::Op> hand_written(const std::shared_ptr<sfem::FunctionSpace> &fs, const real_t mu, const real_t lambda) {
        auto op = sfem::create_op(fs, "LinearElasticity", sfem::EXECUTION_SPACE_HOST);
        if (!op || op->initialize() != SFEM_SUCCESS) return nullptr;
        for (const auto &block : fs->mesh()->blocks()) {
            op->set_value_in_block(block->name(), "mu", mu);
            op->set_value_in_block(block->name(), "lambda", lambda);
        }
        return op;
    }

    // Without inertia: the residual and the tangent action, at the same state and the velocity
    // Newmark implies there.
    int compare(const char *const label, const std::shared_ptr<sfem::Mesh> &mesh) {
        auto            fs    = sfem::FunctionSpace::create(mesh, 3);
        const ptrdiff_t ndofs = fs->n_dofs();

        auto scheme = std::make_shared<sfem::NewmarkScheme>(fs);
        scheme->set_density(0);
        SFEM_TEST_ASSERT(scheme->initialize() == SFEM_SUCCESS);
        seed(ndofs, 0, scheme->state()->data());
        seed(ndofs, 1, scheme->velocity()->data());
        seed(ndofs, 2, scheme->acceleration()->data());
        auto gen = generated(fs, scheme);
        SFEM_TEST_ASSERT(gen != nullptr);
        scheme->begin_step(0, dt);

        auto x = sfem::create_host_buffer<real_t>(ndofs);
        auto d = sfem::create_host_buffer<real_t>(ndofs);
        auto v = sfem::create_host_buffer<real_t>(ndofs);
        auto a = sfem::create_host_buffer<real_t>(ndofs);
        seed(ndofs, 5, x->data());
        seed(ndofs, 6, d->data());
        scheme->reconstruct(x->data(), v->data(), a->data());

        auto elastic = hand_written(fs, k / 2, K - k / 3);
        auto viscous = hand_written(fs, eta / 2, -eta / 3);
        SFEM_TEST_ASSERT(elastic != nullptr && viscous != nullptr);

        auto g_gen  = sfem::create_host_buffer<real_t>(ndofs);
        auto g_hand = sfem::create_host_buffer<real_t>(ndofs);
        SFEM_TEST_ASSERT(gen->gradient(x->data(), g_gen->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(elastic->gradient(x->data(), g_hand->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(viscous->gradient(v->data(), g_hand->data()) == SFEM_SUCCESS);

        auto shifted = sfem::create_host_buffer<real_t>(ndofs);
        for (ptrdiff_t i = 0; i < ndofs; ++i) shifted->data()[i] = scheme->shift() * d->data()[i];

        auto h_gen  = sfem::create_host_buffer<real_t>(ndofs);
        auto h_hand = sfem::create_host_buffer<real_t>(ndofs);
        SFEM_TEST_ASSERT(gen->apply(x->data(), d->data(), h_gen->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(elastic->apply(x->data(), d->data(), h_hand->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(viscous->apply(v->data(), shifted->data(), h_hand->data()) == SFEM_SUCCESS);

        const real_t eg = worst_relative(ndofs, g_gen->data(), g_hand->data());
        const real_t eh = worst_relative(ndofs, h_gen->data(), h_hand->data());
        printf("  %s, %td dofs: residual %.3g, tangent action %.3g\n",
               label,
               ndofs,
               (double)eg,
               (double)eh);
        SFEM_TEST_ASSERT(eg < 1e-6);
        SFEM_TEST_ASSERT(eh < 1e-6);
        return SFEM_TEST_SUCCESS;
    }
}  // namespace

int test_matches_the_hand_written_operator_on_hex8() {
    return compare("HEX8", sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 3, 2, 2, 0, 0, 0, 1, real_t(0.7), real_t(0.5)));
}

// The assembled tangent, the scheme's mass included, is the action's: on HEX8 and on the level-2
// semi-structured mesh in both bases.
int test_the_assembled_tangent_is_the_action() {
    auto box = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 1, 0, 0, 0, 1, real_t(0.7), real_t(0.5));
    auto ss = smesh::to_semistructured(2, box, true, false);
    for (auto [mesh, high_order] : {std::pair{box, false}, std::pair{ss, false}, std::pair{ss, true}}) {
        auto fs = sfem::FunctionSpace::create(mesh, 3);
        if (high_order) fs->set_semi_structured_basis(sfem::FunctionSpace::SemiStructuredBasis::HIGH_ORDER);
        const ptrdiff_t ndofs = fs->n_dofs();
        const int       bs    = 3;

        auto scheme = std::make_shared<sfem::NewmarkScheme>(fs);
        scheme->set_density(2);
        SFEM_TEST_ASSERT(scheme->initialize() == SFEM_SUCCESS);
        auto op = generated(fs, scheme);
        SFEM_TEST_ASSERT(op != nullptr);
        scheme->begin_step(0, dt);

        auto x = sfem::create_host_buffer<real_t>(ndofs);
        auto d = sfem::create_host_buffer<real_t>(ndofs);
        seed(ndofs, 5, x->data());
        seed(ndofs, 6, d->data());

        auto            graph  = fs->element_node_graph();
        const count_t  *rowptr = graph->rowptr()->data();
        const idx_t    *colidx = graph->colidx()->data();
        auto            values = sfem::create_host_buffer<real_t>(graph->nnz() * bs * bs);
        SFEM_TEST_ASSERT(op->hessian_bsr(x->data(), rowptr, colidx, values->data()) == SFEM_SUCCESS);
        auto action = sfem::create_host_buffer<real_t>(ndofs);
        SFEM_TEST_ASSERT(op->apply(x->data(), d->data(), action->data()) == SFEM_SUCCESS);

        auto product = sfem::create_host_buffer<real_t>(ndofs);
        for (ptrdiff_t i = 0; i < ndofs / bs; ++i) {
            for (int r = 0; r < bs; ++r) {
                real_t sum = 0;
                for (count_t k = rowptr[i]; k < rowptr[i + 1]; ++k) {
                    for (int c = 0; c < bs; ++c) sum += values->data()[(k * bs + r) * bs + c] * d->data()[colidx[k] * bs + c];
                }
                product->data()[i * bs + r] = sum;
            }
        }
        const real_t e = worst_relative(ndofs, product->data(), action->data());
        printf("  %td dofs: |K d - J d| / |J d| = %.3g\n", ndofs, (double)e);
        SFEM_TEST_ASSERT(e < 1e-12);
    }
    return SFEM_TEST_SUCCESS;
}

int test_matches_the_hand_written_operator_on_sshex8() {
    auto box = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 1, 0, 0, 0, 1, real_t(0.7), real_t(0.5));
    return compare("SSHEX8 level 2, micro", smesh::to_semistructured(2, box, true, false));
}

// The generator cannot evaluate this material's potential, so the material declares none and the
// merit a line search minimises is the residual's, 1/2 ||R||^2: its derivative along d is R . J d,
// with the scheme's inertia in both. The residual is affine, so the merit is quadratic and a
// central difference is exact up to round-off. J d is `Function::apply`, which skips only the
// operators whose residual does not read the state: this one does.
int test_the_merit_differentiates_to_the_residual() {
    auto            mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 3, 2, 2);
    auto            fs    = sfem::FunctionSpace::create(mesh, 3);
    const ptrdiff_t ndofs = fs->n_dofs();

    auto scheme = std::make_shared<sfem::NewmarkScheme>(fs);
    scheme->set_density(2);
    SFEM_TEST_ASSERT(scheme->initialize() == SFEM_SUCCESS);
    seed(ndofs, 0, scheme->state()->data());
    seed(ndofs, 1, scheme->velocity()->data());
    seed(ndofs, 2, scheme->acceleration()->data());
    auto op = generated(fs, scheme);
    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_ASSERT(!op->energy_or_potential_based());
    SFEM_TEST_ASSERT(op->residual_depends_on_state());
    scheme->begin_step(0, dt);

    auto f = sfem::Function::create(fs);
    f->add_operator(op);
    SFEM_TEST_ASSERT(!f->has_energy_merit());

    auto x  = sfem::create_host_buffer<real_t>(ndofs);
    auto d  = sfem::create_host_buffer<real_t>(ndofs);
    auto g  = sfem::create_host_buffer<real_t>(ndofs);
    auto jd = sfem::create_host_buffer<real_t>(ndofs);
    seed(ndofs, 5, x->data());
    seed(ndofs, 6, d->data());
    SFEM_TEST_ASSERT(f->gradient(x->data(), g->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(f->apply(x->data(), d->data(), jd->data()) == SFEM_SUCCESS);
    real_t slope = 0;
    for (ptrdiff_t i = 0; i < ndofs; ++i) slope += g->data()[i] * jd->data()[i];

    const real_t eps = real_t(1e-3), steps[2] = {eps, -eps};
    real_t       values[2] = {0, 0};
    SFEM_TEST_ASSERT(f->residual_merit(x->data(), d->data(), 2, steps, values) == SFEM_SUCCESS);
    const real_t fd = (values[0] - values[1]) / (2 * eps);
    printf("  slope %.12g, central difference %.12g\n", (double)slope, (double)fd);
    SFEM_TEST_ASSERT(std::abs(slope) > 0);
    SFEM_TEST_ASSERT(std::abs(fd - slope) <= real_t(1e-8) * std::abs(slope));
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_matches_the_hand_written_operator_on_hex8);
    SFEM_RUN_TEST(test_matches_the_hand_written_operator_on_sshex8);
    SFEM_RUN_TEST(test_the_assembled_tangent_is_the_action);
    SFEM_RUN_TEST(test_the_merit_differentiates_to_the_residual);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
