// The generated linear Kelvin-Voigt material against the hand-written `KelvinVoigtNewmark` it
// replaces, and its merit against its residual.
//
// The hand-written operator's law is sigma = K tr(eps) I + k dev(eps) + eta dev(eps_dot), and it
// takes the velocity and the acceleration as fields with Newmark's dt, beta and gamma as its own
// parameters. The generated one holds the scheme, with the standard parameters: mu = k/2,
// lmbda = K - k/3, eta_s = eta/2, eta_b = 0.
//
// On a semi-structured mesh they are the same discretisation in the micro-element basis -- each
// level-2 macro element as eight trilinear micro elements -- and different ones in the high-order
// basis, where the generated operator is one quadratic element on the same 27 nodes.

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_KelvinVoigtNewmark.hpp"
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
        for (const auto &block : fs->mesh_ptr()->blocks()) {
            op->set_value_in_block(block->name(), "mu", k / 2);
            op->set_value_in_block(block->name(), "lmbda", K - k / 3);
            op->set_value_in_block(block->name(), "eta_s", eta / 2);
            op->set_value_in_block(block->name(), "eta_b", 0);
        }
        return op;
    }

    // Without inertia, where the two masses (the scheme's lumped, the hand-written consistent)
    // cannot differ: the residual and the tangent action, at the same state and the velocity and
    // acceleration Newmark implies there.
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

        auto hand = std::dynamic_pointer_cast<sfem::KelvinVoigtNewmark>(
                sfem::create_op(fs, "KelvinVoigtNewmark", sfem::EXECUTION_SPACE_HOST));
        SFEM_TEST_ASSERT(hand != nullptr);
        SFEM_TEST_ASSERT(hand->initialize() == SFEM_SUCCESS);
        hand->set_k(k);
        hand->set_K(K);
        hand->set_eta(eta);
        hand->set_dt(dt);
        hand->set_beta(real_t(0.25));
        hand->set_gamma(real_t(0.5));
        hand->set_rho(0);
        hand->set_field("velocity", v, 0);
        hand->set_field("acceleration", a, 0);

        auto g_gen  = sfem::create_host_buffer<real_t>(ndofs);
        auto g_hand = sfem::create_host_buffer<real_t>(ndofs);
        SFEM_TEST_ASSERT(gen->gradient(x->data(), g_gen->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(hand->gradient(x->data(), g_hand->data()) == SFEM_SUCCESS);

        auto h_gen  = sfem::create_host_buffer<real_t>(ndofs);
        auto h_hand = sfem::create_host_buffer<real_t>(ndofs);
        SFEM_TEST_ASSERT(gen->apply(x->data(), d->data(), h_gen->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(hand->apply(x->data(), d->data(), h_hand->data()) == SFEM_SUCCESS);

        const real_t eg = worst_relative(ndofs, g_gen->data(), g_hand->data());
        const real_t eh = worst_relative(ndofs, h_gen->data(), h_hand->data());
        printf("  %s, %td dofs: residual %.3g, tangent action %.3g\n",
               label,
               ndofs,
               (double)eg,
               (double)eh);
        // The hand-written operator stores its element Jacobians as jacobian_t, single precision.
        // On a semi-structured mesh its gradient is not the residual its action linearises (it is
        // 30% off K x at zero velocity), so there only the action is compared.
        if (!fs->has_semi_structured_mesh()) SFEM_TEST_ASSERT(eg < 1e-6);
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

// The material says it is potential-based, so the merit a line search minimises is a potential:
// its derivative along d is the residual's projection on d, with the scheme's inertia in both.
// The potential is quadratic, so a central difference is exact up to round-off.
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
    SFEM_TEST_ASSERT(op->energy_or_potential_based());
    scheme->begin_step(0, dt);

    auto f = sfem::Function::create(fs);
    f->add_operator(op);

    auto x = sfem::create_host_buffer<real_t>(ndofs);
    auto d = sfem::create_host_buffer<real_t>(ndofs);
    auto g = sfem::create_host_buffer<real_t>(ndofs);
    seed(ndofs, 5, x->data());
    seed(ndofs, 6, d->data());
    SFEM_TEST_ASSERT(f->gradient(x->data(), g->data()) == SFEM_SUCCESS);
    real_t slope = 0;
    for (ptrdiff_t i = 0; i < ndofs; ++i) slope += g->data()[i] * d->data()[i];

    const real_t eps = real_t(1e-3), steps[2] = {eps, -eps};
    real_t       values[2] = {0, 0};
    SFEM_TEST_ASSERT(f->energy_merit(x->data(), d->data(), 2, steps, values) == SFEM_SUCCESS);
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
