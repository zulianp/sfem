#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_BDF2InertiaPotential.hpp"
#include "sfem_Function.hpp"

#include <cmath>
#include <cstdio>

namespace {

    real_t dot(const ptrdiff_t n, const real_t *const a, const real_t *const b) {
        real_t ret = 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            ret += a[i] * b[i];
        }
        return ret;
    }

}  // namespace

int test_bdf2_inertia_potential_derivatives() {
    auto mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 1, 1, 1);
    auto space = sfem::FunctionSpace::create(mesh, 3);

    const ptrdiff_t ndofs = space->n_dofs();
    auto            mass  = sfem::create_host_buffer<real_t>(ndofs);
    auto            u_hat = sfem::create_host_buffer<real_t>(ndofs);
    auto            x     = sfem::create_host_buffer<real_t>(ndofs);
    auto            h     = sfem::create_host_buffer<real_t>(ndofs);
    auto            g     = sfem::create_host_buffer<real_t>(ndofs);
    auto            g_p   = sfem::create_host_buffer<real_t>(ndofs);
    auto            g_m   = sfem::create_host_buffer<real_t>(ndofs);
    auto            ah    = sfem::create_host_buffer<real_t>(ndofs);
    auto            x_p   = sfem::create_host_buffer<real_t>(ndofs);
    auto            x_m   = sfem::create_host_buffer<real_t>(ndofs);

    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        mass->data()[i]  = 1 + real_t(0.125) * ((i % 5) + 1);
        u_hat->data()[i] = real_t(0.01) * ((i % 7) - 3);
        x->data()[i]     = real_t(0.02) * ((i % 11) - 5);
        h->data()[i]     = real_t(0.005) * ((i % 13) - 6);
    }

    sfem::BDF2InertiaPotential op(space);
    op.set_alpha(7.25);
    op.set_mass(mass);
    op.set_u_hat(u_hat);
    SFEM_TEST_ASSERT(op.initialize() == SFEM_SUCCESS);

    real_t value = 0;
    SFEM_TEST_ASSERT(op.value(x->data(), &value) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(value > 0);

    SFEM_TEST_ASSERT(op.gradient(x->data(), g->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op.apply(x->data(), h->data(), ah->data()) == SFEM_SUCCESS);

    const real_t eps = 1e-6;
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        x_p->data()[i] = x->data()[i] + eps * h->data()[i];
        x_m->data()[i] = x->data()[i] - eps * h->data()[i];
    }

    real_t v_p = 0;
    real_t v_m = 0;
    SFEM_TEST_ASSERT(op.value(x_p->data(), &v_p) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op.value(x_m->data(), &v_m) == SFEM_SUCCESS);
    const real_t fd_value = (v_p - v_m) / (2 * eps);
    const real_t gdoth    = dot(ndofs, g->data(), h->data());
    SFEM_TEST_ASSERT(std::abs(fd_value - gdoth) < 1e-8);

    SFEM_TEST_ASSERT(op.gradient(x_p->data(), g_p->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op.gradient(x_m->data(), g_m->data()) == SFEM_SUCCESS);
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        const real_t fd_apply = (g_p->data()[i] - g_m->data()[i]) / (2 * eps);
        SFEM_TEST_ASSERT(std::abs(fd_apply - ah->data()[i]) < 1e-8);
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_bdf2_inertia_potential_derivatives);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
