#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_InertiaPotential.hpp"
#include "sfem_Function.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

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

    sfem::InertiaPotential op(space);
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

#ifdef SFEM_ENABLE_CUDA
/// The device answers what the host answers.
///
/// Everything this operator does to a vector goes through
/// `sfem::blas<real_t>(es)`, and the one thing that cannot -- finding a row's
/// diagonal entry -- has a kernel of its own.  Neither is worth anything unless
/// the two sides agree, so this runs the same problem on both and compares.
int test_inertia_potential_device_matches_host() {
    auto mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 2, 2, 2);
    auto space = sfem::FunctionSpace::create(mesh, 3);

    const ptrdiff_t ndofs = space->n_dofs();
    auto            x     = sfem::create_host_buffer<real_t>(ndofs);
    auto            h     = sfem::create_host_buffer<real_t>(ndofs);
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        x->data()[i] = real_t(0.01) * ((i % 11) - 5);
        h->data()[i] = real_t(0.005) * ((i % 7) - 3);
    }

    sfem::InertiaPotential host_op(space, sfem::EXECUTION_SPACE_HOST);
    sfem::InertiaPotential device_op(space, sfem::EXECUTION_SPACE_DEVICE);
    host_op.set_density(2.5);
    device_op.set_density(2.5);
    SFEM_TEST_ASSERT(host_op.initialize() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(device_op.initialize() == SFEM_SUCCESS);
    host_op.set_alpha(3.25);
    device_op.set_alpha(3.25);
    SFEM_TEST_ASSERT(device_op.execution_space() == sfem::EXECUTION_SPACE_DEVICE);

    auto d_x = smesh::to_device(x);
    auto d_h = smesh::to_device(h);

    // The 0-form, which is a weighted reduction on both sides.
    real_t host_value = 0, device_value = 0;
    SFEM_TEST_ASSERT(host_op.value(x->data(), &host_value) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(device_op.value(d_x->data(), &device_value) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(std::abs(host_value - device_value) <= 1e-10 * std::abs(host_value) + 1e-14);

    // The gradient and the action, compared component by component.
    auto host_g   = sfem::create_host_buffer<real_t>(ndofs);
    auto host_ah  = sfem::create_host_buffer<real_t>(ndofs);
    auto host_dia = sfem::create_host_buffer<real_t>(ndofs);
    SFEM_TEST_ASSERT(host_op.gradient(x->data(), host_g->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(host_op.apply(nullptr, h->data(), host_ah->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(host_op.hessian_diag(nullptr, host_dia->data()) == SFEM_SUCCESS);

    auto d_g   = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_DEVICE);
    auto d_ah  = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_DEVICE);
    auto d_dia = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_DEVICE);
    SFEM_TEST_ASSERT(device_op.gradient(d_x->data(), d_g->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(device_op.apply(nullptr, d_h->data(), d_ah->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(device_op.hessian_diag(nullptr, d_dia->data()) == SFEM_SUCCESS);

    auto back_g   = smesh::to_host(d_g);
    auto back_ah  = smesh::to_host(d_ah);
    auto back_dia = smesh::to_host(d_dia);
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        SFEM_TEST_ASSERT(std::abs(host_g->data()[i] - back_g->data()[i]) <= 1e-12);
        SFEM_TEST_ASSERT(std::abs(host_ah->data()[i] - back_ah->data()[i]) <= 1e-12);
        SFEM_TEST_ASSERT(std::abs(host_dia->data()[i] - back_dia->data()[i]) <= 1e-12);
    }

    // And the line-search 0-form, which is the reduction once per step.
    const real_t        steps[] = {real_t(0), real_t(-0.5), real_t(0.25)};
    std::vector<real_t> host_steps(3, 0), device_steps(3, 0);
    SFEM_TEST_ASSERT(host_op.value_steps(x->data(), h->data(), 3, steps, host_steps.data()) ==
                     SFEM_SUCCESS);
    SFEM_TEST_ASSERT(device_op.value_steps(d_x->data(), d_h->data(), 3, steps, device_steps.data()) ==
                     SFEM_SUCCESS);
    for (int s = 0; s < 3; ++s) {
        SFEM_TEST_ASSERT(std::abs(host_steps[s] - device_steps[s]) <=
                         1e-10 * std::abs(host_steps[s]) + 1e-14);
    }
    // The two sparse assemblies, which are the only part of this operator that
    // is not a vector operation and so the only part with a kernel of its own.
    // A pattern with exactly one entry per row -- its diagonal -- exercises the
    // kernel without needing a real sparsity graph, and the answer is one the
    // test can state outright.
    {
        const int       bs      = space->block_size();
        const ptrdiff_t n_nodes = ndofs / bs;

        auto rowptr = sfem::create_host_buffer<count_t>(ndofs + 1);
        auto colidx = sfem::create_host_buffer<idx_t>(ndofs);
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            rowptr->data()[i] = (count_t)i;
            colidx->data()[i] = (idx_t)i;
        }
        rowptr->data()[ndofs] = (count_t)ndofs;

        auto host_crs = sfem::create_host_buffer<real_t>(ndofs);
        SFEM_TEST_ASSERT(host_op.hessian_crs(nullptr, rowptr->data(), colidx->data(), host_crs->data()) ==
                         SFEM_SUCCESS);

        auto d_rowptr = smesh::to_device(rowptr);
        auto d_colidx = smesh::to_device(colidx);
        auto d_crs    = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_DEVICE);
        SFEM_TEST_ASSERT(device_op.hessian_crs(
                                 nullptr, d_rowptr->data(), d_colidx->data(), d_crs->data()) ==
                         SFEM_SUCCESS);
        auto back_crs = smesh::to_host(d_crs);
        SFEM_TEST_ASSERT(host_crs->data()[0] != real_t(0));
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            SFEM_TEST_ASSERT(std::abs(host_crs->data()[i] - back_crs->data()[i]) <= 1e-12);
        }

        auto brow = sfem::create_host_buffer<count_t>(n_nodes + 1);
        auto bcol = sfem::create_host_buffer<idx_t>(n_nodes);
        for (ptrdiff_t node = 0; node < n_nodes; ++node) {
            brow->data()[node] = (count_t)node;
            bcol->data()[node] = (idx_t)node;
        }
        brow->data()[n_nodes] = (count_t)n_nodes;

        auto host_bsr = sfem::create_host_buffer<real_t>(n_nodes * bs * bs);
        SFEM_TEST_ASSERT(host_op.hessian_bsr(nullptr, brow->data(), bcol->data(), host_bsr->data()) ==
                         SFEM_SUCCESS);

        auto d_brow = smesh::to_device(brow);
        auto d_bcol = smesh::to_device(bcol);
        auto d_bsr  = sfem::create_buffer<real_t>(n_nodes * bs * bs, sfem::EXECUTION_SPACE_DEVICE);
        SFEM_TEST_ASSERT(device_op.hessian_bsr(
                                 nullptr, d_brow->data(), d_bcol->data(), d_bsr->data()) ==
                         SFEM_SUCCESS);
        auto back_bsr = smesh::to_host(d_bsr);
        SFEM_TEST_ASSERT(host_bsr->data()[0] != real_t(0));
        for (ptrdiff_t k = 0; k < n_nodes * bs * bs; ++k) {
            SFEM_TEST_ASSERT(std::abs(host_bsr->data()[k] - back_bsr->data()[k]) <= 1e-12);
        }
    }

    // The block-diagonal symmetric format, which is the third kernel.
    {
        const int       bs      = space->block_size();
        const ptrdiff_t n_nodes = ndofs / bs;
        const int       packed  = bs * (bs + 1) / 2;

        auto host_sym = sfem::create_host_buffer<real_t>(n_nodes * packed);
        SFEM_TEST_ASSERT(host_op.hessian_block_diag_sym(nullptr, host_sym->data()) == SFEM_SUCCESS);

        auto d_sym = sfem::create_buffer<real_t>(n_nodes * packed, sfem::EXECUTION_SPACE_DEVICE);
        SFEM_TEST_ASSERT(device_op.hessian_block_diag_sym(nullptr, d_sym->data()) == SFEM_SUCCESS);
        auto back_sym = smesh::to_host(d_sym);

        SFEM_TEST_ASSERT(host_sym->data()[0] != real_t(0));
        for (ptrdiff_t k = 0; k < n_nodes * packed; ++k) {
            SFEM_TEST_ASSERT(std::abs(host_sym->data()[k] - back_sym->data()[k]) <= 1e-12);
        }
    }

    return SFEM_TEST_SUCCESS;
}
#endif  // SFEM_ENABLE_CUDA

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_bdf2_inertia_potential_derivatives);
#ifdef SFEM_ENABLE_CUDA
    SFEM_RUN_TEST(test_inertia_potential_device_matches_host);
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
