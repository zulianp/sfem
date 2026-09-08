#include "sfem_test.hpp"

#include <algorithm>
#include <cmath>

#include "sfem_API.hpp"
#include "sfem_NeumannConditions.hpp"

namespace {

    real_t dot(const ptrdiff_t n, const real_t *const a, const real_t *const b) {
        real_t value = 0;
        for (ptrdiff_t i = 0; i < n; ++i) value += a[i] * b[i];
        return value;
    }

    int verify_derivatives(const std::shared_ptr<sfem::NeumannConditions> &op,
                           const std::shared_ptr<sfem::FunctionSpace>     &space) {
        const ptrdiff_t ndofs  = space->n_dofs();
        const ptrdiff_t nnodes = space->mesh_ptr()->n_nodes();
        auto            x      = sfem::create_host_buffer<real_t>(ndofs);
        auto            h      = sfem::create_host_buffer<real_t>(ndofs);
        auto            xp     = sfem::create_host_buffer<real_t>(ndofs);
        auto            xm     = sfem::create_host_buffer<real_t>(ndofs);
        auto            g      = sfem::create_host_buffer<real_t>(ndofs);
        auto            gp     = sfem::create_host_buffer<real_t>(ndofs);
        auto            gm     = sfem::create_host_buffer<real_t>(ndofs);
        auto            ah     = sfem::create_host_buffer<real_t>(ndofs);
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            x->data()[i] = real_t(0.003) * ((i % 11) - 5);
            h->data()[i] = real_t(0.002) * ((i % 7) - 3);
        }

        const real_t eps = 1e-6;
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            xp->data()[i] = x->data()[i] + eps * h->data()[i];
            xm->data()[i] = x->data()[i] - eps * h->data()[i];
        }

        std::fill(g->data(), g->data() + ndofs, real_t(0));
        std::fill(gp->data(), gp->data() + ndofs, real_t(0));
        std::fill(gm->data(), gm->data() + ndofs, real_t(0));
        std::fill(ah->data(), ah->data() + ndofs, real_t(0));
        SFEM_TEST_ASSERT(op->gradient(x->data(), g->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(op->gradient(xp->data(), gp->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(op->gradient(xm->data(), gm->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(op->apply(x->data(), h->data(), ah->data()) == SFEM_SUCCESS);

        real_t vp = 0, vm = 0;
        SFEM_TEST_ASSERT(op->value(xp->data(), &vp) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(op->value(xm->data(), &vm) == SFEM_SUCCESS);
        SFEM_TEST_APPROXEQ((vp - vm) / (2 * eps), dot(ndofs, g->data(), h->data()), 1e-9);
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            SFEM_TEST_APPROXEQ((gp->data()[i] - gm->data()[i]) / (2 * eps), ah->data()[i], 2e-9);
        }

        auto rowptr = sfem::create_host_buffer<count_t>(nnodes + 1);
        auto colidx = sfem::create_host_buffer<idx_t>(nnodes * nnodes);
        auto matrix = sfem::create_host_buffer<real_t>(nnodes * nnodes * 9);
        auto mh     = sfem::create_host_buffer<real_t>(ndofs);
        for (ptrdiff_t i = 0; i <= nnodes; ++i) rowptr->data()[i] = i * nnodes;
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            for (ptrdiff_t j = 0; j < nnodes; ++j) colidx->data()[i * nnodes + j] = j;
        }
        std::fill(matrix->data(), matrix->data() + nnodes * nnodes * 9, real_t(0));
        std::fill(mh->data(), mh->data() + ndofs, real_t(0));
        SFEM_TEST_ASSERT(op->hessian_bsr(x->data(), rowptr->data(), colidx->data(), matrix->data()) == SFEM_SUCCESS);
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            for (ptrdiff_t j = 0; j < nnodes; ++j) {
                const real_t *const block = &matrix->data()[(i * nnodes + j) * 9];
                for (int di = 0; di < 3; ++di) {
                    for (int dj = 0; dj < 3; ++dj) mh->data()[3 * i + di] += block[3 * di + dj] * h->data()[3 * j + dj];
                }
            }
        }
        SFEM_ASSERT_ARRAY_APPROX_EQ(ndofs, mh->data(), ah->data(), 1e-12);
        return SFEM_TEST_SUCCESS;
    }

    sfem::SharedBuffer<idx_t *> tetra_surface(const std::shared_ptr<sfem::Mesh> &mesh, const bool reverse) {
        const idx_t *const *elements = mesh->elements(0)->data();
        idx_t               nodes[4] = {elements[0][0], elements[1][0], elements[3][0], elements[4][0]};
        auto                points   = mesh->points()->data();
        real_t              a[3], b[3], c[3];
        for (int d = 0; d < 3; ++d) {
            a[d] = points[d][nodes[1]] - points[d][nodes[0]];
            b[d] = points[d][nodes[2]] - points[d][nodes[0]];
            c[d] = points[d][nodes[3]] - points[d][nodes[0]];
        }
        const real_t det =
                a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
        if (det < 0) std::swap(nodes[1], nodes[2]);

        const int faces[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
        auto      surface     = sfem::create_host_buffer<idx_t>(3, 4);
        for (int e = 0; e < 4; ++e) {
            surface->data()[0][e] = nodes[faces[e][0]];
            surface->data()[1][e] = nodes[faces[e][reverse ? 2 : 1]];
            surface->data()[2][e] = nodes[faces[e][reverse ? 1 : 2]];
        }
        return surface;
    }

    sfem::SharedBuffer<idx_t *> cube_surface(const std::shared_ptr<sfem::Mesh> &mesh) {
        const idx_t *const *elements    = mesh->elements(0)->data();
        const int           faces[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
        auto                surface     = sfem::create_host_buffer<idx_t>(4, 6);
        for (int e = 0; e < 6; ++e) {
            for (int v = 0; v < 4; ++v) surface->data()[v][e] = elements[faces[e][v]][0];
        }
        return surface;
    }

}  // namespace

int test_tri3_follower_pressure() {
    auto                               mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 1, 1, 1);
    auto                               space = sfem::FunctionSpace::create(mesh, 3);
    sfem::NeumannConditions::Condition condition;
    condition.element_type      = smesh::TRI3;
    condition.surface           = tetra_surface(mesh, false);
    condition.value             = 2.5;
    condition.follower_pressure = true;
    auto op                     = sfem::NeumannConditions::create(space, {condition});
    SFEM_TEST_ASSERT(!op->is_linear());
    SFEM_TEST_ASSERT(verify_derivatives(op, space) == SFEM_TEST_SUCCESS);

    auto zero        = sfem::create_host_buffer<real_t>(space->n_dofs());
    auto translation = sfem::create_host_buffer<real_t>(space->n_dofs());
    auto g0          = sfem::create_host_buffer<real_t>(space->n_dofs());
    auto gt          = sfem::create_host_buffer<real_t>(space->n_dofs());
    std::fill(zero->data(), zero->data() + space->n_dofs(), real_t(0));
    std::fill(g0->data(), g0->data() + space->n_dofs(), real_t(0));
    std::fill(gt->data(), gt->data() + space->n_dofs(), real_t(0));
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        translation->data()[3 * i]     = 0.2;
        translation->data()[3 * i + 1] = -0.1;
        translation->data()[3 * i + 2] = 0.3;
    }
    SFEM_TEST_ASSERT(op->gradient(zero->data(), g0->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op->gradient(translation->data(), gt->data()) == SFEM_SUCCESS);
    SFEM_ASSERT_ARRAY_APPROX_EQ(space->n_dofs(), g0->data(), gt->data(), 1e-13);
    real_t translated_value = 0;
    SFEM_TEST_ASSERT(op->value(translation->data(), &translated_value) == SFEM_SUCCESS);
    SFEM_TEST_APPROXEQ(translated_value, 0, 1e-13);
    for (int d = 0; d < 3; ++d) {
        real_t resultant = 0;
        for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) resultant += g0->data()[3 * i + d];
        SFEM_TEST_APPROXEQ(resultant, 0, 1e-13);
    }

    condition.surface = tetra_surface(mesh, true);
    auto reversed     = sfem::NeumannConditions::create(space, {condition});
    auto gr           = sfem::create_host_buffer<real_t>(space->n_dofs());
    std::fill(gr->data(), gr->data() + space->n_dofs(), real_t(0));
    SFEM_TEST_ASSERT(reversed->gradient(zero->data(), gr->data()) == SFEM_SUCCESS);
    for (ptrdiff_t i = 0; i < space->n_dofs(); ++i) SFEM_TEST_APPROXEQ(gr->data()[i], -g0->data()[i], 1e-13);
    return SFEM_TEST_SUCCESS;
}

int test_quad4_follower_pressure() {
    auto                               mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 1, 1, 1);
    auto                               space = sfem::FunctionSpace::create(mesh, 3);
    sfem::NeumannConditions::Condition condition;
    condition.element_type      = smesh::QUAD4;
    condition.surface           = cube_surface(mesh);
    condition.value             = 1.75;
    condition.follower_pressure = true;
    auto op                     = sfem::NeumannConditions::create(space, {condition});
    return verify_derivatives(op, space);
}

#ifdef SFEM_ENABLE_RYAML
int test_neumann_yaml_profile() {
    auto mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 1, 1, 1);
    auto space = sfem::FunctionSpace::create(mesh, 3);
    auto op    = sfem::NeumannConditions::create_from_yaml(
            space,
            "neumann_conditions:\n"
               "- type: sideset\n"
               "  format: expr\n"
               "  parent: [0]\n"
               "  lfi: [0]\n"
               "  value: 4\n"
               "  component: 0\n"
               "  profile: {type: linear_ramp, start_time: 0, end_time: 2, start_value: 0, end_value: 1}\n");
    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_ASSERT(op->set_time(real_t(0.5)) == SFEM_SUCCESS);
    SFEM_TEST_APPROXEQ(op->conditions()[0].value, 1, 1e-15);
    SFEM_TEST_ASSERT(op->set_time(real_t(3)) == SFEM_SUCCESS);
    SFEM_TEST_APPROXEQ(op->conditions()[0].value, 4, 1e-15);
    return SFEM_TEST_SUCCESS;
}
#endif

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_tri3_follower_pressure);
    SFEM_RUN_TEST(test_quad4_follower_pressure);
#ifdef SFEM_ENABLE_RYAML
    SFEM_RUN_TEST(test_neumann_yaml_profile);
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
