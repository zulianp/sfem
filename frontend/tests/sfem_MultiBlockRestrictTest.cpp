#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_FunctionSpace.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include <cmath>
#include <memory>
#include <vector>

namespace {

const real_t val_tol() { return sizeof(real_t) == sizeof(double) ? real_t(1e-12) : real_t(1e-5); }

const geom_t geom_tol() { return sizeof(geom_t) == sizeof(double) ? geom_t(1e-12) : geom_t(1e-5); }

void fill_ones(const sfem::SharedBuffer<real_t> &v) {
    auto            d = v->data();
    const ptrdiff_t n = (ptrdiff_t)v->size();
    for (ptrdiff_t i = 0; i < n; ++i) {
        d[i] = 1;
    }
}

void fill_linear(const smesh::Mesh &mesh, const sfem::SharedBuffer<real_t> &v) {
    auto            p = mesh.points()->data();
    auto            d = v->data();
    const ptrdiff_t n = mesh.n_nodes();
    for (ptrdiff_t i = 0; i < n; ++i) {
        d[i] = (real_t)p[0][i] + real_t(2) * (real_t)p[1][i] + real_t(3) * (real_t)p[2][i];
    }
}

int map_nodes_by_xyz(const smesh::Mesh &from, const smesh::Mesh &to, std::vector<ptrdiff_t> &from_to_to) {
    SFEM_TEST_EQ(from.n_nodes(), to.n_nodes());
    const ptrdiff_t n   = from.n_nodes();
    const geom_t    tol = geom_tol();
    from_to_to.assign((size_t)n, -1);

    auto pa = from.points()->data();
    auto pb = to.points()->data();
    for (ptrdiff_t i = 0; i < n; ++i) {
        ptrdiff_t found = -1;
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (std::fabs(pa[0][i] - pb[0][j]) <= tol && std::fabs(pa[1][i] - pb[1][j]) <= tol &&
                std::fabs(pa[2][i] - pb[2][j]) <= tol) {
                found = j;
                break;
            }
        }
        SFEM_TEST_ASSERT(found >= 0);
        from_to_to[(size_t)i] = found;
    }
    return SFEM_TEST_SUCCESS;
}

int compare_mapped(const sfem::SharedBuffer<real_t> &a,
                   const sfem::SharedBuffer<real_t> &b,
                   const std::vector<ptrdiff_t>     &a2b) {
    const ptrdiff_t n   = (ptrdiff_t)a->size();
    const real_t    tol = val_tol();
    SFEM_TEST_EQ(n, (ptrdiff_t)a2b.size());
    SFEM_TEST_EQ(n, (ptrdiff_t)b->size());
    auto da = a->data();
    auto db = b->data();
    for (ptrdiff_t i = 0; i < n; ++i) {
        SFEM_TEST_ASSERT(std::isfinite(da[i]));
        SFEM_TEST_ASSERT(std::isfinite(db[a2b[(size_t)i]]));
        SFEM_TEST_ASSERT(std::fabs(da[i] - db[a2b[(size_t)i]]) <= tol);
    }
    return SFEM_TEST_SUCCESS;
}

struct SSPair {
    std::shared_ptr<sfem::FunctionSpace> fine;
    std::shared_ptr<sfem::FunctionSpace> coarse;
};

SSPair make_ss_pair(const std::shared_ptr<smesh::Mesh> &hex) {
    const int L  = 4;
    auto      ss = smesh::to_semistructured(L, hex, true, false);
    auto      fine   = sfem::FunctionSpace::create(ss, 1);
    auto      coarse = fine->derefine(2);
    return {fine, coarse};
}

}  // namespace

int test_checkerboard_prolong_ones() {
    auto mesh = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 4, 4, 4);
    SFEM_TEST_ASSERT(mesh != nullptr);
    SFEM_TEST_EQ(mesh->n_blocks(), static_cast<size_t>(2));

    auto pair = make_ss_pair(mesh);
    SFEM_TEST_ASSERT(pair.fine->has_semi_structured_mesh());
    SFEM_TEST_EQ(pair.fine->mesh().n_blocks(), static_cast<size_t>(2));

    auto prolongation = sfem::create_hierarchical_prolongation(pair.coarse, pair.fine, sfem::EXECUTION_SPACE_HOST);
    auto coarse_field = sfem::create_host_buffer<real_t>(pair.coarse->n_dofs());
    auto fine_field   = sfem::create_host_buffer<real_t>(pair.fine->n_dofs());
    fill_ones(coarse_field);

    SFEM_TEST_ASSERT(prolongation->apply(coarse_field->data(), fine_field->data()) == SFEM_SUCCESS);

    const real_t tol = val_tol();
    auto         d   = fine_field->data();
    for (ptrdiff_t i = 0; i < pair.fine->n_dofs(); ++i) {
        SFEM_TEST_ASSERT(std::isfinite(d[i]));
        SFEM_TEST_ASSERT(std::fabs(d[i] - real_t(1)) <= tol);
    }
    return SFEM_TEST_SUCCESS;
}

int test_checkerboard_restrict_finite() {
    auto mesh = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 4, 4, 4);
    auto pair = make_ss_pair(mesh);

    auto restriction  = sfem::create_hierarchical_restriction(pair.fine, pair.coarse, sfem::EXECUTION_SPACE_HOST);
    auto fine_field   = sfem::create_host_buffer<real_t>(pair.fine->n_dofs());
    auto coarse_field = sfem::create_host_buffer<real_t>(pair.coarse->n_dofs());
    fill_ones(fine_field);

    SFEM_TEST_ASSERT(restriction->apply(fine_field->data(), coarse_field->data()) == SFEM_SUCCESS);

    auto d = coarse_field->data();
    for (ptrdiff_t i = 0; i < pair.coarse->n_dofs(); ++i) {
        SFEM_TEST_ASSERT(std::isfinite(d[i]));
    }
    return SFEM_TEST_SUCCESS;
}

int test_checkerboard_vs_cube() {
    auto cb   = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 4, 4, 4);
    auto cube = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 4, 4, 4);

    auto cb_pair   = make_ss_pair(cb);
    auto cube_pair = make_ss_pair(cube);

    SFEM_TEST_EQ(cb_pair.fine->n_dofs(), cube_pair.fine->n_dofs());
    SFEM_TEST_EQ(cb_pair.coarse->n_dofs(), cube_pair.coarse->n_dofs());

    std::vector<ptrdiff_t> fine_cb2cube;
    std::vector<ptrdiff_t> coarse_cb2cube;
    SFEM_TEST_ASSERT(map_nodes_by_xyz(cb_pair.fine->mesh(), cube_pair.fine->mesh(), fine_cb2cube) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(map_nodes_by_xyz(cb_pair.coarse->mesh(), cube_pair.coarse->mesh(), coarse_cb2cube) ==
                     SFEM_TEST_SUCCESS);

    auto es = sfem::EXECUTION_SPACE_HOST;

    {
        auto p_cb   = sfem::create_hierarchical_prolongation(cb_pair.coarse, cb_pair.fine, es);
        auto p_cube = sfem::create_hierarchical_prolongation(cube_pair.coarse, cube_pair.fine, es);

        auto c_cb   = sfem::create_host_buffer<real_t>(cb_pair.coarse->n_dofs());
        auto c_cube = sfem::create_host_buffer<real_t>(cube_pair.coarse->n_dofs());
        auto f_cb   = sfem::create_host_buffer<real_t>(cb_pair.fine->n_dofs());
        auto f_cube = sfem::create_host_buffer<real_t>(cube_pair.fine->n_dofs());

        fill_ones(c_cb);
        fill_ones(c_cube);
        SFEM_TEST_ASSERT(p_cb->apply(c_cb->data(), f_cb->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(p_cube->apply(c_cube->data(), f_cube->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(f_cb, f_cube, fine_cb2cube) == SFEM_TEST_SUCCESS);

        fill_linear(cb_pair.coarse->mesh(), c_cb);
        fill_linear(cube_pair.coarse->mesh(), c_cube);
        SFEM_TEST_ASSERT(p_cb->apply(c_cb->data(), f_cb->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(p_cube->apply(c_cube->data(), f_cube->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(f_cb, f_cube, fine_cb2cube) == SFEM_TEST_SUCCESS);
    }

    {
        auto r_cb   = sfem::create_hierarchical_restriction(cb_pair.fine, cb_pair.coarse, es);
        auto r_cube = sfem::create_hierarchical_restriction(cube_pair.fine, cube_pair.coarse, es);

        auto f_cb   = sfem::create_host_buffer<real_t>(cb_pair.fine->n_dofs());
        auto f_cube = sfem::create_host_buffer<real_t>(cube_pair.fine->n_dofs());
        auto c_cb   = sfem::create_host_buffer<real_t>(cb_pair.coarse->n_dofs());
        auto c_cube = sfem::create_host_buffer<real_t>(cube_pair.coarse->n_dofs());

        fill_ones(f_cb);
        fill_ones(f_cube);
        SFEM_TEST_ASSERT(r_cb->apply(f_cb->data(), c_cb->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(r_cube->apply(f_cube->data(), c_cube->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(c_cb, c_cube, coarse_cb2cube) == SFEM_TEST_SUCCESS);

        fill_linear(cb_pair.fine->mesh(), f_cb);
        fill_linear(cube_pair.fine->mesh(), f_cube);
        c_cb   = sfem::create_host_buffer<real_t>(cb_pair.coarse->n_dofs());
        c_cube = sfem::create_host_buffer<real_t>(cube_pair.coarse->n_dofs());
        SFEM_TEST_ASSERT(r_cb->apply(f_cb->data(), c_cb->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(r_cube->apply(f_cube->data(), c_cube->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(compare_mapped(c_cb, c_cube, coarse_cb2cube) == SFEM_TEST_SUCCESS);
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_checkerboard_prolong_ones);
    SFEM_RUN_TEST(test_checkerboard_restrict_finite);
    SFEM_RUN_TEST(test_checkerboard_vs_cube);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
