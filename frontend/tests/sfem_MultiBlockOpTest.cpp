#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_FunctionSpace.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace {

    const geom_t geom_tol() { return sizeof(geom_t) == sizeof(double) ? geom_t(1e-12) : geom_t(1e-5); }

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

    void fill_scalar_field(const smesh::Mesh &mesh, real_t *const x) {
        auto            points = mesh.points()->data();
        const ptrdiff_t n      = mesh.n_nodes();
        for (ptrdiff_t i = 0; i < n; ++i) {
            const real_t px = points[0][i];
            const real_t py = points[1][i];
            const real_t pz = points[2][i];
            x[i]            = px * px + real_t(0.5) * py - real_t(0.25) * pz * pz + real_t(0.125) * px * py;
        }
    }

    void fill_vector_field(const smesh::Mesh &mesh, real_t *const x) {
        auto            points = mesh.points()->data();
        const ptrdiff_t n      = mesh.n_nodes();
        for (ptrdiff_t i = 0; i < n; ++i) {
            const real_t px = points[0][i];
            const real_t py = points[1][i];
            const real_t pz = points[2][i];
            x[3 * i + 0]    = px + real_t(0.25) * py * py;
            x[3 * i + 1]    = py - real_t(0.125) * px * pz;
            x[3 * i + 2]    = pz + real_t(0.5) * px * px;
        }
    }

    int compare_mapped_dofs(const sfem::SharedBuffer<real_t> &actual,
                            const sfem::SharedBuffer<real_t> &expected,
                            const std::vector<ptrdiff_t>     &actual_to_expected_nodes,
                            const int                         block_size,
                            const real_t                      tol) {
        SFEM_TEST_EQ(actual->size(), expected->size());
        SFEM_TEST_EQ(actual->size(), actual_to_expected_nodes.size() * (size_t)block_size);
        real_t diff2 = 0;
        real_t ref2  = 0;
        for (size_t node = 0; node < actual_to_expected_nodes.size(); ++node) {
            const size_t mapped_node = (size_t)actual_to_expected_nodes[node];
            for (int c = 0; c < block_size; ++c) {
                const size_t actual_idx   = node * (size_t)block_size + (size_t)c;
                const size_t expected_idx = mapped_node * (size_t)block_size + (size_t)c;
                SFEM_TEST_ASSERT(std::isfinite(actual->data()[actual_idx]));
                SFEM_TEST_ASSERT(std::isfinite(expected->data()[expected_idx]));
                const real_t diff = actual->data()[actual_idx] - expected->data()[expected_idx];
                diff2 += diff * diff;
                ref2 += expected->data()[expected_idx] * expected->data()[expected_idx];
            }
        }

        const real_t rel = std::sqrt(diff2) / (std::sqrt(ref2) + real_t(1e-16));
        SFEM_TEST_ASSERT(rel < tol);
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int test_multi_block_op() {
    auto mesh  = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 4, 4, 4);
    auto space = sfem::FunctionSpace::create(mesh, 1);

    auto f  = sfem::Function::create(space);
    auto op = sfem::Factory::create_op(space, "Laplacian");
    op->initialize();
    f->add_operator(op);

    auto lop = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, f, nullptr, f->execution_space());

    auto cg = sfem::create_cg(lop, f->execution_space());
    auto x  = sfem::create_buffer<real_t>(space->n_dofs(), f->execution_space());
    auto b  = sfem::create_buffer<real_t>(space->n_dofs(), f->execution_space());

    SFEM_TEST_ASSERT(cg->apply(x->data(), b->data()) == SFEM_SUCCESS);

    smesh::create_directory("test_multi_block_op");
    SFEM_TEST_ASSERT(mesh->write(smesh::Path("test_multi_block_op")) == SFEM_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int test_checkerboard_sshex_em_laplacian_apply() {
    auto cb_hex   = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 4, 4, 4);
    auto cube_hex = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 4, 4, 4);
    auto cb_ss    = smesh::to_semistructured(2, cb_hex, true, false);
    auto cube_ss  = smesh::to_semistructured(2, cube_hex, true, false);
    SFEM_TEST_ASSERT(cb_ss != nullptr);
    SFEM_TEST_ASSERT(cube_ss != nullptr);
    SFEM_TEST_EQ(cb_ss->n_blocks(), static_cast<size_t>(2));

    std::vector<ptrdiff_t> cb_to_cube;
    SFEM_TEST_ASSERT(map_nodes_by_xyz(*cb_ss, *cube_ss, cb_to_cube) == SFEM_TEST_SUCCESS);

    auto cb_space   = sfem::FunctionSpace::create(cb_ss, 1);
    auto cube_space = sfem::FunctionSpace::create(cube_ss, 1);
    auto x_cb       = sfem::create_host_buffer<real_t>(cb_space->n_dofs());
    auto x_cube     = sfem::create_host_buffer<real_t>(cube_space->n_dofs());
    auto y_cb       = sfem::create_host_buffer<real_t>(cb_space->n_dofs());
    auto y_cube     = sfem::create_host_buffer<real_t>(cube_space->n_dofs());
    fill_scalar_field(*cb_ss, x_cb->data());
    fill_scalar_field(*cube_ss, x_cube->data());
    std::fill(y_cb->data(), y_cb->data() + y_cb->size(), real_t(0));
    std::fill(y_cube->data(), y_cube->data() + y_cube->size(), real_t(0));

    auto cb_em   = sfem::create_op(cb_space, "em:Laplacian", sfem::EXECUTION_SPACE_HOST);
    auto cube_em = sfem::create_op(cube_space, "em:Laplacian", sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(cb_em != nullptr);
    SFEM_TEST_ASSERT(cube_em != nullptr);
    SFEM_TEST_ASSERT(cb_em->initialize() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(cube_em->initialize() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(cb_em->apply(nullptr, x_cb->data(), y_cb->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(cube_em->apply(nullptr, x_cube->data(), y_cube->data()) == SFEM_SUCCESS);

    const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-10) : real_t(1e-5);
    return compare_mapped_dofs(y_cb, y_cube, cb_to_cube, 1, tol);
}

int test_checkerboard_sshex_em_linear_elasticity_apply() {
    auto cb_hex   = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), 4, 4, 4);
    auto cube_hex = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 4, 4, 4);
    auto cb_ss    = smesh::to_semistructured(2, cb_hex, true, false);
    auto cube_ss  = smesh::to_semistructured(2, cube_hex, true, false);
    SFEM_TEST_ASSERT(cb_ss != nullptr);
    SFEM_TEST_ASSERT(cube_ss != nullptr);
    SFEM_TEST_EQ(cb_ss->n_blocks(), static_cast<size_t>(2));

    std::vector<ptrdiff_t> cb_to_cube;
    SFEM_TEST_ASSERT(map_nodes_by_xyz(*cb_ss, *cube_ss, cb_to_cube) == SFEM_TEST_SUCCESS);

    auto cb_space   = sfem::FunctionSpace::create(cb_ss, 3);
    auto cube_space = sfem::FunctionSpace::create(cube_ss, 3);
    auto x_cb       = sfem::create_host_buffer<real_t>(cb_space->n_dofs());
    auto x_cube     = sfem::create_host_buffer<real_t>(cube_space->n_dofs());
    auto y_cb       = sfem::create_host_buffer<real_t>(cb_space->n_dofs());
    auto y_cube     = sfem::create_host_buffer<real_t>(cube_space->n_dofs());
    fill_vector_field(*cb_ss, x_cb->data());
    fill_vector_field(*cube_ss, x_cube->data());
    std::fill(y_cb->data(), y_cb->data() + y_cb->size(), real_t(0));
    std::fill(y_cube->data(), y_cube->data() + y_cube->size(), real_t(0));

    auto cb_em   = sfem::create_op(cb_space, "em:LinearElasticity", sfem::EXECUTION_SPACE_HOST);
    auto cube_em = sfem::create_op(cube_space, "em:LinearElasticity", sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(cb_em != nullptr);
    SFEM_TEST_ASSERT(cube_em != nullptr);
    SFEM_TEST_ASSERT(cb_em->initialize() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(cube_em->initialize() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(cb_em->apply(nullptr, x_cb->data(), y_cb->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(cube_em->apply(nullptr, x_cube->data(), y_cube->data()) == SFEM_SUCCESS);

    const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-10) : real_t(1e-5);
    return compare_mapped_dofs(y_cb, y_cube, cb_to_cube, 3, tol);
}

int test_hex8_tet4_laplacian_apply() {
    auto mesh = sfem::Mesh::create_hex8_tet4_cube(sfem::Communicator::self(), 4, 4, 2);
    SFEM_TEST_ASSERT(mesh != nullptr);
    SFEM_TEST_ASSERT(!mesh->is_distributed());
    SFEM_TEST_EQ(mesh->n_blocks(), static_cast<size_t>(2));
    SFEM_TEST_EQ(mesh->element_type(0), smesh::HEX8);
    SFEM_TEST_EQ(mesh->element_type(1), smesh::TET4);

    auto space = sfem::FunctionSpace::create(mesh, 1);
    auto f     = sfem::Function::create(space);
    auto op    = sfem::Factory::create_op(space, "Laplacian");
    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);
    f->add_operator(op);

    const ptrdiff_t n      = space->n_dofs();
    auto            x      = sfem::create_host_buffer<real_t>(n);
    auto            y      = sfem::create_host_buffer<real_t>(n);
    auto            points = mesh->points()->data();
    for (ptrdiff_t i = 0; i < n; ++i) {
        const geom_t px = points[0][i];
        const geom_t py = points[1][i];
        const geom_t pz = points[2][i];
        x->data()[i]    = px * px + real_t(0.5) * py * py - real_t(0.25) * pz * pz;
        y->data()[i]    = 0;
    }

    SFEM_TEST_ASSERT(f->apply(nullptr, x->data(), y->data()) == SFEM_SUCCESS);

    real_t nrm2 = 0;
    for (ptrdiff_t i = 0; i < n; ++i) {
        nrm2 += y->data()[i] * y->data()[i];
    }
    SFEM_TEST_ASSERT(nrm2 > 0);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_multi_block_op);
    SFEM_RUN_TEST(test_checkerboard_sshex_em_laplacian_apply);
    SFEM_RUN_TEST(test_checkerboard_sshex_em_linear_elasticity_apply);
    SFEM_RUN_TEST(test_hex8_tet4_laplacian_apply);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
