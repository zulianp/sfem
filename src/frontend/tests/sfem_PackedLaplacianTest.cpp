#include "sfem_test.hpp"

#include "sfem_API.hpp"

#include <algorithm>
#include <cmath>

int test_packed_laplacian() {

    enum smesh::ElemType SFEM_ELEM_TYPE = smesh::type_from_string(smesh::Env::read_string("SFEM_ELEM_TYPE", "HEX8").c_str());
    constexpr int        N              = 40;
    auto mesh = sfem::Mesh::create_cube(sfem::Communicator::world(), SFEM_ELEM_TYPE, N, N, N, -1, -1, -1, 1, 1, 1);
    auto fs   = sfem::FunctionSpace::create(mesh, 1);

    SFEM_TEST_ASSERT(fs->initialize_packed_mesh() == SFEM_SUCCESS);

    auto laplacian        = sfem::create_op(fs, "Laplacian", sfem::EXECUTION_SPACE_HOST);
    auto packed_laplacian = sfem::create_op(fs, "PackedLaplacian", sfem::EXECUTION_SPACE_HOST);

    {
        // Make the mesh non-uniform
        auto            b_points = mesh->points()->data();
        int             dim      = mesh->spatial_dimension();
        const ptrdiff_t nnodes   = mesh->n_nodes();
        for (int d = 0; d < dim; d++) {
            for (ptrdiff_t i = 0; i < nnodes; ++i) {
                b_points[d][i] =
                        b_points[d][i] * b_points[d][i] * b_points[d][i] + b_points[0][i] * b_points[0][i] * b_points[0][i] * 0.2;
            }
        }
    }

    // mesh->write(smesh::Path("mesh_r"));

    SFEM_TEST_ASSERT(laplacian != nullptr);
    SFEM_TEST_ASSERT(packed_laplacian != nullptr);
    SFEM_TEST_ASSERT(laplacian->initialize() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(packed_laplacian->initialize() == SFEM_SUCCESS);

    const ptrdiff_t ndofs  = fs->n_dofs();
    const ptrdiff_t nnodes = mesh->n_nodes();

    auto h        = sfem::create_host_buffer<real_t>(ndofs);
    auto y_ref    = sfem::create_host_buffer<real_t>(ndofs);
    auto y_packed = sfem::create_host_buffer<real_t>(ndofs);

    auto points = mesh->points()->data();
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        h->data()[i] = points[0][i] + 2 * points[1][i] - 3 * points[2][i];
    }

    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        y_ref->data()[i]    = 0;
        y_packed->data()[i] = 0;
    }

    SFEM_TEST_ASSERT(laplacian->apply(nullptr, h->data(), y_ref->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(packed_laplacian->apply(nullptr, h->data(), y_packed->data()) == SFEM_SUCCESS);

    SFEM_ASSERT_ARRAY_APPROX_EQ(ndofs, y_ref->data(), y_packed->data(), 1e-10);
    return SFEM_TEST_SUCCESS;
}

// The packed Neo-Hookean (Ogden), or the generic Hyperelasticity, against the standard one at the same
// state: the partial assembly, through the plain apply (one region around the team product).
static int check_neohookean_ogden(const std::shared_ptr<sfem::Mesh> &mesh, const char *const name) {
    auto fs = sfem::FunctionSpace::create(mesh, 3);
    SFEM_TEST_ASSERT(fs->initialize_packed_mesh() == SFEM_SUCCESS);

    auto op        = sfem::create_op(fs, "NeoHookeanOgden", sfem::EXECUTION_SPACE_HOST);
    auto packed_op = sfem::create_op(fs, name, sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_ASSERT(packed_op != nullptr);
    SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(packed_op->initialize() == SFEM_SUCCESS);

    const ptrdiff_t ndofs  = fs->n_dofs();
    const ptrdiff_t nnodes = mesh->n_nodes();
    auto            points = mesh->points()->data();

    auto u        = sfem::create_host_buffer<real_t>(ndofs);
    auto h        = sfem::create_host_buffer<real_t>(ndofs);
    auto y_ref    = sfem::create_host_buffer<real_t>(ndofs);
    auto y_packed = sfem::create_host_buffer<real_t>(ndofs);
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        const real_t x = points[0][i], y = points[1][i], z = points[2][i];
        for (int c = 0; c < 3; ++c) {
            u->data()[i * 3 + c] = real_t(0.01) * (x * y + (c + 1) * z * z - c * x);
            h->data()[i * 3 + c] = x + 2 * y - 3 * z + c;
        }
    }
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        y_ref->data()[i]    = 0;
        y_packed->data()[i] = 0;
    }

    SFEM_TEST_ASSERT(op->update(u->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(packed_op->update(u->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op->apply(u->data(), h->data(), y_ref->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(packed_op->apply(u->data(), h->data(), y_packed->data()) == SFEM_SUCCESS);

    real_t scale = 0;
    for (ptrdiff_t i = 0; i < ndofs; ++i) scale = std::max(scale, std::abs(y_ref->data()[i]));
    SFEM_TEST_ASSERT(scale > 0);
    SFEM_ASSERT_ARRAY_APPROX_EQ(ndofs, y_ref->data(), y_packed->data(), 1e-12 * scale);
    return SFEM_TEST_SUCCESS;
}

int test_packed_neohookean_ogden() {
    constexpr int N    = 12;
    auto          hex8 = sfem::Mesh::create_cube(sfem::Communicator::world(), smesh::HEX8, N, N, N, -1, -1, -1, 1, 1, 1);
    SFEM_TEST_ASSERT(check_neohookean_ogden(hex8, "NeoHookeanOgdenPacked") == SFEM_TEST_SUCCESS);

    auto tet4  = sfem::Mesh::create_cube(sfem::Communicator::world(), smesh::TET4, N, N, N, -1, -1, -1, 1, 1, 1);
    auto tet10 = smesh::promote_to(smesh::TET10, tet4);
    SFEM_TEST_ASSERT(check_neohookean_ogden(tet10, "NeoHookeanOgdenPacked") == SFEM_TEST_SUCCESS);
    return SFEM_TEST_SUCCESS;
}

// Hyperelasticity registers Neo-Hookean (Ogden) kernels for HEX8 only.
int test_hyperelasticity() {
    constexpr int N    = 12;
    auto          hex8 = sfem::Mesh::create_cube(sfem::Communicator::world(), smesh::HEX8, N, N, N, -1, -1, -1, 1, 1, 1);
    SFEM_TEST_ASSERT(check_neohookean_ogden(hex8, "Hyperelasticity") == SFEM_TEST_SUCCESS);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_packed_laplacian);
    SFEM_RUN_TEST(test_packed_neohookean_ogden);
    SFEM_RUN_TEST(test_hyperelasticity);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}

