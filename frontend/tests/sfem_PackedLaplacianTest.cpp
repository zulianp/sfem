#include "sfem_test.hpp"

#include "sfem_API.hpp"

int test_packed_laplacian() {
    MPI_Comm comm = MPI_COMM_WORLD;

    constexpr int N = 3;
    auto          mesh =
            sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm), N, N, N, 0, 0, 0, 1, 1, 1);
    auto fs = sfem::FunctionSpace::create(mesh, 1);

    SFEM_TEST_ASSERT(fs->initialize_packed_mesh() == SFEM_SUCCESS);

    auto laplacian        = sfem::create_op(fs, "Laplacian", sfem::EXECUTION_SPACE_HOST);
    auto packed_laplacian = sfem::create_op(fs, "PackedLaplacian", sfem::EXECUTION_SPACE_HOST);

    SFEM_TEST_ASSERT(laplacian != nullptr);
    SFEM_TEST_ASSERT(packed_laplacian != nullptr);
    SFEM_TEST_ASSERT(laplacian->initialize() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(packed_laplacian->initialize() == SFEM_SUCCESS);

    const ptrdiff_t ndofs = fs->n_dofs();
    const ptrdiff_t nnodes = mesh->n_nodes();

    auto h        = sfem::create_host_buffer<real_t>(ndofs);
    auto y_ref    = sfem::create_host_buffer<real_t>(ndofs);
    auto y_packed = sfem::create_host_buffer<real_t>(ndofs);

    auto points = mesh->points()->data();
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        h->data()[i] = points[0][i] + 2 * points[1][i] - 3 * points[2][i];
    }

    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        y_ref->data()[i] = 0;
        y_packed->data()[i] = 0;
    }

    SFEM_TEST_ASSERT(laplacian->apply(nullptr, h->data(), y_ref->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(packed_laplacian->apply(nullptr, h->data(), y_packed->data()) == SFEM_SUCCESS);

    SFEM_ASSERT_ARRAY_APPROX_EQ(ndofs, y_ref->data(), y_packed->data(), 1e-10);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_packed_laplacian);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
