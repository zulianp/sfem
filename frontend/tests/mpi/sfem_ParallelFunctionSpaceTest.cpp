#include "sfem_test.hpp"

#include "sfem_FunctionSpace.hpp"

#include "smesh_mesh.hpp"

namespace {

    int check_space_counts(const std::shared_ptr<sfem::FunctionSpace> &space,
                           const std::shared_ptr<sfem::Mesh>         &mesh,
                           const int                                  block_size) {
        SFEM_TEST_EQ(space->block_size(), block_size);
        SFEM_TEST_EQ(space->n_blocks(), mesh->n_blocks());

        if (mesh->is_distributed()) {
            auto dist = mesh->distributed();
            SFEM_TEST_EQ(space->n_dofs(), dist->n_nodes_local() * block_size);
            SFEM_TEST_EQ(space->n_owned_dofs(), dist->n_nodes_owned() * block_size);
            SFEM_TEST_EQ(space->n_dofs_global(), dist->n_nodes_global() * block_size);
        } else {
            const ptrdiff_t expected = mesh->n_nodes() * block_size;
            SFEM_TEST_EQ(space->n_dofs(), expected);
            SFEM_TEST_EQ(space->n_owned_dofs(), expected);
            SFEM_TEST_EQ(space->n_dofs_global(), expected);
        }

        ptrdiff_t nlocal  = 0;
        ptrdiff_t nglobal = 0;
        real_t   *values  = nullptr;
        SFEM_TEST_ASSERT(space->create_vector(&nlocal, &nglobal, &values) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(values != nullptr);
        SFEM_TEST_EQ(nlocal, space->n_dofs());
        SFEM_TEST_EQ(nglobal, space->n_dofs_global());
        SFEM_TEST_ASSERT(space->destroy_vector(values) == SFEM_SUCCESS);
        return SFEM_TEST_SUCCESS;
    }

    int check_parallel_function_space(const std::shared_ptr<sfem::Mesh> &serial_mesh,
                                      const std::shared_ptr<sfem::Mesh> &parallel_mesh,
                                      const int                          n_expected_blocks) {
        auto comm = sfem::Communicator::world();

        SFEM_TEST_ASSERT(serial_mesh != nullptr);
        SFEM_TEST_ASSERT(parallel_mesh != nullptr);
        SFEM_TEST_EQ(serial_mesh->n_blocks(), static_cast<size_t>(n_expected_blocks));
        SFEM_TEST_EQ(parallel_mesh->n_blocks(), serial_mesh->n_blocks());
        if (comm->size() > 1) {
            SFEM_TEST_ASSERT(parallel_mesh->is_distributed());
        } else {
            SFEM_TEST_ASSERT(!parallel_mesh->is_distributed());
        }

        for (size_t b = 0; b < serial_mesh->n_blocks(); ++b) {
            SFEM_TEST_EQ(parallel_mesh->element_type(static_cast<smesh::block_idx_t>(b)),
                         serial_mesh->element_type(static_cast<smesh::block_idx_t>(b)));
        }

        const int block_sizes[] = {1, 3};
        for (int bs : block_sizes) {
            auto serial_space   = sfem::FunctionSpace::create(serial_mesh, bs);
            auto parallel_space = sfem::FunctionSpace::create(parallel_mesh, bs);

            SFEM_TEST_ASSERT(check_space_counts(serial_space, serial_mesh, bs) == SFEM_TEST_SUCCESS);
            SFEM_TEST_ASSERT(check_space_counts(parallel_space, parallel_mesh, bs) == SFEM_TEST_SUCCESS);

            SFEM_TEST_EQ(parallel_space->n_blocks(), serial_space->n_blocks());
            SFEM_TEST_EQ(parallel_space->n_dofs_global(), serial_space->n_dofs_global());
            SFEM_TEST_EQ(comm->sum(parallel_space->n_owned_dofs()), parallel_space->n_dofs_global());

            for (size_t b = 0; b < serial_space->n_blocks(); ++b) {
                SFEM_TEST_EQ(parallel_space->element_type(static_cast<int>(b)), serial_space->element_type(static_cast<int>(b)));
            }
        }

        return SFEM_TEST_SUCCESS;
    }

    int test_parallel_function_space_hex8_cube() {
        auto comm = sfem::Communicator::world();
        const ptrdiff_t nx = 8, ny = 6, nz = 4;
        auto serial   = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), nx, ny, nz);
        auto parallel = sfem::Mesh::create_hex8_cube(comm, nx, ny, nz);
        return check_parallel_function_space(serial, parallel, 1);
    }

    int test_parallel_function_space_checkerboard() {
        auto comm = sfem::Communicator::world();
        const ptrdiff_t nx = 8, ny = 6, nz = 4;
        auto serial   = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), nx, ny, nz);
        auto parallel = sfem::Mesh::create_hex8_checkerboard_cube(comm, nx, ny, nz);
        SFEM_TEST_ASSERT(parallel->block(0)->name() == "white");
        SFEM_TEST_ASSERT(parallel->block(1)->name() == "black");
        return check_parallel_function_space(serial, parallel, 2);
    }

    int test_parallel_function_space_hex8_tet4() {
        auto comm = sfem::Communicator::world();
        const ptrdiff_t nx = 4, ny = 4, nz = 2;
        auto serial   = sfem::Mesh::create_hex8_tet4_cube(sfem::Communicator::self(), nx, ny, nz);
        auto parallel = sfem::Mesh::create_hex8_tet4_cube(comm, nx, ny, nz);
        SFEM_TEST_EQ(parallel->element_type(0), smesh::HEX8);
        SFEM_TEST_EQ(parallel->element_type(1), smesh::TET4);
        return check_parallel_function_space(serial, parallel, 2);
    }

    int test_parallel_function_space_tet4_cube() {
        auto comm = sfem::Communicator::world();
        const ptrdiff_t nx = 4, ny = 2, nz = 2;
        auto serial   = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), nx, ny, nz);
        auto parallel = sfem::Mesh::create_tet4_cube(comm, nx, ny, nz);
        SFEM_TEST_EQ(parallel->element_type(0), smesh::TET4);
        return check_parallel_function_space(serial, parallel, 1);
    }

}  // namespace

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_parallel_function_space_hex8_cube);
    SFEM_RUN_TEST(test_parallel_function_space_checkerboard);
    SFEM_RUN_TEST(test_parallel_function_space_hex8_tet4);
    SFEM_RUN_TEST(test_parallel_function_space_tet4_cube);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}


