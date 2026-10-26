#include "sfem_test.hpp"

#include "sfem_FunctionSpace.hpp"

#include "smesh_mesh.hpp"

#include <cmath>
#include <vector>

#ifdef SFEM_ENABLE_CUDA
#include "smesh_device_buffer.hpp"
#endif

namespace {

    int check_space_counts(const std::shared_ptr<sfem::FunctionSpace> &space,
                           const std::shared_ptr<sfem::Mesh>         &mesh,
                           const int                                  block_size) {
        SFEM_TEST_EQ(space->block_size(), block_size);
        SFEM_TEST_EQ(space->n_blocks(), mesh->n_blocks());

        if (mesh->is_distributed()) {
            auto dist = mesh->distributed();
            SFEM_TEST_EQ(space->n_dofs(), dist->n_nodes_local() * block_size);
            SFEM_TEST_EQ(space->n_dofs_owned(), dist->n_nodes_owned() * block_size);
            SFEM_TEST_EQ(space->n_dofs_global(), dist->n_nodes_global() * block_size);
        } else {
            const ptrdiff_t expected = mesh->n_nodes() * block_size;
            SFEM_TEST_EQ(space->n_dofs(), expected);
            SFEM_TEST_EQ(space->n_dofs_owned(), expected);
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
            SFEM_TEST_EQ(comm->sum(parallel_space->n_dofs_owned()), parallel_space->n_dofs_global());

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

// synchronize copies each owned value to the ranks holding it as a ghost or in their aura, packed
// or not. Packing renumbers this rank's owned-not-shared nodes, and the aura reaches some of them:
// a neighbour addresses them by the position they were created at.
int test_synchronize_after_packing() {
    for (const bool packed : {false, true}) {
        auto mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::world(), 6, 6, 6, 0, 0, 0, 1, 1, 1);
        auto space = sfem::FunctionSpace::create(mesh, 3);
        if (packed) SFEM_TEST_ASSERT(space->initialize_packed_mesh() == SFEM_SUCCESS);

        // A smooth field of the coordinates: on the owned nodes, then expected everywhere.
        auto            m = space->mesh();
        auto            p = m->points()->data();
        const ptrdiff_t n = space->n_dofs();
        auto field        = [&](const ptrdiff_t node, const int d) { return std::sin(7 * p[0][node] + 3 * p[1][node] + d) * p[2][node]; };
        std::vector<real_t> x(n, 0);
        for (ptrdiff_t i = 0; i < space->n_dofs_owned(); ++i) x[i] = field(i / 3, int(i % 3));
        SFEM_TEST_ASSERT(space->synchronize(x.data()) == SFEM_SUCCESS);
        int wrong = 0;
        for (ptrdiff_t i = 0; i < n; ++i) wrong += std::abs(x[i] - field(i / 3, int(i % 3))) > 1e-14;
        SFEM_TEST_EQ(wrong, 0);

#ifdef SFEM_ENABLE_CUDA
        // The same gather of device memory, by GPU-aware MPI: the host's values, bitwise.
        std::vector<real_t> y(n, 0);
        for (ptrdiff_t i = 0; i < space->n_dofs_owned(); ++i) y[i] = field(i / 3, int(i % 3));
        auto on_device = smesh::to_device(sfem::Buffer<real_t>::wrap(n, y.data()));
        SFEM_TEST_ASSERT(space->synchronize(on_device->data(), sfem::EXECUTION_SPACE_DEVICE) == SFEM_SUCCESS);
        auto back   = smesh::to_host(on_device);
        int  differ = 0;
        for (ptrdiff_t i = 0; i < n; ++i) differ += back->data()[i] != x[i];
        SFEM_TEST_EQ(differ, 0);
#endif
    }
    return SFEM_TEST_SUCCESS;
}

// A solver's dot products and its step bound reduced in one message: the sums of the first
// entries, the minimum of the last, on every rank, for f64 and f32.
int test_sum_and_min_last() {
    auto      comm = sfem::Communicator::world();
    const int r = comm->rank(), p = comm->size();
    double    d[4] = {1.0 * r, 1.0, 2.5, 10.0 - r};
    float     f[3] = {2.0f * r, 0.5f, 3.0f + r};
    comm->sum_and_min_last(d, 4, smesh::SMESH_FLOAT64);
    comm->sum_and_min_last(f, 3, smesh::SMESH_FLOAT32);
    SFEM_TEST_ASSERT(d[0] == p * (p - 1) / 2.0 && d[1] == p && d[2] == 2.5 * p && d[3] == 10.0 - (p - 1));
    SFEM_TEST_ASSERT(f[0] == float(p * (p - 1)) && f[1] == 0.5f * p && f[2] == 3.0f);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_parallel_function_space_hex8_cube);
    SFEM_RUN_TEST(test_parallel_function_space_checkerboard);
    SFEM_RUN_TEST(test_parallel_function_space_hex8_tet4);
    SFEM_RUN_TEST(test_parallel_function_space_tet4_cube);
    SFEM_RUN_TEST(test_synchronize_after_packing);
    SFEM_RUN_TEST(test_sum_and_min_last);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}


