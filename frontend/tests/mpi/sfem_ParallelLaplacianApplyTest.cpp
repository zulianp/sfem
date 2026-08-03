#include "sfem_test.hpp"

#include "sfem_API.hpp"

#include "smesh_base.hpp"
#include "smesh_env.hpp"
#include "smesh_mesh.hpp"
#include "smesh_mesh_reorder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>

namespace {

    real_t nodal_field(const geom_t x, const geom_t y, const geom_t z) {
        return x * x + real_t(0.5) * y * y - real_t(0.25) * z * z + real_t(0.125) * x * y + real_t(0.0625) * y * z;
    }

    void make_mesh_nonuniform(const std::shared_ptr<sfem::Mesh> &mesh) {
        auto            points = mesh->points()->data();
        const ptrdiff_t nnodes = mesh->n_nodes();

        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            const geom_t x = points[0][i];
            const geom_t y = points[1][i];
            const geom_t z = points[2][i];

            points[0][i] = x + geom_t(0.05) * y * y + geom_t(0.025) * z;
            points[1][i] = y + geom_t(0.04) * x * z + geom_t(0.015) * x;
            points[2][i] = z + geom_t(0.03) * x * x + geom_t(0.02) * y;
        }
    }

    int test_parallel_laplacian_apply() {
        auto comm = sfem::Communicator::world();

        // SFEM_TEST_ASSERT(comm->size() > 1);

        const smesh::Path mesh_path("/private/tmp/sfem_parallel_laplacian_apply_mesh");

        if (comm->rank() == 0) {
            auto mesh       = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 80, 60, 40, -1, -0.5, 0.25, 1, 1.5, 2.0);
            auto reordering = smesh::SFC::create_from_env();
            reordering->reorder(*mesh);
            make_mesh_nonuniform(mesh);
            SFEM_TEST_ASSERT(mesh->write(mesh_path) == SFEM_SUCCESS);
        }

        comm->barrier();

        auto serial_mesh   = sfem::Mesh::create_from_file(sfem::Communicator::self(), mesh_path);
        auto parallel_mesh = sfem::Mesh::create_from_file(comm, mesh_path);

        SFEM_TEST_ASSERT(serial_mesh != nullptr);
        SFEM_TEST_ASSERT(parallel_mesh != nullptr);
        SFEM_TEST_EQ(serial_mesh->n_blocks(), parallel_mesh->n_blocks());
        SFEM_TEST_EQ(serial_mesh->element_type(0), parallel_mesh->element_type(0));

        if (comm->size() > 1) {
            auto dist = parallel_mesh->distributed();
            SFEM_TEST_ASSERT(dist->n_elements_global() == serial_mesh->n_elements());
            SFEM_TEST_ASSERT(dist->n_nodes_global() == serial_mesh->n_nodes());

            const ptrdiff_t local_aura_nodes    = dist->n_nodes_aura();
            const ptrdiff_t local_aura_elements = dist->n_elements_ghosts();
            const ptrdiff_t global_aura_nodes   = comm->sum(local_aura_nodes);
            const ptrdiff_t global_aura_elems   = comm->sum(local_aura_elements);
            SFEM_TEST_ASSERT(global_aura_nodes > 0);
            SFEM_TEST_ASSERT(global_aura_elems > 0);
        }

        auto serial_space   = sfem::FunctionSpace::create(serial_mesh, 1);
        auto parallel_space = sfem::FunctionSpace::create(parallel_mesh, 1);

        auto serial_function   = sfem::Function::create(serial_space);
        auto parallel_function = sfem::Function::create(parallel_space);

        auto serial_laplacian   = sfem::create_op(serial_space, "Laplacian", sfem::EXECUTION_SPACE_HOST);
        auto parallel_laplacian = sfem::create_op(parallel_space, "Laplacian", sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(serial_laplacian != nullptr);
        SFEM_TEST_ASSERT(parallel_laplacian != nullptr);
        SFEM_TEST_ASSERT(serial_laplacian->initialize() == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(parallel_laplacian->initialize() == SFEM_SUCCESS);

        serial_function->add_operator(serial_laplacian);
        parallel_function->add_operator(parallel_laplacian);

        auto parallel_op =
                sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, parallel_function, nullptr, sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(parallel_op != nullptr);

        const ptrdiff_t n_serial_dofs = serial_space->n_dofs();
        ptrdiff_t       n_owned_dofs  = n_serial_dofs;
        if (comm->size() > 1) {
            auto dist    = parallel_mesh->distributed();
            n_owned_dofs = dist->n_nodes_owned();
            SFEM_TEST_EQ(parallel_op->rows(), n_owned_dofs);
            SFEM_TEST_EQ(parallel_op->cols(), n_owned_dofs);
        }

        auto serial_h = sfem::create_host_buffer<real_t>(n_serial_dofs);
        auto serial_y = sfem::create_host_buffer<real_t>(n_serial_dofs);

        auto serial_points = serial_mesh->points()->data();
        for (ptrdiff_t i = 0; i < n_serial_dofs; ++i) {
            serial_h->data()[i] = nodal_field(serial_points[0][i], serial_points[1][i], serial_points[2][i]);
            serial_y->data()[i] = 0;
        }

        SFEM_TEST_ASSERT(serial_function->apply(nullptr, serial_h->data(), serial_y->data()) == SFEM_SUCCESS);

        auto parallel_h = sfem::create_host_buffer<real_t>(n_owned_dofs);
        auto parallel_y = sfem::create_host_buffer<real_t>(n_owned_dofs);

        if (comm->size() > 1) {
            auto dist         = parallel_mesh->distributed();
            auto node_mapping = dist->node_mapping()->data();
            for (ptrdiff_t i = 0; i < n_owned_dofs; ++i) {
                const ptrdiff_t global_node = static_cast<ptrdiff_t>(node_mapping[i]);
                parallel_h->data()[i]       = serial_h->data()[global_node];
                parallel_y->data()[i]       = 0;
            }
        } else {
            std::memcpy(parallel_h->data(), serial_h->data(), sizeof(real_t) * n_owned_dofs);
            std::memcpy(parallel_y->data(), serial_y->data(), sizeof(real_t) * n_owned_dofs);
        }

        SFEM_TEST_ASSERT(parallel_op->apply(parallel_h->data(), parallel_y->data()) == SFEM_SUCCESS);

        const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-10) : real_t(1e-5);

        if (comm->size() > 1) {
            auto dist         = parallel_mesh->distributed();
            auto node_mapping = dist->node_mapping()->data();
            for (ptrdiff_t i = 0; i < n_owned_dofs; ++i) {
                const ptrdiff_t global_node = static_cast<ptrdiff_t>(node_mapping[i]);
                SFEM_TEST_APPROXEQ(parallel_y->data()[i], serial_y->data()[global_node], tol);
            }
        }

        const int bench_reps  = smesh::Env::read<int>("SFEM_PARALLEL_LAPLACIAN_BENCH_REPS", 20);
        const int warmup_reps = smesh::Env::read<int>("SFEM_PARALLEL_LAPLACIAN_WARMUP_REPS", 2);

        double serial_elapsed = 0;
        if (comm->rank() == 0) {
            for (int r = 0; r < warmup_reps; ++r) {
                std::fill(serial_y->data(), serial_y->data() + n_serial_dofs, real_t(0));
                SFEM_TEST_ASSERT(serial_function->apply(nullptr, serial_h->data(), serial_y->data()) == SFEM_SUCCESS);
            }

            const double tick = smesh::time_seconds();
            for (int r = 0; r < bench_reps; ++r) {
                std::fill(serial_y->data(), serial_y->data() + n_serial_dofs, real_t(0));
                SFEM_TEST_ASSERT(serial_function->apply(nullptr, serial_h->data(), serial_y->data()) == SFEM_SUCCESS);
            }
            serial_elapsed = smesh::time_seconds() - tick;
        }

        comm->barrier();
        for (int r = 0; r < warmup_reps; ++r) {
            SFEM_TEST_ASSERT(parallel_op->apply(parallel_h->data(), parallel_y->data()) == SFEM_SUCCESS);
        }

        comm->barrier();
        const double parallel_tick = smesh::time_seconds();
        for (int r = 0; r < bench_reps; ++r) {
            SFEM_TEST_ASSERT(parallel_op->apply(parallel_h->data(), parallel_y->data()) == SFEM_SUCCESS);
        }
        comm->barrier();
        const double parallel_elapsed = smesh::time_seconds() - parallel_tick;

        if (comm->rank() == 0) {
            const ptrdiff_t n_global_dofs = (comm->size() > 1) ? parallel_mesh->distributed()->n_nodes_global() : n_serial_dofs;
            const double    serial_rate   = static_cast<double>(n_serial_dofs) * static_cast<double>(bench_reps) / serial_elapsed;
            const double parallel_rate = static_cast<double>(n_global_dofs) * static_cast<double>(bench_reps) / parallel_elapsed;
            std::printf("Laplacian apply(%ld) DOF rates: serial %.6e dof/s, parallel %.6e dof/s, speedup %.3f, reps %d\n",
                        n_global_dofs,
                        serial_rate,
                        parallel_rate,
                        parallel_rate / serial_rate,
                        bench_reps);
        }

        comm->barrier();
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_parallel_laplacian_apply);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
