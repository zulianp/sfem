#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_ElementScope.hpp"
#include "sfem_ParallelOperator.hpp"

#include "smesh_base.hpp"
#include "smesh_env.hpp"
#include "smesh_exchange.hpp"
#include "smesh_mesh.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_BLAS_cuda.hpp"
#include "smesh_device_buffer.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

    real_t nodal_field(const geom_t x, const geom_t y, const geom_t z) {
        return x * x + real_t(0.5) * y * y - real_t(0.25) * z * z + real_t(0.125) * x * y + real_t(0.0625) * y * z;
    }

    void fill_nodal_field(const std::shared_ptr<sfem::Mesh> &mesh, real_t *const values, const ptrdiff_t n, const int bs = 1) {
        auto points = mesh->points()->data();
        for (ptrdiff_t i = 0; i < n; ++i) {
            for (int c = 0; c < bs; ++c) {
                values[i * bs + c] = real_t(1 + c) * nodal_field(points[0][i], points[1][i], points[2][i]) + real_t(0.25) * c;
            }
        }
    }

    int check_owned_geometry(const std::shared_ptr<sfem::Communicator> &comm,
                             const std::shared_ptr<sfem::Mesh>         &serial_mesh,
                             const std::shared_ptr<sfem::Mesh>         &parallel_mesh,
                             const ptrdiff_t                            n_owned) {
        if (comm->size() <= 1) {
            return SFEM_TEST_SUCCESS;
        }

        auto         dist         = parallel_mesh->distributed();
        auto         node_mapping = dist->node_mapping()->data();
        auto         serial_pts   = serial_mesh->points()->data();
        auto         parallel_pts = parallel_mesh->points()->data();
        const real_t geom_tol     = sizeof(geom_t) == sizeof(double) ? real_t(1e-12) : real_t(1e-5);
        for (ptrdiff_t i = 0; i < n_owned; ++i) {
            const ptrdiff_t g = static_cast<ptrdiff_t>(node_mapping[i]);
            SFEM_TEST_APPROXEQ(parallel_pts[0][i], serial_pts[0][g], geom_tol);
            SFEM_TEST_APPROXEQ(parallel_pts[1][i], serial_pts[1][g], geom_tol);
            SFEM_TEST_APPROXEQ(parallel_pts[2][i], serial_pts[2][g], geom_tol);
        }
        return SFEM_TEST_SUCCESS;
    }

    int compare_owned_to_serial(const std::shared_ptr<sfem::Communicator> &comm,
                                const std::shared_ptr<sfem::Mesh>         &parallel_mesh,
                                const real_t *const                        parallel_y,
                                const real_t *const                        serial_y,
                                const ptrdiff_t                            n_owned_dofs,
                                const ptrdiff_t                            n_serial_dofs,
                                const real_t                               tol,
                                const int                                  bs = 1) {
        real_t local_sum = 0;
        for (ptrdiff_t i = 0; i < n_owned_dofs; ++i) {
            local_sum += parallel_y[i];
        }
        const real_t parallel_sum = comm->sum(local_sum);
        real_t       serial_sum   = 0;
        if (comm->rank() == 0) {
            for (ptrdiff_t i = 0; i < n_serial_dofs; ++i) {
                serial_sum += serial_y[i];
            }
        }
        comm->broadcast(&serial_sum, 1, 0);
        SFEM_TEST_APPROXEQ(parallel_sum, serial_sum, tol * static_cast<real_t>(std::max<ptrdiff_t>(n_serial_dofs, 1)));

        if (comm->size() > 1) {
            auto dist         = parallel_mesh->distributed();
            auto node_mapping = dist->node_mapping()->data();
            for (ptrdiff_t i = 0; i < n_owned_dofs; ++i) {
                const ptrdiff_t global_dof = static_cast<ptrdiff_t>(node_mapping[i / bs]) * bs + i % bs;
                SFEM_TEST_APPROXEQ(parallel_y[i], serial_y[global_dof], tol);
            }
        } else {
            for (ptrdiff_t i = 0; i < n_owned_dofs; ++i) {
                SFEM_TEST_APPROXEQ(parallel_y[i], serial_y[i], tol);
            }
        }
        return SFEM_TEST_SUCCESS;
    }

    int check_parallel_laplacian_apply(const std::shared_ptr<sfem::Mesh> &serial_mesh,
                                       const std::shared_ptr<sfem::Mesh> &parallel_mesh,
                                       const sfem::ExecutionSpace         es,
                                       const bool                         bench,
                                       const char                        *op_name = "Laplacian",
                                       const int                          bs      = 1) {
        auto comm = sfem::Communicator::world();

        SFEM_TEST_ASSERT(serial_mesh != nullptr);
        SFEM_TEST_ASSERT(parallel_mesh != nullptr);
        SFEM_TEST_EQ(serial_mesh->n_blocks(), parallel_mesh->n_blocks());
        for (size_t b = 0; b < serial_mesh->n_blocks(); ++b) {
            SFEM_TEST_EQ(parallel_mesh->element_type(static_cast<smesh::block_idx_t>(b)),
                         serial_mesh->element_type(static_cast<smesh::block_idx_t>(b)));
        }

        if (comm->size() > 1) {
            SFEM_TEST_ASSERT(parallel_mesh->is_distributed());
            auto dist = parallel_mesh->distributed();
            SFEM_TEST_ASSERT(dist->n_elements_global() == serial_mesh->n_elements());
            SFEM_TEST_ASSERT(dist->n_nodes_global() == serial_mesh->n_nodes());

            const ptrdiff_t global_aura_nodes = comm->sum(dist->n_nodes_aura());
            const ptrdiff_t global_aura_elems = comm->sum(dist->n_elements_ghosts());
            SFEM_TEST_ASSERT(global_aura_nodes > 0);
            SFEM_TEST_ASSERT(global_aura_elems > 0);
        } else {
            SFEM_TEST_ASSERT(!parallel_mesh->is_distributed());
        }

        auto serial_space   = sfem::FunctionSpace::create(serial_mesh, bs);
        auto parallel_space = sfem::FunctionSpace::create(parallel_mesh, bs);

        auto serial_function   = sfem::Function::create(serial_space);
        auto parallel_function = sfem::Function::create(parallel_space);

        auto serial_laplacian   = sfem::create_op(serial_space, op_name, sfem::EXECUTION_SPACE_HOST);
        auto parallel_laplacian = sfem::create_op(parallel_space, op_name, es);
        SFEM_TEST_ASSERT(serial_laplacian != nullptr);
        SFEM_TEST_ASSERT(parallel_laplacian != nullptr);
        SFEM_TEST_ASSERT(serial_laplacian->initialize() == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(parallel_laplacian->initialize() == SFEM_SUCCESS);

        serial_function->add_operator(serial_laplacian);
        parallel_function->add_operator(parallel_laplacian);

        auto parallel_op = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, parallel_function, nullptr, es);
        SFEM_TEST_ASSERT(parallel_op != nullptr);

        auto *pop = dynamic_cast<sfem::ParallelOperator<real_t> *>(parallel_op.get());
        SFEM_TEST_ASSERT(pop != nullptr);

        const ptrdiff_t n_serial_dofs = serial_space->n_dofs();
        const ptrdiff_t n_owned_dofs  = pop->rows();
        const ptrdiff_t n_x_alloc     = pop->col_allocation_size();
        const ptrdiff_t n_y_alloc     = pop->row_allocation_size();
        SFEM_TEST_ASSERT(n_x_alloc >= n_owned_dofs);
        SFEM_TEST_ASSERT(n_y_alloc >= n_owned_dofs);

        if (comm->size() > 1) {
            auto dist = parallel_mesh->distributed();
            SFEM_TEST_EQ(n_owned_dofs, dist->n_nodes_owned() * bs);
            SFEM_TEST_EQ(parallel_op->cols(), n_owned_dofs);
            SFEM_TEST_ASSERT(check_owned_geometry(comm, serial_mesh, parallel_mesh, n_owned_dofs / bs) == SFEM_TEST_SUCCESS);
        }

        auto serial_h = sfem::create_host_buffer<real_t>(n_serial_dofs);
        auto serial_y = sfem::create_host_buffer<real_t>(n_serial_dofs);
        fill_nodal_field(serial_mesh, serial_h->data(), n_serial_dofs / bs, bs);
        std::fill(serial_y->data(), serial_y->data() + n_serial_dofs, real_t(0));

        SFEM_TEST_ASSERT(serial_function->apply(nullptr, serial_h->data(), serial_y->data()) == SFEM_SUCCESS);

        auto parallel_h_host = sfem::create_host_buffer<real_t>(n_x_alloc);
        auto parallel_y_host = sfem::create_host_buffer<real_t>(n_y_alloc);
        std::fill(parallel_h_host->data(), parallel_h_host->data() + n_x_alloc, real_t(0));
        std::fill(parallel_y_host->data(), parallel_y_host->data() + n_y_alloc, real_t(0));

        if (comm->size() > 1) {
            auto dist         = parallel_mesh->distributed();
            auto node_mapping = dist->node_mapping()->data();
            for (ptrdiff_t i = 0; i < n_owned_dofs; ++i) {
                const ptrdiff_t global_dof = static_cast<ptrdiff_t>(node_mapping[i / bs]) * bs + i % bs;
                parallel_h_host->data()[i] = serial_h->data()[global_dof];
            }
        } else {
            std::memcpy(parallel_h_host->data(), serial_h->data(), sizeof(real_t) * n_owned_dofs);
        }

        std::shared_ptr<sfem::Buffer<real_t>> parallel_h;
        std::shared_ptr<sfem::Buffer<real_t>> parallel_y;

        if (es == sfem::EXECUTION_SPACE_DEVICE) {
#ifdef SFEM_ENABLE_CUDA
            parallel_h = smesh::create_device_buffer<real_t>(n_x_alloc);
            parallel_y = smesh::create_device_buffer<real_t>(n_y_alloc);
            copy_host_to_device((size_t)n_x_alloc * sizeof(real_t), parallel_h_host->data(), parallel_h->data());
            copy_host_to_device((size_t)n_y_alloc * sizeof(real_t), parallel_y_host->data(), parallel_y->data());
#else
            SFEM_TEST_ASSERT(false && "device test requires CUDA");
#endif
        } else {
            parallel_h = parallel_h_host;
            parallel_y = parallel_y_host;
        }

        auto reset_parallel_output = [&]() {
            std::fill(parallel_y_host->data(), parallel_y_host->data() + n_y_alloc, real_t(0));
            if (es == sfem::EXECUTION_SPACE_DEVICE) {
#ifdef SFEM_ENABLE_CUDA
                copy_host_to_device((size_t)n_y_alloc * sizeof(real_t), parallel_y_host->data(), parallel_y->data());
#endif
            }
        };

        reset_parallel_output();
        SFEM_TEST_ASSERT(parallel_op->apply(parallel_h->data(), parallel_y->data()) == SFEM_SUCCESS);

        if (es == sfem::EXECUTION_SPACE_DEVICE) {
#ifdef SFEM_ENABLE_CUDA
            copy_device_to_host((size_t)n_y_alloc * sizeof(real_t), parallel_y->data(), parallel_y_host->data());
#endif
        }

        // Rank-split vs serial assembly order: ~1e-8 abs; 1e-7 still catches real bugs.
        const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-7) : real_t(1e-4);
        SFEM_TEST_ASSERT(
                compare_owned_to_serial(
                        comm, parallel_mesh, parallel_y_host->data(), serial_y->data(), n_owned_dofs, n_serial_dofs, tol, bs) ==
                SFEM_TEST_SUCCESS);

        if (!bench) {
            return SFEM_TEST_SUCCESS;
        }

        const int bench_reps  = smesh::Env::read<int>("SFEM_PARALLEL_LAPLACIAN_BENCH_REPS", 40);
        const int warmup_reps = smesh::Env::read<int>("SFEM_PARALLEL_LAPLACIAN_WARMUP_REPS", 3);

        auto time_parallel_apply = [&]() -> double {
            comm->barrier();
            for (int r = 0; r < warmup_reps; ++r) {
                SFEM_TEST_ASSERT(parallel_op->apply(parallel_h->data(), parallel_y->data()) == SFEM_SUCCESS);
            }
            comm->barrier();
            const double tick = smesh::time_seconds();
            for (int r = 0; r < bench_reps; ++r) {
                SFEM_TEST_ASSERT(parallel_op->apply(parallel_h->data(), parallel_y->data()) == SFEM_SUCCESS);
            }
            comm->barrier();
            return smesh::time_seconds() - tick;
        };

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

        const double parallel_elapsed = time_parallel_apply();

        if (comm->rank() == 0) {
            const ptrdiff_t n_global_dofs = (comm->size() > 1) ? parallel_mesh->distributed()->n_nodes_global() * bs : n_serial_dofs;
            const double    serial_rate   = static_cast<double>(n_serial_dofs) * static_cast<double>(bench_reps) / serial_elapsed;
            const double    parallel_rate = static_cast<double>(n_global_dofs) * static_cast<double>(bench_reps) / parallel_elapsed;
#ifdef _OPENMP
            const int omp_max = omp_get_max_threads();
#else
            const int omp_max = 1;
#endif
            std::printf(
                    "%s apply(%ld) [%s] ranks=%d omp_max_threads=%d reps=%d n_blocks=%ld\n"
                    "  serial:  %.6e dof/s  (%.6f s)\n"
                    "  parallel: %.6e dof/s  (%.6f s)  speedup_vs_serial %.3f\n",
                    op_name,
                    (long)n_global_dofs,
                    es == sfem::EXECUTION_SPACE_DEVICE ? "device" : "host",
                    comm->size(),
                    omp_max,
                    bench_reps,
                    (long)parallel_mesh->n_blocks(),
                    serial_rate,
                    serial_elapsed,
                    parallel_rate,
                    parallel_elapsed,
                    parallel_rate / serial_rate);
        }

        comm->barrier();
        return SFEM_TEST_SUCCESS;
    }

    int test_parallel_laplacian_apply_hex8_cube() {
        auto            comm = sfem::Communicator::world();
        const ptrdiff_t nx = 8, ny = 6, nz = 4;
        auto serial   = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), nx, ny, nz, -1, -0.5, 0.25, 1, 1.5, 2.0);
        auto parallel = sfem::Mesh::create_hex8_cube(comm, nx, ny, nz, -1, -0.5, 0.25, 1, 1.5, 2.0);
        return check_parallel_laplacian_apply(serial, parallel, sfem::EXECUTION_SPACE_HOST, true);
    }

    int test_parallel_laplacian_apply_checkerboard() {
        auto            comm = sfem::Communicator::world();
        const ptrdiff_t nx = 16, ny = 12, nz = 8;
        auto            serial   = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), nx, ny, nz);
        auto            parallel = sfem::Mesh::create_hex8_checkerboard_cube(comm, nx, ny, nz);
        SFEM_TEST_ASSERT(serial != nullptr);
        SFEM_TEST_ASSERT(parallel != nullptr);
        SFEM_TEST_EQ(parallel->n_blocks(), static_cast<size_t>(2));
        SFEM_TEST_ASSERT(parallel->block(0)->name() == "white");
        SFEM_TEST_ASSERT(parallel->block(1)->name() == "black");
        SFEM_TEST_EQ(parallel->element_type(0), smesh::HEX8);
        SFEM_TEST_EQ(parallel->element_type(1), smesh::HEX8);
        return check_parallel_laplacian_apply(serial, parallel, sfem::EXECUTION_SPACE_HOST, false);
    }

    int test_parallel_laplacian_apply_hex8_tet4() {
        auto            comm = sfem::Communicator::world();
        const ptrdiff_t nx = 8, ny = 6, nz = 4;
        auto            serial   = sfem::Mesh::create_hex8_tet4_cube(sfem::Communicator::self(), nx, ny, nz);
        auto            parallel = sfem::Mesh::create_hex8_tet4_cube(comm, nx, ny, nz);
        SFEM_TEST_ASSERT(serial != nullptr);
        SFEM_TEST_ASSERT(parallel != nullptr);
        SFEM_TEST_EQ(parallel->n_blocks(), static_cast<size_t>(2));
        SFEM_TEST_EQ(parallel->element_type(0), smesh::HEX8);
        SFEM_TEST_EQ(parallel->element_type(1), smesh::TET4);
        return check_parallel_laplacian_apply(serial, parallel, sfem::EXECUTION_SPACE_HOST, false);
    }

    // The generated operators' team form (Op::team_apply over an ElementScope): the elements away
    // from the interface while the halo is in flight, then the others.
    int test_parallel_generated_laplace_apply_hex8_cube() {
        auto            comm = sfem::Communicator::world();
        const ptrdiff_t nx = 8, ny = 6, nz = 4;
        auto serial   = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), nx, ny, nz, -1, -0.5, 0.25, 1, 1.5, 2.0);
        auto parallel = sfem::Mesh::create_hex8_cube(comm, nx, ny, nz, -1, -0.5, 0.25, 1, 1.5, 2.0);
        return check_parallel_laplacian_apply(serial, parallel, sfem::EXECUTION_SPACE_HOST, true, "GeneratedLaplace", 1);
    }

    int test_parallel_generated_laplace_apply_hex8_tet4() {
        auto            comm = sfem::Communicator::world();
        const ptrdiff_t nx = 8, ny = 6, nz = 4;
        auto            serial   = sfem::Mesh::create_hex8_tet4_cube(sfem::Communicator::self(), nx, ny, nz);
        auto            parallel = sfem::Mesh::create_hex8_tet4_cube(comm, nx, ny, nz);
        return check_parallel_laplacian_apply(serial, parallel, sfem::EXECUTION_SPACE_HOST, false, "GeneratedLaplace", 1);
    }

    int test_parallel_generated_linear_elasticity_apply_checkerboard() {
        auto            comm = sfem::Communicator::world();
        const ptrdiff_t nx = 16, ny = 12, nz = 8;
        auto            serial   = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), nx, ny, nz);
        auto            parallel = sfem::Mesh::create_hex8_checkerboard_cube(comm, nx, ny, nz);
        return check_parallel_laplacian_apply(
                serial, parallel, sfem::EXECUTION_SPACE_HOST, true, "GeneratedLinearElasticity", 3);
    }

    int test_parallel_generated_linear_elasticity_apply_hex8_tet4() {
        auto            comm = sfem::Communicator::world();
        const ptrdiff_t nx = 8, ny = 6, nz = 4;
        auto            serial   = sfem::Mesh::create_hex8_tet4_cube(sfem::Communicator::self(), nx, ny, nz);
        auto            parallel = sfem::Mesh::create_hex8_tet4_cube(comm, nx, ny, nz);
        return check_parallel_laplacian_apply(
                serial, parallel, sfem::EXECUTION_SPACE_HOST, false, "GeneratedLinearElasticity", 3);
    }

    // The hand-written mass in team form, over the same two element scopes.
    int test_parallel_mass_apply_hex8_cube() {
        auto            comm = sfem::Communicator::world();
        const ptrdiff_t nx = 8, ny = 6, nz = 4;
        auto serial   = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), nx, ny, nz, -1, -0.5, 0.25, 1, 1.5, 2.0);
        auto parallel = sfem::Mesh::create_hex8_cube(comm, nx, ny, nz, -1, -0.5, 0.25, 1, 1.5, 2.0);
        return check_parallel_laplacian_apply(serial, parallel, sfem::EXECUTION_SPACE_HOST, false, "Mass", 1);
    }

    int test_parallel_mass_apply_checkerboard() {
        auto            comm = sfem::Communicator::world();
        const ptrdiff_t nx = 16, ny = 12, nz = 8;
        auto            serial   = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), nx, ny, nz);
        auto            parallel = sfem::Mesh::create_hex8_checkerboard_cube(comm, nx, ny, nz);
        return check_parallel_laplacian_apply(serial, parallel, sfem::EXECUTION_SPACE_HOST, false, "Mass", 1);
    }

    int test_element_scope_identity_checkerboard() {
        auto            comm = sfem::Communicator::world();
        const ptrdiff_t nx = 8, ny = 6, nz = 4;
        auto            mesh = sfem::Mesh::create_hex8_checkerboard_cube(comm, nx, ny, nz);
        SFEM_TEST_ASSERT(mesh != nullptr);
        SFEM_TEST_EQ(mesh->n_blocks(), static_cast<size_t>(2));

        auto space     = sfem::FunctionSpace::create(mesh, 1);
        auto function  = sfem::Function::create(space);
        auto laplacian = sfem::create_op(space, "Laplacian", sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(laplacian != nullptr);
        SFEM_TEST_ASSERT(laplacian->initialize() == SFEM_SUCCESS);
        function->add_operator(laplacian);

        const ptrdiff_t n_local  = space->n_dofs();
        const ptrdiff_t n_owned  = space->n_dofs_owned();
        auto            x        = sfem::create_host_buffer<real_t>(n_local);
        auto            y_all    = sfem::create_host_buffer<real_t>(n_local);
        auto            y_phased = sfem::create_host_buffer<real_t>(n_local);
        std::fill(x->data(), x->data() + n_local, real_t(0));
        fill_nodal_field(mesh, x->data(), n_owned);

        if (mesh->is_distributed()) {
            auto exchange = smesh::Exchange::create_nodal(mesh, smesh::Exchange::ExchangeScope::GhostsAndAura);
            SFEM_TEST_ASSERT(exchange != nullptr);
            SFEM_TEST_ASSERT(exchange->gather(x->data(), 1) == SFEM_SUCCESS);
        }

        std::fill(y_all->data(), y_all->data() + n_local, real_t(0));
        std::fill(y_phased->data(), y_phased->data() + n_local, real_t(0));
        SFEM_TEST_ASSERT(function->apply(nullptr, x->data(), y_all->data(), sfem::ElementScope::ALL) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(function->apply(nullptr, x->data(), y_phased->data(), sfem::ElementScope::OWNED_NOT_SHARED) ==
                         SFEM_SUCCESS);
        SFEM_TEST_ASSERT(function->apply(nullptr, x->data(), y_phased->data(), sfem::ElementScope::SHARED_AND_AURA) ==
                         SFEM_SUCCESS);

        const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-8) : real_t(1e-4);
        for (ptrdiff_t i = 0; i < n_owned; ++i) {
            SFEM_TEST_APPROXEQ(y_phased->data()[i], y_all->data()[i], tol);
        }

        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_parallel_laplacian_apply_hex8_cube);
    SFEM_RUN_TEST(test_parallel_laplacian_apply_checkerboard);
    SFEM_RUN_TEST(test_parallel_laplacian_apply_hex8_tet4);
    SFEM_RUN_TEST(test_parallel_generated_laplace_apply_hex8_cube);
    SFEM_RUN_TEST(test_parallel_generated_laplace_apply_hex8_tet4);
    SFEM_RUN_TEST(test_parallel_generated_linear_elasticity_apply_checkerboard);
    SFEM_RUN_TEST(test_parallel_generated_linear_elasticity_apply_hex8_tet4);
    SFEM_RUN_TEST(test_parallel_mass_apply_hex8_cube);
    SFEM_RUN_TEST(test_parallel_mass_apply_checkerboard);
    SFEM_RUN_TEST(test_element_scope_identity_checkerboard);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
