#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_ParallelOperator.hpp"

#include "smesh_base.hpp"
#include "smesh_env.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>

namespace {

    // Cube extents passed to Mesh::create_hex8_cube.
    constexpr geom_t k_x_min = -1;
    constexpr geom_t k_x_max = 1;

    void add_left_right_dirichlet(const std::shared_ptr<sfem::Function> &f) {
        auto mesh = f->space()->mesh_ptr();

        // Use global geometry bounds (not local AABB): required for distributed meshes.
        // Keep the mesh axis-aligned so face barycenters stay on x=±1.
        auto left_sideset =
                sfem::Sideset::create_from_selector(mesh, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool {
                    return x > (k_x_min - 1e-5) && x < (k_x_min + 1e-5);
                });
        auto right_sideset =
                sfem::Sideset::create_from_selector(mesh, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool {
                    return x > (k_x_max - 1e-5) && x < (k_x_max + 1e-5);
                });

        sfem::DirichletConditions::Condition left{.sidesets = left_sideset, .value = -1, .component = 0};
        sfem::DirichletConditions::Condition right{.sidesets = right_sideset, .value = 1, .component = 0};
        auto conds = sfem::create_dirichlet_conditions(f->space(), {left, right}, sfem::EXECUTION_SPACE_HOST);
        f->add_constraint(conds);
    }

    int build_laplacian_function(const std::shared_ptr<sfem::Mesh> &mesh, std::shared_ptr<sfem::Function> &function) {
        auto space   = sfem::FunctionSpace::create(mesh, 1);
        function     = sfem::Function::create(space);
        auto laplace = sfem::create_op(space, "Laplacian", sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(laplace != nullptr);
        SFEM_TEST_ASSERT(laplace->initialize() == SFEM_SUCCESS);
        function->add_operator(laplace);
        add_left_right_dirichlet(function);
        return SFEM_TEST_SUCCESS;
    }

    int test_parallel_cg() {
        auto comm = sfem::Communicator::world();

        bool verbose = smesh::Env::read("SFEM_VERBOSE", false);

        // Axis-aligned cube: exact solution of -Δu=0 with u(±1)=±1 is u(x)=x.
        ptrdiff_t factor        = smesh::Env::read("SFEM_MESH_FACTOR", 2);
        auto      parallel_mesh = sfem::Mesh::create_hex8_cube(
                comm, factor * 48, factor * 40, factor * 32, k_x_min, -0.5, 0.25, k_x_max, 1.5, 2.0);
        SFEM_TEST_ASSERT(parallel_mesh != nullptr);

        if (comm->size() > 1) {
            SFEM_TEST_ASSERT(parallel_mesh->is_distributed());
            auto dist = parallel_mesh->distributed();
            SFEM_TEST_ASSERT(dist->n_nodes_global() > 0);
            SFEM_TEST_ASSERT(comm->sum(dist->n_nodes_aura()) > 0);
        } else {
            SFEM_TEST_ASSERT(!parallel_mesh->is_distributed());
        }

        std::shared_ptr<sfem::Function> parallel_function;
        SFEM_TEST_ASSERT(build_laplacian_function(parallel_mesh, parallel_function) == SFEM_TEST_SUCCESS);

        auto parallel_op = sfem::create_parallel_matrix_free_operator(parallel_function, nullptr, sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(parallel_op != nullptr);
        SFEM_TEST_ASSERT(parallel_op->comm() != nullptr);
        SFEM_TEST_EQ(parallel_op->comm()->size(), comm->size());

        const ptrdiff_t n_owned   = parallel_op->rows();
        const ptrdiff_t n_x_alloc = parallel_op->col_allocation_size();
        const ptrdiff_t n_b_alloc = parallel_op->row_allocation_size();
        SFEM_TEST_ASSERT(n_x_alloc >= n_owned);
        SFEM_TEST_ASSERT(n_b_alloc >= n_owned);

        // x/b must be allocation-sized: apply_constraints may touch ghost/aura indices from sidesets,
        // and Parallel CG apply gathers into x in place.
        auto parallel_x   = sfem::create_host_buffer<real_t>(n_x_alloc);
        auto parallel_rhs = sfem::create_host_buffer<real_t>(n_b_alloc);
        parallel_function->apply_constraints(parallel_x->data());
        parallel_function->apply_constraints(parallel_rhs->data());

        {
            real_t local_x2 = 0;
            for (ptrdiff_t i = 0; i < n_owned; ++i) {
                local_x2 += parallel_x->data()[i] * parallel_x->data()[i];
            }
            SFEM_TEST_ASSERT(comm->sum(local_x2) > real_t(0));
        }

        auto parallel_cg = sfem::create_parallel_cg<real_t>(parallel_op);
        SFEM_TEST_ASSERT(parallel_cg != nullptr);
        parallel_cg->set_verbose(verbose);
        parallel_cg->set_max_it(5000);
        parallel_cg->set_rtol(1e-10);
        parallel_cg->set_atol(1e-12);

        {
            comm->barrier();
            SFEM_TRACE_SCOPE("parallel_cg::apply");
            SFEM_TEST_ASSERT(parallel_cg->apply(parallel_rhs->data(), parallel_x->data()) == SFEM_SUCCESS);
            comm->barrier();
        }

        SFEM_TEST_ASSERT(parallel_cg->iterations() > 0);

        const real_t tol       = sizeof(real_t) == sizeof(double) ? real_t(1e-6) : real_t(1e-4);
        auto         points    = parallel_mesh->points()->data();
        real_t       local_err = 0;
        for (ptrdiff_t i = 0; i < n_owned; ++i) {
            local_err = std::max(local_err, std::abs(parallel_x->data()[i] - real_t(points[0][i])));
            SFEM_TEST_APPROXEQ(parallel_x->data()[i], real_t(points[0][i]), tol);
        }
        const real_t    global_err    = comm->max(local_err);
        const ptrdiff_t global_n_dofs = comm->sum(parallel_op->rows());

        if (comm->rank() == 0 && verbose) {
            printf("ParallelCGTest: iterations=%d max_err=%g (nprocs=%d), ndofs=%ld\n",
                   parallel_cg->iterations(),
                   (double)global_err,
                   comm->size(),
                   (ptrdiff_t)global_n_dofs);
        }

        comm->barrier();
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_parallel_cg);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}

