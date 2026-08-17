#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_ssgmg.hpp"

#include "smesh_base.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_sideset.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

    sfem::SharedBuffer<idx_t> nodeset_from_sidesets(const std::shared_ptr<smesh::Mesh>                 &mesh,
                                                    const std::vector<std::shared_ptr<smesh::Sideset>> &sidesets) {
        std::vector<idx_t> ids;
        for (const auto &ss : sidesets) {
            auto ns = smesh::create_nodeset_from_sideset(mesh, ss);
            if (!ns || ns->size() == 0) {
                continue;
            }
            auto d = ns->data();
            ids.insert(ids.end(), d, d + ns->size());
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

        auto out = sfem::create_host_buffer<idx_t>((ptrdiff_t)ids.size());
        if (!ids.empty()) {
            std::memcpy(out->data(), ids.data(), ids.size() * sizeof(idx_t));
        }
        return out;
    }

    template <typename Pred>
    sfem::SharedBuffer<idx_t> nodeset_from_point_selector(const std::shared_ptr<smesh::Mesh> &mesh, Pred pred) {
        auto               pts = mesh->points()->data();
        const ptrdiff_t    n   = mesh->n_nodes();
        std::vector<idx_t> ids;
        ids.reserve((size_t)n);
        for (ptrdiff_t i = 0; i < n; ++i) {
            if (pred(pts[0][i], pts[1][i], pts[2][i])) {
                ids.push_back((idx_t)i);
            }
        }
        auto out = sfem::create_host_buffer<idx_t>((ptrdiff_t)ids.size());
        if (!ids.empty()) {
            std::memcpy(out->data(), ids.data(), ids.size() * sizeof(idx_t));
        }
        return out;
    }

    sfem::SharedBuffer<idx_t> union_nodesets(const sfem::SharedBuffer<idx_t> &a, const sfem::SharedBuffer<idx_t> &b) {
        std::vector<idx_t> ids;
        if (a && a->size() > 0) {
            ids.insert(ids.end(), a->data(), a->data() + a->size());
        }
        if (b && b->size() > 0) {
            ids.insert(ids.end(), b->data(), b->data() + b->size());
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        auto out = sfem::create_host_buffer<idx_t>((ptrdiff_t)ids.size());
        if (!ids.empty()) {
            std::memcpy(out->data(), ids.data(), ids.size() * sizeof(idx_t));
        }
        return out;
    }

    std::shared_ptr<sfem::Function> make_homogeneous_checkerboard_ss_poisson(const int                          element_level,
                                                                             const std::shared_ptr<sfem::Mesh> &hex) {
        auto ss = smesh::to_semistructured(element_level, hex, true, false);
        if (!ss) {
            return nullptr;
        }

        auto fs = sfem::FunctionSpace::create(ss, 1);
        auto f  = sfem::Function::create(fs);
        auto op = sfem::create_op(fs, "Laplacian", sfem::EXECUTION_SPACE_HOST);
        if (!op || op->initialize() != SFEM_SUCCESS) {
            return nullptr;
        }
        f->add_operator(op);

        auto bottom_pred = [](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > -1e-5 && y < 1e-5; };
        auto right_pred  = [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool {
            return x > 1 - 1e-5 && x < 1 + 1e-5;
        };

        auto bottom_ss = sfem::Sideset::create_from_selector(ss, bottom_pred);
        auto right_ss  = sfem::Sideset::create_from_selector(ss, right_pred);

        // Sideset parent ids from SS create_from_selector come from a throwaway derefined HEX8
        // mesh; on MPI they need not match local SS element order. Union with a point selector
        // so every owned/ghost Dirichlet node is constrained (B5.7 block_id grouping is out of scope).
        sfem::DirichletConditions::Condition left{
                .sidesets  = bottom_ss,
                .nodeset   = union_nodesets(nodeset_from_sidesets(ss, bottom_ss), nodeset_from_point_selector(ss, bottom_pred)),
                .value     = -1,
                .component = 0};
        sfem::DirichletConditions::Condition right{
                .sidesets  = right_ss,
                .nodeset   = union_nodesets(nodeset_from_sidesets(ss, right_ss), nodeset_from_point_selector(ss, right_pred)),
                .value     = 1,
                .component = 0};
        f->add_constraint(sfem::create_dirichlet_conditions(fs, {left, right}, sfem::EXECUTION_SPACE_HOST));
        return f;
    }

    int find_serial_node(const geom_t *const *serial_pts,
                         const ptrdiff_t      n_serial,
                         const geom_t         x,
                         const geom_t         y,
                         const geom_t         z,
                         const geom_t         tol) {
        for (ptrdiff_t j = 0; j < n_serial; ++j) {
            if (std::fabs(serial_pts[0][j] - x) <= tol && std::fabs(serial_pts[1][j] - y) <= tol &&
                std::fabs(serial_pts[2][j] - z) <= tol) {
                return (int)j;
            }
        }
        return -1;
    }

    int test_parallel_checkerboard_ssgmg() {
        SFEM_TRACE_SCOPE("test_parallel_checkerboard_ssgmg");
        auto comm = sfem::Communicator::world();

        const ptrdiff_t n = smesh::Env::read("SFEM_BASE_RESOLUTION", 2);
        const int       l = smesh::Env::read("SFEM_ELEMENT_REFINE_LEVEL", 8);

        auto serial_hex   = sfem::Mesh::create_hex8_checkerboard_cube(sfem::Communicator::self(), n, n, n);
        auto parallel_hex = sfem::Mesh::create_hex8_checkerboard_cube(comm, n, n, n);
        SFEM_TEST_ASSERT(serial_hex != nullptr);
        SFEM_TEST_ASSERT(parallel_hex != nullptr);
        SFEM_TEST_EQ(parallel_hex->n_blocks(), static_cast<size_t>(2));
        SFEM_TEST_ASSERT(parallel_hex->block(0)->name() == "white");
        SFEM_TEST_ASSERT(parallel_hex->block(1)->name() == "black");

        auto serial_f   = make_homogeneous_checkerboard_ss_poisson(l, serial_hex);
        auto parallel_f = make_homogeneous_checkerboard_ss_poisson(l, parallel_hex);
        SFEM_TEST_ASSERT(serial_f != nullptr);
        SFEM_TEST_ASSERT(parallel_f != nullptr);

        auto serial_fs     = serial_f->space();
        auto parallel_fs   = parallel_f->space();
        auto serial_mesh   = serial_fs->mesh_ptr();
        auto parallel_mesh = parallel_fs->mesh_ptr();

        const ptrdiff_t n_global = parallel_fs->n_dofs_global();
        SFEM_TEST_EQ(comm->sum(parallel_fs->n_owned_dofs()), n_global);
        SFEM_TEST_EQ(n_global, serial_fs->n_dofs_global());

        // MPI SS gids are not serial local indices (HO order within a layer differs), so the
        // owned-x check is a coordinate scan. Skip it (and the serial solve) on large runs.
        const int  compare_max_dofs = smesh::Env::read("SFEM_SERIAL_COMPARE_MAX_DOFS", 8192);
        const bool compare_serial   = n_global <= static_cast<ptrdiff_t>(compare_max_dofs);

        sfem::SharedBuffer<real_t> serial_x;
        if (compare_serial) {
            serial_x        = sfem::create_host_buffer<real_t>(serial_fs->n_dofs());
            auto serial_rhs = sfem::create_host_buffer<real_t>(serial_fs->n_dofs());
            SFEM_TEST_ASSERT(serial_f->apply_constraints(serial_x->data()) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(serial_f->apply_constraints(serial_rhs->data()) == SFEM_SUCCESS);

            auto serial_mg = sfem::create_ssgmg(serial_f, serial_f->execution_space());
            SFEM_TEST_ASSERT(serial_mg != nullptr);
            serial_mg->verbose = false;
            SFEM_TEST_ASSERT(serial_mg->apply(serial_rhs->data(), serial_x->data()) == SFEM_SUCCESS);
        } else if (comm->rank() == 0) {
            printf("skipping serial solution comparison (%td dofs > %d)\n", (ptrdiff_t)n_global, compare_max_dofs);
        }

        auto parallel_x   = sfem::create_host_buffer<real_t>(parallel_fs->n_dofs());
        auto parallel_rhs = sfem::create_host_buffer<real_t>(parallel_fs->n_dofs());
        SFEM_TEST_ASSERT(parallel_f->apply_constraints(parallel_x->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(parallel_f->apply_constraints(parallel_rhs->data()) == SFEM_SUCCESS);

        auto parallel_mg = sfem::create_ssgmg(parallel_f, parallel_f->execution_space());
        SFEM_TEST_ASSERT(parallel_mg != nullptr);
        parallel_mg->verbose = smesh::Env::read("SFEM_VERBOSE", false);
        SFEM_TEST_ASSERT(parallel_mg->apply(parallel_rhs->data(), parallel_x->data()) == SFEM_SUCCESS);

        auto A  = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, parallel_f, nullptr, parallel_f->execution_space());
        auto ax = sfem::create_host_buffer<real_t>(parallel_fs->n_dofs());
        SFEM_TEST_ASSERT(A->apply(parallel_x->data(), ax->data()) == SFEM_SUCCESS);

        const ptrdiff_t n_owned  = parallel_fs->n_owned_dofs();
        real_t          local_r2 = 0;
        real_t          local_b2 = 0;
        for (ptrdiff_t i = 0; i < n_owned; ++i) {
            const real_t r = parallel_rhs->data()[i] - ax->data()[i];
            local_r2 += r * r;
            local_b2 += parallel_rhs->data()[i] * parallel_rhs->data()[i];
        }
        const real_t abs_res = std::sqrt(comm->sum(local_r2));
        const real_t rhs_nrm = std::sqrt(comm->sum(local_b2));
        const real_t rel_res = abs_res / (rhs_nrm + real_t(1e-16));
        if (comm->rank() == 0) {
            printf("parallel checkerboard ssgmg residual abs %g rel %g\n", (double)abs_res, (double)rel_res);
        }

        const real_t abs_tol = sizeof(real_t) == sizeof(double) ? real_t(1e-6) : real_t(1e-4);
        SFEM_TEST_ASSERT(abs_res < abs_tol || rel_res < abs_tol);

        if (compare_serial) {
            const real_t    sol_tol      = abs_tol;
            const geom_t    geom_tol     = sizeof(geom_t) == sizeof(double) ? geom_t(1e-12) : geom_t(1e-5);
            auto            serial_pts   = serial_mesh->points()->data();
            auto            parallel_pts = parallel_mesh->points()->data();
            const ptrdiff_t n_serial     = serial_fs->n_dofs();
            for (ptrdiff_t i = 0; i < n_owned; ++i) {
                const int j = find_serial_node(
                        serial_pts, n_serial, parallel_pts[0][i], parallel_pts[1][i], parallel_pts[2][i], geom_tol);
                SFEM_TEST_ASSERT(j >= 0);
                SFEM_TEST_APPROXEQ(parallel_x->data()[i], serial_x->data()[j], sol_tol);
            }
        }

        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_parallel_checkerboard_ssgmg);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
