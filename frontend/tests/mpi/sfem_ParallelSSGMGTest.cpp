#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_ssgmg.hpp"
#include "sfem_ssmgc.hpp"

#include "smesh_base.hpp"
#include "smesh_distributed_base.hpp"
#include "smesh_grid.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_sideset.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

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

        sfem::DirichletConditions::Condition left{
                .sidesets  = bottom_ss,
                .value     = -1,
                .component = 0};
        sfem::DirichletConditions::Condition right{
                .sidesets  = right_ss,
                .value     = 1,
                .component = 0};
        f->add_constraint(sfem::create_dirichlet_conditions(fs, {left, right}, sfem::EXECUTION_SPACE_HOST));
        return f;
    }

    std::shared_ptr<sfem::Function> make_homogeneous_ss_poisson(const int                          element_level,
                                                                const std::shared_ptr<sfem::Mesh> &mesh) {
        auto ss = smesh::to_semistructured(element_level, mesh, true, false);
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

        sfem::DirichletConditions::Condition bottom{
                .sidesets  = bottom_ss,
                .value     = -1,
                .component = 0};
        sfem::DirichletConditions::Condition right{
                .sidesets  = right_ss,
                .value     = 1,
                .component = 0};
        f->add_constraint(sfem::create_dirichlet_conditions(fs, {bottom, right}, sfem::EXECUTION_SPACE_HOST));
        return f;
    }

    smesh::ElemType block_source_family(const smesh::ElemType type) {
        return smesh::is_semistructured_type(type) ? smesh::ss_source_family(type) : type;
    }

    void mark_hex_tet_nodes(const smesh::Mesh &mesh, std::vector<char> &hex, std::vector<char> &tet) {
        hex.assign((size_t)mesh.n_nodes(), 0);
        tet.assign((size_t)mesh.n_nodes(), 0);
        for (size_t b = 0; b < mesh.n_blocks(); ++b) {
            auto            block = mesh.block(b);
            const ptrdiff_t ne    = block->n_elements();
            if (ne == 0) {
                continue;
            }
            const smesh::ElemType fam = block_source_family(block->element_type());
            std::vector<char>    *mask = nullptr;
            if (fam == smesh::HEX8) {
                mask = &hex;
            } else if (fam == smesh::TET4) {
                mask = &tet;
            } else {
                continue;
            }
            const int nxe = block->n_nodes_per_element();
            auto      els = block->elements()->data();
            for (int d = 0; d < nxe; ++d) {
                for (ptrdiff_t e = 0; e < ne; ++e) {
                    const idx_t n = els[d][e];
                    if (n >= 0 && (size_t)n < mask->size()) {
                        (*mask)[(size_t)n] = 1;
                    }
                }
            }
        }
    }

    // Mixed HEX+TET SS can place two family-local nodes at the same xyz (HEX face
    // interior vs TET face/edge on the interface). Prefer the serial node with the
    // same HEX/TET occupancy; otherwise the first geometric hit.
    int find_serial_node(const geom_t *const *serial_pts,
                         const int            serial_spatial_dim,
                         const ptrdiff_t      n_serial,
                         const char          *serial_hex,
                         const char          *serial_tet,
                         const geom_t         x,
                         const geom_t         y,
                         const geom_t         z,
                         const geom_t         tol,
                         const char           par_hex,
                         const char           par_tet) {
        const bool has_z     = serial_spatial_dim > 2;
        int        first     = -1;
        int        occ_match = -1;
        for (ptrdiff_t j = 0; j < n_serial; ++j) {
            const geom_t serial_z = has_z ? serial_pts[2][j] : geom_t(0);
            if (std::fabs(serial_pts[0][j] - x) > tol || std::fabs(serial_pts[1][j] - y) > tol ||
                std::fabs(serial_z - z) > tol) {
                continue;
            }
            if (first < 0) {
                first = (int)j;
            }
            if (serial_hex && serial_tet && serial_hex[j] == par_hex && serial_tet[j] == par_tet) {
                occ_match = (int)j;
                break;
            }
        }
        return occ_match >= 0 ? occ_match : first;
    }

    int solve_and_check_parallel_ssgmg(const char                          *label,
                                       const std::shared_ptr<sfem::Function> &serial_f,
                                       const std::shared_ptr<sfem::Function> &parallel_f,
                                       const real_t                          abs_tol,
                                       const real_t                          rel_tol,
                                       const real_t                          sol_tol,
                                       const int                             max_it) {
        auto comm = sfem::Communicator::world();
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
            serial_mg->set_max_it(max_it);
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
        parallel_mg->set_max_it(max_it);
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
            printf("parallel %s ssgmg residual abs %g rel %g\n", label, (double)abs_res, (double)rel_res);
        }

        SFEM_TEST_ASSERT(abs_res < abs_tol || rel_res < rel_tol);

        if (compare_serial) {
            const geom_t    geom_tol       = sizeof(geom_t) == sizeof(double) ? geom_t(1e-12) : geom_t(1e-5);
            auto            serial_pts     = serial_mesh->points()->data();
            auto            parallel_pts   = parallel_mesh->points()->data();
            const ptrdiff_t n_serial       = serial_fs->n_dofs();
            const bool      parallel_has_z = parallel_mesh->spatial_dimension() > 2;
            std::vector<char> serial_hex, serial_tet, parallel_hex, parallel_tet;
            mark_hex_tet_nodes(*serial_mesh, serial_hex, serial_tet);
            mark_hex_tet_nodes(*parallel_mesh, parallel_hex, parallel_tet);
            for (ptrdiff_t i = 0; i < n_owned; ++i) {
                const geom_t z  = parallel_has_z ? parallel_pts[2][i] : geom_t(0);
                const char   ph = (size_t)i < parallel_hex.size() ? parallel_hex[(size_t)i] : 0;
                const char   pt = (size_t)i < parallel_tet.size() ? parallel_tet[(size_t)i] : 0;
                const int    j  = find_serial_node(serial_pts,
                                                serial_mesh->spatial_dimension(),
                                                n_serial,
                                                serial_hex.data(),
                                                serial_tet.data(),
                                                parallel_pts[0][i],
                                                parallel_pts[1][i],
                                                z,
                                                geom_tol,
                                                ph,
                                                pt);
                SFEM_TEST_ASSERT(j >= 0);
                SFEM_TEST_APPROXEQ(parallel_x->data()[i], serial_x->data()[j], sol_tol);
            }
        }

        return SFEM_TEST_SUCCESS;
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

        const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-6) : real_t(1e-4);
        return solve_and_check_parallel_ssgmg("checkerboard",
                                              make_homogeneous_checkerboard_ss_poisson(l, serial_hex),
                                              make_homogeneous_checkerboard_ss_poisson(l, parallel_hex),
                                              tol,
                                              tol,
                                              tol,
                                              smesh::Env::read("SFEM_MG_MAX_IT", 40));
    }

    int test_parallel_quad4_ssgmg() {
        SFEM_TRACE_SCOPE("test_parallel_quad4_ssgmg");
        auto comm = sfem::Communicator::world();

        const ptrdiff_t default_n = std::max<ptrdiff_t>(comm->size(), 4);
        const ptrdiff_t n         = smesh::Env::read("SFEM_QUAD_BASE_RESOLUTION", static_cast<int>(default_n));
        const int       l         = smesh::Env::read("SFEM_QUAD_ELEMENT_REFINE_LEVEL", 4);

        auto serial_quad   = sfem::Mesh::create_quad4_square(sfem::Communicator::self(), n, n, 0, 0, 1, 1);
        auto parallel_quad = sfem::Mesh::create_quad4_square(comm, n, n, 0, 0, 1, 1);
        SFEM_TEST_ASSERT(serial_quad != nullptr);
        SFEM_TEST_ASSERT(parallel_quad != nullptr);

        const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-6) : real_t(1e-4);
        return solve_and_check_parallel_ssgmg("quad4",
                                              make_homogeneous_ss_poisson(l, serial_quad),
                                              make_homogeneous_ss_poisson(l, parallel_quad),
                                              tol,
                                              tol,
                                              tol,
                                              smesh::Env::read("SFEM_MG_MAX_IT", 40));
    }

    int test_parallel_tet4_ssgmg() {
        SFEM_TRACE_SCOPE("test_parallel_tet4_ssgmg");
        auto comm = sfem::Communicator::world();

        const ptrdiff_t default_n = 3;
        const ptrdiff_t n         = smesh::Env::read("SFEM_TET_BASE_RESOLUTION", static_cast<int>(default_n));
        const int       l         = smesh::Env::read("SFEM_TET_ELEMENT_REFINE_LEVEL", 4);

        auto serial_tet   = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), n, n, n);
        auto parallel_tet = sfem::Mesh::create_tet4_cube(comm, n, n, n);
        SFEM_TEST_ASSERT(serial_tet != nullptr);
        SFEM_TEST_ASSERT(parallel_tet != nullptr);

        const real_t abs_tol = sizeof(real_t) == sizeof(double) ? real_t(1e-6) : real_t(1e-4);
        const real_t rel_tol = sizeof(real_t) == sizeof(double) ? real_t(1e-6) : real_t(1e-4);
        const real_t sol_tol = sizeof(real_t) == sizeof(double) ? real_t(5e-6) : real_t(1e-4);
        return solve_and_check_parallel_ssgmg("tet4",
                                              make_homogeneous_ss_poisson(l, serial_tet),
                                              make_homogeneous_ss_poisson(l, parallel_tet),
                                              abs_tol,
                                              rel_tol,
                                              sol_tol,
                                              smesh::Env::read("SFEM_MG_MAX_IT", 40));
    }

    int test_parallel_hex8_tet4_ssgmg() {
        SFEM_TRACE_SCOPE("test_parallel_hex8_tet4_ssgmg");
        auto comm = sfem::Communicator::world();

        const ptrdiff_t n = smesh::Env::read("SFEM_MIXED_BASE_RESOLUTION", 2);
        const int       l = smesh::Env::read("SFEM_MIXED_ELEMENT_REFINE_LEVEL", 4);

        auto serial   = sfem::Mesh::create_hex8_tet4_cube(sfem::Communicator::self(), n, n, n);
        auto parallel = sfem::Mesh::create_hex8_tet4_cube(comm, n, n, n);
        SFEM_TEST_ASSERT(serial != nullptr);
        SFEM_TEST_ASSERT(parallel != nullptr);
        SFEM_TEST_EQ(parallel->n_blocks(), static_cast<size_t>(2));
        SFEM_TEST_ASSERT(parallel->block(0)->name() == "hex");
        SFEM_TEST_ASSERT(parallel->block(1)->name() == "tet");

        const real_t abs_tol = sizeof(real_t) == sizeof(double) ? real_t(1e-6) : real_t(1e-4);
        const real_t rel_tol = sizeof(real_t) == sizeof(double) ? real_t(1e-6) : real_t(1e-4);
        const real_t sol_tol = sizeof(real_t) == sizeof(double) ? real_t(5e-6) : real_t(1e-4);
        return solve_and_check_parallel_ssgmg("hex8_tet4",
                                              make_homogeneous_ss_poisson(l, serial),
                                              make_homogeneous_ss_poisson(l, parallel),
                                              abs_tol,
                                              rel_tol,
                                              sol_tol,
                                              smesh::Env::read("SFEM_MG_MAX_IT", 40));
    }

    int test_parallel_checkerboard_ssmgc() {
        SFEM_TRACE_SCOPE("test_parallel_checkerboard_ssmgc");
        auto comm = sfem::Communicator::world();

        auto hex = sfem::Mesh::create_hex8_checkerboard_cube(comm, 2, 2, 2);
        auto ss  = smesh::to_semistructured(2, hex, true, false);
        SFEM_TEST_ASSERT(ss != nullptr);

        auto fs = sfem::FunctionSpace::create(ss, 3);
        auto f  = sfem::Function::create(fs);
        auto op = sfem::create_op(fs, "LinearElasticity", sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(op != nullptr);
        SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);
        f->add_operator(op);

        auto wall = sfem::Sideset::create_from_selector(
                ss, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool {
                    return fabs(x) < 1e-8 || fabs(x - 1) < 1e-8;
                });
        SFEM_TEST_ASSERT(!wall.empty());
        sfem::DirichletConditions::Condition xc{.sidesets = wall, .value = 0, .component = 0};
        sfem::DirichletConditions::Condition yc{.sidesets = wall, .value = real_t(-0.05), .component = 1};
        sfem::DirichletConditions::Condition zc{.sidesets = wall, .value = 0, .component = 2};
        f->add_constraint(sfem::create_dirichlet_conditions(fs, {xc, yc, zc}, sfem::EXECUTION_SPACE_HOST));

        auto contact_ss = sfem::Sideset::create_from_selector(
                ss, [](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool { return y > -1e-5 && y < 1e-5; });
        SFEM_TEST_ASSERT(contact_ss.size() >= 1);

        auto sdf = smesh::create_sdf(comm,
                                     16,
                                     8,
                                     16,
                                     -0.1,
                                     -0.2,
                                     -0.1,
                                     1.1,
                                     0.2,
                                     1.1,
                                     [](const geom_t x, const geom_t y, const geom_t z) -> geom_t {
                                         const geom_t cx = 0.5, cy = -0.5, cz = 0.5, radius = 0.5;
                                         const geom_t dx = cx - x, dy = cy - y, dz = cz - z;
                                         return radius - sqrt(dx * dx + dy * dy + dz * dz);
                                     });

        auto contact_conds = sfem::ContactConditions::create(fs, sdf, contact_ss, sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(contact_conds != nullptr);
        {
            ptrdiff_t nloc  = contact_conds->n_constrained_dofs();
            ptrdiff_t nglob = 0;
            MPI_Allreduce(&nloc, &nglob, 1, smesh::mpi_type<ptrdiff_t>(), MPI_SUM, comm->get());
            SFEM_TEST_ASSERT(nglob > 0);
        }

        const ptrdiff_t ndofs = fs->n_dofs();
        auto            x     = sfem::create_host_buffer<real_t>(ndofs);
        auto            rhs   = sfem::create_host_buffer<real_t>(ndofs);
        std::fill(x->data(), x->data() + ndofs, real_t(0));
        std::fill(rhs->data(), rhs->data() + ndofs, real_t(0));
        f->apply_constraints(rhs->data());
        contact_conds->init();
        f->apply_constraints(x->data());

        setenv("SFEM_MAX_IT", "1", 1);

        auto solver = sfem::create_ssmgc(f, contact_conds, nullptr);
        SFEM_TEST_ASSERT(solver != nullptr);
        SFEM_TEST_ASSERT(solver->apply(rhs->data(), x->data()) == SFEM_SUCCESS);
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            SFEM_TEST_ASSERT(std::isfinite(x->data()[i]));
        }
        solver.reset();
        comm->barrier();
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_parallel_checkerboard_ssgmg);
    SFEM_RUN_TEST(test_parallel_quad4_ssgmg);
    SFEM_RUN_TEST(test_parallel_tet4_ssgmg);
    SFEM_RUN_TEST(test_parallel_hex8_tet4_ssgmg);
    SFEM_RUN_TEST(test_parallel_checkerboard_ssmgc);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}


