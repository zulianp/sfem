#include <cmath>
#include <memory>

#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_cg.hpp"
#include "vector_amg/sfem_sa_vector_amg.hpp"

namespace {

    std::shared_ptr<sfem::Operator<real_t>> zeroing_op(const std::shared_ptr<sfem::Operator<real_t>>& op) {
        return sfem::make_op<real_t>(
                op->rows(),
                op->cols(),
                [=](const real_t* const x, real_t* const y) {
                    for (ptrdiff_t i = 0; i < op->rows(); ++i) {
                        y[i] = 0;
                    }
                    op->apply(x, y);
                },
                op->execution_space());
    }

    real_t residual_norm(const std::shared_ptr<sfem::Operator<real_t>>& op,
                         const sfem::SharedBuffer<real_t>&              rhs,
                         const sfem::SharedBuffer<real_t>&              x,
                         const sfem::SharedBuffer<real_t>&              work) {
        real_t* const       r = work->data();
        const real_t* const b = rhs->data();

        op->apply(x->data(), r);

        real_t nrm2 = 0;
        for (ptrdiff_t i = 0; i < op->rows(); ++i) {
            const real_t ri = b[i] - r[i];
            nrm2 += ri * ri;
        }

        return std::sqrt(nrm2);
    }

    std::shared_ptr<sfem::Operator<real_t>> sa_coarse_correction(
            const sfem::SAVectorAMGLevel<sfem::count_t, sfem::idx_t, real_t, real_t>& level) {
        auto coarse_op         = zeroing_op(level.coarse_a);
        auto coarse_solver     = sfem::create_cg<real_t>(coarse_op, sfem::EXECUTION_SPACE_HOST);
        coarse_solver->verbose = false;
        coarse_solver->set_rtol(1e-10);
        coarse_solver->set_atol(1e-14);
        coarse_solver->set_max_it(80);

        printf("coarse_solver->rows(): %ld\n", coarse_solver->rows());

        auto coarse_rhs = sfem::create_host_buffer<real_t>(level.r->rows());
        auto coarse_x   = sfem::create_host_buffer<real_t>(level.p->cols());

        return sfem::make_op<real_t>(
                level.p->rows(),
                level.r->cols(),
                [=](const real_t* const x, real_t* const y) {
                    for (ptrdiff_t i = 0; i < level.r->rows(); ++i) {
                        coarse_rhs->data()[i] = 0;
                        coarse_x->data()[i]   = 0;
                    }

                    level.r->apply(x, coarse_rhs->data());
                    coarse_solver->apply(coarse_rhs->data(), coarse_x->data());

                    for (ptrdiff_t i = 0; i < level.p->rows(); ++i) {
                        y[i] = 0;
                    }
                    level.p->apply(coarse_x->data(), y);
                },
                sfem::EXECUTION_SPACE_HOST);
    }

    std::shared_ptr<sfem::Operator<real_t>> additive_sa_preconditioner(
            const sfem::SAVectorAMGLevel<sfem::count_t, sfem::idx_t, real_t, real_t>& level,
            const sfem::SharedBuffer<real_t>&                                         inv_diag) {
        auto coarse      = sa_coarse_correction(level);
        auto coarse_work = sfem::create_host_buffer<real_t>(level.p->rows());

        return sfem::make_op<real_t>(
                level.p->rows(),
                level.r->cols(),
                [=](const real_t* const x, real_t* const y) {
                    coarse->apply(x, coarse_work->data());

                    for (ptrdiff_t i = 0; i < level.p->rows(); ++i) {
                        y[i] = inv_diag->data()[i] * x[i] + coarse_work->data()[i];
                    }
                },
                sfem::EXECUTION_SPACE_HOST);
    }

    std::shared_ptr<sfem::Operator<real_t>> inverse_diagonal_preconditioner(const sfem::SharedBuffer<real_t>& inv_diag) {
        return sfem::make_op<real_t>(
                inv_diag->size(),
                inv_diag->size(),
                [=](const real_t* const x, real_t* const y) {
                    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(inv_diag->size()); ++i) {
                        y[i] = inv_diag->data()[i] * x[i];
                    }
                },
                sfem::EXECUTION_SPACE_HOST);
    }

}  // namespace

int test_linear_elasticity_sa_vector_amg() {
    MPI_Comm comm = MPI_COMM_WORLD;

    sfem::ExecutionSpace es = sfem::EXECUTION_SPACE_HOST;

    ptrdiff_t SFEM_MESH_RESOLUTION = 10;
    SFEM_READ_ENV(SFEM_MESH_RESOLUTION, atoi);

    auto mesh = sfem::Mesh::create_cube(sfem::Communicator::wrap(comm),
                                        smesh::TET4,
                                        SFEM_MESH_RESOLUTION,
                                        SFEM_MESH_RESOLUTION,
                                        SFEM_MESH_RESOLUTION,
                                        0,
                                        0,
                                        0,
                                        1,
                                        1,
                                        1);

    auto fs = sfem::FunctionSpace::create(mesh, 3);

    auto left = sfem::Sideset::create_from_selector(
            mesh, [](const geom_t x, const geom_t, const geom_t) -> bool { return x < static_cast<geom_t>(1e-8); });

    auto conds = sfem::create_dirichlet_conditions(fs,
                                                   {{.sidesets = left, .value = 0, .component = 0},
                                                    {.sidesets = left, .value = 0, .component = 1},
                                                    {.sidesets = left, .value = 0, .component = 2}},
                                                   es);

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, "LinearElasticity", es);
    op->initialize();
    f->add_constraint(conds);
    f->add_operator(op);

    const ptrdiff_t ndofs  = fs->n_dofs();
    auto            x      = sfem::create_host_buffer<real_t>(ndofs);
    auto            x_diag = sfem::create_host_buffer<real_t>(ndofs);
    auto            x_sa   = sfem::create_host_buffer<real_t>(ndofs);
    auto            rhs    = sfem::create_host_buffer<real_t>(ndofs);
    auto            work   = sfem::create_host_buffer<real_t>(ndofs);

    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        x->data()[i]      = 0;
        x_diag->data()[i] = 0;
        x_sa->data()[i]   = 0;
        rhs->data()[i]    = 0;
    }

    auto points = fs->points();
    for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
        const real_t xcoord       = static_cast<real_t>(points->data()[0][node]);
        rhs->data()[node * 3 + 0] = xcoord;
        rhs->data()[node * 3 + 1] = 0.25 * xcoord;
        rhs->data()[node * 3 + 2] = -0.125 * xcoord;
    }

    f->apply_constraints(x->data());
    f->apply_constraints(x_diag->data());
    f->apply_constraints(x_sa->data());
    f->apply_constraints(rhs->data());

    const double common_setup_start = smesh::time_seconds();

    auto graph  = fs->node_to_node_graph();
    auto values = sfem::create_host_buffer<real_t>(graph->nnz() * 9);
    SFEM_TEST_ASSERT(f->hessian_bsr(x->data(), graph->rowptr()->data(), graph->colidx()->data(), values->data()) == SFEM_SUCCESS);

    auto a_bsr = sfem::h_bsr_spmv<sfem::count_t, sfem::idx_t, real_t, real_t>(
            graph->n_nodes(), graph->n_nodes(), 3, graph->rowptr(), graph->colidx(), values, static_cast<real_t>(1));

    printf("a_bsr->rows(): %ld\n", a_bsr->rows());

    auto inv_diag = sfem::create_host_buffer<real_t>(ndofs);
    for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
        for (int d = 0; d < 3; ++d) {
            inv_diag->data()[node * 3 + d] = 1;
        }

        for (sfem::count_t k = graph->rowptr()->data()[node]; k < graph->rowptr()->data()[node + 1]; ++k) {
            if (graph->colidx()->data()[k] != node) continue;

            const real_t* const block = &values->data()[k * 9];
            for (int d = 0; d < 3; ++d) {
                const real_t diag              = block[d * 3 + d];
                inv_diag->data()[node * 3 + d] = std::abs(diag) > 1e-14 ? 1 / diag : 1;
            }
            break;
        }
    }

    const double common_setup_time = smesh::time_seconds() - common_setup_start;

    const double sa_setup_start = smesh::time_seconds();

    auto boundary_nodes = sfem::create_host_buffer<sfem::mask_t>(mesh->n_nodes());
    for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
        boundary_nodes->data()[node] = points->data()[0][node] < static_cast<geom_t>(1e-8);
    }

    const int max_aggregate_size = 120;
    auto      level              = sfem::h_sa_vector_amg_level<sfem::count_t, sfem::idx_t, real_t, geom_t, real_t>(
            a_bsr, const_cast<const geom_t* const*>(points->data()), 3, boundary_nodes, max_aggregate_size);

    printf("max_aggregate_size: %d, aggregates: %ld, coarse rows: %ld\n",
           max_aggregate_size,
           level.aggregates.n_aggregates,
           level.coarse_a->rows());

    SFEM_TEST_EQ(level.n_rigid_body_modes, 6);
    SFEM_TEST_ASSERT(level.coarse_a != nullptr);
    SFEM_TEST_EQ(level.coarse_a->block_size(), 6);
    SFEM_TEST_EQ(level.coarse_a->rows(), level.aggregates.n_aggregates * 6);
    SFEM_TEST_ASSERT(level.aggregates.n_aggregates <= mesh->n_nodes() / 2);
    SFEM_TEST_ASSERT(level.coarse_a->rows() < a_bsr->rows());

    auto solve_op   = zeroing_op(a_bsr);
    auto sa_precond = additive_sa_preconditioner(level, inv_diag);

    auto sa_solver     = sfem::create_cg<real_t>(solve_op, es);
    sa_solver->verbose = true;
    sa_solver->set_preconditioner_op(sa_precond);
    sa_solver->set_rtol(1e-8);
    sa_solver->set_atol(1e-10);
    sa_solver->set_max_it(1000);

    const double sa_setup_time = smesh::time_seconds() - sa_setup_start;

    const double diag_setup_start = smesh::time_seconds();

    auto diag_precond = inverse_diagonal_preconditioner(inv_diag);

    auto diag_solver     = sfem::create_cg<real_t>(solve_op, es);
    diag_solver->verbose = false;
    diag_solver->set_preconditioner_op(diag_precond);
    diag_solver->set_rtol(1e-8);
    diag_solver->set_atol(1e-10);
    diag_solver->set_max_it(40000);

    const double diag_setup_time = smesh::time_seconds() - diag_setup_start;

    const real_t initial_residual = residual_norm(solve_op, rhs, x, work);
    SFEM_TEST_ASSERT(initial_residual > 0);

    const double diag_solve_start = smesh::time_seconds();
    SFEM_TEST_ASSERT(diag_solver->apply(rhs->data(), x_diag->data()) == SFEM_SUCCESS);
    const double diag_solve_time = smesh::time_seconds() - diag_solve_start;

    const double sa_solve_start = smesh::time_seconds();
    SFEM_TEST_ASSERT(sa_solver->apply(rhs->data(), x_sa->data()) == SFEM_SUCCESS);
    const double sa_solve_time = smesh::time_seconds() - sa_solve_start;

    printf("diag_preconditioner_iterations: %d, sa_preconditioner_iterations: %d\n",
           diag_solver->iterations(),
           sa_solver->iterations());
    printf("common_setup_time_seconds: %g\n", common_setup_time);
    printf("diag_setup_time_seconds: %g, diag_solve_time_seconds: %g, diag_total_time_seconds: %g\n",
           diag_setup_time,
           diag_solve_time,
           common_setup_time + diag_setup_time + diag_solve_time);
    printf("sa_setup_time_seconds: %g, sa_solve_time_seconds: %g, sa_total_time_seconds: %g\n",
           sa_setup_time,
           sa_solve_time,
           common_setup_time + sa_setup_time + sa_solve_time);
    SFEM_TEST_ASSERT(sa_solver->iterations() <= diag_solver->iterations());

    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        work->data()[i] = 0;
        x->data()[i]    = x_sa->data()[i];
    }

    const real_t final_residual = residual_norm(solve_op, rhs, x_sa, work);
    SFEM_TEST_ASSERT(final_residual < initial_residual * static_cast<real_t>(1e-4));

    smesh::create_directory("amg");
    mesh->write(smesh::Path("amg/mesh"));
    auto out = f->output();
    out->enable_AoS_to_SoA(true);
    out->set_output_dir(smesh::Path("amg/out"));
    SFEM_TEST_ASSERT(out->write("x", x->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(out->write("rhs", rhs->data()) == SFEM_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char* argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_linear_elasticity_sa_vector_amg);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
