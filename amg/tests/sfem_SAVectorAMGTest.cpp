#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

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
#pragma omp parallel for
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

#pragma omp parallel for reduction(+ : nrm2)
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
#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < level.r->rows(); ++i) {
                        coarse_rhs->data()[i] = 0;
                        coarse_x->data()[i]   = 0;
                    }

                    level.r->apply(x, coarse_rhs->data());
                    coarse_solver->apply(coarse_rhs->data(), coarse_x->data());

#pragma omp parallel for
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

#pragma omp parallel for
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

    sfem::SharedBuffer<real_t> bsr_inverse_diagonal(
            const std::shared_ptr<sfem::BSR<sfem::count_t, sfem::idx_t, real_t, real_t>>& a) {
        const ptrdiff_t block_rows = a->row_ptr->size() - 1;
        const int       block_size = a->block_size();
        auto            inv_diag   = sfem::create_host_buffer<real_t>(a->rows());

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < a->rows(); ++i) {
            inv_diag->data()[i] = 1;
        }

#pragma omp parallel for
        for (ptrdiff_t node = 0; node < block_rows; ++node) {
            for (sfem::count_t k = a->row_ptr->data()[node]; k < a->row_ptr->data()[node + 1]; ++k) {
                if (a->col_idx->data()[k] != node) continue;

                const real_t* const block = &a->values->data()[k * block_size * block_size];
                for (int d = 0; d < block_size; ++d) {
                    const real_t diag                       = block[d * block_size + d];
                    inv_diag->data()[node * block_size + d] = std::abs(diag) > 1e-14 ? 1 / diag : 1;
                }
                break;
            }
        }

        return inv_diag;
    }

    sfem::SharedBuffer<real_t> bsr_inverse_diagonal_blocks(
            const std::shared_ptr<sfem::BSR<sfem::count_t, sfem::idx_t, real_t, real_t>>& a) {
        const ptrdiff_t block_rows = a->row_ptr->size() - 1;
        const int       block_size = a->block_size();
        const int       block_area = block_size * block_size;
        auto            inv_diag   = sfem::create_host_buffer<real_t>(block_rows * block_area);

#pragma omp parallel for
        for (ptrdiff_t node = 0; node < block_rows; ++node) {
            real_t* const inv = &inv_diag->data()[node * block_area];
            for (int i = 0; i < block_area; ++i) {
                inv[i] = 0;
            }
            for (int d = 0; d < block_size; ++d) {
                inv[d * block_size + d] = 1;
            }

            for (sfem::count_t k = a->row_ptr->data()[node]; k < a->row_ptr->data()[node + 1]; ++k) {
                if (a->col_idx->data()[k] != node) continue;

                std::vector<real_t> mat(block_area);
                for (int i = 0; i < block_area; ++i) {
                    mat[i] = a->values->data()[k * block_area + i];
                    inv[i] = 0;
                }
                for (int d = 0; d < block_size; ++d) {
                    inv[d * block_size + d] = 1;
                }

                bool invertible = true;
                for (int p = 0; p < block_size; ++p) {
                    int    pivot_row = p;
                    real_t pivot_abs = std::abs(mat[p * block_size + p]);
                    for (int r = p + 1; r < block_size; ++r) {
                        const real_t candidate_abs = std::abs(mat[r * block_size + p]);
                        if (candidate_abs > pivot_abs) {
                            pivot_abs = candidate_abs;
                            pivot_row = r;
                        }
                    }

                    if (pivot_abs <= 1e-14) {
                        invertible = false;
                        break;
                    }

                    if (pivot_row != p) {
                        for (int c = 0; c < block_size; ++c) {
                            std::swap(mat[p * block_size + c], mat[pivot_row * block_size + c]);
                            std::swap(inv[p * block_size + c], inv[pivot_row * block_size + c]);
                        }
                    }

                    const real_t inv_pivot = 1 / mat[p * block_size + p];
                    for (int c = 0; c < block_size; ++c) {
                        mat[p * block_size + c] *= inv_pivot;
                        inv[p * block_size + c] *= inv_pivot;
                    }

                    for (int r = 0; r < block_size; ++r) {
                        if (r == p) continue;
                        const real_t factor = mat[r * block_size + p];
                        if (factor == 0) continue;
                        for (int c = 0; c < block_size; ++c) {
                            mat[r * block_size + c] -= factor * mat[p * block_size + c];
                            inv[r * block_size + c] -= factor * inv[p * block_size + c];
                        }
                    }
                }

                if (!invertible) {
                    for (int i = 0; i < block_area; ++i) {
                        inv[i] = 0;
                    }
                    for (int d = 0; d < block_size; ++d) {
                        const real_t diag       = a->values->data()[k * block_area + d * block_size + d];
                        inv[d * block_size + d] = std::abs(diag) > 1e-14 ? 1 / diag : 1;
                    }
                }

                break;
            }
        }

        return inv_diag;
    }

    std::shared_ptr<sfem::Operator<real_t>> block_diagonal_preconditioner(const sfem::SharedBuffer<real_t>& inv_diag_blocks,
                                                                          const int                         block_size) {
        const ptrdiff_t block_rows = inv_diag_blocks->size() / (block_size * block_size);

        return sfem::make_op<real_t>(
                block_rows * block_size,
                block_rows * block_size,
                [=](const real_t* const x, real_t* const y) {
#pragma omp parallel for
                    for (ptrdiff_t node = 0; node < block_rows; ++node) {
                        const real_t* const inv = &inv_diag_blocks->data()[node * block_size * block_size];
                        for (int d = 0; d < block_size; ++d) {
                            real_t value = 0;
                            for (int e = 0; e < block_size; ++e) {
                                value += inv[d * block_size + e] * x[node * block_size + e];
                            }
                            y[node * block_size + d] = value;
                        }
                    }
                },
                sfem::EXECUTION_SPACE_HOST);
    }

    std::shared_ptr<sfem::Operator<real_t>> sa_multilevel_additive_preconditioner(
            const sfem::SAVectorAMGHierarchy<sfem::count_t, sfem::idx_t, real_t>& hierarchy,
            const ptrdiff_t                                                       level_idx,
            const sfem::SharedBuffer<real_t>&                                     inv_diag_blocks) {
        const auto level      = hierarchy.levels[level_idx];
        const int  block_size = level.a->block_size();

        std::shared_ptr<sfem::Operator<real_t>> coarse_apply;
        if (level_idx + 1 < static_cast<ptrdiff_t>(hierarchy.levels.size())) {
            auto next_inv_diag_blocks = bsr_inverse_diagonal_blocks(hierarchy.levels[level_idx + 1].a);
            coarse_apply              = sa_multilevel_additive_preconditioner(hierarchy, level_idx + 1, next_inv_diag_blocks);
        } else {
            auto coarse_op         = zeroing_op(level.coarse_a);
            auto coarse_solver     = sfem::create_cg<real_t>(coarse_op, sfem::EXECUTION_SPACE_HOST);
            coarse_solver->verbose = false;
            coarse_solver->set_preconditioner_op(
                    block_diagonal_preconditioner(bsr_inverse_diagonal_blocks(level.coarse_a), level.coarse_a->block_size()));
            coarse_solver->set_rtol(1e-10);
            coarse_solver->set_atol(1e-14);
            coarse_solver->set_max_it(200);

            printf("hierarchy_level: %ld, coarse_solver->rows(): %ld\n", level_idx, coarse_solver->rows());

            coarse_apply = sfem::make_op<real_t>(
                    coarse_op->rows(),
                    coarse_op->cols(),
                    [=](const real_t* const x, real_t* const y) {
                        for (ptrdiff_t i = 0; i < coarse_op->rows(); ++i) {
                            y[i] = 0;
                        }
                        coarse_solver->apply(x, y);
                    },
                    sfem::EXECUTION_SPACE_HOST);
        }

        printf("hierarchy_level: %ld, additive_rows: %ld, coarse_rows: %ld\n",
               level_idx,
               level.a->rows(),
               level.coarse_a->rows());

        auto coarse_rhs  = sfem::create_host_buffer<real_t>(level.r->rows());
        auto coarse_x    = sfem::create_host_buffer<real_t>(level.p->cols());
        auto coarse_work = sfem::create_host_buffer<real_t>(level.p->rows());

        return sfem::make_op<real_t>(
                level.p->rows(),
                level.r->cols(),
                [=](const real_t* const x, real_t* const y) {
                    for (ptrdiff_t i = 0; i < level.r->rows(); ++i) {
                        coarse_rhs->data()[i] = 0;
                        coarse_x->data()[i]   = 0;
                    }

                    level.r->apply(x, coarse_rhs->data());
                    coarse_apply->apply(coarse_rhs->data(), coarse_x->data());

                    for (ptrdiff_t i = 0; i < level.p->rows(); ++i) {
                        coarse_work->data()[i] = 0;
                    }
                    level.p->apply(coarse_x->data(), coarse_work->data());

                    const ptrdiff_t block_rows = level.a->row_ptr->size() - 1;

#pragma omp parallel for
                    for (ptrdiff_t node = 0; node < block_rows; ++node) {
                        const real_t* const inv = &inv_diag_blocks->data()[node * block_size * block_size];
                        for (int d = 0; d < block_size; ++d) {
                            real_t smooth = 0;
                            for (int e = 0; e < block_size; ++e) {
                                smooth += inv[d * block_size + e] * x[node * block_size + e];
                            }
                            y[node * block_size + d] = smooth + coarse_work->data()[node * block_size + d];
                        }
                    }
                },
                sfem::EXECUTION_SPACE_HOST);
    }

    std::shared_ptr<sfem::Operator<real_t>> sa_multilevel_vcycle_preconditioner(
            const sfem::SAVectorAMGHierarchy<sfem::count_t, sfem::idx_t, real_t>& hierarchy,
            const ptrdiff_t                                                       level_idx,
            const sfem::SharedBuffer<real_t>&                                     inv_diag_blocks) {
        const auto level      = hierarchy.levels[level_idx];
        const int  block_size = level.a->block_size();

        std::shared_ptr<sfem::Operator<real_t>> coarse_apply;
        if (level_idx + 1 < static_cast<ptrdiff_t>(hierarchy.levels.size())) {
            auto next_inv_diag_blocks = bsr_inverse_diagonal_blocks(hierarchy.levels[level_idx + 1].a);
            coarse_apply              = sa_multilevel_vcycle_preconditioner(hierarchy, level_idx + 1, next_inv_diag_blocks);
        } else {
            auto coarse_op         = zeroing_op(level.coarse_a);
            auto coarse_solver     = sfem::create_cg<real_t>(coarse_op, sfem::EXECUTION_SPACE_HOST);
            coarse_solver->verbose = false;
            coarse_solver->set_preconditioner_op(
                    block_diagonal_preconditioner(bsr_inverse_diagonal_blocks(level.coarse_a), level.coarse_a->block_size()));
            coarse_solver->set_rtol(1e-10);
            coarse_solver->set_atol(1e-14);
            coarse_solver->set_max_it(200);

            printf("hierarchy_level: %ld, coarse_solver->rows(): %ld\n", level_idx, coarse_solver->rows());

            coarse_apply = sfem::make_op<real_t>(
                    coarse_op->rows(),
                    coarse_op->cols(),
                    [=](const real_t* const x, real_t* const y) {
#pragma omp parallel for
                        for (ptrdiff_t i = 0; i < coarse_op->rows(); ++i) {
                            y[i] = 0;
                        }
                        coarse_solver->apply(x, y);
                    },
                    sfem::EXECUTION_SPACE_HOST);
        }

        printf("hierarchy_level: %ld, vcycle_rows: %ld, coarse_rows: %ld\n", level_idx, level.a->rows(), level.coarse_a->rows());

        auto smooth_op   = block_diagonal_preconditioner(inv_diag_blocks, block_size);
        auto residual    = sfem::create_host_buffer<real_t>(level.a->rows());
        auto smooth_work = sfem::create_host_buffer<real_t>(level.a->rows());
        auto coarse_rhs  = sfem::create_host_buffer<real_t>(level.r->rows());
        auto coarse_x    = sfem::create_host_buffer<real_t>(level.p->cols());
        auto coarse_work = sfem::create_host_buffer<real_t>(level.p->rows());

        return sfem::make_op<real_t>(
                level.a->rows(),
                level.a->cols(),
                [=](const real_t* const x, real_t* const y) {
                    static const real_t omega = 0.5;

                    smooth_op->apply(x, y);

#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < level.a->rows(); ++i) {
                        y[i] *= omega;
                        residual->data()[i] = 0;
                    }

                    level.a->apply(y, residual->data());

#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < level.a->rows(); ++i) {
                        residual->data()[i] = x[i] - residual->data()[i];
                    }

#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < level.r->rows(); ++i) {
                        coarse_rhs->data()[i] = 0;
                        coarse_x->data()[i]   = 0;
                    }

                    level.r->apply(residual->data(), coarse_rhs->data());
                    coarse_apply->apply(coarse_rhs->data(), coarse_x->data());

#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < level.p->rows(); ++i) {
                        coarse_work->data()[i] = 0;
                    }

                    level.p->apply(coarse_x->data(), coarse_work->data());

#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < level.a->rows(); ++i) {
                        y[i] += coarse_work->data()[i];
                        residual->data()[i] = 0;
                    }

                    level.a->apply(y, residual->data());

#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < level.a->rows(); ++i) {
                        residual->data()[i] = x[i] - residual->data()[i];
                    }

                    smooth_op->apply(residual->data(), smooth_work->data());

#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < level.a->rows(); ++i) {
                        y[i] += omega * smooth_work->data()[i];
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
                                        smesh::Env::read("SFEM_ELEM_TYPE", smesh::TET4),
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

    const int       max_aggregate_size        = 120;
    const int       coarse_max_aggregate_size = 16;
    const int       max_levels                = 10;
    const ptrdiff_t coarsest_block_rows       = 12;
    auto            hierarchy                 = sfem::h_sa_vector_amg_hierarchy<sfem::count_t, sfem::idx_t, geom_t, real_t>(
            a_bsr,
            const_cast<const geom_t* const*>(points->data()),
            3,
            boundary_nodes,
            max_aggregate_size,
            coarse_max_aggregate_size,
            max_levels,
            coarsest_block_rows);

    SFEM_TEST_ASSERT(!hierarchy.levels.empty());
    const auto& level = hierarchy.levels[0];

    printf("max_aggregate_size: %d, aggregates: %ld, coarse rows: %ld\n",
           max_aggregate_size,
           level.aggregates.n_aggregates,
           level.coarse_a->rows());
    printf("coarse_max_aggregate_size: %d, hierarchy_levels: %ld\n", coarse_max_aggregate_size, hierarchy.levels.size());
    for (ptrdiff_t l = 0; l < static_cast<ptrdiff_t>(hierarchy.levels.size()); ++l) {
        printf("hierarchy_level: %ld, rows: %ld, coarse_rows: %ld, aggregates: %ld\n",
               l,
               hierarchy.levels[l].a->rows(),
               hierarchy.levels[l].coarse_a->rows(),
               hierarchy.levels[l].aggregates.n_aggregates);
    }

    SFEM_TEST_EQ(level.n_rigid_body_modes, 6);
    SFEM_TEST_ASSERT(level.coarse_a != nullptr);
    SFEM_TEST_EQ(level.coarse_a->block_size(), 6);
    SFEM_TEST_EQ(level.coarse_a->rows(), level.aggregates.n_aggregates * 6);
    SFEM_TEST_ASSERT(level.aggregates.n_aggregates <= mesh->n_nodes() / 2);
    SFEM_TEST_ASSERT(level.coarse_a->rows() < a_bsr->rows());

    auto solve_op   = zeroing_op(a_bsr);
    auto sa_precond = sa_multilevel_vcycle_preconditioner(hierarchy, 0, bsr_inverse_diagonal_blocks(a_bsr));

    auto sa_solver     = sfem::create_cg<real_t>(solve_op, es);
    sa_solver->verbose = true;
    sa_solver->set_preconditioner_op(sa_precond);
    sa_solver->set_rtol(1e-8);
    sa_solver->set_atol(1e-10);
    sa_solver->set_max_it(1000);

    const double sa_setup_time = smesh::time_seconds() - sa_setup_start;

    const double diag_setup_start = smesh::time_seconds();

    auto diag_precond = inverse_diagonal_preconditioner(inv_diag);
    // auto diag_precond = block_diagonal_preconditioner(bsr_inverse_diagonal_blocks(a_bsr), 3);

    auto diag_solver     = sfem::create_cg<real_t>(solve_op, es);
    diag_solver->verbose = true;
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
