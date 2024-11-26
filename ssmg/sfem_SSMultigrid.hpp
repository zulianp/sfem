#ifndef SFEM_SS_MULTIGRID_HPP
#define SFEM_SS_MULTIGRID_HPP

#include <cassert>
#include <csignal>

#include "sfem_API.hpp"
#include "sfem_ShiftableJacobi.hpp"
#include "sfem_Stationary.hpp"

#ifdef SFEM_ENABLE_AMG

#include "partitioner.h"
#include "sfem_pwc_interpolator.hpp"
#include "smoother.h"

#endif

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_ShiftedPenalty_impl.hpp"
#else
#include "sfem_ShiftedPenalty_impl.hpp"
#endif

namespace sfem {

    template <class MG>
    std::shared_ptr<MG> create_ssmg(const std::shared_ptr<Function> &f,
                                    const enum ExecutionSpace es) {
        if (!f->space()->has_semi_structured_mesh()) {
            fprintf(stderr, "[Error] create_ssmg cannot build MG without a semistructured mesh");
            MPI_Abort(MPI_COMM_WORLD, -1);
            return nullptr;
        }

        auto fs = f->space();
        auto fs_coarse = fs->derefine();
        auto f_coarse = f->derefine(fs_coarse, true);
        // auto linear_op = sfem::create_linear_operator("BSR", f, nullptr, es);
        // auto linear_op_coarse = sfem::create_linear_operator("BCRS_SYM", f_coarse, nullptr, es);

        auto linear_op = sfem::create_linear_operator("MF", f, nullptr, es);
        auto linear_op_coarse = sfem::create_linear_operator(
                fs->block_size() == 1 ? "COO_SYM" : "BCRS_SYM", f_coarse, nullptr, es);
        // auto linear_op_coarse = sfem::create_linear_operator("MF", f_coarse, nullptr, es);


        // auto smoother = sfem::create_cheb3<real_t>(linear_op, es);
        // smoother->eigen_solver_tol = 1e-2;
        // smoother->init_with_ones();
        // smoother->scale_eig_max = 1.02;
        // smoother->set_max_it(5);
        // smoother->set_initial_guess_zero(false);

        // auto smoother = sfem::create_cg<real_t>(linear_op, es);
        // smoother->set_max_it(10);
        // smoother->verbose = false;

        auto d = sfem::create_buffer<real_t>(fs->n_dofs(), es);
        f->hessian_diag(nullptr, d->data());
        f->set_value_to_constrained_dofs(1, d->data());

        auto sj = sfem::create_shiftable_jacobi(d, es);
        sj->relaxation_parameter = 1./fs->block_size();
        auto smoother = sfem::create_stationary<real_t>(linear_op, sj, es);
        smoother->set_max_it(3);

        auto restriction = sfem::create_hierarchical_restriction(fs, fs_coarse, es);
        auto prolong_unconstr = sfem::create_hierarchical_prolongation(fs_coarse, fs, es);
        auto prolongation = sfem::make_op<real_t>(
                prolong_unconstr->rows(),
                prolong_unconstr->cols(),
                [=](const real_t *const from, real_t *const to) {
                    prolong_unconstr->apply(from, to);
                    f->apply_zero_constraints(to);
                },
                es);

        auto mg = std::make_shared<MG>();
        mg->set_nlsmooth_steps(20);

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            // FIXME this should not be here!
            CUDA_BLAS<real_t>::build_blas(mg->blas());
            CUDA_ShiftedPenalty<real_t>::build(mg->impl());
            mg->execution_space_ = EXECUTION_SPACE_DEVICE;
        } else
#endif
        {
            mg->default_init();
        }

        mg->add_level(linear_op, smoother, nullptr, restriction);

#ifdef SFEM_ENABLE_AMG

        auto crs_graph = f_coarse->space()->mesh_ptr()->node_to_node_graph_upper_triangular();
        auto diag_values = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
        auto off_diag_values = sfem::create_buffer<real_t>(crs_graph->nnz(), es);
        auto off_diag_rows = sfem::create_buffer<idx_t>(crs_graph->nnz(), es);

        auto x = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
        f_coarse->hessian_crs_sym(x->data(),
                                  crs_graph->rowptr()->data(),
                                  crs_graph->colidx()->data(),
                                  diag_values->data(),
                                  off_diag_values->data());

        count_t *row_ptr = crs_graph->rowptr()->data();
        idx_t *col_indices = crs_graph->colidx()->data();
        for (idx_t i = 0; i < fs_coarse->n_dofs(); i++) {
            for (idx_t idx = row_ptr[i]; idx < row_ptr[i + 1]; idx++) {
                off_diag_rows->data()[idx] = i;
                assert(col_indices[idx] > i);
            }
        }

        auto bdy_dofs_buff = sfem::create_buffer<mask_t>(mask_count(fs_coarse->n_dofs()), es);
        auto bdy_dofs = bdy_dofs_buff->data();
        f_coarse->constaints_mask(bdy_dofs_buff->data());
        auto fine_mat = sfem::h_coosym<idx_t, real_t>(
                bdy_dofs_buff, off_diag_rows, crs_graph->colidx(), off_diag_values, diag_values);
        ptrdiff_t fine_ndofs = fine_mat->rows();
        count_t offdiag_nnz = fine_mat->values->size();
        auto near_null = sfem::create_buffer<real_t>(fine_ndofs, es);
        for (idx_t i = 0; i < fine_ndofs; i++) {
            near_null->data()[i] = 1.0;
        }

        PartitionerWorkspace *ws = create_partition_ws(fine_ndofs, offdiag_nnz);
        // Weighted connectivity graph in COO format
        idx_t *offdiag_row_indices = (idx_t *)malloc(offdiag_nnz * sizeof(idx_t));
        idx_t *offdiag_col_indices = (idx_t *)malloc(offdiag_nnz * sizeof(idx_t));
        real_t *offdiag_values = (real_t *)malloc(offdiag_nnz * sizeof(real_t));

        idx_t amg_levels = 1;

        // AMG tunable paramaters
        real_t coarsening_factor = 2.0;
        count_t smoothing_steps = 3;
        count_t coarsest_ndofs = 50;
        count_t max_levels = 20;

        auto prev_mat = fine_mat;
        std::shared_ptr<sfem::PiecewiseConstantInterpolator<idx_t, real_t>> p, pt = nullptr;

        ptrdiff_t ndofs = fine_ndofs;
        while (true) {
            for (idx_t k = 0; k < offdiag_nnz; k++) {
                offdiag_row_indices[k] = prev_mat->offdiag_rowidx->data()[k];
                offdiag_col_indices[k] = prev_mat->offdiag_colidx->data()[k];
                offdiag_values[k] = prev_mat->values->data()[k];
            }

            auto ptr_lp = sfem::create_buffer<real_t>(ndofs, es);
            
            l2_smoother(ndofs,
                        bdy_dofs,
                        prev_mat->values->size(),
                        prev_mat->diag_values->data(),
                        prev_mat->values->data(),
                        prev_mat->offdiag_rowidx->data(),
                        prev_mat->offdiag_colidx->data(),
                        ptr_lp->data());

            auto l2_smoother_op = sfem::create_shiftable_jacobi(ptr_lp, es);

            l2_smoother_op->relaxation_parameter = 1.0;
            auto stat_iter = sfem::create_stationary<real_t>(prev_mat, l2_smoother_op, es);

            ptrdiff_t finer_dim = ndofs;
            int failure = partition(bdy_dofs,
                                    coarsening_factor,
                                    near_null->data(),
                                    offdiag_row_indices,
                                    offdiag_col_indices,
                                    offdiag_values,
                                    &offdiag_nnz,
                                    &ndofs,
                                    ws);

            if (failure || ndofs < coarsest_ndofs || amg_levels == max_levels) {
                auto cg = sfem::create_cg<real_t>(prev_mat, es);
                cg->verbose = false;
                cg->set_max_it(100);
                cg->set_op(prev_mat);
                cg->set_rtol(1e-6);
                cg->set_preconditioner_op(l2_smoother_op);
                mg->add_level(prev_mat, cg, p, nullptr);

                if (failure) {
                    printf("Failed to add new level, AMG levels: %d\n", amg_levels);
                } else {
                    printf("Coarsest size achieved with %d ndofs at AMG level: %d\n",
                           (int)ndofs,
                           amg_levels);
                }
                // std::raise(SIGINT);
                break;
            }

            ptrdiff_t coarser_dim = ndofs;
            idx_t *partition = (idx_t *)malloc(finer_dim * sizeof(idx_t));
            real_t *weights = (real_t *)malloc(finer_dim * sizeof(real_t));

            for (idx_t k = 0; k < finer_dim; k++) {
                partition[k] = ws->partition[k];
                weights[k] = near_null->data()[k];
            }

            auto ptr_weights =
                    sfem::Buffer<real_t>::own(finer_dim, weights, free, sfem::MEMORY_SPACE_HOST);
            auto ptr_partition =
                    sfem::Buffer<idx_t>::own(finer_dim, partition, free, sfem::MEMORY_SPACE_HOST);

            auto pt = h_pwc_interp(ptr_weights, ptr_partition, coarser_dim);
            pt->transpose();

            if (amg_levels == 1) {
                mg->add_level(linear_op_coarse, stat_iter, prolongation, pt);
            } else {
                stat_iter->set_max_it(smoothing_steps);
                mg->add_level(prev_mat, stat_iter, p, pt);
            }
            p = h_pwc_interp(ptr_weights, ptr_partition, coarser_dim);

            real_t resulting_cf = ((real_t)finer_dim) / ((real_t)coarser_dim);
            amg_levels++;

            auto a_coarse = p->coarsen(prev_mat);
            offdiag_nnz = a_coarse->values->size();
            printf("Added level %d\n\tcf: %.2f nrows: %td offdiag nnz: %d, nnz: %td\n",
                   amg_levels,
                   resulting_cf,
                   ndofs,
                   offdiag_nnz,
                   ndofs + (offdiag_nnz * 2));
            prev_mat = a_coarse;
            bdy_dofs = nullptr;
        }

        free_partition_ws(ws);
        free(offdiag_row_indices);
        free(offdiag_col_indices);
        free(offdiag_values);

#else
        auto solver_coarse = sfem::create_cg<real_t>(linear_op_coarse, es);
        solver_coarse->verbose = false;
         mg->add_level(linear_op_coarse, solver_coarse, prolongation, nullptr);
#endif
        return mg;
    }

}  // namespace sfem

#endif  // SFEM_SS_MULTIGRID_HPP
