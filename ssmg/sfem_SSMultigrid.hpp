#ifndef SFEM_SS_MULTIGRID_HPP
#define SFEM_SS_MULTIGRID_HPP

#include <algorithm>
#include <cassert>
#include <csignal>
#include <cstdio>

#include "sfem_API.hpp"
#include "sfem_Buffer.hpp"
#include "sfem_ShiftableJacobi.hpp"
#include "sfem_Stationary.hpp"
#include "sfem_config.h"

#ifdef SFEM_ENABLE_AMG

#include "partitioner.h"
#include "sfem_pwc_interpolator.hpp"
#include "smoother.h"

#endif

#include "sfem_ShiftedPenaltyMultigrid.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_ShiftedPenalty_impl.hpp"
#else
#include "sfem_ShiftedPenalty_impl.hpp"
#endif

namespace sfem {

    // TODO improve contact integration in SSMG
    // auto coarse_contact_conditions = contact_conds->derefine(fs_coarse, true);
    // auto coarse_contact_mask = sfem::create_buffer<mask_t>(fs_coarse->n_dofs(), es);
    // coarse_contact_conditions->mask(coarse_contact_mask->data());

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

        const char *SFEM_FINE_OP_TYPE = "MF";
        const char *SFEM_COARSE_OP_TYPE = fs->block_size() == 1 ? "COO_SYM" : "BCRS_SYM";

        SFEM_READ_ENV(SFEM_FINE_OP_TYPE, );
        SFEM_READ_ENV(SFEM_COARSE_OP_TYPE, );

        auto linear_op = sfem::create_linear_operator(SFEM_FINE_OP_TYPE, f, nullptr, es);
        auto linear_op_coarse =
                sfem::create_linear_operator(SFEM_COARSE_OP_TYPE, f_coarse, nullptr, es);

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
        sj->relaxation_parameter = 1. / fs->block_size();
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

        auto spmg = std::dynamic_pointer_cast<ShiftedPenaltyMultigrid<real_t>>(mg);
        if (spmg) {
            // spmg->set_nlsmooth_steps(30);
            int SFEM_MG_NL_SMOOTH_STEPS = 10;
            SFEM_READ_ENV(SFEM_MG_NL_SMOOTH_STEPS, atoi);
            spmg->set_nlsmooth_steps(SFEM_MG_NL_SMOOTH_STEPS);

            int SFEM_MG_PROJECT_COARSE_CORRECTION = 0;
            SFEM_READ_ENV(SFEM_MG_PROJECT_COARSE_CORRECTION, atoi);
            spmg->set_project_coarse_space_correction(SFEM_MG_PROJECT_COARSE_CORRECTION);
            printf("SFEM_MG_PROJECT_COARSE_CORRECTION=%d\n", SFEM_MG_PROJECT_COARSE_CORRECTION);

            int SFEM_MG_ENABLE_LINESEARCH = 0;
            SFEM_READ_ENV(SFEM_MG_ENABLE_LINESEARCH, atoi);
            spmg->enable_line_search(SFEM_MG_ENABLE_LINESEARCH);
        }

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            // FIXME this should not be here!
            CUDA_BLAS<real_t>::build_blas(mg->blas());

            if (spmg) {
                CUDA_ShiftedPenalty<real_t>::build(spmg->impl());
            }

            mg->execution_space_ = EXECUTION_SPACE_DEVICE;
        } else
#endif
        {
            mg->default_init();
        }

        mg->add_level(linear_op, smoother, nullptr, restriction);

#ifdef SFEM_ENABLE_AMG
        // TODO could probably abstract this part and make it a function somewhere...
        // I just don't know where
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
        // END TODO. function could be fs -> symcoo
        ptrdiff_t fine_ndofs = fine_mat->rows();
        count_t offdiag_nnz = fine_mat->values->size();

        // TODO could experiment with smoothing this vector some, it may help
        auto near_null = sfem::create_buffer<real_t>(fine_ndofs, es);
        for (idx_t i = 0; i < fine_ndofs; i++) {
            near_null->data()[i] = 1.0;
        }

        PartitionerWorkspace *ws = create_partition_ws(fine_ndofs, offdiag_nnz);
        // Weighted connectivity graph in COO format
        auto offdiag_row_indices = h_buffer<idx_t>(offdiag_nnz);
        auto offdiag_col_indices = h_buffer<idx_t>(offdiag_nnz);
        auto offdiag_values = h_buffer<real_t>(offdiag_nnz);

        idx_t amg_levels = 1;

        // AMG tunable paramaters

        float SFEM_MG_COARSENING_FACTOR = 1.7;
        float SFEM_MG_RELAXATION = 1;
        int SFEM_MG_CYCLE_TYPE = 4;
        int SFEM_MG_MAX_LEVELS = 20;
        int SFEM_MG_PROJECT_COARSE_CORRECTION = 0;
        int SFEM_MG_COARSE_SMOOTH_STEPS = 3;

        SFEM_READ_ENV(SFEM_MG_COARSENING_FACTOR, atof);
        SFEM_READ_ENV(SFEM_MG_RELAXATION, atof);
        SFEM_READ_ENV(SFEM_MG_CYCLE_TYPE, atoi);
        SFEM_READ_ENV(SFEM_MG_MAX_LEVELS, atoi);
        SFEM_READ_ENV(SFEM_MG_COARSE_SMOOTH_STEPS, atoi);

        printf("SFEM_MG_COARSENING_FACTOR=%f\n"
               "SFEM_MG_RELAXATION=%f\n"
               "SFEM_MG_MAX_LEVELS=%d\n"
               "SFEM_MG_CYCLE_TYPE=%d\n"
               "SFEM_MG_COARSE_SMOOTH_STEPS=%d\n",
               SFEM_MG_COARSENING_FACTOR,
               SFEM_MG_RELAXATION,
               SFEM_MG_MAX_LEVELS,
               SFEM_MG_CYCLE_TYPE,
               SFEM_MG_COARSE_SMOOTH_STEPS);

        mg->set_cycle_type(SFEM_MG_CYCLE_TYPE);

        count_t coarsest_ndofs = 500;

        auto prev_mat = fine_mat;
        std::shared_ptr<sfem::PiecewiseConstantInterpolator<idx_t, real_t>> p, pt = nullptr;

        auto diag_smoother = sfem::create_buffer<real_t>(fine_ndofs, es);
        l2_smoother(fine_ndofs,
                    bdy_dofs,
                    prev_mat->values->size(),
                    prev_mat->diag_values->data(),
                    prev_mat->values->data(),
                    prev_mat->offdiag_rowidx->data(),
                    prev_mat->offdiag_colidx->data(),
                    diag_smoother->data());
        auto amg_smoother = sfem::create_shiftable_jacobi(diag_smoother, es);
        amg_smoother->relaxation_parameter = SFEM_MG_RELAXATION;

        ptrdiff_t ndofs = fine_ndofs;
        count_t fine_memory = fine_ndofs + offdiag_nnz;
        count_t fine_nnz = fine_ndofs + offdiag_nnz * 2;

        count_t prev_nnz = fine_nnz;
        count_t total_memory = fine_memory;
        count_t total_complexity = fine_nnz;

        printf("\nAMG info:\n level      cf         dofs       offdiag nnz    true nnz   "
               "sparsity "
               "factor\n");
        printf("|%-10d|%-10.2f|%-10td|%-14d|%-10ld|%-16.2f|\n",
               1,
               1.0,
               fine_ndofs,
               offdiag_nnz,
               (long)fine_nnz,
               1.0);

        while (amg_levels < SFEM_MG_MAX_LEVELS && ndofs > coarsest_ndofs) {
            for (idx_t k = 0; k < offdiag_nnz; k++) {
                offdiag_row_indices->data()[k] = prev_mat->offdiag_rowidx->data()[k];
                offdiag_col_indices->data()[k] = prev_mat->offdiag_colidx->data()[k];
                offdiag_values->data()[k] = prev_mat->values->data()[k];
            }

            ptrdiff_t finer_dim = ndofs;
            int failure = partition(bdy_dofs,
                                    SFEM_MG_COARSENING_FACTOR,
                                    near_null->data(),
                                    offdiag_row_indices->data(),
                                    offdiag_col_indices->data(),
                                    offdiag_values->data(),
                                    &offdiag_nnz,
                                    &ndofs,
                                    ws);

            if (failure) {
                printf("Failed to add new level, AMG levels: %d\n", amg_levels);
                // Otherwise no AMG :(
                assert(amg_levels > 1);
                break;
            }

            ptrdiff_t coarser_dim = ndofs;
            auto partition_buff = h_buffer<idx_t>(finer_dim);
            auto weights_buff = h_buffer<real_t>(finer_dim);

            for (idx_t k = 0; k < finer_dim; k++) {
                partition_buff->data()[k] = ws->partition[k];
                weights_buff->data()[k] = near_null->data()[k];
            }

            auto pt = h_pwc_interp(weights_buff, partition_buff, coarser_dim);
            pt->transpose();

            // Convert matrix?
            std::shared_ptr<sfem::Operator<real_t>> coarse_op = prev_mat;
            auto stat_iter = sfem::create_stationary<real_t>(coarse_op, amg_smoother, es);

            if (amg_levels == 1) {
                mg->add_level(linear_op_coarse, stat_iter, prolongation, pt);
            } else {
                stat_iter->set_max_it(SFEM_MG_COARSE_SMOOTH_STEPS);
                mg->add_level(coarse_op, stat_iter, p, pt);
            }
            p = h_pwc_interp(weights_buff, partition_buff, coarser_dim);

            amg_levels++;

            auto a_coarse = p->coarsen(prev_mat);
            offdiag_nnz = a_coarse->values->size();
            count_t coarse_nnz = ndofs + (offdiag_nnz * 2);
            printf("|%-10d|%-10.2f|%-10td|%-14d|%-10ld|%-16.2f|\n",
                   amg_levels,
                   (real_t)finer_dim / (real_t)coarser_dim,
                   ndofs,
                   offdiag_nnz,
                   (long)coarse_nnz,
                   (real_t)coarse_nnz / (real_t)prev_nnz);
            total_memory += ndofs + offdiag_nnz;
            total_complexity += ndofs + (offdiag_nnz * 2);
            prev_nnz = coarse_nnz;
            prev_mat = a_coarse;
            bdy_dofs = nullptr;

            diag_smoother = sfem::create_buffer<real_t>(ndofs, es);
            l2_smoother(ndofs,
                        bdy_dofs,
                        prev_mat->values->size(),
                        prev_mat->diag_values->data(),
                        prev_mat->values->data(),
                        prev_mat->offdiag_rowidx->data(),
                        prev_mat->offdiag_colidx->data(),
                        diag_smoother->data());
            amg_smoother = sfem::create_shiftable_jacobi(diag_smoother, es);
            amg_smoother->relaxation_parameter = SFEM_MG_RELAXATION;
        }

        // Create a coarsest level solver, could also just smooth here if coarsest problem isn't
        // small enough to solve exactly. Direct solver by cholesky is also probably better here
        auto cg = sfem::create_cg<real_t>(prev_mat, es);
        cg->verbose = false;
        cg->set_max_it(40000);  // Keep it large just to be sure!
        cg->set_rtol(1e-12);
        cg->set_atol(1e-20);
        cg->set_preconditioner_op(amg_smoother);
        mg->add_level(prev_mat, cg, p, nullptr);

        if (amg_levels == SFEM_MG_MAX_LEVELS) {
            printf("AMG constructed successfully with max levels hit (%d levels)\n", amg_levels);
        } else if (ndofs <= coarsest_ndofs) {
            printf("AMG constructed successfully with coarsest target hit (%ld dofs on coarsest)\n",
                   (long)ndofs);
        }
        printf("Memory complexity: %.2f\n", (real_t)total_memory / (real_t)fine_memory);
        printf("Operator complexity: %.2f\n\n", (real_t)total_complexity / (real_t)fine_nnz);

        free_partition_ws(ws);

#else
        auto solver_coarse = sfem::create_cg<real_t>(linear_op_coarse, es);
        solver_coarse->verbose = false;
        // solver_coarse->verbose = true;

        int SFEM_MG_ENABLE_COARSE_SPACE_PRECONDITIONER = 0;
        SFEM_READ_ENV(SFEM_MG_ENABLE_COARSE_SPACE_PRECONDITIONER, atoi);

        if (SFEM_MG_ENABLE_COARSE_SPACE_PRECONDITIONER) {
            printf("SFEM_MG_ENABLE_COARSE_SPACE_PRECONDITIONER=1\n");
            auto diag = sfem::create_buffer<real_t>(fs_coarse->n_dofs(), es);
            f_coarse->hessian_diag(nullptr, diag->data());
            auto sj_coarse = sfem::create_shiftable_jacobi(diag, es);
            solver_coarse->set_preconditioner_op(sj_coarse);
        }

        mg->add_level(linear_op_coarse, solver_coarse, prolongation, nullptr);
#endif
        return mg;
    }

}  // namespace sfem

#endif  // SFEM_SS_MULTIGRID_HPP
