#include "mg_builder.hpp"
#include <stdio.h>
#include <cstddef>
#include "partitioner.h"
#include "sfem_API.hpp"
#include "sfem_Buffer.hpp"
#include "sfem_CooSym.hpp"
#include "sfem_LpSmoother.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_Stationary.hpp"
#include "sfem_pwc_interpolator.hpp"
#include "smoother.h"

std::shared_ptr<sfem::Multigrid<real_t>> builder(
        const real_t coarsening_factor, const mask_t *bdy_dofs, real_t *near_null,
        std::shared_ptr<sfem::CooSymSpMV<idx_t, real_t>> &fine_mat) {
    std::shared_ptr<sfem::Multigrid<real_t>> amg = sfem::h_mg<real_t>();

    ptrdiff_t fine_ndofs = fine_mat->rows();
    idx_t levels = 1;
    idx_t max_level = 20;

    PartitionerWorkspace ws;
    count_t offdiag_nnz = fine_mat->values->size();
    printf("Starting AMG builder. Matrix info:\n\tndofs: %d\n\toffdiag nnz: %d\n",
           (int)fine_ndofs,
           offdiag_nnz);

    ptrdiff_t ndofs = fine_ndofs;
    ws.partition = (idx_t *)malloc((fine_ndofs) * sizeof(idx_t));
    ws.rowsums = (real_t *)calloc((fine_ndofs), sizeof(real_t));
    ws.ptr_i = (idx_t *)malloc(offdiag_nnz * sizeof(idx_t));
    ws.ptr_j = (idx_t *)malloc(offdiag_nnz * sizeof(idx_t));
    ws.weights = (real_t *)malloc(offdiag_nnz * sizeof(real_t));
    ws.sort_indices = (count_t *)malloc(offdiag_nnz * sizeof(count_t));

    real_t *zeros = (real_t *)calloc((fine_ndofs), sizeof(real_t));

    // Weighted connectivity graph in COO format
    idx_t *offdiag_row_indices = (idx_t *)malloc(offdiag_nnz * sizeof(idx_t));
    idx_t *offdiag_col_indices = (idx_t *)malloc(offdiag_nnz * sizeof(idx_t));
    real_t *offdiag_values = (real_t *)malloc(offdiag_nnz * sizeof(real_t));

    auto prev_mat = fine_mat;
    std::shared_ptr<sfem::PiecewiseConstantInterpolator<idx_t, real_t>> p = nullptr;
    std::shared_ptr<sfem::PiecewiseConstantInterpolator<idx_t, real_t>> pt = nullptr;

    while (true) {
        for (idx_t k = 0; k < offdiag_nnz; k++) {
            offdiag_row_indices[k] = prev_mat->offdiag_rowidx->data()[k];
            offdiag_col_indices[k] = prev_mat->offdiag_colidx->data()[k];
            offdiag_values[k] = prev_mat->values->data()[k];
        }

        real_t *diag = (real_t *)malloc(ndofs * sizeof(real_t));
        l2_smoother(ndofs,
                    bdy_dofs,
                    prev_mat->values->size(),
                    prev_mat->diag_values->data(),
                    prev_mat->values->data(),
                    prev_mat->offdiag_rowidx->data(),
                    prev_mat->offdiag_colidx->data(),
                    diag);
        auto ptr_lp = sfem::Buffer<real_t>::own(ndofs, diag, free, sfem::MEMORY_SPACE_HOST);

        sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST)
                ->reciprocal(ptr_lp->size(), 1., ptr_lp->data());

        auto l2_smoother_op = sfem::h_lpsmoother(ptr_lp);

        for (idx_t i = 0; i < ndofs; i++) {
            near_null[i] = 1.0;
        }

        auto stat_iter = sfem::h_stationary<real_t>(prev_mat, l2_smoother_op);

        /*
        if (bdy_dofs) {
#pragma omp parallel for
            for (idx_t k = 0; k < ndofs; k++) {
                if (mask_get(k, bdy_dofs)) {
                    near_null[k] = 0;
                }
            }

            stat_iter->set_max_it(10);
            stat_iter->apply(zeros, near_null);
        }
        */

        ptrdiff_t finer_dim = ndofs;
        int failure = partition(bdy_dofs,
                                coarsening_factor,
                                near_null,
                                offdiag_row_indices,
                                offdiag_col_indices,
                                offdiag_values,
                                &offdiag_nnz,
                                &ndofs,
                                &ws);
        if (failure || ndofs < 500 || levels == max_level) {
            auto cg = sfem::create_cg<real_t>(prev_mat, sfem::EXECUTION_SPACE_HOST);
            cg->verbose = false;
            cg->set_max_it(10000);
            cg->set_op(prev_mat);
            cg->set_rtol(1e-12);
            cg->set_preconditioner_op(l2_smoother_op);
            amg->add_level(prev_mat, cg, p, nullptr);
            // amg->add_level(prev_mat, stat_iter, p, nullptr);

            if (failure) {
                printf("Failed to add new level, levels: %d\n", levels);
            } else {
                printf("Coarsest size achieved with %d ndofs at levels: %d\n", (int)ndofs, levels);
            }
            break;
        }

        ptrdiff_t coarser_dim = ndofs;
        idx_t *partition = (idx_t *)malloc(finer_dim * sizeof(idx_t));
        real_t *weights = (real_t *)malloc(finer_dim * sizeof(real_t));

        for (idx_t k = 0; k < finer_dim; k++) {
            partition[k] = ws.partition[k];
            weights[k] = near_null[k];
        }

        auto ptr_weights =
                sfem::Buffer<real_t>::own(finer_dim, weights, free, sfem::MEMORY_SPACE_HOST);
        auto ptr_partition =
                sfem::Buffer<idx_t>::own(finer_dim, partition, free, sfem::MEMORY_SPACE_HOST);

        auto pt = h_pwc_interp(ptr_weights, ptr_partition, coarser_dim);
        pt->transpose();

        stat_iter->set_max_it(3);
        // stat_iter->set_max_it(3);
        //  stat_iter->set_max_it(30);
        amg->add_level(prev_mat, stat_iter, p, pt);
        p = h_pwc_interp(ptr_weights, ptr_partition, coarser_dim);

        real_t resulting_cf = ((real_t)finer_dim) / ((real_t)coarser_dim);
        levels++;

        auto a_coarse = p->coarsen(prev_mat);
        offdiag_nnz = a_coarse->values->size();
        printf("Added level %d\n\tcf: %.2f nrows: %td offdiag nnz: %d, nnz: %td\n",
               levels,
               resulting_cf,
               ndofs,
               offdiag_nnz,
               ndofs + (offdiag_nnz * 2));
        prev_mat = a_coarse;
        bdy_dofs = nullptr;
    }

    // TODO need to add solver for coarsest level in AMG!!

    free(ws.partition);
    free(ws.rowsums);
    free(ws.ptr_i);
    free(ws.ptr_j);
    free(ws.weights);
    free(ws.sort_indices);

    free(offdiag_row_indices);
    free(offdiag_col_indices);
    free(offdiag_values);

    free(zeros);

    return amg;
}
