#include "mg_builder.hpp"
#include <stdio.h>
#include <cstddef>
#include "partitioner.h"
#include "sfem_Buffer.hpp"
#include "sfem_CooSym.hpp"
#include "sfem_LpSmoother.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_pwc_interpolator.hpp"
#include "smoother.h"
#include "sparse.h"

std::shared_ptr<sfem::Multigrid<real_t>> builder(
        const real_t coarsening_factor,
        const mask_t *bdy_dofs,
        std::shared_ptr<sfem::CooSymSpMV<idx_t, real_t>> &fine_mat) {
    std::shared_ptr<sfem::Multigrid<real_t>> amg = sfem::h_mg<real_t>();

    ptrdiff_t dim = fine_mat->rows();
    real_t *near_null = (real_t *)malloc(dim * sizeof(real_t));

    for (idx_t i = 0; i < dim; i++) {
        near_null[i] = 1.0;
    }

    idx_t current_dim = dim;
    idx_t levels = 1;

    PartitionerWorkspace ws;
    count_t nweights = fine_mat->values->size();
    ws.partition = (idx_t *)malloc((dim) * sizeof(idx_t));
    ws.rowsums = (real_t *)calloc((dim), sizeof(real_t));
    ws.ptr_i = (idx_t *)malloc(nweights * sizeof(idx_t));
    ws.ptr_j = (idx_t *)malloc(nweights * sizeof(idx_t));
    ws.weights = (real_t *)malloc(nweights * sizeof(real_t));
    ws.sort_indices = (idx_t *)malloc(nweights * sizeof(idx_t));

    // Don't need `prev.diag` for partitioner so don't allocate it
    SymmCOOMatrix a_bar;
    a_bar.offdiag_row_indices = (idx_t *)malloc(nweights * sizeof(idx_t));
    a_bar.offdiag_col_indices = (idx_t *)malloc(nweights * sizeof(idx_t));
    a_bar.offdiag_values = (real_t *)malloc(nweights * sizeof(real_t));

    auto prev_mat = fine_mat;

    while (current_dim > 100) {
        a_bar.dim = current_dim;
        a_bar.offdiag_nnz = prev_mat->values->size();
        for (idx_t k = 0; k < a_bar.offdiag_nnz; k++) {
            a_bar.offdiag_row_indices[k] = prev_mat->offdiag_rowidx->data()[k];
            a_bar.offdiag_col_indices[k] = prev_mat->offdiag_colidx->data()[k];
            a_bar.offdiag_values[k] = prev_mat->values->data()[k];
        }

        int failure = partition(near_null, bdy_dofs, coarsening_factor, &a_bar, &ws);
        if (failure) {
            break;
        }

        ptrdiff_t fine_dim = current_dim;
        ptrdiff_t coarse_dim = a_bar.dim;
        idx_t *partition = (idx_t *)malloc(current_dim * sizeof(idx_t));
        real_t *weights = (real_t *)malloc(current_dim * sizeof(real_t));

        for (idx_t k = 0; k < current_dim; k++) {
            partition[k] = ws.partition[k];
            weights[k] = near_null[k];
        }

        auto ptr_weights =
                sfem::Buffer<real_t>::own(fine_dim, weights, free, sfem::MEMORY_SPACE_HOST);
        auto ptr_partition =
                sfem::Buffer<idx_t>::own(fine_dim, partition, free, sfem::MEMORY_SPACE_HOST);
        auto p = h_pwc_interp(ptr_weights, ptr_partition, coarse_dim);

        auto pt = h_pwc_interp(ptr_weights, ptr_partition, coarse_dim);
        pt->transpose();

        real_t *inv_diag = (real_t *)malloc(fine_dim * sizeof(real_t));
        l2_smoother(fine_dim,
                    prev_mat->values->size(),
                    prev_mat->diag_values->data(),
                    prev_mat->offdiag_rowidx->data(),
                    prev_mat->offdiag_colidx->data(),
                    inv_diag);
        auto ptr_lp = sfem::Buffer<real_t>::own(fine_dim, inv_diag, free, sfem::MEMORY_SPACE_HOST);
        auto l2_smoother = sfem::h_lpsmoother(ptr_lp);

        amg->add_level(prev_mat, l2_smoother, p, pt);

        real_t resulting_cf = ((real_t)current_dim) / ((real_t)p->coarse_dim);
        levels++;

        auto a_coarse = p->coarsen(prev_mat);
        printf("Added level %d\n\tcf: %.2f nrows: %td nnz: %td\n",
               levels,
               resulting_cf,
               a_bar.dim,
               a_bar.dim + (a_bar.offdiag_nnz * 2));
        current_dim = p->coarse_dim;
        prev_mat = a_coarse;
    }

    // TODO need to add solver for coarsest level in AMG!!

    free(ws.partition);
    free(ws.rowsums);
    free(ws.ptr_i);
    free(ws.ptr_j);
    free(ws.weights);
    free(ws.sort_indices);

    free(a_bar.offdiag_row_indices);
    free(a_bar.offdiag_col_indices);
    free(a_bar.offdiag_values);

    free(near_null);
    return amg;
}
