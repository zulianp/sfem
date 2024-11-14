#include "mg_builder.h"
#include <stdio.h>
#include "interpolation.h"
#include "partitioner.h"
#include "sparse.h"

int builder(const real_t coarsening_factor,
            const idx_t *free_dofs,
            SymmCOOMatrix *fine_mat,
            PWCHierarchy *hierarchy) {
    idx_t max_levels = 1;
    real_t size = 1;
    while (size < (real_t)fine_mat->dim) {
        size *= coarsening_factor;
        max_levels++;
    }

    ptrdiff_t dim = fine_mat->dim;
    real_t *near_null = (real_t *)malloc(dim * sizeof(real_t));

    for (idx_t i = 0; i < dim; i++) {
        near_null[i] = 1.0;
    }

    PiecewiseConstantTransfer **transer_operators =
            (PiecewiseConstantTransfer **)malloc(max_levels * sizeof(PiecewiseConstantTransfer *));
    SymmCOOMatrix **matrices = (SymmCOOMatrix **)malloc(max_levels * sizeof(SymmCOOMatrix *));

    matrices[0] = fine_mat;

    idx_t current_dim = fine_mat->dim;
    idx_t levels = 1;

    PartitionerWorkspace ws;
    count_t nweights = fine_mat->offdiag_nnz;
    ws.partition = (idx_t *)malloc((fine_mat->dim) * sizeof(idx_t));
    ws.rowsums = (real_t *)calloc((fine_mat->dim), sizeof(real_t));
    ws.ptr_i = (idx_t *)malloc(nweights * sizeof(idx_t));
    ws.ptr_j = (idx_t *)malloc(nweights * sizeof(idx_t));
    ws.weights = (real_t *)malloc(nweights * sizeof(real_t));
    ws.sort_indices = (idx_t *)malloc(nweights * sizeof(idx_t));

    SymmCOOMatrix prev;
    // Don't need `prev.diag` for partitioner so don't allocate it
    prev.offdiag_row_indices = (idx_t *)malloc(nweights * sizeof(idx_t));
    prev.offdiag_col_indices = (idx_t *)malloc(nweights * sizeof(idx_t));
    prev.offdiag_values = (real_t *)malloc(nweights * sizeof(real_t));

    while (current_dim > 100) {
        prev.dim = current_dim;
        prev.offdiag_nnz = matrices[levels - 1]->offdiag_nnz;

        for (idx_t k = 0; k < prev.offdiag_nnz; k++) {
            prev.offdiag_row_indices[k] = matrices[levels - 1]->offdiag_row_indices[k];
            prev.offdiag_col_indices[k] = matrices[levels - 1]->offdiag_col_indices[k];
            prev.offdiag_values[k] = matrices[levels - 1]->offdiag_values[k];
        }

        int failure = partition(near_null, free_dofs, coarsening_factor, &prev, &ws);
        if (failure) {
            break;
        }

        SymmCOOMatrix *a_coarse = (SymmCOOMatrix *)malloc(sizeof(SymmCOOMatrix));
        PiecewiseConstantTransfer *p =
                (PiecewiseConstantTransfer *)malloc(sizeof(PiecewiseConstantTransfer));

        p->fine_dim = current_dim;
        p->coarse_dim = prev.dim;
        p->partition = (idx_t *)malloc(current_dim * sizeof(idx_t));
        p->weights = (real_t *)malloc(current_dim * sizeof(real_t));

        for (idx_t k = 0; k < current_dim; k++) {
            p->partition[k] = ws.partition[k];
            p->weights[k] = near_null[k];
        }

        coarsen(matrices[levels - 1], p, a_coarse);
        matrices[levels] = a_coarse;
        transer_operators[levels - 1] = p;

        real_t resulting_cf = ((real_t)current_dim) / ((real_t)p->coarse_dim);
        levels++;
        printf("Added level %d\n\tcf: %.2f nrows: %td nnz: %td\n",
               levels,
               resulting_cf,
               prev.dim,
               prev.dim + (prev.offdiag_nnz * 2));
        current_dim = p->coarse_dim;
    }

    hierarchy->coarsening_factor = coarsening_factor;
    hierarchy->levels = levels;
    hierarchy->matrices = matrices;
    hierarchy->transer_operators = transer_operators;

    free(ws.partition);
    free(ws.rowsums);
    free(ws.ptr_i);
    free(ws.ptr_j);
    free(ws.weights);
    free(ws.sort_indices);

    free(prev.offdiag_row_indices);
    free(prev.offdiag_col_indices);
    free(prev.offdiag_values);

    free(near_null);
    return 0;
}
