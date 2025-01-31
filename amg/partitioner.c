#include "partitioner.h"
#include <math.h>
#include "coo_sort.h"
#include "sfem_base.h"
#include "sfem_mask.h"

int pairwise_aggregation(const real_t          inv_total,
                         const ptrdiff_t       fine_ndofs,
                         const count_t         max_agg_size,
                         count_t              *offdiag_nnz,
                         ptrdiff_t            *ndofs,
                         idx_t                *offdiag_row_indices,
                         idx_t                *offdiag_col_indices,
                         real_t               *offdiag_values,
                         PartitionerWorkspace *ws);

// TODO cap agg sizes...
int partition(const mask_t         *bdy_dofs,
              const real_t          coarsening_factor,
              const real_t         *near_null,
              idx_t                *offdiag_row_indices,
              idx_t                *offdiag_col_indices,
              real_t               *offdiag_values,
              count_t              *offdiag_nnz,
              ptrdiff_t            *ndofs,
              PartitionerWorkspace *ws) {
    real_t  current_cf   = 1.0;
    count_t max_agg_size = ceil(coarsening_factor);

    ptrdiff_t fine_ndofs = *ndofs;

    // We start with each free DOF in its own aggregate
    idx_t aggs = 0;
    for (idx_t row = 0; row < *ndofs; row++) {
        if ((!bdy_dofs) || !mask_get(row, bdy_dofs)) {
            ws->partition[row]  = aggs;
            ws->agg_sizes[aggs] = 1;
            ws->near_null[aggs] = near_null[row];
            aggs += 1;
        } else {
            ws->partition[row] = -1;
        }
    }
    *ndofs = aggs;

    // If we have boundary restriction shift everything
    if (bdy_dofs) {
        count_t restricted_nnz = 0;
        for (idx_t k = 0; k < *offdiag_nnz; k++) {
            idx_t  i   = offdiag_row_indices[k];
            idx_t  j   = offdiag_col_indices[k];
            real_t val = offdiag_values[k];
            if (ws->partition[i] >= 0 && ws->partition[j] >= 0 && val != 0.0) {
                offdiag_row_indices[restricted_nnz] = ws->partition[i];
                offdiag_col_indices[restricted_nnz] = ws->partition[j];
                offdiag_values[restricted_nnz]      = val;
                restricted_nnz++;
            }
        }
        *offdiag_nnz = restricted_nnz;
    }

    // Calculate the row sums of the augmented matrix and the weights for the
    // strength of connection graph
    for (idx_t k = 0; k < *offdiag_nnz; k++) {
        idx_t  i   = offdiag_row_indices[k];
        idx_t  j   = offdiag_col_indices[k];
        real_t val = offdiag_values[k];

        // printf("%d, %d\n", i, j);
        real_t weight = -val * ws->near_null[i] * ws->near_null[j];
        // only not thread safe part here, could parallel this...
        ws->rowsums[i] += weight;
        ws->rowsums[j] += weight;
        offdiag_values[k] = weight;
    }

    // Calculate the total sum of the augmented matrix and fix any negative
    // rowsums
    real_t inv_total = 0.0;
    for (idx_t row = 0; row < *ndofs; row++) {
        // TODO negative rowsums handling.... maybe log it? idk
        real_t rowsum = ws->rowsums[row];
        if (rowsum < 0.0) {
            ws->rowsums[row] = 0.0;
        } else {
            inv_total += rowsum;
        }
    }
    inv_total = 1.0 / inv_total;

    real_t fine_ndofs_float = (real_t)fine_ndofs;

    int finished = 0;
    while (current_cf < coarsening_factor && !finished) {
        finished = pairwise_aggregation(inv_total,
                                        fine_ndofs,
                                        max_agg_size,
                                        offdiag_nnz,
                                        ndofs,
                                        offdiag_row_indices,
                                        offdiag_col_indices,
                                        offdiag_values,
                                        ws);

        if (fine_ndofs == *ndofs) {
            return 1;
        }

        real_t coarse_nrows = (real_t)*ndofs;
        current_cf          = fine_ndofs_float / coarse_nrows;

        // update the aggsizes
        for (idx_t row = 0; row < coarse_nrows; row++) {
            ws->agg_sizes[row] = 0;
        }

        for (idx_t row = 0; row < fine_ndofs; row++) {
            idx_t agg_id = ws->partition[row];
            if (agg_id > 0) {
                ws->agg_sizes[agg_id]++;
                assert(ws->agg_sizes[agg_id] <= max_agg_size);
            }
        }

        /*
        printf("Matching step completed, cf: %.2f nrows: %d offdiag nnz: %d\n",
               current_cf,
               (int)*ndofs,
               (int)*offdiag_nnz);
        */
    }

    return 0;
}

int pairwise_aggregation(const real_t          inv_total,
                         const ptrdiff_t       fine_ndofs,
                         const count_t         max_agg_size,
                         count_t              *offdiag_nnz,
                         ptrdiff_t            *ndofs,
                         idx_t                *offdiag_row_indices,
                         idx_t                *offdiag_col_indices,
                         real_t               *offdiag_values,
                         PartitionerWorkspace *ws) {
    // Compute the modularity weights for the augmented graph, storing only
    // positive modularity weights
    count_t n_mod_weights = 0;
    for (idx_t k = 0; k < *offdiag_nnz; k++) {
        idx_t  i          = offdiag_row_indices[k];
        idx_t  j          = offdiag_col_indices[k];
        real_t val        = offdiag_values[k];
        real_t mod_weight = val - inv_total * ws->rowsums[i] * ws->rowsums[j];

        count_t agg_i = ws->agg_sizes[i];
        count_t agg_j = ws->agg_sizes[j];
        // if (mod_weight > 0.0 && agg_i < max_agg_size && agg_j < max_agg_size) {
        if (agg_i < max_agg_size && agg_j < max_agg_size) {
            ws->weights[n_mod_weights] = mod_weight;
            ws->ptr_i[n_mod_weights]   = i;
            ws->ptr_j[n_mod_weights]   = j;
            n_mod_weights += 1;
        }
    }

    // Cannot coarsen any more 'usefully'
    if (!n_mod_weights) {
        return 1;
    }

    // Sorting modularity weights makes greedy matching efficient
    sort_weights(ws->sort_indices, ws->ptr_i, ws->ptr_j, ws->weights, n_mod_weights);

    // Repurpose sorting array to track which DOFs have been assigned coarse
    // locations
    count_t *alive = ws->sort_indices;
#pragma omp parallel for
    for (idx_t row = 0; row < *ndofs; row++) {
        alive[row] = -1;
    }

    // Assign match pairs to coarse grid values
    idx_t coarse_counter = 0;
    for (idx_t k = 0; k < n_mod_weights; k++) {
        idx_t   i            = ws->ptr_i[k];
        idx_t   j            = ws->ptr_j[k];
        count_t new_agg_size = ws->agg_sizes[i] + ws->agg_sizes[j];
        if (alive[i] < 0 && alive[j] < 0 && new_agg_size <= max_agg_size) {
            alive[i] = coarse_counter;
            alive[j] = coarse_counter;
            coarse_counter += 1;
        }
    }

    // no pairs to match so we are done
    if (!coarse_counter) {
        return 1;
    }

    // Assign any unmatched DOF to singleton on coarse grid
    for (idx_t row = 0; row < *ndofs; row++) {
        if (alive[row] < 0) {
            alive[row] = coarse_counter;
            coarse_counter += 1;
        }
    }

    // Update the partition to reflect the assigned matching
    for (idx_t row = 0; row < fine_ndofs; row++) {
        if (ws->partition[row] >= 0) {
            idx_t old_agg      = ws->partition[row];
            ws->partition[row] = alive[old_agg];
        }
    }

    // Update the augmented matrix
    // TODO verify this with a spmm implementation
    // also could be abstracted away to the PWCpartition->coarsen API
    idx_t new_nnz = 0;
    for (idx_t k = 0; k < *offdiag_nnz; k++) {
        idx_t i = alive[offdiag_row_indices[k]];
        idx_t j = alive[offdiag_col_indices[k]];

        if (j > i) {
            offdiag_row_indices[new_nnz] = i;
            offdiag_col_indices[new_nnz] = j;
            offdiag_values[new_nnz]      = offdiag_values[k];
            new_nnz += 1;
        } else if (i > j) {
            offdiag_row_indices[new_nnz] = j;
            offdiag_col_indices[new_nnz] = i;
            offdiag_values[new_nnz]      = offdiag_values[k];
            new_nnz += 1;
        }
    }
    *ndofs       = coarse_counter;
    *offdiag_nnz = new_nnz;

    // Fix augmented matrix
    sum_duplicates(ws->sort_indices, offdiag_row_indices, offdiag_col_indices, offdiag_values, offdiag_nnz);

    // Update rowsums using weights workspace
    for (idx_t row = 0; row < coarse_counter; row++) {
        ws->weights[row] = 0.0;
    }
    for (idx_t row = 0; row < *ndofs; row++) {
        if (alive[row] >= 0) {
            idx_t coarse_idx = alive[row];
            ws->weights[coarse_idx] += ws->rowsums[row];
        }
    }
    for (idx_t row = 0; row < coarse_counter; row++) {
        ws->rowsums[row] = ws->weights[row];
    }

    return 0;
}

PartitionerWorkspace *create_partition_ws(const ptrdiff_t fine_ndofs, const count_t offdiag_nnz) {
    PartitionerWorkspace *ws = malloc(sizeof(PartitionerWorkspace));
    ws->partition            = (idx_t *)malloc((fine_ndofs) * sizeof(idx_t));
    ws->rowsums              = (real_t *)calloc((fine_ndofs), sizeof(real_t));
    ws->near_null            = (real_t *)calloc((fine_ndofs), sizeof(real_t));
    ws->ptr_i                = (idx_t *)malloc(offdiag_nnz * sizeof(idx_t));
    ws->ptr_j                = (idx_t *)malloc(offdiag_nnz * sizeof(idx_t));
    ws->weights              = (real_t *)malloc(offdiag_nnz * sizeof(real_t));
    ws->sort_indices         = (count_t *)malloc(offdiag_nnz * sizeof(count_t));
    ws->agg_sizes            = (count_t *)malloc(fine_ndofs * sizeof(count_t));
    return ws;
}

int free_partition_ws(PartitionerWorkspace *ws) {
    free(ws->partition);
    free(ws->rowsums);
    free(ws->ptr_i);
    free(ws->ptr_j);
    free(ws->weights);
    free(ws->sort_indices);
    free(ws->agg_sizes);
    free(ws);

    return SFEM_SUCCESS;
}
