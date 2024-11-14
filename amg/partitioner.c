#include "partitioner.h"
#include <stdlib.h>
#include "sparse.h"

// Helper functions... not public API so I don't think they go in header file?
// idk I don't code in C.
void sort_weights(PartitionerWorkspace *ws, count_t nweights);
int pairwise_aggregation(const real_t coarsening_factor,
                         const real_t inv_total,
                         const ptrdiff_t fine_nrows,
                         SymmCOOMatrix *a_bar,
                         PartitionerWorkspace *ws);

// Global variables needed for qsort...
real_t *global_weights;

int partition(const real_t *near_null,
              const idx_t *free_dofs,
              const real_t coarsening_factor,
              SymmCOOMatrix *symm_coo,
              PartitionerWorkspace *ws) {
    ptrdiff_t nrows = symm_coo->dim;
    count_t nnz = symm_coo->offdiag_nnz;
    real_t current_cf = 1.0;

    // We start with each free DOF in its own aggregate
    idx_t agg_id = 0;
    for (idx_t row = 0; row < nrows; row++) {
        if (free_dofs[row]) {
            ws->partition[row] = agg_id;
        } else {
            ws->partition[row] = -1;
        }
    }

    // Calculate the row sums of the augmented matrix and the weights for the
    // strength of connection graph
    for (idx_t k = 0; k < nnz; k++) {
        idx_t i = symm_coo->offdiag_row_indices[k];
        idx_t j = symm_coo->offdiag_col_indices[k];
        if (free_dofs[i] && free_dofs[j]) {
            real_t val = symm_coo->offdiag_values[k];

            real_t weight = -val * near_null[i] * near_null[j];
            // only not thread safe part here, could parallel this...
            ws->rowsums[i] += weight;
            ws->rowsums[j] += weight;
            symm_coo->offdiag_values[k] = weight;
        }
    }

    // Calculate the total sum of the augmented matrix and fix any negative
    // rowsums
    real_t inv_total = 0.0;
    for (idx_t row = 0; row < nrows; row++) {
        // TODO negative rowsums handling.... maybe log it? idk
        real_t rowsum = ws->rowsums[row];
        if (rowsum < 0.0) {
            ws->rowsums[row] = 0.0;
        } else {
            inv_total += rowsum;
        }
    }
    inv_total = 1.0 / inv_total;

    real_t fine_nrows_float = (real_t)nrows;

    while (current_cf < coarsening_factor) {
        if (pairwise_aggregation(coarsening_factor, inv_total, nrows, symm_coo, ws)) {
            return 1;
        }

        real_t coarse_nrows = (real_t)symm_coo->dim;
        current_cf = fine_nrows_float / coarse_nrows;
        /*
        printf("Matching step completed, cf: %.2f nrows: %d nnz: %d\n", current_cf,
               symm_coo->dim, symm_coo->dim + (symm_coo->offdiag_nnz * 2));
        */
    }

    return 0;
}

int pairwise_aggregation(const real_t coarsening_factor,
                         const real_t inv_total,
                         const ptrdiff_t fine_nrows,
                         SymmCOOMatrix *a_bar,
                         PartitionerWorkspace *ws) {
    count_t nnz = a_bar->offdiag_nnz;
    ptrdiff_t nrows = a_bar->dim;

    // Compute the modularity weights for the augmented graph, storing only
    // positive modularity weights
    count_t n_mod_weights = 0;
    for (idx_t k = 0; k < nnz; k++) {
        idx_t i = a_bar->offdiag_row_indices[k];
        idx_t j = a_bar->offdiag_col_indices[k];

        if (ws->partition[i] >= 0 && ws->partition[j] >= 0) {
            real_t val = a_bar->offdiag_values[k];
            real_t mod_weight = val - inv_total * ws->rowsums[i] * ws->rowsums[j];

            if (mod_weight > 0.0) {
                ws->weights[n_mod_weights] = mod_weight;
                ws->ptr_i[n_mod_weights] = i;
                ws->ptr_j[n_mod_weights] = j;
                n_mod_weights += 1;
            }
        }
    }

    // Cannot coarsen any more 'usefully' if all modularity weights are negative
    if (!n_mod_weights) {
        return 1;
    }

    // Sorting modularity weights makes greedy matching efficient
    sort_weights(ws, n_mod_weights);

    // Repurpose sorting array to track which DOFs have been assigned coarse
    // locations
    idx_t *alive = ws->sort_indices;
#pragma omp parallel for
    for (idx_t row = 0; row < nrows; row++) {
        alive[row] = -1;
    }

    // Assign match pairs to coarse grid values
    idx_t coarse_counter = 0;
    for (idx_t k = 0; k < n_mod_weights; k++) {
        idx_t i = ws->ptr_i[k];
        idx_t j = ws->ptr_j[k];
        if (alive[i] < 0 && alive[j] < 0) {
            alive[i] = coarse_counter;
            alive[j] = coarse_counter;
            coarse_counter += 1;
        }
    }

    // Assign any unmatched DOF to singleton on coarse grid
    for (idx_t row = 0; row < nrows; row++) {
        if (alive[row] < 0 && ws->partition[row] >= 0) {
            alive[row] = coarse_counter;
            coarse_counter += 1;
        }
    }

    // Update the partition to reflect the assigned matching
    for (idx_t row = 0; row < fine_nrows; row++) {
        if (ws->partition[row] >= 0) {
            idx_t old_agg = ws->partition[row];
            ws->partition[row] = alive[old_agg];
        }
    }

    // Update the augmented matrix
    idx_t new_nnz = 0;
    for (idx_t k = 0; k < nnz; k++) {
        idx_t i = alive[a_bar->offdiag_row_indices[k]];
        idx_t j = alive[a_bar->offdiag_col_indices[k]];

        if (j > i) {
            a_bar->offdiag_row_indices[new_nnz] = i;
            a_bar->offdiag_col_indices[new_nnz] = j;
            a_bar->offdiag_values[new_nnz] = a_bar->offdiag_values[k];
            new_nnz += 1;
        } else if (i > j) {
            a_bar->offdiag_row_indices[new_nnz] = j;
            a_bar->offdiag_col_indices[new_nnz] = i;
            a_bar->offdiag_values[new_nnz] = a_bar->offdiag_values[k];
            new_nnz += 1;
        }
    }
    a_bar->dim = coarse_counter;

    // Fix augmented matrix
    sum_duplicates(a_bar->offdiag_row_indices,
                   a_bar->offdiag_col_indices,
                   a_bar->offdiag_values,
                   ws->sort_indices,
                   &new_nnz);
    a_bar->offdiag_nnz = new_nnz;

    // Update rowsums using weights workspace
    for (idx_t row = 0; row < coarse_counter; row++) {
        ws->weights[row] = 0.0;
    }
    for (idx_t row = 0; row < nrows; row++) {
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

int compare_indices(const void *a, const void *b) {
    idx_t idx_a = *(const idx_t *)a;
    idx_t idx_b = *(const idx_t *)b;
    if (global_weights[idx_a] < global_weights[idx_b]) return 1;
    if (global_weights[idx_a] > global_weights[idx_b]) return -1;
    return 0;
}

void sort_weights(PartitionerWorkspace *ws, count_t nweights) {
    idx_t *indices = ws->sort_indices;
    idx_t *ptr_i = ws->ptr_i;
    idx_t *ptr_j = ws->ptr_j;
    real_t *weights = ws->weights;
    global_weights = weights;

    for (idx_t i = 0; i < nweights; i++) {
        indices[i] = i;
    }

    // TODO parallel sort is must here
    qsort(indices, nweights, sizeof(idx_t), compare_indices);
    cycle_leader_swap(ptr_i, ptr_j, weights, indices, nweights);
}
