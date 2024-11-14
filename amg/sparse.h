#ifndef SPARSE_H
#define SPARSE_H

#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include "sfem_base.h"

typedef struct {
    ptrdiff_t dim;  // Number of rows / columns

    count_t offdiag_nnz;         // Length of 3 below arrays, or half the actualy number of
                                 // off diagonal non-zero entries
    idx_t *offdiag_row_indices;  // Row indices of off diagonal non-zero entries
    idx_t *offdiag_col_indices;  // Column indices of off diagonal non-zero entries
    real_t *offdiag_values;      // Off diagonal non-zero values

    real_t *diag;  // Diagonal values
} SymmCOOMatrix;

/* Function declarations */
void csr_to_symmcoo(const ptrdiff_t nrows,
                    const count_t nnz,
                    const count_t *row_ptr,
                    const idx_t *col_indices,
                    const real_t *values,
                    SymmCOOMatrix *coo);
void coo_symm_spmv(const SymmCOOMatrix *a, const real_t *x, real_t *y);

/* Helper Functions */
// Ignores elements with negative row or column indices as way to filter
void sum_duplicates(idx_t *rows, idx_t *cols, real_t *values, idx_t *indices, count_t *N_ptr);
void cycle_leader_swap(idx_t *rows, idx_t *cols, real_t *values, idx_t *indices, count_t N);
void load_binary_file(const char *filename, void *buffer, size_t size);
#endif  // SPARSE_H
