#include "sparse.h"
#include <assert.h>
#include <stdio.h>

idx_t *row_indices;
idx_t *col_indices;

void csr_to_symmcoo(const ptrdiff_t nrows,
                    const count_t nnz,
                    const count_t *row_ptr,
                    const idx_t *col_indices,
                    const real_t *values,
                    SymmCOOMatrix *coo) {
    assert(nnz % 2 == 0);

    count_t nweights = (nnz - nrows) / 2;
    coo->offdiag_nnz = nweights;
    coo->dim = nrows;

    idx_t k = 0;
    for (idx_t i = 0; i < nrows; i++) {
        for (idx_t idx = row_ptr[i]; idx < row_ptr[i + 1]; idx++) {
            idx_t j = col_indices[idx];
            if (j > i) {
                coo->offdiag_row_indices[k] = i;
                coo->offdiag_col_indices[k] = j;
                coo->offdiag_values[k] = values[idx];
                k += 1;
            } else if (i == j) {
                coo->diag[i] = values[idx];
            }
        }
    }
}

/* Sparse Matrix-Vector Multiplication using SymmCOO format */
void coo_symm_spmv(const SymmCOOMatrix *a, const real_t *x, real_t *y) {
#pragma omp parallel for
    for (idx_t k = 0; k < a->dim; k++) {
        y[k] = a->diag[k] * x[k];
    }

    for (idx_t k = 0; k < a->offdiag_nnz; k++) {
        idx_t i = a->offdiag_row_indices[k];
        idx_t j = a->offdiag_col_indices[k];
        real_t val = a->offdiag_values[k];
        y[i] += x[j] * val;
        y[j] += x[i] * val;
    }
}

// Function to load binary data from file
void load_binary_file(const char *filename, void *buffer, size_t size) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fread(buffer, size, 1, file);
    fclose(file);
}

int compare_indices_coo(const void *a, const void *b) {
    idx_t idx_a = *(const idx_t *)a;
    idx_t idx_b = *(const idx_t *)b;

    if (row_indices[idx_a] != row_indices[idx_b])
        return row_indices[idx_a] - row_indices[idx_b];
    else
        return col_indices[idx_a] - col_indices[idx_b];
}

void sum_duplicates(idx_t *rows, idx_t *cols, real_t *values, idx_t *indices, count_t *N_ptr) {
    count_t N = *N_ptr;

    // Assign the global pointers
    row_indices = rows;
    col_indices = cols;

    for (idx_t i = 0; i < N; i++) {
        indices[i] = i;
    }

    // Sort the indices array
    qsort(indices, N, sizeof(idx_t), compare_indices_coo);
    cycle_leader_swap(rows, cols, values, indices, N);

    // Compact the arrays
    idx_t write_pos = 0;
    idx_t prev_row = -1;
    idx_t prev_col = -1;

    for (idx_t k = 0; k < N; k++) {
        idx_t idx = indices[k];
        idx_t i = rows[idx];
        idx_t j = cols[idx];
        real_t v = values[idx];

        if (i >= 0 && j >= 0) {
            if (i == prev_row && j == prev_col) {
                // Duplicate entry found; sum the values
                values[write_pos - 1] += v;
            } else {
                // New unique entry; copy to the write position
                rows[write_pos] = i;
                cols[write_pos] = j;
                values[write_pos] = v;
                write_pos++;
            }

            prev_row = i;
            prev_col = j;
        }
    }

    *N_ptr = write_pos;
}

void cycle_leader_swap(idx_t *rows, idx_t *cols, real_t *values, idx_t *indices, count_t N) {
    for (idx_t i = 0; i < N; i++) {
        idx_t current = i;
        while (indices[current] != i) {
            idx_t next = indices[current];

            idx_t temp_row = rows[current];
            rows[current] = rows[next];
            rows[next] = temp_row;

            idx_t temp_col = cols[current];
            cols[current] = cols[next];
            cols[next] = temp_col;

            real_t temp_val = values[current];
            values[current] = values[next];
            values[next] = temp_val;

            indices[current] = current;
            current = next;
        }
        indices[current] = current;
    }
}
