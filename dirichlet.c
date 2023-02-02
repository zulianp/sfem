#include "dirichlet.h"

#include "sfem_vec.h"
#include "sortreduce.h"

#include <assert.h>
#include <string.h>


static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - SFEM_VECTOR_SIZE; i += SFEM_VECTOR_SIZE) {
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }
    for (; i < size; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

static SFEM_INLINE int find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    if (lenrow <= 32) {
        return linear_search(key, row, lenrow);

        // Using sentinel (potentially dangerous if matrix is buggy and column does not exist)
        // while (key > row[++k]) {
        //     // Hi
        // }
        // assert(k < lenrow);
        // assert(key == row[k]);
    } else {
        // Use this for larger number of dofs per row
        return find_idx_binary_search(key, row, lenrow);
    }
}

void constraint_nodes_to_value(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const real_t value,
    real_t *values
    )
{
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        idx_t i = dirichlet_nodes[node];
        values[i] = value;
    }
}

void crs_constraint_nodes_to_identity(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const real_t diag_value,
    const idx_t *rowptr,
    const idx_t *colidx,
    real_t *values
    ) {
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        idx_t i = dirichlet_nodes[node];

        idx_t begin = rowptr[i];
        idx_t end = rowptr[i + 1];
        idx_t lenrow = end - begin;
        const idx_t *cols = &colidx[begin];
		real_t *row = &values[begin];

        memset(row, 0, sizeof(real_t) * lenrow);

        int k = find_col(i, cols, lenrow);
        assert(k >= 0);
        row[k] = diag_value;
    }
}
