#include "dirichlet.h"

#include "sfem_vec.h"
#include "sortreduce.h"

#include <assert.h>
#include <string.h>

#include <stdio.h>

static SFEM_INLINE idx_t linear_search(const idx_t target, const idx_t *const arr, const int size) {
    idx_t i;
    for (i = 0; i < size - 4; i += 4) {
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }
    for (; i < size; i++) {
        if (arr[i] == target) return i;
    }
    return SFEM_IDX_INVALID;
}

static SFEM_INLINE idx_t find_col(const idx_t key, const idx_t *const row, const int lenrow) {
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

void        constraint_nodes_to_value(const ptrdiff_t n_dirichlet_nodes,
                                      const idx_t    *dirichlet_nodes,
                                      const real_t    value,
                                      real_t         *values) {
#pragma omp parallel for
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        idx_t i   = dirichlet_nodes[node];
        values[i] = value;
    }
}

void        constraint_nodes_to_values(const ptrdiff_t             n_dirichlet_nodes,
                                       const idx_t                *dirichlet_nodes,
                                       const real_t *SFEM_RESTRICT dirichlet_values,
                                       real_t *SFEM_RESTRICT       values) {
#pragma omp parallel for
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        idx_t i   = dirichlet_nodes[node];
        values[i] = dirichlet_values[node];
    }
}

void constraint_nodes_copy(const ptrdiff_t n_dirichlet_nodes, const idx_t *dirichlet_nodes, const real_t *source, real_t *dest) {
#pragma omp parallel for
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        idx_t i = dirichlet_nodes[node];
        dest[i] = source[i];
    }
}

void        crs_constraint_nodes_to_identity(const ptrdiff_t n_dirichlet_nodes,
                                             const idx_t    *dirichlet_nodes,
                                             const real_t    diag_value,
                                             const count_t  *rowptr,
                                             const idx_t    *colidx,
                                             real_t         *values) {
#pragma omp parallel for
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        idx_t i = dirichlet_nodes[node];

        idx_t        begin  = rowptr[i];
        idx_t        end    = rowptr[i + 1];
        idx_t        lenrow = end - begin;
        const idx_t *cols   = &colidx[begin];
        real_t      *row    = &values[begin];

        memset(row, 0, sizeof(real_t) * lenrow);

        int k = find_col(i, cols, lenrow);
        assert(k >= 0);
        row[k] = diag_value;
    }
}

void        constraint_nodes_to_value_vec(const ptrdiff_t n_dirichlet_nodes,
                                          const idx_t    *dirichlet_nodes,
                                          const int       block_size,
                                          const int       component,
                                          const real_t    value,
                                          real_t         *values) {
#pragma omp parallel for
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        ptrdiff_t i = (ptrdiff_t)dirichlet_nodes[node] * block_size + component;
        values[i]   = value;
    }
}

void constraint_objective_nodes_to_value_vec(const ptrdiff_t                   n_dirichlet_nodes,
                                             const idx_t                      *dirichlet_nodes,
                                             const int                         block_size,
                                             const int                         component,
                                             const real_t                      value,
                                             const real_t *const SFEM_RESTRICT x,
                                             real_t *const SFEM_RESTRICT       out) {
    real_t acc = 0;
#pragma omp parallel for reduction(+ : acc)
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        ptrdiff_t    i    = (ptrdiff_t)dirichlet_nodes[node] * block_size + component;
        const real_t diff = x[i] - value;
        acc += diff * diff;
    }

    out[0] += acc / 2;
}

<<<<<<< HEAD
=======
void constraint_objective_nodes_to_values_vec(const ptrdiff_t                   n_dirichlet_nodes,
                                              const idx_t                      *dirichlet_nodes,
                                              const int                         block_size,
                                              const int                         component,
                                              const real_t                     *values,
                                              const real_t *const SFEM_RESTRICT x,
                                              real_t *const SFEM_RESTRICT       out) {
    real_t acc = 0;
#pragma omp parallel for reduction(+ : acc)
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        ptrdiff_t    i    = (ptrdiff_t)dirichlet_nodes[node] * block_size + component;
        const real_t diff = x[i] - values[node];
        acc += diff * diff;
    }

    out[0] += acc / 2;
}

>>>>>>> origin/main
void constraint_objective_nodes_to_value_vec_steps(const ptrdiff_t                   n_dirichlet_nodes,
                                                   const idx_t                      *dirichlet_nodes,
                                                   const int                         block_size,
                                                   const int                         component,
                                                   const real_t                      value,
                                                   const real_t *const SFEM_RESTRICT x,
                                                   const real_t *const SFEM_RESTRICT h,
                                                   const int                         nsteps,
                                                   const real_t *const SFEM_RESTRICT steps,
                                                   real_t *const SFEM_RESTRICT       out) {
#pragma omp parallel
    {
        real_t *acc = (real_t *)calloc(nsteps, sizeof(real_t));

        for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
            ptrdiff_t i = (ptrdiff_t)dirichlet_nodes[node] * block_size + component;
            for (int s = 0; s < nsteps; s++) {
                const real_t diff = (x[i] + steps[s] * h[i]) - value;
                acc[s] += diff * diff;
            }
        }

        for (int s = 0; s < nsteps; s++) {
#pragma omp atomic update
            out[s] += acc[s];
        }

        free(acc);
    }
}

<<<<<<< HEAD
=======
void constraint_objective_nodes_to_values_vec_steps(const ptrdiff_t                   n_dirichlet_nodes,
                                                    const idx_t                      *dirichlet_nodes,
                                                    const int                         block_size,
                                                    const int                         component,
                                                    const real_t                     *values,
                                                    const real_t *const SFEM_RESTRICT x,
                                                    const real_t *const SFEM_RESTRICT h,
                                                    const int                         nsteps,
                                                    const real_t *const SFEM_RESTRICT steps,
                                                    real_t *const SFEM_RESTRICT       out) {
#pragma omp parallel
    {
        real_t *acc = (real_t *)calloc(nsteps, sizeof(real_t));

        for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
            ptrdiff_t i = (ptrdiff_t)dirichlet_nodes[node] * block_size + component;
            for (int s = 0; s < nsteps; s++) {
                const real_t diff = (x[i] + steps[s] * h[i]) - values[node];
                acc[s] += diff * diff;
            }
        }

        for (int s = 0; s < nsteps; s++) {
#pragma omp atomic update
            out[s] += acc[s];
        }

        free(acc);
    }
}

>>>>>>> origin/main
void        constraint_gradient_nodes_to_value_vec(const ptrdiff_t                   n_dirichlet_nodes,
                                                   const idx_t                      *dirichlet_nodes,
                                                   const int                         block_size,
                                                   const int                         component,
                                                   const real_t                      value,
                                                   const real_t *const SFEM_RESTRICT x,
                                                   real_t *const SFEM_RESTRICT       g) {
#pragma omp parallel for
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        ptrdiff_t i = (ptrdiff_t)dirichlet_nodes[node] * block_size + component;
        g[i]        = x[i] - value;
    }
}

void        constraint_nodes_to_values_vec(const ptrdiff_t n_dirichlet_nodes,
                                           const idx_t    *dirichlet_nodes,
                                           const int       block_size,
                                           const int       component,
                                           const real_t   *dirichlet_values,
                                           real_t         *values) {
#pragma omp parallel for
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        ptrdiff_t i = (ptrdiff_t)dirichlet_nodes[node] * block_size + component;
        values[i]   = dirichlet_values[node];
    }
}

void        constraint_gradient_nodes_to_values_vec(const ptrdiff_t                   n_dirichlet_nodes,
                                                    const idx_t                      *dirichlet_nodes,
                                                    const int                         block_size,
                                                    const int                         component,
                                                    const real_t                     *dirichlet_values,
                                                    const real_t *const SFEM_RESTRICT x,
                                                    real_t *const SFEM_RESTRICT       g) {
#pragma omp parallel for
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        ptrdiff_t i = (ptrdiff_t)dirichlet_nodes[node] * block_size + component;
        g[i]        = x[i] - dirichlet_values[node];
    }
}

void        constraint_nodes_copy_vec(const ptrdiff_t n_dirichlet_nodes,
                                      const idx_t    *dirichlet_nodes,
                                      const int       block_size,
                                      const int       component,
                                      const real_t   *source,
                                      real_t         *dest) {
#pragma omp parallel for
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        ptrdiff_t i = (ptrdiff_t)dirichlet_nodes[node] * block_size + component;
        dest[i]     = source[i];
    }
}

void        crs_constraint_nodes_to_identity_vec(const ptrdiff_t n_dirichlet_nodes,
                                                 const idx_t    *dirichlet_nodes,
                                                 const int       block_size,
                                                 const int       component,
                                                 const real_t    diag_value,
                                                 const count_t  *rowptr,
                                                 const idx_t    *colidx,
                                                 real_t         *values) {
#pragma omp parallel for
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        ptrdiff_t i = (ptrdiff_t)dirichlet_nodes[node] * block_size + component;

        const count_t begin  = rowptr[i];
        const count_t end    = rowptr[i + 1];
        const count_t lenrow = end - begin;
        const idx_t  *cols   = &colidx[begin];
        real_t       *row    = &values[begin];

        memset(row, 0, sizeof(real_t) * lenrow);

        int k = find_col(i, cols, lenrow);
        assert(k >= 0);
        row[k] = diag_value;
    }
}

void bsr_constraint_nodes_to_identity_vec(const ptrdiff_t n_dirichlet_nodes,
                                          const idx_t    *dirichlet_nodes,
                                          const int       block_size,
                                          const int       component,
                                          const real_t    diag_value,
                                          const count_t  *rowptr,
                                          const idx_t    *colidx,
                                          real_t         *values) {
    const int mat_block_size = block_size * block_size;
    const int b_offset       = component * block_size;
#pragma omp parallel for
    for (ptrdiff_t node = 0; node < n_dirichlet_nodes; ++node) {
        const idx_t i = dirichlet_nodes[node];

        const count_t begin  = rowptr[i];
        const count_t end    = rowptr[i + 1];
        const count_t lenrow = end - begin;
        const idx_t  *cols   = &colidx[begin];

        for (count_t k = 0; k < lenrow; k++) {
            const idx_t   col   = cols[k];
            real_t *const block = &values[(begin + k) * mat_block_size];

            if (col == i) {
                for (int d = 0; d < block_size; d++) {
                    block[b_offset + d] = (d == component) * diag_value;
                }
            } else {
                for (int d = 0; d < block_size; d++) {
                    block[b_offset + d] = 0;
                }
            }
        }
    }
}
