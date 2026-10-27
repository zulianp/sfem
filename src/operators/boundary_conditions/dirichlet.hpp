#ifndef SFEM_DIRICHLET_H
#define SFEM_DIRICHLET_H

#include <stddef.h>
#include "sfem_base.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void constraint_nodes_to_value(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const real_t value,
    real_t *values
    );

void constraint_nodes_to_values(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const real_t * SFEM_RESTRICT dirichlet_values,
    real_t * SFEM_RESTRICT values
    );

void constraint_nodes_copy(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const real_t *source,
    real_t *dest
    );


void crs_constraint_nodes_to_identity(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const real_t diag_value,
    const count_t *rowptr,
    const idx_t *colidx,
    real_t *values
    );

// Vector version

void constraint_nodes_to_value_vec(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const int block_size,
    const int component,
    const real_t value,
    real_t *values
    );

void constraint_objective_nodes_to_value_vec(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const int block_size,
    const int component,
    const real_t value,
    const real_t *const SFEM_RESTRICT x,
    real_t *const SFEM_RESTRICT out);

void constraint_objective_nodes_to_values_vec(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const int block_size,
    const int component,
    const real_t *values,
    const real_t *const SFEM_RESTRICT x,
    real_t *const SFEM_RESTRICT out);

void constraint_objective_nodes_to_value_vec_steps(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const int block_size,
    const int component,
    const real_t value,
    const real_t *const SFEM_RESTRICT x,
    const real_t *const SFEM_RESTRICT h,
    const int nsteps,
    const real_t *const SFEM_RESTRICT steps,
    real_t *const SFEM_RESTRICT out);


void constraint_objective_nodes_to_values_vec_steps(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const int block_size,
    const int component,
    const real_t *values,
    const real_t *const SFEM_RESTRICT x,
    const real_t *const SFEM_RESTRICT h,
    const int nsteps,
    const real_t *const SFEM_RESTRICT steps,
    real_t *const SFEM_RESTRICT out);

void constraint_gradient_nodes_to_value_vec(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const int block_size,
    const int component,
    const real_t value,
    const real_t *const SFEM_RESTRICT x,
    real_t *const SFEM_RESTRICT g);

void constraint_nodes_to_values_vec(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const int block_size,
    const int component,
    const real_t *dirichlet_values,
    real_t *values
    );

void constraint_gradient_nodes_to_values_vec(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const int block_size,
    const int component,
    const real_t *dirichlet_values,
    const real_t *const SFEM_RESTRICT x,
    real_t *const SFEM_RESTRICT g
);

void constraint_nodes_copy_vec(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const int block_size,
    const int component,
    const real_t *source,
    real_t *dest
    );


void crs_constraint_nodes_to_identity_vec(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const int block_size,
    const int component,
    const real_t diag_value,
    const count_t *rowptr,
    const idx_t *colidx,
    real_t *values
    );

#ifdef __cplusplus
}

// The rows of `component` at the nodes, of BSR values in the tangent's value type V: diag_value on
// the diagonal, zero elsewhere.
template <typename V>
void bsr_constraint_nodes_to_identity_vec(const ptrdiff_t n_dirichlet_nodes,
                                          const idx_t    *dirichlet_nodes,
                                          const int       block_size,
                                          const int       component,
                                          const real_t    diag_value,
                                          const count_t  *rowptr,
                                          const idx_t    *colidx,
                                          V              *values) {
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
            const idx_t col   = cols[k];
            V *const    block = &values[(begin + k) * mat_block_size];

            if (col == i) {
                for (int d = 0; d < block_size; d++) {
                    block[b_offset + d] = V((d == component) * diag_value);
                }
            } else {
                for (int d = 0; d < block_size; d++) {
                    block[b_offset + d] = 0;
                }
            }
        }
    }
}
#endif
#endif //SFEM_DIRICHLET_H

