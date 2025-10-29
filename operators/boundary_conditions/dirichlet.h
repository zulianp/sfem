#ifndef SFEM_DIRICHLET_H
#define SFEM_DIRICHLET_H

#include <stddef.h>
#include "sfem_base.h"

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

void bsr_constraint_nodes_to_identity_vec(const ptrdiff_t n_dirichlet_nodes,
                                          const idx_t *dirichlet_nodes,
                                          const int block_size,
                                          const int component,
                                          const real_t diag_value,
                                          const count_t *rowptr,
                                          const idx_t *colidx,
                                          real_t *values);

#ifdef __cplusplus
}
#endif
#endif //SFEM_DIRICHLET_H
