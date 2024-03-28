#ifndef BOUNDARY_CONDITION_H
#define BOUNDARY_CONDITION_H

#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ptrdiff_t local_size, global_size;
    idx_t *idx;
    int component;
    // Set for uniform
    real_t value;

    // Set for varying
    real_t *values;
} boundary_condition_t;

void add_neumann_condition_to_gradient_vec(const enum ElemType element_type,
                                           geom_t **const SFEM_RESTRICT points,
                                           const int n_conditions,
                                           const boundary_condition_t *const cond,
                                           const int block_size,
                                           real_t *g);

void apply_dirichlet_condition_vec(const int n_conditions,
                                   const boundary_condition_t *const cond,
                                   const int block_size,
                                   real_t *const x);

void apply_zero_dirichlet_condition_vec(const int n_conditions,
                                        const boundary_condition_t *const cond,
                                        const int block_size,
                                        real_t *const x);

void copy_at_dirichlet_nodes_vec(const int n_conditions,
                                 const boundary_condition_t *const cond,
                                 const int block_size,
                                 const real_t *const in,
                                 real_t *const out);

void apply_dirichlet_condition_to_hessian_crs_vec(const int n_conditions,
                                                  const boundary_condition_t *const cond,
                                                  const int block_size,
                                                  const count_t *const rowptr,
                                                  const idx_t *const colidx,
                                                  real_t *const values);

void destroy_conditions(const int n_conditions, boundary_condition_t *cond);

#ifdef __cplusplus
}
#endif
#endif  // BOUNDARY_CONDITION_H
