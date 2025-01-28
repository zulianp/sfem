#ifndef BOUNDARY_CONDITION_INCORE_CUDA_H
#define BOUNDARY_CONDITION_INCORE_CUDA_H

#include <stddef.h>

#include "boundary_condition.h"
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void boundary_conditions_host_to_device(const boundary_condition_t *const host,
                                        boundary_condition_t *const device);

void d_constraint_nodes_copy_vec(const ptrdiff_t n_dirichlet_nodes,
                                 const idx_t *dirichlet_nodes,
                                 const int block_size,
                                 const int component,
                                 const real_t *source,
                                 real_t *dest);

void d_copy_at_dirichlet_nodes_vec(const int n_conditions,
                                   const boundary_condition_t *const cond,
                                   const int block_size,
                                   const real_t *const in,
                                   real_t *const out);

void d_constraint_nodes_to_value_vec(const ptrdiff_t n_dirichlet_nodes,
                                     const idx_t *dirichlet_nodes,
                                     const int block_size,
                                     const int component,
                                     const real_t value,
                                     real_t *values);

void d_constraint_nodes_to_values_vec(const ptrdiff_t n_dirichlet_nodes,
                                      const idx_t *dirichlet_nodes,
                                      const int block_size,
                                      const int component,
                                      const real_t *dirichlet_values,
                                      real_t *values);

void d_apply_dirichlet_condition_vec(const int n_conditions,
                                     const boundary_condition_t *const cond,
                                     const int block_size,
                                     real_t *const x);

void d_constraint_gradient_nodes_to_value_vec(const ptrdiff_t n_dirichlet_nodes,
                                              const idx_t *dirichlet_nodes,
                                              const int block_size,
                                              const int component,
                                              const real_t value,
                                              const real_t *const SFEM_RESTRICT x,
                                              real_t *const SFEM_RESTRICT g);

void cu_crs_constraint_nodes_to_identity_vec(const ptrdiff_t n_dirichlet_nodes,
                                             const idx_t *const SFEM_RESTRICT dirichlet_nodes,
                                             const int block_size,
                                             const int component,
                                             const real_t diag_value,
                                             const count_t *const SFEM_RESTRICT rowptr,
                                             const idx_t *const SFEM_RESTRICT colidx,
                                             real_t *const SFEM_RESTRICT values);

void d_destroy_conditions(const int n_conditions, boundary_condition_t *cond);

#ifdef __cplusplus
}
#endif
#endif  // BOUNDARY_CONDITION_INCORE_CUDA_H
