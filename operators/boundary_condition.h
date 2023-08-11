#ifndef BOUNDARY_CONDITION_H
#define BOUNDARY_CONDITION_H

#include <mpi.h>
#include <stddef.h>

#include "sfem_base.h"
#include "sfem_mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ptrdiff_t local_size, global_size;
    idx_t *idx;
    int component;
    real_t value;
} boundary_condition_t;

void read_boundary_conditions(MPI_Comm comm,
                              char *sets,
                              char *values,
                              char *components,
                              boundary_condition_t **bcs,
                              int *nbc);

void read_dirichlet_conditions(const mesh_t *const mesh,
                               char *sets,
                               char *values,
                               char *components,
                               boundary_condition_t **bcs,
                               int *nbc);

void read_neumann_conditions(const mesh_t *const mesh,
                             char *sets,
                             char *values,
                             char *components,
                             boundary_condition_t **bcs,
                             int *nbc);

void add_neumann_condition_to_gradient_vec(const int n_conditions,
                                           const boundary_condition_t *const cond,
                                           const mesh_t *const mesh,
                                           const int block_size,
                                           real_t *g);

void apply_dirichlet_condition_vec(const int n_conditions,
                                   const boundary_condition_t *const cond,
                                   const mesh_t *const mesh,
                                   const int block_size,
                                   real_t *const x);

void apply_zero_dirichlet_condition_vec(const int n_conditions,
                                   const boundary_condition_t *const cond,
                                   const mesh_t *const mesh,
                                   const int block_size,
                                   real_t *const x);

void copy_at_dirichlet_nodes_vec(const int n_conditions,
                                   const boundary_condition_t *const cond,
                                   const mesh_t *const mesh,
                                   const int block_size,
                                   const real_t *const in,
                                   real_t *const out);

void apply_dirichlet_condition_to_hessian_crs_vec(const int n_conditions,
                                                  const boundary_condition_t *const cond,
                                                  const mesh_t *const mesh,
                                                  const int block_size,
                                                  const count_t *const rowptr,
                                                  const idx_t *const colidx,
                                                  real_t *const values);

void destroy_conditions(const int n_conditions, boundary_condition_t * cond);

#ifdef __cplusplus
}
#endif
#endif  // BOUNDARY_CONDITION_H
