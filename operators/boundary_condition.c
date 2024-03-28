#include "boundary_condition.h"

#include "dirichlet.h"
#include "neumann.h"
#include "sfem_defs.h"

#include <assert.h>
#include <math.h>

void add_neumann_condition_to_gradient_vec(const enum ElemType element_type,
                                           geom_t **const SFEM_RESTRICT points,
                                           const int n_conditions,
                                           const boundary_condition_t *const cond,
                                           const int block_size,
                                           real_t *g) {
    for (int i = 0; i < n_conditions; i++) {
        surface_forcing_function_vec(side_type(element_type),
                                     cond[i].local_size,
                                     cond[i].idx,
                                     points,
                                     -  // Use negative sign since we are on LHS
                                     cond[i].value,
                                     block_size,
                                     cond[i].component,
                                     g);
    }
}

void apply_dirichlet_condition_vec(const int n_conditions,
                                   const boundary_condition_t *const cond,
                                   const int block_size,
                                   real_t *const x) {
    for (int i = 0; i < n_conditions; i++) {
        if (cond[i].values) {
            constraint_nodes_to_values_vec(
                cond[i].local_size, cond[i].idx, block_size, cond[i].component, cond[i].values, x);
        } else {
            constraint_nodes_to_value_vec(
                cond[i].local_size, cond[i].idx, block_size, cond[i].component, cond[i].value, x);
        }
    }
}

void apply_zero_dirichlet_condition_vec(const int n_conditions,
                                        const boundary_condition_t *const cond,
                                        const int block_size,
                                        real_t *const x) {
    for (int i = 0; i < n_conditions; i++) {
        constraint_nodes_to_value_vec(
            cond[i].local_size, cond[i].idx, block_size, cond[i].component, 0, x);
    }
}

void copy_at_dirichlet_nodes_vec(const int n_conditions,
                                 const boundary_condition_t *const cond,
                                 const int block_size,
                                 const real_t *const in,
                                 real_t *const out) {
    for (int i = 0; i < n_conditions; i++) {
        constraint_nodes_copy_vec(
            cond[i].local_size, cond[i].idx, block_size, cond[i].component, in, out);
    }
}

void apply_dirichlet_condition_to_hessian_crs_vec(const int n_conditions,
                                                  const boundary_condition_t *const cond,
                                                  const int block_size,
                                                  const count_t *const rowptr,
                                                  const idx_t *const colidx,
                                                  real_t *const values) {
    for (int i = 0; i < n_conditions; i++) {
        crs_constraint_nodes_to_identity_vec(cond[i].local_size,
                                             cond[i].idx,
                                             block_size,
                                             cond[i].component,
                                             1,
                                             rowptr,
                                             colidx,
                                             values);
    }
}

void destroy_conditions(const int n_conditions, boundary_condition_t *cond) {
    for (int i = 0; i < n_conditions; i++) {
        free(cond[i].idx);
    }

    free(cond);
}
