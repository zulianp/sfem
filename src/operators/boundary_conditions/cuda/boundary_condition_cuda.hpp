#ifndef BOUNDARY_CONDITION_INCORE_CUDA_H
#define BOUNDARY_CONDITION_INCORE_CUDA_H

#include <stddef.h>

#include "boundary_condition.hpp"
#include "sfem_base.hpp"

namespace sfem {
    /// The condition with its indices and values copied to the device (d_destroy_conditions frees them).
    boundary_condition_t to_device(const boundary_condition_t &host);

    /// The device bsr_constraint_nodes_to_identity_vec (dirichlet.hpp): the rows of `component` at
    /// the nodes, of BSR values of type V, diag_value on the diagonal and zero elsewhere.
    template <typename V>
    int d_bsr_constraint_nodes_to_identity_vec(const ptrdiff_t                    n_dirichlet_nodes,
                                               const idx_t *const SFEM_RESTRICT   dirichlet_nodes,
                                               const int                          block_size,
                                               const int                          component,
                                               const real_t                       diag_value,
                                               const count_t *const SFEM_RESTRICT rowptr,
                                               const idx_t *const SFEM_RESTRICT   colidx,
                                               V *const SFEM_RESTRICT             values,
                                               void *const                        stream);

    /// out[s] += sum over the nodes of (x_i + steps[s] h_i - v_i)^2, v_i the node's entry of
    /// `values` or, when that is null, `value`; h null: x alone.  x, h, values and the nodes on the
    /// device, steps and out on the host (the host constraint_objective_nodes_to_value(s)_vec_steps).
    int d_constraint_objective_vec_steps(const ptrdiff_t                   n_dirichlet_nodes,
                                         const idx_t *const SFEM_RESTRICT  dirichlet_nodes,
                                         const int                         block_size,
                                         const int                         component,
                                         const real_t                      value,
                                         const real_t *const SFEM_RESTRICT values,
                                         const real_t *const SFEM_RESTRICT x,
                                         const real_t *const SFEM_RESTRICT h,
                                         const int                         nsteps,
                                         const real_t *const               steps,
                                         real_t *const                     out,
                                         void *const                       stream);
}  // namespace sfem

#ifdef __cplusplus
extern "C" {
#endif

void d_constraint_nodes_copy_vec(const ptrdiff_t n_dirichlet_nodes,
                                 const idx_t    *dirichlet_nodes,
                                 const int       block_size,
                                 const int       component,
                                 const real_t   *source,
                                 real_t         *dest,
                                 void *const     stream);

void d_copy_at_dirichlet_nodes_vec(const int                         n_conditions,
                                   const boundary_condition_t *const cond,
                                   const int                         block_size,
                                   const real_t *const               in,
                                   real_t *const                     out,
                                   void *const                       stream);

void d_constraint_nodes_to_value_vec(const ptrdiff_t n_dirichlet_nodes,
                                     const idx_t    *dirichlet_nodes,
                                     const int       block_size,
                                     const int       component,
                                     const real_t    value,
                                     real_t         *values,
                                     void *const     stream);

void d_constraint_nodes_to_values_vec(const ptrdiff_t n_dirichlet_nodes,
                                      const idx_t    *dirichlet_nodes,
                                      const int       block_size,
                                      const int       component,
                                      const real_t   *dirichlet_values,
                                      real_t         *values,
                                      void *const     stream);

void d_apply_dirichlet_condition_vec(const int                         n_conditions,
                                     const boundary_condition_t *const cond,
                                     const int                         block_size,
                                     real_t *const                     x,
                                     void *const                       stream);

void d_constraint_gradient_nodes_to_value_vec(const ptrdiff_t                   n_dirichlet_nodes,
                                              const idx_t                      *dirichlet_nodes,
                                              const int                         block_size,
                                              const int                         component,
                                              const real_t                      value,
                                              const real_t *const SFEM_RESTRICT x,
                                              real_t *const SFEM_RESTRICT       g,
                                              void *const                       stream);

void d_constraint_gradient_nodes_to_values_vec(const ptrdiff_t                   n_dirichlet_nodes,
                                               const idx_t                      *dirichlet_nodes,
                                               const int                         block_size,
                                               const int                         component,
                                               const real_t *const SFEM_RESTRICT dirichlet_values,
                                               const real_t *const SFEM_RESTRICT x,
                                               real_t *const SFEM_RESTRICT       g,
                                               void *const                       stream);
void cu_crs_constraint_nodes_to_identity_vec(const ptrdiff_t                    n_dirichlet_nodes,
                                             const idx_t *const SFEM_RESTRICT   dirichlet_nodes,
                                             const int                          block_size,
                                             const int                          component,
                                             const real_t                       diag_value,
                                             const count_t *const SFEM_RESTRICT rowptr,
                                             const idx_t *const SFEM_RESTRICT   colidx,
                                             real_t *const SFEM_RESTRICT        values,
                                             void *const                        stream);

void d_destroy_conditions(const int n_conditions, boundary_condition_t *cond);

#ifdef __cplusplus
}
#endif
#endif  // BOUNDARY_CONDITION_INCORE_CUDA_H
