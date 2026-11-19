#ifndef SFEM_OBSTACLE_H
#define SFEM_OBSTACLE_H

#include <stddef.h>
#include "sfem_base.hpp"
#include "sfem_defs.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/// obstacle_normal_project on the rows [begin, end) (out += the normal component of h at the
/// rows' nodes): a thread's share, inside or outside a region.
int obstacle_normal_project_range(const int                         dim,
                                  const ptrdiff_t                   begin,
                                  const ptrdiff_t                   end,
                                  const idx_t *const SFEM_RESTRICT  idx,
                                  real_t **const SFEM_RESTRICT      normals,
                                  const real_t *const SFEM_RESTRICT h,
                                  real_t *const SFEM_RESTRICT       out);

int obstacle_normal_project(const int                         dim,
                            const ptrdiff_t                   n,
                            const idx_t *const SFEM_RESTRICT  idx,
                            real_t **const SFEM_RESTRICT      normals,
                            const real_t *const SFEM_RESTRICT h,
                            real_t *const SFEM_RESTRICT       out);

/// obstacle_distribute_contact_forces from inside a parallel region, by every thread of it
/// (orphaned worksharing): the rows split among them, ending on the loop's barrier.
int obstacle_distribute_contact_forces_team(const int                         dim,
                                            const ptrdiff_t                   n,
                                            const idx_t *const SFEM_RESTRICT  idx,
                                            real_t **const SFEM_RESTRICT      normals,
                                            const real_t *const SFEM_RESTRICT m,
                                            const real_t *const               f,
                                            real_t *const                     out);

int obstacle_distribute_contact_forces(const int                         dim,
                                       const ptrdiff_t                   n,
                                       const idx_t *const SFEM_RESTRICT  idx,
                                       real_t **const SFEM_RESTRICT      normals,
                                       const real_t *const SFEM_RESTRICT m,
                                       const real_t *const               f,
                                       real_t *const                     out);

int obstacle_hessian_block_diag_sym(const int                         dim,
                                    const ptrdiff_t                   n,
                                    const idx_t *const SFEM_RESTRICT  idx,
                                    real_t **const SFEM_RESTRICT      normals,
                                    const real_t *const SFEM_RESTRICT m,
                                    const real_t *const               x,
                                    real_t *const                     values);

int obstacle_contact_stress(const int                         dim,
                            const ptrdiff_t                   n,
                            const idx_t *const SFEM_RESTRICT  idx,
                            real_t **const SFEM_RESTRICT      normals,
                            const real_t *const SFEM_RESTRICT m,
                            const real_t *const               r,
                            real_t *const                     s);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_OBSTACLE_H
