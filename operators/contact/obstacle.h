#ifndef SFEM_OBSTACLE_H
#define SFEM_OBSTACLE_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int obstacle_normal_project(const int                         dim,
                            const ptrdiff_t                   n,
                            const idx_t *const SFEM_RESTRICT  idx,
                            real_t **const SFEM_RESTRICT      normals,
                            const real_t *const SFEM_RESTRICT h,
                            real_t *const SFEM_RESTRICT       out);

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
