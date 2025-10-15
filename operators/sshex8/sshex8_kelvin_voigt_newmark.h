#ifndef SFEM_SSHEX8_KELVIN_VOIGT_NEWMARK_H
#define SFEM_SSHEX8_KELVIN_VOIGT_NEWMARK_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif
// Semi-structured Kelvin-Voigt Newmark (affine) LHS apply
// Current exported symbol implemented in sshex8_kv.c (name kept for compatibility)
int affine_sshex8_kelvin_voigt_newmark_apply(const int                    level,
                                          const ptrdiff_t              nelements,
                                          const ptrdiff_t              nnodes,
                                          idx_t **const SFEM_RESTRICT  elements,
                                          geom_t **const SFEM_RESTRICT points,
                                          const real_t                 k,
                                          const real_t                 K,
                                          const real_t                 eta,
                                          const real_t                 rho,
                                          const real_t                 dt,
                                          const real_t                 gamma,
                                          const real_t                 beta,
                                          const ptrdiff_t              u_stride,
                                          const real_t *const          ux,
                                          const real_t *const          uy,
                                          const real_t *const          uz,
                                          const ptrdiff_t              out_stride,
                                          real_t *const                outx,
                                          real_t *const                outy,
                                          real_t *const                outz);

int affine_sshex8_kelvin_voigt_newmark_gradient(const int                    level,
                                          const ptrdiff_t              nelements,
                                          const ptrdiff_t              nnodes,
                                          idx_t **const SFEM_RESTRICT  elements,
                                          geom_t **const SFEM_RESTRICT points,
                                          const real_t                 k,
                                          const real_t                 K,
                                          const real_t                 eta,
                                          const real_t                 rho,
                                          const ptrdiff_t              u_stride,
                                          const real_t *const          ux,
                                          const real_t *const          uy,
                                          const real_t *const          uz,
                                          const real_t *const          vx,
                                          const real_t *const          vy,
                                          const real_t *const          vz,
                                          const real_t *const          ax,
                                          const real_t *const          ay,
                                          const real_t *const          az,
                                          const ptrdiff_t              out_stride,
                                          real_t *const                outx,
                                          real_t *const                outy,
                                          real_t *const                outz);



int affine_sshex8_kelvin_voigt_newmark_diag(const int                    level,
                                            const ptrdiff_t              nelements,
                                            const ptrdiff_t              nnodes,
                                            idx_t **const SFEM_RESTRICT  elements,
                                            geom_t **const SFEM_RESTRICT points,
                                            const real_t                 beta,
                                            const real_t                 gamma,
                                            const real_t                 dt,
                                            const real_t                 k,
                                            const real_t                 K,
                                            const real_t                 eta,
                                            const real_t                 rho,
                                            const ptrdiff_t              out_stride,
                                            real_t *const                outx,
                                            real_t *const                outy,
                                            real_t *const                outz);



#ifdef __cplusplus
}
#endif
#endif  // SFEM_SSHEX8_KELVIN_VOIGT_NEWMARK_H