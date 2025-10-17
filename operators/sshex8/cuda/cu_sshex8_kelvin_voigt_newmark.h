#ifndef CU_SSHEX8_KELVIN_VOIGT_NEWMARK_H
#define CU_SSHEX8_KELVIN_VOIGT_NEWMARK_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_affine_sshex8_kelvin_voigt_newmark_apply(const int                       level,
                                             const ptrdiff_t                 nelements,
                                             idx_t **const SFEM_RESTRICT     elements,
                                             const ptrdiff_t                 jacobian_stride,
                                             const void *const SFEM_RESTRICT jacobian_adjugate,
                                             const void *const SFEM_RESTRICT jacobian_determinant,
                                             const real_t                    k,
                                             const real_t                    K,
                                             const real_t                    eta,
                                             const real_t                    rho,
                                             const real_t                    dt,
                                             const real_t                    gamma,
                                             const real_t                    beta,
                                             const enum RealType             real_type,
                                             const ptrdiff_t                 u_stride,
                                             const void *const SFEM_RESTRICT ux,
                                             const void *const SFEM_RESTRICT uy,
                                             const void *const SFEM_RESTRICT uz,
                                             const void *const SFEM_RESTRICT vx,
                                             const void *const SFEM_RESTRICT vy,
                                             const void *const SFEM_RESTRICT vz,
                                             const void *const SFEM_RESTRICT ax,
                                             const void *const SFEM_RESTRICT ay,
                                             const void *const SFEM_RESTRICT az,
                                             const ptrdiff_t                 out_stride,
                                             void *const SFEM_RESTRICT       outx,
                                             void *const SFEM_RESTRICT       outy,
                                             void *const SFEM_RESTRICT       outz,
                                             void                           *stream);

int cu_affine_sshex8_kelvin_voigt_newmark_diag(const int                       level,
                                            const ptrdiff_t                 nelements,
                                            idx_t **const SFEM_RESTRICT     elements,
                                            const ptrdiff_t                 jacobian_stride,
                                            const void *const SFEM_RESTRICT jacobian_adjugate,
                                            const void *const SFEM_RESTRICT jacobian_determinant,
                                            const real_t                    k,
                                            const real_t                    K,
                                            const real_t                    eta,
                                            const real_t                    rho,
                                            const real_t                    dt,
                                            const real_t                    gamma,
                                            const real_t                    beta,
                                            const enum RealType             real_type,
                                            const ptrdiff_t                 out_stride,
                                            void *const SFEM_RESTRICT       outx,
                                            void *const SFEM_RESTRICT       outy,
                                            void *const SFEM_RESTRICT       outz,
                                            void                           *stream);


#ifdef __cplusplus
}
#endif
#endif  // CU_SSHEX8_KELVIN_VOIGT_NEWMARK_H
