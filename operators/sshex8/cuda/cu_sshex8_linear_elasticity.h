#ifndef CU_SSHEX8_LINEAR_ELASTICITY_H
#define CU_SSHEX8_LINEAR_ELASTICITY_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_affine_sshex8_linear_elasticity_apply(const int                       level,
                                             const ptrdiff_t                 nelements,
                                             idx_t **const SFEM_RESTRICT     elements,
                                             const ptrdiff_t                 jacobian_stride,
                                             const void *const SFEM_RESTRICT jacobian_adjugate,
                                             const void *const SFEM_RESTRICT jacobian_determinant,
                                             const real_t                    mu,
                                             const real_t                    lambda,
                                             const enum RealType             real_type,
                                             const ptrdiff_t                 u_stride,
                                             const void *const SFEM_RESTRICT ux,
                                             const void *const SFEM_RESTRICT uy,
                                             const void *const SFEM_RESTRICT uz,
                                             const ptrdiff_t                 out_stride,
                                             void *const SFEM_RESTRICT       outx,
                                             void *const SFEM_RESTRICT       outy,
                                             void *const SFEM_RESTRICT       outz,
                                             void                           *stream);

int cu_affine_sshex8_linear_elasticity_diag(const int                       level,
                                            const ptrdiff_t                 nelements,
                                            idx_t **const SFEM_RESTRICT     elements,
                                            const ptrdiff_t                 jacobian_stride,
                                            const void *const SFEM_RESTRICT jacobian_adjugate,
                                            const void *const SFEM_RESTRICT jacobian_determinant,
                                            const real_t                    mu,
                                            const real_t                    lambda,
                                            const enum RealType             real_type,
                                            const ptrdiff_t                 out_stride,
                                            void *const SFEM_RESTRICT       outx,
                                            void *const SFEM_RESTRICT       outy,
                                            void *const SFEM_RESTRICT       outz,
                                            void                           *stream);

int cu_affine_sshex8_linear_elasticity_block_diag_sym(const int                       level,
                                                      const ptrdiff_t                 nelements,
                                                      idx_t **const SFEM_RESTRICT     elements,
                                                      const ptrdiff_t                 jacobian_stride,
                                                      const void *const SFEM_RESTRICT jacobian_adjugate,
                                                      const void *const SFEM_RESTRICT jacobian_determinant,
                                                      const real_t                    mu,
                                                      const real_t                    lambda,
                                                      const ptrdiff_t                 out_stride,
                                                      const enum RealType             real_type,
                                                      void *const                     out0,
                                                      void *const                     out1,
                                                      void *const                     out2,
                                                      void *const                     out3,
                                                      void *const                     out4,
                                                      void *const                     out5,
                                                      void                           *stream);

#ifdef __cplusplus
}
#endif
#endif  // CU_SSHEX8_LINEAR_ELASTICITY_H
