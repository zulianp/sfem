#ifndef CU_HEX8_LINEAR_ELASTICITY_H
#define CU_HEX8_LINEAR_ELASTICITY_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_affine_hex8_linear_elasticity_apply(const ptrdiff_t                  nelements,
                                           const ptrdiff_t                  stride,
                                           const idx_t *const SFEM_RESTRICT elements,
                                           const void *const SFEM_RESTRICT  jacobian_adjugate,
                                           const void *const SFEM_RESTRICT  jacobian_determinant,
                                           const real_t                     mu,
                                           const real_t                     lambda,
                                           const enum RealType              real_type,
                                           const ptrdiff_t                  u_stride,
                                           const void *const SFEM_RESTRICT  ux,
                                           const void *const SFEM_RESTRICT  uy,
                                           const void *const SFEM_RESTRICT  uz,
                                           const ptrdiff_t                  out_stride,
                                           void *const SFEM_RESTRICT        outx,
                                           void *const SFEM_RESTRICT        outy,
                                           void *const SFEM_RESTRICT        outz,
                                           void                            *stream);

// Block sparse row (BSR) https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-storage-formats
int cu_affine_hex8_linear_elasticity_bsr(const ptrdiff_t                    nelements,
                                         const ptrdiff_t                    stride,
                                         const idx_t *const SFEM_RESTRICT   elements,
                                         const void *const SFEM_RESTRICT    jacobian_adjugate,
                                         const void *const SFEM_RESTRICT    jacobian_determinant,
                                         const real_t                       mu,
                                         const real_t                       lambda,
                                         const enum RealType                real_type,
                                         const count_t *const SFEM_RESTRICT rowptr,
                                         const idx_t *const SFEM_RESTRICT   colidx,
                                         void *const SFEM_RESTRICT          values,
                                         void                              *stream);

int cu_affine_hex8_linear_elasticity_block_diag_sym(const ptrdiff_t                 nelements,
                                                    const ptrdiff_t                 stride,
                                                    idx_t *const SFEM_RESTRICT      elements,
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
#endif  // CU_HEX8_LINEAR_ELASTICITY_H
