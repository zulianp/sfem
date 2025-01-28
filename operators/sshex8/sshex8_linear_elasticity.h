#ifndef SFEM_SSHEX8_LINEAR_ELASTICITY_H
#define SFEM_SSHEX8_LINEAR_ELASTICITY_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int sshex8_linear_elasticity_apply(const int                    level,
                                   const ptrdiff_t              nelements,
                                   const ptrdiff_t              nnodes,
                                   idx_t **const SFEM_RESTRICT  elements,
                                   geom_t **const SFEM_RESTRICT points,
                                   const real_t                 mu,
                                   const real_t                 lambda,
                                   const ptrdiff_t              u_stride,
                                   const real_t *const          ux,
                                   const real_t *const          uy,
                                   const real_t *const          uz,
                                   const ptrdiff_t              out_stride,
                                   real_t *const                outx,
                                   real_t *const                outy,
                                   real_t *const                outz);

int affine_sshex8_linear_elasticity_apply(const int                    level,
                                          const ptrdiff_t              nelements,
                                          const ptrdiff_t              nnodes,
                                          idx_t **const SFEM_RESTRICT  elements,
                                          geom_t **const SFEM_RESTRICT points,
                                          const real_t                 mu,
                                          const real_t                 lambda,
                                          const ptrdiff_t              u_stride,
                                          const real_t *const          ux,
                                          const real_t *const          uy,
                                          const real_t *const          uz,
                                          const ptrdiff_t              out_stride,
                                          real_t *const                outx,
                                          real_t *const                outy,
                                          real_t *const                outz);

int affine_sshex8_linear_elasticity_diag(const int                    level,
                                         const ptrdiff_t              nelements,
                                         const ptrdiff_t              nnodes,
                                         idx_t **const SFEM_RESTRICT  elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const real_t                 mu,
                                         const real_t                 lambda,
                                         const ptrdiff_t              out_stride,
                                         real_t *const                outx,
                                         real_t *const                outy,
                                         real_t *const                outz);

int affine_sshex8_elasticity_bsr(const int                          level,
                                 const ptrdiff_t                    nelements,
                                 const ptrdiff_t                    nnodes,
                                 idx_t **const SFEM_RESTRICT        elements,
                                 geom_t **const SFEM_RESTRICT       points,
                                 const real_t                       mu,
                                 const real_t                       lambda,
                                 const count_t *const SFEM_RESTRICT rowptr,
                                 const idx_t *const SFEM_RESTRICT   colidx,
                                 real_t *const SFEM_RESTRICT        values);

int affine_sshex8_elasticity_crs_sym(const int                          level,
                                     const ptrdiff_t                    nelements,
                                     const ptrdiff_t                    nnodes,
                                     idx_t **const SFEM_RESTRICT        elements,
                                     geom_t **const SFEM_RESTRICT       points,
                                     const real_t                       mu,
                                     const real_t                       lambda,
                                     const count_t *const SFEM_RESTRICT rowptr,
                                     const idx_t *const SFEM_RESTRICT   colidx,
                                     // Output in SoA format (6)
                                     real_t **const SFEM_RESTRICT block_diag,
                                     real_t **const SFEM_RESTRICT block_offdiag);

int affine_sshex8_linear_elasticity_block_diag_sym(const int                    level,
                                                   const ptrdiff_t              nelements,
                                                   const ptrdiff_t              nnodes,
                                                   idx_t **const SFEM_RESTRICT  elements,
                                                   geom_t **const SFEM_RESTRICT points,
                                                   const real_t                 mu,
                                                   const real_t                 lambda,
                                                   const ptrdiff_t              out_stride,
                                                   real_t *const                out0,
                                                   real_t *const                out1,
                                                   real_t *const                out2,
                                                   real_t *const                out3,
                                                   real_t *const                out4,
                                                   real_t *const                out5);

// Optional mapping for selecting a subset of rows
//    const idx_t *const SFEM_RESTRICT mapping

#ifdef __cplusplus
}
#endif
#endif  // SFEM_SSHEX8_LINEAR_ELASTICITY_H
