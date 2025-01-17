#ifndef HEX8_LINEAR_ELASTICITY_H
#define HEX8_LINEAR_ELASTICITY_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int hex8_linear_elasticity_apply(const ptrdiff_t              nelements,
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

int affine_hex8_linear_elasticity_apply(const ptrdiff_t              nelements,
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

int affine_hex8_linear_elasticity_bsr(const ptrdiff_t                    nelements,
                                      const ptrdiff_t                    nnodes,
                                      idx_t **const SFEM_RESTRICT        elements,
                                      geom_t **const SFEM_RESTRICT       points,
                                      const real_t                       mu,
                                      const real_t                       lambda,
                                      const count_t *const SFEM_RESTRICT rowptr,
                                      const idx_t *const SFEM_RESTRICT   colidx,
                                      real_t *const SFEM_RESTRICT        values);
/**
 * @brief Assembles the symmetric linear elasticity matrix for affine hexahedral elements in CRS
 * format. The matrix is divided into diagonal and upper-triangular off-diagonal blocks.
 *
 * @param nelements Number of elements
 * @param nnodes Number of nodes
 * @param elements Element connectivity array (8 nodes per element)
 * @param points Nodal coordinates array
 * @param mu First Lamé parameter (shear modulus)
 * @param lambda Second Lamé parameter
 * @param rowptr Row pointer array for CRS format
 * @param colidx Column indices array for CRS format
 * @param block_stride Stride between blocks in the output arrays
 * @param block_diag Array of diagonal blocks of size 6
 * @param block_offdiag Array of off-diagonal blocks of size 6
 * @return int Error code (0 on success)
 */
int affine_hex8_linear_elasticity_crs_sym(const ptrdiff_t                    nelements,
                                          const ptrdiff_t                    nnodes,
                                          idx_t **const SFEM_RESTRICT        elements,
                                          geom_t **const SFEM_RESTRICT       points,
                                          const real_t                       mu,
                                          const real_t                       lambda,
                                          const count_t *const SFEM_RESTRICT rowptr,
                                          const idx_t *const SFEM_RESTRICT   colidx,
                                          const ptrdiff_t                    block_stride,
                                          real_t **const SFEM_RESTRICT       block_diag,
                                          real_t **const SFEM_RESTRICT       block_offdiag);

int affine_hex8_linear_elasticity_diag(const ptrdiff_t              nelements,
                                       const ptrdiff_t              nnodes,
                                       idx_t **const SFEM_RESTRICT  elements,
                                       geom_t **const SFEM_RESTRICT points,
                                       const real_t                 mu,
                                       const real_t                 lambda,
                                       const ptrdiff_t              out_stride,
                                       real_t *const                outx,
                                       real_t *const                outy,
                                       real_t *const                outz);

int affine_hex8_linear_elasticity_block_diag_sym(const ptrdiff_t              nelements,
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

#ifdef __cplusplus
}
#endif
#endif  // HEX8_LINEAR_ELASTICITY_H
