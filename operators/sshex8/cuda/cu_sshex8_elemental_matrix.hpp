#ifndef CU_SSHEX8_ELEMENTAL_MATRIX_H
#define CU_SSHEX8_ELEMENTAL_MATRIX_H

#include <stddef.h>
#include "sfem_base.hpp"
#include "sfem_defs.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int cu_affine_hex8_elemental_matrix_apply(const ptrdiff_t                 nelements,
                                          idx_t **const SFEM_RESTRICT     elements,
                                          const enum smesh::PrimitiveType             real_type,
                                          void **const SFEM_RESTRICT      elemental_matrix,
                                          const void *const SFEM_RESTRICT x,
                                          void *const SFEM_RESTRICT       y,
                                          void                           *stream);

int cu_affine_sshex8_elemental_matrix_apply(const int                       level,
                                            const ptrdiff_t                 nelements,
                                            idx_t **const SFEM_RESTRICT     elements,
                                            const enum smesh::PrimitiveType             real_type,
                                            void **const SFEM_RESTRICT      elemental_matrix,
                                            const void *const SFEM_RESTRICT x,
                                            void *const SFEM_RESTRICT       y,
                                            void                           *stream);

int cu_affine_sshex8_elemental_matrix_apply_AoS(const int                        level,
                                                const ptrdiff_t                  nelements,
                                                const idx_t *const SFEM_RESTRICT elements,
                                                const enum smesh::PrimitiveType              real_type,
                                                const void *const SFEM_RESTRICT  elemental_matrix,
                                                const void *const SFEM_RESTRICT  x,
                                                void *const SFEM_RESTRICT        y,
                                                void                            *stream);

/** Vector (block_size=3) elemental-matrix apply for affine SSHEX8.
 *  elemental_matrix is AoS with 24x24 entries per macro-element (cartesian HEX8 node order
 *  within each component block). Vectors x/y are AoS with stride 3. */
int cu_affine_sshex8_elemental_matrix_apply_AoS_vector(const int                        level,
                                                       const ptrdiff_t                  nelements,
                                                       const idx_t *const SFEM_RESTRICT elements,
                                                       const enum smesh::PrimitiveType  real_type,
                                                       const void *const SFEM_RESTRICT  elemental_matrix,
                                                       const void *const SFEM_RESTRICT  x,
                                                       void *const SFEM_RESTRICT        y,
                                                       void                            *stream);

#ifdef __cplusplus
}
#endif
#endif  // CU_SSHEX8_ELEMENTAL_MATRIX_H
