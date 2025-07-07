#ifndef CU_SSHEX8_ELEMENTAL_MATRIX_H
#define CU_SSHEX8_ELEMENTAL_MATRIX_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_affine_hex8_elemental_matrix_apply(const ptrdiff_t                 nelements,
                                          idx_t **const SFEM_RESTRICT     elements,
                                          const enum RealType             real_type,
                                          void **const SFEM_RESTRICT      elemental_matrix,
                                          const void *const SFEM_RESTRICT x,
                                          void *const SFEM_RESTRICT       y,
                                          void                           *stream);

int cu_affine_sshex8_elemental_matrix_apply(const int                       level,
                                            const ptrdiff_t                 nelements,
                                            idx_t **const SFEM_RESTRICT     elements,
                                            const enum RealType             real_type,
                                            void **const SFEM_RESTRICT      elemental_matrix,
                                            const void *const SFEM_RESTRICT x,
                                            void *const SFEM_RESTRICT       y,
                                            void                           *stream);

int cu_affine_sshex8_elemental_matrix_apply_AoS(const int                        level,
                                                const ptrdiff_t                  nelements,
                                                const idx_t *const SFEM_RESTRICT elements,
                                                const enum RealType              real_type,
                                                const void *const SFEM_RESTRICT  elemental_matrix,
                                                const void *const SFEM_RESTRICT  x,
                                                void *const SFEM_RESTRICT        y,
                                                void                            *stream);

#ifdef __cplusplus
}
#endif
#endif  // CU_SSHEX8_ELEMENTAL_MATRIX_H
