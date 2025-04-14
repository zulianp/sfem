#ifndef CU_QUADSHELL4_INTEGRATE_VALUES_H
#define CU_QUADSHELL4_INTEGRATE_VALUES_H

#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_quadshell4_integrate_value(const ptrdiff_t                    nelements,
                                  const ptrdiff_t                    stride,  // Stride for elements and coords
                                  const idx_t *const SFEM_RESTRICT   elements,
                                  const geom_t **const SFEM_RESTRICT coords,
                                  const real_t                       value,
                                  const int                          block_size,
                                  const int                          component,
                                  const enum RealType                real_type,
                                  void *const SFEM_RESTRICT          out,
                                  void                              *stream);

int cu_quadshell4_integrate_values(const ptrdiff_t                    nelements,
                                   const ptrdiff_t                    stride,  // Stride for elements and coords
                                   const idx_t *const SFEM_RESTRICT   elements,
                                   const geom_t **const SFEM_RESTRICT coords,
                                   const enum RealType                real_type,
                                   void *const SFEM_RESTRICT          values,
                                   const int                          block_size,
                                   const int                          component,
                                   void *const SFEM_RESTRICT          out,
                                   void                              *stream);

#ifdef __cplusplus
}
#endif

#endif  // CU_QUADSHELL4_INTEGRATE_VALUES_H