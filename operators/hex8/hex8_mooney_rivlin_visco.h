#ifndef HEX8_MOONEY_RIVLIN_VISCO_H
#define HEX8_MOONEY_RIVLIN_VISCO_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int hex8_mooney_rivlin_visco_gradient(const ptrdiff_t                   nelements,
                                    const ptrdiff_t                   stride,
                                    const ptrdiff_t                   nnodes,
                                    idx_t **const SFEM_RESTRICT       elements,
                                    geom_t **const SFEM_RESTRICT      points,
                                      const real_t                      C10,
                                      const real_t                      K,
                                      const real_t                      C01,
                                      const real_t                      dt,
                                      const int                         num_prony_terms,
                                      const real_t *const SFEM_RESTRICT g,
                                      const real_t *const SFEM_RESTRICT tau,
                                      const ptrdiff_t                   history_stride,
                                      const real_t *const SFEM_RESTRICT history,
                                   const ptrdiff_t                   u_stride,
                                   const real_t *const SFEM_RESTRICT ux,
                                   const real_t *const SFEM_RESTRICT uy,
                                   const real_t *const SFEM_RESTRICT uz,
                                   const ptrdiff_t                   out_stride,
                                   real_t *const SFEM_RESTRICT       outx,
                                   real_t *const SFEM_RESTRICT       outy,
                                   real_t *const SFEM_RESTRICT       outz);

int hex8_mooney_rivlin_visco_update_history(const ptrdiff_t                   nelements,
                                            const ptrdiff_t                   stride,
                                            const ptrdiff_t                   nnodes,
                                            idx_t **const SFEM_RESTRICT       elements,
                                            geom_t **const SFEM_RESTRICT      points,
                                            const real_t                      C10,
                                            const real_t                      K,
                                            const real_t                      C01,
                                            const real_t                      dt,
                                            const int                         num_prony_terms,
                                            const real_t *const SFEM_RESTRICT g,
                                            const real_t *const SFEM_RESTRICT tau,
                                            const ptrdiff_t                   history_stride,
                                            const real_t *const SFEM_RESTRICT history,
                                            real_t *const SFEM_RESTRICT       new_history,
                                            const ptrdiff_t                   u_stride,
                                            const real_t *const SFEM_RESTRICT ux,
                                            const real_t *const SFEM_RESTRICT uy,
                                            const real_t *const SFEM_RESTRICT uz);

int hex8_mooney_rivlin_visco_bsr(const ptrdiff_t                   nelements,
                                                const ptrdiff_t                   stride,
                                     const ptrdiff_t                   nnodes,
                                                idx_t **const SFEM_RESTRICT       elements,
                                                geom_t **const SFEM_RESTRICT      points,
                                     const real_t                      C10,
                                     const real_t                      K,
                                     const real_t                      C01,
                                     const real_t                      dt,
                                     const int                         num_prony_terms,
                                     const real_t *const SFEM_RESTRICT g,
                                     const real_t *const SFEM_RESTRICT tau,
                                     const ptrdiff_t                   history_stride,
                                     const real_t *const SFEM_RESTRICT history,
                                                const ptrdiff_t                   u_stride,
                                                const real_t *const SFEM_RESTRICT ux,
                                                const real_t *const SFEM_RESTRICT uy,
                                                const real_t *const SFEM_RESTRICT uz,
                                                const ptrdiff_t                   out_stride,
                                     real_t *const SFEM_RESTRICT       values,
                                     const ptrdiff_t                   row_stride,
                                     const idx_t *const SFEM_RESTRICT  rowptr,
                                     const idx_t *const SFEM_RESTRICT  colidx);

int hex8_mooney_rivlin_visco_hessian_diag(const ptrdiff_t                   nelements,
                                          const ptrdiff_t                   stride,
                                          const ptrdiff_t                   nnodes,
                                          idx_t **const SFEM_RESTRICT       elements,
                                          geom_t **const SFEM_RESTRICT      points,
                                          const real_t                      C10,
                                          const real_t                      K,
                                          const real_t                      C01,
                                          const real_t                      dt,
                                          const int                         num_prony_terms,
                                          const real_t *const SFEM_RESTRICT g,
                                          const real_t *const SFEM_RESTRICT tau,
                                          const ptrdiff_t                   history_stride,
                                          const real_t *const SFEM_RESTRICT history,
                                          const ptrdiff_t                   u_stride,
                                          const real_t *const SFEM_RESTRICT ux,
                                          const real_t *const SFEM_RESTRICT uy,
                                          const real_t *const SFEM_RESTRICT uz,
                                          const ptrdiff_t                   out_stride,
                                          real_t *const SFEM_RESTRICT       outx,
                                          real_t *const SFEM_RESTRICT       outy,
                                          real_t *const SFEM_RESTRICT       outz);

// Flexible version: supports arbitrary number of Prony terms at runtime
int hex8_mooney_rivlin_visco_bsr_flexible(const ptrdiff_t                   nelements,
                                          const ptrdiff_t                   stride,
                                          const ptrdiff_t                   nnodes,
                                          idx_t **const SFEM_RESTRICT       elements,
                                          geom_t **const SFEM_RESTRICT      points,
                                          const real_t                      C10,
                                          const real_t                      K,
                                          const real_t                      C01,
                                          const real_t                      dt,
                                          const int                         num_prony_terms,
                                          const real_t *const SFEM_RESTRICT g,
                                          const real_t *const SFEM_RESTRICT tau,
                                          const ptrdiff_t                   history_stride,
                                          const real_t *const SFEM_RESTRICT history,
                                          const ptrdiff_t                   u_stride,
                                          const real_t *const SFEM_RESTRICT ux,
                                          const real_t *const SFEM_RESTRICT uy,
                                          const real_t *const SFEM_RESTRICT uz,
                                          const ptrdiff_t                   out_stride,
                                          real_t *const SFEM_RESTRICT       values,
                                          const ptrdiff_t                   row_stride,
                                          const idx_t *const SFEM_RESTRICT  rowptr,
                                          const idx_t *const SFEM_RESTRICT  colidx);

#ifdef __cplusplus
}
#endif

#endif  // HEX8_MOONEY_RIVLIN_VISCO_H
