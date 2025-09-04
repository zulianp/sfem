#ifndef KELVIN_VOIGT_NEWMARK
#define KELVIN_VOIGT_NEWMARK

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// HAOYU

int kelvin_voigt_newmark_apply_adjugate_soa(const enum ElemType                   element_type,
                                         const ptrdiff_t                       nelements,
                                         const ptrdiff_t                       nnodes,
                                         idx_t **const SFEM_RESTRICT           elements,
                                         const jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                         const jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                         const real_t                          dt,
                                         const real_t                          gamma,
                                         const real_t                          beta, 
                                         const real_t                          k,
                                         const real_t                          K,
                                         const real_t                          eta,
                                         const real_t                          rho,
                                         const real_t *const SFEM_RESTRICT     u,
                                         real_t *const SFEM_RESTRICT           values);


int kelvin_voigt_newmark_gradient_soa(const enum ElemType                   element_type,
                                         const ptrdiff_t                       nelements,
                                         const ptrdiff_t                       nnodes,
                                         idx_t **const SFEM_RESTRICT           elements,
                                         const jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                         const jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                         const real_t                          k,
                                         const real_t                          K,
                                         const real_t                          eta,
                                         const real_t                          rho,
                                         const real_t *const SFEM_RESTRICT     u,
                                         const real_t *const SFEM_RESTRICT     vx,
                                         const real_t *const SFEM_RESTRICT     vy,
                                         const real_t *const SFEM_RESTRICT     vz,
                                         const real_t *const SFEM_RESTRICT     ax,
                                         const real_t *const SFEM_RESTRICT     ay,
                                         const real_t *const SFEM_RESTRICT     az,
                                         real_t *const SFEM_RESTRICT           values);



///////////////////////////////////////////////// AOS /////////////////////////////////////////////////


int kelvin_voigt_newmark_apply_adjugate_aos(const enum ElemType                   element_type,
                                         const ptrdiff_t                       nelements,
                                         const ptrdiff_t                       nnodes,
                                         idx_t **const SFEM_RESTRICT           elements,
                                         const jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                         const jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                         const real_t                          dt,
                                         const real_t                          gamma,
                                         const real_t                          beta, 
                                         const real_t                          k,
                                         const real_t                          K,
                                         const real_t                          eta,
                                         const real_t                          rho,
                                         const real_t *const SFEM_RESTRICT     u,
                                         real_t *const SFEM_RESTRICT           values);





int kelvin_voigt_newmark_gradient_aos(const enum ElemType                   element_type,
                                         const ptrdiff_t                       nelements,
                                         const ptrdiff_t                       nnodes,
                                         idx_t **const SFEM_RESTRICT           elements, 
                                         const jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                                         const jacobian_t *const SFEM_RESTRICT jacobian_determinant,
                                         const real_t                          k,
                                         const real_t                          K,
                                         const real_t                          eta,
                                         const real_t                          rho,
                                         const real_t *const SFEM_RESTRICT     u,
                                         const real_t *const SFEM_RESTRICT     vx,
                                         const real_t *const SFEM_RESTRICT     vy,
                                         const real_t *const SFEM_RESTRICT     vz,
                                         const real_t *const SFEM_RESTRICT     ax,
                                         const real_t *const SFEM_RESTRICT     ay,
                                         const real_t *const SFEM_RESTRICT     az,
                                         real_t *const SFEM_RESTRICT           values);



#ifdef __cplusplus
}
#endif

#endif  // KELVIN_VOIGT_NEWMARK
