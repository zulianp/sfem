#ifndef HEX8_KELVIN_VOIGT_NEWMARK_H
#define HEX8_KELVIN_VOIGT_NEWMARK_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// int hex8_kelvin_voigt_newmark_lhs_apply(const ptrdiff_t              nelements,
//                                     const ptrdiff_t              nnodes,
//                                     idx_t **const SFEM_RESTRICT  elements,
//                                     geom_t **const SFEM_RESTRICT points,
//                                     const ptrdiff_t              in_stride,
//                                     // unified interface for both SoA and AoS
//                                     const real_t *const SFEM_RESTRICT ux,
//                                     const real_t *const SFEM_RESTRICT uy,
//                                     const real_t *const SFEM_RESTRICT uz,
//                                     const ptrdiff_t                   out_stride,
//                                     real_t *const SFEM_RESTRICT       outx,
//                                     real_t *const SFEM_RESTRICT       outy,
//                                     real_t *const SFEM_RESTRICT       outz);

// //  F(x, x', x'') = 0
// int hex8_kelvin_voigt_newmark_gradient(const ptrdiff_t              nelements,
//                                        const ptrdiff_t              nnodes,
//                                        idx_t **const SFEM_RESTRICT  elements,
//                                        geom_t **const SFEM_RESTRICT points,
//                                        // unified interface for both SoA and AoS
//                                        const ptrdiff_t in_stride,
//                                        // Displacement
//                                        const real_t *const SFEM_RESTRICT u_oldx,
//                                        const real_t *const SFEM_RESTRICT u_oldy,
//                                        const real_t *const SFEM_RESTRICT u_oldz,
//                                        // Velocity
//                                        const real_t *const SFEM_RESTRICT v_oldx,
//                                        const real_t *const SFEM_RESTRICT v_oldy,
//                                        const real_t *const SFEM_RESTRICT v_oldz,
//                                        // Accleration
//                                        const real_t *const SFEM_RESTRICT a_oldx,
//                                        const real_t *const SFEM_RESTRICT a_oldy,
//                                        const real_t *const SFEM_RESTRICT a_oldz,
//                                        // Current input
//                                        const real_t *const SFEM_RESTRICT ux,
//                                        const real_t *const SFEM_RESTRICT uy,
//                                        const real_t *const SFEM_RESTRICT uz,
//                                        // Output
//                                        const ptrdiff_t             out_stride,
//                                        real_t *const SFEM_RESTRICT outx,
//                                        real_t *const SFEM_RESTRICT outy,
//                                        real_t *const SFEM_RESTRICT outz);

int affine_hex8_kelvin_voigt_newmark_lhs_apply(const ptrdiff_t              nelements,
                                 const ptrdiff_t              nnodes,
                                 idx_t **const SFEM_RESTRICT  elements,

                                 const jacobian_t *const          g_jacobian_adjugate,
                                 const jacobian_t *const          g_jacobian_determinant,

                                 const real_t                 dt,
                                 const real_t                 gamma,
                                 const real_t                 beta, 

                                 const real_t                 k,
                                 const real_t                 K,
                                 const real_t                 eta, 

                                 const ptrdiff_t              u_stride,
                                 const real_t *const          ux,
                                 const real_t *const          uy,
                                 const real_t *const          uz,
                                 const ptrdiff_t              out_stride,
                                 real_t *const                outx,
                                 real_t *const                outy,
                                 real_t *const                outz);


void newmark_increment_update(
                                const real_t dt,         
                                const real_t beta,       
                                const real_t gamma,      
                                
                                const real_t *const ux,        
                                const real_t *const uy,     
                                const real_t *const uz,  

                                real_t *const vx,      
                                real_t *const vy,       
                                real_t *const vz,        
                                real_t *const ax,        
                                real_t *const ay,        
                                real_t *const az,        

                                const real_t *const u_oldx,
                                const real_t *const u_oldy,
                                const real_t *const u_oldz,
                                const real_t *const v_oldx,
                                const real_t *const v_oldy,
                                const real_t *const v_oldz,
                                const real_t *const a_oldx,
                                const real_t *const a_oldy,
                                const real_t *const a_oldz,

                                const ptrdiff_t nnodes,
                                const ptrdiff_t stride);



// int affine_hex8_kelvin_voigt_newmark_gradient(const ptrdiff_t              nelements,
//                                  const ptrdiff_t              nnodes,
//                                  idx_t **const SFEM_RESTRICT  elements,

//                                  const real_t *const          g_jacobian_adjugate,
//                                  const real_t *const          g_jacobian_determinant,

//                                  const real_t                 dt,
//                                  const real_t                 gamma,
//                                  const real_t                 beta, 

//                                  const real_t                 k,
//                                  const real_t                 K,
//                                  const real_t                 eta,
//                                  const real_t                 rho,

//                                  const ptrdiff_t              u_stride,

//                                  const real_t *const          u_oldx,
//                                  const real_t *const          u_oldy,
//                                  const real_t *const          u_oldz,
//                                  const real_t *const          v_oldx,
//                                  const real_t *const          v_oldy,
//                                  const real_t *const          v_oldz,
//                                  const real_t *const          a_oldx,
//                                  const real_t *const          a_oldy,
//                                  const real_t *const          a_oldz,

//                                  const real_t *const          ux,
//                                  const real_t *const          uy,
//                                  const real_t *const          uz,

//                                  const ptrdiff_t              out_stride,
//                                  real_t *const                outx,
//                                  real_t *const                outy,
//                                  real_t *const                outz); 

int affine_hex8_kelvin_voigt_newmark_gradient(const ptrdiff_t              nelements,
                                 const ptrdiff_t              nnodes,
                                 idx_t **const SFEM_RESTRICT  elements,

                                 const jacobian_t *const          g_jacobian_adjugate,
                                 const jacobian_t *const          g_jacobian_determinant,

                                 const real_t                 k,
                                 const real_t                 K,
                                 const real_t                 eta,

                                 const ptrdiff_t              u_stride,

                                 const real_t *const          ux,
                                 const real_t *const          uy,
                                 const real_t *const          uz,
                                 const real_t *const          vx,
                                 const real_t *const          vy,
                                 const real_t *const          vz,

                                 const ptrdiff_t              out_stride,
                                 real_t *const                outx,
                                 real_t *const                outy,
                                 real_t *const                outz);


#ifdef __cplusplus
}
#endif
#endif  // HEX8_KELVIN_VOIGT_NEWMARK_H
