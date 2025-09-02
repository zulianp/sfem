#ifndef __SFEM_ADJOINT_MINI_TET_GPU_WRAPPER_H__
#define __SFEM_ADJOINT_MINI_TET_GPU_WRAPPER_H__

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "bit_array.h"
#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_mesh.h"
#include "sfem_resample_field_adjoint_hyteg.h"

#ifdef __cplusplus
extern "C" {
#endif

void                                                                                    //
call_sfem_adjoint_mini_tet_kernel_gpu(const ptrdiff_t             start_element,        // Mesh
                                      const ptrdiff_t             end_element,          //
                                      const ptrdiff_t             nnodes,               //
                                      const idx_t** const         elems,                //
                                      const geom_t** const        xyz,                  //
                                      const ptrdiff_t             n0,                   // SDF
                                      const ptrdiff_t             n1,                   //
                                      const ptrdiff_t             n2,                   //
                                      const ptrdiff_t             stride0,              // Stride
                                      const ptrdiff_t             stride1,              //
                                      const ptrdiff_t             stride2,              //
                                      const geom_t                origin0,              // Origin
                                      const geom_t                origin1,              //
                                      const geom_t                origin2,              //
                                      const geom_t                dx,                   // Delta
                                      const geom_t                dy,                   //
                                      const geom_t                dz,                   //
                                      const real_t* const         weighted_field,       // Input weighted field
                                      const mini_tet_parameters_t mini_tet_parameters,  // Threshold for alpha
                                      real_t* const               data);                              //

int                                                                                                      //
tet4_resample_field_local_refine_adjoint_hyteg_gpu(const ptrdiff_t                      start_element,   // Mesh
                                                   const ptrdiff_t                      end_element,     //
                                                   const ptrdiff_t                      nnodes,          //
                                                   const idx_t** const SFEM_RESTRICT    elems,           //
                                                   const geom_t** const SFEM_RESTRICT   xyz,             //
                                                   const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                                                   const ptrdiff_t* const SFEM_RESTRICT stride,          //
                                                   const geom_t* const SFEM_RESTRICT    origin,          //
                                                   const geom_t* const SFEM_RESTRICT    delta,           //
                                                   const real_t* const SFEM_RESTRICT    weighted_field,  // Input weighted field
                                                   const real_t                         alpha_th,        // Threshold for alpha
                                                   real_t* const SFEM_RESTRICT          data);

#ifdef __cplusplus
}
#endif

#endif  // __SFEM_ADJOINT_MINI_TET_GPU_WRAPPER_H__