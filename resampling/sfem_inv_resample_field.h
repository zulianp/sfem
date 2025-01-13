#ifndef __SFEM_INV_RESAMPLE_FIELD_H__
#define __SFEM_INV_RESAMPLE_FIELD_H__

#include "matrixio_array.h"
#include "sfem_resample_field.h"

#include "mass.h"

#include "mesh_aura.h"
#include "quadratures_rule.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {  // begin extern "C"
#endif

real_t                 //
min4(const real_t a,   //
     const real_t b,   //
     const real_t c,   //
     const real_t d);  //

real_t                 //
max4(const real_t a,   //
     const real_t b,   //
     const real_t c,   //
     const real_t d);  //

/**
 * @brief Perform the inverse resampling of a field on a mesh using tet4 elements
 *
 * @param start_element
 * @param end_element
 * @param nnodes
 * @param elems
 * @param xyz
 * @param weighted_field
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param data
 * @return int
 */
int                                                                                 //
tet4_inv_resample_field_local(const ptrdiff_t                      start_element,   // Mesh
                              const ptrdiff_t                      end_element,     // Mesh
                              const ptrdiff_t                      nnodes,          // Mesh
                              const idx_t** const SFEM_RESTRICT    elems,           // Mesh
                              const geom_t** const SFEM_RESTRICT   xyz,             // Mesh
                              const real_t* const SFEM_RESTRICT    weighted_field,  // Input (weighted field)
                              const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                              const ptrdiff_t* const SFEM_RESTRICT stride,          // SDF
                              const geom_t* const SFEM_RESTRICT    origin,          // SDF
                              const geom_t* const SFEM_RESTRICT    delta,           // SDF
                              real_t* const SFEM_RESTRICT          data);                    // SDF: Output

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // __SFEM_INV_RESAMPLE_FIELD_H__