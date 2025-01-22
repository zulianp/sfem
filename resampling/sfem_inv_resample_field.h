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

typedef real_t (*function_XYZ_t)(real_t x, real_t y, real_t z);

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

/**
 * @brief Check if a point is inside a tetrahedron
 *
 * This function determines whether a point (px, py, pz) is inside a tetrahedron
 * defined by its four vertices (x0, y0, z0), (x1, y1, z1), (x2, y2, z2), and (x3, y3, z3).
 *
 * @param px x coordinate of the point
 * @param py y coordinate of the point
 * @param pz z coordinate of the point
 * @param x0 x coordinate of the 1st vertex
 * @param y0 y coordinate of the 1st vertex
 * @param z0 z coordinate of the 1st vertex
 * @param x1 x coordinate of the 2nd vertex
 * @param y1 y coordinate of the 2nd vertex
 * @param z1 z coordinate of the 2nd vertex
 * @param x2 x coordinate of the 3rd vertex
 * @param y2 y coordinate of the 3rd vertex
 * @param z2 z coordinate of the 3rd vertex
 * @param x3 x coordinate of the 4th vertex
 * @param y3 y coordinate of the 4th vertex
 * @param z3 z coordinate of the 4th vertex
 * @return int Returns 1 if the point is inside the tetrahedron, 0 otherwise.
 */
int                                           //
check_p_inside_tetrahedron(const real_t px,   //
                           const real_t py,   //
                           const real_t pz,   //
                           const real_t x0,   //
                           const real_t y0,   //
                           const real_t z0,   //
                           const real_t x1,   //
                           const real_t y1,   //
                           const real_t z1,   //
                           const real_t x2,   //
                           const real_t y2,   //
                           const real_t z2,   //
                           const real_t x3,   //
                           const real_t y3,   //
                           const real_t z3);  //

/**
 * @brief Apply a function to a mesh
 *
 * This function applies a given function to a mesh within the specified range of elements.
 *
 * @param start_element [in] The starting element index of the mesh to apply the function to.
 * @param end_element [in] The ending element index of the mesh to apply the function to.
 * @param nnodes [in] Number of nodes in the mesh.
 * @param elems [in] Pointer to the array of element indices in the mesh.
 * @param xyz [in] Pointer to the array of geometric coordinates of the nodes in the mesh.
 * @param fun [in] The function to be applied to the mesh.
 * @param weighted_field [out] Pointer to the output array where the weighted field will be stored.
 */
int                                                                  //
apply_fun_to_mesh(const ptrdiff_t                    start_element,  // Mesh
                  const ptrdiff_t                    end_element,    // Mesh
                  const ptrdiff_t                    nnodes,         // Mesh
                  const idx_t** const SFEM_RESTRICT  elems,          // Mesh
                  const geom_t** const SFEM_RESTRICT xyz,            // Mesh
                  const function_XYZ_t               fun,            // Function
                  real_t* const SFEM_RESTRICT        weighted_field);       //   Output (weighted field)

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // __SFEM_INV_RESAMPLE_FIELD_H__