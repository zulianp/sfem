#ifndef TET10_RESAMPLE_FIELD_V2_H
#define TET10_RESAMPLE_FIELD_V2_H

#include <mpi.h>
#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Assemble the dual mass vector for the tet10 element.
 *
 * This function assembles the dual mass vector for a quadratic tetrahedral (tet10) element.
 * The dual mass vector is used in finite element methods to represent the mass distribution
 * across the elements.
 *
 * @param nelements Number of elements in the mesh.
 * @param nnodes Number of nodes in the mesh.
 * @param elems Connectivity array where elems[i][j] gives the global node index of the i-th local node in the j-th element.
 * @param xyz Coordinates of mesh nodes where xyz[d][i] gives the d-th coordinate (d=0 for x, d=1 for y, d=2 for z) of the i-th
 * node.
 * @param values Output array to store the assembled dual mass vector values.
 * @return int 0 if successful, non-zero error code otherwise.
 */
int                                                                         //
tet10_assemble_dual_mass_vector_V2(const ptrdiff_t              nelements,  // Number of elements
                                   const ptrdiff_t              nnodes,     // Number of nodes
                                   idx_t** const SFEM_RESTRICT  elems,      // Connectivity
                                   geom_t** const SFEM_RESTRICT xyz,        // Coordinates
                                   real_t* const                values);                   // Output values

/**
 * @brief Assemble the dual mass vector for the tet10 element using isoparametric mapping.
 *
 * This function assembles the dual mass vector for a quadratic tetrahedral (tet10) element
 * using isoparametric mapping, where both the geometry and the field are interpolated using
 * the same basis functions.
 *
 * @param nelements Number of elements in the mesh.
 * @param nnodes Number of nodes in the mesh.
 * @param elems Connectivity array where elems[i][j] gives the global node index of the i-th local node in the j-th element.
 * @param xyz Coordinates of mesh nodes where xyz[d][i] gives the d-th coordinate (d=0 for x, d=1 for y, d=2 for z) of the i-th
 * node.
 * @param diag Output array to store the diagonal of the mass matrix.
 * @return int 0 if successful, non-zero error code otherwise.
 */
int                                                                                      //
isoparametric_tet10_assemble_dual_mass_vector_V(const ptrdiff_t              nelements,  // Number of elements
                                                const ptrdiff_t              nnodes,     // Number of nodes
                                                idx_t** const SFEM_RESTRICT  elems,      // Connectivity
                                                geom_t** const SFEM_RESTRICT xyz,        // Coordinates
                                                real_t* const                diag);                     // Diagonal matrix

/**
 * @brief Compute the indices of the field for third order interpolation.
 *
 * This function computes the indices of the field for third order interpolation
 * from a structured hexahedral grid to a quadratic tetrahedral (tet10) mesh.
 *
 * @param nelements Number of elements in the mesh.
 * @param nnodes Number of nodes in the mesh.
 * @param elems Connectivity array where elems[i][j] gives the global node index of the i-th local node in the j-th element.
 * @param xyz Coordinates of mesh nodes where xyz[d][i] gives the d-th coordinate (d=0 for x, d=1 for y, d=2 for z) of the i-th
 * node.
 * @param n Number of nodes in each direction of the structured grid.
 * @param stride Memory stride values for the structured grid.
 * @param origin Origin coordinates of the structured grid.
 * @param delta Grid spacing in each dimension.
 * @param data Input field values defined on the structured grid.
 * @param weighted_field Output array to store the resampled field values.
 * @return int 0 if successful, non-zero error code otherwise.
 */
int                                                                                    //
hex8_to_tet10_resample_field_local_V2(const ptrdiff_t                      nelements,  // Number of elements
                                      const ptrdiff_t                      nnodes,     // Number of nodes
                                      idx_t** const SFEM_RESTRICT          elems,      // Connectivity
                                      geom_t** const SFEM_RESTRICT         xyz,        // Coordinates
                                      const ptrdiff_t* const SFEM_RESTRICT n,          // Number of nodes in each direction
                                      const ptrdiff_t* const SFEM_RESTRICT stride,     // Stride of the data
                                      const geom_t* const SFEM_RESTRICT    origin,     // Origin of the domain
                                      const geom_t* const SFEM_RESTRICT    delta,      // Delta of the domain
                                      const real_t* const SFEM_RESTRICT    data,       // SDF data
                                      real_t* const SFEM_RESTRICT          weighted_field);     // Output field

/**
 * @brief Compute the indices of the field for third order interpolation using WENO interpolation.
 *
 * This function computes the indices of the field for third order interpolation
 * from a structured hexahedral grid to a quadratic tetrahedral (tet10) mesh using
 * Weighted Essentially Non-Oscillatory (WENO) interpolation.
 *
 * @param nelements Number of elements in the mesh.
 * @param nnodes Number of nodes in the mesh.
 * @param elems Connectivity array where elems[i][j] gives the global node index of the i-th local node in the j-th element.
 * @param xyz Coordinates of mesh nodes where xyz[d][i] gives the d-th coordinate (d=0 for x, d=1 for y, d=2 for z) of the i-th
 * node.
 * @param n Number of nodes in each direction of the structured grid.
 * @param stride Memory stride values for the structured grid.
 * @param origin Origin coordinates of the structured grid.
 * @param delta Grid spacing in each dimension.
 * @param data Input field values defined on the structured grid.
 * @param weighted_field Output array to store the resampled field values.
 * @return int 0 if successful, non-zero error code otherwise.
 */
int                                                                                                       //
hex8_to_isoparametric_tet10_resample_field_local_cube1_V(const ptrdiff_t                      nelements,  // Number of elements
                                                         const ptrdiff_t                      nnodes,     // Number of nodes
                                                         idx_t** const SFEM_RESTRICT          elems,      // Connectivity
                                                         geom_t** const SFEM_RESTRICT         xyz,        // Coordinates
                                                         const ptrdiff_t* const SFEM_RESTRICT n,          // Number of nodes
                                                         const ptrdiff_t* const SFEM_RESTRICT stride,     // Stride of the data
                                                         const geom_t* const SFEM_RESTRICT    origin,     // Origin of the domain
                                                         const geom_t* const SFEM_RESTRICT    delta,      // Delta of the domain
                                                         const real_t* const SFEM_RESTRICT    data,       // SDF data
                                                         real_t* const SFEM_RESTRICT          weighted_field);     // Output field

/**
 * @brief Resample a field from a structured grid to a quadratic (10-node) tetrahedral mesh using subparametric mapping.
 *
 * This function resamples a scalar field defined on a structured hexahedral grid onto a
 * quadratic tetrahedral mesh (TET10). It uses subparametric mapping, where the geometric
 * mapping is of lower order than the field interpolation. Specifically, the geometric
 * transformation uses linear basis functions (P1) while the field interpolation uses
 * quadratic basis functions (P2).
 *
 * The subparametric approach provides a compromise between computational efficiency and accuracy,
 * maintaining second-order accurate field representation while using simplified geometry mapping.
 *
 * @param[in] nelements     Number of tetrahedral elements in the mesh.
 * @param[in] nnodes        Total number of nodes in the tetrahedral mesh.
 * @param[in] elems         Element connectivity array where elems[i][j] gives the global node index
 *                          of the i-th local node in the j-th element.
 * @param[in] xyz           Coordinates of mesh nodes where xyz[d][i] gives the d-th coordinate
 *                          (d=0 for x, d=1 for y, d=2 for z) of the i-th node.
 * @param[in] n             Number of grid points in each dimension of the structured grid [nx,ny,nz].
 * @param[in] stride        Memory stride values for the structured grid, defining how to
 *                          navigate through the linearized 3D array.
 * @param[in] origin        Origin coordinates of the structured grid [x0,y0,z0].
 * @param[in] delta         Grid spacing in each dimension [dx,dy,dz].
 * @param[in] data          Input field values defined on the structured grid.
 * @param[out] weighted_field Output field values on the tetrahedral mesh nodes (pre-multiplied
 *                          by the mass matrix).
 *
 * @return 0 on success, non-zero error code otherwise.
 *
 * @note This function computes the weighted field without applying the mass matrix inverse.
 *       To obtain the actual nodal values, divide the result by the diagonal of the mass matrix
 *       using tet10_assemble_dual_mass_vector functions.
 */
int                                                                                                 //
hex8_to_subparametric_tet10_resample_field_local_V(const ptrdiff_t                      nelements,  // Number of elements
                                                   const ptrdiff_t                      nnodes,     // Number of nodes
                                                   idx_t** const SFEM_RESTRICT          elems,      // Connectivity
                                                   geom_t** const SFEM_RESTRICT         xyz,        // Coordinates
                                                   const ptrdiff_t* const SFEM_RESTRICT n,  // Number of nodes in each direction
                                                   const ptrdiff_t* const SFEM_RESTRICT stride,  // Stride of the data
                                                   const geom_t* const SFEM_RESTRICT    origin,  // Origin of the domain
                                                   const geom_t* const SFEM_RESTRICT    delta,   // Delta of the domain
                                                   const real_t* const SFEM_RESTRICT    data,    // SDF data
                                                   real_t* const SFEM_RESTRICT          weighted_field);  // Output field

//////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

// TODO (full 2nd order FEM)
// For the structured grid we can decide to go directy high-order splines instead of tri-linear
// hexas int hex{8,?}_to_isoparametric_tet10_resample_field_local(...) int
// isoparametric_tet10_assemble_dual_mass_vector(...)

#endif  // TET10_RESAMPLE_FIELD_V2_H