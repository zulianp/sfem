#ifndef TET10_RESAMPLE_FIELD_H
#define TET10_RESAMPLE_FIELD_H

#include <mpi.h>
#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * sub-parametric assumes that the mesh geometry is 1st order (i.e., tet4 with no warped
 elements)
 * but approximation space (coefficients) is 2nd order.
 */
int hex8_to_subparametric_tet10_resample_field_local(
        // Mesh
        const ptrdiff_t nelements,          // number of elements
        const ptrdiff_t nnodes,             // number of nodes
        idx_t** const SFEM_RESTRICT elems,  // connectivity
        geom_t** const SFEM_RESTRICT xyz,   // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
        const geom_t* const SFEM_RESTRICT origin,     // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,      // delta of the domain
        const real_t* const SFEM_RESTRICT data,       // SDF
        // Output
        real_t* const SFEM_RESTRICT weighted_field);

// int subparametric_tet10_assemble_dual_mass_vector(const ptrdiff_t nelements,
//                                                   const ptrdiff_t nnodes,
//                                                   idx_t** const SFEM_RESTRICT elems,
//                                                   geom_t** const SFEM_RESTRICT xyz,
//                                                   real_t* const values);

int tet10_assemble_dual_mass_vector(const ptrdiff_t nelements, const ptrdiff_t nnodes,
                                    idx_t** const SFEM_RESTRICT elems,
                                    geom_t** const SFEM_RESTRICT xyz, real_t* const values);

int hex8_to_tet10_resample_field_local(
        // Mesh
        const ptrdiff_t nelements,          // number of elements
        const ptrdiff_t nnodes,             // number of nodes
        idx_t** const SFEM_RESTRICT elems,  // connectivity
        geom_t** const SFEM_RESTRICT xyz,   // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
        const geom_t* const SFEM_RESTRICT origin,     // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,      // delta of the domain
        const real_t* const SFEM_RESTRICT data,       // SDF
        // Output
        real_t* const SFEM_RESTRICT weighted_field);

int hex8_to_isoparametric_tet10_resample_field_local_cube1(
        /// Mesh
        const ptrdiff_t nelements,          // number of elements
        const ptrdiff_t nnodes,             // number of nodes
        idx_t** const SFEM_RESTRICT elems,  // connectivity
        geom_t** const SFEM_RESTRICT xyz,   // coordinates
        /// SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
        const geom_t* const SFEM_RESTRICT origin,     // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,      // delta of the domain
        const real_t* const SFEM_RESTRICT data,       // SDF
        // Output
        real_t* const SFEM_RESTRICT weighted_field);

real_t hex_aa_8_eval_weno4_3D_Unit(  //
        const real_t x_unit,         //
        const real_t y_unit,         //
        const real_t z_unit,         //
        const real_t ox_unit,        // Coordinates of the origin of the grid in the unitary space
        const real_t oy_unit,        // for the structured grid
        const real_t oz_unit,        // X, Y and Z
        const ptrdiff_t i,           // it must be the absolute index
        const ptrdiff_t j,           // Used to get the data
        const ptrdiff_t k,           // From the data array
        const ptrdiff_t* stride,     //
        const real_t* const SFEM_RESTRICT data);

#ifdef __cplusplus
}
#endif

// TODO (full 2nd order FEM)
// For the structured grid we can decide to go directy high-order splines instead of tri-linear
// hexas int hex{8,?}_to_isoparametric_tet10_resample_field_local(...) int
// isoparametric_tet10_assemble_dual_mass_vector(...)

#endif  // TET10_RESAMPLE_FIELD_H
