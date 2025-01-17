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
 * @brief Assemble the dual mass vector for the tet10 element
 *
 * @param nelements
 * @param nnodes
 * @param elems
 * @param xyz
 * @param values
 * @return int
 */
int tet10_assemble_dual_mass_vector_V2(const ptrdiff_t nelements, const ptrdiff_t nnodes,
                                       idx_t** const SFEM_RESTRICT elems,
                                       geom_t** const SFEM_RESTRICT xyz, real_t* const values);

/**
 * @brief Assemble the dual mass vector for the tet10 element
 *
 * @param nelements number of elements
 * @param nnodes number of nodes
 * @param elems connectivity
 * @param xyz coordinates
 * @param diag diagonal
 * @return int 0 if successful
 */
int isoparametric_tet10_assemble_dual_mass_vector_V(const ptrdiff_t nelements,
                                                    const ptrdiff_t nnodes,
                                                    idx_t** const SFEM_RESTRICT elems,
                                                    geom_t** const SFEM_RESTRICT xyz,
                                                    real_t* const diag);

/**
 * @brief Compute the indices of the field for third order interpolation
 *
 * @param nelements
 * @param nnodes
 * @param elems
 * @param xyz
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param data
 * @param weighted_field
 * @return int
 */
int hex8_to_tet10_resample_field_local_V2(
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

/**
 * @brief Compute the indices of the field for third order interpolation
 * by using the WENO interpolation in the resampling.
 *
 * @param nelements
 * @param nnodes
 * @param elems
 * @param xyz
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param data
 * @param weighted_field
 * @return int
 */
int hex8_to_isoparametric_tet10_resample_field_local_cube1_V(
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

/**
 * @brief Compute the resampling of the field using the subparametric
 *
 * @param nelements
 * @param nnodes
 * @param elems
 * @param xyz
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param data
 * @param weighted_field
 * @return int
 */
int hex8_to_subparametric_tet10_resample_field_local_V(
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

//////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

// TODO (full 2nd order FEM)
// For the structured grid we can decide to go directy high-order splines instead of tri-linear
// hexas int hex{8,?}_to_isoparametric_tet10_resample_field_local(...) int
// isoparametric_tet10_assemble_dual_mass_vector(...)

#endif  // TET10_RESAMPLE_FIELD_V2_H