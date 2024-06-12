#ifndef TET10_RESAMPLE_FIELD_H
#define TET10_RESAMPLE_FIELD_H

#include <mpi.h>
#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int tet10_resample_field_local(
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

#ifdef __cplusplus
}
#endif

#endif  // TET10_RESAMPLE_FIELD_H
