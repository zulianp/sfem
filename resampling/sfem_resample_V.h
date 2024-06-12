#ifndef SFEM_RESAMPLE_V
#define SFEM_RESAMPLE_V

#include <mpi.h>
#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif
int tet4_resample_field_local_v2(
        // Mesh
        const ptrdiff_t start_element,
        const ptrdiff_t end_element,
        const ptrdiff_t nnodes,
        idx_t** const SFEM_RESTRICT elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT origin,
        const geom_t* const SFEM_RESTRICT delta,
        const real_t* const SFEM_RESTRICT data,
        // Output
        real_t* const SFEM_RESTRICT weighted_field);

int tet4_resample_field_local_V8(
        // Mesh
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t** const SFEM_RESTRICT elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT origin,
        const geom_t* const SFEM_RESTRICT delta,
        const real_t* const SFEM_RESTRICT data,
        // Output
        real_t* const SFEM_RESTRICT weighted_field);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_RESAMPLE_V
