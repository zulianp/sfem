#ifndef SFEM_RESAMPLE_GAP_H
#define SFEM_RESAMPLE_GAP_H

#include <mpi.h>
#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int resample_gap_local(
        // Mesh
        const enum ElemType element_type, const ptrdiff_t nelements, const ptrdiff_t nnodes,
        idx_t** const SFEM_RESTRICT elems, geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n, const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT origin, const geom_t* const SFEM_RESTRICT delta,
        const geom_t* const SFEM_RESTRICT data,
        // Output
        real_t* const SFEM_RESTRICT wg, real_t* const SFEM_RESTRICT xnormal,
        real_t* const SFEM_RESTRICT ynormal, real_t* const SFEM_RESTRICT znormal);

int resample_gap(
        // Mesh
        const enum ElemType element_type, const ptrdiff_t nelements, const ptrdiff_t nnodes,
        idx_t** const SFEM_RESTRICT elems, geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n, const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT origin, const geom_t* const SFEM_RESTRICT delta,
        const geom_t* const SFEM_RESTRICT data,
        // Output
        real_t* const SFEM_RESTRICT g, real_t* const SFEM_RESTRICT xnormal,
        real_t* const SFEM_RESTRICT ynormal, real_t* const SFEM_RESTRICT znormal);

int sdf_view(MPI_Comm comm, const ptrdiff_t nnodes, const geom_t* SFEM_RESTRICT z_coordinate,
             const ptrdiff_t* const nlocal, const ptrdiff_t* const SFEM_RESTRICT nglobal,
             const ptrdiff_t* const SFEM_RESTRICT stride, const geom_t* const origin,
             const geom_t* const SFEM_RESTRICT delta, const geom_t* const sdf, geom_t** sdf_out,
             ptrdiff_t* z_nlocal_out, geom_t* const SFEM_RESTRICT z_origin_out);

int sdf_view_ensure_margin(MPI_Comm comm, const ptrdiff_t nnodes,
                           const geom_t* SFEM_RESTRICT z_coordinate, const ptrdiff_t* const nlocal,
                           const ptrdiff_t* const SFEM_RESTRICT nglobal,
                           const ptrdiff_t* const SFEM_RESTRICT stride, const geom_t* const origin,
                           const geom_t* const SFEM_RESTRICT delta, const geom_t* const sdf,
                           const ptrdiff_t z_margin, geom_t** sdf_out, ptrdiff_t* z_nlocal_out,
                           geom_t* const SFEM_RESTRICT z_origin_out);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_RESAMPLE_GAP_H
