#ifndef SFEM_RESAMPLE_FIELD_H
#define SFEM_RESAMPLE_FIELD_H

#include <mpi.h>
#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// typedef enum {
//     TET4,
//     TET10
// } element_type;

typedef struct {
    ptrdiff_t quad_nodes_cnt;
    ptrdiff_t nelements;
    //     element_type element_type;
    enum ElemType element_type;
} sfem_resample_field_info;

int resample_field_local(
        // Mesh
        const enum ElemType element_type, const ptrdiff_t nelements, const ptrdiff_t nnodes,
        idx_t** const SFEM_RESTRICT elems, geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n, const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT origin, const geom_t* const SFEM_RESTRICT delta,
        const real_t* const SFEM_RESTRICT data,
        // Output
        real_t* const SFEM_RESTRICT wg, sfem_resample_field_info* info);

int resample_field(
        // Mesh
        const enum ElemType element_type, const ptrdiff_t nelements, const ptrdiff_t nnodes,
        idx_t** const SFEM_RESTRICT elems, geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n, const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT origin, const geom_t* const SFEM_RESTRICT delta,
        const real_t* const SFEM_RESTRICT data,
        // Output
        real_t* const SFEM_RESTRICT field, sfem_resample_field_info* info);

int interpolate_field(const ptrdiff_t nnodes, geom_t** const SFEM_RESTRICT xyz,
                      // SDF
                      const ptrdiff_t* const SFEM_RESTRICT n,
                      const ptrdiff_t* const SFEM_RESTRICT stride,
                      const geom_t* const SFEM_RESTRICT origin,
                      const geom_t* const SFEM_RESTRICT delta,
                      const real_t* const SFEM_RESTRICT data,
                      // Output
                      real_t* const SFEM_RESTRICT field);

int field_view(MPI_Comm comm, const ptrdiff_t nnodes, const geom_t* SFEM_RESTRICT z_coordinate,
               const ptrdiff_t* const nlocal, const ptrdiff_t* const SFEM_RESTRICT nglobal,
               const ptrdiff_t* const SFEM_RESTRICT stride, const geom_t* const origin,
               const geom_t* const SFEM_RESTRICT delta, const real_t* const field,
               real_t** field_out, ptrdiff_t* z_nlocal_out,
               geom_t* const SFEM_RESTRICT z_origin_out);

int field_view_ensure_margin(MPI_Comm comm, const ptrdiff_t nnodes,
                             const geom_t* SFEM_RESTRICT z_coordinate,
                             const ptrdiff_t* const nlocal,
                             const ptrdiff_t* const SFEM_RESTRICT nglobal,
                             const ptrdiff_t* const SFEM_RESTRICT stride,
                             const geom_t* const origin, const geom_t* const SFEM_RESTRICT delta,
                             const real_t* const field, const ptrdiff_t z_margin,
                             real_t** field_out, ptrdiff_t* z_nlocal_out,
                             geom_t* const SFEM_RESTRICT z_origin_out);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_RESAMPLE_FIELD_H
