#include "sfem_resample_gap.h"

#include "beam2_resample.h"
#include "quadshell4_resample.h"
#include "trishell3_resample.h"

#include "mass.h"
#include "read_mesh.h"

#include "matrixio_array.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int resample_gap_local(
        // Mesh
        const enum ElemType          element_type,
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT    origin,
        const geom_t* const SFEM_RESTRICT    delta,
        const geom_t* const SFEM_RESTRICT    data,
        // Output
        real_t* const SFEM_RESTRICT wg,
        real_t* const SFEM_RESTRICT xnormal,
        real_t* const SFEM_RESTRICT ynormal,
        real_t* const SFEM_RESTRICT znormal) {
    if (!nelements) return 0;

    enum ElemType st = shell_type(element_type);

    switch (st) {
        case TRISHELL3:
            return trishell3_resample_gap_local(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, wg, xnormal, ynormal, znormal);
        case BEAM2:
            return beam2_resample_gap_local(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, wg, xnormal, ynormal, znormal);
        case QUADSHELL4: {
            return quadshell4_resample_gap_local(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, wg, xnormal, ynormal, znormal);
        }
        default: {
            SFEM_ERROR("Invalid shell_element_type: %d from  element_type: %d\n", st, element_type);
            return EXIT_FAILURE;
        }
    }
}

int resample_weight_local(
        // Mesh
        const enum ElemType          element_type,
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT    origin,
        const geom_t* const SFEM_RESTRICT    delta,
        const geom_t* const SFEM_RESTRICT    data,
        // Output
        real_t* const SFEM_RESTRICT w) {
    if (!nelements) return SFEM_SUCCESS;

    enum ElemType st = shell_type(element_type);

    switch (st) {
        case TRISHELL3:
            return trishell3_resample_weight_local(nelements, nnodes, elems, xyz, w);
        case BEAM2:
            return beam2_resample_weight_local(nelements, nnodes, elems, xyz, w);
        case QUADSHELL4: {
            return quadshell4_resample_weight_local(nelements, nnodes, elems, xyz, w);
        }
        default: {
            SFEM_ERROR("Invalid shell_element_type: %d from  element_type: %d\n", st, element_type);
            return EXIT_FAILURE;
        }
    }
}

int resample_gap(
        // Mesh
        const enum ElemType          element_type,
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT    origin,
        const geom_t* const SFEM_RESTRICT    delta,
        const geom_t* const SFEM_RESTRICT    data,
        // Output
        real_t* const SFEM_RESTRICT g,
        real_t* const SFEM_RESTRICT xnormal,
        real_t* const SFEM_RESTRICT ynormal,
        real_t* const SFEM_RESTRICT znormal) {
    if (!nelements) return SFEM_SUCCESS;

    enum ElemType st = shell_type(element_type);
    memset(g, 0, nnodes * sizeof(real_t));

    if (resample_gap_local(st, nelements, nnodes, elems, xyz, n, stride, origin, delta, data, g, xnormal, ynormal, znormal) !=
        SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }

    real_t* w = calloc(nnodes, sizeof(real_t));
    if (resample_weight_local(st, nelements, nnodes, elems, xyz, n, stride, origin, delta, data, w) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        g[i] /= w[i];
    }

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        real_t denom = sqrt(xnormal[i] * xnormal[i] + ynormal[i] * ynormal[i] + znormal[i] * znormal[i]);
        xnormal[i] /= denom;
        ynormal[i] /= denom;
        znormal[i] /= denom;
    }

    free(w);
    return SFEM_SUCCESS;
}

SFEM_INLINE static void minmax(const ptrdiff_t n, const geom_t* const SFEM_RESTRICT x, geom_t* xmin, geom_t* xmax) {
    if (!n) return;

    *xmin = x[0];
    *xmax = x[0];
    for (ptrdiff_t i = 1; i < n; i++) {
        *xmin = MIN(*xmin, x[i]);
        *xmax = MAX(*xmax, x[i]);
    }
}

int sdf_view(MPI_Comm                             comm,
             const ptrdiff_t                      nnodes,
             const geom_t* SFEM_RESTRICT          z_coordinate,
             const ptrdiff_t* const               nlocal,
             const ptrdiff_t* const SFEM_RESTRICT nglobal,
             const ptrdiff_t* const SFEM_RESTRICT stride,
             const geom_t* const                  origin,
             const geom_t* const SFEM_RESTRICT    delta,
             const geom_t* const                  sdf,
             geom_t**                             sdf_out,
             ptrdiff_t*                           z_nlocal_out,
             geom_t* const SFEM_RESTRICT          z_origin_out) {
    return sdf_view_ensure_margin(
            comm, nnodes, z_coordinate, nlocal, nglobal, stride, origin, delta, sdf, 0, sdf_out, z_nlocal_out, z_origin_out);
}

int sdf_view_ensure_margin(MPI_Comm                             comm,
                           const ptrdiff_t                      nnodes,
                           const geom_t* SFEM_RESTRICT          z_coordinate,
                           const ptrdiff_t* const               nlocal,
                           const ptrdiff_t* const SFEM_RESTRICT nglobal,
                           const ptrdiff_t* const SFEM_RESTRICT stride,
                           const geom_t* const                  origin,
                           const geom_t* const SFEM_RESTRICT    delta,
                           const geom_t* const                  sdf,
                           const ptrdiff_t                      z_margin,
                           geom_t**                             sdf_out,
                           ptrdiff_t*                           z_nlocal_out,
                           geom_t* const SFEM_RESTRICT          z_origin_out) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size == 1) {
        SFEM_ERROR("[%d] resample_grid_view cannot be used in serial runs!\n", rank);
    }

    geom_t zmin = origin[2], zmax = origin[2];
    minmax(nnodes, z_coordinate, &zmin, &zmax);

    // Z is distributed
    ptrdiff_t zoffset = 0;
    MPI_Exscan(&nlocal[2], &zoffset, 1, MPI_LONG, MPI_SUM, comm);

    // // Compute Local z-tile
    ptrdiff_t sdf_start = (zmin - origin[2]) / delta[2];
    ptrdiff_t sdf_end   = (zmax - origin[2]) / delta[2];

    // Make sure we are inside the grid and get also the required margin for resampling
    sdf_start = MAX(0, sdf_start - 1 - z_margin);
    sdf_end   = MIN(nglobal[2],
                  sdf_end + 2 + z_margin);  // 1 for the rightside of the cell 1 for the exclusive range

    ptrdiff_t pnlocal_z = (sdf_end - sdf_start);
    geom_t*   psdf      = malloc(pnlocal_z * stride[2] * sizeof(geom_t));

    array_range_select(comm,
                       SFEM_MPI_GEOM_T,
                       (void*)sdf,
                       (void*)psdf,
                       // Size of z-slice
                       nlocal[2] * stride[2],
                       // starting offset
                       sdf_start * stride[2],
                       // ending offset
                       sdf_end * stride[2]);

    *sdf_out      = psdf;
    *z_nlocal_out = pnlocal_z;
    *z_origin_out = origin[2] + sdf_start * delta[2];

    return SFEM_SUCCESS;
}
