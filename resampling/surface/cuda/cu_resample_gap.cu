#include "cu_resample_gap.h"

#include "cu_quadshell4_resample.h"

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
        // case TRISHELL3:
        //     return cu_trishell3_resample_gap_local(
        //             nelements, nnodes, elems, xyz, n, stride, origin, delta, data, wg, xnormal, ynormal, znormal);
        // case BEAM2:
        //     return cu_beam2_resample_gap_local(
        //             nelements, nnodes, elems, xyz, n, stride, origin, delta, data, wg, xnormal, ynormal, znormal);
        case QUADSHELL4: {
            return cu_quadshell4_resample_gap_local(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, wg, xnormal, ynormal, znormal);
        }
        default: {
            SFEM_ERROR("Invalid shell_element_type: %d from  element_type: %d\n", st, element_type);
            return SFEM_FAILURE;
        }
    }
}

int cu_resample_weight_local(
        // Mesh
        const enum ElemType          element_type,
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // Output
        real_t* const SFEM_RESTRICT w) {
    if (!nelements) return SFEM_SUCCESS;

    enum ElemType st = shell_type(element_type);

    switch (st) {
        // case TRISHELL3:
        //     return cu_trishell3_resample_weight_local(nelements, nnodes, elems, xyz, w);
        // case BEAM2:
        //     return cu_beam2_resample_weight_local(nelements, nnodes, elems, xyz, w);
        case QUADSHELL4: {
            return cu_quadshell4_resample_weight_local(nelements, nnodes, elems, xyz, w);
        }
        default: {
            SFEM_ERROR("Invalid shell_element_type: %d from  element_type: %d\n", st, element_type);
            return SFEM_FAILURE;
        }
    }
}

int cu_resample_gap(
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

    if (cu_resample_gap_local(st, nelements, nnodes, elems, xyz, n, stride, origin, delta, data, g, xnormal, ynormal, znormal) !=
        SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }

    SFEM_ERROR("IMPLEMENT ME!\n");

    //     real_t* w = calloc(nnodes, sizeof(real_t));
    //     if (resample_weight_local(st, nelements, nnodes, elems, xyz, w) != SFEM_SUCCESS) {
    //         return SFEM_FAILURE;
    //     }

    // #pragma omp parallel for
    //     for (ptrdiff_t i = 0; i < nnodes; i++) {
    //         g[i] /= w[i];
    //     }

    // #pragma omp parallel for
    //     for (ptrdiff_t i = 0; i < nnodes; i++) {
    //         real_t denom = sqrt(xnormal[i] * xnormal[i] + ynormal[i] * ynormal[i] + znormal[i] * znormal[i]);
    //         xnormal[i] /= denom;
    //         ynormal[i] /= denom;
    //         znormal[i] /= denom;
    //     }

    //     free(w);
    return SFEM_SUCCESS;
}

int cu_resample_gap_value_local(
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
        real_t* const SFEM_RESTRICT g) {
    if (!nelements) return SFEM_SUCCESS;

    enum ElemType st = shell_type(element_type);

    switch (st) {
        case QUADSHELL4: {
            return cu_quadshell4_resample_gap_value_local(nelements, nnodes, elems, xyz, n, stride, origin, delta, data, g);
        }
        default: {
            SFEM_ERROR("Invalid shell_element_type: %d from  element_type: %d\n", st, element_type);
            return SFEM_FAILURE;
        }
    }
}

int cu_resample_gap_value(
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
        real_t* const SFEM_RESTRICT g) {
    if (!nelements) return SFEM_SUCCESS;

    enum ElemType st = shell_type(element_type);
    memset(g, 0, nnodes * sizeof(real_t));

    if (cu_resample_gap_value_local(st, nelements, nnodes, elems, xyz, n, stride, origin, delta, data, g) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }

    SFEM_ERROR("IMPLEMENT ME!");

    //     real_t* w = calloc(nnodes, sizeof(real_t));
    //     if (resample_weight_local(st, nelements, nnodes, elems, xyz, w) != SFEM_SUCCESS) {
    //         return SFEM_FAILURE;
    //     }

    // #pragma omp parallel for
    //     for (ptrdiff_t i = 0; i < nnodes; i++) {
    //         g[i] /= w[i];
    //     }

    //     free(w);
    return SFEM_SUCCESS;
}

int cu_resample_gap_normals_local(
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
        real_t* const SFEM_RESTRICT xnormal,
        real_t* const SFEM_RESTRICT ynormal,
        real_t* const SFEM_RESTRICT znormal) {
    if (!nelements) return SFEM_SUCCESS;

    enum ElemType st = shell_type(element_type);

    switch (st) {
        case QUADSHELL4: {
            return cu_quadshell4_resample_gap_normals_local(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, xnormal, ynormal, znormal);
        }
        default: {
            SFEM_ERROR("Invalid shell_element_type: %d from element_type: %d\n", st, element_type);
            return SFEM_FAILURE;
        }
    }
}

int cu_resample_gap_normals(
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
        real_t* const SFEM_RESTRICT xnormal,
        real_t* const SFEM_RESTRICT ynormal,
        real_t* const SFEM_RESTRICT znormal) {
    if (!nelements) return SFEM_SUCCESS;

    if (cu_resample_gap_normals_local(shell_type(element_type),
                                      nelements,
                                      nnodes,
                                      elems,
                                      xyz,
                                      n,
                                      stride,
                                      origin,
                                      delta,
                                      data,
                                      xnormal,
                                      ynormal,
                                      znormal) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }

    SFEM_ERROR("IMPLEMENT ME!");

    // #pragma omp parallel for
    //     for (ptrdiff_t i = 0; i < nnodes; i++) {
    //         real_t denom = sqrt(xnormal[i] * xnormal[i] + ynormal[i] * ynormal[i] + znormal[i] * znormal[i]);
    //         xnormal[i] /= denom;
    //         ynormal[i] /= denom;
    //         znormal[i] /= denom;
    //     }

    return SFEM_SUCCESS;
}
