#include "cu_resample_gap.h"

#include "cu_quadshell4_resample.h"
#include "sfem_cuda_base.h"
#include "sfem_macros.h"

extern "C" int cu_resample_gap_local(
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
        real_t* const SFEM_RESTRICT  wg,
        real_t** const SFEM_RESTRICT normals) {
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
            return cu_quadshell4_resample_gap_local(nelements, nnodes, elems, xyz, n, stride, origin, delta, data, wg, normals);
        }
        default: {
            SFEM_ERROR("Invalid shell_element_type: %d from  element_type: %d\n", st, element_type);
            return SFEM_FAILURE;
        }
    }
}

extern "C" int cu_resample_weight_local(
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

__global__ void cu_in_place_div(const ptrdiff_t n, real_t* const SFEM_RESTRICT inout, const real_t* const SFEM_RESTRICT w) {
    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        inout[i] /= w[i];
    }
}

static int rescale_with_weight(const enum ElemType          element_type,
                               const ptrdiff_t              nelements,
                               const ptrdiff_t              nnodes,
                               idx_t** const SFEM_RESTRICT  elems,
                               geom_t** const SFEM_RESTRICT xyz,
                               real_t* const SFEM_RESTRICT  g) {
    real_t* w;
    cudaMalloc(&w, nnodes * sizeof(real_t));
    cudaMemset(w, 0, nnodes * sizeof(real_t));

    if (cu_resample_weight_local(element_type, nelements, nnodes, elems, xyz, w) != SFEM_SUCCESS) {
        cudaFree(w);
        return SFEM_FAILURE;
    }

    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (nnodes + block_size - 1) / block_size);
    cu_in_place_div<<<n_blocks, block_size, 0>>>(nnodes, g, w);

    cudaFree(w);
    return SFEM_SUCCESS;
}

extern "C" int cu_resample_gap(
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
        real_t* const SFEM_RESTRICT  g,
        real_t** const SFEM_RESTRICT normals) {
    if (!nelements) return SFEM_SUCCESS;

    enum ElemType st = shell_type(element_type);
    cudaMemset(g, 0, nnodes * sizeof(real_t));

    int err = 0;
    err     = cu_resample_gap_local(st, nelements, nnodes, elems, xyz, n, stride, origin, delta, data, g, normals);
    err |= rescale_with_weight(element_type, nelements, nnodes, elems, xyz, g);
    err |= cu_normalize(nnodes, normals);

    return err;
}

extern "C" int cu_resample_gap_value_local(
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

extern "C" int cu_resample_gap_value(
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
    cudaMemset(g, 0, nnodes * sizeof(real_t));
    int err = cu_resample_gap_value_local(st, nelements, nnodes, elems, xyz, n, stride, origin, delta, data, g);
    err |= rescale_with_weight(element_type, nelements, nnodes, elems, xyz, g);
    return err;
}

extern "C" int cu_resample_gap_normals_local(
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
        real_t** const SFEM_RESTRICT normals) {
    if (!nelements) return SFEM_SUCCESS;

    enum ElemType st = shell_type(element_type);

    switch (st) {
        case QUADSHELL4: {
            return cu_quadshell4_resample_gap_normals_local(
                    nelements, nnodes, elems, xyz, n, stride, origin, delta, data, normals);
        }
        default: {
            SFEM_ERROR("Invalid shell_element_type: %d from element_type: %d\n", st, element_type);
            return SFEM_FAILURE;
        }
    }
}

__global__ void cu_normalize_kernel(const ptrdiff_t nnodes, real_t** const SFEM_RESTRICT normals) {
    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nnodes; i += gridDim.x * blockDim.x) {
        const real_t nx = normals[0][i];
        const real_t ny = normals[1][i];
        const real_t nz = normals[2][i];

        const real_t denom = sqrt(nx * nx + ny * ny + nz * nz);

        if (denom > 0) {
            normals[0][i] = nx / denom;
            normals[1][i] = ny / denom;
            normals[2][i] = nz / denom;
        }
    }
}

extern "C" int cu_normalize(const ptrdiff_t nnodes, real_t** const SFEM_RESTRICT normals) {
    if (!nnodes) return SFEM_SUCCESS;

    SFEM_DEBUG_SYNCHRONIZE();

    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (nnodes + block_size - 1) / block_size);
    cu_normalize_kernel<<<n_blocks, block_size, 0>>>(nnodes, normals);

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern "C" int cu_resample_gap_normals(
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
        real_t** const SFEM_RESTRICT normals) {
    if (!nelements) return SFEM_SUCCESS;

    int err = cu_resample_gap_normals_local(
                   shell_type(element_type), nelements, nnodes, elems, xyz, n, stride, origin, delta, data, normals);
    err |= cu_normalize(nnodes, normals);
    return err;
}
