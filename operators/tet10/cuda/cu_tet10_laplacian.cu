#include "cu_tet10_laplacian.hpp"
#include "cu_tet10_laplacian_inline.hpp"
#include "sfem_cuda_base.hpp"

#include <cassert>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

template <typename real_t>
__global__ void cu_tet10_laplacian_apply_kernel(const ptrdiff_t                          nelements,
                                                idx_t **const SFEM_RESTRICT              elems,
                                                const ptrdiff_t                          fff_stride,
                                                const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                const real_t *const SFEM_RESTRICT        x,
                                                real_t *const SFEM_RESTRICT              y) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        scalar_t ex[10];
        scalar_t ey[10];
        idx_t    vidx[10];

        for (int v = 0; v < 10; ++v) {
            ey[v] = 0;
        }

        // collect coeffs
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            vidx[v] = elems[v][e];
            ex[v]   = x[vidx[v]];
        }

        geom_t fffe[6];
#pragma unroll(6)
        for (int d = 0; d < 6; d++) {
            fffe[d] = fff[d * fff_stride + e];
        }

        cu_tet10_laplacian_apply_fff(fffe, 1, ex, ey);

        // redistribute coeffs
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            atomicAdd(&y[vidx[v]], ey[v]);
        }
    }
}

template <typename T>
static int cu_tet10_laplacian_apply_tpl(const ptrdiff_t                          nelements,
                                        idx_t **const SFEM_RESTRICT              elements,
                                        const ptrdiff_t                          fff_stride,
                                        const cu_jacobian_t *const SFEM_RESTRICT fff,
                                        const T *const SFEM_RESTRICT             x,
                                        T *const SFEM_RESTRICT                   y,
                                        void                                    *stream) {
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_tet10_laplacian_apply_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_tet10_laplacian_apply_kernel<<<n_blocks, block_size, 0, s>>>(nelements, elements, fff_stride, fff, x, y);
    } else {
        cu_tet10_laplacian_apply_kernel<<<n_blocks, block_size, 0>>>(nelements, elements, fff_stride, fff, x, y);
    }

    return SFEM_SUCCESS;
}

extern int cu_tet10_laplacian_apply(const ptrdiff_t                  nelements,
                                    idx_t **const SFEM_RESTRICT elements,
                                    const ptrdiff_t                  fff_stride,
                                    const void *const SFEM_RESTRICT  fff,
                                    const enum smesh::PrimitiveType              real_type_xy,
                                    const void *const SFEM_RESTRICT  x,
                                    void *const SFEM_RESTRICT        y,
                                    void                            *stream) {
    switch (real_type_xy) {
        case smesh::SMESH_DEFAULT: {
            return cu_tet10_laplacian_apply_tpl(
                    nelements, elements, fff_stride, (cu_jacobian_t *)fff, (real_t *)x, (real_t *)y, stream);
        }
        case smesh::SMESH_FLOAT32: {
            return cu_tet10_laplacian_apply_tpl(
                    nelements, elements, fff_stride, (cu_jacobian_t *)fff, (float *)x, (float *)y, stream);
        }
        case smesh::SMESH_FLOAT64: {
            return cu_tet10_laplacian_apply_tpl(
                    nelements, elements, fff_stride, (cu_jacobian_t *)fff, (double *)x, (double *)y, stream);
        }
        default: {
            SFEM_ERROR(
                    "[Error] cu_tet10_laplacian_apply: not implemented for type %s (code %d)\n",
                    smesh::to_string(real_type_xy),
                    real_type_xy);
            return SFEM_FAILURE;
        }
    }
}
