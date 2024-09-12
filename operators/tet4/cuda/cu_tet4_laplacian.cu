#include "cu_tet4_laplacian.h"

#include "sfem_cuda_base.h"
#include "sfem_defs.h"

#include "cu_tet4_inline.hpp"
#include "cu_tet4_laplacian_inline.hpp"

#include <cassert>

template <typename real_t>
__global__ void cu_tet4_laplacian_apply_kernel(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        const real_t *const SFEM_RESTRICT x,
        real_t *const SFEM_RESTRICT y) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        scalar_t ex[4];
        scalar_t ey[4];
        idx_t vidx[4];

        // collect coeffs
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            vidx[v] = elements[v * stride + e];
            ex[v] = x[vidx[v]];
        }

        scalar_t fffe[6];
#pragma unroll(6)
        for (int d = 0; d < 6; d++) {
            fffe[d] = fff[d * stride + e];
        }

        cu_tet4_laplacian_apply_fff(fffe, 1, ex, ey);

        // redistribute coeffs
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            atomicAdd(&y[vidx[v]], ey[v]);
        }
    }
}

template <typename T>
static int cu_tet4_laplacian_apply_tpl(const ptrdiff_t nelements,
                                       const ptrdiff_t stride,  // Stride for elements and fff
                                       const idx_t *const SFEM_RESTRICT elements,
                                       const cu_jacobian_t *const SFEM_RESTRICT fff,
                                       const T *const x,
                                       T *const y,
                                       void *stream) {
    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_tet4_laplacian_apply_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_tet4_laplacian_apply_kernel<<<n_blocks, block_size, 0, s>>>(
                nelements, stride, elements, fff, x, y);
    } else {
        cu_tet4_laplacian_apply_kernel<<<n_blocks, block_size, 0>>>(
                nelements, stride, elements, fff, x, y);
    }

    return SFEM_SUCCESS;
}

extern int cu_tet4_laplacian_apply(const ptrdiff_t nelements,
                                   const ptrdiff_t stride,  // Stride for elements and fff
                                   const idx_t *const SFEM_RESTRICT elements,
                                   const void *const SFEM_RESTRICT fff,
                                   const enum RealType real_type_xy,
                                   const void *const x,
                                   void *const y,
                                   void *stream) {
    switch (real_type_xy) {
        case SFEM_REAL_DEFAULT: {
            return cu_tet4_laplacian_apply_tpl(nelements,
                                               stride,
                                               elements,
                                               (cu_jacobian_t *)fff,
                                               (real_t *)x,
                                               (real_t *)y,
                                               stream);
        }
        case SFEM_FLOAT32: {
            return cu_tet4_laplacian_apply_tpl(nelements,
                                               stride,
                                               elements,
                                               (cu_jacobian_t *)fff,
                                               (float *)x,
                                               (float *)y,
                                               stream);
        }
        case SFEM_FLOAT64: {
            return cu_tet4_laplacian_apply_tpl(nelements,
                                               stride,
                                               elements,
                                               (cu_jacobian_t *)fff,
                                               (double *)x,
                                               (double *)y,
                                               stream);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_tet4_laplacian_apply: not implemented for type %s (code %d)\n",
                    real_type_to_string(real_type_xy),
                    real_type_xy);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}

template <typename real_t>
__global__ void cu_tet4_laplacian_diag_kernel(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        real_t *const SFEM_RESTRICT diag) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        scalar_t ed[4];
        idx_t vidx[4];

#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            vidx[v] = elements[v * stride + e];
        }

        scalar_t fffe[6];
#pragma unroll(6)
        for (int d = 0; d < 6; d++) {
            fffe[d] = fff[d * stride + e];
        }

        // Assembler operator diagonal
        cu_tet4_laplacian_diag_fff(fffe, 1, ed);

        // redistribute coeffs
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            assert(ed[v] != 0);
            atomicAdd(&diag[vidx[v]], ed[v]);
        }
    }
}

template <typename T>
static int cu_tet4_laplacian_diag_tpl(const ptrdiff_t nelements,
                                      const ptrdiff_t stride,  // Stride for elements and fff
                                      const idx_t *const SFEM_RESTRICT elements,
                                      const cu_jacobian_t *const SFEM_RESTRICT fff,
                                      T *const diag,
                                      void *stream) {
    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_tet4_laplacian_diag_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_tet4_laplacian_diag_kernel<<<n_blocks, block_size, 0, s>>>(
                nelements, stride, elements, fff, diag);
    } else {
        cu_tet4_laplacian_diag_kernel<<<n_blocks, block_size, 0>>>(
                nelements, stride, elements, fff, diag);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_tet4_laplacian_diag(const ptrdiff_t nelements,
                                  const ptrdiff_t stride,  // Stride for elements and fff
                                  const idx_t *const SFEM_RESTRICT elements,
                                  const void *const SFEM_RESTRICT fff,
                                  const enum RealType real_type_diag,
                                  void *const diag,
                                  void *stream) {
    switch (real_type_diag) {
        case SFEM_REAL_DEFAULT: {
            return cu_tet4_laplacian_diag_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, (real_t *)diag, stream);
        }
        case SFEM_FLOAT32: {
            return cu_tet4_laplacian_diag_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, (float *)diag, stream);
        }
        case SFEM_FLOAT64: {
            return cu_tet4_laplacian_diag_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, (double *)diag, stream);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_tet4_laplacian_diag: not implemented for type %s (code %d)\n",
                    real_type_to_string(real_type_diag),
                    real_type_diag);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}

// ------------ CRS

template <typename T>
__global__ void cu_tet4_laplacian_crs_kernel(const ptrdiff_t nelements,
                                             const ptrdiff_t stride,  // Stride for elements and fff
                                             const idx_t *const SFEM_RESTRICT elements,
                                             const cu_jacobian_t *const SFEM_RESTRICT fff,
                                             const count_t *const SFEM_RESTRICT rowptr,
                                             const idx_t *const SFEM_RESTRICT colidx,
                                             T *const SFEM_RESTRICT values) {

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        accumulator_t element_matrix[4 * 4];
        idx_t ev[4];

#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v * stride + e];
        }

        scalar_t fffe[6];
#pragma unroll(6)
        for (int d = 0; d < 6; d++) {
            fffe[d] = fff[d * stride + e];
        }

        cu_tet4_laplacian_matrix_fff(
                fffe,
                element_matrix); 

        cu_tet4_local_to_global(
                ev,
                element_matrix,
                rowptr,
                colidx,
                values);
    }
}

template <typename T>
int cu_tet4_laplacian_crs_tpl(const ptrdiff_t nelements,
                              const ptrdiff_t stride,  // Stride for elements and fff
                              const idx_t *const SFEM_RESTRICT elements,
                              const cu_jacobian_t *const SFEM_RESTRICT fff,
                              const count_t *const SFEM_RESTRICT rowptr,
                              const idx_t *const SFEM_RESTRICT colidx,
                              T *const SFEM_RESTRICT values,
                              void *stream) {
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_tet4_laplacian_crs_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_tet4_laplacian_crs_kernel<<<n_blocks, block_size, 0, s>>>(
                nelements, stride, elements, fff, rowptr, colidx, values);
    } else {
        cu_tet4_laplacian_crs_kernel<<<n_blocks, block_size, 0>>>(
                nelements, stride, elements, fff, rowptr, colidx, values);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_tet4_laplacian_crs(const ptrdiff_t nelements,
                                 const ptrdiff_t stride,  // Stride for elements and fff
                                 const idx_t *const SFEM_RESTRICT elements,
                                 const void *const SFEM_RESTRICT fff,
                                 const count_t *const SFEM_RESTRICT rowptr,
                                 const idx_t *const SFEM_RESTRICT colidx,
                                 const enum RealType real_type,
                                 void *const SFEM_RESTRICT values,
                                 void *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_tet4_laplacian_crs_tpl(nelements,
                                             stride,
                                             elements,
                                             (cu_jacobian_t *)fff,
                                             rowptr,
                                             colidx,
                                             (real_t *)values,
                                             stream);
        }
        case SFEM_FLOAT32: {
            return cu_tet4_laplacian_crs_tpl(nelements,
                                             stride,
                                             elements,
                                             (cu_jacobian_t *)fff,
                                             rowptr,
                                             colidx,
                                             (float *)values,
                                             stream);
        }
        case SFEM_FLOAT64: {
            return cu_tet4_laplacian_crs_tpl(nelements,
                                             stride,
                                             elements,
                                             (cu_jacobian_t *)fff,
                                             rowptr,
                                             colidx,
                                             (double *)values,
                                             stream);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_tet4_laplacian_crs: not implemented for type %s (code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}
