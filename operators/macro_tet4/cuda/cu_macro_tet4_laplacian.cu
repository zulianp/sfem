#include "cu_macro_tet4_laplacian.h"

#include "cu_macro_tet4_inline.hpp"
#include "cu_tet4_inline.hpp"
#include "cu_tet4_laplacian_inline.hpp"

#include "sfem_cuda_base.h"

#include <cassert>

template <typename real_t>
__global__ void cu_macro_tet4_laplacian_apply_kernel(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const idx_t *const SFEM_RESTRICT elems,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        const real_t *const SFEM_RESTRICT x,
        real_t *const SFEM_RESTRICT y) {
    scalar_t ex[10];
    scalar_t ey[10];
    geom_t sub_fff[6];

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ey[v] = 0;
        }

        // collect coeffs
        for (int v = 0; v < 10; ++v) {
            ex[v] = x[elems[v * stride + e]];
        }

        geom_t offf[6];
#pragma unroll(6)
        for (int d = 0; d < 6; d++) {
            offf[d] = fff[d * stride + e];
        }

        // Apply operator
        {  // Corner tets
            cu_macro_tet4_sub_fff_0(offf, 1, sub_fff);

            // [0, 4, 6, 7],
            cu_macro_tet4_laplacian_apply_fff(sub_fff,
                                              ex[0],
                                              ex[4],
                                              ex[6],
                                              ex[7],  //
                                              &ey[0],
                                              &ey[4],
                                              &ey[6],
                                              &ey[7]);

            // [4, 1, 5, 8],
            cu_macro_tet4_laplacian_apply_fff(sub_fff,
                                              ex[4],
                                              ex[1],
                                              ex[5],
                                              ex[8],  //
                                              &ey[4],
                                              &ey[1],
                                              &ey[5],
                                              &ey[8]);

            // [6, 5, 2, 9],
            cu_macro_tet4_laplacian_apply_fff(sub_fff,
                                              ex[6],
                                              ex[5],
                                              ex[2],
                                              ex[9],  //
                                              &ey[6],
                                              &ey[5],
                                              &ey[2],
                                              &ey[9]);

            // [7, 8, 9, 3],
            cu_macro_tet4_laplacian_apply_fff(sub_fff,
                                              ex[7],
                                              ex[8],
                                              ex[9],
                                              ex[3],  //
                                              &ey[7],
                                              &ey[8],
                                              &ey[9],
                                              &ey[3]);
        }

        {  // Octahedron tets

            // [4, 5, 6, 8],
            cu_macro_tet4_sub_fff_4(offf, 1, sub_fff);
            cu_macro_tet4_laplacian_apply_fff(sub_fff,
                                              ex[4],
                                              ex[5],
                                              ex[6],
                                              ex[8],  //
                                              &ey[4],
                                              &ey[5],
                                              &ey[6],
                                              &ey[8]);

            // [7, 4, 6, 8],
            cu_macro_tet4_sub_fff_5(offf, 1, sub_fff);
            cu_macro_tet4_laplacian_apply_fff(sub_fff,
                                              ex[7],
                                              ex[4],
                                              ex[6],
                                              ex[8],  //
                                              &ey[7],
                                              &ey[4],
                                              &ey[6],
                                              &ey[8]);

            // [6, 5, 9, 8],
            cu_macro_tet4_sub_fff_6(offf, 1, sub_fff);
            cu_macro_tet4_laplacian_apply_fff(sub_fff,
                                              ex[6],
                                              ex[5],
                                              ex[9],
                                              ex[8],  //
                                              &ey[6],
                                              &ey[5],
                                              &ey[9],
                                              &ey[8]);

            // [7, 6, 9, 8]]
            cu_macro_tet4_sub_fff_7(offf, 1, sub_fff);
            cu_macro_tet4_laplacian_apply_fff(sub_fff,
                                              ex[7],
                                              ex[6],
                                              ex[9],
                                              ex[8],  //
                                              &ey[7],
                                              &ey[6],
                                              &ey[9],
                                              &ey[8]);
        }

        // redistribute coeffs
        for (int v = 0; v < 10; ++v) {
            atomicAdd(&y[elems[v * stride + e]], ey[v]);
        }
    }
}

template <typename T>
static int cu_macro_tet4_laplacian_apply_tpl(const ptrdiff_t nelements,
                                             const ptrdiff_t stride,  // Stride for elements and fff
                                             const idx_t *const SFEM_RESTRICT elements,
                                             const cu_jacobian_t *const SFEM_RESTRICT fff,
                                             const T *const SFEM_RESTRICT x,
                                             T *const SFEM_RESTRICT y,
                                             void *stream) {
    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_macro_tet4_laplacian_apply_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_macro_tet4_laplacian_apply_kernel<T>
                <<<n_blocks, block_size, 0, s>>>(nelements, stride, elements, fff, x, y);
    } else {
        cu_macro_tet4_laplacian_apply_kernel<T>
                <<<n_blocks, block_size, 0>>>(nelements, stride, elements, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_macro_tet4_laplacian_apply(const ptrdiff_t nelements,
                                         const ptrdiff_t stride,  // Stride for elements and fff
                                         const idx_t *const SFEM_RESTRICT elements,
                                         const void *const SFEM_RESTRICT fff,
                                         const enum RealType real_type_xy,
                                         const void *const SFEM_RESTRICT x,
                                         void *const SFEM_RESTRICT y,
                                         void *stream) {
    switch (real_type_xy) {
        case SFEM_REAL_DEFAULT: {
            return cu_macro_tet4_laplacian_apply_tpl(nelements,
                                                     stride,
                                                     elements,
                                                     (cu_jacobian_t *)fff,
                                                     (real_t *)x,
                                                     (real_t *)y,
                                                     stream);
        }
        case SFEM_FLOAT32: {
            return cu_macro_tet4_laplacian_apply_tpl(nelements,
                                                     stride,
                                                     elements,
                                                     (cu_jacobian_t *)fff,
                                                     (float *)x,
                                                     (float *)y,
                                                     stream);
        }
        case SFEM_FLOAT64: {
            return cu_macro_tet4_laplacian_apply_tpl(nelements,
                                                     stride,
                                                     elements,
                                                     (cu_jacobian_t *)fff,
                                                     (double *)x,
                                                     (double *)y,
                                                     stream);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_macro_tet4_laplacian_apply: not implemented for type %s (code "
                    "%d)\n",
                    real_type_to_string(real_type_xy),
                    real_type_xy);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}

// DIAG

template <typename real_t>
__global__ void cu_macro_tet4_laplacian_diag_kernel(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const idx_t *const SFEM_RESTRICT elems,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        real_t *const SFEM_RESTRICT diag) {
    scalar_t ed[10];
    geom_t sub_fff[6];

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        for (int v = 0; v < 10; ++v) {
            ed[v] = 0;
        }

        geom_t offf[6];
        for (int d = 0; d < 6; d++) {
            offf[d] = fff[d * stride + e];
        }

        // Apply operator
        {  // Corner tets
            cu_macro_tet4_sub_fff_0(offf, 1, sub_fff);

            // [0, 4, 6, 7],
            cu_macro_tet4_laplacian_diag_fff(sub_fff, &ed[0], &ed[4], &ed[6], &ed[7]);

            // [4, 1, 5, 8],
            cu_macro_tet4_laplacian_diag_fff(sub_fff, &ed[4], &ed[1], &ed[5], &ed[8]);

            // [6, 5, 2, 9],
            cu_macro_tet4_laplacian_diag_fff(sub_fff, &ed[6], &ed[5], &ed[2], &ed[9]);

            // [7, 8, 9, 3],
            cu_macro_tet4_laplacian_diag_fff(sub_fff, &ed[7], &ed[8], &ed[9], &ed[3]);
        }

        {  // Octahedron tets

            // [4, 5, 6, 8],
            cu_macro_tet4_sub_fff_4(offf, 1, sub_fff);
            cu_macro_tet4_laplacian_diag_fff(sub_fff, &ed[4], &ed[5], &ed[6], &ed[8]);

            // [7, 4, 6, 8],
            cu_macro_tet4_sub_fff_5(offf, 1, sub_fff);
            cu_macro_tet4_laplacian_diag_fff(sub_fff, &ed[7], &ed[4], &ed[6], &ed[8]);

            // [6, 5, 9, 8],
            cu_macro_tet4_sub_fff_6(offf, 1, sub_fff);
            cu_macro_tet4_laplacian_diag_fff(sub_fff, &ed[6], &ed[5], &ed[9], &ed[8]);

            // [7, 6, 9, 8]]
            cu_macro_tet4_sub_fff_7(offf, 1, sub_fff);
            cu_macro_tet4_laplacian_diag_fff(sub_fff, &ed[7], &ed[6], &ed[9], &ed[8]);
        }

        // redistribute coeffs
        // #pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            const idx_t idx = elems[v * stride + e];
            assert(ed[v] != 0);
            atomicAdd(&diag[idx], ed[v]);
        }
    }
}

template <typename T>
static int cu_macro_tet4_laplacian_diag_tpl(const ptrdiff_t nelements,
                                            const ptrdiff_t stride,  // Stride for elements and fff
                                            const idx_t *const SFEM_RESTRICT elements,
                                            const cu_jacobian_t *const SFEM_RESTRICT fff,
                                            T *const SFEM_RESTRICT diag,
                                            void *stream) {
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_macro_tet4_laplacian_diag_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_macro_tet4_laplacian_diag_kernel<<<n_blocks, block_size, 0, s>>>(
                nelements, stride, elements, fff, diag);
    } else {
        cu_macro_tet4_laplacian_diag_kernel<<<n_blocks, block_size, 0>>>(
                nelements, stride, elements, fff, diag);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_macro_tet4_laplacian_diag(const ptrdiff_t nelements,
                                        const ptrdiff_t stride,  // Stride for elements and fff
                                        const idx_t *const SFEM_RESTRICT elements,
                                        const void *const SFEM_RESTRICT fff,
                                        const enum RealType real_type_diag,
                                        void *const SFEM_RESTRICT diag,
                                        void *stream) {
    switch (real_type_diag) {
        case SFEM_REAL_DEFAULT: {
            return cu_macro_tet4_laplacian_diag_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, (real_t *)diag, stream);
        }
        case SFEM_FLOAT32: {
            return cu_macro_tet4_laplacian_diag_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, (float *)diag, stream);
        }
        case SFEM_FLOAT64: {
            return cu_macro_tet4_laplacian_diag_tpl(
                    nelements, stride, elements, (cu_jacobian_t *)fff, (double *)diag, stream);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_macro_tet4_laplacian_diag: not implemented for type %s (code %d)\n",
                    real_type_to_string(real_type_diag),
                    real_type_diag);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}

// ------------ CRS

template <typename T>
__global__ void cu_macro_tet4_laplacian_crs_kernel(
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const idx_t *const SFEM_RESTRICT elements,
        const cu_jacobian_t *const SFEM_RESTRICT fff,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        T *const SFEM_RESTRICT values) {
    idx_t ev10[10];
    idx_t ev[4];

    accumulator_t element_matrix[4 * 4];
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        for (int v = 0; v < 10; ++v) {
            ev10[v] = elements[v * stride + e];
        }

        geom_t offf[6];
        scalar_t sub_fff[6];

        for (int d = 0; d < 6; d++) {
            offf[d] = fff[d * stride + e];
        }

        // Apply operator
        {  // Corner tets
            cu_macro_tet4_sub_fff_0(offf, 1, sub_fff);
            cu_tet4_laplacian_matrix_fff(sub_fff, element_matrix);

            cu_tet4_gather_idx(ev10, 0, 4, 6, 7, ev);
            cu_tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);

            cu_tet4_gather_idx(ev10, 4, 1, 5, 8, ev);
            cu_tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);

            cu_tet4_gather_idx(ev10, 6, 5, 2, 9, ev);
            cu_tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);

            cu_tet4_gather_idx(ev10, 7, 8, 9, 3, ev);
            cu_tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);
        }

        {  // Octahedron tets

            cu_macro_tet4_sub_fff_4(offf, 1, sub_fff);
            cu_tet4_laplacian_matrix_fff(sub_fff, element_matrix);
            cu_tet4_gather_idx(ev10, 4, 5, 6, 8, ev);
            cu_tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);

            cu_macro_tet4_sub_fff_5(offf, 1, sub_fff);
            cu_tet4_laplacian_matrix_fff(sub_fff, element_matrix);
            cu_tet4_gather_idx(ev10, 7, 4, 6, 8, ev);
            cu_tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);

            cu_macro_tet4_sub_fff_6(offf, 1, sub_fff);
            cu_tet4_laplacian_matrix_fff(sub_fff, element_matrix);
            cu_tet4_gather_idx(ev10, 6, 5, 9, 8, ev);
            cu_tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);

            cu_macro_tet4_sub_fff_7(offf, 1, sub_fff);
            cu_tet4_laplacian_matrix_fff(sub_fff, element_matrix);
            cu_tet4_gather_idx(ev10, 7, 6, 9, 8, ev);
            cu_tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);
        }
    }
}

template <typename T>
int cu_macro_tet4_laplacian_crs_tpl(const ptrdiff_t nelements,
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
        cu_macro_tet4_laplacian_crs_kernel<<<n_blocks, block_size, 0, s>>>(
                nelements, stride, elements, fff, rowptr, colidx, values);
    } else {
        cu_macro_tet4_laplacian_crs_kernel<<<n_blocks, block_size, 0>>>(
                nelements, stride, elements, fff, rowptr, colidx, values);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_macro_tet4_laplacian_crs(const ptrdiff_t nelements,
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
            return cu_macro_tet4_laplacian_crs_tpl(nelements,
                                                   stride,
                                                   elements,
                                                   (cu_jacobian_t *)fff,
                                                   rowptr,
                                                   colidx,
                                                   (real_t *)values,
                                                   stream);
        }
        case SFEM_FLOAT32: {
            return cu_macro_tet4_laplacian_crs_tpl(nelements,
                                                   stride,
                                                   elements,
                                                   (cu_jacobian_t *)fff,
                                                   rowptr,
                                                   colidx,
                                                   (float *)values,
                                                   stream);
        }
        case SFEM_FLOAT64: {
            return cu_macro_tet4_laplacian_crs_tpl(nelements,
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
                    "[Error] cu_macro_tet4_laplacian_crs: not implemented for type %s (code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}
