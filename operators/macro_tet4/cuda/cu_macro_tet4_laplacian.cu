#include "cu_macro_tet4_laplacian.h"

#include "cu_tet4_inline.hpp"

#include "sfem_cuda_base.h"

#include <cassert>

// NOTE: Promotes to more expressive type duiring computation

template <typename fff_t, typename sub_fff_t>
static /*inline*/ __device__ __host__ void sub_fff_0(const fff_t *const SFEM_RESTRICT fff,
                                                     const ptrdiff_t stride,
                                                     sub_fff_t *const SFEM_RESTRICT sub_fff) {
    sub_fff[0] = (sub_fff_t)(1.0 / 2.0) * fff[0 * stride];
    sub_fff[1] = (sub_fff_t)(1.0 / 2.0) * fff[1 * stride];
    sub_fff[2] = (sub_fff_t)(1.0 / 2.0) * fff[2 * stride];
    sub_fff[3] = (sub_fff_t)(1.0 / 2.0) * fff[3 * stride];
    sub_fff[4] = (sub_fff_t)(1.0 / 2.0) * fff[4 * stride];
    sub_fff[5] = (sub_fff_t)(1.0 / 2.0) * fff[5 * stride];
}

template <typename fff_t, typename sub_fff_t>
static /*inline*/ __device__ __host__ void sub_fff_4(const fff_t *const SFEM_RESTRICT fff,
                                                     const ptrdiff_t stride,
                                                     sub_fff_t *const SFEM_RESTRICT sub_fff) {
    const fff_t x0 = (sub_fff_t)(1.0 / 2.0) * fff[0 * stride];
    const fff_t x1 = (sub_fff_t)(1.0 / 2.0) * fff[2 * stride];
    sub_fff[0] = fff[1 * stride] + (1.0 / 2.0) * fff[3 * stride] + x0;
    sub_fff[1] = (sub_fff_t)(-1.0 / 2.0) * fff[1 * stride] - x0;
    sub_fff[2] = (sub_fff_t)(1.0 / 2.0) * fff[4 * stride] + x1;
    sub_fff[3] = x0;
    sub_fff[4] = -x1;
    sub_fff[5] = (sub_fff_t)(1.0 / 2.0) * fff[5 * stride];
}

template <typename fff_t, typename sub_fff_t>
static /*inline*/ __device__ __host__ void sub_fff_5(const fff_t *const SFEM_RESTRICT fff,
                                                     const ptrdiff_t stride,
                                                     sub_fff_t *const SFEM_RESTRICT sub_fff) {
    const fff_t x0 = (sub_fff_t)(1.0 / 2.0) * fff[3 * stride];
    const fff_t x1 = fff[4 * stride] + (sub_fff_t)(1.0 / 2.0) * fff[5 * stride] + x0;
    const fff_t x2 = (sub_fff_t)(1.0 / 2.0) * fff[4 * stride] + x0;
    const fff_t x3 = (sub_fff_t)(1.0 / 2.0) * fff[1 * stride];
    sub_fff[0] = x1;
    sub_fff[1] = -x2;
    sub_fff[2] = (sub_fff_t)(-1.0 / 2.0) * fff[2 * stride] - x1 - x3;
    sub_fff[3] = x0;
    sub_fff[4] = x2 + x3;
    sub_fff[5] = (sub_fff_t)(1.0 / 2.0) * fff[0 * stride] + fff[1 * stride] + fff[2 * stride] + x1;
}

template <typename fff_t, typename sub_fff_t>
static /*inline*/ __device__ __host__ void sub_fff_6(const fff_t *const SFEM_RESTRICT fff,
                                                     const ptrdiff_t stride,
                                                     sub_fff_t *const SFEM_RESTRICT sub_fff) {
    const fff_t x0 = (sub_fff_t)(1.0 / 2.0) * fff[3 * stride];
    const fff_t x1 = (sub_fff_t)(1.0 / 2.0) * fff[4 * stride];
    const fff_t x2 = (sub_fff_t)(1.0 / 2.0) * fff[1 * stride] + x0;
    sub_fff[0] = (sub_fff_t)(1.0 / 2.0) * fff[0 * stride] + fff[1 * stride] + x0;
    sub_fff[1] = (sub_fff_t)(1.0 / 2.0) * fff[2 * stride] + x1 + x2;
    sub_fff[2] = -x2;
    sub_fff[3] = fff[4 * stride] + (sub_fff_t)(1.0 / 2.0) * fff[5 * stride] + x0;
    sub_fff[4] = -x0 - x1;
    sub_fff[5] = x0;
}

template <typename fff_t, typename sub_fff_t>
static /*inline*/ __device__ __host__ void sub_fff_7(const fff_t *const SFEM_RESTRICT fff,
                                                     const ptrdiff_t stride,
                                                     sub_fff_t *const SFEM_RESTRICT sub_fff) {
    const fff_t x0 = (sub_fff_t)(1.0 / 2.0) * fff[5 * stride];
    const fff_t x1 = (sub_fff_t)(1.0 / 2.0) * fff[2 * stride];
    sub_fff[0] = x0;
    sub_fff[1] = (sub_fff_t)(-1.0 / 2.0) * fff[4 * stride] - x0;
    sub_fff[2] = -x1;
    sub_fff[3] = (sub_fff_t)(1.0 / 2.0) * fff[3 * stride] + fff[4 * stride] + x0;
    sub_fff[4] = (sub_fff_t)(1.0 / 2.0) * fff[1 * stride] + x1;
    sub_fff[5] = (sub_fff_t)(1.0 / 2.0) * fff[0 * stride];
}

// APPLY

template <typename fff_t, typename scalar_t, typename accumulator_t>
static /*inline*/ __device__ __host__ void lapl_apply_micro_kernel(
        const fff_t *const SFEM_RESTRICT fff,
        const scalar_t u0,
        const scalar_t u1,
        const scalar_t u2,
        const scalar_t u3,
        accumulator_t *const SFEM_RESTRICT e0,
        accumulator_t *const SFEM_RESTRICT e1,
        accumulator_t *const SFEM_RESTRICT e2,
        accumulator_t *const SFEM_RESTRICT e3) {
    const scalar_t x0 = fff[0] + fff[1] + fff[2];
    const scalar_t x1 = fff[1] + fff[3] + fff[4];
    const scalar_t x2 = fff[2] + fff[4] + fff[5];
    const scalar_t x3 = fff[1] * u0;
    const scalar_t x4 = fff[2] * u0;
    const scalar_t x5 = fff[4] * u0;
    *e0 += u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
    *e1 += -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
    *e2 += fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
    *e3 += fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
}

template <typename fff_t, typename accumulator_t>
static inline __device__ __host__ void lapl_diag_micro_kernel(const fff_t *const SFEM_RESTRICT fff,
                                                              accumulator_t *const SFEM_RESTRICT e0,
                                                              accumulator_t *const SFEM_RESTRICT e1,
                                                              accumulator_t *const SFEM_RESTRICT e2,
                                                              accumulator_t *const SFEM_RESTRICT e3) {
    *e0 += fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] + 2 * fff[4] + fff[5];
    *e1 += fff[0];
    *e2 += fff[3];
    *e3 += fff[5];
}

template <typename real_t>
__global__ void cu_macro_tet4_laplacian_apply_kernel(const ptrdiff_t nelements,
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
            ex[v] = x[elems[v * nelements + e]];
        }

        const ptrdiff_t stride = 1;
        geom_t offf[6];
#pragma unroll(6)
        for (int d = 0; d < 6; d++) {
            offf[d] = fff[d * nelements + e];
        }

        // Apply operator
        {  // Corner tets
            sub_fff_0(offf, stride, sub_fff);

            // [0, 4, 6, 7],
            lapl_apply_micro_kernel(sub_fff,
                                    ex[0],
                                    ex[4],
                                    ex[6],
                                    ex[7],  //
                                    &ey[0],
                                    &ey[4],
                                    &ey[6],
                                    &ey[7]);

            // [4, 1, 5, 8],
            lapl_apply_micro_kernel(sub_fff,
                                    ex[4],
                                    ex[1],
                                    ex[5],
                                    ex[8],  //
                                    &ey[4],
                                    &ey[1],
                                    &ey[5],
                                    &ey[8]);

            // [6, 5, 2, 9],
            lapl_apply_micro_kernel(sub_fff,
                                    ex[6],
                                    ex[5],
                                    ex[2],
                                    ex[9],  //
                                    &ey[6],
                                    &ey[5],
                                    &ey[2],
                                    &ey[9]);

            // [7, 8, 9, 3],
            lapl_apply_micro_kernel(sub_fff,
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
            sub_fff_4(offf, stride, sub_fff);
            lapl_apply_micro_kernel(sub_fff,
                                    ex[4],
                                    ex[5],
                                    ex[6],
                                    ex[8],  //
                                    &ey[4],
                                    &ey[5],
                                    &ey[6],
                                    &ey[8]);

            // [7, 4, 6, 8],
            sub_fff_5(offf, stride, sub_fff);
            lapl_apply_micro_kernel(sub_fff,
                                    ex[7],
                                    ex[4],
                                    ex[6],
                                    ex[8],  //
                                    &ey[7],
                                    &ey[4],
                                    &ey[6],
                                    &ey[8]);

            // [6, 5, 9, 8],
            sub_fff_6(offf, stride, sub_fff);
            lapl_apply_micro_kernel(sub_fff,
                                    ex[6],
                                    ex[5],
                                    ex[9],
                                    ex[8],  //
                                    &ey[6],
                                    &ey[5],
                                    &ey[9],
                                    &ey[8]);

            // [7, 6, 9, 8]]
            sub_fff_7(offf, stride, sub_fff);
            lapl_apply_micro_kernel(sub_fff,
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
            atomicAdd(&y[elems[v * nelements + e]], ey[v]);
        }
    }
}

template <typename T>
static int cu_macro_tet4_laplacian_apply_tpl(const ptrdiff_t nelements,
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

        cu_macro_tet4_laplacian_apply_kernel<T><<<n_blocks, block_size, 0, s>>>(
                nelements, elements, fff, x, y);
    } else {

        cu_macro_tet4_laplacian_apply_kernel<T><<<n_blocks, block_size, 0>>>(
                nelements, elements, fff, x, y);
    }

    return SFEM_SUCCESS;
}

extern int cu_macro_tet4_laplacian_apply(const ptrdiff_t nelements,
                                         const idx_t *const SFEM_RESTRICT elements,
                                         const void *const SFEM_RESTRICT fff,
                                         const enum RealType real_type_xy,
                                         const void *const SFEM_RESTRICT x,
                                         void *const SFEM_RESTRICT y,
                                         void *stream) {
    switch (real_type_xy) {
        case SFEM_REAL_DEFAULT: {
            return cu_macro_tet4_laplacian_apply_tpl(
                    nelements, elements, (cu_jacobian_t *)fff, (real_t *)x, (real_t *)y, stream);
        }
        case SFEM_FLOAT32: {
            return cu_macro_tet4_laplacian_apply_tpl(
                    nelements, elements, (cu_jacobian_t *)fff, (float *)x, (float *)y, stream);
        }
        case SFEM_FLOAT64: {
            return cu_macro_tet4_laplacian_apply_tpl(
                    nelements, elements, (cu_jacobian_t *)fff, (double *)x, (double *)y, stream);
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
__global__ void cu_macro_tet4_laplacian_diag_kernel(const ptrdiff_t nelements,
                                                    const idx_t *const SFEM_RESTRICT elems,
                                                    const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                    real_t *const SFEM_RESTRICT diag) {
    scalar_t ed[10];
    geom_t sub_fff[6];

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ed[v] = 0;
        }

        const ptrdiff_t stride = 1;
        geom_t offf[6];
#pragma unroll(6)
        for (int d = 0; d < 6; d++) {
            offf[d] = fff[d * nelements + e];
        }

        // Apply operator
        {  // Corner tets
            sub_fff_0(offf, stride, sub_fff);

            // [0, 4, 6, 7],
            lapl_diag_micro_kernel(sub_fff, &ed[0], &ed[4], &ed[6], &ed[7]);

            // [4, 1, 5, 8],
            lapl_diag_micro_kernel(sub_fff, &ed[4], &ed[1], &ed[5], &ed[8]);

            // [6, 5, 2, 9],
            lapl_diag_micro_kernel(sub_fff, &ed[6], &ed[5], &ed[2], &ed[9]);

            // [7, 8, 9, 3],
            lapl_diag_micro_kernel(sub_fff, &ed[7], &ed[8], &ed[9], &ed[3]);
        }

        {  // Octahedron tets

            // [4, 5, 6, 8],
            sub_fff_4(offf, stride, sub_fff);
            lapl_diag_micro_kernel(sub_fff, &ed[4], &ed[5], &ed[6], &ed[8]);

            // [7, 4, 6, 8],
            sub_fff_5(offf, stride, sub_fff);
            lapl_diag_micro_kernel(sub_fff, &ed[7], &ed[4], &ed[6], &ed[8]);

            // [6, 5, 9, 8],
            sub_fff_6(offf, stride, sub_fff);
            lapl_diag_micro_kernel(sub_fff, &ed[6], &ed[5], &ed[9], &ed[8]);

            // [7, 6, 9, 8]]
            sub_fff_7(offf, stride, sub_fff);
            lapl_diag_micro_kernel(sub_fff, &ed[7], &ed[6], &ed[9], &ed[8]);
        }

        // redistribute coeffs
        // #pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            atomicAdd(&diag[elems[v * nelements + e]], ed[v]);
        }
    }
}

template <typename T>
static int cu_macro_tet4_laplacian_diag_tpl(const ptrdiff_t nelements,
                                            const idx_t *const SFEM_RESTRICT elements,
                                            const cu_jacobian_t *const SFEM_RESTRICT fff,
                                            T *const SFEM_RESTRICT diag,
                                            void *stream) {
    // Hand tuned
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
                nelements, elements, fff, diag);
    } else {
        cu_macro_tet4_laplacian_diag_kernel<<<n_blocks, block_size, 0>>>(
                nelements, elements, fff, diag);
    }

    return SFEM_SUCCESS;
}

extern int cu_macro_tet4_laplacian_diag(const ptrdiff_t nelements,
                                        const idx_t *const SFEM_RESTRICT elements,
                                        const void *const SFEM_RESTRICT fff,
                                        const enum RealType real_type_diag,
                                        void *const SFEM_RESTRICT diag,
                                        void *stream) {
    switch (real_type_diag) {
        case SFEM_REAL_DEFAULT: {
            return cu_macro_tet4_laplacian_diag_tpl(
                    nelements, elements, (cu_jacobian_t *)fff, (real_t *)diag, stream);
        }
        case SFEM_FLOAT32: {
            return cu_macro_tet4_laplacian_diag_tpl(
                    nelements, elements, (cu_jacobian_t *)fff, (float *)diag, stream);
        }
        case SFEM_FLOAT64: {
            return cu_macro_tet4_laplacian_diag_tpl(
                    nelements, elements, (cu_jacobian_t *)fff, (double *)diag, stream);
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
