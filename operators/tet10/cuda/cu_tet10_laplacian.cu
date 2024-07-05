#include "cu_tet10_laplacian.h"
#include "sfem_cuda_base.h"

#include <cassert>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define POW2(a) ((a) * (a))

static inline __device__ __host__ void trial_operand(const scalar_t qx,
                                                     const scalar_t qy,
                                                     const scalar_t qz,
                                                     const scalar_t qw,
                                                     const ptrdiff_t stride,
                                                     const geom_t *const SFEM_RESTRICT fff,
                                                     const scalar_t *const SFEM_RESTRICT u,
                                                     scalar_t *const SFEM_RESTRICT out) {
    const scalar_t x0 = 4 * qx;
    const scalar_t x1 = 4 * qy;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x1 + x2;
    const scalar_t x4 = -u[6] * x1;
    const scalar_t x5 = u[0] * (x0 + x3 - 3);
    const scalar_t x6 = -u[7] * x2 + x5;
    const scalar_t x7 =
            u[1] * (x0 - 1) + u[4] * (-8 * qx - x3 + 4) + u[5] * x1 + u[8] * x2 + x4 + x6;
    const scalar_t x8 = x0 - 4;
    const scalar_t x9 = -u[4] * x0;
    const scalar_t x10 =
            u[2] * (x1 - 1) + u[5] * x0 + u[6] * (-8 * qy - x2 - x8) + u[9] * x2 + x6 + x9;
    const scalar_t x11 =
            u[3] * (x2 - 1) + u[7] * (-8 * qz - x1 - x8) + u[8] * x0 + u[9] * x1 + x4 + x5 + x9;
    out[0] = qw * (fff[0 * stride] * x7 + fff[1 * stride] * x10 + fff[2 * stride] * x11);
    out[1] = qw * (fff[1 * stride] * x7 + fff[3 * stride] * x10 + fff[4 * stride] * x11);
    out[2] = qw * (fff[2 * stride] * x7 + fff[4 * stride] * x10 + fff[5 * stride] * x11);
}

static inline __device__ __host__ void ref_shape_grad_x(const scalar_t qx,
                                                        const scalar_t qy,
                                                        const scalar_t qz,
                                                        scalar_t *const out) {
    const scalar_t x0 = 4 * qx;
    const scalar_t x1 = 4 * qy;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x1 + x2;
    out[0] = x0 + x3 - 3;
    out[1] = x0 - 1;
    out[2] = 0;
    out[3] = 0;
    out[4] = -8 * qx - x3 + 4;
    out[5] = x1;
    out[6] = -x1;
    out[7] = -x2;
    out[8] = x2;
    out[9] = 0;
}

static inline __device__ __host__ void ref_shape_grad_y(const scalar_t qx,
                                                        const scalar_t qy,
                                                        const scalar_t qz,
                                                        scalar_t *const out) {
    const scalar_t x0 = 4 * qy;
    const scalar_t x1 = 4 * qx;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x1 + x2;
    out[0] = x0 + x3 - 3;
    out[1] = 0;
    out[2] = x0 - 1;
    out[3] = 0;
    out[4] = -x1;
    out[5] = x1;
    out[6] = -8 * qy - x3 + 4;
    out[7] = -x2;
    out[8] = 0;
    out[9] = x2;
}

static inline __device__ __host__ void ref_shape_grad_z(const scalar_t qx,
                                                        const scalar_t qy,
                                                        const scalar_t qz,
                                                        scalar_t *const out) {
    const scalar_t x0 = 4 * qz;
    const scalar_t x1 = 4 * qx;
    const scalar_t x2 = 4 * qy;
    const scalar_t x3 = x1 + x2;
    out[0] = x0 + x3 - 3;
    out[1] = 0;
    out[2] = 0;
    out[3] = x0 - 1;
    out[4] = -x1;
    out[5] = 0;
    out[6] = -x2;
    out[7] = -8 * qz - x3 + 4;
    out[8] = x1;
    out[9] = x2;
}

static inline __device__ __host__ void lapl_apply_micro_kernel(
        const scalar_t qx,
        const scalar_t qy,
        const scalar_t qz,
        const scalar_t qw,
        const ptrdiff_t stride,
        const geom_t *const SFEM_RESTRICT fff,
        const scalar_t *const SFEM_RESTRICT u,
        scalar_t *const SFEM_RESTRICT element_vector) {
    // Registers
    scalar_t ref_grad[10];
    scalar_t grad_u[3];

    // Evaluate gradient fe function transformed with fff and scaling factors
    trial_operand(qx, qy, qz, qw, stride, fff, u, grad_u);

    {  // X-components
        ref_shape_grad_x(qx, qy, qz, ref_grad);
#pragma unroll(10)
        for (int i = 0; i < 10; i++) {
            element_vector[i] += ref_grad[i] * grad_u[0];
        }
    }
    {  // Y-components
        ref_shape_grad_y(qx, qy, qz, ref_grad);
#pragma unroll(10)
        for (int i = 0; i < 10; i++) {
            element_vector[i] += ref_grad[i] * grad_u[1];
        }
    }

    {  // Z-components
        ref_shape_grad_z(qx, qy, qz, ref_grad);
#pragma unroll(10)
        for (int i = 0; i < 10; i++) {
            element_vector[i] += ref_grad[i] * grad_u[2];
        }
    }
}

template <typename real_t>
__global__ void cu_tet10_laplacian_apply_kernel(const ptrdiff_t nelements,
                                                const idx_t *const SFEM_RESTRICT elems,
                                                const cu_jacobian_t *const SFEM_RESTRICT fff,
                                                const real_t *const SFEM_RESTRICT x,
                                                real_t *const SFEM_RESTRICT y) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements;
         e += blockDim.x * gridDim.x) {
        scalar_t ex[10];
        scalar_t ey[10];
        idx_t vidx[10];

        // collect coeffs
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            vidx[v] = elems[v * nelements + e];
            ex[v] = x[vidx[v]];
        }

        geom_t fffe[6];
#pragma unroll(6)
        for (int d = 0; d < 6; d++) {
            fffe[d] = fff[d * nelements + e];
        }

        {  // Numerical quadrature
            lapl_apply_micro_kernel(0, 0, 0, 0.025, 1, fffe, ex, ey);
            lapl_apply_micro_kernel(1, 0, 0, 0.025, 1, fffe, ex, ey);
            lapl_apply_micro_kernel(0, 1, 0, 0.025, 1, fffe, ex, ey);
            lapl_apply_micro_kernel(0, 0, 1, 0.025, 1, fffe, ex, ey);

            static const scalar_t athird = 1. / 3;
            lapl_apply_micro_kernel(athird, athird, 0., 0.225, 1, fffe, ex, ey);
            lapl_apply_micro_kernel(athird, 0., athird, 0.225, 1, fffe, ex, ey);
            lapl_apply_micro_kernel(0., athird, athird, 0.225, 1, fffe, ex, ey);
            lapl_apply_micro_kernel(athird, athird, athird, 0.225, 1, fffe, ex, ey);
        }

        // redistribute coeffs
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            atomicAdd(&y[vidx[v]], ey[v]);
        }
    }
}

template <typename T>
static int cu_tet10_laplacian_apply_tpl(const ptrdiff_t nelements,
                                    const idx_t *const SFEM_RESTRICT elements,
                                    const cu_jacobian_t *const SFEM_RESTRICT fff,
                                    const T *const SFEM_RESTRICT x,
                                    T *const SFEM_RESTRICT y,
                                    void *stream) {
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, cu_tet10_laplacian_apply_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    const ptrdiff_t n_blocks = MIN(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);
        cu_tet10_laplacian_apply_kernel<<<n_blocks, block_size, 0, s>>>(
                nelements, elements, fff, x, y);
    } else {
        cu_tet10_laplacian_apply_kernel<<<n_blocks, block_size, 0>>>(
                nelements, elements, fff, x, y);
    }

    return SFEM_SUCCESS;
}

extern int cu_tet10_laplacian_apply(const ptrdiff_t nelements,
                                    const idx_t *const SFEM_RESTRICT elements,
                                    const void *const SFEM_RESTRICT fff,
                                    const enum RealType real_type_xy,
                                    const void *const SFEM_RESTRICT x,
                                    void *const SFEM_RESTRICT y,
                                    void *stream) {
    switch (real_type_xy) {
        case SFEM_REAL_DEFAULT: {
            return cu_tet10_laplacian_apply_tpl(
                    nelements, elements, (cu_jacobian_t *)fff, (real_t *)x, (real_t *)y, stream);
        }
        case SFEM_FLOAT32: {
            return cu_tet10_laplacian_apply_tpl(
                    nelements, elements, (cu_jacobian_t *)fff, (float *)x, (float *)y, stream);
        }
        case SFEM_FLOAT64: {
            return cu_tet10_laplacian_apply_tpl(
                    nelements, elements, (cu_jacobian_t *)fff, (double *)x, (double *)y, stream);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_tet10_laplacian_apply: not implemented for type %s (code %d)\n",
                    real_type_to_string(real_type_xy),
                    real_type_xy);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}
