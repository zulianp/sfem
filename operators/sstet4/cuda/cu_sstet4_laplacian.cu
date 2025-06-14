#include "cu_sstet4_laplacian.h"

#include "cu_tet4_inline.hpp"
#include "sfem_cuda_base.h"

#include <cassert>

#define POW3(x) ((x) * (x) * (x))

template <typename fff_t, typename scalar_t, typename accumulator_t>
static /*inline*/ __device__ __host__ void cu_sstet4_laplacian_apply_fff(const fff_t *const SFEM_RESTRICT   fff,
                                                                         const scalar_t                     u0,
                                                                         const scalar_t                     u1,
                                                                         const scalar_t                     u2,
                                                                         const scalar_t                     u3,
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
    *e0               = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
    *e1               = -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
    *e2               = fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
    *e3               = fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
}

static inline __device__ __host__ int sstet4_nxe(int level) {
    int num_nodes = 0;
    if (level % 2 == 0) {
        for (int i = 0; i < level / 2; i++) {
            num_nodes += (level - i + 1) * (i + 1) * 2;
        }
        num_nodes += (level / 2 + 1) * (level / 2 + 1);
    } else {
        for (int i = 0; i < level / 2 + 1; i++) {
            num_nodes += (level - i + 1) * (i + 1) * 2;
        }
    }

    return num_nodes;
}

static inline __device__ __host__ int sstet4_txe(int level) { return (int)POW3(level); }

template <typename fff_t, typename scalar_t>
static inline __device__ __host__ void cu_sstet4_sub_fff_0(const int                        L,
                                                           const fff_t *const SFEM_RESTRICT fff,
                                                           scalar_t *const SFEM_RESTRICT    sub_fff) {
    const scalar_t x0 = 1.0 / L;
    sub_fff[0]        = fff[0] * x0;
    sub_fff[1]        = fff[1] * x0;
    sub_fff[2]        = fff[2] * x0;
    sub_fff[3]        = fff[3] * x0;
    sub_fff[4]        = fff[4] * x0;
    sub_fff[5]        = fff[5] * x0;
}

template <typename fff_t, typename scalar_t>
static inline __device__ __host__ void cu_sstet4_sub_fff_1(const int                        L,
                                                           const fff_t *const SFEM_RESTRICT fff,
                                                           scalar_t *const SFEM_RESTRICT    sub_fff) {
    const scalar_t x0 = 1. / POW3(L);
    const scalar_t x1 = L * fff[0];
    const scalar_t x2 = L * fff[1];
    const scalar_t x3 = L * (-x1 - x2);
    const scalar_t x4 = -L * fff[3] - x2;
    const scalar_t x5 = L * fff[2];
    const scalar_t x6 = L * fff[4];
    const scalar_t x7 = (1 / POW2(L));
    sub_fff[0]        = x0 * (-L * x4 - x3);
    sub_fff[1]        = x0 * (L * (-x5 - x6) + x3);
    sub_fff[2]        = x4 * x7;
    sub_fff[3]        = x0 * (L * (x1 + x5) + L * (L * fff[5] + x5));
    sub_fff[4]        = x7 * (x2 + x6);
    sub_fff[5]        = fff[3] / L;
}

template <typename fff_t, typename scalar_t>
static inline __device__ __host__ void cu_sstet4_sub_fff_2(const int                        L,
                                                           const fff_t *const SFEM_RESTRICT fff,
                                                           scalar_t *const SFEM_RESTRICT    sub_fff) {
    const scalar_t x0 = 1. / POW3(L);
    const scalar_t x1 = L * fff[0];
    const scalar_t x2 = L * fff[1];
    const scalar_t x3 = x1 + x2;
    const scalar_t x4 = L * x3;
    const scalar_t x5 = POW2(L);
    const scalar_t x6 = L * fff[2];
    sub_fff[0]        = x0 * (L * (L * fff[3] + x2) + x4);
    sub_fff[1]        = -x3 / x5;
    sub_fff[2]        = x0 * (L * (L * fff[4] + x6) + x4);
    sub_fff[3]        = fff[0] / L;
    sub_fff[4]        = x0 * (-fff[0] * x5 - fff[2] * x5);
    sub_fff[5]        = x0 * (L * (x1 + x6) + L * (L * fff[5] + x6));
}

template <typename fff_t, typename scalar_t>
static inline __device__ __host__ void cu_sstet4_sub_fff_3(const int                        L,
                                                           const fff_t *const SFEM_RESTRICT fff,
                                                           scalar_t *const SFEM_RESTRICT    sub_fff) {
    const scalar_t x0  = 1. / POW3(L);
    const scalar_t x1  = L * fff[0];
    const scalar_t x2  = L * fff[2];
    const scalar_t x3  = x1 + x2;
    const scalar_t x4  = -L * x3;
    const scalar_t x5  = L * fff[5] + x2;
    const scalar_t x6  = -L * x5 + x4;
    const scalar_t x7  = L * fff[1];
    const scalar_t x8  = L * fff[4];
    const scalar_t x9  = x7 + x8;
    const scalar_t x10 = -L * x9;
    const scalar_t x11 = L * fff[3];
    const scalar_t x12 = L * (-x1 - x7) + L * (-x11 - x7);
    sub_fff[0]         = -x0 * x6;
    sub_fff[1]         = x0 * (-x10 - x4);
    sub_fff[2]         = x0 * (x10 + x6);
    sub_fff[3]         = -x0 * x12;
    sub_fff[4]         = x0 * (L * (-x2 - x8) + x12);
    sub_fff[5]         = x0 * (L * (x11 + x9) + L * (x3 + x7) + L * (x5 + x8));
}

template <typename fff_t, typename scalar_t>
static inline __device__ __host__ void cu_sstet4_sub_fff_4(const int                        L,
                                                           const fff_t *const SFEM_RESTRICT fff,
                                                           scalar_t *const SFEM_RESTRICT    sub_fff) {
    const scalar_t x0 = 1.0 / L;
    const scalar_t x1 = 1. / POW3(L);
    const scalar_t x2 = POW2(L);
    const scalar_t x3 = L * fff[1];
    const scalar_t x4 = L * fff[2];
    const scalar_t x5 = L * fff[0] + x3 + x4;
    const scalar_t x6 = L * fff[4];
    sub_fff[0]        = fff[3] * x0;
    sub_fff[1]        = x1 * (-fff[1] * x2 - fff[3] * x2 - fff[4] * x2);
    sub_fff[2]        = fff[1] * x0;
    sub_fff[3]        = x1 * (L * x5 + L * (L * fff[3] + x3 + x6) + L * (L * fff[5] + x4 + x6));
    sub_fff[4]        = -x5 / x2;
    sub_fff[5]        = fff[0] * x0;
}

template <typename fff_t, typename scalar_t>
static inline __device__ __host__ void cu_sstet4_sub_fff_5(const int                        L,
                                                           const fff_t *const SFEM_RESTRICT fff,
                                                           scalar_t *const SFEM_RESTRICT    sub_fff) {
    const scalar_t x0 = 1. / POW3(L);
    const scalar_t x1 = L * fff[0];
    const scalar_t x2 = L * fff[2];
    const scalar_t x3 = L * (-x1 - x2);
    const scalar_t x4 = -L * fff[5] - x2;
    const scalar_t x5 = POW2(L);
    const scalar_t x6 = L * fff[1];
    sub_fff[0]        = x0 * (-L * x4 - x3);
    sub_fff[1]        = x4 / x5;
    sub_fff[2]        = x0 * (L * (-L * fff[4] - x6) + x3);
    sub_fff[3]        = fff[5] / L;
    sub_fff[4]        = x0 * (fff[2] * x5 + fff[4] * x5);
    sub_fff[5]        = x0 * (L * (x1 + x6) + L * (L * fff[3] + x6));
}

template <typename real_t>
__global__ void cu_sstet4_laplacian_apply_kernel(const int                                level,
                                                 const ptrdiff_t                          nelements,
                                                 const ptrdiff_t                          stride,
                                                 const cu_jacobian_t *const SFEM_RESTRICT g_fff,
                                                 const real_t *const SFEM_RESTRICT        u,
                                                 real_t *const SFEM_RESTRICT              values) {
    const int nxe = sstet4_nxe(level);
    const int txe = sstet4_txe(level);

    int           ev[4];
    scalar_t      fff[6];
    geom_t        offf[6];
    accumulator_t v[4];

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
#pragma unroll(6)
        for (int d = 0; d < 6; d++) {
            offf[d] = g_fff[d * stride + e];
        }

        const real_t *const element_u      = &u[e];
        real_t *const       element_vector = &values[e];

        ///////////////////////////////////
        // Cat 0
        ///////////////////////////////////
        {
            cu_sstet4_sub_fff_0(level, offf, fff);

            int p = 0;
            for (int i = 0; i < level - 1; i++) {
                int layer_items = (level - i + 1) * (level - i) / 2;
                for (int j = 0; j < level - i - 1; j++) {
                    for (int k = 0; k < level - i - j - 1; k++) {
                        ev[0] = p;
                        ev[1] = p + 1;
                        ev[2] = p + level - i - j;
                        ev[3] = p + layer_items - j;

                        cu_sstet4_laplacian_apply_fff(fff,
                                                      element_u[ev[0] * stride],
                                                      element_u[ev[1] * stride],
                                                      element_u[ev[2] * stride],
                                                      element_u[ev[3] * stride],
                                                      &v[0],
                                                      &v[1],
                                                      &v[2],
                                                      &v[3]);

                        for (int d = 0; d < 4; d++) {
                            element_vector[ev[d] * stride] += v[d];
                        }

                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        ///////////////////////////////////
        // Cat 1
        ///////////////////////////////////
        {
            cu_sstet4_sub_fff_1(level, offf, fff);

            int p = 0;
            for (int i = 0; i < level - 1; i++) {
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++) {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++) {
                        ev[0] = p;
                        ev[1] = p + layer_items + level - i - j - 1;
                        ev[2] = p + layer_items + level - i - j;
                        ev[3] = p + layer_items + level - i - j - 1 + level - i - j - 1;

                        cu_sstet4_laplacian_apply_fff(fff,
                                                      element_u[ev[0] * stride],
                                                      element_u[ev[1] * stride],
                                                      element_u[ev[2] * stride],
                                                      element_u[ev[3] * stride],
                                                      &v[0],
                                                      &v[1],
                                                      &v[2],
                                                      &v[3]);

                        for (int d = 0; d < 4; d++) {
                            element_vector[ev[d] * stride] += v[d];
                        }
                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        ///////////////////////////////////
        // Cat 2
        ///////////////////////////////////
        {
            cu_sstet4_sub_fff_2(level, offf, fff);

            int p = 0;
            for (int i = 0; i < level - 1; i++) {
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++) {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++) {
                        ev[0] = p;
                        ev[1] = p + level - i - j;
                        ev[3] = p + layer_items + level - i - j;
                        ev[2] = p + layer_items + level - i - j - 1 + level - i - j - 1;

                        cu_sstet4_laplacian_apply_fff(fff,
                                                      element_u[ev[0] * stride],
                                                      element_u[ev[1] * stride],
                                                      element_u[ev[2] * stride],
                                                      element_u[ev[3] * stride],
                                                      &v[0],
                                                      &v[1],
                                                      &v[2],
                                                      &v[3]);

                        for (int d = 0; d < 4; d++) {
                            element_vector[ev[d] * stride] += v[d];
                        }

                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        ///////////////////////////////////
        // Cat 3
        ///////////////////////////////////
        {
            cu_sstet4_sub_fff_3(level, offf, fff);

            int p = 0;
            for (int i = 0; i < level - 1; i++) {
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++) {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++) {
                        ev[0] = p;
                        ev[1] = p + level - i - j - 1;
                        ev[2] = p + layer_items + level - i - j - 1;
                        ev[3] = p + layer_items + level - i - j - 1 + level - i - j - 1;

                        cu_sstet4_laplacian_apply_fff(fff,
                                                      element_u[ev[0] * stride],
                                                      element_u[ev[1] * stride],
                                                      element_u[ev[2] * stride],
                                                      element_u[ev[3] * stride],
                                                      &v[0],
                                                      &v[1],
                                                      &v[2],
                                                      &v[3]);

                        for (int d = 0; d < 4; d++) {
                            element_vector[ev[d] * stride] += v[d];
                        }

                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        ///////////////////////////////////
        // Cat 4
        ///////////////////////////////////
        {
            cu_sstet4_sub_fff_4(level, offf, fff);

            int p = 0;
            for (int i = 1; i < level - 1; i++) {
                p               = p + level - i + 1;
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++) {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++) {
                        ev[0] = p;
                        ev[1] = p + layer_items + level - i;
                        ev[2] = p + layer_items + level - i - j + level - i;
                        ev[3] = p + layer_items + level - i - j + level - i - 1;

                        cu_sstet4_laplacian_apply_fff(fff,
                                                      element_u[ev[0] * stride],
                                                      element_u[ev[1] * stride],
                                                      element_u[ev[2] * stride],
                                                      element_u[ev[3] * stride],
                                                      &v[0],
                                                      &v[1],
                                                      &v[2],
                                                      &v[3]);

                        for (int d = 0; d < 4; d++) {
                            element_vector[ev[d] * stride] += v[d];
                        }

                        p++;
                    }
                    p++;
                }
                p++;
            }
        }

        ///////////////////////////////////
        // Cat 5
        ///////////////////////////////////
        {
            cu_sstet4_sub_fff_5(level, offf, fff);

            int p = 0;
            for (int i = 0; i < level - 1; i++) {
                int layer_items = (level - i) * (level - i - 1) / 2;
                for (int j = 0; j < level - i - 1; j++) {
                    p++;
                    for (int k = 1; k < level - i - j - 1; k++) {
                        ev[0] = p;
                        ev[1] = p + level - i - j - 1;
                        ev[2] = p + layer_items + level - i - j - 1 + level - i - j - 1;
                        ev[3] = p + level - i - j;

                        cu_sstet4_laplacian_apply_fff(fff,
                                                      element_u[ev[0] * stride],
                                                      element_u[ev[1] * stride],
                                                      element_u[ev[2] * stride],
                                                      element_u[ev[3] * stride],
                                                      &v[0],
                                                      &v[1],
                                                      &v[2],
                                                      &v[3]);

                        for (int d = 0; d < 4; d++) {
                            element_vector[ev[d] * stride] += v[d];
                        }

                        p++;
                    }
                    p++;
                }
                p++;
            }
        }
    }
}

template <typename T>
static int cu_sstet4_laplacian_apply_tpl(const int                                level,
                                         const ptrdiff_t                          nelements,
                                         const ptrdiff_t                          stride,
                                         const cu_jacobian_t *const SFEM_RESTRICT fff,
                                         const T *const SFEM_RESTRICT             x,
                                         T *const SFEM_RESTRICT                   y,
                                         void                                    *stream) {
    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_sstet4_laplacian_apply_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_sstet4_laplacian_apply_kernel<T><<<n_blocks, block_size, 0, s>>>(level, nelements, stride, fff, x, y);
    } else {
        cu_sstet4_laplacian_apply_kernel<T><<<n_blocks, block_size, 0>>>(level, nelements, stride, fff, x, y);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

extern int cu_sstet4_laplacian_apply(const int                       level,
                                     const ptrdiff_t                 nelements,
                                     const ptrdiff_t                 fff_stride,
                                     const void *const SFEM_RESTRICT fff,
                                     const enum RealType             real_type_xy,
                                     const void *const SFEM_RESTRICT x,
                                     void *const SFEM_RESTRICT       y,
                                     void                           *stream) {
    switch (real_type_xy) {
        case SFEM_REAL_DEFAULT: {
            return cu_sstet4_laplacian_apply_tpl(
                    level, nelements, fff_stride, (cu_jacobian_t *)fff, (real_t *)x, (real_t *)y, stream);
        }
        case SFEM_FLOAT32: {
            return cu_sstet4_laplacian_apply_tpl(level, nelements, fff_stride, (cu_jacobian_t *)fff, (float *)x, (float *)y, stream);
        }
        case SFEM_FLOAT64: {
            return cu_sstet4_laplacian_apply_tpl(
                    level, nelements, fff_stride, (cu_jacobian_t *)fff, (double *)x, (double *)y, stream);
        }
        default: {
            SFEM_ERROR(
                    "[Error] cu_sstet4_laplacian_apply: not implemented for type %s (code "
                    "%d)\n",
                    real_type_to_string(real_type_xy),
                    real_type_xy);
            return SFEM_FAILURE;
        }
    }
}
