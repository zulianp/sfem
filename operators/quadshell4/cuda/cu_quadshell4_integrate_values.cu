#include "cu_quadshell4_integrate_values.h"

#include "sfem_cuda_base.h"
#include "sfem_macros.h"

#include <cassert>
#include <cstdio>

template <typename T>
static inline __device__ __host__ void cu_quadshell_4_integrate(const T  px0,
                                                                const T  px1,
                                                                const T  px2,
                                                                const T  px3,
                                                                const T  py0,
                                                                const T  py1,
                                                                const T  py2,
                                                                const T  py3,
                                                                const T  pz0,
                                                                const T  pz1,
                                                                const T  pz2,
                                                                const T  pz3,
                                                                const T  val,
                                                                T *const element_vector) {
    static const T   rule_qx[4] = {0.211324865405187, 0.788675134594813, 0.211324865405187, 0.788675134594813};
    static const T   rule_qy[4] = {0.211324865405187, 0.211324865405187, 0.788675134594813, 0.788675134594813};
    static const T   rule_qw[4] = {0.25, 0.25, 0.25, 0.25};
    static const int rule_n_qp  = 4;

    element_vector[0] = 0;
    element_vector[1] = 0;
    element_vector[2] = 0;
    element_vector[3] = 0;

    for (int q = 0; q < rule_n_qp; q++) {
        const T qx = rule_qx[q];
        const T qy = rule_qy[q];
        const T qw = rule_qw[q];

        const T x0 = qx - 1;
        const T x1 = -x0;
        const T x2 = qy - 1;
        const T x3 = -x2;
        const T x4 = px0 * x0 - px1 * qx + px2 * qx + px3 * x1;
        const T x5 = px0 * x2 + px1 * x3 + px2 * qy - px3 * qy;
        const T x6 = py0 * x0 - py1 * qx + py2 * qx + py3 * x1;
        const T x7 = py0 * x2 + py1 * x3 + py2 * qy - py3 * qy;
        const T x8 = pz0 * x0 - pz1 * qx + pz2 * qx + pz3 * x1;
        const T x9 = pz0 * x2 + pz1 * x3 + pz2 * qy - pz3 * qy;
        const T x10 =
                qw * val *
                sqrt((POW2(x4) + POW2(x6) + POW2(x8)) * (POW2(x5) + POW2(x7) + POW2(x9)) - POW2(x4 * x5 + x6 * x7 + x8 * x9));
        const T x11 = x10 * x3;
        const T x12 = qy * x10;
        element_vector[0] += x1 * x11;
        element_vector[1] += qx * x11;
        element_vector[2] += qx * x12;
        element_vector[3] += x1 * x12;
    }
}

template <typename T>
__global__ void cu_quadshell4_integrate_value_kernel(const ptrdiff_t                    nelements,
                                                     const ptrdiff_t                    stride,  // Stride for elements and coords
                                                     const idx_t *const SFEM_RESTRICT   elements,
                                                     const geom_t **const SFEM_RESTRICT coords,
                                                     const real_t                       value,
                                                     const int                          block_size,
                                                     const int                          component,
                                                     T *const SFEM_RESTRICT             out) {
    const geom_t *const x = coords[0];
    const geom_t *const y = coords[1];
    const geom_t *const z = coords[2];

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        T element_vector[4];

        const geom_t *const xe = &x[e];
        const geom_t *const ye = &y[e];
        const geom_t *const ze = &z[e];

        cu_quadshell_4_integrate<T>(
                // X
                xe[0 * stride],
                xe[1 * stride],
                xe[2 * stride],
                xe[3 * stride],
                // Y
                ye[0 * stride],
                ye[1 * stride],
                ye[2 * stride],
                ye[3 * stride],
                // Z
                ze[0 * stride],
                ze[1 * stride],
                ze[2 * stride],
                ze[3 * stride],
                value,
                element_vector);

        for (int v = 0; v < 4; v++) {
            const ptrdiff_t idx = elements[v * stride + e];
            atomicAdd(&out[idx * block_size + component], element_vector[v]);
        }
    }
}

template <typename T>
__global__ void cu_quadshell4_integrate_values_kernel(const ptrdiff_t                  nelements,
                                                      const ptrdiff_t                  stride,  // Stride for elements and coords
                                                      const idx_t *const SFEM_RESTRICT elements,
                                                      const geom_t **const SFEM_RESTRICT coords,
                                                      const T *const SFEM_RESTRICT       values,
                                                      const int                          block_size,
                                                      const int                          component,
                                                      T *const SFEM_RESTRICT             out) {
    const geom_t *const x = coords[0];
    const geom_t *const y = coords[1];
    const geom_t *const z = coords[2];

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        T element_vector[4];

        const geom_t *const xe = &x[e];
        const geom_t *const ye = &y[e];
        const geom_t *const ze = &z[e];

        cu_quadshell_4_integrate<T>(
                // X
                xe[0 * stride],
                xe[1 * stride],
                xe[2 * stride],
                xe[3 * stride],
                // Y
                ye[0 * stride],
                ye[1 * stride],
                ye[2 * stride],
                ye[3 * stride],
                // Z
                ze[0 * stride],
                ze[1 * stride],
                ze[2 * stride],
                ze[3 * stride],
                values[e],
                element_vector);

        for (int v = 0; v < 4; v++) {
            const ptrdiff_t idx = elements[v * stride + e];
            atomicAdd(&out[idx * block_size + component], element_vector[v]);
        }
    }
}

template <typename T>
int cu_quadshell4_integrate_value_tpl(const ptrdiff_t                    nelements,
                                      const ptrdiff_t                    stride,  // Stride for elements and coords
                                      const idx_t *const SFEM_RESTRICT   elements,
                                      const geom_t **const SFEM_RESTRICT coords,
                                      const real_t                       value,
                                      const int                          vec_size,
                                      const int                          component,
                                      T *const SFEM_RESTRICT             out,
                                      void                              *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_quadshell4_integrate_value_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_quadshell4_integrate_value_kernel<T>
                <<<n_blocks, block_size, 0, s>>>(nelements, stride, elements, coords, value, vec_size, component, out);
    } else {
        cu_quadshell4_integrate_value_kernel<T>
                <<<n_blocks, vec_size, 0>>>(nelements, stride, elements, coords, value, block_size, component, out);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

template <typename T>
int cu_quadshell4_integrate_values_tpl(const ptrdiff_t                    nelements,
                                      const ptrdiff_t                    stride,  // Stride for elements and coords
                                      const idx_t *const SFEM_RESTRICT   elements,
                                      const geom_t **const SFEM_RESTRICT coords,
                                      const T * const SFEM_RESTRICT                       values,
                                      const int                          vec_size,
                                      const int                          component,
                                      T *const SFEM_RESTRICT             out,
                                      void                              *stream) {
    SFEM_DEBUG_SYNCHRONIZE();

    // Hand tuned
    int block_size = 128;
#ifdef SFEM_USE_OCCUPANCY_MAX_POTENTIAL
    {
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_quadshell4_integrate_value_kernel<T>, 0, 0);
    }
#endif  // SFEM_USE_OCCUPANCY_MAX_POTENTIAL

    ptrdiff_t n_blocks = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);

    if (stream) {
        cudaStream_t s = *static_cast<cudaStream_t *>(stream);

        cu_quadshell4_integrate_values_kernel<T>
                <<<n_blocks, block_size, 0, s>>>(nelements, stride, elements, coords, values, vec_size, component, out);
    } else {
        cu_quadshell4_integrate_values_kernel<T>
                <<<n_blocks, vec_size, 0>>>(nelements, stride, elements, coords, values, block_size, component, out);
    }

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}


extern int cu_quadshell4_integrate_value(const ptrdiff_t                    nelements,
                                         const ptrdiff_t                    stride,  // Stride for elements and coords
                                         const idx_t *const SFEM_RESTRICT   elements,
                                         const geom_t **const SFEM_RESTRICT coords,
                                         const real_t                       value,
                                         const int                          vec_size,
                                         const int                          component,
                                         const enum RealType                real_type,
                                         void *const SFEM_RESTRICT          out,
                                         void                              *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_quadshell4_integrate_value_tpl<real_t>(
                    nelements, stride, elements, coords, value, vec_size, component, (real_t *)out, stream);
        }
        case SFEM_FLOAT32: {
            return cu_quadshell4_integrate_value_tpl<float>(
                    nelements, stride, elements, coords, value, vec_size, component, (float *)out, stream);
        }
        case SFEM_FLOAT64: {
            return cu_quadshell4_integrate_value_tpl<double>(
                    nelements, stride, elements, coords, value, vec_size, component, (double *)out, stream);
        }
        default: {
            SFEM_ERROR(
                    "[Error] cu_quadshell4_integrate_value: not implemented "
                    "for "
                    "type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}

extern int cu_quadshell4_integrate_values(const ptrdiff_t                    nelements,
                                          const ptrdiff_t                    stride,  // Stride for elements and coords
                                          const idx_t *const SFEM_RESTRICT   elements,
                                          const geom_t **const SFEM_RESTRICT coords,
                                          const enum RealType                real_type,
                                          void *const SFEM_RESTRICT          values,
                                          const int                          vec_size,
                                          const int                          component,
                                          void *const SFEM_RESTRICT          out,
                                          void                              *stream) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_quadshell4_integrate_values_tpl<real_t>(
                    nelements, stride, elements, coords, (real_t *)values, vec_size, component, (real_t *)out, stream);
        }
        case SFEM_FLOAT32: {
            return cu_quadshell4_integrate_values_tpl<float>(
                    nelements, stride, elements, coords, (float *)values, vec_size, component, (float *)out, stream);
        }
        case SFEM_FLOAT64: {
            return cu_quadshell4_integrate_values_tpl<double>(
                    nelements, stride, elements, coords, (double *)values, vec_size, component, (double *)out, stream);
        }
        default: {
            SFEM_ERROR(
                    "[Error] cu_quadshell4_integrate_values: not implemented "
                    "for "
                    "type %s "
                    "(code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            return SFEM_FAILURE;
        }
    }
}
