#ifndef __SFEM_ADJOINT_MINI_TET_FUN_CUH__
#define __SFEM_ADJOINT_MINI_TET_FUN_CUH__

#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <cuda/atomic>
#include <set>

#include "quadratures_rule_cuda.cuh"
#include "sfem_adjoint_mini_tet_fun.cuh"
#include "sfem_config.h"
#include "sfem_resample_field_adjoint_hyteg.h"
#include "sfem_resample_field_cuda_fun.cuh"

// typedef float   geom_t;
// typedef int32_t idx_t;

namespace cg = cooperative_groups;

#define LANES_PER_TILE 32  // Number of lanes per tile (warp)
#define HYTEG_MAX_REFINEMENT_LEVEL 22

// typedef struct {
//     float        alpha_min_threshold;
//     float        alpha_max_threshold;
//     unsigned int min_refinement_L;
//     unsigned int max_refinement_L;
// } mini_tet_parameters_t;

// Float3 template class that provides type mapping for float3/double3
template <typename T>
struct Float3 {
    // Default implementation (intentionally left incomplete)
};

// Specialization for double
template <>
struct Float3<double> {
    using type = double3;

    __device__ static inline type make(double x, double y, double z) { return make_double3(x, y, z); }
};

// Specialization for float
template <>
struct Float3<float> {
    using type = float3;

    __device__ static inline type make(float x, float y, float z) { return make_float3(x, y, z); }
};

// Warp-aggregated atomicAdd to reduce contention (1 atomic per unique address in a warp)
template <typename T>
__device__ inline void warpAggregatedAdd(T* data, ptrdiff_t idx, T val) {
#if __CUDA_ARCH__ < 700

#pragma error "Warp-aggregated Add requires CUDA Compute Capability 7.0 or higher"

#else

    // #pragma message("Using warp-aggregated atomicAdd")

    const unsigned full_mask = __activemask();
    // Group lanes by target address
    unsigned long long key    = reinterpret_cast<unsigned long long>(data + idx);
    const unsigned     peers  = __match_any_sync(full_mask, key);
    const int          leader = __ffs(peers) - 1;
    const int          lane   = threadIdx.x & (warpSize - 1);

    // Leader accumulates contributions from its peer group using uniform shuffles
    T sum = val;
    for (int src = 0; src < warpSize; ++src) {
        // Everyone executes the same shuffle; only the leader of a group uses selected values
        T value = __shfl_sync(full_mask, val, src);
        if (lane == leader && (peers & (1u << src)) && src != leader) {
            sum += value;
        }
    }

    if (lane == leader) data[idx] += sum;

#endif
}

// Fast floor for float/double -> int (round down)
__device__ __forceinline__ int fast_floorf(float x) { return __float2int_rd(x); }
__device__ __forceinline__ int fast_floord(double x) { return __double2int_rd(x); }
template <typename T>
__device__ __forceinline__ int fast_floor(T x);
template <>
__device__ __forceinline__ int fast_floor<float>(float x) {
    return fast_floorf(x);
}
template <>
__device__ __forceinline__ int fast_floor<double>(double x) {
    return fast_floord(x);
}

// Fast ceil for float/double -> int (round up)
__device__ __forceinline__ int fast_ceilf(float x) { return __float2int_ru(x); }
__device__ __forceinline__ int fast_ceild(double x) { return __double2int_ru(x); }

template <typename T>
__device__ __forceinline__ int fast_ceil(T x);
template <>
__device__ __forceinline__ int fast_ceil<float>(float x) {
    return fast_ceilf(x);
}
template <>
__device__ __forceinline__ int fast_ceil<double>(double x) {
    return fast_ceild(x);
}

// Fast min/max for float/double
__device__ __forceinline__ float  fast_minf(float a, float b) { return a < b ? a : b; }
__device__ __forceinline__ double fast_mind(double a, double b) { return a < b ? a : b; }

template <typename T>
__device__ __forceinline__ T fast_min(T a, T b);
template <>
__device__ __forceinline__ float fast_min<float>(float a, float b) {
    return fast_minf(a, b);
}
template <>
__device__ __forceinline__ double fast_min<double>(double a, double b) {
    return fast_mind(a, b);
}

// Fast max for float/double
__device__ __forceinline__ float  fast_maxf(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ double fast_maxd(double a, double b) { return a > b ? a : b; }

template <typename T>
__device__ __forceinline__ T fast_max(T a, T b);
template <>
__device__ __forceinline__ float fast_max<float>(float a, float b) {
    return fast_maxf(a, b);
}
template <>
__device__ __forceinline__ double fast_max<double>(double a, double b) {
    return fast_maxd(a, b);
}

// Fast fused multiply-add for float/double
__device__ __forceinline__ float  fast_fmaf(float a, float b, float c) { return fmaf(a, b, c); }
__device__ __forceinline__ double fast_fmad(double a, double b, double c) { return fma(a, b, c); }
template <typename T>
__device__ __forceinline__ T fast_fma(T a, T b, T c);

template <>
__device__ __forceinline__ float fast_fma<float>(float a, float b, float c) {
    return fast_fmaf(a, b, c);
}
template <>
__device__ __forceinline__ double fast_fma<double>(double a, double b, double c) {
    return fast_fmad(a, b, c);
}

// Fast sqrt for float/double
__device__ __forceinline__ float  fast_sqrtf(float x) { return sqrtf(x); }
__device__ __forceinline__ double fast_sqrtd(double x) { return sqrt(x); }
template <typename T>
__device__ __forceinline__ T fast_sqrt(T x);
template <>
__device__ __forceinline__ float fast_sqrt<float>(float x) {
    return fast_sqrtf(x);
}
template <>
__device__ __forceinline__ double fast_sqrt<double>(double x) {
    return fast_sqrtd(x);
}

// Fast abs for float/double
__device__ __forceinline__ float  fast_absf(float x) { return fabsf(x); }
__device__ __forceinline__ double fast_absd(double x) { return fabs(x); }

template <typename T>
T __device__ __forceinline__ fast_abs(T x);
template <>
__device__ __forceinline__ float fast_abs<float>(float x) {
    return fast_absf(x);
}
template <>
__device__ __forceinline__ double fast_abs<double>(double x) {
    return fast_absd(x);
}

////////////////////////////////////////////////////////////////////////////////
// Compute matrix J_phys * J_ref = J_tot
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __inline__ void                                                         //
compute_matrix_mult_3x3_gpu(const typename Float3<FloatType>::type* const J_phys,  // Jacobian matrix of the physical tetrahedron
                            const typename Float3<FloatType>::type* const J_ref,   // Jacobian matrix of the reference tetrahedron
                            typename Float3<FloatType>::type*             J_tot) {             // Output Jacobian matrix
    // Row 1
    J_tot[0].x = fast_fma(J_phys[0].y, J_ref[1].x, J_phys[0].x * J_ref[0].x);
    J_tot[0].x = fast_fma(J_phys[0].z, J_ref[2].x, J_tot[0].x);
    J_tot[0].y = fast_fma(J_phys[0].y, J_ref[1].y, J_phys[0].x * J_ref[0].y);
    J_tot[0].y = fast_fma(J_phys[0].z, J_ref[2].y, J_tot[0].y);
    J_tot[0].z = fast_fma(J_phys[0].y, J_ref[1].z, J_phys[0].x * J_ref[0].z);
    J_tot[0].z = fast_fma(J_phys[0].z, J_ref[2].z, J_tot[0].z);

    // Row 2
    J_tot[1].x = fast_fma(J_phys[1].y, J_ref[1].x, J_phys[1].x * J_ref[0].x);
    J_tot[1].x = fast_fma(J_phys[1].z, J_ref[2].x, J_tot[1].x);
    J_tot[1].y = fast_fma(J_phys[1].y, J_ref[1].y, J_phys[1].x * J_ref[0].y);
    J_tot[1].y = fast_fma(J_phys[1].z, J_ref[2].y, J_tot[1].y);
    J_tot[1].z = fast_fma(J_phys[1].y, J_ref[1].z, J_phys[1].x * J_ref[0].z);
    J_tot[1].z = fast_fma(J_phys[1].z, J_ref[2].z, J_tot[1].z);

    // Row 3
    J_tot[2].x = fast_fma(J_phys[2].y, J_ref[1].x, J_phys[2].x * J_ref[0].x);
    J_tot[2].x = fast_fma(J_phys[2].z, J_ref[2].x, J_tot[2].x);
    J_tot[2].y = fast_fma(J_phys[2].y, J_ref[1].y, J_phys[2].x * J_ref[0].y);
    J_tot[2].y = fast_fma(J_phys[2].z, J_ref[2].y, J_tot[2].y);
    J_tot[2].z = fast_fma(J_phys[2].y, J_ref[1].z, J_phys[2].x * J_ref[0].z);
    J_tot[2].z = fast_fma(J_phys[2].z, J_ref[2].z, J_tot[2].z);
    return;
}

////////////////////////////////////////////////////////////////////////////////
// Function to get the Jacobian matrix for a given category
// get_category_Jacobian
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __forceinline__ bool                                        //
get_category_Jacobian(const unsigned int                category,      //
                      const FloatType                   L,             //
                      typename Float3<FloatType>::type* Jacobian_c) {  //

    const FloatType invL = FloatType(1.0) / L;
    const FloatType zero = FloatType(0.0);

    switch (category) {
        case 0:
            Jacobian_c[0] = Float3<FloatType>::make(invL, zero, zero);
            Jacobian_c[1] = Float3<FloatType>::make(zero, invL, zero);
            Jacobian_c[2] = Float3<FloatType>::make(zero, zero, invL);
            break;

        case 1:
            Jacobian_c[0] = Float3<FloatType>::make(zero, -invL, -invL);
            Jacobian_c[1] = Float3<FloatType>::make(zero, invL, zero);
            Jacobian_c[2] = Float3<FloatType>::make(invL, invL, invL);
            break;

        case 2:
            Jacobian_c[0] = Float3<FloatType>::make(-invL, zero, zero);
            Jacobian_c[1] = Float3<FloatType>::make(invL, zero, invL);
            Jacobian_c[2] = Float3<FloatType>::make(invL, invL, zero);
            break;

        case 3:
            Jacobian_c[0] = Float3<FloatType>::make(-invL, -invL, -invL);
            Jacobian_c[1] = Float3<FloatType>::make(zero, invL, invL);
            Jacobian_c[2] = Float3<FloatType>::make(invL, invL, zero);
            break;

        case 4:
            Jacobian_c[0] = Float3<FloatType>::make(-invL, -invL, zero);
            Jacobian_c[1] = Float3<FloatType>::make(invL, invL, invL);
            Jacobian_c[2] = Float3<FloatType>::make(zero, invL, zero);
            break;

        case 5:
            Jacobian_c[0] = Float3<FloatType>::make(zero, zero, -invL);
            Jacobian_c[1] = Float3<FloatType>::make(zero, -invL, zero);
            Jacobian_c[2] = Float3<FloatType>::make(invL, invL, invL);
            break;

        default:
            __trap();
            return false;
            break;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Function to compute the Jacobian matrix and its determinant for a tetrahedron
// make_Jocobian_matrix_tet_cu
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __forceinline__ FloatType                                 //
make_Jacobian_matrix_tet_gpu(const FloatType                   fx0,  // Tetrahedron vertices X-coordinates
                             const FloatType                   fx1,  //
                             const FloatType                   fx2,  //
                             const FloatType                   fx3,  //
                             const FloatType                   fy0,  // Tetrahedron vertices Y-coordinates
                             const FloatType                   fy1,  //
                             const FloatType                   fy2,  //
                             const FloatType                   fy3,  //
                             const FloatType                   fz0,  // Tetrahedron vertices Z-coordinates
                             const FloatType                   fz1,  //
                             const FloatType                   fz2,  //
                             const FloatType                   fz3,
                             typename Float3<FloatType>::type* J) {  // Jacobian matrix

    J[0] = Float3<FloatType>::make(fx1 - fx0, fx2 - fx0, fx3 - fx0);
    J[1] = Float3<FloatType>::make(fy1 - fy0, fy2 - fy0, fy3 - fy0);
    J[2] = Float3<FloatType>::make(fz1 - fz0, fz2 - fz0, fz3 - fz0);

    // FMA-accelerated evaluation
    const FloatType m00 = fast_fma(J[1].y, J[2].z, -(J[1].z * J[2].y));  // J[1].y*J[2].z - J[1].z*J[2].y
    const FloatType m01 = fast_fma(J[1].x, J[2].z, -(J[1].z * J[2].x));  // J[1].x*J[2].z - J[1].z*J[2].x
    const FloatType m02 = fast_fma(J[1].x, J[2].y, -(J[1].y * J[2].x));  // J[1].x*J[2].y - J[1].y*J[2].x
    const FloatType det = fast_fma(J[0].x, m00, fast_fma(-J[0].y, m01, J[0].z * m02));

    return det;
}

////////////////////////////////////////////////////////////////////////////////
// Function to evaluate the 8 trilinear shape functions of a hexahedron
// hex_aa_8_eval_fun_T_cu
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>                //
__device__ void                              //
hex_aa_8_eval_fun_T_gpu(const FloatType x,   // Local coordinates (in the unit cube)
                        const FloatType y,   //
                        const FloatType z,   //
                        FloatType&      f0,  // Output
                        FloatType&      f1,  //
                        FloatType&      f2,  //
                        FloatType&      f3,  //
                        FloatType&      f4,  //
                        FloatType&      f5,  //
                        FloatType&      f6,  //
                        FloatType&      f7) {
    const FloatType one = FloatType(1.0);
    const FloatType mx  = one - x;
    const FloatType my  = one - y;
    const FloatType mz  = one - z;
    f0                  = mx * my * mz;
    f1                  = x * my * mz;
    f2                  = x * y * mz;
    f3                  = mx * y * mz;
    f4                  = mx * my * z;
    f5                  = x * my * z;
    f6                  = x * y * z;
    f7                  = mx * y * z;
}

////////////////////////////////////////////////////////////////////////////////
// Function to collect the indices of the 8 vertices of a hexahedron
// hex_aa_8_collect_coeffs_indices_cu
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ void                               //
hex_aa_8_collect_coeffs_indices_gpu(const ptrdiff_t stride0,  // Stride
                                    const ptrdiff_t stride1,  //
                                    const ptrdiff_t stride2,  //
                                    const ptrdiff_t i,        // Indices of the element
                                    const ptrdiff_t j,        //
                                    const ptrdiff_t k,        //
                                    ptrdiff_t&      i0,       //
                                    ptrdiff_t&      i1,       //
                                    ptrdiff_t&      i2,       //
                                    ptrdiff_t&      i3,       //
                                    ptrdiff_t&      i4,       //
                                    ptrdiff_t&      i5,       //
                                    ptrdiff_t&      i6,       //
                                    ptrdiff_t&      i7) {          //

    const ptrdiff_t i_plus_1 = i + 1;
    const ptrdiff_t j_plus_1 = j + 1;
    const ptrdiff_t k_plus_1 = k + 1;

    i0 = i * stride0 + j * stride1 + k * stride2;
    i1 = i_plus_1 * stride0 + j * stride1 + k * stride2;
    i2 = i_plus_1 * stride0 + j_plus_1 * stride1 + k * stride2;
    i3 = i * stride0 + j_plus_1 * stride1 + k * stride2;
    i4 = i * stride0 + j * stride1 + k_plus_1 * stride2;
    i5 = i_plus_1 * stride0 + j * stride1 + k_plus_1 * stride2;
    i6 = i_plus_1 * stride0 + j_plus_1 * stride1 + k_plus_1 * stride2;
    i7 = i * stride0 + j_plus_1 * stride1 + k_plus_1 * stride2;
}

////////////////////////////////////////////////////////////////////////////////
// Function to compute the distance between two points in 3D space
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __forceinline__ FloatType       //
points_distance_gpu(const FloatType x0,    //
                    const FloatType y0,    //
                    const FloatType z0,    //
                    const FloatType x1,    //
                    const FloatType y1,    //
                    const FloatType z1) {  //

    const FloatType dx = x1 - x0;
    const FloatType dy = y1 - y0;
    const FloatType dz = z1 - z0;

    // dx*dx + dy*dy + dz*dz using fused multiply-adds
    const FloatType sum = fast_fma(dx, dx, fast_fma(dy, dy, dz * dz));
    return fast_sqrt(sum);
}

////////////////////////////////////////////////////////////////////////////////
// Function to compute the maximum edge length of a tetrahedron
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __forceinline__ FloatType                     //
tet_edge_max_length_gpu(const FloatType  v0x,            //
                        const FloatType  v0y,            //
                        const FloatType  v0z,            //
                        const FloatType  v1x,            //
                        const FloatType  v1y,            //
                        const FloatType  v1z,            //
                        const FloatType  v2x,            //
                        const FloatType  v2y,            //
                        const FloatType  v2z,            //
                        const FloatType  v3x,            //
                        const FloatType  v3y,            //
                        const FloatType  v3z,            //
                        int*             vertex_a,       //
                        int*             vertex_b,       //
                        FloatType* const edge_length) {  //

    FloatType max_length = 0.0;

    // Edge 0 (v0, v1)
    edge_length[0] = points_distance_gpu(v0x, v0y, v0z, v1x, v1y, v1z);
    if (edge_length[0] > max_length) {
        max_length = edge_length[0];
        *vertex_a  = 0;
        *vertex_b  = 1;
    }

    // Edge 1 (v0, v2)
    edge_length[1] = points_distance_gpu(v0x, v0y, v0z, v2x, v2y, v2z);
    if (edge_length[1] > max_length) {
        max_length = edge_length[1];
        *vertex_a  = 0;
        *vertex_b  = 2;
    }

    // Edge 2 (v0, v3)
    edge_length[2] = points_distance_gpu(v0x, v0y, v0z, v3x, v3y, v3z);
    if (edge_length[2] > max_length) {
        max_length = edge_length[2];
        *vertex_a  = 0;
        *vertex_b  = 3;
    }

    // Edge 3 (v1, v2)
    edge_length[3] = points_distance_gpu(v1x, v1y, v1z, v2x, v2y, v2z);
    if (edge_length[3] > max_length) {
        max_length = edge_length[3];
        *vertex_a  = 1;
        *vertex_b  = 2;
    }

    // Edge 4 (v1, v3)
    edge_length[4] = points_distance_gpu(v1x, v1y, v1z, v3x, v3y, v3z);
    if (edge_length[4] > max_length) {
        max_length = edge_length[4];
        *vertex_a  = 1;
        *vertex_b  = 3;
    }

    // Edge 5 (v2, v3)
    edge_length[5] = points_distance_gpu(v2x, v2y, v2z, v3x, v3y, v3z);
    if (edge_length[5] > max_length) {
        max_length = edge_length[5];
        *vertex_a  = 2;
        *vertex_b  = 3;
    }

    return max_length;
}

/////////////////////////////////////////////////////////////////////////////////
// Function to map alpha to HYTEG refinement level
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __forceinline__ int                                    //
alpha_to_hyteg_level_gpu(const FloatType    alpha,                //
                         const FloatType    alpha_min_threshold,  //
                         const FloatType    alpha_max_threshold,  //
                         const unsigned int min_refinement_L,     //
                         const unsigned int max_refinement_L) {   //

    // return 1;  ///// TODO

    // const int min_refinement_L = 2;  // Minimum refinement level

    if (alpha < alpha_min_threshold) return min_refinement_L;  // No refinement
    if (alpha > alpha_max_threshold) return max_refinement_L;  // Maximum refinement

    const FloatType alpha_x = alpha - alpha_min_threshold;  // Shift the alpha to start from 0
    const FloatType L_real =
            (alpha_x / (alpha_max_threshold - alpha_min_threshold) * (FloatType)(HYTEG_MAX_REFINEMENT_LEVEL - 1)) + 1.0;

    int L = L_real >= FloatType(1.0) ? (int)L_real : min_refinement_L;        // Convert to integer
    L     = L > HYTEG_MAX_REFINEMENT_LEVEL ? HYTEG_MAX_REFINEMENT_LEVEL : L;  // Clamp to maximum level

    const int ret = L >= max_refinement_L ? max_refinement_L : L;
    return (ret) < min_refinement_L ? min_refinement_L : (ret);  // Ensure L is within bounds
}

/////////////////////////////////////////////////////////////////////////////////
// Resampling function for a mini-tetrahedron and a given category
/////////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ __forceinline__ void store_add(T* dst, T v) {
// #define SFEM_NON_ATOMIC_UPDATES
#if defined(SFEM_NON_ATOMIC_UPDATES)
    *dst += v;  // non-atomic update (use only if no overlaps)
#else
    atomicAdd(dst, v);  // default: safe atomic add
#endif
}

/////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
struct __align__(16) buffer_cluster_t {
    FloatType* buffer = nullptr;
    ptrdiff_t  size   = 0;
};

template <typename FloatType>
cudaError_t clear_buffer_cluster_async(buffer_cluster_t<FloatType>& buffer_cluster, cudaStream_t stream) {
    if (buffer_cluster.buffer) {
        cudaError_t error = cudaFreeAsync(buffer_cluster.buffer, stream);
        if (error != cudaSuccess) {
            printf("CUDA error in clear_buffer_cluster_async: %s\n", cudaGetErrorString(error));
            return error;
        }
        buffer_cluster.buffer = nullptr;
        buffer_cluster.size   = 0;
    }
    return cudaSuccess;
}

template <typename FloatType>
cudaError_t allocate_buffer_cluster(buffer_cluster_t<FloatType>& buffer_cluster, ptrdiff_t size, cudaStream_t stream,
                                    bool force_realloc = false) {
    if (force_realloc && buffer_cluster.buffer != nullptr) {
        clear_buffer_cluster_async(buffer_cluster, stream);
    }

    if (size > buffer_cluster.size || force_realloc) {
        cudaError_t error;
        error = cudaMallocAsync((void**)&buffer_cluster.buffer, size * sizeof(FloatType), stream);
        if (error != cudaSuccess) {
            printf("CUDA error in allocate_buffer_cluster: %s\n", cudaGetErrorString(error));
            return error;
        }
        buffer_cluster.size = size;
    }
    return cudaSuccess;
}

template <typename FloatType>
cudaError_t clear_buffer_cluster(buffer_cluster_t<FloatType>& buffer_cluster) {
    if (buffer_cluster.buffer) {
        cudaError_t error = cudaFree(buffer_cluster.buffer);
        if (error != cudaSuccess) {
            printf("CUDA error in clear_buffer_cluster: %s\n", cudaGetErrorString(error));
            return error;
        }
        buffer_cluster.buffer = nullptr;
        buffer_cluster.size   = 0;
    }
    return cudaSuccess;
}

template <typename FloatType>
bool is_buffer_valid(const buffer_cluster_t<FloatType>& buffer_cluster, ptrdiff_t required_size = 0) {
    if (!buffer_cluster.buffer) {
        return false;
    }

    if (required_size > 0 && buffer_cluster.size < required_size) {
        return false;
    }

    return true;
}

template <typename FloatType>
cudaError_t reset_buffer_values(buffer_cluster_t<FloatType>& buffer_cluster, FloatType value, cudaStream_t stream) {
    if (!buffer_cluster.buffer) {
        return cudaErrorInvalidDevicePointer;
    }

    return cudaMemsetAsync(buffer_cluster.buffer, value, buffer_cluster.size * sizeof(FloatType), stream);
}

#endif  // __SFEM_ADJOINT_MINI_TET_FUN_CUH__