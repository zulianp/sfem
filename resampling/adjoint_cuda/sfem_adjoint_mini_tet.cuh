#ifndef __SFEM_ADJOINT_MINI_TET_CUH__
#define __SFEM_ADJOINT_MINI_TET_CUH__

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <cuda/atomic>

#include "quadratures_rule_cuda.cuh"
#include "sfem_config.h"
#include "sfem_resample_field_adjoint_hyteg.h"
#include "sfem_resample_field_cuda_fun.cuh"

// typedef float   geom_t;
// typedef int32_t idx_t;

namespace cg = cooperative_groups;

#define LANES_PER_TILE 32
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

////////////////////////////////////////////////////////////////////////////////
// Function to get the Jacobian matrix for a given category
// get_category_Jacobian
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ bool get_category_c_Jacobian(const unsigned int category,  //
                                        const FloatType    L,         //
                                        FloatType*         Jacobian_c) {      //

    const FloatType invL = FloatType(1.0) / FloatType(L);
    const FloatType zero = FloatType(0.0);

    switch (category) {
        case 0:
            // Row 0: indices 0,1,2
            Jacobian_c[0] = invL;
            Jacobian_c[1] = zero;
            Jacobian_c[2] = zero;
            // Row 1: indices 3,4,5
            Jacobian_c[3] = zero;
            Jacobian_c[4] = invL;
            Jacobian_c[5] = zero;
            // Row 2: indices 6,7,8
            Jacobian_c[6] = zero;
            Jacobian_c[7] = zero;
            Jacobian_c[8] = invL;
            break;

        case 1:
            // Row 0: indices 0,1,2
            Jacobian_c[0] = zero;
            Jacobian_c[1] = -invL;
            Jacobian_c[2] = -invL;
            // Row 1: indices 3,4,5
            Jacobian_c[3] = zero;
            Jacobian_c[4] = invL;
            Jacobian_c[5] = zero;
            // Row 2: indices 6,7,8
            Jacobian_c[6] = invL;
            Jacobian_c[7] = invL;
            Jacobian_c[8] = invL;
            break;

        case 2:
            // Row 0: indices 0,1,2
            Jacobian_c[0] = -invL;
            Jacobian_c[1] = zero;
            Jacobian_c[2] = zero;
            // Row 1: indices 3,4,5
            Jacobian_c[3] = invL;
            Jacobian_c[4] = zero;
            Jacobian_c[5] = invL;
            // Row 2: indices 6,7,8
            Jacobian_c[6] = invL;
            Jacobian_c[7] = invL;
            Jacobian_c[8] = zero;
            break;

        case 3:
            // Row 0: indices 0,1,2
            Jacobian_c[0] = -invL;
            Jacobian_c[1] = -invL;
            Jacobian_c[2] = -invL;
            // Row 1: indices 3,4,5
            Jacobian_c[3] = zero;
            Jacobian_c[4] = invL;
            Jacobian_c[5] = invL;
            // Row 2: indices 6,7,8
            Jacobian_c[6] = invL;
            Jacobian_c[7] = invL;
            Jacobian_c[8] = zero;
            break;

        case 4:
            // Row 0: indices 0,1,2
            Jacobian_c[0] = -invL;
            Jacobian_c[1] = -invL;
            Jacobian_c[2] = zero;
            // Row 1: indices 3,4,5
            Jacobian_c[3] = invL;
            Jacobian_c[4] = invL;
            Jacobian_c[5] = invL;
            // Row 2: indices 6,7,8
            Jacobian_c[6] = zero;
            Jacobian_c[7] = invL;
            Jacobian_c[8] = zero;
            break;

        case 5:
            // Row 0: indices 0,1,2
            Jacobian_c[0] = zero;
            Jacobian_c[1] = zero;
            Jacobian_c[2] = -invL;
            // Row 1: indices 3,4,5
            Jacobian_c[3] = zero;
            Jacobian_c[4] = -invL;
            Jacobian_c[5] = zero;
            // Row 2: indices 6,7,8
            Jacobian_c[6] = invL;
            Jacobian_c[7] = invL;
            Jacobian_c[8] = invL;
            break;

        default:
            __trap();
            return false;
            break;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Function to get the Jacobian matrix for a given category
// get_category_Jacobian
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ bool get_category_Jacobian(const unsigned int                category,      //
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
__device__ FloatType                                 //
make_Jacobian_matrix_tet_c_gpu(const FloatType fx0,  // Tetrahedron vertices X-coordinates
                               const FloatType fx1,  //
                               const FloatType fx2,  //
                               const FloatType fx3,  //
                               const FloatType fy0,  // Tetrahedron vertices Y-coordinates
                               const FloatType fy1,  //
                               const FloatType fy2,  //
                               const FloatType fy3,  //
                               const FloatType fz0,  // Tetrahedron vertices Z-coordinates
                               const FloatType fz1,  //
                               const FloatType fz2,  //
                               const FloatType fz3,
                               FloatType*      J) {  // Jacobian matrix
    // Compute the Jacobian matrix for tetrahedron transformation
    // J = [x1-x0, x2-x0, x3-x0]   <- Row 0: indices 0,1,2
    //     [y1-y0, y2-y0, y3-y0]   <- Row 1: indices 3,4,5
    //     [z1-z0, z2-z0, z3-z0]   <- Row 2: indices 6,7,8

    // Row 0: x-components (indices 0,1,2)
    J[0] = fx1 - fx0;  // dx/dxi
    J[1] = fx2 - fx0;  // dx/deta
    J[2] = fx3 - fx0;  // dx/dzeta

    // J[0] = Float3<FloatType>::make(fx1 - fx0, fx2 - fx0, fx3 - fx0);

    // Row 1: y-components (indices 3,4,5)
    J[3] = fy1 - fy0;  // dy/dxi
    J[4] = fy2 - fy0;  // dy/deta
    J[5] = fy3 - fy0;  // dy/dzeta

    // J[1] = Float3<FloatType>::make(fy1 - fy0, fy2 - fy0, fy3 - fy0);

    // Row 2: z-components (indices 6,7,8)
    J[6] = fz1 - fz0;  // dz/dxi
    J[7] = fz2 - fz0;  // dz/deta
    J[8] = fz3 - fz0;  // dz/dzeta

    // J[2] = Float3<FloatType>::make(fz1 - fz0, fz2 - fz0, fz3 - fz0);

    // Compute determinant of the 3x3 Jacobian matrix
    const FloatType det = J[0] * (J[4] * J[8] - J[5] * J[7]) -  //
                          J[1] * (J[3] * J[8] - J[5] * J[6]) +  //
                          J[2] * (J[3] * J[7] - J[4] * J[6]);   //

    // const FloatType det = J[0].x * (J[1].y * J[2].z - J[1].z * J[2].y) -  //
    //                       J[0].y * (J[1].x * J[2].z - J[1].z * J[2].x) +  //
    //                       J[0].z * (J[1].x * J[2].y - J[1].y * J[2].x);   //

    return det;
}

////////////////////////////////////////////////////////////////////////////////
// Function to compute the Jacobian matrix and its determinant for a tetrahedron
// make_Jocobian_matrix_tet_cu
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ FloatType                                                 //
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
    // Compute the Jacobian matrix for tetrahedron transformation
    // J = [x1-x0, x2-x0, x3-x0]   <- Row 0: indices 0,1,2
    //     [y1-y0, y2-y0, y3-y0]   <- Row 1: indices 3,4,5
    //     [z1-z0, z2-z0, z3-z0]   <- Row 2: indices 6,7,8

    // Row 0: x-components (indices 0,1,2)
    // J[0] = fx1 - fx0;  // dx/dxi
    // J[1] = fx2 - fx0;  // dx/deta
    // J[2] = fx3 - fx0;  // dx/dzeta

    J[0] = Float3<FloatType>::make(fx1 - fx0, fx2 - fx0, fx3 - fx0);

    // Row 1: y-components (indices 3,4,5)
    // J[3] = fy1 - fy0;  // dy/dxi
    // J[4] = fy2 - fy0;  // dy/deta
    // J[5] = fy3 - fy0;  // dy/dzeta

    J[1] = Float3<FloatType>::make(fy1 - fy0, fy2 - fy0, fy3 - fy0);

    // Row 2: z-components (indices 6,7,8)
    // J[6] = fz1 - fz0;  // dz/dxi
    // J[7] = fz2 - fz0;  // dz/deta
    // J[8] = fz3 - fz0;  // dz/dzeta

    J[2] = Float3<FloatType>::make(fz1 - fz0, fz2 - fz0, fz3 - fz0);

    // Compute determinant of the 3x3 Jacobian matrix
    // const FloatType det = J[0] * (J[4] * J[8] - J[5] * J[7]) -  //
    //                      J[1] * (J[3] * J[8] - J[5] * J[6]) +  //
    //                      J[2] * (J[3] * J[7] - J[4] * J[6]);   //

    const FloatType det = J[0].x * (J[1].y * J[2].z - J[1].z * J[2].y) -  //
                          J[0].y * (J[1].x * J[2].z - J[1].z * J[2].x) +  //
                          J[0].z * (J[1].x * J[2].y - J[1].y * J[2].x);   //

    return det;
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

__device__ __forceinline__ float  fast_fmaf(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ double fast_fmad(double a, double b, double c) { return __fma_rn(a, b, c); }
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

////////////////////////////////////////////////////////////////////////////////
// Function to evaluate the 8 trilinear shape functions of a hexahedron
// hex_aa_8_eval_fun_T_cu
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
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
__device__ void                                               //
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

    i0 = i * stride0 + j * stride1 + k * stride2;
    i1 = (i + 1) * stride0 + j * stride1 + k * stride2;
    i2 = (i + 1) * stride0 + (j + 1) * stride1 + k * stride2;
    i3 = i * stride0 + (j + 1) * stride1 + k * stride2;
    i4 = i * stride0 + j * stride1 + (k + 1) * stride2;
    i5 = (i + 1) * stride0 + j * stride1 + (k + 1) * stride2;
    i6 = (i + 1) * stride0 + (j + 1) * stride1 + (k + 1) * stride2;
    i7 = i * stride0 + (j + 1) * stride1 + (k + 1) * stride2;
}

////////////////////////////////////////////////////////////////////////////////
// Function to compute the distance between two points in 3D space
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ FloatType                       //
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
__device__ FloatType                                     //
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
__device__ int                                                    //
alpha_to_hyteg_level_gpu(const FloatType    alpha,                //
                         const FloatType    alpha_min_threshold,  //
                         const FloatType    alpha_max_threshold,  //
                         const unsigned int min_refinement_L,     //
                         const unsigned int max_refinement_L) {   //

    // return 1;  ///// TODO

    // const int min_refinement_L = 2;  // Minimum refinement level

    if (alpha < alpha_min_threshold) return min_refinement_L;  // No refinement
    if (alpha > alpha_max_threshold) return max_refinement_L;  // Maximum refinement

    FloatType alpha_x = alpha - alpha_min_threshold;  // Shift the alpha to start from 0
    FloatType L_real =
            (alpha_x / (alpha_max_threshold - alpha_min_threshold) * (FloatType)(HYTEG_MAX_REFINEMENT_LEVEL - 1)) + 1.0;

    int L = L_real >= FloatType(1.0) ? (int)L_real : min_refinement_L;        // Convert to integer
    L     = L > HYTEG_MAX_REFINEMENT_LEVEL ? HYTEG_MAX_REFINEMENT_LEVEL : L;  // Clamp to maximum level

    const int ret = L >= max_refinement_L ? max_refinement_L : L;
    return (ret) < min_refinement_L ? min_refinement_L : (ret);  // Ensure L is within bounds
}

/////////////////////////////////////////////////////////////////////////////////
// Resampling function for a mini-tetrahedron and a given category
/////////////////////////////////////////////////////////////////////////////////

// Warp-aggregated atomicAdd to reduce contention (1 atomic per unique address in a warp)
template <typename T>
__device__ inline void warpAggAtomicAdd(T* data, ptrdiff_t idx, T val) {
#if __CUDA_ARCH__ >= 700
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
        T v = __shfl_sync(full_mask, val, src);
        if (lane == leader && (peers & (1u << src)) && src != leader) {
            sum += v;
        }
    }

    if (lane == leader) {
        atomicAdd(&data[idx], sum);
    }
#else
    // Fallback for older architectures
    atomicAdd(&data[idx], val);
#endif
}

template <typename FloatType>
__device__ FloatType  //
tet4_resample_tetrahedron_local_adjoint_category_gpu(
        const unsigned int                     category,    //
        const unsigned int                     L,           // Refinement level
        const typename Float3<FloatType>::type bc,          // Fixed double const
        const typename Float3<FloatType>::type J_phys[3],   // Jacobian matrix
        const typename Float3<FloatType>::type J_ref[3],    // Jacobian matrix
        const FloatType                        det_J_phys,  // Determinant of the Jacobian matrix (changed from vector type)
        const typename Float3<FloatType>::type fxyz,        // Tetrahedron origin vertex XYZ-coordinates
        const FloatType                        wf0,         // Weighted field at the vertices
        const FloatType                        wf1,         //
        const FloatType                        wf2,         //
        const FloatType                        wf3,         //
        const FloatType                        ox,          // Origin of the grid
        const FloatType                        oy,          //
        const FloatType                        oz,          //
        const FloatType                        dx,          // Spacing of the grid
        const FloatType                        dy,          //
        const FloatType                        dz,          //
        const ptrdiff_t                        stride0,     // Stride
        const ptrdiff_t                        stride1,     //
        const ptrdiff_t                        stride2,     //
        const ptrdiff_t                        n0,          // Size of the grid
        const ptrdiff_t                        n1,          //
        const ptrdiff_t                        n2,          //
        FloatType* const                       data) {                            // Output

    const FloatType N_micro_tet     = (FloatType)(L) * (FloatType)(L) * (FloatType)(L);
    const FloatType inv_N_micro_tet = 1.0 / N_micro_tet;  // Inverse of the number of micro-tetrahedra

    const FloatType theta_volume = det_J_phys / ((FloatType)(6.0));  // Volume of the mini-tetrahedron in the physical space

    // FloatType cumulated_dV = 0.0;

    // const int tile_id = threadIdx.x / LANES_PER_TILE;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id   = thread_id % LANES_PER_TILE;

    const FloatType inv_dx = FloatType(1.0) / dx;
    const FloatType inv_dy = FloatType(1.0) / dy;
    const FloatType inv_dz = FloatType(1.0) / dz;

    // Per-thread cell-aware accumulation (one-entry cache)
    ptrdiff_t cache_base = -1;  // invalid
    // Relative offsets for the 8 hex nodes (constant for given strides)
    const ptrdiff_t off0 = 0;
    const ptrdiff_t off1 = stride0;
    const ptrdiff_t off2 = stride0 + stride1;
    const ptrdiff_t off3 = stride1;
    const ptrdiff_t off4 = stride2;
    const ptrdiff_t off5 = stride0 + stride2;
    const ptrdiff_t off6 = stride0 + stride1 + stride2;
    const ptrdiff_t off7 = stride1 + stride2;

    FloatType acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    FloatType acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;

    for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i += LANES_PER_TILE) {  // loop over the quadrature points

        const int quad_i_tile = quad_i + lane_id;
        if (quad_i_tile >= TET_QUAD_NQP) continue;  // skip inactive lanes early

        // Direct loads (avoid ternaries)
        const FloatType qx = tet_qx[quad_i_tile];
        const FloatType qy = tet_qy[quad_i_tile];
        const FloatType qz = tet_qz[quad_i_tile];
        const FloatType qw = tet_qw[quad_i_tile];

        // Mapping the quadrature point from the reference space to the mini-tetrahedron
        const FloatType xq_mref = J_ref[0].x * qx + J_ref[0].y * qy + J_ref[0].z * qz + bc.x;
        const FloatType yq_mref = J_ref[1].x * qx + J_ref[1].y * qy + J_ref[1].z * qz + bc.y;
        const FloatType zq_mref = J_ref[2].x * qx + J_ref[2].y * qy + J_ref[2].z * qz + bc.z;

        // Mapping the quadrature point from the mini-tetrahedron to the physical space
        const FloatType xq_phys = J_phys[0].x * xq_mref + J_phys[0].y * yq_mref + J_phys[0].z * zq_mref + fxyz.x;
        const FloatType yq_phys = J_phys[1].x * xq_mref + J_phys[1].y * yq_mref + J_phys[1].z * zq_mref + fxyz.y;
        const FloatType zq_phys = J_phys[2].x * xq_mref + J_phys[2].y * yq_mref + J_phys[2].z * zq_mref + fxyz.z;

        const FloatType grid_x = (xq_phys - ox) * inv_dx;
        const FloatType grid_y = (yq_phys - oy) * inv_dy;
        const FloatType grid_z = (zq_phys - oz) * inv_dz;

        // Fast floor
        const ptrdiff_t i = (ptrdiff_t)fast_floor<FloatType>(grid_x);
        const ptrdiff_t j = (ptrdiff_t)fast_floor<FloatType>(grid_y);
        const ptrdiff_t k = (ptrdiff_t)fast_floor<FloatType>(grid_z);

        const FloatType l_x = (grid_x - (FloatType)(i));
        const FloatType l_y = (grid_y - (FloatType)(j));
        const FloatType l_z = (grid_z - (FloatType)(k));

        const FloatType f0 = FloatType(1.0) - xq_mref - yq_mref - zq_mref;
        const FloatType f1 = xq_mref;
        const FloatType f2 = yq_mref;
        const FloatType f3 = zq_mref;

        // printf("theta_volume = %e, inv_N_micro_tet = %e, qw = %e\n", theta_volume, inv_N_micro_tet, qw);

        const FloatType wf_quad = f0 * wf0 + f1 * wf1 + f2 * wf2 + f3 * wf3;
        // const FloatType wf_quad = f0 * 1.0 + f1 * 1.0 + f2 * 1.0 + f3 * 1.0;
        const FloatType dV = theta_volume * inv_N_micro_tet * qw;
        const FloatType It = wf_quad * dV;

        // cumulated_dV += dV;  // Cumulative volume for debugging

        FloatType hex8_f0 = 0.0,  //
                hex8_f1   = 0.0,  //
                hex8_f2   = 0.0,  //
                hex8_f3   = 0.0,  //
                hex8_f4   = 0.0,  //
                hex8_f5   = 0.0,  //
                hex8_f6   = 0.0,  //
                hex8_f7   = 0.0;  //

        hex_aa_8_eval_fun_T_gpu(l_x,  //
                                l_y,
                                l_z,
                                hex8_f0,
                                hex8_f1,
                                hex8_f2,
                                hex8_f3,
                                hex8_f4,
                                hex8_f5,
                                hex8_f6,
                                hex8_f7);

        const FloatType d0 = It * hex8_f0;
        const FloatType d1 = It * hex8_f1;
        const FloatType d2 = It * hex8_f2;
        const FloatType d3 = It * hex8_f3;
        const FloatType d4 = It * hex8_f4;
        const FloatType d5 = It * hex8_f5;
        const FloatType d6 = It * hex8_f6;
        const FloatType d7 = It * hex8_f7;

        // Base linear index for the current cell
        const ptrdiff_t base = i * stride0 + j * stride1 + k * stride2;

        if (base == cache_base) {
            // Same cell as previous iteration: accumulate locally
            acc0 += d0;
            acc1 += d1;
            acc2 += d2;
            acc3 += d3;
            acc4 += d4;
            acc5 += d5;
            acc6 += d6;
            acc7 += d7;
        } else {
            // Flush previous cell if any
            if (cache_base != -1) {
                atomicAdd(&data[cache_base + off0], acc0);
                atomicAdd(&data[cache_base + off1], acc1);
                atomicAdd(&data[cache_base + off2], acc2);
                atomicAdd(&data[cache_base + off3], acc3);
                atomicAdd(&data[cache_base + off4], acc4);
                atomicAdd(&data[cache_base + off5], acc5);
                atomicAdd(&data[cache_base + off6], acc6);
                atomicAdd(&data[cache_base + off7], acc7);
            }
            // Start accumulating for the new cell
            cache_base = base;
            acc0       = d0;
            acc1       = d1;
            acc2       = d2;
            acc3       = d3;
            acc4       = d4;
            acc5       = d5;
            acc6       = d6;
            acc7       = d7;
        }

        // Update the data with atomic operations to prevent race conditions
        // atomicAdd(&data[i0], d0);
        // atomicAdd(&data[i1], d1);
        // atomicAdd(&data[i2], d2);
        // atomicAdd(&data[i3], d3);
        // atomicAdd(&data[i4], d4);
        // atomicAdd(&data[i5], d5);
        // atomicAdd(&data[i6], d6);
        // atomicAdd(&data[i7], d7);
    }

    // Flush tail
    if (cache_base != -1) {
        atomicAdd(&data[cache_base + off0], acc0);
        atomicAdd(&data[cache_base + off1], acc1);
        atomicAdd(&data[cache_base + off2], acc2);
        atomicAdd(&data[cache_base + off3], acc3);
        atomicAdd(&data[cache_base + off4], acc4);
        atomicAdd(&data[cache_base + off5], acc5);
        atomicAdd(&data[cache_base + off6], acc6);
        atomicAdd(&data[cache_base + off7], acc7);
    }

    // Reduce the cumulated_dV across all lanes in the tile
    // unsigned int mask = 0xFF;  // Mask for 8 lanes

    // // Reduction using warp shuffle operations
    // for (int offset = LANES_PER_TILE / 2; offset > 0; offset >>= 1) {
    //     cumulated_dV += __shfl_down_sync(mask, cumulated_dV, offset);
    // }

    // // Broadcast the result from lane 0 to all other lanes in the tile
    // cumulated_dV = __shfl_sync(mask, cumulated_dV, 0);

    return 0.0;  // cumulated_dV;  // Return the cumulative volume for debugging
}

////////////////////////////////////////////////////////////////////////////////
// Main loop over the mini-tetrahedron
// main_tet_loop_gpu
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ void main_tet_loop_gpu(const int                               L,
                                  const typename Float3<FloatType>::type* J_phys,      // Jacobian matrix
                                  const FloatType                         det_J_phys,  // Determinant of the Jacobian matrix
                                  const typename Float3<FloatType>::type  fxyz,     // Tetrahedron origin vertex XYZ-coordinates
                                  const FloatType                         wf0,      // Weighted field at the vertices
                                  const FloatType                         wf1,      //
                                  const FloatType                         wf2,      //
                                  const FloatType                         wf3,      //
                                  const FloatType                         ox,       // Origin of the grid
                                  const FloatType                         oy,       //
                                  const FloatType                         oz,       //
                                  const FloatType                         dx,       // Spacing of the grid
                                  const FloatType                         dy,       //
                                  const FloatType                         dz,       //
                                  const ptrdiff_t                         stride0,  // Stride
                                  const ptrdiff_t                         stride1,  //
                                  const ptrdiff_t                         stride2,  //
                                  const ptrdiff_t                         n0,       // Size of the grid
                                  const ptrdiff_t                         n1,       //
                                  const ptrdiff_t                         n2,       //
                                  FloatType* const                        data) {                          // Output

    const FloatType zero = 0.0;

    using FloatType3 = typename Float3<FloatType>::type;

    FloatType3      Jacobian_c[6][3];
    const FloatType h = FloatType(1.0) / FloatType(L);

    for (int cat_i = 0; cat_i < 6; cat_i++) {
        bool status = get_category_Jacobian<FloatType>(cat_i, FloatType(L), Jacobian_c[cat_i]);
        if (!status) {
            // Handle error: invalid category
            // For example, you might want to set a default value or log an error
        }
    }

    for (int k = 0; k <= L; ++k) {  // Loop over z

        const int nodes_per_side = (L - k) + 1;
        // const int nodes_per_layer = nodes_per_side * (nodes_per_side + 1) / 2;
        // Removed unused variable Ns
        // const int Nl = nodes_per_layer;

        // Layer loop info
        // printf("Layer %d: Ik = %d, Ns = %d, Nl = %d\n", k, Ik, Ns, Nl);

        for (int j = 0; j < nodes_per_side - 1; ++j) {          // Loop over y
            for (int i = 0; i < nodes_per_side - 1 - j; ++i) {  // Loop over x

                const FloatType3 bc = Float3<FloatType>::make(FloatType(i) * h,   //
                                                              FloatType(j) * h,   //
                                                              FloatType(k) * h);  //

                // Category 0
                // ... category 0 logic here ...
                {
                    const unsigned int cat_0 = 0;
                    tet4_resample_tetrahedron_local_adjoint_category_gpu(cat_0,  //
                                                                         L,
                                                                         bc,
                                                                         J_phys,
                                                                         Jacobian_c[cat_0],
                                                                         det_J_phys,
                                                                         fxyz,
                                                                         wf0,
                                                                         wf1,
                                                                         wf2,
                                                                         wf3,
                                                                         ox,
                                                                         oy,
                                                                         oz,
                                                                         dx,
                                                                         dy,
                                                                         dz,
                                                                         stride0,
                                                                         stride1,
                                                                         stride2,
                                                                         n0,
                                                                         n1,
                                                                         n2,
                                                                         data);
                }

                if (i >= 1) {
                    for (int cat_ii = 1; cat_ii <= 4; cat_ii++) {
                        tet4_resample_tetrahedron_local_adjoint_category_gpu(cat_ii,  //
                                                                             L,
                                                                             bc,
                                                                             J_phys,
                                                                             Jacobian_c[cat_ii],
                                                                             det_J_phys,
                                                                             fxyz,
                                                                             wf0,
                                                                             wf1,
                                                                             wf2,
                                                                             wf3,
                                                                             ox,
                                                                             oy,
                                                                             oz,
                                                                             dx,
                                                                             dy,
                                                                             dz,
                                                                             stride0,
                                                                             stride1,
                                                                             stride2,
                                                                             n0,
                                                                             n1,
                                                                             n2,
                                                                             data);
                    }
                }  // END if (i >= 1)

                if (j >= 1 && i >= 1) {
                    // Category 5
                    // ... category 5 logic here ...
                    const unsigned int cat_5 = 5;
                    tet4_resample_tetrahedron_local_adjoint_category_gpu(cat_5,  //
                                                                         L,
                                                                         bc,
                                                                         J_phys,
                                                                         Jacobian_c[cat_5],
                                                                         det_J_phys,
                                                                         fxyz,
                                                                         wf0,
                                                                         wf1,
                                                                         wf2,
                                                                         wf3,
                                                                         ox,
                                                                         oy,
                                                                         oz,
                                                                         dx,
                                                                         dy,
                                                                         dz,
                                                                         stride0,
                                                                         stride1,
                                                                         stride2,
                                                                         n0,
                                                                         n1,
                                                                         n2,
                                                                         data);
                }
            }
        }
        // Ik = Ik + Nl;
    }
}

/////////////////////////////////////////////////////////////////////////////////
// Kernel to perform adjoint mini-tetrahedron resampling
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__global__ void                                                                    //
sfem_adjoint_mini_tet_kernel_gpu(const ptrdiff_t             start_element,        // Mesh
                                 const ptrdiff_t             end_element,          //
                                 const ptrdiff_t             nnodes,               //
                                 const elems_tet4_device     elems,                //
                                 const xyz_tet4_device       xyz,                  //
                                 const ptrdiff_t             n0,                   // SDF
                                 const ptrdiff_t             n1,                   //
                                 const ptrdiff_t             n2,                   //
                                 const ptrdiff_t             stride0,              // Stride
                                 const ptrdiff_t             stride1,              //
                                 const ptrdiff_t             stride2,              //
                                 const geom_t                origin0,              // Origin
                                 const geom_t                origin1,              //
                                 const geom_t                origin2,              //
                                 const geom_t                dx,                   // Delta
                                 const geom_t                dy,                   //
                                 const geom_t                dz,                   //
                                 const FloatType* const      weighted_field,       // Input weighted field
                                 const mini_tet_parameters_t mini_tet_parameters,  // Threshold for alpha
                                 FloatType* const            data) {                          //

    const int tet_id    = (blockIdx.x * blockDim.x + threadIdx.x) / LANES_PER_TILE;
    const int element_i = start_element + tet_id;  // Global element index

    if (element_i >= end_element) return;  // Out of range

    // printf("Processing element %ld / %ld\n", element_i, end_element);

    const FloatType d_min             = dx < dy ? (dx < dz ? dx : dz) : (dy < dz ? dy : dz);
    const FloatType hexahedron_volume = dx * dy * dz;

    // printf("Exaedre volume: %e\n", hexahedron_volume);

    idx_t ev[4] = {0, 0, 0, 0};  // Indices of the vertices of the tetrahedron

    ev[0] = elems.elems_v0[element_i];
    ev[1] = elems.elems_v1[element_i];
    ev[2] = elems.elems_v2[element_i];
    ev[3] = elems.elems_v3[element_i];

    // Read the coordinates of the vertices of the tetrahedron
    // In the physical space
    const FloatType x0_n = FloatType(xyz.x[ev[0]]);
    const FloatType x1_n = FloatType(xyz.x[ev[1]]);
    const FloatType x2_n = FloatType(xyz.x[ev[2]]);
    const FloatType x3_n = FloatType(xyz.x[ev[3]]);

    const FloatType y0_n = FloatType(xyz.y[ev[0]]);
    const FloatType y1_n = FloatType(xyz.y[ev[1]]);
    const FloatType y2_n = FloatType(xyz.y[ev[2]]);
    const FloatType y3_n = FloatType(xyz.y[ev[3]]);

    const FloatType z0_n = FloatType(xyz.z[ev[0]]);
    const FloatType z1_n = FloatType(xyz.z[ev[1]]);
    const FloatType z2_n = FloatType(xyz.z[ev[2]]);
    const FloatType z3_n = FloatType(xyz.z[ev[3]]);

    const FloatType wf0 = weighted_field[ev[0]];  // Weighted field at vertex 0
    const FloatType wf1 = weighted_field[ev[1]];  // Weighted field at vertex 1
    const FloatType wf2 = weighted_field[ev[2]];  // Weighted field at vertex 2
    const FloatType wf3 = weighted_field[ev[3]];  // Weighted field at vertex 3

    FloatType edges_length[6];

    int vertex_a = -1;
    int vertex_b = -1;

    const FloatType max_edges_length =              //
            tet_edge_max_length_gpu(x0_n,           //
                                    y0_n,           //
                                    z0_n,           //
                                    x1_n,           //
                                    y1_n,           //
                                    z1_n,           //
                                    x2_n,           //
                                    y2_n,           //
                                    z2_n,           //
                                    x3_n,           //
                                    y3_n,           //
                                    z3_n,           //
                                    &vertex_a,      // Output
                                    &vertex_b,      // Output
                                    edges_length);  // Output

    const FloatType alpha_tet = max_edges_length / d_min;

    const int L = alpha_to_hyteg_level_gpu(alpha_tet,                                           //
                                           FloatType(mini_tet_parameters.alpha_min_threshold),  //
                                           FloatType(mini_tet_parameters.alpha_max_threshold),  //
                                           mini_tet_parameters.min_refinement_L,                //
                                           mini_tet_parameters.max_refinement_L);               //

    typename Float3<FloatType>::type Jacobian_phys[3];

    const FloatType det_J_phys =                               //
            abs(make_Jacobian_matrix_tet_gpu<FloatType>(x0_n,  //
                                                        x1_n,  //
                                                        x2_n,
                                                        x3_n,
                                                        y0_n,
                                                        y1_n,
                                                        y2_n,
                                                        y3_n,
                                                        z0_n,
                                                        z1_n,
                                                        z2_n,
                                                        z3_n,
                                                        Jacobian_phys));

    // if (element_i == 37078) {
    //     printf("---- Debug info (sfem_adjoint_mini_tet_kernel_gpu) ----\n");
    //     printf("Element %ld / %ld\n", element_i, end_element);
    //     printf("Vertex indices: %ld, %ld, %ld, %ld\n", (long)ev[0], (long)ev[1], (long)ev[2], (long)ev[3]);
    //     printf("Vertex coordinates:\n");
    //     printf("V0: (%e, %e, %e), wf0: %e\n", (double)x0_n, (double)y0_n, (double)z0_n, (double)wf0);
    //     printf("V1: (%e, %e, %e), wf1: %e\n", (double)x1_n, (double)y1_n, (double)z1_n, (double)wf1);
    //     printf("V2: (%e, %e, %e), wf2: %e\n", (double)x2_n, (double)y2_n, (double)z2_n, (double)wf2);
    //     printf("V3: (%e, %e, %e), wf3: %e\n", (double)x3_n, (double)y3_n, (double)z3_n, (double)wf3);
    //     printf("det_J_phys = %e\n", (double)det_J_phys);
    //     printf("Jacobian_phys[0] = (%e, %e, %e)\n",
    //            (double)Jacobian_phys[0].x,
    //            (double)Jacobian_phys[0].y,
    //            (double)Jacobian_phys[0].z);
    //     printf("Jacobian_phys[1] = (%e, %e, %e)\n",
    //            (double)Jacobian_phys[1].x,
    //            (double)Jacobian_phys[1].y,
    //            (double)Jacobian_phys[1].z);
    //     printf("Jacobian_phys[2] = (%e, %e, %e)\n",
    //            (double)Jacobian_phys[2].x,
    //            (double)Jacobian_phys[2].y,
    //            (double)Jacobian_phys[2].z);
    //     printf("Max edge length: %e (between vertices %d and %d)\n", (double)max_edges_length, vertex_a, vertex_b);

    //     printf("---------------------------------------------------\n");
    //     __trap();
    // }

    main_tet_loop_gpu<FloatType>(L,                                          //
                                 Jacobian_phys,                              //
                                 det_J_phys,                                 //
                                 Float3<FloatType>::make(x0_n, y0_n, z0_n),  //
                                 wf0,                                        //
                                 wf1,                                        //
                                 wf2,                                        //
                                 wf3,                                        //
                                 origin0,                                    //
                                 origin1,                                    //
                                 origin2,                                    //
                                 dx,                                         //
                                 dy,                                         //
                                 dz,                                         //
                                 stride0,                                    //
                                 stride1,                                    //
                                 stride2,                                    //
                                 n0,                                         //
                                 n1,                                         //
                                 n2,                                         //
                                 data);                                      //
}
/////////////////////////////////////////////////////////////////////////////////

extern "C" void                                                                         //
call_sfem_adjoint_mini_tet_kernel_gpu(const ptrdiff_t             start_element,        // Mesh
                                      const ptrdiff_t             end_element,          //
                                      const ptrdiff_t             nelements,            //
                                      const ptrdiff_t             nnodes,               //
                                      const idx_t** const         elems,                //
                                      const geom_t** const        xyz,                  //
                                      const ptrdiff_t             n0,                   // SDF
                                      const ptrdiff_t             n1,                   //
                                      const ptrdiff_t             n2,                   //
                                      const ptrdiff_t             stride0,              // Stride
                                      const ptrdiff_t             stride1,              //
                                      const ptrdiff_t             stride2,              //
                                      const geom_t                origin0,              // Origin
                                      const geom_t                origin1,              //
                                      const geom_t                origin2,              //
                                      const geom_t                dx,                   // Delta
                                      const geom_t                dy,                   //
                                      const geom_t                dz,                   //
                                      const real_t* const         weighted_field,       // Input weighted field
                                      const mini_tet_parameters_t mini_tet_parameters,  // Threshold for alpha
                                      real_t* const               data);

#endif  // __SFEM_ADJOINT_MINI_TET_CUH__