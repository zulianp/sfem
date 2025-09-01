#include <cuda_runtime.h>
#include <stddef.h>

#include "quadratures_rule_cuda.cuh"
// #include "sfem_config.h"

typedef float   geom_t;
typedef int32_t idx_t;

#define LANES_PER_TILE 8

typedef struct {
    float        alpha;
    float        alpha_min_threshold;
    float        alpha_max_threshold;
    unsigned int min_refinement_L;
    unsigned int max_refinement_L;
} mini_tet_parameters_t;

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

template <typename FloatType>
__device__ bool get_category_Jacobian(const unsigned int category, const FloatType L,
                                      typename Float3<FloatType>::type* Jacobian_c) {
    const FloatType invL = FloatType(1.0) / FloatType(L);
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

template <typename FloatType>
__device__ void                             //
hex_aa_8_eval_fun_T(const FloatType  x,     // Local coordinates (in the unit cube)
                    const FloatType  y,     //
                    const FloatType  z,     //
                    FloatType* const f0,    // Output
                    FloatType* const f1,    //
                    FloatType* const f2,    //
                    FloatType* const f3,    //
                    FloatType* const f4,    //
                    FloatType* const f5,    //
                    FloatType* const f6,    //
                    FloatType* const f7) {  //
    // Quadrature point (local coordinates)
    // With respect to the hat functions of a cube element
    // In a local coordinate system
    //
    *f0 = (1.0 - x) * (1.0 - y) * (1.0 - z);
    *f1 = x * (1.0 - y) * (1.0 - z);
    *f2 = x * y * (1.0 - z);
    *f3 = (1.0 - x) * y * (1.0 - z);
    *f4 = (1.0 - x) * (1.0 - y) * z;
    *f5 = x * (1.0 - y) * z;
    *f6 = x * y * z;
    *f7 = (1.0 - x) * y * z;
}

void                                                                   //
hex_aa_8_collect_coeffs_indices_T(const ptrdiff_t stride0,             // Stride
                                  const ptrdiff_t stride1,             //
                                  const ptrdiff_t stride2,             //
                                  const ptrdiff_t i,                   // Indices of the element
                                  const ptrdiff_t j,                   //
                                  const ptrdiff_t k,                   //
                                  ptrdiff_t* const __restrict__ i0,    //
                                  ptrdiff_t* const __restrict__ i1,    //
                                  ptrdiff_t* const __restrict__ i2,    //
                                  ptrdiff_t* const __restrict__ i3,    //
                                  ptrdiff_t* const __restrict__ i4,    //
                                  ptrdiff_t* const __restrict__ i5,    //
                                  ptrdiff_t* const __restrict__ i6,    //
                                  ptrdiff_t* const __restrict__ i7) {  //

    *i0 = i * stride0 + j * stride1 + k * stride2;
    *i1 = (i + 1) * stride0 + j * stride1 + k * stride2;
    *i2 = (i + 1) * stride0 + (j + 1) * stride1 + k * stride2;
    *i3 = i * stride0 + (j + 1) * stride1 + k * stride2;
    *i4 = i * stride0 + j * stride1 + (k + 1) * stride2;
    *i5 = (i + 1) * stride0 + j * stride1 + (k + 1) * stride2;
    *i6 = (i + 1) * stride0 + (j + 1) * stride1 + (k + 1) * stride2;
    *i7 = i * stride0 + (j + 1) * stride1 + (k + 1) * stride2;
}

template <typename RealType>
__device__ RealType                      //
points_distance_cu(const RealType x0,    //
                   const RealType y0,    //
                   const RealType z0,    //
                   const RealType x1,    //
                   const RealType y1,    //
                   const RealType z1) {  //

    const RealType dx = x1 - x0;
    const RealType dy = y1 - y0;
    const RealType dz = z1 - z0;

    return sqrt(dx * dx + dy * dy + dz * dz);
}

template <typename RealType>
__device__ RealType                                    //
tet_edge_max_length_cu(const RealType  v0x,            //
                       const RealType  v0y,            //
                       const RealType  v0z,            //
                       const RealType  v1x,            //
                       const RealType  v1y,            //
                       const RealType  v1z,            //
                       const RealType  v2x,            //
                       const RealType  v2y,            //
                       const RealType  v2z,            //
                       const RealType  v3x,            //
                       const RealType  v3y,            //
                       const RealType  v3z,            //
                       int*            vertex_a,       //
                       int*            vertex_b,       //
                       RealType* const edge_length) {  //

    real_t max_length = 0.0;

    // Edge 0 (v0, v1)
    edge_length[0] = points_distance_cu(v0x, v0y, v0z, v1x, v1y, v1z);
    if (edge_length[0] > max_length) {
        max_length = edge_length[0];
        *vertex_a  = 0;
        *vertex_b  = 1;
    }

    // Edge 1 (v0, v2)
    edge_length[1] = points_distance_cu(v0x, v0y, v0z, v2x, v2y, v2z);
    if (edge_length[1] > max_length) {
        max_length = edge_length[1];
        *vertex_a  = 0;
        *vertex_b  = 2;
    }

    // Edge 2 (v0, v3)
    edge_length[2] = points_distance_cu(v0x, v0y, v0z, v3x, v3y, v3z);
    if (edge_length[2] > max_length) {
        max_length = edge_length[2];
        *vertex_a  = 0;
        *vertex_b  = 3;
    }

    // Edge 3 (v1, v2)
    edge_length[3] = points_distance_cu(v1x, v1y, v1z, v2x, v2y, v2z);
    if (edge_length[3] > max_length) {
        max_length = edge_length[3];
        *vertex_a  = 1;
        *vertex_b  = 2;
    }

    // Edge 4 (v1, v3)
    edge_length[4] = points_distance_cu(v1x, v1y, v1z, v3x, v3y, v3z);
    if (edge_length[4] > max_length) {
        max_length = edge_length[4];
        *vertex_a  = 1;
        *vertex_b  = 3;
    }

    // Edge 5 (v2, v3)
    edge_length[5] = points_distance_cu(v2x, v2y, v2z, v3x, v3y, v3z);
    if (edge_length[5] > max_length) {
        max_length = edge_length[5];
        *vertex_a  = 2;
        *vertex_b  = 3;
    }

    return max_length;
}

template <typename FloatType>
__device__ FloatType                                                                  //
tet4_resample_tetrahedron_local_adjoint_category(const unsigned int      category,    //
                                                 const unsigned int      L,           // Refinement level
                                                 const Float3<FloatType> bc,          // Fixed double const
                                                 const Float3<FloatType> J_phys[3],   // Jacobian matrix
                                                 const Float3<FloatType> J_ref[3],    // Jacobian matrix
                                                 const Float3<FloatType> det_J_phys,  // Determinant of the Jacobian matrix
                                                 const Float3<FloatType> fxyz,        // Tetrahedron origin vertex XYZ-coordinates
                                                 const FloatType         wf0,         // Weighted field at the vertices
                                                 const FloatType         wf1,         //
                                                 const FloatType         wf2,         //
                                                 const FloatType         wf3,         //
                                                 const FloatType         ox,          // Origin of the grid
                                                 const FloatType         oy,          //
                                                 const FloatType         oz,          //
                                                 const FloatType         dx,          // Spacing of the grid
                                                 const FloatType         dy,          //
                                                 const FloatType         dz,          //
                                                 const ptrdiff_t         stride0,     // Stride
                                                 const ptrdiff_t         stride1,     //
                                                 const ptrdiff_t         stride2,     //
                                                 const ptrdiff_t         n0,          // Size of the grid
                                                 const ptrdiff_t         n1,          //
                                                 const ptrdiff_t         n2,          //
                                                 FloatType* const        data) {             // Output

    const FloatType N_micro_tet     = (FloatType)(L) * (FloatType)(L) * (FloatType)(L);
    const FloatType inv_N_micro_tet = 1.0 / N_micro_tet;  // Inverse of the number of micro-tetrahedra

    const FloatType theta_volume = det_J_phys / ((FloatType)(6.0));  // Volume of the mini-tetrahedron in the physical space

    FloatType cumulated_dV = 0.0;

    const int tile_id = threadIdx.x / LANES_PER_TILE;
    const int lane_id = threadIdx.x % LANES_PER_TILE;

    for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i += LANES_PER_TILE) {  // loop over the quadrature points

        const int quad_i_tile = quad_i + lane_id;

        const FloatType qx = quad_i_tile < TET_QUAD_NQP ? tet_qx[quad_i_tile] : 0.0;
        const FloatType qy = quad_i_tile < TET_QUAD_NQP ? tet_qy[quad_i_tile] : 0.0;
        const FloatType qz = quad_i_tile < TET_QUAD_NQP ? tet_qz[quad_i_tile] : 0.0;
        const FloatType qw = quad_i_tile < TET_QUAD_NQP ? tet_qw[quad_i_tile] : 0.0;

        // Mapping the quadrature point from the reference space to the mini-tetrahedron
        const FloatType xq_mref = J_ref[0].x * qx + J_ref[0].y * qy + J_ref[0].z * qz + bc.x;
        const FloatType yq_mref = J_ref[1].x * qx + J_ref[1].y * qy + J_ref[1].z * qz + bc.y;
        const FloatType zq_mref = J_ref[2].x * qx + J_ref[2].y * qy + J_ref[2].z * qz + bc.z;

        // Mapping the quadrature point from the mini-tetrahedron to the physical space
        const FloatType xq_phys = J_phys[0].x * xq_mref + J_phys[0].y * yq_mref + J_phys[0].z * zq_mref + fxyz.x;
        const FloatType yq_phys = J_phys[1].x * xq_mref + J_phys[1].y * yq_mref + J_phys[1].z * zq_mref + fxyz.y;
        const FloatType zq_phys = J_phys[2].x * xq_mref + J_phys[2].y * yq_mref + J_phys[2].z * zq_mref + fxyz.z;

        const FloatType grid_x = (xq_phys - ox) / dx;
        const FloatType grid_y = (yq_phys - oy) / dy;
        const FloatType grid_z = (zq_phys - oz) / dz;

        const ptrdiff_t i = floor(grid_x);  /// ATTENTION: To be replaced with the adaptive floor
        const ptrdiff_t j = floor(grid_y);  /// In the sfem math library
        const ptrdiff_t k = floor(grid_z);

        const FloatType l_x = (grid_x - (FloatType)i);
        const FloatType l_y = (grid_y - (FloatType)j);
        const FloatType l_z = (grid_z - (FloatType)k);

        const FloatType f0 = 1.0 - xq_mref - yq_mref - zq_mref;
        const FloatType f1 = xq_mref;
        const FloatType f2 = yq_mref;
        const FloatType f3 = zq_mref;

        const FloatType wf_quad = f0 * wf0 + f1 * wf1 + f2 * wf2 + f3 * wf3;
        const FloatType dV      = theta_volume * inv_N_micro_tet * tet_qw[quad_i];
        const FloatType It      = wf_quad * dV;

        cumulated_dV += dV;  // Cumulative volume for debugging

        FloatType hex8_f0, hex8_f1, hex8_f2, hex8_f3, hex8_f4, hex8_f5, hex8_f6, hex8_f7;

        hex_aa_8_eval_fun_T(l_x,  //
                            l_y,
                            l_z,
                            &hex8_f0,
                            &hex8_f1,
                            &hex8_f2,
                            &hex8_f3,
                            &hex8_f4,
                            &hex8_f5,
                            &hex8_f6,
                            &hex8_f7);

        ptrdiff_t i0, i1, i2, i3, i4, i5, i6, i7;
        hex_aa_8_collect_coeffs_indices_T(stride0,  //
                                          stride1,
                                          stride2,
                                          i,
                                          j,
                                          k,
                                          &i0,
                                          &i1,
                                          &i2,
                                          &i3,
                                          &i4,
                                          &i5,
                                          &i6,
                                          &i7);

        const FloatType d0 = It * hex8_f0;
        const FloatType d1 = It * hex8_f1;
        const FloatType d2 = It * hex8_f2;
        const FloatType d3 = It * hex8_f3;
        const FloatType d4 = It * hex8_f4;
        const FloatType d5 = It * hex8_f5;
        const FloatType d6 = It * hex8_f6;
        const FloatType d7 = It * hex8_f7;

        // Update the data with atomic operations to prevent race conditions
        atomicAdd(&data[i0], d0);
        atomicAdd(&data[i1], d1);
        atomicAdd(&data[i2], d2);
        atomicAdd(&data[i3], d3);
        atomicAdd(&data[i4], d4);
        atomicAdd(&data[i5], d5);
        atomicAdd(&data[i6], d6);
        atomicAdd(&data[i7], d7);
    }

    // Reduce the cumulated_dV across all lanes in the tile
    unsigned int mask = 0xFF;  // Mask for 8 lanes

    // Reduction using warp shuffle operations
    for (int offset = LANES_PER_TILE / 2; offset > 0; offset >>= 1) {
        cumulated_dV += __shfl_down_sync(mask, cumulated_dV, offset);
    }

    // Broadcast the result from lane 0 to all other lanes in the tile
    cumulated_dV = __shfl_sync(mask, cumulated_dV, 0);

    return cumulated_dV;
}

template <typename FloatType>
__device__ void main_tet_loop(const int L) {
    const FloatType zero = 0.0;

    int Ik = 0;

    using FloatType3 = typename Float3<FloatType>::type;

    FloatType3      Jacobian_c[6][3];
    const FloatType h = FloatType(1.0) / FloatType(L);

    for (int c = 0; c < 6; ++c) {
        bool status = get_category_Jacobian<FloatType>(c, FloatType(L), Jacobian_c[c]);
        if (!status) {
            // Handle error: invalid category
            // For example, you might want to set a default value or log an error
        }
    }

    for (int k = 0; k <= L; ++k) {  // Loop over z

        const int nodes_per_side  = (L - k) + 1;
        const int nodes_per_layer = nodes_per_side * (nodes_per_side + 1) / 2;
        // Removed unused variable Ns
        const int Nl = nodes_per_layer;

        // Layer loop info
        // printf("Layer %d: Ik = %d, Ns = %d, Nl = %d\n", k, Ik, Ns, Nl);

        for (int j = 0; j < nodes_per_side - 1; ++j) {          // Loop over y
            for (int i = 0; i < nodes_per_side - 1 - j; ++i) {  // Loop over x

                const FloatType3 bc = Float3<FloatType>::make(FloatType(i) * h, FloatType(j) * h, FloatType(k) * h);

                // Category 0
                // ... category 0 logic here ...

                if (i >= 1) {
                    // Category 1
                    // ... category 1 logic here ...

                    // Category 2
                    // ... category 2 logic here ...

                    // Category 3
                    // ... category 3 logic here ...

                    // Category 4
                    // ... category 4 logic here ...
                }

                if (j >= 1 && i >= 1) {
                    // Category 5
                    // ... category 5 logic here ...
                }
            }
        }
        Ik = Ik + Nl;
    }
}

#define HYTEG_MAX_REFINEMENT_LEVEL 20

template <typename RealType>
__device__ int                                                   //
alpha_to_hyteg_level_cu(const RealType     alpha,                //
                        const RealType     alpha_min_threshold,  //
                        const RealType     alpha_max_threshold,  //
                        const unsigned int min_refinement_L,     //
                        const unsigned int max_refinement_L) {   //

    // return 1;  ///// TODO

    // const int min_refinement_L = 2;  // Minimum refinement level

    if (alpha < alpha_min_threshold) return min_refinement_L;  // No refinement
    if (alpha > alpha_max_threshold) return max_refinement_L;  // Maximum refinement

    real_t alpha_x = alpha - alpha_min_threshold;  // Shift the alpha to start from 0
    real_t L_real  = (alpha_x / (alpha_max_threshold - alpha_min_threshold) * (real_t)(HYTEG_MAX_REFINEMENT_LEVEL - 1)) + 1.0;

    int L = L_real >= RealType(1.0) ? (int)L_real : min_refinement_L;         // Convert to integer
    L     = L > HYTEG_MAX_REFINEMENT_LEVEL ? HYTEG_MAX_REFINEMENT_LEVEL : L;  // Clamp to maximum level

    const int ret = L >= max_refinement_L ? max_refinement_L : L;
    return (ret) < min_refinement_L ? min_refinement_L : (ret);  // Ensure L is within bounds
}

template <typename RealType>
__global__ void                                                                //
sfem_adjoint_mini_tet_kernel(const ptrdiff_t             start_element,        // Mesh
                             const ptrdiff_t             end_element,          //
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
                             const RealType* const       weighted_field,       // Input weighted field
                             const mini_tet_parameters_t mini_tet_parameters,  // Threshold for alpha
                             RealType* const             data) {                           //

    const int tet_id    = (blockIdx.x * blockDim.x + threadIdx.x) / LANES_PER_TILE;
    const int element_i = start_element + tet_id;  // Global element index

    const RealType d_min             = dx < dy ? (dx < dz ? dx : dz) : (dy < dz ? dy : dz);
    const RealType hexahedron_volume = dx * dy * dz;

    idx_t ev[4];
    for (int v = 0; v < 4; ++v) {
        ev[v] = elems[v][element_i];
    }

    // Read the coordinates of the vertices of the tetrahedron
    // In the physical space
    const RealType x0_n = xyz[0][ev[0]];
    const RealType x1_n = xyz[0][ev[1]];
    const RealType x2_n = xyz[0][ev[2]];
    const RealType x3_n = xyz[0][ev[3]];

    const RealType y0_n = xyz[1][ev[0]];
    const RealType y1_n = xyz[1][ev[1]];
    const RealType y2_n = xyz[1][ev[2]];
    const RealType y3_n = xyz[1][ev[3]];

    const RealType z0_n = xyz[2][ev[0]];
    const RealType z1_n = xyz[2][ev[1]];
    const RealType z2_n = xyz[2][ev[2]];
    const RealType z3_n = xyz[2][ev[3]];

    const RealType wf0 = weighted_field[ev[0]];  // Weighted field at vertex 0
    const RealType wf1 = weighted_field[ev[1]];  // Weighted field at vertex 1
    const RealType wf2 = weighted_field[ev[2]];  // Weighted field at vertex 2
    const RealType wf3 = weighted_field[ev[3]];  // Weighted field at vertex 3

    RealType edges_length[6];

    int vertex_a = -1;
    int vertex_b = -1;

    const RealType max_edges_length =              //
            tet_edge_max_length_cu(x0_n,           //
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

    const RealType alpha_tet = max_edges_length / d_min;

    const int L = alpha_to_hyteg_level_cu(alpha_tet,                                          //
                                          RealType(mini_tet_parameters.alpha_min_threshold),  //
                                          RealType(mini_tet_parameters.alpha_max_threshold),  //
                                          mini_tet_parameters.min_refinement_L,               //
                                          mini_tet_parameters.max_refinement_L);              //

    main_tet_loop<double>(L);
}

#define __TESTING__
#ifdef __TESTING__

int main() {
    mini_tet_parameters_t mini_tet_parameters;
    mini_tet_parameters.alpha               = 0.5;
    mini_tet_parameters.alpha_min_threshold = 0.0;
    mini_tet_parameters.alpha_max_threshold = 1.0;
    mini_tet_parameters.min_refinement_L    = 1;
    mini_tet_parameters.max_refinement_L    = 20;

    // const int L = 4;  // Example refinement level
    sfem_adjoint_mini_tet_kernel<double><<<1, 1>>>(0,        //
                                                   1,        //
                                                   1,        //
                                                   nullptr,  //
                                                   nullptr,
                                                   1,        //
                                                   1,        //
                                                   1,        //
                                                   1,        //
                                                   1,        //
                                                   1,        //
                                                   0.0,      //
                                                   0.0,      //
                                                   0.0,      //
                                                   1.0,      //
                                                   1.0,      //
                                                   1.0,      //
                                                   nullptr,  //
                                                   mini_tet_parameters,
                                                   nullptr);  //
    cudaDeviceSynchronize();
    return 0;
}

#endif