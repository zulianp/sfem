#include "sfem_resample_field.h"
#include "sfem_resample_field_adjoint_hyteg.h"
#include "sfem_resample_field_tet4_math.h"
#include "sfem_stack.h"

#include "mass.h"
// #include "read_mesh.h"
#include "matrixio_array.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "hyteg.h"
#include "hyteg_Jacobian_matrix_real_t.h"

#include "quadratures_rule.h"

#include "sfem_resample_field_adjoint_hex_quad.h"

#define out_real_t real_t

#define N_QUADRATURE_POINTS_MAX 256

static int    dim_quad_cube_p                      = 1;  // Midpoint quadrature in 3D
static real_t Q_nodes_x_p[N_QUADRATURE_POINTS_MAX] = {0.5};
static real_t Q_nodes_y_p[N_QUADRATURE_POINTS_MAX] = {0.5};
static real_t Q_nodes_z_p[N_QUADRATURE_POINTS_MAX] = {0.5};
static real_t Q_weights_p[N_QUADRATURE_POINTS_MAX] = {1.0};

int init_quad_points_hex_qtet(const int dim_quad) {
    const int dim_quad_cube = dim_quad * dim_quad * dim_quad;

    if (dim_quad_cube < N_QUADRATURE_POINTS_MAX) {
        dim_quad_cube_p = dim_quad_cube;
        sfem_quad_rule_3D(TET_QUAD_MIDPOINT_NQP, dim_quad, Q_nodes_x_p, Q_nodes_y_p, Q_nodes_z_p, Q_weights_p);
    } else {
        return -1;  // Unsupported quadrature
    }

    return 0;  // Success
}

static int                   //
get_dim_qad() {              //
    return dim_quad_cube_p;  //
}  // END: get_dim_qad

int                                                                         //
compute_tet_bounding_box_norm(const real_t                   x0,            //
                              const real_t                   x1,            //
                              const real_t                   x2,            //
                              const real_t                   x3,            //
                              const real_t                   y0,            //
                              const real_t                   y1,            //
                              const real_t                   y2,            //
                              const real_t                   y3,            //
                              const real_t                   z0,            //
                              const real_t                   z1,            //
                              const real_t                   z2,            //
                              const real_t                   z3,            //
                              const ptrdiff_t                stride0,       //
                              const ptrdiff_t                stride1,       //
                              ptrdiff_t* const SFEM_RESTRICT min_grid_x,    //
                              ptrdiff_t* const SFEM_RESTRICT max_grid_x,    //
                              ptrdiff_t* const SFEM_RESTRICT min_grid_y,    //
                              ptrdiff_t* const SFEM_RESTRICT max_grid_y,    //
                              ptrdiff_t* const SFEM_RESTRICT min_grid_z,    //
                              ptrdiff_t* const SFEM_RESTRICT max_grid_z) {  //

    const real_t x_min = fmin(fmin(x0, x1), fmin(x2, x3));
    const real_t x_max = fmax(fmax(x0, x1), fmax(x2, x3));

    const real_t y_min = fmin(fmin(y0, y1), fmin(y2, y3));
    const real_t y_max = fmax(fmax(y0, y1), fmax(y2, y3));

    const real_t z_min = fmin(fmin(z0, z1), fmin(z2, z3));
    const real_t z_max = fmax(fmax(z0, z1), fmax(z2, z3));

    // const real_t dx = delta0;
    // const real_t dy = delta1;
    // const real_t dz = delta2;

    // const real_t ox = origin0;
    // const real_t oy = origin1;
    // const real_t oz = origin2;

    // Step 2: Convert to grid indices accounting for the origin
    // Formula: grid_index = (physical_coord - origin) / delta
    // Using floor for minimum indices (with safety margin of -1)
    *min_grid_x = floor(x_min) - 1;
    *min_grid_y = floor(y_min) - 1;
    *min_grid_z = floor(z_min) - 1;

    // Using ceil for maximum indices (with safety margin of +1)
    *max_grid_x = ceil(x_max) + 1;
    *max_grid_y = ceil(y_max) + 1;
    *max_grid_z = ceil(z_max) + 1;

    return 0;  // Success
}

/////////////////////////////////////////////////////////
// transform_quadrature_point_n /////////////////////////
/////////////////////////////////////////////////////////
quadrature_point_result_t                                                //
transform_quadrature_point_norm(const int q_ijk,                         //
                                const real_t* const restrict Q_nodes_x,  //
                                const real_t* const restrict Q_nodes_y,  //
                                const real_t* const restrict Q_nodes_z,  //
                                const real_t* const restrict Q_weights,  //
                                const ptrdiff_t i_grid,                  //
                                const ptrdiff_t j_grid,                  //
                                const ptrdiff_t k_grid) {                //

    quadrature_point_result_t result;
    result.is_inside = true;

    result.x = ((real_t)i_grid + Q_nodes_x[q_ijk]);
    result.y = ((real_t)j_grid + Q_nodes_y[q_ijk]);
    result.z = ((real_t)k_grid + Q_nodes_z[q_ijk]);

    result.weight = Q_weights[q_ijk];  // Could be removed since we use normalized weights

    return result;
}  // END: transform_and_check_quadrature_point

/////////////////////////////////////////////////////////
// transfer_weighted_field_tet4_to_hex //////////////////
/////////////////////////////////////////////////////////
ijk_index_t                                                                                //
transfer_weighted_field_tet4_to_hex_norm(const real_t                wf0,                  //
                                         const real_t                wf1,                  //
                                         const real_t                wf2,                  //
                                         const real_t                wf3,                  //
                                         const real_t                q_phys_x,             //
                                         const real_t                q_phys_y,             //
                                         const real_t                q_phys_z,             //
                                         const real_t                q_ref_x,              //
                                         const real_t                q_ref_y,              //
                                         const real_t                q_ref_z,              //
                                         const real_t                QW_phys_hex,          //
                                         real_t* const SFEM_RESTRICT hex_element_field) {  //

    // Compute the weighted contribution from the tetrahedron
    // Using linear shape functions for tetrahedron

    // Check if the reference coordinates are valid
    // If they are outside the tetrahedron, skip the contribution
    // Here we check if the ref coords are below the x-z, y-z, and x-y planes.
    // The others check in a previous step.
    // Check if the reference coordinates are valid (all 4 tet constraints)
    if (q_ref_x < 0.0 || q_ref_y < 0.0 || q_ref_z < 0.0 || (q_ref_x + q_ref_y + q_ref_z) > 1.0) {
        return (ijk_index_t){-1, -1, -1, false};
    }  // END if (outside tet)

    const real_t grid_x = q_phys_x;
    const real_t grid_y = q_phys_y;
    const real_t grid_z = q_phys_z;

    const ptrdiff_t i = floor(grid_x);
    const ptrdiff_t j = floor(grid_y);
    const ptrdiff_t k = floor(grid_z);

    const real_t l_x = (grid_x - (real_t)i);
    const real_t l_y = (grid_y - (real_t)j);
    const real_t l_z = (grid_z - (real_t)k);

    const real_t f0 = 1.0 - q_ref_x - q_ref_y - q_ref_z;
    const real_t f1 = q_ref_x;
    const real_t f2 = q_ref_y;
    const real_t f3 = q_ref_z;

    const real_t wf_quad = f0 * wf0 + f1 * wf1 + f2 * wf2 + f3 * wf3;

    real_t hex8_f0, hex8_f1, hex8_f2, hex8_f3, hex8_f4, hex8_f5, hex8_f6, hex8_f7;
    // hex_aa_8_eval_fun_V(l_x,        // Local coordinates
    //                     l_y,        //
    //                     l_z,        //
    //                     &hex8_f0,   // Output shape functions
    //                     &hex8_f1,   //
    //                     &hex8_f2,   //
    //                     &hex8_f3,   //
    //                     &hex8_f4,   //
    //                     &hex8_f5,   //
    //                     &hex8_f6,   //
    //                     &hex8_f7);  //

    hex8_f0 = (1.0 - l_x) * (1.0 - l_y) * (1.0 - l_z);
    hex8_f1 = l_x * (1.0 - l_y) * (1.0 - l_z);
    hex8_f2 = l_x * l_y * (1.0 - l_z);
    hex8_f3 = (1.0 - l_x) * l_y * (1.0 - l_z);
    hex8_f4 = (1.0 - l_x) * (1.0 - l_y) * l_z;
    hex8_f5 = l_x * (1.0 - l_y) * l_z;
    hex8_f6 = l_x * l_y * l_z;
    hex8_f7 = (1.0 - l_x) * l_y * l_z;

    const real_t wf_quad_QW = wf_quad * QW_phys_hex;

    hex_element_field[0] += wf_quad_QW * hex8_f0;
    hex_element_field[1] += wf_quad_QW * hex8_f1;
    hex_element_field[2] += wf_quad_QW * hex8_f2;
    hex_element_field[3] += wf_quad_QW * hex8_f3;
    hex_element_field[4] += wf_quad_QW * hex8_f4;
    hex_element_field[5] += wf_quad_QW * hex8_f5;
    hex_element_field[6] += wf_quad_QW * hex8_f6;
    hex_element_field[7] += wf_quad_QW * hex8_f7;

    return (ijk_index_t){i, j, k, true};
}  // END transfer_weighted_field_tet4_to_hex

/////////////////////////////////////////////////////////
// is_hex_out_of_tet ////////////////////////////
/////////////////////////////////////////////////////////
bool                                                        //
is_hex_out_of_tet_norm_v(const real_t inv_J_tet[9],         //
                         const real_t tet_origin_x,         //
                         const real_t tet_origin_y,         //
                         const real_t tet_origin_z,         //
                         const real_t hex_vertices_x[8],    //
                         const real_t hex_vertices_y[8],    //
                         const real_t hex_vertices_z[8]) {  //

    /**
     * ****************************************************************************************
     * Check if a hexahedral element is completely outside a tetrahedral element
     * Using the inverse Jacobian of the tetrahedron to transform hex vertices to tet reference space
     * and check against tet reference space constraints.
     * This function return true if the hex is completely outside the tet.
     * And return false if it is unsure (partially inside, intersecting, completely inside, or UNDETECTED outside).
     * This must be used as a fast culling test before more expensive intersection tests.
     *
     * Tet reference space constraints for a point to be INSIDE:
     *   ref_x >= 0 AND ref_y >= 0 AND ref_z >= 0 AND (ref_x + ref_y + ref_z) <= 1
     *
     * A hex is completely OUTSIDE if ALL vertices violate at least ONE of these constraints:
     *   - All ref_x < 0 (all on negative x side)
     *   - All ref_y < 0 (all on negative y side)
     *   - All ref_z < 0 (all on negative z side)
     *   - All sum > 1 (all beyond the diagonal plane)
     * *****************************************************************************************
     */
    // Precompute inverse Jacobian components
    const real_t inv_J00 = inv_J_tet[0];
    const real_t inv_J01 = inv_J_tet[1];
    const real_t inv_J02 = inv_J_tet[2];
    const real_t inv_J10 = inv_J_tet[3];
    const real_t inv_J11 = inv_J_tet[4];
    const real_t inv_J12 = inv_J_tet[5];
    const real_t inv_J20 = inv_J_tet[6];
    const real_t inv_J21 = inv_J_tet[7];
    const real_t inv_J22 = inv_J_tet[8];

    // Use 512-bit AVX-512 vectors (8 doubles at a time - perfect for 8 vertices!)
    typedef double vec8 __attribute__((vector_size(8 * sizeof(double))));

    // Load all 8 hex vertices at once
    vec8 hex_x, hex_y, hex_z;
    for (int i = 0; i < 8; i++) {
        hex_x[i] = hex_vertices_x[i];
        hex_y[i] = hex_vertices_y[i];
        hex_z[i] = hex_vertices_z[i];
    }

    // Broadcast tet origin to all 8 lanes
    vec8 origin_x = {
            tet_origin_x, tet_origin_x, tet_origin_x, tet_origin_x, tet_origin_x, tet_origin_x, tet_origin_x, tet_origin_x};
    vec8 origin_y = {
            tet_origin_y, tet_origin_y, tet_origin_y, tet_origin_y, tet_origin_y, tet_origin_y, tet_origin_y, tet_origin_y};
    vec8 origin_z = {
            tet_origin_z, tet_origin_z, tet_origin_z, tet_origin_z, tet_origin_z, tet_origin_z, tet_origin_z, tet_origin_z};

    // Compute dx, dy, dz for all 8 vertices simultaneously
    vec8 dx = hex_x - origin_x;
    vec8 dy = hex_y - origin_y;
    vec8 dz = hex_z - origin_z;

    // Broadcast inverse Jacobian elements to all 8 lanes
    vec8 J00 = {inv_J00, inv_J00, inv_J00, inv_J00, inv_J00, inv_J00, inv_J00, inv_J00};
    vec8 J01 = {inv_J01, inv_J01, inv_J01, inv_J01, inv_J01, inv_J01, inv_J01, inv_J01};
    vec8 J02 = {inv_J02, inv_J02, inv_J02, inv_J02, inv_J02, inv_J02, inv_J02, inv_J02};
    vec8 J10 = {inv_J10, inv_J10, inv_J10, inv_J10, inv_J10, inv_J10, inv_J10, inv_J10};
    vec8 J11 = {inv_J11, inv_J11, inv_J11, inv_J11, inv_J11, inv_J11, inv_J11, inv_J11};
    vec8 J12 = {inv_J12, inv_J12, inv_J12, inv_J12, inv_J12, inv_J12, inv_J12, inv_J12};
    vec8 J20 = {inv_J20, inv_J20, inv_J20, inv_J20, inv_J20, inv_J20, inv_J20, inv_J20};
    vec8 J21 = {inv_J21, inv_J21, inv_J21, inv_J21, inv_J21, inv_J21, inv_J21, inv_J21};
    vec8 J22 = {inv_J22, inv_J22, inv_J22, inv_J22, inv_J22, inv_J22, inv_J22, inv_J22};

    // Transform all 8 vertices to reference space simultaneously
    vec8 ref_x = J00 * dx + J01 * dy + J02 * dz;
    vec8 ref_y = J10 * dx + J11 * dy + J12 * dz;
    vec8 ref_z = J20 * dx + J21 * dy + J22 * dz;

    // Compute sum for all 8 vertices
    vec8 sum_ref = ref_x + ref_y + ref_z;

    // Constants for comparison
    vec8 zero = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    vec8 one  = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    // Perform comparisons (returns vector of comparison results)
    vec8 neg_x   = ref_x < zero;
    vec8 neg_y   = ref_y < zero;
    vec8 neg_z   = ref_z < zero;
    vec8 out_sum = sum_ref > one;

    // Check if all 8 vertices satisfy each constraint
    // All bits set means all comparisons were true
    bool all_negative_x = (neg_x[0] && neg_x[1] && neg_x[2] && neg_x[3] && neg_x[4] && neg_x[5] && neg_x[6] && neg_x[7]);
    bool all_negative_y = (neg_y[0] && neg_y[1] && neg_y[2] && neg_y[3] && neg_y[4] && neg_y[5] && neg_y[6] && neg_y[7]);
    bool all_negative_z = (neg_z[0] && neg_z[1] && neg_z[2] && neg_z[3] && neg_z[4] && neg_z[5] && neg_z[6] && neg_z[7]);
    bool all_outside_sum =
            (out_sum[0] && out_sum[1] && out_sum[2] && out_sum[3] && out_sum[4] && out_sum[5] && out_sum[6] && out_sum[7]);

    // Hex is completely outside if ALL vertices violate at least one constraint together
    return (all_negative_x || all_negative_y || all_negative_z || all_outside_sum);

}  // END Function: is_hex_out_of_tet

// #if defined(__AVX512F__)
#include <immintrin.h>

// Macro for 3x1 matrix-vector multiplication using FP32 and YMM registers
#define MATVEC_MUL_3x1_PS(result, j0, j1, j2, d0, d1, d2)                    \
    do {                                                                     \
        __m256 _tmp0_##result = _mm256_mul_ps((j2), (d2));                   \
        __m256 _tmp1_##result = _mm256_fmadd_ps((j1), (d1), _tmp0_##result); \
        (result)              = _mm256_fmadd_ps((j0), (d0), _tmp1_##result); \
    } while (0)

bool                                                                    //
is_hex_out_of_tet_norm_v_avx512_fp32(const real_t inv_J_tet[9],         //
                                     const real_t tet_origin_x,         //
                                     const real_t tet_origin_y,         //
                                     const real_t tet_origin_z,         //
                                     const real_t hex_vertices_x[8],    //
                                     const real_t hex_vertices_y[8],    //
                                     const real_t hex_vertices_z[8]) {  //

    // Load all 8 vertices (8 floats = 256 bits)
    __m256 hex_x = _mm256_loadu_ps(hex_vertices_x);
    __m256 hex_y = _mm256_loadu_ps(hex_vertices_y);
    __m256 hex_z = _mm256_loadu_ps(hex_vertices_z);

    // Broadcast tet origin (single float to all 8 lanes)
    __m256 origin_x = _mm256_set1_ps(tet_origin_x);
    __m256 origin_y = _mm256_set1_ps(tet_origin_y);
    __m256 origin_z = _mm256_set1_ps(tet_origin_z);

    // Compute dx, dy, dz
    __m256 dx = _mm256_sub_ps(hex_x, origin_x);
    __m256 dy = _mm256_sub_ps(hex_y, origin_y);
    __m256 dz = _mm256_sub_ps(hex_z, origin_z);

    // Broadcast inverse Jacobian elements (single float to all 8 lanes)
    __m256 J00 = _mm256_set1_ps(inv_J_tet[0]);
    __m256 J01 = _mm256_set1_ps(inv_J_tet[1]);
    __m256 J02 = _mm256_set1_ps(inv_J_tet[2]);
    __m256 J10 = _mm256_set1_ps(inv_J_tet[3]);
    __m256 J11 = _mm256_set1_ps(inv_J_tet[4]);
    __m256 J12 = _mm256_set1_ps(inv_J_tet[5]);
    __m256 J20 = _mm256_set1_ps(inv_J_tet[6]);
    __m256 J21 = _mm256_set1_ps(inv_J_tet[7]);
    __m256 J22 = _mm256_set1_ps(inv_J_tet[8]);

    // Transform to reference space using FMA (FP32)
    __m256 ref_x, ref_y, ref_z;
    MATVEC_MUL_3x1_PS(ref_x, J00, J01, J02, dx, dy, dz);
    MATVEC_MUL_3x1_PS(ref_y, J10, J11, J12, dx, dy, dz);
    MATVEC_MUL_3x1_PS(ref_z, J20, J21, J22, dx, dy, dz);

    // Compute sum
    __m256 sum_ref = _mm256_add_ps(_mm256_add_ps(ref_x, ref_y), ref_z);

    // Constants
    __m256 zero = _mm256_setzero_ps();
    __m256 one  = _mm256_set1_ps(1.0f);

    // Perform comparisons and get masks (using k registers)
    // Note: AVX-512 mask registers work with 256-bit ops via _mm256_cmp_ps_mask
    __mmask8 k1 = _mm256_cmp_ps_mask(ref_x, zero, _CMP_LT_OQ);   // ref_x < 0
    __mmask8 k2 = _mm256_cmp_ps_mask(ref_y, zero, _CMP_LT_OQ);   // ref_y < 0
    __mmask8 k3 = _mm256_cmp_ps_mask(ref_z, zero, _CMP_LT_OQ);   // ref_z < 0
    __mmask8 k4 = _mm256_cmp_ps_mask(sum_ref, one, _CMP_GT_OQ);  // sum > 1

    // Reduce masks: check if ALL vertices satisfy each constraint
    // For 8 elements (8 floats), all bits set = 0xFF
    bool all_negative_x  = (k1 == 0xFF);
    bool all_negative_y  = (k2 == 0xFF);
    bool all_negative_z  = (k3 == 0xFF);
    bool all_outside_sum = (k4 == 0xFF);

    // Return true if at least one constraint is satisfied by all vertices
    return (all_negative_x || all_negative_y || all_negative_z || all_outside_sum);
}

// Macro for 3x1 matrix-vector multiplication using FP32 and YMM registers
#define MATVEC_MUL_3x1_PS_512(result, j0, j1, j2, d0, d1, d2)                \
    do {                                                                     \
        __m512 _tmp0_##result = _mm512_mul_ps((j2), (d2));                   \
        __m512 _tmp1_##result = _mm512_fmadd_ps((j1), (d1), _tmp0_##result); \
        (result)              = _mm512_fmadd_ps((j0), (d0), _tmp1_##result); \
    } while (0)

void is_hex_out_of_tet_norm_v_avx512_fp32_step2h(const float inv_J_tet[9],        //
                                                 const float tet_origin_x,        //
                                                 const float tet_origin_y,        //
                                                 const float tet_origin_z,        //
                                                 const float hex_vertices_x[16],  //
                                                 const float hex_vertices_y[16],  //
                                                 const float hex_vertices_z[16],  //
                                                 bool        in_out[2]) {                //

    // Load all 8 vertices (8 floats = 256 bits)
    __m512 hex_x = _mm512_loadu_ps(hex_vertices_x);
    __m512 hex_y = _mm512_loadu_ps(hex_vertices_y);
    __m512 hex_z = _mm512_loadu_ps(hex_vertices_z);

    // Broadcast tet origin (single float to all 8 lanes)
    __m512 origin_x = _mm512_set1_ps(tet_origin_x);
    __m512 origin_y = _mm512_set1_ps(tet_origin_y);
    __m512 origin_z = _mm512_set1_ps(tet_origin_z);

    // Compute dx, dy, dz
    __m512 dx = _mm512_sub_ps(hex_x, origin_x);
    __m512 dy = _mm512_sub_ps(hex_y, origin_y);
    __m512 dz = _mm512_sub_ps(hex_z, origin_z);

    // Broadcast inverse Jacobian elements (single float to all 8 lanes)
    __m512 J00 = _mm512_set1_ps(inv_J_tet[0]);
    __m512 J01 = _mm512_set1_ps(inv_J_tet[1]);
    __m512 J02 = _mm512_set1_ps(inv_J_tet[2]);
    __m512 J10 = _mm512_set1_ps(inv_J_tet[3]);
    __m512 J11 = _mm512_set1_ps(inv_J_tet[4]);
    __m512 J12 = _mm512_set1_ps(inv_J_tet[5]);
    __m512 J20 = _mm512_set1_ps(inv_J_tet[6]);
    __m512 J21 = _mm512_set1_ps(inv_J_tet[7]);
    __m512 J22 = _mm512_set1_ps(inv_J_tet[8]);

    // Transform to reference space using FMA (FP32)
    __m512 ref_x, ref_y, ref_z;
    MATVEC_MUL_3x1_PS_512(ref_x, J00, J01, J02, dx, dy, dz);
    MATVEC_MUL_3x1_PS_512(ref_y, J10, J11, J12, dx, dy, dz);
    MATVEC_MUL_3x1_PS_512(ref_z, J20, J21, J22, dx, dy, dz);

    // Compute sum
    __m512 sum_ref = _mm512_add_ps(_mm512_add_ps(ref_x, ref_y), ref_z);

    // Constants
    __m512 zero = _mm512_setzero_ps();
    __m512 one  = _mm512_set1_ps(1.0f);
    // Perform comparisons and get masks (using k registers)
    // Note: AVX-512 mask registers work with 256-bit ops via _mm256_cmp_ps_mask
    __mmask16 k1 = _mm512_cmp_ps_mask(ref_x, zero, _CMP_LT_OQ);   // ref_x < 0
    __mmask16 k2 = _mm512_cmp_ps_mask(ref_y, zero, _CMP_LT_OQ);   // ref_y < 0
    __mmask16 k3 = _mm512_cmp_ps_mask(ref_z, zero, _CMP_LT_OQ);   // ref_z < 0
    __mmask16 k4 = _mm512_cmp_ps_mask(sum_ref, one, _CMP_GT_OQ);  // sum > 1

    // Reduce masks: check if ALL vertices satisfy each constraint
    // For 8 elements (8 floats), all bits set = 0xFF
    bool all_negative_x_0  = (k1 == 0xFF00);
    bool all_negative_y_0  = (k2 == 0xFF00);
    bool all_negative_z_0  = (k3 == 0xFF00);
    bool all_outside_sum_0 = (k4 == 0xFF00);

    bool all_negative_x_1  = (k1 == 0x00FF);
    bool all_negative_y_1  = (k2 == 0x00FF);
    bool all_negative_z_1  = (k3 == 0x00FF);
    bool all_outside_sum_1 = (k4 == 0x00FF);

    // Return true if at least one constraint is satisfied by all vertices
    in_out[0] = (all_negative_x_0 || all_negative_y_0 || all_negative_z_0 || all_outside_sum_0);
    in_out[1] = (all_negative_x_1 || all_negative_y_1 || all_negative_z_1 || all_outside_sum_1);
}
// #endif

// #endif

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_adjoint_tet_quad_d ////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                            //
tet4_resample_field_adjoint_tet_norm(const real_t                    x0_n,     // Tet vertices //
                                     const real_t                    x1_n,     //
                                     const real_t                    x2_n,     //
                                     const real_t                    x3_n,     //
                                     const real_t                    y0_n,     //
                                     const real_t                    y1_n,     //
                                     const real_t                    y2_n,     //
                                     const real_t                    y3_n,     //
                                     const real_t                    z0_n,     //
                                     const real_t                    z1_n,     //
                                     const real_t                    z2_n,     //
                                     const real_t                    z3_n,     //
                                     const real_t                    wf0,      // Weighted field at tet vertices
                                     const real_t                    wf1,      //
                                     const real_t                    wf2,      //
                                     const real_t                    wf3,      //
                                     const ptrdiff_t                 stride1,  // Stride of hex grid
                                     const ptrdiff_t                 stride2,  //
                                     out_real_t* const SFEM_RESTRICT data) {   // Outut data array HEX
    // Placeholder implementation

#if SFEM_LOG_LEVEL >= 6
    printf("Stride0: %td, Stride1: %td \n", stride0, stride1);
#endif

    // const int off0 = 0;
    // const int off1 = stride0;
    // const int off2 = stride0 + stride1;
    // const int off3 = stride1;
    // const int off4 = 0;
    // const int off5 = stride0;
    // const int off6 = stride0 + stride1;
    // const int off7 = stride1;

    const int off0 = 0;
    const int off1 = 1;
    const int off2 = 1 + stride1;
    const int off3 = stride1;
    const int off4 = stride2;
    const int off5 = 1 + stride2;
    const int off6 = 1 + stride1 + stride2;
    const int off7 = stride1 + stride2;

    const int           dim_quad  = get_dim_qad();
    const real_t* const Q_nodes_x = Q_nodes_x_p;
    const real_t* const Q_nodes_y = Q_nodes_y_p;
    const real_t* const Q_nodes_z = Q_nodes_z_p;
    const real_t* const Q_weights = Q_weights_p;

    // const real_t inv_dx = 1.0;
    // const real_t inv_dy = 1.0;
    // const real_t inv_dz = 1.0;

    real_t    inv_J_tet[9];
    ptrdiff_t min_grid_x, max_grid_x;
    ptrdiff_t min_grid_y, max_grid_y;
    ptrdiff_t min_grid_z, max_grid_z;

    tet4_inv_Jacobian(x0_n,        //
                      x1_n,        //
                      x2_n,        //
                      x3_n,        //
                      y0_n,        //
                      y1_n,        //
                      y2_n,        //
                      y3_n,        //
                      z0_n,        //
                      z1_n,        //
                      z2_n,        //
                      z3_n,        //
                      inv_J_tet);  //

    compute_tet_bounding_box_norm(x0_n,          //
                                  x1_n,          //
                                  x2_n,          //
                                  x3_n,          //
                                  y0_n,          //
                                  y1_n,          //
                                  y2_n,          //
                                  y3_n,          //
                                  z0_n,          //
                                  z1_n,          //
                                  z2_n,          //
                                  z3_n,          //
                                  stride1,       //
                                  stride2,       //
                                  &min_grid_x,   //
                                  &max_grid_x,   //
                                  &min_grid_y,   //
                                  &max_grid_y,   //
                                  &min_grid_z,   //
                                  &max_grid_z);  //

    real_t hex_element_field[8] = {0.0};

    for (int k_grid_z = min_grid_z; k_grid_z < max_grid_z; k_grid_z++) {
        const real_t z_hex_min = ((real_t)k_grid_z);
        const real_t z_hex_max = z_hex_min + 1.0;

        for (int j_grid_y = min_grid_y; j_grid_y < max_grid_y; j_grid_y++) {
            const real_t y_hex_min = ((real_t)j_grid_y);
            const real_t y_hex_max = y_hex_min + 1.0;

            for (int i_grid_x = min_grid_x; i_grid_x < max_grid_x; i_grid_x++) {
                //
                const real_t x_hex_min = ((real_t)i_grid_x);
                const real_t x_hex_max = x_hex_min + 1.0;

                const real_t hex_vertices_x[8] = {x_hex_min,
                                                  x_hex_max,
                                                  x_hex_max,
                                                  x_hex_min,  //
                                                  x_hex_min,
                                                  x_hex_max,
                                                  x_hex_max,
                                                  x_hex_min};

                const real_t hex_vertices_y[8] = {y_hex_min,
                                                  y_hex_min,
                                                  y_hex_max,
                                                  y_hex_max,  //
                                                  y_hex_min,
                                                  y_hex_min,
                                                  y_hex_max,
                                                  y_hex_max};

                const real_t hex_vertices_z[8] = {z_hex_min,
                                                  z_hex_min,
                                                  z_hex_min,
                                                  z_hex_min,  //
                                                  z_hex_max,
                                                  z_hex_max,
                                                  z_hex_max,
                                                  z_hex_max};

                const bool is_out_of_tet = is_hex_out_of_tet_norm_v_avx512_fp32  //
                                                                                 // is_hex_out_of_tet  //
                        (inv_J_tet,                                              //
                         x0_n,                                                   //
                         y0_n,                                                   //
                         z0_n,                                                   //
                         hex_vertices_x,                                         //
                         hex_vertices_y,                                         //
                         hex_vertices_z);                                        //

                // printf("Is out of tet: %d \n", is_out_of_tet);

                if (is_out_of_tet) continue;  // c Skip this hex cell

                // Midpoint quadrature rule in 3D

                memset(hex_element_field, 0, 8 * sizeof(real_t));

                for (int q_ijk = 0; q_ijk < dim_quad; q_ijk++) {
                    quadrature_point_result_t Qpoint_phys =             //
                            transform_quadrature_point_norm(q_ijk,      //
                                                            Q_nodes_x,  //
                                                            Q_nodes_y,  //
                                                            Q_nodes_z,  //
                                                            Q_weights,  //
                                                            i_grid_x,   //
                                                            j_grid_y,   //
                                                            k_grid_z);  //

                    real_t Q_ref_x, Q_ref_y, Q_ref_z;

                    tet4_inv_transform_J(inv_J_tet,      // Inverse Jacobian matrix
                                         Qpoint_phys.x,  // Physical coordinates of the quadrature point
                                         Qpoint_phys.y,  //
                                         Qpoint_phys.z,  //
                                         x0_n,           //
                                         y0_n,           //
                                         z0_n,           //
                                         &Q_ref_x,       // Reference coordinates of the quadrature point
                                         &Q_ref_y,       //
                                         &Q_ref_z);      //

                    // for (int v = 0; v < 8; v++) hex_element_field[v] = 0.0;

                    ijk_index_t ijk_indices =                                             //
                            transfer_weighted_field_tet4_to_hex_norm(wf0,                 //
                                                                     wf1,                 //
                                                                     wf2,                 //
                                                                     wf3,                 //
                                                                     Qpoint_phys.x,       //
                                                                     Qpoint_phys.y,       //
                                                                     Qpoint_phys.z,       //
                                                                     Q_ref_x,             //
                                                                     Q_ref_y,             //
                                                                     Q_ref_z,             //
                                                                     Qpoint_phys.weight,  //
                                                                     hex_element_field);  //

                }  // END: for q_ijk

                const ptrdiff_t base_index = i_grid_x * 1 +        //
                                             j_grid_y * stride1 +  //
                                             k_grid_z * stride2;   //

#pragma omp critical
                {
                    data[base_index + off0] += hex_element_field[0];  //
                    data[base_index + off1] += hex_element_field[1];  //
                    data[base_index + off2] += hex_element_field[2];  //
                    data[base_index + off3] += hex_element_field[3];  //
                    data[base_index + off4] += hex_element_field[4];  //
                    data[base_index + off5] += hex_element_field[5];  //
                    data[base_index + off6] += hex_element_field[6];  //
                    data[base_index + off7] += hex_element_field[7];  //
                }  // END omp critical

                // }  // END: for i_grid_x

            }  // END: for k_grid_z
        }  // END: for i_grid_y
    }  // END: for j_grid_y

    return 0;
}  // END: Function: tet4_resample_field_adjoint_tet_quad_d

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_adjoint_tet_quad_d ////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                                   //
tet4_resample_field_adjoint_tet_norm_step2h(const real_t                    x0_n,     // Tet vertices //
                                            const real_t                    x1_n,     //
                                            const real_t                    x2_n,     //
                                            const real_t                    x3_n,     //
                                            const real_t                    y0_n,     //
                                            const real_t                    y1_n,     //
                                            const real_t                    y2_n,     //
                                            const real_t                    y3_n,     //
                                            const real_t                    z0_n,     //
                                            const real_t                    z1_n,     //
                                            const real_t                    z2_n,     //
                                            const real_t                    z3_n,     //
                                            const real_t                    wf0,      // Weighted field at tet vertices
                                            const real_t                    wf1,      //
                                            const real_t                    wf2,      //
                                            const real_t                    wf3,      //
                                            const ptrdiff_t                 stride1,  // Stride of hex grid
                                            const ptrdiff_t                 stride2,  //
                                            out_real_t* const SFEM_RESTRICT data) {   // Outut data array HEX
    // Placeholder implementation

#if SFEM_LOG_LEVEL >= 5
    // printf("Stride0: %td, Stride1: %td \n", stride0, stride1);
#endif

    // const int off0 = 0;
    // const int off1 = stride0;
    // const int off2 = stride0 + stride1;
    // const int off3 = stride1;
    // const int off4 = 0;
    // const int off5 = stride0;
    // const int off6 = stride0 + stride1;
    // const int off7 = stride1;

    const int off0 = 0;
    const int off1 = 1;
    const int off2 = 1 + stride1;
    const int off3 = stride1;
    const int off4 = stride2;
    const int off5 = 1 + stride2;
    const int off6 = 1 + stride1 + stride2;
    const int off7 = stride1 + stride2;

    const int           dim_quad  = get_dim_qad();
    const real_t* const Q_nodes_x = Q_nodes_x_p;
    const real_t* const Q_nodes_y = Q_nodes_y_p;
    const real_t* const Q_nodes_z = Q_nodes_z_p;
    const real_t* const Q_weights = Q_weights_p;

    // const real_t inv_dx = 1.0;
    // const real_t inv_dy = 1.0;
    // const real_t inv_dz = 1.0;

    real_t    inv_J_tet[9];
    ptrdiff_t min_grid_x, max_grid_x;
    ptrdiff_t min_grid_y, max_grid_y;
    ptrdiff_t min_grid_z, max_grid_z;

    tet4_inv_Jacobian(x0_n,        //
                      x1_n,        //
                      x2_n,        //
                      x3_n,        //
                      y0_n,        //
                      y1_n,        //
                      y2_n,        //
                      y3_n,        //
                      z0_n,        //
                      z1_n,        //
                      z2_n,        //
                      z3_n,        //
                      inv_J_tet);  //

    compute_tet_bounding_box_norm(x0_n,          //
                                  x1_n,          //
                                  x2_n,          //
                                  x3_n,          //
                                  y0_n,          //
                                  y1_n,          //
                                  y2_n,          //
                                  y3_n,          //
                                  z0_n,          //
                                  z1_n,          //
                                  z2_n,          //
                                  z3_n,          //
                                  stride1,       //
                                  stride2,       //
                                  &min_grid_x,   //
                                  &max_grid_x,   //
                                  &min_grid_y,   //
                                  &max_grid_y,   //
                                  &min_grid_z,   //
                                  &max_grid_z);  //

    real_t hex_element_field[8] = {0.0};

    int       size_x = max_grid_x - min_grid_x + 1;
    const int size_y = max_grid_y - min_grid_y + 1;
    const int size_z = max_grid_z - min_grid_z + 1;

    if (size_x % 2 != 0) {
        // Ensure size_x is even for step2h processing
        // This is a simple fix; in practice, you might want to handle this differently
        size_x += 1;
    }

    const int total_grid_points = size_x * size_y * size_z;

    bool* in_out_array = NULL;
    in_out_array       = (bool*)malloc(total_grid_points * sizeof(bool));
    // memset(in_out_array, 0, total_grid_points * sizeof(bool));

    for (int k_grid_z = min_grid_z; k_grid_z < max_grid_z; k_grid_z++) {
        const real_t z_hex_min = ((real_t)k_grid_z);
        const real_t z_hex_max = z_hex_min + 1.0;

        for (int j_grid_y = min_grid_y; j_grid_y < max_grid_y; j_grid_y++) {
            const real_t y_hex_min = ((real_t)j_grid_y);
            const real_t y_hex_max = y_hex_min + 1.0;

            for (int i_grid_x = min_grid_x; i_grid_x < max_grid_x; i_grid_x += 2) {
                //
                const real_t x_hex_min_0 = ((real_t)i_grid_x);
                const real_t x_hex_max_0 = x_hex_min_0 + 1.0;

                const real_t x_hex_min_1 = ((real_t)(i_grid_x + 1));
                const real_t x_hex_max_1 = x_hex_min_1 + 1.0;

                int local_idx = (i_grid_x - min_grid_x) +                   //
                                (j_grid_y - min_grid_y) * size_x +          //
                                (k_grid_z - min_grid_z) * size_x * size_y;  //

                const real_t hex_vertices_x[16] = {x_hex_min_0,
                                                   x_hex_max_0,
                                                   x_hex_max_0,
                                                   x_hex_min_0,  //
                                                   x_hex_min_0,
                                                   x_hex_max_0,
                                                   x_hex_max_0,
                                                   x_hex_min_0,  // second hex
                                                   x_hex_min_1,
                                                   x_hex_max_1,
                                                   x_hex_max_1,
                                                   x_hex_min_1,  //
                                                   x_hex_min_1,
                                                   x_hex_max_1,
                                                   x_hex_max_1,
                                                   x_hex_min_1};

                const real_t hex_vertices_y[16] = {y_hex_min,
                                                   y_hex_min,
                                                   y_hex_max,
                                                   y_hex_max,  //
                                                   y_hex_min,
                                                   y_hex_min,
                                                   y_hex_max,
                                                   y_hex_max,  // second hex
                                                   y_hex_min,
                                                   y_hex_min,
                                                   y_hex_max,
                                                   y_hex_max,  //
                                                   y_hex_min,
                                                   y_hex_min,
                                                   y_hex_max,
                                                   y_hex_max};

                const real_t hex_vertices_z[16] = {z_hex_min,
                                                   z_hex_min,
                                                   z_hex_min,
                                                   z_hex_min,  //
                                                   z_hex_max,
                                                   z_hex_max,
                                                   z_hex_max,
                                                   z_hex_max,
                                                   z_hex_min,  // second hex
                                                   z_hex_min,
                                                   z_hex_min,
                                                   z_hex_min,  //
                                                   z_hex_max,
                                                   z_hex_max,
                                                   z_hex_max,
                                                   z_hex_max};

                is_hex_out_of_tet_norm_v_avx512_fp32_step2h  //
                                                             // is_hex_out_of_tet_step2h//
                        (inv_J_tet,                          //
                         x0_n,                               //
                         y0_n,                               //
                         z0_n,                               //
                         (const real_t*)hex_vertices_x,      //
                         (const real_t*)hex_vertices_y,      //
                         (const real_t*)hex_vertices_z,      //
                         &in_out_array[local_idx]);          //
            }
        }
    }

    for (int k_grid_z = min_grid_z; k_grid_z < max_grid_z; k_grid_z++) {
        const real_t z_hex_min = ((real_t)k_grid_z);
        const real_t z_hex_max = z_hex_min + 1.0;

        const real_t hex_vertices_z[8] = {z_hex_min,
                                          z_hex_min,
                                          z_hex_min,
                                          z_hex_min,  //
                                          z_hex_max,
                                          z_hex_max,
                                          z_hex_max,
                                          z_hex_max};

        for (int j_grid_y = min_grid_y; j_grid_y < max_grid_y; j_grid_y++) {
            const real_t y_hex_min = ((real_t)j_grid_y);
            const real_t y_hex_max = y_hex_min + 1.0;

            const real_t hex_vertices_y[8] = {y_hex_min,
                                              y_hex_min,
                                              y_hex_max,
                                              y_hex_max,  //
                                              y_hex_min,
                                              y_hex_min,
                                              y_hex_max,
                                              y_hex_max};

            for (int i_grid_x = min_grid_x; i_grid_x < max_grid_x; i_grid_x++) {
                //
                const real_t x_hex_min = ((real_t)i_grid_x);
                const real_t x_hex_max = x_hex_min + 1.0;

                int local_idx = (i_grid_x - min_grid_x) +                   //
                                (j_grid_y - min_grid_y) * size_x +          //
                                (k_grid_z - min_grid_z) * size_x * size_y;  //

                const real_t hex_vertices_x[8] = {x_hex_min,
                                                  x_hex_max,
                                                  x_hex_max,
                                                  x_hex_min,  //
                                                  x_hex_min,
                                                  x_hex_max,
                                                  x_hex_max,
                                                  x_hex_min};

                // printf("Is out of tet: %d \n", is_out_of_tet);

                if (in_out_array[local_idx]) continue;  // c Skip this hex cell

                // Midpoint quadrature rule in 3D

                memset(hex_element_field, 0, 8 * sizeof(real_t));

                for (int q_ijk = 0; q_ijk < dim_quad; q_ijk++) {
                    quadrature_point_result_t Qpoint_phys =             //
                            transform_quadrature_point_norm(q_ijk,      //
                                                            Q_nodes_x,  //
                                                            Q_nodes_y,  //
                                                            Q_nodes_z,  //
                                                            Q_weights,  //
                                                            i_grid_x,   //
                                                            j_grid_y,   //
                                                            k_grid_z);  //

                    real_t Q_ref_x, Q_ref_y, Q_ref_z;

                    tet4_inv_transform_J(inv_J_tet,      // Inverse Jacobian matrix
                                         Qpoint_phys.x,  // Physical coordinates of the quadrature point
                                         Qpoint_phys.y,  //
                                         Qpoint_phys.z,  //
                                         x0_n,           //
                                         y0_n,           //
                                         z0_n,           //
                                         &Q_ref_x,       // Reference coordinates of the quadrature point
                                         &Q_ref_y,       //
                                         &Q_ref_z);      //

                    // for (int v = 0; v < 8; v++) hex_element_field[v] = 0.0;

                    ijk_index_t ijk_indices =                                             //
                            transfer_weighted_field_tet4_to_hex_norm(wf0,                 //
                                                                     wf1,                 //
                                                                     wf2,                 //
                                                                     wf3,                 //
                                                                     Qpoint_phys.x,       //
                                                                     Qpoint_phys.y,       //
                                                                     Qpoint_phys.z,       //
                                                                     Q_ref_x,             //
                                                                     Q_ref_y,             //
                                                                     Q_ref_z,             //
                                                                     Qpoint_phys.weight,  //
                                                                     hex_element_field);  //

                }  // END: for q_ijk

                const ptrdiff_t base_index = i_grid_x * 1 +        //
                                             j_grid_y * stride1 +  //
                                             k_grid_z;             //

                data[base_index + off0] += hex_element_field[0];  //
                data[base_index + off1] += hex_element_field[1];  //
                data[base_index + off2] += hex_element_field[2];  //
                data[base_index + off3] += hex_element_field[3];  //
                data[base_index + off4] += hex_element_field[4];  //
                data[base_index + off5] += hex_element_field[5];  //
                data[base_index + off6] += hex_element_field[6];  //
                data[base_index + off7] += hex_element_field[7];  //

            }  // END: for k_grid_z
        }  // END: for i_grid_y
    }  // END: for j_grid_y

    free(in_out_array);

    return 0;
}  // END: Function: tet4_resample_field_adjoint_tet_quad_d

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_refine_adjoint_hyteg ////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                                                  //
tet4_resample_field_adjoint_hex_quad_norm(const ptrdiff_t                      start_element,        // Mesh
                                          const ptrdiff_t                      end_element,          //
                                          const ptrdiff_t                      nnodes,               //
                                          const idx_t** const SFEM_RESTRICT    elems,                //
                                          const geom_t** const SFEM_RESTRICT   xyz,                  //
                                          const ptrdiff_t* const SFEM_RESTRICT n,                    // SDF
                                          const ptrdiff_t* const SFEM_RESTRICT stride,               //
                                          const geom_t* const SFEM_RESTRICT    origin,               //
                                          const geom_t* const SFEM_RESTRICT    delta,                //
                                          const real_t* const SFEM_RESTRICT    weighted_field,       // Input weighted field
                                          const mini_tet_parameters_t          mini_tet_parameters,  //
                                          real_t* const SFEM_RESTRICT          data) {                        //

    init_quad_points_hex_qtet(2);  //

#pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        // Read the element vertex indices
        idx_t ev[4];

        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }  // END: for vq

#if SFEM_LOG_LEVEL >= 5
        if (element_i % 100000 == 0) {
            printf("*** Processing element %td / %td \n", element_i, end_element);
        }
#endif

        // Read the coordinates of the vertices of the tetrahedron
        // In the physical space
        const real_t x0_n = xyz[0][ev[0]];
        const real_t x1_n = xyz[0][ev[1]];
        const real_t x2_n = xyz[0][ev[2]];
        const real_t x3_n = xyz[0][ev[3]];

        const real_t y0_n = xyz[1][ev[0]];
        const real_t y1_n = xyz[1][ev[1]];
        const real_t y2_n = xyz[1][ev[2]];
        const real_t y3_n = xyz[1][ev[3]];

        const real_t z0_n = xyz[2][ev[0]];
        const real_t z1_n = xyz[2][ev[1]];
        const real_t z2_n = xyz[2][ev[2]];
        const real_t z3_n = xyz[2][ev[3]];

        const real_t wf0 = weighted_field[ev[0]];  // Weighted field at vertex 0
        const real_t wf1 = weighted_field[ev[1]];  // Weighted field at vertex 1
        const real_t wf2 = weighted_field[ev[2]];  // Weighted field at vertex 2
        const real_t wf3 = weighted_field[ev[3]];  // Weighted field at vertex 3

        tet4_resample_field_adjoint_tet_norm(x0_n,       //
                                             x1_n,       //
                                             x2_n,       //
                                             x3_n,       //
                                             y0_n,       //
                                             y1_n,       //
                                             y2_n,       //
                                             y3_n,       //
                                             z0_n,       //
                                             z1_n,       //
                                             z2_n,       //
                                             z3_n,       //
                                             wf0,        //
                                             wf1,        //
                                             wf2,        //
                                             wf3,        //
                                             stride[1],  //
                                             stride[2],  //
                                             data);      //
    }

}  // END: Function: tet4_resample_field_adjoint_hex_quad_norm