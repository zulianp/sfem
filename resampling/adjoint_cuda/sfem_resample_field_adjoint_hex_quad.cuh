#ifndef __SFEM_RESAMPLE_FIELD_ADJOINT_HEX_QUAD_CUH__
#define __SFEM_RESAMPLE_FIELD_ADJOINT_HEX_QUAD_CUH__

#include "sfem_adjoint_mini_tet.cuh"
#include "sfem_adjoint_mini_tet_fun.cuh"
#include "sfem_resample_field_cuda_fun.cuh"
#include "sfem_resample_field_quad_rules.cuh"

typedef enum { TET_QUAD_MIDPOINT_NQP } tet_quad_midpoint_nqp_gpu_t;

#define LANES_PER_TILE_HEX_QUAD (1)  // Number of lanes per CUDA tile for hex quad elements

template <typename FloatType>
struct quadrature_point_result_gpu_t {
    FloatType physical_x;
    FloatType physical_y;
    FloatType physical_z;  // Physical coordinates
    FloatType weight;      // Physical weight
    bool      is_inside;   // Containment result
};  // END struct: quadrature_point_result_gpu_t

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_inv_transform_J_gpu //////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType>
__device__ void                                         //
tet4_inv_transform_J_gpu(const FloatType* const J_inv,  //
                         const FloatType        pfx,    //
                         const FloatType        pfy,    //
                         const FloatType        pfz,    //
                         const FloatType        px0,    //
                         const FloatType        py0,    //
                         const FloatType        pz0,    //
                         FloatType&             out_x,  //
                         FloatType&             out_y,  //
                         FloatType&             out_z) {            //
    //
    //

    /**
     ****************************************************************************************
    \begin{bmatrix}
    out_x \\
    out_y \\
    out_z
    \end{bmatrix}
    =
    J^{-1} \cdot
    \begin{bmatrix}
    pfx - px0 \\
    pfy - py0 \\
    pfz - pz0
    \end{bmatrix}
    *************************************************************************************************
    */

    // Compute the difference between the physical point and the origin (common subexpressions)
    const FloatType dx = pfx - px0;
    const FloatType dy = pfy - py0;
    const FloatType dz = pfz - pz0;

    // Preload inverse Jacobian components for better register utilization
    const FloatType J00 = J_inv[0];
    const FloatType J01 = J_inv[1];
    const FloatType J02 = J_inv[2];
    const FloatType J10 = J_inv[3];
    const FloatType J11 = J_inv[4];
    const FloatType J12 = J_inv[5];
    const FloatType J20 = J_inv[6];
    const FloatType J21 = J_inv[7];
    const FloatType J22 = J_inv[8];

    // Apply the inverse transformation using FMA for better precision
    out_x = fast_fma(J00, dx, fast_fma(J01, dy, J02 * dz));
    out_y = fast_fma(J10, dx, fast_fma(J11, dy, J12 * dz));
    out_z = fast_fma(J20, dx, fast_fma(J21, dy, J22 * dz));
}  // END Function: tet4_inv_transform_J_gpu

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// transfer_weighted_field_tet4_to_hex_gpu //////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType, typename IntType = ptrdiff_t>
__device__ __inline__ bool                                                       //
transfer_weighted_field_tet4_to_hex_ckp_gpu(const FloatType  inv_J_tet[9],       //
                                            const FloatType  wf0,                //
                                            const FloatType  wf1,                //
                                            const FloatType  wf2,                //
                                            const FloatType  wf3,                //
                                            const FloatType  q_phys_x,           //
                                            const FloatType  q_phys_y,           //
                                            const FloatType  q_phys_z,           //
                                            const FloatType  QW_phys_hex,        //
                                            const FloatType  x0_n,               //
                                            const FloatType  y0_n,               //
                                            const FloatType  z0_n,               //
                                            const FloatType  ox,                 //
                                            const FloatType  oy,                 //
                                            const FloatType  oz,                 //
                                            const FloatType  inv_dx,             //
                                            const FloatType  inv_dy,             //
                                            const FloatType  inv_dz,             //
                                            FloatType* const hex_element_field,  //
                                            IntType&         out_i,              //
                                            IntType&         out_j,              //
                                            IntType&         out_k) {                    //

    // Compute the weighted contribution from the tetrahedron
    // Using linear shape functions for tetrahedron

    FloatType q_ref_x;  //
    FloatType q_ref_y;  //
    FloatType q_ref_z;  //

    tet4_inv_transform_J_gpu(inv_J_tet,  //
                             q_phys_x,   //
                             q_phys_y,   //
                             q_phys_z,   //
                             x0_n,       //
                             y0_n,       //
                             z0_n,       //
                             q_ref_x,    //
                             q_ref_y,    //
                             q_ref_z);   //

    if (q_ref_x < 0.0 || q_ref_y < 0.0 || q_ref_z < 0.0 || (q_ref_x + q_ref_y + q_ref_z) > 1.0) {
        return false;
    }  // END if (outside tet)

    const FloatType grid_x = (q_phys_x - ox) * inv_dx;
    const FloatType grid_y = (q_phys_y - oy) * inv_dy;
    const FloatType grid_z = (q_phys_z - oz) * inv_dz;

    const IntType i = (IntType)fast_floor(grid_x);
    const IntType j = (IntType)fast_floor(grid_y);
    const IntType k = (IntType)fast_floor(grid_z);

    const FloatType l_x = grid_x - (FloatType)i;
    const FloatType l_y = grid_y - (FloatType)j;
    const FloatType l_z = grid_z - (FloatType)k;

    // Tet4 linear shape functions in reference coordinates
    const FloatType f0 = FloatType(1.0) - q_ref_x - q_ref_y - q_ref_z;
    const FloatType f1 = q_ref_x;
    const FloatType f2 = q_ref_y;
    const FloatType f3 = q_ref_z;

    // Interpolate weighted field at quadrature point using FMA for precision
    const FloatType wf_quad = fast_fma(f0, wf0, fast_fma(f1, wf1, fast_fma(f2, wf2, f3 * wf3)));

    // Quadrature point (local coordinates)
    // With respect to the hat functions of a cube element
    // In a local coordinate system
    //
    // Precompute common subexpressions for hex8 shape functions
    const FloatType one_minus_lx = FloatType(1.0) - l_x;
    const FloatType one_minus_ly = FloatType(1.0) - l_y;
    const FloatType one_minus_lz = FloatType(1.0) - l_z;

    // Precompute products that are reused multiple times
    const FloatType lx_ly           = l_x * l_y;
    const FloatType lx_lz           = l_x * l_z;
    const FloatType ly_lz           = l_y * l_z;
    const FloatType one_minus_lx_ly = one_minus_lx * one_minus_ly;
    const FloatType one_minus_lx_lz = one_minus_lx * l_z;
    const FloatType lx_one_minus_ly = l_x * one_minus_ly;

    // Compute hex8 shape functions using precomputed subexpressions
    const FloatType hex8_f0 = one_minus_lx_ly * one_minus_lz;
    const FloatType hex8_f1 = lx_one_minus_ly * one_minus_lz;
    const FloatType hex8_f2 = lx_ly * one_minus_lz;
    const FloatType hex8_f3 = one_minus_lx * l_y * one_minus_lz;
    const FloatType hex8_f4 = one_minus_lx_ly * l_z;
    const FloatType hex8_f5 = lx_one_minus_ly * l_z;
    const FloatType hex8_f6 = lx_ly * l_z;
    const FloatType hex8_f7 = one_minus_lx_lz * l_y;

    // Accumulate contributions to hex element field using FMA for precision
    const FloatType contribution = wf_quad * QW_phys_hex;

    hex_element_field[0] += contribution * hex8_f0;
    hex_element_field[1] += contribution * hex8_f1;
    hex_element_field[2] += contribution * hex8_f2;
    hex_element_field[3] += contribution * hex8_f3;
    hex_element_field[4] += contribution * hex8_f4;
    hex_element_field[5] += contribution * hex8_f5;
    hex_element_field[6] += contribution * hex8_f6;
    hex_element_field[7] += contribution * hex8_f7;

    // Return grid indices via output reference parameters
    out_i = i;
    out_j = j;
    out_k = k;

    return true;
}  // END Function: transfer_weighted_field_tet4_to_hex_gpu

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// transfer_weighted_field_tet4_to_hex_gpu //////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType, typename IntType = ptrdiff_t>
__device__ __inline__ void                                                   //
transfer_weighted_field_tet4_to_hex_gpu(const FloatType  wf0,                //
                                        const FloatType  wf1,                //
                                        const FloatType  wf2,                //
                                        const FloatType  wf3,                //
                                        const FloatType  q_phys_x,           //
                                        const FloatType  q_phys_y,           //
                                        const FloatType  q_phys_z,           //
                                        const FloatType  q_ref_x,            //
                                        const FloatType  q_ref_y,            //
                                        const FloatType  q_ref_z,            //
                                        const FloatType  QW_phys_hex,        //
                                        const FloatType  ox,                 //
                                        const FloatType  oy,                 //
                                        const FloatType  oz,                 //
                                        const FloatType  dx,                 //
                                        const FloatType  dy,                 //
                                        const FloatType  dz,                 //
                                        FloatType* const hex_element_field,  //
                                        IntType&         out_i,              //
                                        IntType&         out_j,              //
                                        IntType&         out_k) {                    //

    // Compute the weighted contribution from the tetrahedron
    // Using linear shape functions for tetrahedron

    const FloatType grid_x = (q_phys_x - ox) / dx;
    const FloatType grid_y = (q_phys_y - oy) / dy;
    const FloatType grid_z = (q_phys_z - oz) / dz;

    const IntType i = (IntType)fast_floor(grid_x);
    const IntType j = (IntType)fast_floor(grid_y);
    const IntType k = (IntType)fast_floor(grid_z);

    const FloatType l_x = grid_x - (FloatType)i;
    const FloatType l_y = grid_y - (FloatType)j;
    const FloatType l_z = grid_z - (FloatType)k;

    // Tet4 linear shape functions in reference coordinates
    const FloatType f0 = FloatType(1.0) - q_ref_x - q_ref_y - q_ref_z;
    const FloatType f1 = q_ref_x;
    const FloatType f2 = q_ref_y;
    const FloatType f3 = q_ref_z;

    // Interpolate weighted field at quadrature point using FMA for precision
    const FloatType wf_quad = fast_fma(f0, wf0, fast_fma(f1, wf1, fast_fma(f2, wf2, f3 * wf3)));

    // Quadrature point (local coordinates)
    // With respect to the hat functions of a cube element
    // In a local coordinate system
    //
    // Precompute common subexpressions for hex8 shape functions
    const FloatType one_minus_lx = FloatType(1.0) - l_x;
    const FloatType one_minus_ly = FloatType(1.0) - l_y;
    const FloatType one_minus_lz = FloatType(1.0) - l_z;

    // Precompute products that are reused multiple times
    const FloatType lx_ly           = l_x * l_y;
    const FloatType lx_lz           = l_x * l_z;
    const FloatType ly_lz           = l_y * l_z;
    const FloatType one_minus_lx_ly = one_minus_lx * one_minus_ly;
    const FloatType one_minus_lx_lz = one_minus_lx * l_z;
    const FloatType lx_one_minus_ly = l_x * one_minus_ly;

    // Compute hex8 shape functions using precomputed subexpressions
    const FloatType hex8_f0 = one_minus_lx_ly * one_minus_lz;
    const FloatType hex8_f1 = lx_one_minus_ly * one_minus_lz;
    const FloatType hex8_f2 = lx_ly * one_minus_lz;
    const FloatType hex8_f3 = one_minus_lx * l_y * one_minus_lz;
    const FloatType hex8_f4 = one_minus_lx_ly * l_z;
    const FloatType hex8_f5 = lx_one_minus_ly * l_z;
    const FloatType hex8_f6 = lx_ly * l_z;
    const FloatType hex8_f7 = one_minus_lx_lz * l_y;

    // Accumulate contributions to hex element field using FMA for precision
    const FloatType contribution = wf_quad * QW_phys_hex;

    hex_element_field[0] = fast_fma(contribution, hex8_f0, hex_element_field[0]);
    hex_element_field[1] = fast_fma(contribution, hex8_f1, hex_element_field[1]);
    hex_element_field[2] = fast_fma(contribution, hex8_f2, hex_element_field[2]);
    hex_element_field[3] = fast_fma(contribution, hex8_f3, hex_element_field[3]);
    hex_element_field[4] = fast_fma(contribution, hex8_f4, hex_element_field[4]);
    hex_element_field[5] = fast_fma(contribution, hex8_f5, hex_element_field[5]);
    hex_element_field[6] = fast_fma(contribution, hex8_f6, hex_element_field[6]);
    hex_element_field[7] = fast_fma(contribution, hex8_f7, hex_element_field[7]);

    // Return grid indices via output reference parameters
    out_i = i;
    out_j = j;
    out_k = k;
}  // END Function: transfer_weighted_field_tet4_to_hex_gpu

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// is_point_in_tet_n_gpu /////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType>
__device__ void                                                            //
is_point_in_tet_n_gpu(const int                 N,                         //
                      const FloatType           tet4_faces_normals[4][3],  //
                      const FloatType           faces_centroids[4][3],     //
                      const FloatType* const    ptx,                       //
                      const FloatType* const    pty,                       //
                      const FloatType* const    ptz,                       //
                      bool* const SFEM_RESTRICT results) {                 //

    for (int i = 0; i < N; i++) {
        results[i] = true;

        for (int f = 0; f < 4; f++) {
            const FloatType nx = tet4_faces_normals[f][0];
            const FloatType ny = tet4_faces_normals[f][1];
            const FloatType nz = tet4_faces_normals[f][2];

            const FloatType cx = faces_centroids[f][0];
            const FloatType cy = faces_centroids[f][1];
            const FloatType cz = faces_centroids[f][2];

            // Vector from face centroid to point
            const FloatType vx = ptx[i] - cx;
            const FloatType vy = pty[i] - cy;
            const FloatType vz = ptz[i] - cz;

            // Use fast_fma for better precision in dot product computation
            const FloatType dot = fast_fma(nx, vx, fast_fma(ny, vy, nz * vz));

            // For outward normals: point is inside if dot product is negative
            // (vector to point is opposite to outward normal)
            if (dot > FloatType(0.0)) {
                results[i] = false;
                break;  // No need to check other faces
            }  // END if (dot > 0.0)
        }  // END for (int f = 0; f < 4; f++)
    }  // END for (int i = 0; i < N; i++)
}  // END Function: is_point_in_tet_n_gpu

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// transform_and_check_quadrature_point_n_gpu ///////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType, typename IntType = ptrdiff_t>
__device__                                                                                           //
        quadrature_point_result_gpu_t<FloatType>                                                     //
        transform_and_check_quadrature_point_n_gpu(const int              q_ijk,                     //
                                                   const FloatType        tet4_faces_normals[4][3],  //
                                                   const FloatType        faces_centroids[4][3],     //
                                                   const FloatType* const Q_nodes_x,                 //
                                                   const FloatType* const Q_nodes_y,                 //
                                                   const FloatType* const Q_nodes_z,                 //
                                                   const FloatType* const Q_weights,                 //
                                                   const FloatType        origin0,                   //
                                                   const FloatType        origin1,                   //
                                                   const FloatType        origin2,                   //
                                                   const FloatType        dx,                        //
                                                   const FloatType        dy,                        //
                                                   const FloatType        dz,                        //
                                                   const IntType          i_grid,                    //
                                                   const IntType          j_grid,                    //
                                                   const IntType          k_grid,                    //
                                                   const FloatType        tet_vertices_x[4],         //
                                                   const FloatType        tet_vertices_y[4],         //
                                                   const FloatType        tet_vertices_z[4]) {              //

    quadrature_point_result_gpu_t<FloatType> result;

    // Transform to physical coordinates
    // Q_nodes are in [0,1] reference space, need to map to the specific grid cell [i_grid, i_grid+1]
    result.physical_x = fast_fma((FloatType)i_grid + Q_nodes_x[q_ijk], dx, origin0);
    result.physical_y = fast_fma((FloatType)j_grid + Q_nodes_y[q_ijk], dy, origin1);
    result.physical_z = fast_fma((FloatType)k_grid + Q_nodes_z[q_ijk], dz, origin2);

    // Check if point is inside tetrahedron
    is_point_in_tet_n_gpu<FloatType>(1,                   //
                                     tet4_faces_normals,  //
                                     faces_centroids,     //
                                     &result.physical_x,  //
                                     &result.physical_y,  //
                                     &result.physical_z,  //
                                     &result.is_inside);  //

    // Compute physical weight using FMA for precision
    // Q_weights[q_ijk] is already the product of 3 1D weights, so just scale by volume
    result.weight = Q_weights[q_ijk] * dx * dy * dz;

    return result;
}  // END Function: transform_and_check_quadrature_point_n_gpu

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// transform_quadrature_point_n_gpu //////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType, typename IntType = ptrdiff_t>
__device__                                                           //
        quadrature_point_result_gpu_t<FloatType>                     //
        transform_quadrature_point_n_gpu(const FloatType Q_nodes_x,  //
                                         const FloatType Q_nodes_y,  //
                                         const FloatType Q_nodes_z,  //
                                         const FloatType Q_weights,  //
                                         const FloatType origin0,    //
                                         const FloatType origin1,    //
                                         const FloatType origin2,    //
                                         const FloatType dx,         //
                                         const FloatType dy,         //
                                         const FloatType dz,         //
                                         const IntType   i_grid,     //
                                         const IntType   j_grid,     //
                                         const IntType   k_grid) {     //

    quadrature_point_result_gpu_t<FloatType> result;
    result.is_inside = false;

    // Transform to physical coordinates
    // Q_nodes are in [0,1] reference space, need to map to the specific grid cell [i_grid, i_grid+1]
    result.physical_x = fast_fma((FloatType)i_grid + Q_nodes_x, dx, origin0);
    result.physical_y = fast_fma((FloatType)j_grid + Q_nodes_y, dy, origin1);
    result.physical_z = fast_fma((FloatType)k_grid + Q_nodes_z, dz, origin2);

    // Compute physical weight using FMA for precision
    // Q_weights[q_ijk] is already the product of 3 1D weights, so just scale by volume
    result.weight = Q_weights * dx * dy * dz;

    return result;
}  // END Function: transform_and_check_quadrature_point_n_gpu

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// is_hex_out_of_tet_gpu /////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType>
__device__ bool                                             //
is_hex_out_of_tet_gpu(const FloatType inv_J_tet[9],         //
                      const FloatType tet_origin_x,         //
                      const FloatType tet_origin_y,         //
                      const FloatType tet_origin_z,         //
                      const FloatType hex_vertices_x[8],    //
                      const FloatType hex_vertices_y[8],    //
                      const FloatType hex_vertices_z[8]) {  //

    /**
     * ****************************************************************************************
     * Check if a hexahedral element is completely outside a tetrahedral element
     * Using the inverse Jacobian of the tetrahedron to transform hex vertices to tet reference space
     * and check against tet reference space constraints.
     * This function returns true if the hex is completely outside the tet.
     * And returns false if it is unsure (partially inside, intersecting, completely inside, or UNDETECTED outside).
     * This must be used as a fast culling test before more expensive intersection tests.
     * *****************************************************************************************
     */

    // Precompute inverse Jacobian components for better cache utilization
    const FloatType inv_J00 = inv_J_tet[0];
    const FloatType inv_J01 = inv_J_tet[1];
    const FloatType inv_J02 = inv_J_tet[2];
    const FloatType inv_J10 = inv_J_tet[3];
    const FloatType inv_J11 = inv_J_tet[4];
    const FloatType inv_J12 = inv_J_tet[5];
    const FloatType inv_J20 = inv_J_tet[6];
    const FloatType inv_J21 = inv_J_tet[7];
    const FloatType inv_J22 = inv_J_tet[8];

    // Track if all vertices violate each constraint
    int all_negative_x  = 1;  // All ref_x < 0
    int all_negative_y  = 1;  // All ref_y < 0
    int all_negative_z  = 1;  // All ref_z < 0
    int all_outside_sum = 1;  // All ref_x + ref_y + ref_z > 1

#pragma unroll
    for (int v = 0; v < 8; v++) {
        // Transform hex vertex to tet reference space
        const FloatType dx = hex_vertices_x[v] - tet_origin_x;
        const FloatType dy = hex_vertices_y[v] - tet_origin_y;
        const FloatType dz = hex_vertices_z[v] - tet_origin_z;

        // Use fast_fma for better precision in matrix-vector multiplication
        const FloatType ref_x = fast_fma(inv_J00, dx, fast_fma(inv_J01, dy, inv_J02 * dz));
        const FloatType ref_y = fast_fma(inv_J10, dx, fast_fma(inv_J11, dy, inv_J12 * dz));
        const FloatType ref_z = fast_fma(inv_J20, dx, fast_fma(inv_J21, dy, inv_J22 * dz));

        // Point is inside tet if: ref_x >= 0 AND ref_y >= 0 AND ref_z >= 0 AND ref_x + ref_y + ref_z <= 1
        // Point is outside if it violates ANY of these constraints

        // Update flags - use bitwise AND for branchless execution
        all_negative_x &= (ref_x < FloatType(0.0));
        all_negative_y &= (ref_y < FloatType(0.0));
        all_negative_z &= (ref_z < FloatType(0.0));

        const FloatType sum_ref = ref_x + ref_y + ref_z;
        all_outside_sum &= (sum_ref > FloatType(1.0));

        // Early exit optimization: if we know hex is not completely outside, stop
        if (!all_negative_x &&   //
            !all_negative_y &&   //
            !all_negative_z &&   //
            !all_outside_sum) {  //
            return false;
        }  // END if early exit
    }  // END for (int v = 0; v < 8; v++)

    // Hex is completely outside if ALL vertices violate at least one constraint
    return (all_negative_x || all_negative_y || all_negative_z || all_outside_sum);
}  // END Function: is_hex_out_of_tet_gpu

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// is_hex_out_of_tet_gpu /////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType>
__device__ bool                                                    //
is_hex_out_of_tet_vertices_gpu(const FloatType inv_J_tet[9],       //
                               const FloatType tet_origin_x,       //
                               const FloatType tet_origin_y,       //
                               const FloatType tet_origin_z,       //
                               const FloatType hex_vertices_x[8],  //
                               const FloatType hex_vertices_y[8],  //
                               const FloatType hex_vertices_z[8],  //
                               FloatType       is_vertex_outside[8]) {   //

    /**
     * ****************************************************************************************
     * Check if a hexahedral element is completely outside a tetrahedral element
     * Using the inverse Jacobian of the tetrahedron to transform hex vertices to tet reference space
     * and check against tet reference space constraints.
     * This function returns true if the hex is completely outside the tet.
     * And returns false if it is unsure (partially inside, intersecting, completely inside, or UNDETECTED outside).
     * This must be used as a fast culling test before more expensive intersection tests.
     * *****************************************************************************************
     */

    // Precompute inverse Jacobian components for better cache utilization
    const FloatType inv_J00 = inv_J_tet[0];
    const FloatType inv_J01 = inv_J_tet[1];
    const FloatType inv_J02 = inv_J_tet[2];
    const FloatType inv_J10 = inv_J_tet[3];
    const FloatType inv_J11 = inv_J_tet[4];
    const FloatType inv_J12 = inv_J_tet[5];
    const FloatType inv_J20 = inv_J_tet[6];
    const FloatType inv_J21 = inv_J_tet[7];
    const FloatType inv_J22 = inv_J_tet[8];

    // Track if all vertices violate each constraint
    int all_negative_x  = 1;  // All ref_x < 0
    int all_negative_y  = 1;  // All ref_y < 0
    int all_negative_z  = 1;  // All ref_z < 0
    int all_outside_sum = 1;  // All ref_x + ref_y + ref_z > 1

#pragma unroll
    for (int v = 0; v < 8; v++) {
        // Transform hex vertex to tet reference space
        const FloatType dx = hex_vertices_x[v] - tet_origin_x;
        const FloatType dy = hex_vertices_y[v] - tet_origin_y;
        const FloatType dz = hex_vertices_z[v] - tet_origin_z;

        // Use fast_fma for better precision in matrix-vector multiplication
        const FloatType ref_x = fast_fma(inv_J00, dx, fast_fma(inv_J01, dy, inv_J02 * dz));
        const FloatType ref_y = fast_fma(inv_J10, dx, fast_fma(inv_J11, dy, inv_J12 * dz));
        const FloatType ref_z = fast_fma(inv_J20, dx, fast_fma(inv_J21, dy, inv_J22 * dz));

        // Point is inside tet if: ref_x >= 0 AND ref_y >= 0 AND ref_z >= 0 AND ref_x + ref_y + ref_z <= 1
        // Point is outside if it violates ANY of these constraints

        // Update flags - use bitwise AND for branchless execution
        all_negative_x &= (ref_x < FloatType(0.0));
        all_negative_y &= (ref_y < FloatType(0.0));
        all_negative_z &= (ref_z < FloatType(0.0));

        const FloatType sum_ref = ref_x + ref_y + ref_z;
        all_outside_sum &= (sum_ref > FloatType(1.0));

        // Early exit optimization: if we know hex is not completely outside, stop
        if (!all_negative_x &&   //
            !all_negative_y &&   //
            !all_negative_z &&   //
            !all_outside_sum) {  //
            return false;
        }  // END if early exit
    }  // END for (int v = 0; v < 8; v++)

    // Hex is completely outside if ALL vertices violate at least one constraint
    return (all_negative_x || all_negative_y || all_negative_z || all_outside_sum);
}  // END Function: is_hex_out_of_tet_gpu

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// is_hex_out_of_tet_gpu_optimized //////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType>
__device__ bool                                                       //
is_hex_out_of_tet_gpu_optimized(const FloatType inv_J_tet[9],         //
                                const FloatType tet_origin_x,         //
                                const FloatType tet_origin_y,         //
                                const FloatType tet_origin_z,         //
                                const FloatType hex_vertices_x[8],    //
                                const FloatType hex_vertices_y[8],    //
                                const FloatType hex_vertices_z[8]) {  //

    /**
     * ****************************************************************************************
     * Optimized version using bitmask operations for maximum GPU performance
     * *****************************************************************************************
     */

    // Precompute inverse Jacobian components
    const FloatType inv_J00 = inv_J_tet[0], inv_J01 = inv_J_tet[1], inv_J02 = inv_J_tet[2];
    const FloatType inv_J10 = inv_J_tet[3], inv_J11 = inv_J_tet[4], inv_J12 = inv_J_tet[5];
    const FloatType inv_J20 = inv_J_tet[6], inv_J21 = inv_J_tet[7], inv_J22 = inv_J_tet[8];

    // Use bitmask for branchless constraint tracking
    unsigned int       constraint_mask = 0xF;  // Start with all 4 constraints potentially satisfied
    const unsigned int MASK_NEG_X = 1u, MASK_NEG_Y = 2u, MASK_NEG_Z = 4u, MASK_SUM_GT1 = 8u;

// Complete loop unrolling for maximum performance
#define CHECK_VERTEX_GPU(idx)                                                                  \
    do {                                                                                       \
        const FloatType dx    = hex_vertices_x[idx] - tet_origin_x;                            \
        const FloatType dy    = hex_vertices_y[idx] - tet_origin_y;                            \
        const FloatType dz    = hex_vertices_z[idx] - tet_origin_z;                            \
        const FloatType ref_x = fast_fma(inv_J00, dx, fast_fma(inv_J01, dy, inv_J02 * dz));    \
        const FloatType ref_y = fast_fma(inv_J10, dx, fast_fma(inv_J11, dy, inv_J12 * dz));    \
        const FloatType ref_z = fast_fma(inv_J20, dx, fast_fma(inv_J21, dy, inv_J22 * dz));    \
        constraint_mask &= ~((ref_x >= FloatType(0.0)) ? 0u : MASK_NEG_X);                     \
        constraint_mask &= ~((ref_y >= FloatType(0.0)) ? 0u : MASK_NEG_Y);                     \
        constraint_mask &= ~((ref_z >= FloatType(0.0)) ? 0u : MASK_NEG_Z);                     \
        constraint_mask &= ~(((ref_x + ref_y + ref_z) <= FloatType(1.0)) ? 0u : MASK_SUM_GT1); \
        if (!constraint_mask) return false;                                                    \
    } while (0)

    CHECK_VERTEX_GPU(0);
    CHECK_VERTEX_GPU(1);
    CHECK_VERTEX_GPU(2);
    CHECK_VERTEX_GPU(3);
    CHECK_VERTEX_GPU(4);
    CHECK_VERTEX_GPU(5);
    CHECK_VERTEX_GPU(6);
    CHECK_VERTEX_GPU(7);

#undef CHECK_VERTEX_GPU

    return (constraint_mask != 0);
}  // END Function: is_hex_out_of_tet_gpu_optimized

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// compute_tet_bounding_box_gpu //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType, typename IntType = ptrdiff_t>
__device__ int                                            //
compute_tet_bounding_box_gpu(const FloatType x0,          //
                             const FloatType x1,          //
                             const FloatType x2,          //
                             const FloatType x3,          //
                             const FloatType y0,          //
                             const FloatType y1,          //
                             const FloatType y2,          //
                             const FloatType y3,          //
                             const FloatType z0,          //
                             const FloatType z1,          //
                             const FloatType z2,          //
                             const FloatType z3,          //
                             const FloatType stride0,     //
                             const FloatType stride1,     //
                             const FloatType stride2,     //
                             const FloatType origin0,     //
                             const FloatType origin1,     //
                             const FloatType origin2,     //
                             const FloatType inv_delta0,  //
                             const FloatType inv_delta1,  //
                             const FloatType inv_delta2,  //
                             IntType&        min_grid_x,  //
                             IntType&        max_grid_x,  //
                             IntType&        min_grid_y,  //
                             IntType&        max_grid_y,  //
                             IntType&        min_grid_z,  //
                             IntType&        max_grid_z) {       //

    // Use fast_fma for precision optimization where beneficial
    const FloatType x_min = fast_min(fast_min(x0, x1), fast_min(x2, x3));
    const FloatType x_max = fast_max(fast_max(x0, x1), fast_max(x2, x3));

    const FloatType y_min = fast_min(fast_min(y0, y1), fast_min(y2, y3));
    const FloatType y_max = fast_max(fast_max(y0, y1), fast_max(y2, y3));

    const FloatType z_min = fast_min(fast_min(z0, z1), fast_min(z2, z3));
    const FloatType z_max = fast_max(fast_max(z0, z1), fast_max(z2, z3));

    // const FloatType dx = (FloatType)delta0;
    // const FloatType dy = (FloatType)delta1;
    // const FloatType dz = (FloatType)delta2;

    const FloatType ox = (FloatType)origin0;
    const FloatType oy = (FloatType)origin1;
    const FloatType oz = (FloatType)origin2;

    // Step 2: Convert to grid indices accounting for the origin
    // Formula: grid_index = (physical_coord - origin) / delta
    // Using floor for minimum indices (with safety margin of -1)
    min_grid_x = (IntType)fast_floor((x_min - ox) * inv_delta0) - 1;
    min_grid_y = (IntType)fast_floor((y_min - oy) * inv_delta1) - 1;
    min_grid_z = (IntType)fast_floor((z_min - oz) * inv_delta2) - 1;

    // Using ceil for maximum indices (with safety margin of +1)
    max_grid_x = (IntType)fast_ceil((x_max - ox) * inv_delta0) + 1;
    max_grid_y = (IntType)fast_ceil((y_max - oy) * inv_delta1) + 1;
    max_grid_z = (IntType)fast_ceil((z_max - oz) * inv_delta2) + 1;

    return 0;  // Success
}  // END Function: compute_tet_bounding_box_gpu

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_inv_Jacobian_gpu /////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __inline__ void                   //
tet4_inv_Jacobian_gpu(const FloatType px0,   //
                      const FloatType px1,   //
                      const FloatType px2,   //
                      const FloatType px3,   //
                      const FloatType py0,   //
                      const FloatType py1,   //
                      const FloatType py2,   //
                      const FloatType py3,   //
                      const FloatType pz0,   //
                      const FloatType pz1,   //
                      const FloatType pz2,   //
                      const FloatType pz3,   //
                      FloatType       J_inv[9]) {  //

    /**
     ****************************************************************************************
    J^{-1} =
    \begin{bmatrix}
    inv_J11 & inv_J12 & inv_J13 \\
    inv_J21 & inv_J22 & inv_J23 \\
    inv_J31 & inv_J32 & inv_J33
    \end{bmatrix}
    *************************************************************************************************
     */

    // Compute the Jacobian matrix components
    const FloatType J11 = -px0 + px1;
    const FloatType J12 = -px0 + px2;
    const FloatType J13 = -px0 + px3;

    const FloatType J21 = -py0 + py1;
    const FloatType J22 = -py0 + py2;
    const FloatType J23 = -py0 + py3;

    const FloatType J31 = -pz0 + pz1;
    const FloatType J32 = -pz0 + pz2;
    const FloatType J33 = -pz0 + pz3;

    // Compute the determinant of the Jacobian using FMA operations for better precision
    const FloatType det_J = fast_fma(J11,
                                     fast_fma(J22, J33, -J23 * J32),
                                     fast_fma(-J12, fast_fma(J21, J33, -J23 * J31), J13 * fast_fma(J21, J32, -J22 * J31)));

    const FloatType inv_det_J = FloatType(1.0) / det_J;

    // Compute the inverse of the Jacobian matrix using FMA for precision
    J_inv[0] = fast_fma(J22, J33, -J23 * J32) * inv_det_J;
    J_inv[1] = fast_fma(J13, J32, -J12 * J33) * inv_det_J;
    J_inv[2] = fast_fma(J12, J23, -J13 * J22) * inv_det_J;
    J_inv[3] = fast_fma(J23, J31, -J21 * J33) * inv_det_J;
    J_inv[4] = fast_fma(J11, J33, -J13 * J31) * inv_det_J;
    J_inv[5] = fast_fma(J13, J21, -J11 * J23) * inv_det_J;
    J_inv[6] = fast_fma(J21, J32, -J22 * J31) * inv_det_J;
    J_inv[7] = fast_fma(J12, J31, -J11 * J32) * inv_det_J;
    J_inv[8] = fast_fma(J11, J22, -J12 * J21) * inv_det_J;

}  // END Function: tet4_inv_Jacobian_gpu

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_face_normal_gpu //////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType>
__device__ void                            //
tet4_face_normal_gpu(const FloatType px0,  //
                     const FloatType py0,  //
                     const FloatType pz0,  //
                     const FloatType px1,  //
                     const FloatType py1,  //
                     const FloatType pz1,  //
                     const FloatType px2,  //
                     const FloatType py2,  //
                     const FloatType pz2,  //
                     FloatType*      nx,   //
                     FloatType*      ny,   //
                     FloatType*      nz) {      //

    // Compute two edge vectors
    const FloatType e1x = px1 - px0;
    const FloatType e1y = py1 - py0;
    const FloatType e1z = pz1 - pz0;

    const FloatType e2x = px2 - px0;
    const FloatType e2y = py2 - py0;
    const FloatType e2z = pz2 - pz0;

    // Cross product e1 × e2 to get face normal
    *nx = e1y * e2z - e1z * e2y;
    *ny = e1z * e2x - e1x * e2z;
    *nz = e1x * e2y - e1y * e2x;

}  // END Function: tet4_face_normal_gpu

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_faces_normals_gpu ////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType>
__device__ void                                           //
tet4_faces_normals_gpu(const FloatType px0,               //
                       const FloatType px1,               //
                       const FloatType px2,               //
                       const FloatType px3,               //
                       const FloatType py0,               //
                       const FloatType py1,               //
                       const FloatType py2,               //
                       const FloatType py3,               //
                       const FloatType pz0,               //
                       const FloatType pz1,               //
                       const FloatType pz2,               //
                       const FloatType pz3,               //
                       FloatType       normals[4][3],     //
                       FloatType       faces_centroid[4][3]) {  //

    // Compute tetrahedron centroid
    const FloatType cx = (px0 + px1 + px2 + px3) * FloatType(0.25);
    const FloatType cy = (py0 + py1 + py2 + py3) * FloatType(0.25);
    const FloatType cz = (pz0 + pz1 + pz2 + pz3) * FloatType(0.25);

    // Store face centroids
    faces_centroid[0][0] = (px1 + px2 + px3) / FloatType(3.0);
    faces_centroid[0][1] = (py1 + py2 + py3) / FloatType(3.0);
    faces_centroid[0][2] = (pz1 + pz2 + pz3) / FloatType(3.0);
    faces_centroid[1][0] = (px0 + px3 + px2) / FloatType(3.0);
    faces_centroid[1][1] = (py0 + py3 + py2) / FloatType(3.0);
    faces_centroid[1][2] = (pz0 + pz3 + pz2) / FloatType(3.0);
    faces_centroid[2][0] = (px0 + px1 + px3) / FloatType(3.0);
    faces_centroid[2][1] = (py0 + py1 + py3) / FloatType(3.0);
    faces_centroid[2][2] = (pz0 + pz1 + pz3) / FloatType(3.0);
    faces_centroid[3][0] = (px0 + px2 + px1) / FloatType(3.0);
    faces_centroid[3][1] = (py0 + py2 + py1) / FloatType(3.0);
    faces_centroid[3][2] = (pz0 + pz2 + pz1) / FloatType(3.0);

    // Face 0: vertices 1, 2, 3 (opposite to vertex 0)
    tet4_face_normal_gpu(px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, &normals[0][0], &normals[0][1], &normals[0][2]);

    // Check orientation: vector from centroid to face center should align with normal
    const FloatType fc0_x = faces_centroid[0][0] - cx;
    const FloatType fc0_y = faces_centroid[0][1] - cy;
    const FloatType fc0_z = faces_centroid[0][2] - cz;
    const FloatType dot0  = normals[0][0] * fc0_x + normals[0][1] * fc0_y + normals[0][2] * fc0_z;
    if (dot0 < FloatType(0.0)) {
        normals[0][0] = -normals[0][0];
        normals[0][1] = -normals[0][1];
        normals[0][2] = -normals[0][2];
    }  // END if (dot0 < 0.0)

    // Face 1: vertices 0, 3, 2 (opposite to vertex 1)
    tet4_face_normal_gpu(px0, py0, pz0, px3, py3, pz3, px2, py2, pz2, &normals[1][0], &normals[1][1], &normals[1][2]);

    const FloatType fc1_x = faces_centroid[1][0] - cx;
    const FloatType fc1_y = faces_centroid[1][1] - cy;
    const FloatType fc1_z = faces_centroid[1][2] - cz;
    const FloatType dot1  = normals[1][0] * fc1_x + normals[1][1] * fc1_y + normals[1][2] * fc1_z;
    if (dot1 < FloatType(0.0)) {
        normals[1][0] = -normals[1][0];
        normals[1][1] = -normals[1][1];
        normals[1][2] = -normals[1][2];
    }  // END if (dot1 < 0.0)

    // Face 2: vertices 0, 1, 3 (opposite to vertex 2)
    tet4_face_normal_gpu(px0, py0, pz0, px1, py1, pz1, px3, py3, pz3, &normals[2][0], &normals[2][1], &normals[2][2]);

    const FloatType fc2_x = faces_centroid[2][0] - cx;
    const FloatType fc2_y = faces_centroid[2][1] - cy;
    const FloatType fc2_z = faces_centroid[2][2] - cz;
    const FloatType dot2  = normals[2][0] * fc2_x + normals[2][1] * fc2_y + normals[2][2] * fc2_z;
    if (dot2 < FloatType(0.0)) {
        normals[2][0] = -normals[2][0];
        normals[2][1] = -normals[2][1];
        normals[2][2] = -normals[2][2];
    }  // END if (dot2 < 0.0)

    // Face 3: vertices 0, 2, 1 (opposite to vertex 3)
    tet4_face_normal_gpu(px0, py0, pz0, px2, py2, pz2, px1, py1, pz1, &normals[3][0], &normals[3][1], &normals[3][2]);

    const FloatType fc3_x = faces_centroid[3][0] - cx;
    const FloatType fc3_y = faces_centroid[3][1] - cy;
    const FloatType fc3_z = faces_centroid[3][2] - cz;
    const FloatType dot3  = normals[3][0] * fc3_x + normals[3][1] * fc3_y + normals[3][2] * fc3_z;
    if (dot3 < FloatType(0.0)) {
        normals[3][0] = -normals[3][0];
        normals[3][1] = -normals[3][1];
        normals[3][2] = -normals[3][2];
    }  // END if (dot3 < 0.0)

}  // END Function: tet4_faces_normals_gpu

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// midpoint_quadrature_gpu ////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType, typename IntType = ptrdiff_t>
__device__ int                                 //
midpoint_quadrature_gpu(const IntType N,       //
                        FloatType*    nodes,   //
                        FloatType*    weights) {  //
    if (N <= 0) {
        return -1;  // Invalid number of points
    }

    const FloatType weight = 1.0 / (FloatType)N;  // Equal weights for midpoint rule

    for (int i = 0; i < N; i++) {
        nodes[i]   = ((FloatType)(i) + 0.5) / (FloatType)N;  // Midpoint in each subinterval
        weights[i] = weight;                                 // Assign equal weight
    }

    return 0;  // Success
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// Gauss_Legendre_quadrature_gpu /////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType, typename IntType = ptrdiff_t>
__device__ int                                       //
Gauss_Legendre_quadrature_gpu(const int  N,          //
                              FloatType* nodes,      //
                              FloatType* weights) {  //
    if (N <= 0) return -1;                           // Invalid number of points

    using FloatType2_loc = typename declare_FloatType2<FloatType>::type;

    for (int i = 0; i < N; i++) {
        const FloatType2_loc quad_pair = Gauss_Legendre_quadrature_pairs<FloatType, FloatType2_loc, IntType>(N, i);

        nodes[i]   = quad_pair.x;
        weights[i] = quad_pair.y;
    }  // END for (int i = 0; i < N; i++)

    return 0;  // Success
}  // END Function: Gauss_Legendre_quadrature_gpu

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
// sfem_quad_rule_3D ///////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType, typename IntType = ptrdiff_t>
__device__ __inline_hint__ int                                    //
sfem_quad_rule_3D_gpu(const tet_quad_midpoint_nqp_gpu_t rule,     //
                      const IntType                     N,        //
                      FloatType*                        nodes,    // Buffer for 1D nodes
                      FloatType*                        weights,  // Buffer for 1D weights
                      FloatType*                        qx,       //
                      FloatType*                        qy,       //
                      FloatType*                        qz,       //
                      FloatType*                        qw) {                            //
    switch (rule) {
        case TET_QUAD_MIDPOINT_NQP: {
            // FloatType *nodes = (FloatType*)malloc(N * sizeof(FloatType)),  //
            //         *weights = (FloatType*)malloc(N * sizeof(FloatType));

            // Compute 1D midpoint quadrature points and weights
            midpoint_quadrature_gpu(N, nodes, weights);

            // Compute weights for the 3D quadrature points
            int idx = 0;
#pragma unroll
            for (int i = 0; i < N; i++) {
                const FloatType nodes_i   = nodes[i];
                const FloatType weights_i = weights[i];
#pragma unroll
                for (int j = 0; j < N; j++) {
                    const FloatType nodes_j    = nodes[j];
                    const FloatType weights_ij = weights[j] * weights_i;
#pragma unroll
                    for (int k = 0; k < N; k++) {
                        qw[idx] = weights_ij * weights[k];  // Product of weights
                        qx[idx] = nodes_i;
                        qy[idx] = nodes_j;
                        qz[idx] = nodes[k];
                        idx++;
                    }  // END for k
                }  // END for j
            }  // END for i

            return N * N * N;  // Total number of quadrature points
        }  // END case TET_QUAD_MIDPOINT_NQP

        default:
            return -1;  // Unknown rule
    }  // END switch
}  // END sfem_quad_rule_3D

/////////////////////////////////////////////////////////////////////////////////
// Kernel to perform adjoint mini-tetrahedron resampling
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType,                                                            //
          typename IntType = ptrdiff_t>                                                  //
__global__ void                                                                          //
tet4_resample_field_adjoint_hex_quad_kernel_gpu(const IntType           start_element,   // Mesh
                                                const IntType           end_element,     //
                                                const IntType           nnodes,          //
                                                const elems_tet4_device elems,           //
                                                const xyz_tet4_device   xyz,             //
                                                const IntType           n0,              // SDF
                                                const IntType           n1,              //
                                                const IntType           n2,              //
                                                const IntType           stride0,         // Stride
                                                const IntType           stride1,         //
                                                const IntType           stride2,         //
                                                const FloatType         origin0,         // Origin
                                                const FloatType         origin1,         //
                                                const FloatType         origin2,         //
                                                const FloatType         dx,              // Delta
                                                const FloatType         dy,              //
                                                const FloatType         dz,              //
                                                const FloatType* const  weighted_field,  // Input weighted field
                                                FloatType* const        data) {                 // Output data

    // const int tet_id    = (blockIdx.x * blockDim.x + threadIdx.x) / LANES_PER_TILE_HEX_QUAD;
    const int tet_id    = blockIdx.x;
    const int element_i = start_element + tet_id;  // Global element index
    const int warp_id   = threadIdx.x / LANES_PER_TILE_HEX_QUAD;
    const int lane_id   = threadIdx.x % LANES_PER_TILE_HEX_QUAD;
    const int n_warps   = blockDim.x / LANES_PER_TILE_HEX_QUAD;

    if (element_i >= end_element) return;  // Out of range

    // printf("Processing element %ld / %ld\n", element_i, end_element);

    const IntType off0 = 0;
    const IntType off1 = stride0;
    const IntType off2 = stride0 + stride1;
    const IntType off3 = stride1;
    const IntType off4 = stride2;
    const IntType off5 = stride0 + stride2;
    const IntType off6 = stride0 + stride1 + stride2;
    const IntType off7 = stride1 + stride2;

#define N_midpoint (4)
#define dim_quad (N_midpoint * N_midpoint * N_midpoint)
    FloatType Q_nodes_x[dim_quad];
    FloatType Q_nodes_y[dim_quad];
    FloatType Q_nodes_z[dim_quad];
    FloatType Q_weights[dim_quad];
    FloatType Q_Nodes[N_midpoint];
    FloatType Q_Weights[N_midpoint];

    sfem_quad_rule_3D_gpu<FloatType, IntType>(TET_QUAD_MIDPOINT_NQP,  //
                                              N_midpoint,
                                              Q_Nodes,
                                              Q_Weights,
                                              Q_nodes_x,
                                              Q_nodes_y,
                                              Q_nodes_z,
                                              Q_weights);

    const FloatType d_min             = dx < dy ? (dx < dz ? dx : dz) : (dy < dz ? dy : dz);
    const FloatType hexahedron_volume = dx * dy * dz;

    // printf("Exaedre volume: %e\n", hexahedron_volume);

    IntType   ev[4] = {0, 0, 0, 0};  // Indices of the vertices of the tetrahedron
    FloatType inv_J_tet[9];

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

    IntType min_grid_x, max_grid_x;
    IntType min_grid_y, max_grid_y;
    IntType min_grid_z, max_grid_z;

    FloatType face_normals_array[4][3];
    FloatType faces_centroids_array[4][3];

    tet4_faces_normals_gpu(x0_n,                    //
                           x1_n,                    //
                           x2_n,                    //
                           x3_n,                    //
                           y0_n,                    //
                           y1_n,                    //
                           y2_n,                    //
                           y3_n,                    //
                           z0_n,                    //
                           z1_n,                    //
                           z2_n,                    //
                           z3_n,                    //
                           face_normals_array,      //
                           faces_centroids_array);  //

    tet4_inv_Jacobian_gpu(x0_n,        //
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

    compute_tet_bounding_box_gpu<FloatType, IntType>(x0_n,         //
                                                     x1_n,         //
                                                     x2_n,         //
                                                     x3_n,         //
                                                     y0_n,         //
                                                     y1_n,         //
                                                     y2_n,         //
                                                     y3_n,         //
                                                     z0_n,         //
                                                     z1_n,         //
                                                     z2_n,         //
                                                     z3_n,         //
                                                     stride0,      //
                                                     stride1,      //
                                                     stride2,      //
                                                     origin0,      //
                                                     origin1,      //
                                                     origin2,      //
                                                     dx,           //
                                                     dy,           //
                                                     dz,           //
                                                     min_grid_x,   //
                                                     max_grid_x,   //
                                                     min_grid_y,   //
                                                     max_grid_y,   //
                                                     min_grid_z,   //
                                                     max_grid_z);  //

    FloatType hex_element_field[8] = {0.0};

    const IntType size_x = max_grid_x - min_grid_x + 1;
    const IntType size_y = max_grid_y - min_grid_y + 1;
    const IntType size_z = max_grid_z - min_grid_z + 1;

    const IntType total_grid_points = size_x * size_y * size_z;

    // Loop over all grid points in the bounding box
    for (IntType idx = 0; idx < total_grid_points; idx += n_warps) {
        const IntType grid_idx = idx + warp_id;
        if (grid_idx >= total_grid_points) continue;

        const IntType ix_local = grid_idx % size_x;
        const IntType iy_local = (grid_idx / size_x) % size_y;
        const IntType iz_local = grid_idx / (size_x * size_y);

        // Convert to absolute grid coordinates
        const IntType ix = min_grid_x + ix_local;
        const IntType iy = min_grid_y + iy_local;
        const IntType iz = min_grid_z + iz_local;

        const FloatType z_hex_min = ((FloatType)iz) * dz + origin2;
        const FloatType z_hex_max = z_hex_min + dz;

        const FloatType y_hex_min = ((FloatType)iy) * dy + origin1;
        const FloatType y_hex_max = y_hex_min + dy;

        const FloatType x_hex_min = ((FloatType)ix) * dx + origin0;
        const FloatType x_hex_max = x_hex_min + dx;

        const FloatType hex_vertices_x[8] = {x_hex_min,
                                             x_hex_max,
                                             x_hex_max,
                                             x_hex_min,  //
                                             x_hex_min,
                                             x_hex_max,
                                             x_hex_max,
                                             x_hex_min};

        const FloatType hex_vertices_y[8] = {y_hex_min,
                                             y_hex_min,
                                             y_hex_max,
                                             y_hex_max,  //
                                             y_hex_min,
                                             y_hex_min,
                                             y_hex_max,
                                             y_hex_max};

        const FloatType hex_vertices_z[8] = {z_hex_min,
                                             z_hex_min,
                                             z_hex_min,
                                             z_hex_min,  //
                                             z_hex_max,
                                             z_hex_max,
                                             z_hex_max,
                                             z_hex_max};

        const bool is_out_of_tet = is_hex_out_of_tet_gpu(inv_J_tet,        //
                                                         x0_n,             //
                                                         y0_n,             //
                                                         z0_n,             //
                                                         hex_vertices_x,   //
                                                         hex_vertices_y,   //
                                                         hex_vertices_z);  //

        if (is_out_of_tet) continue;  // Skip this hex cell

        // printf("Element %d, Hex cell at (%d, %d, %d) may overlap tet\n", element_i, ix, iy, iz);

        for (int q_ijk = lane_id; q_ijk < dim_quad; q_ijk += LANES_PER_TILE_HEX_QUAD) {
            quadrature_point_result_gpu_t Qpoint_phys =                //
                    transform_and_check_quadrature_point_n_gpu(q_ijk,  //
                                                                       //    vol_tet_main,                          //
                                                               face_normals_array,                       //
                                                               faces_centroids_array,                    //
                                                               Q_nodes_x,                                //
                                                               Q_nodes_y,                                //
                                                               Q_nodes_z,                                //
                                                               Q_weights,                                //
                                                               origin0,                                  //
                                                               origin1,                                  //
                                                               origin2,                                  //
                                                               dx,                                       //
                                                               dy,                                       //
                                                               dz,                                       //
                                                               ix,                                       //
                                                               iy,                                       //
                                                               iz,                                       //
                                                               (FloatType[4]){x0_n, x1_n, x2_n, x3_n},   //
                                                               (FloatType[4]){y0_n, y1_n, y2_n, y3_n},   //
                                                               (FloatType[4]){z0_n, z1_n, z2_n, z3_n});  //

            if (Qpoint_phys.is_inside) {
#pragma unroll
                for (int v = 0; v < 8; v++) hex_element_field[v] = FloatType(0.0);

                FloatType Q_ref_x, Q_ref_y, Q_ref_z;

                tet4_inv_transform_J_gpu(inv_J_tet,               //
                                         Qpoint_phys.physical_x,  //
                                         Qpoint_phys.physical_y,  //
                                         Qpoint_phys.physical_z,  //
                                         x0_n,                    //
                                         y0_n,                    //
                                         z0_n,                    //
                                         Q_ref_x,                 //
                                         Q_ref_y,                 //
                                         Q_ref_z);                //

                IntType out_i, out_j, out_k;

                transfer_weighted_field_tet4_to_hex_gpu<FloatType, IntType>(wf0,                     //
                                                                            wf1,                     //
                                                                            wf2,                     //
                                                                            wf3,                     //
                                                                            Qpoint_phys.physical_x,  //
                                                                            Qpoint_phys.physical_y,  //
                                                                            Qpoint_phys.physical_z,  //
                                                                            Q_ref_x,                 //
                                                                            Q_ref_y,                 //
                                                                            Q_ref_z,                 //
                                                                            Qpoint_phys.weight,      //
                                                                            origin0,                 //
                                                                            origin1,                 //
                                                                            origin2,                 //
                                                                            dx,                      //
                                                                            dy,                      //
                                                                            dz,                      //
                                                                            hex_element_field,       //
                                                                            out_i,                   //
                                                                            out_j,                   //
                                                                            out_k);                  //

                const ptrdiff_t base_index = ix * stride0 +  //
                                             iy * stride1 +  //
                                             iz * stride2;   //

                atomicAdd(&data[base_index + off0], hex_element_field[0]);  //
                atomicAdd(&data[base_index + off1], hex_element_field[1]);  //
                atomicAdd(&data[base_index + off2], hex_element_field[2]);  //
                atomicAdd(&data[base_index + off3], hex_element_field[3]);  //
                atomicAdd(&data[base_index + off4], hex_element_field[4]);  //
                atomicAdd(&data[base_index + off5], hex_element_field[5]);  //
                atomicAdd(&data[base_index + off6], hex_element_field[6]);  //
                atomicAdd(&data[base_index + off7], hex_element_field[7]);  //

            }  // END if (Qpoint_phys.is_inside)
        }  // END for (int q_ijk = lane_id; q_ijk < dim_quad; q_ijk += LANES_PER_TILE_HEX_QUAD)
    }  // END for (IntType idx = 0; idx < total_grid_points; idx += n_warps)
}  // END Function: tet4_resample_field_adjoint_hex_quad_kernel_gpu

/////////////////////////////////////////////////////////////////////////////////
// Kernel to perform adjoint mini-tetrahedron resampling Version 2
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType,                                                                    //
          typename IntType = ptrdiff_t>                                                          //
__device__ __forceinline__ void                                                                  //
tet4_resample_field_adjoint_hex_quad_element_method_gpu(const IntType           element_i,       // element index
                                                        const IntType           nnodes,          //
                                                        const elems_tet4_device elems,           //
                                                        const xyz_tet4_device   xyz,             //
                                                        const IntType           n0,              // SDF
                                                        const IntType           n1,              //
                                                        const IntType           n2,              //
                                                        const IntType           stride0,         // Stride
                                                        const IntType           stride1,         //
                                                        const IntType           stride2,         //
                                                        const FloatType         origin0,         // Origin
                                                        const FloatType         origin1,         //
                                                        const FloatType         origin2,         //
                                                        const FloatType         dx,              // Delta
                                                        const FloatType         dy,              //
                                                        const FloatType         dz,              //
                                                        const FloatType* const  weighted_field,  // Input weighted field
                                                        FloatType* const        data) {                 // Output data

    // printf("Processing element %ld / %ld\n", element_i, end_element);

    const int warp_id = threadIdx.x / LANES_PER_TILE_HEX_QUAD;
    const int lane_id = threadIdx.x % LANES_PER_TILE_HEX_QUAD;
    const int n_warps = blockDim.x / LANES_PER_TILE_HEX_QUAD;

    const FloatType inv_dx = FloatType(1.0) / dx;
    const FloatType inv_dy = FloatType(1.0) / dy;
    const FloatType inv_dz = FloatType(1.0) / dz;

    const IntType off0 = 0;
    const IntType off1 = stride0;
    const IntType off2 = stride0 + stride1;
    const IntType off3 = stride1;
    const IntType off4 = stride2;
    const IntType off5 = stride0 + stride2;
    const IntType off6 = stride0 + stride1 + stride2;
    const IntType off7 = stride1 + stride2;

    const IntType N_quadnodes_loc = 2;

    FloatType Q_nodes[N_quadnodes_loc];
    FloatType Q_weights[N_quadnodes_loc];

    Gauss_Legendre_quadrature_gpu<FloatType, IntType>(N_quadnodes_loc, Q_nodes, Q_weights);
    // midpoint_quadrature_gpu<FloatType, IntType>(N_midpoint_loc, Q_nodes, Q_weights);

    // sfem_quad_rule_3D_gpu<FloatType, IntType>(TET_QUAD_MIDPOINT_NQP,  //
    //                                           N_midpoint,             //
    //                                           Q_Nodes,                //
    //                                           Q_Weights,
    //                                           Q_nodes_x,
    //                                           Q_nodes_y,
    //                                           Q_nodes_z,
    //                                           Q_weights);

    // const FloatType d_min             = dx < dy ? (dx < dz ? dx : dz) : (dy < dz ? dy : dz);
    // const FloatType hexahedron_volume = dx * dy * dz;

    // printf("Exaedre volume: %e\n", hexahedron_volume);

    IntType   ev[4] = {0, 0, 0, 0};  // Indices of the vertices of the tetrahedron
    FloatType inv_J_tet[9];

    const IntType ev0 = __ldg(&elems.elems_v0[element_i]);
    const IntType ev1 = __ldg(&elems.elems_v1[element_i]);
    const IntType ev2 = __ldg(&elems.elems_v2[element_i]);
    const IntType ev3 = __ldg(&elems.elems_v3[element_i]);

    // Read the coordinates of the vertices of the tetrahedron
    // In the physical space
    const FloatType x0_n = FloatType(__ldg(&xyz.x[ev0]));
    const FloatType x1_n = FloatType(__ldg(&xyz.x[ev1]));
    const FloatType x2_n = FloatType(__ldg(&xyz.x[ev2]));
    const FloatType x3_n = FloatType(__ldg(&xyz.x[ev3]));

    const FloatType y0_n = FloatType(__ldg(&xyz.y[ev0]));
    const FloatType y1_n = FloatType(__ldg(&xyz.y[ev1]));
    const FloatType y2_n = FloatType(__ldg(&xyz.y[ev2]));
    const FloatType y3_n = FloatType(__ldg(&xyz.y[ev3]));

    const FloatType z0_n = FloatType(__ldg(&xyz.z[ev0]));
    const FloatType z1_n = FloatType(__ldg(&xyz.z[ev1]));
    const FloatType z2_n = FloatType(__ldg(&xyz.z[ev2]));
    const FloatType z3_n = FloatType(__ldg(&xyz.z[ev3]));

    const FloatType wf0 = FloatType(__ldg(&weighted_field[ev0]));  // Weighted field at vertex 0
    const FloatType wf1 = FloatType(__ldg(&weighted_field[ev1]));  // Weighted field at vertex 1
    const FloatType wf2 = FloatType(__ldg(&weighted_field[ev2]));  // Weighted field at vertex 2
    const FloatType wf3 = FloatType(__ldg(&weighted_field[ev3]));  // Weighted field at vertex 3

    IntType min_grid_x, max_grid_x;
    IntType min_grid_y, max_grid_y;
    IntType min_grid_z, max_grid_z;

    tet4_inv_Jacobian_gpu<FloatType>(x0_n,        //
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

    compute_tet_bounding_box_gpu<FloatType, IntType>(x0_n,         //
                                                     x1_n,         //
                                                     x2_n,         //
                                                     x3_n,         //
                                                     y0_n,         //
                                                     y1_n,         //
                                                     y2_n,         //
                                                     y3_n,         //
                                                     z0_n,         //
                                                     z1_n,         //
                                                     z2_n,         //
                                                     z3_n,         //
                                                     stride0,      //
                                                     stride1,      //
                                                     stride2,      //
                                                     origin0,      //
                                                     origin1,      //
                                                     origin2,      //
                                                     inv_dx,       //
                                                     inv_dy,       //
                                                     inv_dz,       //
                                                     min_grid_x,   //
                                                     max_grid_x,   //
                                                     min_grid_y,   //
                                                     max_grid_y,   //
                                                     min_grid_z,   //
                                                     max_grid_z);  //

    FloatType hex_element_field[8] = {0.0};

    const IntType size_x = max_grid_x - min_grid_x + 1;
    const IntType size_y = max_grid_y - min_grid_y + 1;
    const IntType size_z = max_grid_z - min_grid_z + 1;

    const IntType total_grid_points = size_x * size_y * size_z;

    // Loop over all grid points in the bounding box
    for (IntType idx = 0; idx < total_grid_points; idx += n_warps) {
        const IntType grid_idx = idx + warp_id;
        if (grid_idx >= total_grid_points) continue;

        const IntType ix_local = grid_idx % size_x;
        const IntType iy_local = (grid_idx / size_x) % size_y;
        const IntType iz_local = grid_idx / (size_x * size_y);

        // Convert to absolute grid coordinates
        const IntType ix = min_grid_x + ix_local;
        const IntType iy = min_grid_y + iy_local;
        const IntType iz = min_grid_z + iz_local;

        const FloatType x_hex_min = fast_fma((FloatType)ix, dx, origin0);
        const FloatType y_hex_min = fast_fma((FloatType)iy, dy, origin1);
        const FloatType z_hex_min = fast_fma((FloatType)iz, dz, origin2);

        const FloatType x_hex_max = x_hex_min + dx;
        const FloatType y_hex_max = y_hex_min + dy;
        const FloatType z_hex_max = z_hex_min + dz;

        const FloatType hex_vertices_x[8] = {x_hex_min,
                                             x_hex_max,
                                             x_hex_max,
                                             x_hex_min,  //
                                             x_hex_min,
                                             x_hex_max,
                                             x_hex_max,
                                             x_hex_min};

        const FloatType hex_vertices_y[8] = {y_hex_min,
                                             y_hex_min,
                                             y_hex_max,
                                             y_hex_max,  //
                                             y_hex_min,
                                             y_hex_min,
                                             y_hex_max,
                                             y_hex_max};

        const FloatType hex_vertices_z[8] = {z_hex_min,
                                             z_hex_min,
                                             z_hex_min,
                                             z_hex_min,  //
                                             z_hex_max,
                                             z_hex_max,
                                             z_hex_max,
                                             z_hex_max};

        const bool is_out_of_tet = is_hex_out_of_tet_gpu(inv_J_tet,        //
                                                         x0_n,             //
                                                         y0_n,             //
                                                         z0_n,             //
                                                         hex_vertices_x,   //
                                                         hex_vertices_y,   //
                                                         hex_vertices_z);  //

        if (is_out_of_tet) continue;  // Skip this hex cell

        // printf("Element %d, Hex cell at (%d, %d, %d) may overlap tet\n", element_i, ix, iy, iz);

#pragma unroll
        for (int v = 0; v < 8; v++) hex_element_field[v] = FloatType(0.0);

        // for (int q_ijk = lane_id; q_ijk < dim_quad; q_ijk += LANES_PER_TILE_HEX_QUAD) {

#pragma unroll
        for (int q_i = 0; q_i < N_quadnodes_loc; q_i++) {
            const FloatType q_i_node   = Q_nodes[q_i];
            const FloatType q_i_weight = Q_weights[q_i];

#pragma unroll
            for (int q_j = 0; q_j < N_quadnodes_loc; q_j++) {
                const FloatType q_j_node    = Q_nodes[q_j];
                const FloatType q_ij_weight = Q_weights[q_j] * q_i_weight;

#pragma unroll
                for (int q_k = 0; q_k < N_quadnodes_loc; q_k++) {
                    // const int q_ijk = q_i * N_midpoint * N_midpoint + q_j * N_midpoint + q_k;
                    //

                    const FloatType Q_weight = q_ij_weight * Q_weights[q_k];

                    quadrature_point_result_gpu_t Qpoint_phys =                      //
                            transform_quadrature_point_n_gpu<FloatType,              //
                                                             IntType>(q_i_node,      //
                                                                      q_j_node,      //
                                                                      Q_nodes[q_k],  //
                                                                      Q_weight,      //
                                                                      origin0,       //
                                                                      origin1,       //
                                                                      origin2,       //
                                                                      dx,            //
                                                                      dy,            //
                                                                      dz,            //
                                                                      ix,            //
                                                                      iy,            //
                                                                      iz);           //

                    IntType out_i, out_j, out_k;

                    const bool is_in_tet =                                                                           //
                            transfer_weighted_field_tet4_to_hex_ckp_gpu<FloatType, IntType>(inv_J_tet,               //
                                                                                            wf0,                     //
                                                                                            wf1,                     //
                                                                                            wf2,                     //
                                                                                            wf3,                     //
                                                                                            Qpoint_phys.physical_x,  //
                                                                                            Qpoint_phys.physical_y,  //
                                                                                            Qpoint_phys.physical_z,  //
                                                                                            Qpoint_phys.weight,      //
                                                                                            x0_n,                    //
                                                                                            y0_n,                    //
                                                                                            z0_n,                    //
                                                                                            origin0,                 //
                                                                                            origin1,                 //
                                                                                            origin2,                 //
                                                                                            inv_dx,                  //
                                                                                            inv_dy,                  //
                                                                                            inv_dz,                  //
                                                                                            hex_element_field,       //
                                                                                            out_i,                   //
                                                                                            out_j,                   //
                                                                                            out_k);                  //

                }  // END: for (int q_ijk = lane_id; q_ijk < dim_quad; q_ijk += LANES_PER_TILE_HEX_QUAD)
            }  // END: for (int q_j = 0; q_j < N_midpoint; q_j++)
        }  // END: for (int q_i = 0; q_i < N_midpoint; q_i++)

        const IntType base_index = ix * stride0 +                   //
                                   iy * stride1 +                   //
                                   iz * stride2;                    //
                                                                    //
        atomicAdd(&data[base_index + off0], hex_element_field[0]);  //
        atomicAdd(&data[base_index + off1], hex_element_field[1]);  //
        atomicAdd(&data[base_index + off2], hex_element_field[2]);  //
        atomicAdd(&data[base_index + off3], hex_element_field[3]);  //
        atomicAdd(&data[base_index + off4], hex_element_field[4]);  //
        atomicAdd(&data[base_index + off5], hex_element_field[5]);  //
        atomicAdd(&data[base_index + off6], hex_element_field[6]);  //
        atomicAdd(&data[base_index + off7], hex_element_field[7]);  //

    }  // END for (IntType idx = 0; idx < total_grid_points; idx += n_warps)
}  // END Function: tet4_resample_field_adjoint_hex_quad_element_method_gpu

/////////////////////////////////////////////////////////////////////////////////
// Kernel to perform adjoint mini-tetrahedron resampling Version 2
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType,                                                               //
          typename IntType = ptrdiff_t>                                                     //
__device__ void                                                                             //
tet4_resample_field_adjoint_hex_quad_v2_method_gpu(const IntType           start_element,   // Mesh
                                                   const IntType           end_element,     //
                                                   const IntType           nnodes,          //
                                                   const elems_tet4_device elems,           //
                                                   const xyz_tet4_device   xyz,             //
                                                   const IntType           n0,              // SDF
                                                   const IntType           n1,              //
                                                   const IntType           n2,              //
                                                   const IntType           stride0,         // Stride
                                                   const IntType           stride1,         //
                                                   const IntType           stride2,         //
                                                   const FloatType         origin0,         // Origin
                                                   const FloatType         origin1,         //
                                                   const FloatType         origin2,         //
                                                   const FloatType         dx,              // Delta
                                                   const FloatType         dy,              //
                                                   const FloatType         dz,              //
                                                   const FloatType* const  weighted_field,  // Input weighted field
                                                   FloatType* const        data) {                 // Output data

    // const int tet_id    = (blockIdx.x * blockDim.x + threadIdx.x) / LANES_PER_TILE_HEX_QUAD;
    const int tet_id    = blockIdx.x;
    const int element_i = start_element + tet_id;  // Global element index

    if (element_i >= end_element) return;  // Out of range

    tet4_resample_field_adjoint_hex_quad_element_method_gpu<FloatType, IntType>(  //
            element_i,                                                            //
            nnodes,                                                               //
            elems,                                                                //
            xyz,                                                                  //
            n0,                                                                   //
            n1,                                                                   //
            n2,                                                                   //
            stride0,                                                              //
            stride1,                                                              //
            stride2,                                                              //
            origin0,                                                              //
            origin1,                                                              //
            origin2,                                                              //
            dx,                                                                   //
            dy,                                                                   //
            dz,                                                                   //
            weighted_field,                                                       //
            data);                                                                //

}  // END Function: tet4_resample_field_adjoint_hex_quad_kernel_gpu

/////////////////////////////////////////////////////////////////////////////////
// Kernel to perform adjoint mini-tetrahedron resampling
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType,                                                               //
          typename IntType = ptrdiff_t>                                                     //
__global__ void                                                                             //
tet4_resample_field_adjoint_hex_quad_v2_kernel_gpu(const IntType           start_element,   // Mesh
                                                   const IntType           end_element,     //
                                                   const IntType           nnodes,          //
                                                   const elems_tet4_device elems,           //
                                                   const xyz_tet4_device   xyz,             //
                                                   const IntType           n0,              // SDF
                                                   const IntType           n1,              //
                                                   const IntType           n2,              //
                                                   const IntType           stride0,         // Stride
                                                   const IntType           stride1,         //
                                                   const IntType           stride2,         //
                                                   const FloatType         origin0,         // Origin
                                                   const FloatType         origin1,         //
                                                   const FloatType         origin2,         //
                                                   const FloatType         dx,              // Delta
                                                   const FloatType         dy,              //
                                                   const FloatType         dz,              //
                                                   const FloatType* const  weighted_field,  // Input weighted field
                                                   FloatType* const        data) {                 // Output data

    // const IntType elems_block_size = gridDim.x;

    for (int element_i = start_element + blockIdx.x; element_i < end_element; element_i += gridDim.x) {
        //
        tet4_resample_field_adjoint_hex_quad_element_method_gpu<FloatType, IntType>(  //
                element_i,                                                            //
                nnodes,                                                               //
                elems,                                                                //
                xyz,                                                                  //
                n0,                                                                   //
                n1,                                                                   //
                n2,                                                                   //
                stride0,                                                              //
                stride1,                                                              //
                stride2,                                                              //
                origin0,                                                              //
                origin1,                                                              //
                origin2,                                                              //
                dx,                                                                   //
                dy,                                                                   //
                dz,                                                                   //
                weighted_field,                                                       //
                data);                                                                //
    }  // END for ( int element_i = start_element + blockIdx.x; element_i < end_element;

}  // END Function: tet4_resample_field_adjoint_hex_quad_v2_kernel_gpu

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// is_hex_out_of_tet_gpu /////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType>
__device__ void                                                    //
is_hex_vertices_out_of_tet_gpu(const FloatType inv_J_tet[9],       //
                               const FloatType tet_origin_x,       //
                               const FloatType tet_origin_y,       //
                               const FloatType tet_origin_z,       //
                               const FloatType hex_vertices_x[8],  //
                               const FloatType hex_vertices_y[8],  //
                               const FloatType hex_vertices_z[8],  //
                               int32_t         is_inside[8]) {
    /**
     * ****************************************************************************************
     * Check if a hexahedral element is completely outside a tetrahedral element
     * Using the inverse Jacobian of the tetrahedron to transform hex vertices to tet reference space
     * and check against tet reference space constraints.
     * This function returns true if the hex is completely outside the tet.
     * And returns false if it is unsure (partially inside, intersecting, completely inside, or UNDETECTED outside).
     * This must be used as a fast culling test before more expensive intersection tests.
     * *****************************************************************************************
     */

    // Precompute inverse Jacobian components for better cache utilization
    const FloatType inv_J00 = inv_J_tet[0];
    const FloatType inv_J01 = inv_J_tet[1];
    const FloatType inv_J02 = inv_J_tet[2];
    const FloatType inv_J10 = inv_J_tet[3];
    const FloatType inv_J11 = inv_J_tet[4];
    const FloatType inv_J12 = inv_J_tet[5];
    const FloatType inv_J20 = inv_J_tet[6];
    const FloatType inv_J21 = inv_J_tet[7];
    const FloatType inv_J22 = inv_J_tet[8];

#pragma unroll
    for (int v = 0; v < 8; v++) {
        // Transform hex vertex to tet reference space
        const FloatType dx = hex_vertices_x[v] - tet_origin_x;
        const FloatType dy = hex_vertices_y[v] - tet_origin_y;
        const FloatType dz = hex_vertices_z[v] - tet_origin_z;

        // Use fast_fma for better precision in matrix-vector multiplication
        const FloatType ref_x = fast_fma(inv_J00, dx, fast_fma(inv_J01, dy, inv_J02 * dz));
        const FloatType ref_y = fast_fma(inv_J10, dx, fast_fma(inv_J11, dy, inv_J12 * dz));
        const FloatType ref_z = fast_fma(inv_J20, dx, fast_fma(inv_J21, dy, inv_J22 * dz));

        const bool outside_condition = (ref_x < 0.0) || (ref_y < 0.0) || (ref_z < 0.0) || ((ref_x + ref_y + ref_z) > 1.0);
        is_inside[v]                 = int32_t(!outside_condition);  // true if inside, false if outside
    }  // END for (int v = 0; v < 8; v++)

}  // END Function: is_hex_out_of_tet_gpu

/**
 * @brief Kernel to identify hexahedral boundary elements from tetrahedral mesh.
 */
template <typename FloatType, typename IntType>
__device__ __forceinline__                                                     //
        void tet_grid_hex_indicator_IO_gpu(const IntType           element_i,  // element index    //
                                           const elems_tet4_device elems,      //
                                           const xyz_tet4_device   xyz,        //
                                           const IntType           n0,         // SDF
                                           const IntType           n1,         //
                                           const IntType           n2,         //
                                           const IntType           stride0,    // Stride
                                           const IntType           stride1,    //
                                           const IntType           stride2,    //
                                           const FloatType         origin0,    // Origin
                                           const FloatType         origin1,    //
                                           const FloatType         origin2,    //
                                           const FloatType         dx,         // Delta
                                           const FloatType         dy,         //
                                           const FloatType         dz,         //
                                           int32_t* const          hex_inout_device) {  //

    const int warp_id = threadIdx.x / LANES_PER_TILE_HEX_QUAD;
    const int lane_id = threadIdx.x % LANES_PER_TILE_HEX_QUAD;
    const int n_warps = blockDim.x / LANES_PER_TILE_HEX_QUAD;

    const FloatType inv_dx = FloatType(1.0) / dx;
    const FloatType inv_dy = FloatType(1.0) / dy;
    const FloatType inv_dz = FloatType(1.0) / dz;

    const IntType hex8_offsets[8] = {
            0,                            // Vertex 0: (0,0,0)
            stride0,                      // Vertex 1: (1,0,0)
            stride0 + stride1,            // Vertex 2: (1,1,0)
            stride1,                      // Vertex 3: (0,1,0)
            stride2,                      // Vertex 4: (0,0,1)
            stride0 + stride2,            // Vertex 5: (1,0,1)
            stride0 + stride1 + stride2,  // Vertex 6: (1,1,1)
            stride1 + stride2             // Vertex 7: (0,1,1)
    };

    IntType   ev[4] = {0, 0, 0, 0};  // Indices of the vertices of the tetrahedron
    FloatType inv_J_tet[9];

    const IntType ev0 = __ldg(&elems.elems_v0[element_i]);
    const IntType ev1 = __ldg(&elems.elems_v1[element_i]);
    const IntType ev2 = __ldg(&elems.elems_v2[element_i]);
    const IntType ev3 = __ldg(&elems.elems_v3[element_i]);

    // Read the coordinates of the vertices of the tetrahedron
    // In the physical space
    const FloatType x0_n = FloatType(__ldg(&xyz.x[ev0]));
    const FloatType x1_n = FloatType(__ldg(&xyz.x[ev1]));
    const FloatType x2_n = FloatType(__ldg(&xyz.x[ev2]));
    const FloatType x3_n = FloatType(__ldg(&xyz.x[ev3]));

    const FloatType y0_n = FloatType(__ldg(&xyz.y[ev0]));
    const FloatType y1_n = FloatType(__ldg(&xyz.y[ev1]));
    const FloatType y2_n = FloatType(__ldg(&xyz.y[ev2]));
    const FloatType y3_n = FloatType(__ldg(&xyz.y[ev3]));

    const FloatType z0_n = FloatType(__ldg(&xyz.z[ev0]));
    const FloatType z1_n = FloatType(__ldg(&xyz.z[ev1]));
    const FloatType z2_n = FloatType(__ldg(&xyz.z[ev2]));
    const FloatType z3_n = FloatType(__ldg(&xyz.z[ev3]));

    IntType min_grid_x, max_grid_x;
    IntType min_grid_y, max_grid_y;
    IntType min_grid_z, max_grid_z;

    tet4_inv_Jacobian_gpu<FloatType>(x0_n,        //
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

    compute_tet_bounding_box_gpu<FloatType, IntType>(x0_n,         //
                                                     x1_n,         //
                                                     x2_n,         //
                                                     x3_n,         //
                                                     y0_n,         //
                                                     y1_n,         //
                                                     y2_n,         //
                                                     y3_n,         //
                                                     z0_n,         //
                                                     z1_n,         //
                                                     z2_n,         //
                                                     z3_n,         //
                                                     stride0,      //
                                                     stride1,      //
                                                     stride2,      //
                                                     origin0,      //
                                                     origin1,      //
                                                     origin2,      //
                                                     inv_dx,       //
                                                     inv_dy,       //
                                                     inv_dz,       //
                                                     min_grid_x,   //
                                                     max_grid_x,   //
                                                     min_grid_y,   //
                                                     max_grid_y,   //
                                                     min_grid_z,   //
                                                     max_grid_z);  //

    const IntType size_x = max_grid_x - min_grid_x + 1;
    const IntType size_y = max_grid_y - min_grid_y + 1;
    const IntType size_z = max_grid_z - min_grid_z + 1;

    const IntType total_grid_points = size_x * size_y * size_z;

    // Loop over all grid points in the bounding box
    for (IntType idx = 0; idx < total_grid_points; idx += n_warps) {
        const IntType grid_idx = idx + warp_id;

        if (grid_idx >= total_grid_points) continue;

        const IntType ix_local = grid_idx % size_x;
        const IntType iy_local = (grid_idx / size_x) % size_y;
        const IntType iz_local = grid_idx / (size_x * size_y);

        // Convert to absolute grid coordinates
        const IntType ix = min_grid_x + ix_local;
        const IntType iy = min_grid_y + iy_local;
        const IntType iz = min_grid_z + iz_local;

        const FloatType x_hex_min = fast_fma((FloatType)ix, dx, origin0);
        const FloatType y_hex_min = fast_fma((FloatType)iy, dy, origin1);
        const FloatType z_hex_min = fast_fma((FloatType)iz, dz, origin2);

        const FloatType x_hex_max = x_hex_min + dx;
        const FloatType y_hex_max = y_hex_min + dy;
        const FloatType z_hex_max = z_hex_min + dz;

        const FloatType hex_vertices_x[8] = {x_hex_min,
                                             x_hex_max,
                                             x_hex_max,
                                             x_hex_min,  //
                                             x_hex_min,
                                             x_hex_max,
                                             x_hex_max,
                                             x_hex_min};

        const FloatType hex_vertices_y[8] = {y_hex_min,
                                             y_hex_min,
                                             y_hex_max,
                                             y_hex_max,  //
                                             y_hex_min,
                                             y_hex_min,
                                             y_hex_max,
                                             y_hex_max};

        const FloatType hex_vertices_z[8] = {z_hex_min,
                                             z_hex_min,
                                             z_hex_min,
                                             z_hex_min,  //
                                             z_hex_max,
                                             z_hex_max,
                                             z_hex_max,
                                             z_hex_max};

        int32_t is_inside[8];
        is_hex_vertices_out_of_tet_gpu(inv_J_tet,       //
                                       x0_n,            //
                                       y0_n,            //
                                       z0_n,            //
                                       hex_vertices_x,  //
                                       hex_vertices_y,  //
                                       hex_vertices_z,  //
                                       is_inside);      //

        const IntType base_index = ix * stride0 +  //
                                   iy * stride1 +  //
                                   iz * stride2;   //

#pragma unroll
        for (int v = 0; v < 8; v++) {
            if (is_inside[v]) hex_inout_device[base_index + hex8_offsets[v]] = 1;
        }
    }
}

/**
 * @brief Kernel wrapper to identify hexahedral boundary elements from tetrahedral mesh.
 */
template <typename FloatType, typename IntType>
__global__ void tet_grid_hex_indicator_IO_kernel_gpu(const IntType           start_element,  // element index    //
                                                     const IntType           end_element,    //
                                                     const elems_tet4_device elems,          //
                                                     const xyz_tet4_device   xyz,            //
                                                     const IntType           n0,             // SDF
                                                     const IntType           n1,             //
                                                     const IntType           n2,             //
                                                     const IntType           stride0,        // Stride
                                                     const IntType           stride1,        //
                                                     const IntType           stride2,        //
                                                     const FloatType         origin0,        // Origin
                                                     const FloatType         origin1,        //
                                                     const FloatType         origin2,        //
                                                     const FloatType         dx,             // Delta
                                                     const FloatType         dy,             //
                                                     const FloatType         dz,             //
                                                     int32_t* const          hex_inout_device) {      //

    for (int element_i = start_element + blockIdx.x; element_i < end_element; element_i += gridDim.x) {
        tet_grid_hex_indicator_IO_gpu<FloatType, IntType>(element_i,          //
                                                          elems,              //
                                                          xyz,                //
                                                          n0,                 //
                                                          n1,                 //
                                                          n2,                 //
                                                          stride0,            //
                                                          stride1,            //
                                                          stride2,            //
                                                          origin0,            //
                                                          origin1,            //
                                                          origin2,            //
                                                          dx,                 //
                                                          dy,                 //
                                                          dz,                 //
                                                          hex_inout_device);  //
    }
}

#ifdef __cplusplus
extern "C" {
#endif

void                                                                                       //
call_tet4_resample_field_adjoint_hex_quad_kernel_gpu(const ptrdiff_t      start_element,   //
                                                     const ptrdiff_t      end_element,     //
                                                     const ptrdiff_t      nelements,       //
                                                     const ptrdiff_t      nnodes,          //
                                                     const idx_t** const  elems,           //
                                                     const geom_t** const xyz,             //
                                                     const ptrdiff_t      n0,              //
                                                     const ptrdiff_t      n1,              //
                                                     const ptrdiff_t      n2,              //
                                                     const ptrdiff_t      stride0,         //
                                                     const ptrdiff_t      stride1,         //
                                                     const ptrdiff_t      stride2,         //
                                                     const geom_t         origin0,         //
                                                     const geom_t         origin1,         //
                                                     const geom_t         origin2,         //
                                                     const geom_t         dx,              //
                                                     const geom_t         dy,              //
                                                     const geom_t         dz,              //
                                                     const real_t* const  weighted_field,  //
                                                     real_t* const        data);                  //
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // __SFEM_RESAMPLE_FIELD_ADJOINT_HEX_QUAD_CUH__