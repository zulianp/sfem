#include "sfem_resample_field_adjoint_hex_wquad.cuh"

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// transfer_weighted_field_tet4_to_hex_gpu //////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType,                                                               //
          typename IntType = ptrdiff_t,                                                     //
          IntType Dim,                                                                      //
          bool    Generate_I_data>                                                             //
__device__ __inline__ bool                                                                  //
transfer_weighted_field_tet4_to_hex_dim_vec_gpu(const FloatType inv_J_tet[9],               //
                                                const FloatType wf0[Dim],                   //
                                                const FloatType wf1[Dim],                   //
                                                const FloatType wf2[Dim],                   //
                                                const FloatType wf3[Dim],                   //
                                                const FloatType q_phys_x,                   //
                                                const FloatType q_phys_y,                   //
                                                const FloatType q_phys_z,                   //
                                                const FloatType QW_phys_hex,                //
                                                const FloatType x0_n,                       //
                                                const FloatType y0_n,                       //
                                                const FloatType z0_n,                       //
                                                const FloatType ox,                         //
                                                const FloatType oy,                         //
                                                const FloatType oz,                         //
                                                const FloatType inv_dx,                     //
                                                const FloatType inv_dy,                     //
                                                const FloatType inv_dz,                     //
                                                FloatType       hex_element_field[Dim + 1][8]) {  //

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
    const FloatType hex8_f[8] = {
            one_minus_lx_ly * one_minus_lz,     // hex8_f0
            lx_one_minus_ly * one_minus_lz,     // hex8_f1
            lx_ly * one_minus_lz,               // hex8_f2
            one_minus_lx * l_y * one_minus_lz,  // hex8_f3
            one_minus_lx_ly * l_z,              // hex8_f4
            lx_one_minus_ly * l_z,              // hex8_f5
            lx_ly * l_z,                        // hex8_f6
            one_minus_lx_lz * l_y               // hex8_f7
    };

    // Tet4 linear shape functions in reference coordinates
    const FloatType f0 = FloatType(1.0) - q_ref_x - q_ref_y - q_ref_z;
    const FloatType f1 = q_ref_x;
    const FloatType f2 = q_ref_y;
    const FloatType f3 = q_ref_z;

    if constexpr (Generate_I_data) {
#pragma unroll
        for (int v = 0; v < 8; v++) {
            hex_element_field[0][v] += QW_phys_hex;
        }  // END for (v < 8)
    }

#pragma unroll
    for (IntType di = 1; di < Dim + 1; di++) {
        // Interpolate weighted field at quadrature point using FMA for precision
        const FloatType wf_quad =
                fast_fma(f0, wf0[di - 1], fast_fma(f1, wf1[di - 1], fast_fma(f2, wf2[di - 1], f3 * wf3[di - 1])));

        // Accumulate contributions to hex element field using FMA for precision
        const FloatType contribution = wf_quad * QW_phys_hex;

#pragma unroll
        for (int v = 0; v < 8; v++) {
            hex_element_field[di][v] += contribution * hex8_f[v];
        }  // END for (v < 8)
    }

    return true;
}  // END Function: transfer_weighted_field_tet4_to_hex_gpu
