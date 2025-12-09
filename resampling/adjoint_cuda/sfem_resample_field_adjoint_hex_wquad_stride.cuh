#include "sfem_resample_field_adjoint_hex_wquad.cuh"

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// transfer_weighted_field_tet4_to_hex_gpu //////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType,                                                               //
          typename IntType = ptrdiff_t,                                                     //
          IntType DIM,                                                                      //
          bool    Generate_I_data>                                                             //
__device__ __inline__ bool                                                                  //
transfer_weighted_field_tet4_to_hex_dim_vec_gpu(const FloatType inv_J_tet[9],               //
                                                const FloatType wf0[DIM],                   //
                                                const FloatType wf1[DIM],                   //
                                                const FloatType wf2[DIM],                   //
                                                const FloatType wf3[DIM],                   //
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
                                                FloatType       hex_element_field[DIM + 1][8]) {  //

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
    for (IntType di = 1; di < DIM + 1; di++) {
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

/////////////////////////////////////////////////////////////////////////////////
// Main method to perform adjoint mini-tetrahedron resampling Version 2
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType,                                                                           //
          typename IntType,                                                                             //
          IntType           DIM,                                                                        //
          matrix_ordering_t Ordering_IN,                                                                //
          matrix_ordering_t Ordering_OUT,                                                               //
          bool              Generate_I_data>                                                                         //
__device__ __forceinline__ void                                                                         //
tet4_resample_field_adjoint_hex_dim_quad_element_nw_gpu(const IntType           element_i,              // element index
                                                        const IntType           nnodes,                 //
                                                        const elems_tet4_device elems,                  //
                                                        const xyz_tet4_device   xyz,                    //
                                                        const IntType           n0,                     // SDF
                                                        const IntType           n1,                     //
                                                        const IntType           n2,                     //
                                                        const IntType           stride0,                // Stride
                                                        const IntType           stride1,                //
                                                        const IntType           stride2,                //
                                                        const FloatType         origin0,                // Origin
                                                        const FloatType         origin1,                //
                                                        const FloatType         origin2,                //
                                                        const FloatType         dx,                     // Delta
                                                        const FloatType         dy,                     //
                                                        const FloatType         dz,                     //
                                                        const FloatType* const  weighted_field_v[DIM],  // Input weighted field
                                                        FloatType* const        data[DIM],              //
                                                        FloatType*              I_data) {                            // Output data

    // printf("Processing element %ld / %ld\n", element_i, end_element);

    const int warp_id = threadIdx.x / LANES_PER_TILE_HEX_QUAD;
    const int lane_id = threadIdx.x % LANES_PER_TILE_HEX_QUAD;
    const int n_warps = blockDim.x / LANES_PER_TILE_HEX_QUAD;

    FloatType wf0[DIM];
    FloatType wf1[DIM];
    FloatType wf2[DIM];
    FloatType wf3[DIM];

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

    const IntType n_nodes_data = n0 * n1 * n2;

    const IntType N_quadnodes_loc = 2;

    FloatType Q_nodes[N_quadnodes_loc];
    FloatType Q_weights[N_quadnodes_loc];

    Gauss_Legendre_quadrature_gpu<FloatType, IntType>(N_quadnodes_loc, Q_nodes, Q_weights);
    //     // midpoint_quadrature_gpu<FloatType, IntType>(N_midpoint_loc, Q_nodes, Q_weights);

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

#pragma unroll
    for (IntType di = 0; di < DIM; di++) {
        IntType wi = 0;
        if constexpr (Ordering_IN == matrix_ordering_t::ROW_MAJOR)
            wi = di * nnodes + ev0;
        else if constexpr (Ordering_IN == matrix_ordering_t::COL_MAJOR)
            wi = ev0 * DIM + di;
        else
            static_assert(Ordering_IN == matrix_ordering_t::ROW_MAJOR ||        //
                                  Ordering_IN == matrix_ordering_t::COL_MAJOR,  //
                          "Invalid Ordering_IN");                               //

        wf0[di] = FloatType(weighted_field_v[wi]);
        wf1[di] = FloatType(weighted_field_v[wi]);
        wf2[di] = FloatType(weighted_field_v[wi]);
        wf3[di] = FloatType(weighted_field_v[wi]);
    }

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

    FloatType hex_element_field[DIM + 1][8] = {0.0};

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

        // #pragma unroll
        //         for (IntType di = 0; di < DIM + 1; di++)
        // #pragma unroll
        //             for (int v = 0; v < 8; v++) hex_element_field[di][v] = FloatType(0.0);

        memset(hex_element_field, 0, sizeof(hex_element_field));

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

                    const bool is_in_tet =                                                                            //
                            transfer_weighted_field_tet4_to_hex_dim_vec_gpu<FloatType,                                //
                                                                            IntType,                                  //
                                                                            DIM,                                      //
                                                                            Generate_I_data>(inv_J_tet,               //
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
                                                                                             hex_element_field);      //

                }  // END: for (int q_ijk = lane_id; q_ijk < dim_quad; q_ijk += LANES_PER_TILE_HEX_QUAD)
            }  // END: for (int q_j = 0; q_j < N_midpoint; q_j++)
        }  // END: for (int q_i = 0; q_i < N_midpoint; q_i++)

        const IntType base_index = ix * stride0 +  //
                                   iy * stride1 +  //
                                   iz * stride2;   //

        if constexpr (Generate_I_data) {
            atomicAdd(&I_data[base_index + off0], hex_element_field[0][0]);  //
            atomicAdd(&I_data[base_index + off1], hex_element_field[0][1]);  //
            atomicAdd(&I_data[base_index + off2], hex_element_field[0][2]);  //
            atomicAdd(&I_data[base_index + off3], hex_element_field[0][3]);  //
            atomicAdd(&I_data[base_index + off4], hex_element_field[0][4]);  //
            atomicAdd(&I_data[base_index + off5], hex_element_field[0][5]);  //
            atomicAdd(&I_data[base_index + off6], hex_element_field[0][6]);  //
            atomicAdd(&I_data[base_index + off7], hex_element_field[0][7]);  //
        }

        {  // Block for data accumulation //////////////////////////
#pragma unroll
            for (IntType di = 0; di < DIM; di++) {
                IntType data_i = 0;
                if constexpr (Ordering_IN == matrix_ordering_t::ROW_MAJOR) {
                    data_i = di * n_nodes_data + base_index;

                    atomicAdd(&data[data_i + off0], hex_element_field[di + 1][0]);  //
                    atomicAdd(&data[data_i + off1], hex_element_field[di + 1][1]);  //
                    atomicAdd(&data[data_i + off2], hex_element_field[di + 1][2]);  //
                    atomicAdd(&data[data_i + off3], hex_element_field[di + 1][3]);  //
                    atomicAdd(&data[data_i + off4], hex_element_field[di + 1][4]);  //
                    atomicAdd(&data[data_i + off5], hex_element_field[di + 1][5]);  //
                    atomicAdd(&data[data_i + off6], hex_element_field[di + 1][6]);  //
                    atomicAdd(&data[data_i + off7], hex_element_field[di + 1][7]);  //

                } else if constexpr (Ordering_IN == matrix_ordering_t::COL_MAJOR) {
                    atomicAdd(&data[(base_index + off0) * DIM + di], hex_element_field[di + 1][0]);  //
                    atomicAdd(&data[(base_index + off1) * DIM + di], hex_element_field[di + 1][1]);  //
                    atomicAdd(&data[(base_index + off2) * DIM + di], hex_element_field[di + 1][2]);  //
                    atomicAdd(&data[(base_index + off3) * DIM + di], hex_element_field[di + 1][3]);  //
                    atomicAdd(&data[(base_index + off4) * DIM + di], hex_element_field[di + 1][4]);  //
                    atomicAdd(&data[(base_index + off5) * DIM + di], hex_element_field[di + 1][5]);  //
                    atomicAdd(&data[(base_index + off6) * DIM + di], hex_element_field[di + 1][6]);  //
                    atomicAdd(&data[(base_index + off7) * DIM + di], hex_element_field[di + 1][7]);  //
                } else {
                    static_assert(Ordering_IN == matrix_ordering_t::ROW_MAJOR ||        //
                                          Ordering_IN == matrix_ordering_t::COL_MAJOR,  //
                                  "Invalid Ordering_IN");                               //
                }
            }
        }  // END block for data accumulation //////////////////////////
    }  // END for (IntType idx = 0; idx < total_grid_points; idx += n_warps)
}  // END Function: tet4_resample_field_adjoint_hex_dim_quad_element_nw_gpu

template <typename FloatType,                                                                       //
          typename IntType,                                                                         //
          IntType           DIM,                                                                    //
          matrix_ordering_t Ordering_IN,                                                            //
          matrix_ordering_t Ordering_OUT,                                                           //
          bool              Generate_I_data>                                                                     //
__global__ void tet_grid_hex_dim_quad_element_nw_kernel_gpu(const IntType           start_element,  // element index    //
                                                            const IntType           end_element,    //
                                                            const IntType           nnodes,         //
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
                                                            FloatType* const        weighted_field_v[DIM],
                                                            FloatType* const        data[DIM],  //
                                                            int32_t* const          hex_inout_device) {  //

    for (int element_i = start_element + blockIdx.x; element_i < end_element; element_i += gridDim.x) {
        tet4_resample_field_adjoint_hex_dim_quad_element_nw_gpu<FloatType,         //
                                                                IntType,           //
                                                                DIM,               //
                                                                Ordering_IN,       //
                                                                Ordering_OUT,      //
                                                                Generate_I_data>(  //
                element_i,                                                         //
                nnodes,                                                            //
                elems,                                                             //
                xyz,                                                               //
                n0,                                                                //
                n1,                                                                //
                n2,                                                                //
                stride0,                                                           //
                stride1,                                                           //
                stride2,                                                           //
                origin0,                                                           //
                origin1,                                                           //
                origin2,                                                           //
                dx,                                                                //
                dy,                                                                //
                dz,                                                                //
                weighted_field_v,                                                  //
                data,                                                              //
                (Generate_I_data ? hex_inout_device : nullptr));                   //
    }
}
