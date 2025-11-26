#ifndef SFEM_RESAMPLE_FIELD_ADJOINT_HEX_WQUAD_CUH
#define SFEM_RESAMPLE_FIELD_ADJOINT_HEX_WQUAD_CUH

#include "sfem_resample_field_adjoint_hex_quad.cuh"

/////////////////////////////////////////////////////////////////////////////////
// Kernel to perform adjoint mini-tetrahedron resampling Version 2
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType,                                                                //
          typename IntType = ptrdiff_t,                                                      //
          IntType N_wf>                                                                      //
__device__ __forceinline__ void                                                              //
tet4_resample_field_adjoint_hex_quad_element_nw_gpu(const IntType           element_i,       // element index
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
                                                    FloatType* const        data,            //
                                                    FloatType*              I_data) {                     // Output data

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

#endif  // SFEM_RESAMPLE_FIELD_ADJOINT_HEX_WQUAD_CUH