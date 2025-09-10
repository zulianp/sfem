#ifndef __SFEM_ADJOINT_MINI_TET_CUH__
#define __SFEM_ADJOINT_MINI_TET_CUH__

#include <cuda_runtime_api.h>
#include <cstddef>
#include "sfem_adjoint_mini_tet_fun.cuh"

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

    // Precompute offsets for grid mapping: (x - o) * inv_d == fma(inv_d, x, (-o)*inv_d)
    const FloatType neg_ox_inv_dx = (-ox) * inv_dx;
    const FloatType neg_oy_inv_dy = (-oy) * inv_dy;
    const FloatType neg_oz_inv_dz = (-oz) * inv_dz;

    const ptrdiff_t quad_iterations = TET_QUAD_NQP / LANES_PER_TILE + ((TET_QUAD_NQP % LANES_PER_TILE) ? 1 : 0);
    const ptrdiff_t quad_start      = lane_id * quad_iterations;

    // printf("TET_QUAD_NQP=%d, LANES_PER_TILE=%d, quad_iterations=%ld, quad_start=%ld\n", TET_QUAD_NQP, LANES_PER_TILE,
    // quad_iterations, quad_start);

    // for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i += LANES_PER_TILE) {  // loop over the quadrature points
    for (ptrdiff_t quad_i = 0; quad_i < quad_iterations; ++quad_i) {  // loop over the quadrature points

        const ptrdiff_t quad_i_tile = quad_start + quad_i;
        // const int quad_i_tile = quad_i + lane_id;

        if (quad_i_tile >= TET_QUAD_NQP) continue;  // skip inactive lanes early

        // Direct loads (avoid ternaries)
        const FloatType qx = tet_qx[quad_i_tile];
        const FloatType qy = tet_qy[quad_i_tile];
        const FloatType qz = tet_qz[quad_i_tile];
        const FloatType qw = tet_qw[quad_i_tile];

        // // Mapping the quadrature point from the reference space to the mini-tetrahedron
        // const FloatType xq_mref = J_ref[0].x * qx + J_ref[0].y * qy + J_ref[0].z * qz + bc.x;
        // const FloatType yq_mref = J_ref[1].x * qx + J_ref[1].y * qy + J_ref[1].z * qz + bc.y;
        // const FloatType zq_mref = J_ref[2].x * qx + J_ref[2].y * qy + J_ref[2].z * qz + bc.z;

        // // Mapping the quadrature point from the mini-tetrahedron to the physical space
        // const FloatType xq_phys = J_phys[0].x * xq_mref + J_phys[0].y * yq_mref + J_phys[0].z * zq_mref + fxyz.x;
        // const FloatType yq_phys = J_phys[1].x * xq_mref + J_phys[1].y * yq_mref + J_phys[1].z * zq_mref + fxyz.y;
        // const FloatType zq_phys = J_phys[2].x * xq_mref + J_phys[2].y * yq_mref + J_phys[2].z * zq_mref + fxyz.z;

        // const FloatType grid_x = (xq_phys - ox) * inv_dx;
        // const FloatType grid_y = (yq_phys - oy) * inv_dy;
        // const FloatType grid_z = (zq_phys - oz) * inv_dz;

        // Mapping the quadrature point from the reference space to the mini-tetrahedron
        const FloatType xq_mref = fast_fma(J_ref[0].z, qz, fast_fma(J_ref[0].y, qy, fast_fma(J_ref[0].x, qx, bc.x)));
        const FloatType yq_mref = fast_fma(J_ref[1].z, qz, fast_fma(J_ref[1].y, qy, fast_fma(J_ref[1].x, qx, bc.y)));
        const FloatType zq_mref = fast_fma(J_ref[2].z, qz, fast_fma(J_ref[2].y, qy, fast_fma(J_ref[2].x, qx, bc.z)));

        // Mapping the quadrature point from the mini-tetrahedron to the physical space
        const FloatType xq_phys =
                fast_fma(J_phys[0].z, zq_mref, fast_fma(J_phys[0].y, yq_mref, fast_fma(J_phys[0].x, xq_mref, fxyz.x)));
        const FloatType yq_phys =
                fast_fma(J_phys[1].z, zq_mref, fast_fma(J_phys[1].y, yq_mref, fast_fma(J_phys[1].x, xq_mref, fxyz.y)));
        const FloatType zq_phys =
                fast_fma(J_phys[2].z, zq_mref, fast_fma(J_phys[2].y, yq_mref, fast_fma(J_phys[2].x, xq_mref, fxyz.z)));

        // Grid coords with fused multiply-add
        const FloatType grid_x = fast_fma(inv_dx, xq_phys, neg_ox_inv_dx);
        const FloatType grid_y = fast_fma(inv_dy, yq_phys, neg_oy_inv_dy);
        const FloatType grid_z = fast_fma(inv_dz, zq_phys, neg_oz_inv_dz);

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

        const FloatType wf_quad = fast_fma(f0, wf0, fast_fma(f1, wf1, fast_fma(f2, wf2, f3 * wf3)));

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
                store_add(&data[cache_base + off0], acc0);
                store_add(&data[cache_base + off1], acc1);
                store_add(&data[cache_base + off2], acc2);
                store_add(&data[cache_base + off3], acc3);
                store_add(&data[cache_base + off4], acc4);
                store_add(&data[cache_base + off5], acc5);
                store_add(&data[cache_base + off6], acc6);
                store_add(&data[cache_base + off7], acc7);
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
    }

    // Flush tail
    if (cache_base != -1) {
        store_add(&data[cache_base + off0], acc0);
        store_add(&data[cache_base + off1], acc1);
        store_add(&data[cache_base + off2], acc2);
        store_add(&data[cache_base + off3], acc3);
        store_add(&data[cache_base + off4], acc4);
        store_add(&data[cache_base + off5], acc5);
        store_add(&data[cache_base + off6], acc6);
        store_add(&data[cache_base + off7], acc7);
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

template <typename FloatType>
class tet_properties_info_t {
public:
    // Arrays allocated on device (length = count)
    ptrdiff_t* delta0_index;  // delta with respect to the global grid in the global grid
    ptrdiff_t* delta1_index;
    ptrdiff_t* delta2_index;

    ptrdiff_t* size0_local;
    ptrdiff_t* size1_local;
    ptrdiff_t* size2_local;

    ptrdiff_t* stride0_local;
    ptrdiff_t* stride1_local;
    ptrdiff_t* stride2_local;

    ptrdiff_t* n0_local;
    ptrdiff_t* n1_local;
    ptrdiff_t* n2_local;

    FloatType* min_x;
    FloatType* min_y;
    FloatType* min_z;

    FloatType* max_x;
    FloatType* max_y;
    FloatType* max_z;

    // Host-side meta
    size_t count;

    __device__ size_t get_tet_grid_size(const ptrdiff_t tet_i) const {
        return (size0_local[tet_i] * size1_local[tet_i] * size2_local[tet_i]);
    }

    __host__ tet_properties_info_t()
        : delta0_index(nullptr),
          delta1_index(nullptr),
          delta2_index(nullptr),
          size0_local(nullptr),
          size1_local(nullptr),
          size2_local(nullptr),
          stride0_local(nullptr),
          stride1_local(nullptr),
          stride2_local(nullptr),
          n0_local(nullptr),
          n1_local(nullptr),
          n2_local(nullptr),
          min_x(nullptr),
          min_y(nullptr),
          min_z(nullptr),
          max_x(nullptr),
          max_y(nullptr),
          max_z(nullptr),
          count(0) {}

    __host__ cudaError_t alloc_async(size_t n, cudaStream_t stream) {
        count           = n;
        cudaError_t err = cudaSuccess;

        auto fail_cleanup = [&](cudaError_t e) {
            (void)free_async(stream);  // best-effort cleanup
            return e;
        };

        // ptrdiff_t arrays
        if ((err = cudaMallocAsync((void**)&delta0_index, n * sizeof(ptrdiff_t), stream)) != cudaSuccess)
            return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&delta1_index, n * sizeof(ptrdiff_t), stream)) != cudaSuccess)
            return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&delta2_index, n * sizeof(ptrdiff_t), stream)) != cudaSuccess)
            return fail_cleanup(err);

        if ((err = cudaMallocAsync((void**)&size0_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&size1_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&size2_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess) return fail_cleanup(err);

        if ((err = cudaMallocAsync((void**)&stride0_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess)
            return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&stride1_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess)
            return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&stride2_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess)
            return fail_cleanup(err);

        if ((err = cudaMallocAsync((void**)&n0_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&n1_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&n2_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess) return fail_cleanup(err);

        // FloatType arrays
        if ((err = cudaMallocAsync((void**)&min_x, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&min_y, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&min_z, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);

        if ((err = cudaMallocAsync((void**)&max_x, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&max_y, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&max_z, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);

        return cudaSuccess;
    }

    __host__ cudaError_t free_async(cudaStream_t stream) {
        // Free if non-null; ignore errors to attempt freeing all
        auto free_if = [&](void* p) {
            if (p) (void)cudaFreeAsync(p, stream);
        };

        free_if(delta0_index);
        free_if(delta1_index);
        free_if(delta2_index);

        free_if(size0_local);
        free_if(size1_local);
        free_if(size2_local);

        free_if(stride0_local);
        free_if(stride1_local);
        free_if(stride2_local);

        free_if(n0_local);
        free_if(n1_local);
        free_if(n2_local);

        free_if(min_x);
        free_if(min_y);
        free_if(min_z);

        free_if(max_x);
        free_if(max_y);
        free_if(max_z);

        reset();
        // Optionally, user can cudaStreamSynchronize(stream) after this
        return cudaSuccess;
    }

private:
    __host__ void reset() {
        delta0_index = delta1_index = delta2_index = nullptr;
        size0_local = size1_local = size2_local = nullptr;
        stride0_local = stride1_local = stride2_local = nullptr;
        n0_local = n1_local = n2_local = nullptr;
        min_x = min_y = min_z = nullptr;
        max_x = max_y = max_z = nullptr;
        count                 = 0;
    }
};

/////////////////////////////////////////////////////////////////////////////////
// Kernel to perform adjoint mini-tetrahedron resampling
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__global__ void                                                                          //
sfem_make_local_tets_kernel_gpu(const ptrdiff_t                  shared_memory_size,     //
                                const ptrdiff_t                  start_element,          // Mesh
                                const ptrdiff_t                  end_element,            //
                                const ptrdiff_t                  nnodes,                 //
                                const elems_tet4_device          elems,                  //
                                const xyz_tet4_device            xyz,                    //
                                const ptrdiff_t                  n0,                     // SDF
                                const ptrdiff_t                  n1,                     //
                                const ptrdiff_t                  n2,                     //
                                const ptrdiff_t                  stride0,                // Stride
                                const ptrdiff_t                  stride1,                //
                                const ptrdiff_t                  stride2,                //
                                const geom_t                     origin0,                // Origin
                                const geom_t                     origin1,                //
                                const geom_t                     origin2,                //
                                const geom_t                     dx,                     // Delta
                                const geom_t                     dy,                     //
                                const geom_t                     dz,                     //
                                tet_properties_info_t<FloatType> tet_properties_info) {  //

    const int tet_id    = (blockIdx.x * blockDim.x + threadIdx.x);
    const int element_i = start_element + tet_id;  // Global element index

    if (element_i >= end_element) return;  // Out of range

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

    const FloatType x_min = fast_min(fast_min(x0_n, x1_n), fast_min(x2_n, x3_n));
    const FloatType y_min = fast_min(fast_min(y0_n, y1_n), fast_min(y2_n, y3_n));
    const FloatType z_min = fast_min(fast_min(z0_n, z1_n), fast_min(z2_n, z3_n));

    const FloatType x_max = fast_max(fast_max(x0_n, x1_n), fast_max(x2_n, x3_n));
    const FloatType y_max = fast_max(fast_max(y0_n, y1_n), fast_max(y2_n, y3_n));
    const FloatType z_max = fast_max(fast_max(z0_n, z1_n), fast_max(z2_n, z3_n));

    const ptrdiff_t min_grid_x = fast_floor((x_min - origin0) / dx) - 1;
    const ptrdiff_t min_grid_y = fast_floor((y_min - origin1) / dy) - 1;
    const ptrdiff_t min_grid_z = fast_floor((z_min - origin2) / dz) - 1;

    const FloatType max_grid_x = fast_ceil((x_max - origin0) / dx) + 1;
    const FloatType max_grid_y = fast_ceil((y_max - origin1) / dy) + 1;
    const FloatType max_grid_z = fast_ceil((z_max - origin2) / dz) + 1;
}

/////////////////////////////////////////////////////////////////////////////////
// Kernel to perform adjoint mini-tetrahedron resampling
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__global__ void                                                                               //
sfem_adjoint_mini_tet_shared_loc_kernel_gpu(const ptrdiff_t             shared_memory_size,   //
                                            const ptrdiff_t             start_element,        // Mesh
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

    const int threading_block_size = blockDim.x;
    if (shared_memory_size < threading_block_size) {
        if (threadIdx.x == 0) {
            printf("Error: Not enough shared memory allocated! Required: %ld, allocated: %ld\n",
                   threading_block_size * sizeof(FloatType),
                   shared_memory_size);
        }
        return;
    }

    extern __shared__ FloatType shared_local_data[];  // Shared memory for local accumulation

    for (int i = threadIdx.x; i < threading_block_size; i += blockDim.x) {
        if (i < threading_block_size) shared_local_data[i] = 0.0;
    }
    __syncthreads();

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