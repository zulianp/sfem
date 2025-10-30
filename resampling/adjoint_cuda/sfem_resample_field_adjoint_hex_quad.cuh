#ifndef __SFEM_RESAMPLE_FIELD_ADJOINT_HEX_QUAD_CUH__
#define __SFEM_RESAMPLE_FIELD_ADJOINT_HEX_QUAD_CUH__

#include "sfem_adjoint_mini_tet.cuh"
#include "sfem_adjoint_mini_tet_fun.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

typedef enum { TET_QUAD_MIDPOINT_NQP } tet_quad_midpoint_nqp_gpu_t;

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// compute_tet_bounding_box_gpu //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType, typename IntType = ptrdiff_t>
__device__ int                                                           //
compute_tet_bounding_box_gpu(const FloatType              x0,            //
                             const FloatType              x1,            //
                             const FloatType              x2,            //
                             const FloatType              x3,            //
                             const FloatType              y0,            //
                             const FloatType              y1,            //
                             const FloatType              y2,            //
                             const FloatType              y3,            //
                             const FloatType              z0,            //
                             const FloatType              z1,            //
                             const FloatType              z2,            //
                             const FloatType              z3,            //
                             const IntType                stride0,       //
                             const IntType                stride1,       //
                             const IntType                stride2,       //
                             const IntType                origin0,       //
                             const IntType                origin1,       //
                             const IntType                origin2,       //
                             const IntType                delta0,        //
                             const IntType                delta1,        //
                             const IntType                delta2,        //
                             IntType* const SFEM_RESTRICT min_grid_x,    //
                             IntType* const SFEM_RESTRICT max_grid_x,    //
                             IntType* const SFEM_RESTRICT min_grid_y,    //
                             IntType* const SFEM_RESTRICT max_grid_y,    //
                             IntType* const SFEM_RESTRICT min_grid_z,    //
                             IntType* const SFEM_RESTRICT max_grid_z) {  //

    // Use fast_fma for precision optimization where beneficial
    const FloatType x_min = fast_min(fast_min(x0, x1), fast_min(x2, x3));
    const FloatType x_max = fast_max(fast_max(x0, x1), fast_max(x2, x3));

    const FloatType y_min = fast_min(fast_min(y0, y1), fast_min(y2, y3));
    const FloatType y_max = fast_max(fast_max(y0, y1), fast_max(y2, y3));

    const FloatType z_min = fast_min(fast_min(z0, z1), fast_min(z2, z3));
    const FloatType z_max = fast_max(fast_max(z0, z1), fast_max(z2, z3));

    const FloatType dx = (FloatType)delta0;
    const FloatType dy = (FloatType)delta1;
    const FloatType dz = (FloatType)delta2;

    const FloatType ox = (FloatType)origin0;
    const FloatType oy = (FloatType)origin1;
    const FloatType oz = (FloatType)origin2;

    // Step 2: Convert to grid indices accounting for the origin
    // Formula: grid_index = (physical_coord - origin) / delta
    // Using floor for minimum indices (with safety margin of -1)
    *min_grid_x = (IntType)fast_floor((x_min - ox) / dx) - 1;
    *min_grid_y = (IntType)fast_floor((y_min - oy) / dy) - 1;
    *min_grid_z = (IntType)fast_floor((z_min - oz) / dz) - 1;

    // Using ceil for maximum indices (with safety margin of +1)
    *max_grid_x = (IntType)fast_ceil((x_max - ox) / dx) + 1;
    *max_grid_y = (IntType)fast_ceil((y_max - oy) / dy) + 1;
    *max_grid_z = (IntType)fast_ceil((z_max - oz) / dz) + 1;

    return 0;  // Success
}  // END Function: compute_tet_bounding_box_gpu

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_inv_Jacobian_gpu /////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType>
__device__ void                              //
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
midpoint_quadrature_gpu(const int  N,          //
                        FloatType* nodes,      //
                        FloatType* weights) {  //
    if (N <= 0) {
        return -1;  // Invalid number of points
    }

    const FloatType weight = 1.0 / (FloatType)N;  // Equal weights for midpoint rule

    for (int i = 0; i < N; i++) {
        nodes[i]   = (FloatType)(i + 0.5) / (FloatType)N;  // Midpoint in each subinterval
        weights[i] = weight;                               // Assign equal weight
    }

    return 0;  // Success
}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
// sfem_quad_rule_3D ///////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType, typename IntType = ptrdiff_t>
__device__ int                                                 //
sfem_quad_rule_3D_gpu(const tet_quad_midpoint_nqp_gpu_t rule,  //
                      const IntType                     N,     //
                      FloatType*                        qx,    //
                      FloatType*                        qy,    //
                      FloatType*                        qz,    //
                      FloatType*                        qw) {                         //
    switch (rule) {
        case TET_QUAD_MIDPOINT_NQP: {
            FloatType *nodes = (FloatType*)malloc(N * sizeof(FloatType)),  //
                    *weights = (FloatType*)malloc(N * sizeof(FloatType));

            // Compute 1D midpoint quadrature points and weights
            midpoint_quadrature_gpu(N, nodes, weights);

            // Compute weights for the 3D quadrature points
            int idx = 0;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < N; k++) {
                        qw[idx] = weights[i] * weights[j] * weights[k];  // Product of weights
                        qx[idx] = nodes[i];
                        qy[idx] = nodes[j];
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
                                                const geom_t            origin0,         // Origin
                                                const geom_t            origin1,         //
                                                const geom_t            origin2,         //
                                                const geom_t            dx,              // Delta
                                                const geom_t            dy,              //
                                                const geom_t            dz,              //
                                                const FloatType* const  weighted_field,  // Input weighted field
                                                FloatType* const        data) {                 // Output data

    const int tet_id    = (blockIdx.x * blockDim.x + threadIdx.x) / LANES_PER_TILE;
    const int element_i = start_element + tet_id;  // Global element index
    const int warp_id   = threadIdx.x / LANES_PER_TILE;

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

    const IntType N_midpoint = 4;
    const IntType dim_quad   = N_midpoint * N_midpoint * N_midpoint;
    FloatType     Q_nodes_x[dim_quad];
    FloatType     Q_nodes_y[dim_quad];
    FloatType     Q_nodes_z[dim_quad];
    FloatType     Q_weights[dim_quad];

    sfem_quad_rule_3D_gpu<FloatType, IntType>(TET_QUAD_MIDPOINT_NQP,  //
                                              N_midpoint,
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

    compute_tet_bounding_box_gpu(x0_n,          //
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
                                 stride0,       //
                                 stride1,       //
                                 stride2,       //
                                 origin0,       //
                                 origin1,       //
                                 origin2,       //
                                 dx,            //
                                 dy,            //
                                 dz,            //
                                 &min_grid_x,   //
                                 &max_grid_x,   //
                                 &min_grid_y,   //
                                 &max_grid_y,   //
                                 &min_grid_z,   //
                                 &max_grid_z);  //

    real_t hex_element_field[8] = {0.0};
}

#endif  // __SFEM_RESAMPLE_FIELD_ADJOINT_HEX_QUAD_CUH__