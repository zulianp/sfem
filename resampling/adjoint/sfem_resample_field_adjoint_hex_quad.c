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

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
// generate_poly_bounding_box //////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
int                                                                  //
generate_poly_bounding_box(const real_t* const SFEM_RESTRICT x,      //
                           const real_t* const SFEM_RESTRICT y,      //
                           const real_t* const SFEM_RESTRICT z,      //
                           const ptrdiff_t                   n,      //
                           real_t* const SFEM_RESTRICT       x_min,  //
                           real_t* const SFEM_RESTRICT       x_max,  //
                           real_t* const SFEM_RESTRICT       y_min,  //
                           real_t* const SFEM_RESTRICT       y_max,  //
                           real_t* const SFEM_RESTRICT       z_min,  //
                           real_t* const SFEM_RESTRICT       z_max) {      //

    if (n <= 0) {
        return -1;  // Invalid number of points
    }

    // Initialize min and max with the first point
    *x_min = x[0];
    *x_max = x[0];
    *y_min = y[0];
    *y_max = y[0];
    *z_min = z[0];
    *z_max = z[0];

    // Loop through all points to find min and max
    for (ptrdiff_t i = 1; i < n; i++) {
        if (x[i] < *x_min) {
            *x_min = x[i];
        }
        if (x[i] > *x_max) {
            *x_max = x[i];
        }
        if (y[i] < *y_min) {
            *y_min = y[i];
        }
        if (y[i] > *y_max) {
            *y_max = y[i];
        }
        if (z[i] < *z_min) {
            *z_min = z[i];
        }
        if (z[i] > *z_max) {
            *z_max = z[i];
        }
    }

    return 0;  // Success
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_inv_transform ////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void                                                     //
tet4_inv_transform(const real_t                px0,      // X-coordinate
                   const real_t                px1,      //
                   const real_t                px2,      //
                   const real_t                px3,      //
                   const real_t                py0,      // Y-coordinate
                   const real_t                py1,      //
                   const real_t                py2,      //
                   const real_t                py3,      //
                   const real_t                pz0,      // Z-coordinate
                   const real_t                pz1,      //
                   const real_t                pz2,      //
                   const real_t                pz3,      //
                   const real_t                pfx,      // Input point int the physical space
                   const real_t                pfy,      //
                   const real_t                pfz,      //
                   real_t* const SFEM_RESTRICT out_x,    // Output point in the reference space
                   real_t* const SFEM_RESTRICT out_y,    //
                   real_t* const SFEM_RESTRICT out_z) {  //
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

    // Compute the Jacobian matrix components
    const real_t J11 = -px0 + px1;
    const real_t J12 = -px0 + px2;
    const real_t J13 = -px0 + px3;

    const real_t J21 = -py0 + py1;
    const real_t J22 = -py0 + py2;
    const real_t J23 = -py0 + py3;

    const real_t J31 = -pz0 + pz1;
    const real_t J32 = -pz0 + pz2;
    const real_t J33 = -pz0 + pz3;

    // Compute the determinant of the Jacobian
    const real_t det_J = J11 * (J22 * J33 - J23 * J32) - J12 * (J21 * J33 - J23 * J31) + J13 * (J21 * J32 - J22 * J31);

    // Compute the inverse of the Jacobian matrix
    const real_t inv_J11 = (J22 * J33 - J23 * J32) / det_J;
    const real_t inv_J12 = (J13 * J32 - J12 * J33) / det_J;
    const real_t inv_J13 = (J12 * J23 - J13 * J22) / det_J;

    const real_t inv_J21 = (J23 * J31 - J21 * J33) / det_J;
    const real_t inv_J22 = (J11 * J33 - J13 * J31) / det_J;
    const real_t inv_J23 = (J13 * J21 - J11 * J23) / det_J;

    const real_t inv_J31 = (J21 * J32 - J22 * J31) / det_J;
    const real_t inv_J32 = (J12 * J31 - J11 * J32) / det_J;
    const real_t inv_J33 = (J11 * J22 - J12 * J21) / det_J;

    // Compute the difference between the physical point and the origin
    const real_t dx = pfx - px0;
    const real_t dy = pfy - py0;
    const real_t dz = pfz - pz0;

    // Apply the inverse transformation
    *out_x = inv_J11 * dx + inv_J12 * dy + inv_J13 * dz;
    *out_y = inv_J21 * dx + inv_J22 * dy + inv_J23 * dz;
    *out_z = inv_J31 * dx + inv_J32 * dy + inv_J33 * dz;
}  // END: sfem_resample_field_adjoint_hex_quad

void tet4_face_normal(const real_t px0,  //
                      const real_t py0,  //
                      const real_t pz0,  //
                      const real_t px1,  //
                      const real_t py1,  //
                      const real_t pz1,  //
                      const real_t px2,  //
                      const real_t py2,  //
                      const real_t pz2,  //
                      real_t*      nx,   //
                      real_t*      ny,   //
                      real_t*      nz) {      //

    // Compute edge vectors
    const real_t e1x = px1 - px0;
    const real_t e1y = py1 - py0;
    const real_t e1z = pz1 - pz0;

    const real_t e2x = px2 - px0;
    const real_t e2y = py2 - py0;
    const real_t e2z = pz2 - pz0;

    // Compute cross product e1 x e2
    *nx = e1y * e2z - e1z * e2y;
    *ny = e1z * e2x - e1x * e2z;
    *nz = e1x * e2y - e1y * e2x;
}  // END: sfem_resample_field_adjoint_hex_quad

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_faces_normals ////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void                                               //
tet4_faces_normals(const real_t px0,               //
                   const real_t px1,               //
                   const real_t px2,               //
                   const real_t px3,               //
                   const real_t py0,               //
                   const real_t py1,               //
                   const real_t py2,               //
                   const real_t py3,               //
                   const real_t pz0,               //
                   const real_t pz1,               //
                   const real_t pz2,               //
                   const real_t pz3,               //
                   real_t       normals[4][3],     //
                   real_t       faces_centroid[4][3]) {  //

    // Compute tetrahedron centroid
    const real_t cx = (px0 + px1 + px2 + px3) * 0.25;
    const real_t cy = (py0 + py1 + py2 + py3) * 0.25;
    const real_t cz = (pz0 + pz1 + pz2 + pz3) * 0.25;

    // Store face centroids
    faces_centroid[0][0] = (px1 + px2 + px3) / 3.0;
    faces_centroid[0][1] = (py1 + py2 + py3) / 3.0;
    faces_centroid[0][2] = (pz1 + pz2 + pz3) / 3.0;
    faces_centroid[1][0] = (px0 + px3 + px2) / 3.0;
    faces_centroid[1][1] = (py0 + py3 + py2) / 3.0;
    faces_centroid[1][2] = (pz0 + pz3 + pz2) / 3.0;
    faces_centroid[2][0] = (px0 + px1 + px3) / 3.0;
    faces_centroid[2][1] = (py0 + py1 + py3) / 3.0;
    faces_centroid[2][2] = (pz0 + pz1 + pz3) / 3.0;
    faces_centroid[3][0] = (px0 + px2 + px1) / 3.0;
    faces_centroid[3][1] = (py0 + py2 + py1) / 3.0;
    faces_centroid[3][2] = (pz0 + pz2 + pz1) / 3.0;

    // Face 0: vertices 1, 2, 3 (opposite to vertex 0)
    tet4_face_normal(px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, &normals[0][0], &normals[0][1], &normals[0][2]);

    // Check orientation: vector from centroid to face center should align with normal
    const real_t fc0_x = faces_centroid[0][0] - cx;
    const real_t fc0_y = faces_centroid[0][1] - cy;
    const real_t fc0_z = faces_centroid[0][2] - cz;
    const real_t dot0  = normals[0][0] * fc0_x + normals[0][1] * fc0_y + normals[0][2] * fc0_z;
    if (dot0 < 0.0) {
        normals[0][0] = -normals[0][0];
        normals[0][1] = -normals[0][1];
        normals[0][2] = -normals[0][2];
    }  // END if (dot0 < 0.0)

    // Face 1: vertices 0, 3, 2 (opposite to vertex 1)
    tet4_face_normal(px0, py0, pz0, px3, py3, pz3, px2, py2, pz2, &normals[1][0], &normals[1][1], &normals[1][2]);

    const real_t fc1_x = faces_centroid[1][0] - cx;
    const real_t fc1_y = faces_centroid[1][1] - cy;
    const real_t fc1_z = faces_centroid[1][2] - cz;
    const real_t dot1  = normals[1][0] * fc1_x + normals[1][1] * fc1_y + normals[1][2] * fc1_z;
    if (dot1 < 0.0) {
        normals[1][0] = -normals[1][0];
        normals[1][1] = -normals[1][1];
        normals[1][2] = -normals[1][2];
    }  // END if (dot1 < 0.0)

    // Face 2: vertices 0, 1, 3 (opposite to vertex 2)
    tet4_face_normal(px0, py0, pz0, px1, py1, pz1, px3, py3, pz3, &normals[2][0], &normals[2][1], &normals[2][2]);

    const real_t fc2_x = faces_centroid[2][0] - cx;
    const real_t fc2_y = faces_centroid[2][1] - cy;
    const real_t fc2_z = faces_centroid[2][2] - cz;
    const real_t dot2  = normals[2][0] * fc2_x + normals[2][1] * fc2_y + normals[2][2] * fc2_z;
    if (dot2 < 0.0) {
        normals[2][0] = -normals[2][0];
        normals[2][1] = -normals[2][1];
        normals[2][2] = -normals[2][2];
    }  // END if (dot2 < 0.0)

    // Face 3: vertices 0, 2, 1 (opposite to vertex 3)
    tet4_face_normal(px0, py0, pz0, px2, py2, pz2, px1, py1, pz1, &normals[3][0], &normals[3][1], &normals[3][2]);

    const real_t fc3_x = faces_centroid[3][0] - cx;
    const real_t fc3_y = faces_centroid[3][1] - cy;
    const real_t fc3_z = faces_centroid[3][2] - cz;
    const real_t dot3  = normals[3][0] * fc3_x + normals[3][1] * fc3_y + normals[3][2] * fc3_z;
    if (dot3 < 0.0) {
        normals[3][0] = -normals[3][0];
        normals[3][1] = -normals[3][1];
        normals[3][2] = -normals[3][2];
    }  // END if (dot3 < 0.0)

}  // END Function: tet4_faces_normals

/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
// is_point_in_tet_n ////////////////////////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
void                                                       //
is_point_in_tet_n(const int     N,                         //
                  const real_t  tet4_faces_normals[4][3],  //
                  const real_t  faces_centroids[4][3],     //
                  const real_t* ptx,                       //
                  const real_t* pty,                       //
                  const real_t* ptz,                       //
                  bool*         results) {                         //

    for (int i = 0; i < N; i++) {
        results[i] = true;

        for (int f = 0; f < 4; f++) {
            const real_t nx = tet4_faces_normals[f][0];
            const real_t ny = tet4_faces_normals[f][1];
            const real_t nz = tet4_faces_normals[f][2];

            const real_t cx = faces_centroids[f][0];
            const real_t cy = faces_centroids[f][1];
            const real_t cz = faces_centroids[f][2];

            // Vector from face centroid to point
            const real_t vx = ptx[i] - cx;
            const real_t vy = pty[i] - cy;
            const real_t vz = ptz[i] - cz;

            const real_t dot = nx * vx + ny * vy + nz * vz;

            // For outward normals: point is inside if dot product is negative
            // (vector to point is opposite to outward normal)
            if (dot > 0.0) {
                results[i] = false;
                break;  // No need to check other faces
            }  // END if (dot > 0.0)
        }  // END for (int f = 0; f < 4; f++)
    }  // END for (int i = 0; i < N; i++)
}  // END Function: is_point_in_tet_n

/////////////////////////////////////////////////////////
// tet4_inv_Jacobian ////////////////////////////
/////////////////////////////////////////////////////////
void                                  //
tet4_inv_Jacobian(const real_t px0,   //
                  const real_t px1,   //
                  const real_t px2,   //
                  const real_t px3,   //
                  const real_t py0,   //
                  const real_t py1,   //
                  const real_t py2,   //
                  const real_t py3,   //
                  const real_t pz0,   //
                  const real_t pz1,   //
                  const real_t pz2,   //
                  const real_t pz3,   //
                  real_t       J_inv[9]) {  //
    //
    //

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
    const real_t J11 = -px0 + px1;
    const real_t J12 = -px0 + px2;
    const real_t J13 = -px0 + px3;

    const real_t J21 = -py0 + py1;
    const real_t J22 = -py0 + py2;
    const real_t J23 = -py0 + py3;

    const real_t J31 = -pz0 + pz1;
    const real_t J32 = -pz0 + pz2;
    const real_t J33 = -pz0 + pz3;

    // Compute the determinant of the Jacobian
    const real_t det_J     = J11 * (J22 * J33 - J23 * J32) - J12 * (J21 * J33 - J23 * J31) + J13 * (J21 * J32 - J22 * J31);
    const real_t inv_det_J = 1.0 / det_J;

    // Compute the inverse of the Jacobian matrix
    J_inv[0] = (J22 * J33 - J23 * J32) * inv_det_J;
    J_inv[1] = (J13 * J32 - J12 * J33) * inv_det_J;
    J_inv[2] = (J12 * J23 - J13 * J22) * inv_det_J;
    J_inv[3] = (J23 * J31 - J21 * J33) * inv_det_J;
    J_inv[4] = (J11 * J33 - J13 * J31) * inv_det_J;
    J_inv[5] = (J13 * J21 - J11 * J23) * inv_det_J;
    J_inv[6] = (J21 * J32 - J22 * J31) * inv_det_J;
    J_inv[7] = (J12 * J31 - J11 * J32) * inv_det_J;
    J_inv[8] = (J11 * J22 - J12 * J21) * inv_det_J;
}  // END: sfem_resample_field_adjoint_hex_quad

/////////////////////////////////////////////////////////
// tet4_inv_transform_J ////////////////////////////
/////////////////////////////////////////////////////////
void tet4_inv_transform_J(const real_t*               J_inv,    //
                          const real_t                pfx,      //
                          const real_t                pfy,      //
                          const real_t                pfz,      //
                          const real_t                px0,      //
                          const real_t                py0,      //
                          const real_t                pz0,      //
                          real_t* const SFEM_RESTRICT out_x,    //
                          real_t* const SFEM_RESTRICT out_y,    //
                          real_t* const SFEM_RESTRICT out_z) {  //
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

    // Compute the difference between the physical point and the origin
    const real_t dx = pfx - px0;
    const real_t dy = pfy - py0;
    const real_t dz = pfz - pz0;

    // Apply the inverse transformation
    *out_x = J_inv[0] * dx + J_inv[1] * dy + J_inv[2] * dz;
    *out_y = J_inv[3] * dx + J_inv[4] * dy + J_inv[5] * dz;
    *out_z = J_inv[6] * dx + J_inv[7] * dy + J_inv[8] * dz;
}  // END: sfem_resample_field_adjoint_hex_quad

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
// compute_tet_bounding_box ////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
int                                                                        //
compute_tet_bounding_box(const real_t                         x0,          //
                         const real_t                         x1,          //
                         const real_t                         x2,          //
                         const real_t                         x3,          //
                         const real_t                         y0,          //
                         const real_t                         y1,          //
                         const real_t                         y2,          //
                         const real_t                         y3,          //
                         const real_t                         z0,          //
                         const real_t                         z1,          //
                         const real_t                         z2,          //
                         const real_t                         z3,          //
                         const ptrdiff_t* const SFEM_RESTRICT stride,      //
                         const geom_t* const SFEM_RESTRICT    origin,      //
                         const geom_t* const SFEM_RESTRICT    delta,       //
                         ptrdiff_t* const SFEM_RESTRICT       min_grid_x,  //
                         ptrdiff_t* const SFEM_RESTRICT       max_grid_x,  //
                         ptrdiff_t* const SFEM_RESTRICT       min_grid_y,  //
                         ptrdiff_t* const SFEM_RESTRICT       max_grid_y,  //
                         ptrdiff_t* const SFEM_RESTRICT       min_grid_z,  //
                         ptrdiff_t* const SFEM_RESTRICT       max_grid_z) {      //

    const real_t x_min = fmin(fmin(x0, x1), fmin(x2, x3));
    const real_t x_max = fmax(fmax(x0, x1), fmax(x2, x3));

    const real_t y_min = fmin(fmin(y0, y1), fmin(y2, y3));
    const real_t y_max = fmax(fmax(y0, y1), fmax(y2, y3));

    const real_t z_min = fmin(fmin(z0, z1), fmin(z2, z3));
    const real_t z_max = fmax(fmax(z0, z1), fmax(z2, z3));

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    // Step 2: Convert to grid indices accounting for the origin
    // Formula: grid_index = (physical_coord - origin) / delta
    // Using floor for minimum indices (with safety margin of -1)
    *min_grid_x = floor((x_min - ox) / dx) - 1;
    *min_grid_y = floor((y_min - oy) / dy) - 1;
    *min_grid_z = floor((z_min - oz) / dz) - 1;

    // Using ceil for maximum indices (with safety margin of +1)
    *max_grid_x = ceil((x_max - ox) / dx) + 1;
    *max_grid_y = ceil((y_max - oy) / dy) + 1;
    *max_grid_z = ceil((z_max - oz) / dz) + 1;

    return 0;  // Success
}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
// midpoint_quadrature /////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
int                                     //
midpoint_quadrature(const int N,        //
                    real_t*   nodes,    //
                    real_t*   weights) {  //
    if (N <= 0) {
        return -1;  // Invalid number of points
    }

    const real_t weight = 1.0 / (real_t)N;  // Equal weights for midpoint rule

    for (int i = 0; i < N; i++) {
        nodes[i]   = (real_t)(i + 0.5) / (real_t)N;  // Midpoint in each subinterval
        weights[i] = weight;                         // Assign equal weight
    }

    return 0;  // Success
}

typedef enum { TET_QUAD_MIDPOINT_NQP } tet_quad_midpoint_nqp_t;

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
// sfem_quad_rule_3D ///////////////////////////////////
//////////////////////////////////////////////////////////
int                                                    //
sfem_quad_rule_3D(const tet_quad_midpoint_nqp_t rule,  //
                  const int                     N,     //
                  real_t*                       qx,    //
                  real_t*                       qy,    //
                  real_t*                       qz,    //
                  real_t*                       qw) {                        //
    switch (rule) {
        case TET_QUAD_MIDPOINT_NQP: {
            real_t *nodes    = (real_t*)malloc(N * sizeof(real_t)),  //
                    *weights = (real_t*)malloc(N * sizeof(real_t));

            // Compute 1D midpoint quadrature points and weights
            midpoint_quadrature(N, nodes, weights);

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

typedef struct {
    real_t x, y, z;    // Physical coordinates
    real_t weight;     // Physical weight
    bool   is_inside;  // Containment result
} quadrature_point_result_t;

static inline quadrature_point_result_t                       //
transform_and_check_quadrature_point(                         //
        const int                         q_ijk,              //
        const real_t                      volume_main_tet,    //
        const real_t* const SFEM_RESTRICT Q_nodes_x,          //
        const real_t* const SFEM_RESTRICT Q_nodes_y,          //
        const real_t* const SFEM_RESTRICT Q_nodes_z,          //
        const real_t* const SFEM_RESTRICT Q_weights,          //
        const geom_t* const SFEM_RESTRICT origin,             //
        const geom_t* const SFEM_RESTRICT delta,              //
        const ptrdiff_t                   i_grid,             //
        const ptrdiff_t                   j_grid,             //
        const ptrdiff_t                   k_grid,             //
        const real_t                      tet_vertices_x[4],  //
        const real_t                      tet_vertices_y[4],  //
        const real_t                      tet_vertices_z[4]) {                     //

    quadrature_point_result_t result;

    // Transform to physical coordinates
    // Q_nodes are in [0,1] reference space, need to map to the specific grid cell [i_grid, i_grid+1]

    result.x = ((real_t)i_grid + Q_nodes_x[q_ijk]) * delta[0] + origin[0];
    result.y = ((real_t)j_grid + Q_nodes_y[q_ijk]) * delta[1] + origin[1];
    result.z = ((real_t)k_grid + Q_nodes_z[q_ijk]) * delta[2] + origin[2];

    // printf("delta: %.2e %.2e %.2e, ", delta[0], delta[1], delta[2]);
    // printf("origin: %.2e %.2e %.2e, ", origin[0], origin[1], origin[2]);
    // printf(" phys coord: %.3e %.3e %.3e \n", result.x, result.y, result.z);

    // Compute physical weight

    // Q_weights[q_ijk] is already the product of 3 1D weights, so just scale by volume
    result.weight = Q_weights[q_ijk] * delta[0] * delta[1] * delta[2];

    // Check containment
    check_point_in_tet(1,
                       &result.x,
                       &result.y,
                       &result.z,
                       volume_main_tet,
                       tet_vertices_x[0],
                       tet_vertices_x[1],
                       tet_vertices_x[2],
                       tet_vertices_x[3],
                       tet_vertices_y[0],
                       tet_vertices_y[1],
                       tet_vertices_y[2],
                       tet_vertices_y[3],
                       tet_vertices_z[0],
                       tet_vertices_z[1],
                       tet_vertices_z[2],
                       tet_vertices_z[3],
                       &result.is_inside);

    return result;
}  // END: transform_and_check_quadrature_point

static inline quadrature_point_result_t                  //
transform_and_check_quadrature_point_n(const int q_ijk,  //
                                                         // const real_t                      volume_main_tet,           //
                                       const real_t                      tet4_faces_normals[4][3],  //
                                       const real_t                      faces_centroids[4][3],     //
                                       const real_t* const SFEM_RESTRICT Q_nodes_x,                 //
                                       const real_t* const SFEM_RESTRICT Q_nodes_y,                 //
                                       const real_t* const SFEM_RESTRICT Q_nodes_z,                 //
                                       const real_t* const SFEM_RESTRICT Q_weights,                 //
                                       const geom_t* const SFEM_RESTRICT origin,                    //
                                       const geom_t* const SFEM_RESTRICT delta,                     //
                                       const ptrdiff_t                   i_grid,                    //
                                       const ptrdiff_t                   j_grid,                    //
                                       const ptrdiff_t                   k_grid,                    //
                                       const real_t                      tet_vertices_x[4],         //
                                       const real_t                      tet_vertices_y[4],         //
                                       const real_t                      tet_vertices_z[4]) {                            //

    quadrature_point_result_t result;

    // Transform to physical coordinates
    // Q_nodes are in [0,1] reference space, need to map to the specific grid cell [i_grid, i_grid+1]

    result.x = ((real_t)i_grid + Q_nodes_x[q_ijk]) * delta[0] + origin[0];
    result.y = ((real_t)j_grid + Q_nodes_y[q_ijk]) * delta[1] + origin[1];
    result.z = ((real_t)k_grid + Q_nodes_z[q_ijk]) * delta[2] + origin[2];

    // printf("delta: %.2e %.2e %.2e, ", delta[0], delta[1], delta[2]);
    // printf("origin: %.2e %.2e %.2e, ", origin[0], origin[1], origin[2]);
    // printf(" phys coord: %.3e %.3e %.3e \n", result.x, result.y, result.z);

    // Compute physical weight

    is_point_in_tet_n(1,                   //
                      tet4_faces_normals,  //
                      faces_centroids,     //
                      &result.x,           //
                      &result.y,           //
                      &result.z,           //
                      &result.is_inside);  //

    // Q_weights[q_ijk] is already the product of 3 1D weights, so just scale by volume
    result.weight = Q_weights[q_ijk] * delta[0] * delta[1] * delta[2];

    return result;
}  // END: transform_and_check_quadrature_point

/////////////////////////////////////////////////////////
// is_hex_out_of_tet ////////////////////////////
/////////////////////////////////////////////////////////
bool                                                 //
is_hex_out_of_tet(const real_t inv_J_tet[9],         //
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
     * *****************************************************************************************
     */

    // Precompute inverse Jacobian components for better cache utilization
    const real_t inv_J00 = inv_J_tet[0];
    const real_t inv_J01 = inv_J_tet[1];
    const real_t inv_J02 = inv_J_tet[2];
    const real_t inv_J10 = inv_J_tet[3];
    const real_t inv_J11 = inv_J_tet[4];
    const real_t inv_J12 = inv_J_tet[5];
    const real_t inv_J20 = inv_J_tet[6];
    const real_t inv_J21 = inv_J_tet[7];
    const real_t inv_J22 = inv_J_tet[8];

    // Track if all vertices violate each constraint
    int all_negative_x  = 1;  // All ref_x < 0
    int all_negative_y  = 1;  // All ref_y < 0
    int all_negative_z  = 1;  // All ref_z < 0
    int all_outside_sum = 1;  // All ref_x + ref_y + ref_z > 1
    // int all_larger_than_one = 1;  // All ref_x > 1, ref_y > 1, ref_z > 1 (not used slow)

    for (int v = 0; v < 8; v++) {
        // Transform hex vertex to tet reference space
        const real_t dx = hex_vertices_x[v] - tet_origin_x;
        const real_t dy = hex_vertices_y[v] - tet_origin_y;
        const real_t dz = hex_vertices_z[v] - tet_origin_z;

        const real_t ref_x = inv_J00 * dx + inv_J01 * dy + inv_J02 * dz;
        const real_t ref_y = inv_J10 * dx + inv_J11 * dy + inv_J12 * dz;
        const real_t ref_z = inv_J20 * dx + inv_J21 * dy + inv_J22 * dz;

        // Point is inside tet if: ref_x >= 0 AND ref_y >= 0 AND ref_z >= 0 AND ref_x + ref_y + ref_z <= 1
        // Point is outside if it violates ANY of these constraints

        // Update flags - use bitwise AND for branchless execution
        all_negative_x &= (ref_x < 0.0);
        all_negative_y &= (ref_y < 0.0);
        all_negative_z &= (ref_z < 0.0);
        all_outside_sum &= ((ref_x + ref_y + ref_z) > 1.0);

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
}  // END Function: is_hex_out_of_tet

//////////////////////////////////////////////////////////
// ijk_index_t ////////////////////////////
//////////////////////////////////////////////////////////
typedef struct ijk_index {
    ptrdiff_t i;
    ptrdiff_t j;
    ptrdiff_t k;
} ijk_index_t;  // END ijk_index_t

/////////////////////////////////////////////////////////
// transfer_weighted_field_tet4_to_hex //////////////////
/////////////////////////////////////////////////////////
ijk_index_t                                                                           //
transfer_weighted_field_tet4_to_hex(const real_t                wf0,                  //
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
                                    const real_t                ox,                   //
                                    const real_t                oy,                   //
                                    const real_t                oz,                   //
                                    const real_t                dx,                   //
                                    const real_t                dy,                   //
                                    const real_t                dz,                   //
                                    real_t* const SFEM_RESTRICT hex_element_field) {  //

    // Compute the weighted contribution from the tetrahedron
    // Using linear shape functions for tetrahedron

    const real_t grid_x = (q_phys_x - ox) / dx;
    const real_t grid_y = (q_phys_y - oy) / dy;
    const real_t grid_z = (q_phys_z - oz) / dz;

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
    hex_aa_8_eval_fun_V(l_x,        // Local coordinates
                        l_y,        //
                        l_z,        //
                        &hex8_f0,   // Output shape functions
                        &hex8_f1,   //
                        &hex8_f2,   //
                        &hex8_f3,   //
                        &hex8_f4,   //
                        &hex8_f5,   //
                        &hex8_f6,   //
                        &hex8_f7);  //

    hex_element_field[0] += wf_quad * hex8_f0 * QW_phys_hex;
    hex_element_field[1] += wf_quad * hex8_f1 * QW_phys_hex;
    hex_element_field[2] += wf_quad * hex8_f2 * QW_phys_hex;
    hex_element_field[3] += wf_quad * hex8_f3 * QW_phys_hex;
    hex_element_field[4] += wf_quad * hex8_f4 * QW_phys_hex;
    hex_element_field[5] += wf_quad * hex8_f5 * QW_phys_hex;
    hex_element_field[6] += wf_quad * hex8_f6 * QW_phys_hex;
    hex_element_field[7] += wf_quad * hex8_f7 * QW_phys_hex;

    return (ijk_index_t){i, j, k};
}  // END transfer_weighted_field_tet4_to_hex

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_refine_adjoint_hyteg ////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                                               //
tet4_resample_field_adjoint_hex_quad_d(const ptrdiff_t                      start_element,        // Mesh
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

    PRINT_CURRENT_FUNCTION;

#if SFEM_LOG_LEVEL >= 5
    printf("Processing elements from %td to %td \n", start_element, end_element);
    printf("Stride: %td %td %td \n", stride[0], stride[1], stride[2]);
#endif

    const real_t volume_hex = delta[0] * delta[1] * delta[2];

    const int off0 = 0;
    const int off1 = stride[0];
    const int off2 = stride[0] + stride[1];
    const int off3 = stride[1];
    const int off4 = stride[2];
    const int off5 = stride[0] + stride[2];
    const int off6 = stride[0] + stride[1] + stride[2];
    const int off7 = stride[1] + stride[2];

    const int N_midpoint = 4;
    const int dim_quad   = N_midpoint * N_midpoint * N_midpoint;
    real_t    Q_nodes_x[dim_quad];
    real_t    Q_nodes_y[dim_quad];
    real_t    Q_nodes_z[dim_quad];
    real_t    Q_weights[dim_quad];

    sfem_quad_rule_3D(TET_QUAD_MIDPOINT_NQP, N_midpoint, Q_nodes_x, Q_nodes_y, Q_nodes_z, Q_weights);

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        // loop over the 4 vertices of the tetrahedron

        idx_t  ev[4];
        real_t inv_J_tet[9];

        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }  // END: for v

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

        ptrdiff_t min_grid_x, max_grid_x;
        ptrdiff_t min_grid_y, max_grid_y;
        ptrdiff_t min_grid_z, max_grid_z;

        real_t face_normals_array[4][3];
        real_t faces_centroids_array[4][3];

        // const real_t vol_tet_main = fabs(tet4_measure_v3(x0_n,    //
        //                                                  x1_n,    //
        //                                                  x2_n,    //
        //                                                  x3_n,    //
        //                                                  y0_n,    //
        //                                                  y1_n,    //
        //                                                  y2_n,    //
        //                                                  y3_n,    //
        //                                                  z0_n,    //
        //                                                  z1_n,    //
        //                                                  z2_n,    //
        //                                                  z3_n));  //

        tet4_faces_normals(x0_n,                    //
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

        compute_tet_bounding_box(x0_n,          //
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
                                 stride,        //
                                 origin,        //
                                 delta,         //
                                 &min_grid_x,   //
                                 &max_grid_x,   //
                                 &min_grid_y,   //
                                 &max_grid_y,   //
                                 &min_grid_z,   //
                                 &max_grid_z);  //

        real_t hex_element_field[8] = {0.0};

        for (int k_grid_z = min_grid_z; k_grid_z < max_grid_z; k_grid_z++) {
            const real_t z_hex_min = ((real_t)k_grid_z) * delta[2] + origin[2];
            const real_t z_hex_max = z_hex_min + delta[2];

            for (int j_grid_y = min_grid_y; j_grid_y < max_grid_y; j_grid_y++) {
                const real_t y_hex_min = ((real_t)j_grid_y) * delta[1] + origin[1];
                const real_t y_hex_max = y_hex_min + delta[1];

                for (int i_grid_x = min_grid_x; i_grid_x < max_grid_x; i_grid_x++) {
                    const real_t x_hex_min = ((real_t)i_grid_x) * delta[0] + origin[0];
                    const real_t x_hex_max = x_hex_min + delta[0];

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

                    const bool is_out_of_tet = is_hex_out_of_tet(inv_J_tet,        //
                                                                 x0_n,             //
                                                                 y0_n,             //
                                                                 z0_n,             //
                                                                 hex_vertices_x,   //
                                                                 hex_vertices_y,   //
                                                                 hex_vertices_z);  //

                    const int i = i_grid_x - min_grid_x;
                    const int j = j_grid_y - min_grid_y;
                    const int k = k_grid_z - min_grid_z;

                    if (is_out_of_tet) continue;  // Skip this hex cell

                    // Midpoint quadrature rule in 3D

                    for (int q_ijk = 0; q_ijk < dim_quad; q_ijk++) {
                        quadrature_point_result_t Qpoint_phys =                //
                                transform_and_check_quadrature_point_n(q_ijk,  //
                                                                               //    vol_tet_main,                          //
                                                                       face_normals_array,                    //
                                                                       faces_centroids_array,                 //
                                                                       Q_nodes_x,                             //
                                                                       Q_nodes_y,                             //
                                                                       Q_nodes_z,                             //
                                                                       Q_weights,                             //
                                                                       origin,                                //
                                                                       delta,                                 //
                                                                       i_grid_x,                              //
                                                                       j_grid_y,                              //
                                                                       k_grid_z,                              //
                                                                       (real_t[4]){x0_n, x1_n, x2_n, x3_n},   //
                                                                       (real_t[4]){y0_n, y1_n, y2_n, y3_n},   //
                                                                       (real_t[4]){z0_n, z1_n, z2_n, z3_n});  //

                        if (Qpoint_phys.is_inside) {
                            for (int v = 0; v < 8; v++) hex_element_field[v] = 0.0;

                            // printf("Element %td, grid (%td,%td,%td), quad point %d is inside tet at phys (%.6f,%.6f,%.6f) \n",
                            //        element_i,
                            //        i_grid_x,
                            //        j_grid_y,
                            //        k_grid_z,
                            //        q_ijk,
                            //        result.x,
                            //        result.y,
                            //        result.z);

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

                            ijk_index_t ijk_indices =                                        //
                                    transfer_weighted_field_tet4_to_hex(wf0,                 //
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
                                                                        origin[0],           //
                                                                        origin[1],           //
                                                                        origin[2],           //
                                                                        delta[0],            //
                                                                        delta[1],            //
                                                                        delta[2],            //
                                                                        hex_element_field);  //

                            const ptrdiff_t base_index = i_grid_x * stride[0] +  //
                                                         j_grid_y * stride[1] +  //
                                                         k_grid_z * stride[2];

                            data[base_index + off0] += hex_element_field[0];  //
                            data[base_index + off1] += hex_element_field[1];  //
                            data[base_index + off2] += hex_element_field[2];  //
                            data[base_index + off3] += hex_element_field[3];  //
                            data[base_index + off4] += hex_element_field[4];  //
                            data[base_index + off5] += hex_element_field[5];  //
                            data[base_index + off6] += hex_element_field[6];  //
                            data[base_index + off7] += hex_element_field[7];  //

                        }  // END: if is_inside
                    }  // END: for q_ijk
                }  // END: for k_grid_z
            }  // END: for i_grid_y
        }  // END: for j_grid_y
    }  // END: for element_i

    RETURN_FROM_FUNCTION(0);
}  // END: Function: tet4_resample_field_adjoint_hex_quad_data
