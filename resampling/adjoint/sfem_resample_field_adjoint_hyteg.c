#include "sfem_resample_field.h"
#include "sfem_resample_field_tet4_math.h"
#include "sfem_stack.h"

#include "mass.h"
// #include "read_mesh.h"
#include "matrixio_array.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

// #define real_t  real_t

#include "hyteg.h"
#include "quadratures_rule.h"

#define real_type real_t
#define SFEM_RESTRICT __restrict__

#define SFEM_RESAMPLE_GAP_DUAL

/**
 * @brief Compute tetrahedral dual basis functions and their weighted sum for a point in physical space
 *
 * This function computes the DUAL basis functions for tetrahedral elements at a point in
 * physical space coordinates, and returns their weighted sum.
 *
 * @param[in] x       x-coordinate of the point in physical space
 * @param[in] y       y-coordinate of the point in physical space
 * @param[in] z       z-coordinate of the point in physical space
 * @param[in] x0,y0,z0  Coordinates of the first tetrahedron vertex
 * @param[in] x1,y1,z1  Coordinates of the second tetrahedron vertex
 * @param[in] x2,y2,z2  Coordinates of the third tetrahedron vertex
 * @param[in] x3,y3,z3  Coordinates of the fourth tetrahedron vertex
 * @param[in] wf0     Weight for the first tetrahedral vertex
 * @param[in] wf1     Weight for the second tetrahedral vertex
 * @param[in] wf2     Weight for the third tetrahedral vertex
 * @param[in] wf3     Weight for the fourth tetrahedral vertex
 * @param[out] f0_out Optional pointer to store the first basis function value (can be NULL)
 * @param[out] f1_out Optional pointer to store the second basis function value (can be NULL)
 * @param[out] f2_out Optional pointer to store the third basis function value (can be NULL)
 * @param[out] f3_out Optional pointer to store the fourth basis function value (can be NULL)
 *
 * @return The weighted sum of the basis functions
 */
static real_t tet4_eval_dual_basis_weighted_physical(
        // Point in physical space
        const real_t x, const real_t y, const real_t z,

        // Coordinates of tetrahedron vertices
        const real_t x0, const real_t y0, const real_t z0,  // Vertex 0
        const real_t x1, const real_t y1, const real_t z1,  // Vertex 1
        const real_t x2, const real_t y2, const real_t z2,  // Vertex 2
        const real_t x3, const real_t y3, const real_t z3,  // Vertex 3

        // Weights for vertices
        const real_t wf0, const real_t wf1, const real_t wf2, const real_t wf3,

        // Output basis function values (can be NULL)
        real_t* f0_out, real_t* f1_out, real_t* f2_out, real_t* f3_out) {
    // Convert physical coordinates to local coordinates using barycentric coordinates

    // Matrix components for coordinate transformation

    real_t X[3][3], b[3], det;

    // Set up the system matrix
    X[0][0] = x1 - x0;
    X[0][1] = x2 - x0;
    X[0][2] = x3 - x0;
    X[1][0] = y1 - y0;
    X[1][1] = y2 - y0;
    X[1][2] = y3 - y0;
    X[2][0] = z1 - z0;
    X[2][1] = z2 - z0;
    X[2][2] = z3 - z0;

    // Right-hand side
    b[0] = x - x0;
    b[1] = y - y0;
    b[2] = z - z0;

    // Compute determinant of X
    det = X[0][0] * (X[1][1] * X[2][2] - X[1][2] * X[2][1]) - X[0][1] * (X[1][0] * X[2][2] - X[1][2] * X[2][0]) +
          X[0][2] * (X[1][0] * X[2][1] - X[1][1] * X[2][0]);

    if (fabs(det) < 1e-10) {
        // Degenerate tetrahedron or point outside
        if (f0_out) *f0_out = 0.0;
        if (f1_out) *f1_out = 0.0;
        if (f2_out) *f2_out = 0.0;
        if (f3_out) *f3_out = 0.0;
        return 0.0;
    }

    // Solve the system using Cramer's rule
    real_t qx, qy, qz;

    // Compute local coordinates using inverse mapping
    qx = ((b[0] * (X[1][1] * X[2][2] - X[1][2] * X[2][1]) - X[0][1] * (b[1] * X[2][2] - X[1][2] * b[2]) +
           X[0][2] * (b[1] * X[2][1] - X[1][1] * b[2])) /
          det);

    qy = ((X[0][0] * (b[1] * X[2][2] - X[1][2] * b[2]) - b[0] * (X[1][0] * X[2][2] - X[1][2] * X[2][0]) +
           X[0][2] * (X[1][0] * b[2] - b[1] * X[2][0])) /
          det);

    qz = ((X[0][0] * (X[1][1] * b[2] - b[1] * X[2][1]) - X[0][1] * (X[1][0] * b[2] - b[1] * X[2][0]) +
           b[0] * (X[1][0] * X[2][1] - X[1][1] * X[2][0])) /
          det);

    // Now compute basis functions using local coordinates
    // DUAL basis function (Shape functions for tetrahedral elements)
    const real_t f0 = 1.0 - qx - qy - qz;
    const real_t f1 = qx;
    const real_t f2 = qy;
    const real_t f3 = qz;

    // Values of the shape functions at the quadrature point
    // In the local coordinate system of the tetrahedral element
    const real_t tet4_f0 = 4.0 * f0 - f1 - f2 - f3;
    const real_t tet4_f1 = -f0 + 4.0 * f1 - f2 - f3;
    const real_t tet4_f2 = -f0 - f1 + 4.0 * f2 - f3;
    const real_t tet4_f3 = -f0 - f1 - f2 + 4.0 * f3;

    // Store the basis function values if pointers are provided
    if (f0_out) *f0_out = tet4_f0;
    if (f1_out) *f1_out = tet4_f1;
    if (f2_out) *f2_out = tet4_f2;
    if (f3_out) *f3_out = tet4_f3;

    // Return the weighted sum
    return tet4_f0 * wf0 + tet4_f1 * wf1 + tet4_f2 * wf2 + tet4_f3 * wf3;
}

int                                                                                     //
tet4_resample_tetrahedron_local_hyteg_adjoint(const real_t                         x0,  // Tetrahedron vertices X-coordinates
                                              const real_t                         x1,  //
                                              const real_t                         x2,  //
                                              const real_t                         x3,  //
                                              const real_t                         y0,  // Tetrahedron vertices Y-coordinates
                                              const real_t                         y1,  //
                                              const real_t                         y2,  //
                                              const real_t                         y3,  //
                                              const real_t                         z0,  // Tetrahedron vertices Z-coordinates
                                              const real_t                         z1,  //
                                              const real_t                         z2,  //
                                              const real_t                         z3,  //
                                              const real_t                         det_jacobian,  // determinant of the category.
                                              const real_t                         wf0,     // Weighted field at the vertices
                                              const real_t                         wf1,     //
                                              const real_t                         wf2,     //
                                              const real_t                         wf3,     //
                                              const real_t                         ox,      // Origin of the grid
                                              const real_t                         oy,      //
                                              const real_t                         oz,      //
                                              const real_t                         dx,      // Spacing of the grid
                                              const real_t                         dy,      //
                                              const real_t                         dz,      //
                                              const ptrdiff_t* const SFEM_RESTRICT stride,  // Stride
                                              const ptrdiff_t* const SFEM_RESTRICT n,       // Size of the grid
                                              real_t* const SFEM_RESTRICT          data) {           //

    for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i++) {  // loop over the quadrature points

        real_t g_qx, g_qy, g_qz;

        tet4_transform_v2(x0,              // x-coordinates of the vertices
                          x1,              //
                          x2,              //
                          x3,              //
                          y0,              // y-coordinates of the vertices
                          y1,              //
                          y2,              //
                          y3,              //
                          z0,              // z-coordinates of the vertices
                          z1,              //
                          z2,              //
                          z3,              //
                          tet_qx[quad_i],  // Quadrature point
                          tet_qy[quad_i],  //
                          tet_qz[quad_i],  //
                          &g_qx,           // Output coordinates
                          &g_qy,           //
                          &g_qz);          //

        real_t tet4_f0, tet4_f1, tet4_f2, tet4_f3;
        {
            // DUAL basis function (Shape functions for tetrahedral elements)
            // at the quadrature point
            const real_t f0 = 1.0 - tet_qx[quad_i] - tet_qy[quad_i] - tet_qz[quad_i];
            const real_t f1 = tet_qx[quad_i];
            const real_t f2 = tet_qy[quad_i];
            const real_t f3 = tet_qz[quad_i];

            // Values of the shape functions at the quadrature point
            // In the local coordinate system of the tetrahedral element
            // For each vertex of the tetrahedral element
            tet4_f0 = 4.0 * f0 - f1 - f2 - f3;
            tet4_f1 = -f0 + 4.0 * f1 - f2 - f3;
            tet4_f2 = -f0 - f1 + 4.0 * f2 - f3;
            tet4_f3 = -f0 - f1 - f2 + 4.0 * f3;
        }

        const real_t grid_x = (g_qx - ox) / dx;
        const real_t grid_y = (g_qy - oy) / dy;
        const real_t grid_z = (g_qz - oz) / dz;

        const ptrdiff_t i = floor(grid_x);
        const ptrdiff_t j = floor(grid_y);
        const ptrdiff_t k = floor(grid_z);

        real_t l_x = (grid_x - (real_t)i);
        real_t l_y = (grid_y - (real_t)j);
        real_t l_z = (grid_z - (real_t)k);

        // Critical point
        // Compute the shape functions of the hexahedral (cubic) element
        // at the quadrature point
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

        // Indices of the vertices of the hexahedral element
        ptrdiff_t i0, i1, i2, i3, i4, i5, i6, i7;
        hex_aa_8_collect_coeffs_indices_V(stride,  // Stride
                                          i,       // Indices of the element
                                          j,       //
                                          k,       //
                                          &i0,     // Output indices
                                          &i1,     //
                                          &i2,     //
                                          &i3,     //
                                          &i4,     //
                                          &i5,     //
                                          &i6,     //
                                          &i7);    //

        // Integrate the values of the field at the vertices of the tetrahedral element
        const real_t dV = (1.0 / 6.0) * det_jacobian * tet_qw[quad_i];
        const real_t It = (tet4_f0 * wf0 + tet4_f1 * wf1 + tet4_f2 * wf2 + tet4_f3 * wf3) * dV;

        const real_t d0 = It * hex8_f0;
        const real_t d1 = It * hex8_f1;
        const real_t d2 = It * hex8_f2;
        const real_t d3 = It * hex8_f3;
        const real_t d4 = It * hex8_f4;
        const real_t d5 = It * hex8_f5;
        const real_t d6 = It * hex8_f6;
        const real_t d7 = It * hex8_f7;

        // Update the data
        data[i0] += d0;
        data[i1] += d1;
        data[i2] += d2;
        data[i3] += d3;
        data[i4] += d4;
        data[i5] += d5;
        data[i6] += d6;
        data[i7] += d7;
    }
}

real_t                                                      //
calculate_det_Jacobian_for_category(const int    category,  //
                                    const real_t px0,       //
                                    const real_t py0,       //
                                    const real_t pz0,       //
                                    const real_t px1,       //
                                    const real_t py1,       //
                                    const real_t pz1,       //
                                    const real_t px2,       //
                                    const real_t py2,       //
                                    const real_t pz2,       //
                                    const real_t px3,       //
                                    const real_t py3,       //
                                    const real_t pz3,       //
                                    const real_t L,         //
                                    const int    tet_i,     //
                                    int*         error_flag) {      //
    real_t det_jacobian = 0.0;
    *error_flag         = 0;

    switch (category) {
        case 0:  // Category 0
            det_jacobian = det_jacobian_cat0_real(px0, py0, pz0, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, L);
            break;
        case 1:  // Category 1
            det_jacobian = det_jacobian_cat1_real(px0, py0, pz0, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, L);
            break;
        case 2:  // Category 2
            det_jacobian = det_jacobian_cat2_real(px0, py0, pz0, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, L);
            break;
        case 3:  // Category 3
            det_jacobian = det_jacobian_cat3_real(px0, py0, pz0, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, L);
            break;
        case 4:  // Category 4
            det_jacobian = det_jacobian_cat4_real(px0, py0, pz0, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, L);
            break;
        case 5:  // Category 5
            det_jacobian = det_jacobian_cat5_real(px0, py0, pz0, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, L);
            break;
        default:  // Invalid category
            fprintf(stderr,
                    "calculate_det_Jacobian_for_category: Invalid category %d for tetrahedron %d at level "
                    "%d\n",
                    category,
                    tet_i,
                    L);
            *error_flag = -1;
            // The decision to exit should be made by the caller
            break;
    }
    return det_jacobian;
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
// alpha_to_hyteg_level //////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
int                                                     //
alpha_to_hyteg_level(const real_t alpha,                //
                     const real_t alpha_min_threshold,  //
                     const real_t alpha_max_threshold,  //
                     const int    max_refinement_L) {      //

    // return 1;  ///// TODO

    if (alpha < alpha_min_threshold) return 1;                           // No refinement
    if (alpha > alpha_max_threshold) return HYTEG_MAX_REFINEMENT_LEVEL;  // Maximum refinement

    real_t alpha_x = alpha - alpha_min_threshold;  // Shift the alpha to start from 0
    real_t L_real  = (alpha_x / (alpha_max_threshold - alpha_min_threshold) * (real_t)(HYTEG_MAX_REFINEMENT_LEVEL - 1)) + 1.0;

    int L = L_real >= 1.0 ? (int)L_real : 1;                                  // Convert to integer
    L     = L > HYTEG_MAX_REFINEMENT_LEVEL ? HYTEG_MAX_REFINEMENT_LEVEL : L;  // Clamp to maximum level

    return L >= max_refinement_L ? max_refinement_L : L;  // Return the level, clamped to max_refinement_L
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_refine_adjoint_hyteg ////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                                                  //
tet4_resample_field_local_refine_adjoint_hyteg(const ptrdiff_t                      start_element,   // Mesh
                                               const ptrdiff_t                      end_element,     //
                                               const ptrdiff_t                      nnodes,          //
                                               const idx_t** const SFEM_RESTRICT    elems,           //
                                               const geom_t** const SFEM_RESTRICT   xyz,             //
                                               const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                                               const ptrdiff_t* const SFEM_RESTRICT stride,          //
                                               const geom_t* const SFEM_RESTRICT    origin,          //
                                               const geom_t* const SFEM_RESTRICT    delta,           //
                                               const real_t* const SFEM_RESTRICT    weighted_field,  // Input weighted field
                                               const real_t                         alpha_th,        // Threshold for alpha
                                               real_t* const SFEM_RESTRICT          data) {                   //

    PRINT_CURRENT_FUNCTION;
    int ret = 0;

    // The minimum and maximum thresholds for alpha are used to determine the level of refinement.
    // If the alpha value is below the minimum threshold, no refinement is applied.
    // If the alpha value is above the maximum threshold, the maximum level of refinement is applied.

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    const real_t d_min             = dx < dy ? (dx < dz ? dx : dz) : (dy < dz ? dy : dz);
    const real_t hexahedron_volume = dx * dy * dz;

#if SFEM_LOG_LEVEL >= 5
    printf("============================================================\n");
    printf("= Start: %s: %s:%d \n", __FUNCTION__, __FILE__, __LINE__);
    printf("= Hexahedron volume = %g\n", hexahedron_volume);
    printf("============================================================\n");
#endif

    int degenerated_tetrahedra_cnt = 0;
    int uniform_refine_cnt         = 0;

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        // loop over the 4 vertices of the tetrahedron
        idx_t ev[4];
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }

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

        // Compute the alpha_tet to decide if the tetrahedron is refined
        // Sides of the tetrahedron
        real_t edges_length[6];

        int vertex_a = -1;
        int vertex_b = -1;

        const real_t max_edges_length =             //
                tet_edge_max_length(x0_n,           //
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

        const real_t alpha_tet = max_edges_length / d_min;

        const real_t alpha_min_threshold = 1.7;  // Minimum threshold for alpha
        const real_t alpha_max_threshold = 8.0;  // Maximum threshold for alpha. Less: make more refinements.
        const int    max_refinement_L    = 4;    // Maximum refinement level

        const int L = alpha_to_hyteg_level(alpha_tet,            //
                                           alpha_min_threshold,  //
                                           alpha_max_threshold,  //
                                           max_refinement_L);    //

        const int     hyteg_num_tetrahedra = get_hyteg_num_tetrahedra(L);
        const real_t* x_hyteg              = get_hyteg_x(L);
        const real_t* y_hyteg              = get_hyteg_y(L);
        const real_t* z_hyteg              = get_hyteg_z(L);
        const int*    categories_hyteg     = get_hyteg_categories(L);
        const int*    v0_array_hyteg       = get_hyteg_v0(L);
        const int*    v1_array_hyteg       = get_hyteg_v1(L);
        const int*    v2_array_hyteg       = get_hyteg_v2(L);
        const int*    v3_array_hyteg       = get_hyteg_v3(L);

        // if (L > 1 && alpha_tet > alpha_min_threshold) {
        //     printf("Refinement level for tetrahedron %ld: %d, alpha_tet = %g, max_edges_length = %g, d_min = %g, num tet:
        //     %d\n",
        //            element_i,
        //            L,
        //            alpha_tet,
        //            max_edges_length,
        //            d_min,
        //            hteg_num_tetrahedra);
        // }

        // DEBUG: theta_volume_main
        const real_t theta_volume_main = tet4_measure_v2(x0_n,  //
                                                         x1_n,  //
                                                         x2_n,  //
                                                         x3_n,  //
                                                         //
                                                         y0_n,  //
                                                         y1_n,  //
                                                         y2_n,  //
                                                         y3_n,  //
                                                         //
                                                         z0_n,   //
                                                         z1_n,   //
                                                         z2_n,   //
                                                         z3_n);  //

        real_t theta_volume_acc = 0.0;  // DEBUG: theta_volume_acc

        // printf("Num tet = %d, L = %d, alpha_tet = %g, max_edges_length = %g, d_min = %g\n",
        //        hteg_num_tetrahedra,
        //        L,
        //        alpha_tet,
        //        max_edges_length,
        //        d_min);

        for (int tet_i = 0; tet_i < hyteg_num_tetrahedra; tet_i++) {  //
            //
            const int category = categories_hyteg[tet_i];

            const int v0 = v0_array_hyteg[tet_i];
            const int v1 = v1_array_hyteg[tet_i];
            const int v2 = v2_array_hyteg[tet_i];
            const int v3 = v3_array_hyteg[tet_i];

            // Coordinates of the HyTeg tetrahedron in the reference space
            const real_t px0 = x_hyteg[v0];
            const real_t px1 = x_hyteg[v1];
            const real_t px2 = x_hyteg[v2];
            const real_t px3 = x_hyteg[v3];

            const real_t py0 = y_hyteg[v0];
            const real_t py1 = y_hyteg[v1];
            const real_t py2 = y_hyteg[v2];
            const real_t py3 = y_hyteg[v3];

            const real_t pz0 = z_hyteg[v0];
            const real_t pz1 = z_hyteg[v1];
            const real_t pz2 = z_hyteg[v2];
            const real_t pz3 = z_hyteg[v3];

            // printf("Tet vertices: px0 = %g, py0 = %g, pz0 = %g || px1 = %g, py1 = %g, pz1 = %g || "
            //        "px2 = %g, py2 = %g, pz2 = %g || px3 = %g, py3 = %g, pz3 = %g\n",
            //        px0,
            //        py0,
            //        pz0,
            //        px1,
            //        py1,
            //        pz1,
            //        px2,
            //        py2,
            //        pz2,
            //        px3,
            //        py3,
            //        pz3);

            real_t fx0, fx1, fx2, fx3;
            real_t fy0, fy1, fy2, fy3;
            real_t fz0, fz1, fz2, fz3;

            // Transform the vertices of the sub-tetrahedron to the physical space
            tet4_transform_v2(x0_n,   // x-coordinates of the vertices
                              x1_n,   //
                              x2_n,   //
                              x3_n,   //
                              y0_n,   // y-coordinates of the vertices
                              y1_n,   //
                              y2_n,   //
                              y3_n,   //
                              z0_n,   // z-coordinates of the vertices
                              z1_n,   //
                              z2_n,   //
                              z3_n,   //
                              px0,    // Vertex of the sub-tetrahedron
                              py0,    //
                              pz0,    //
                              &fx0,   // Output coordinates
                              &fy0,   //
                              &fz0);  //

            // printf("px0: %g, py0: %g, pz0: %g => fx0: %g, fy0: %g, fz0: %g || x0_n: %g, y0_n: %g, z0_n: %g\n",
            //        px0,
            //        py0,
            //        pz0,
            //        fx0,
            //        fy0,
            //        fz0,
            //        x0_n,
            //        y0_n,
            //        z0_n);

            tet4_transform_v2(x0_n,   // x-coordinates of the vertices
                              x1_n,   //
                              x2_n,   //
                              x3_n,   //
                              y0_n,   // y-coordinates of the vertices
                              y1_n,   //
                              y2_n,   //
                              y3_n,   //
                              z0_n,   // z-coordinates of the vertices
                              z1_n,   //
                              z2_n,   //
                              z3_n,   //
                              px1,    // Vertex of the sub-tetrahedron
                              py1,    //
                              pz1,    //
                              &fx1,   // Output coordinates
                              &fy1,   //
                              &fz1);  //

            tet4_transform_v2(x0_n,   // x-coordinates of the vertices
                              x1_n,   //
                              x2_n,   //
                              x3_n,   //
                              y0_n,   // y-coordinates of the vertices
                              y1_n,   //
                              y2_n,   //
                              y3_n,   //
                              z0_n,   // z-coordinates of the vertices
                              z1_n,   //
                              z2_n,   //
                              z3_n,   //
                              px2,    // Vertex of the sub-tetrahedron
                              py2,    //
                              pz2,    //
                              &fx2,   // Output coordinates
                              &fy2,   //
                              &fz2);  //

            tet4_transform_v2(x0_n,   // x-coordinates of the vertices
                              x1_n,   //
                              x2_n,   //
                              x3_n,   //
                              y0_n,   // y-coordinates of the vertices
                              y1_n,   //
                              y2_n,   //
                              y3_n,   //
                              z0_n,   // z-coordinates of the vertices
                              z1_n,   //
                              z2_n,   //
                              z3_n,   //
                              px3,    // Vertex of the sub-tetrahedron
                              py3,    //
                              pz3,    //
                              &fx3,   // Output coordinates
                              &fy3,   //
                              &fz3);  //

            real_t f0_out, f1_out, f2_out, f3_out;

            const real_t wf0 =
                    tet4_eval_dual_basis_weighted_physical(fx0,  // Coordinates of the vertex where the value is calculated.
                                                           fy0,
                                                           fz0,
                                                           x0_n,  // Coordinates of the first tet vertex in the physical space
                                                           y0_n,
                                                           z0_n,
                                                           x1_n,  // Coordinates of the second tet vertex in physical space
                                                           y1_n,
                                                           z1_n,
                                                           x2_n,  // Coordinates of the third tet vertex in physical space
                                                           y2_n,
                                                           z2_n,
                                                           x3_n,  // Coordinates of the fourth tet vertex in physical space
                                                           y3_n,
                                                           z3_n,
                                                           weighted_field[ev[0]],  // Weighted field at the first vertex
                                                           weighted_field[ev[1]],  // Weighted field at the second vertex
                                                           weighted_field[ev[2]],  // Weighted field at the third vertex
                                                           weighted_field[ev[3]],  // Weighted field at the fourth vertex
                                                           &f0_out,
                                                           &f1_out,
                                                           &f2_out,
                                                           &f3_out);  //

            const real_t wf1 =
                    tet4_eval_dual_basis_weighted_physical(fx1,
                                                           fy1,
                                                           fz1,   // Coordinates of the first vertex
                                                           x0_n,  // Coordinates of the first tet vertex in the physical space
                                                           y0_n,
                                                           z0_n,
                                                           x1_n,  // Coordinates of the second tet vertex in physical space
                                                           y1_n,
                                                           z1_n,
                                                           x2_n,  // Coordinates of the third tet vertex in physical space
                                                           y2_n,
                                                           z2_n,
                                                           x3_n,  // Coordinates of the fourth tet vertex in physical space
                                                           y3_n,
                                                           z3_n,
                                                           weighted_field[ev[0]],  // Weighted field at the first vertex
                                                           weighted_field[ev[1]],  // Weighted field at the second vertex
                                                           weighted_field[ev[2]],  // Weighted field at the third vertex
                                                           weighted_field[ev[3]],  // Weighted field at the fourth vertex
                                                           &f0_out,
                                                           &f1_out,
                                                           &f2_out,
                                                           &f3_out);  //

            const real_t wf2 =
                    tet4_eval_dual_basis_weighted_physical(fx2,
                                                           fy2,
                                                           fz2,   // Coordinates of the first vertex
                                                           x0_n,  // Coordinates of the first tet vertex in the physical space
                                                           y0_n,
                                                           z0_n,
                                                           x1_n,  // Coordinates of the second tet vertex in physical space
                                                           y1_n,
                                                           z1_n,
                                                           x2_n,  // Coordinates of the third tet vertex in physical space
                                                           y2_n,
                                                           z2_n,
                                                           x3_n,  // Coordinates of the fourth tet vertex in physical space
                                                           y3_n,
                                                           z3_n,
                                                           weighted_field[ev[0]],  // Weighted field at the first vertex
                                                           weighted_field[ev[1]],  // Weighted field at the second vertex
                                                           weighted_field[ev[2]],  // Weighted field at the third vertex
                                                           weighted_field[ev[3]],  // Weighted field at the fourth vertex
                                                           &f0_out,
                                                           &f1_out,
                                                           &f2_out,
                                                           &f3_out);  //

            const real_t wf3 =
                    tet4_eval_dual_basis_weighted_physical(fx3,
                                                           fy3,
                                                           fz3,   // Coordinates of the first vertex
                                                           x0_n,  // Coordinates of the first tet vertex in the physical space
                                                           y0_n,
                                                           z0_n,
                                                           x1_n,  // Coordinates of the second tet vertex in physical space
                                                           y1_n,
                                                           z1_n,
                                                           x2_n,  // Coordinates of the third tet vertex in physical space
                                                           y2_n,
                                                           z2_n,
                                                           x3_n,  // Coordinates of the fourth tet vertex in physical space
                                                           y3_n,
                                                           z3_n,
                                                           weighted_field[ev[0]],  // Weighted field at the first vertex
                                                           weighted_field[ev[1]],  // Weighted field at the second vertex
                                                           weighted_field[ev[2]],  // Weighted field at the third vertex
                                                           weighted_field[ev[3]],  // Weighted field at the fourth vertex
                                                           &f0_out,
                                                           &f1_out,
                                                           &f2_out,
                                                           &f3_out);  //

            // printf("Weighted field values: %g, %g, %g, %g\n", wf0, wf1, wf2, wf3);

            // real_t wf1 = tet4_eval_dual_basis_weighted_physical

            int          error_flag   = 0;                             //
            const real_t det_jacobian =                                //
                    calculate_det_Jacobian_for_category(category,      //
                                                        x0_n,          //
                                                        y0_n,          //
                                                        z0_n,          //
                                                        x1_n,          //
                                                        y1_n,          //
                                                        z1_n,          //
                                                        x2_n,          //
                                                        y2_n,          //
                                                        z2_n,          //
                                                        x3_n,          //
                                                        y3_n,          //
                                                        z3_n,          //
                                                        L,             //
                                                        tet_i,         //
                                                        &error_flag);  //

            theta_volume_acc += det_jacobian * (1.0 / 6.0);  // DEBUG code

            // // // Calculate the volume of the HyTeg tetrahedron
            // // // In the physical space
            // const  real_t  theta_volume = tet4_measure_v2(fx0,
            //                                                fx1,
            //                                                fx2,
            //                                                fx3,
            //                                                //
            //                                                fy0,
            //                                                fy1,
            //                                                fy2,
            //                                                fy3,
            //                                                //
            //                                                fz0,
            //                                                fz1,
            //                                                fz2,
            //                                                fz3);

            // printf("det_jacobian = %g, theta_volume = %g, det_jacobian / theta_volume = %g\n",
            //        det_jacobian,
            //        theta_volume,
            //        (det_jacobian / theta_volume));

            // printf("Weighted field values: %g, %g, %g, %g, det_jacobian = %g, theta_volume = %g\n",
            //        wf0,
            //        wf1,
            //        wf2,
            //        wf3,
            //        det_jacobian,
            //        theta_volume);

            tet4_resample_tetrahedron_local_adjoint(   //
                    fx0,                               // Tetrahedron vertices X-coordinates
                    fx1,                               //
                    fx2,                               //
                    fx3,                               //
                    fy0,                               // Tetrahedron vertices Y-coordinates
                    fy1,                               //
                    fy2,                               //
                    fy3,                               //
                    fz0,                               // Tetrahedron vertices Z-coordinates
                    fz1,                               //
                    fz2,                               //
                    fz3,                               //
                    fabs(det_jacobian) * (1.0 / 6.0),  // Determinant of the Jacobian (or volume of the tetrahedron)
                    wf0,                               // Weighted field at the vertices
                    wf1,                               //
                    wf2,                               //
                    wf3,                               //
                    ox,                                // Origin of the grid
                    oy,                                //
                    oz,                                //
                    dx,                                // Spacing of the grid
                    dy,                                //
                    dz,                                //
                    stride,                            // Stride
                    n,                                 // Size of the grid
                    data);                             // Output

        }  // END: for (int tet_i = 0; tet_i < hteg_num_tetrahedra; tet_i++)

        // printf("Theta volume for tetrahedron %ld: %g, theta_volume_acc = %g, volume ratio = %.12e\n",
        //        element_i,
        //        theta_volume_main,
        //        theta_volume_acc,
        //        (theta_volume_acc / theta_volume_main));
    }

    RETURN_FROM_FUNCTION(ret);
}  // END OF FUNCTION tet4_resample_field_local_refine_adjoint_hyteg
//////////////////////////////////////////////////////////
