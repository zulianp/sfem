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
#include "hyteg_Jacobian_matrix_real_t.h"

#include "quadratures_rule.h"

#define real_type real_t
#define SFEM_RESTRICT __restrict__

#define SFEM_RESAMPLE_GAP_DUAL

/**
 * @brief Compute tetrahedral dual basis functions and weighted values for all vertices of a tetrahedron
 *
 * This function computes the DUAL basis functions for tetrahedral elements at all 4 vertices
 * of a sub-tetrahedron in physical space coordinates, and returns their weighted values.
 *
 * @param[in] fx0,fy0,fz0  Coordinates of the first sub-tetrahedron vertex
 * @param[in] fx1,fy1,fz1  Coordinates of the second sub-tetrahedron vertex
 * @param[in] fx2,fy2,fz2  Coordinates of the third sub-tetrahedron vertex
 * @param[in] fx3,fy3,fz3  Coordinates of the fourth sub-tetrahedron vertex
 * @param[in] x0,y0,z0     Coordinates of the first original tetrahedron vertex
 * @param[in] x1,y1,z1     Coordinates of the second original tetrahedron vertex
 * @param[in] x2,y2,z2     Coordinates of the third original tetrahedron vertex
 * @param[in] x3,y3,z3     Coordinates of the fourth original tetrahedron vertex
 * @param[in] wfield0      Weight for the first original tetrahedron vertex
 * @param[in] wfield1      Weight for the second original tetrahedron vertex
 * @param[in] wfield2      Weight for the third original tetrahedron vertex
 * @param[in] wfield3      Weight for the fourth original tetrahedron vertex
 * @param[out] wf0         Weighted value at first sub-tetrahedron vertex
 * @param[out] wf1         Weighted value at second sub-tetrahedron vertex
 * @param[out] wf2         Weighted value at third sub-tetrahedron vertex
 * @param[out] wf3         Weighted value at fourth sub-tetrahedron vertex
 *
 * @return 0 on success, -1 if tetrahedron is degenerate
 */
int tet4_eval_all_vertices_dual_basis_weighted_physical(
        // Sub-tetrahedron vertices in physical space
        const real_t fx0, const real_t fy0, const real_t fz0,  // Sub-tet vertex 0
        const real_t fx1, const real_t fy1, const real_t fz1,  // Sub-tet vertex 1
        const real_t fx2, const real_t fy2, const real_t fz2,  // Sub-tet vertex 2
        const real_t fx3, const real_t fy3, const real_t fz3,  // Sub-tet vertex 3

        // Original tetrahedron vertices
        const real_t x0, const real_t y0, const real_t z0,  // Original vertex 0
        const real_t x1, const real_t y1, const real_t z1,  // Original vertex 1
        const real_t x2, const real_t y2, const real_t z2,  // Original vertex 2
        const real_t x3, const real_t y3, const real_t z3,  // Original vertex 3

        // Weights for original vertices
        const real_t wfield0, const real_t wfield1, const real_t wfield2, const real_t wfield3,

        // Output weighted values for sub-tetrahedron vertices
        real_t* wf0, real_t* wf1, real_t* wf2, real_t* wf3) {
    //
    // Precompute transformation matrix components
    const real_t X00 = x1 - x0, X01 = x2 - x0, X02 = x3 - x0;
    const real_t X10 = y1 - y0, X11 = y2 - y0, X12 = y3 - y0;
    const real_t X20 = z1 - z0, X21 = z2 - z0, X22 = z3 - z0;

    // Compute determinant once
    const real_t det = X00 * (X11 * X22 - X12 * X21) - X01 * (X10 * X22 - X12 * X20) + X02 * (X10 * X21 - X11 * X20);

    if (fabs(det) < 1e-10) {
        // Degenerate tetrahedron
        *wf0 = *wf1 = *wf2 = *wf3 = 0.0;
        return -1;
    }

    const real_t inv_det = 1.0 / det;

    // Precompute matrix elements for inverse transformation (using adjugate matrix)
    const real_t A00 = (X11 * X22 - X12 * X21) * inv_det;
    const real_t A01 = -(X01 * X22 - X02 * X21) * inv_det;
    const real_t A02 = (X01 * X12 - X02 * X11) * inv_det;
    const real_t A10 = -(X10 * X22 - X12 * X20) * inv_det;
    const real_t A11 = (X00 * X22 - X02 * X20) * inv_det;
    const real_t A12 = -(X00 * X12 - X02 * X10) * inv_det;
    const real_t A20 = (X10 * X21 - X11 * X20) * inv_det;
    const real_t A21 = -(X00 * X21 - X01 * X20) * inv_det;
    const real_t A22 = (X00 * X11 - X01 * X10) * inv_det;

    // Pack vertices into arrays for easier processing
    const real_t  vertices[4][3] = {{fx0, fy0, fz0}, {fx1, fy1, fz1}, {fx2, fy2, fz2}, {fx3, fy3, fz3}};
    real_t* const outputs[4]     = {wf0, wf1, wf2, wf3};

    // Process all 4 vertices
    for (int v = 0; v < 4; v++) {
        // Compute relative coordinates
        const real_t b0 = vertices[v][0] - x0;
        const real_t b1 = vertices[v][1] - y0;
        const real_t b2 = vertices[v][2] - z0;

        // Apply inverse transformation using precomputed matrix
        const real_t qx = A00 * b0 + A01 * b1 + A02 * b2;
        const real_t qy = A10 * b0 + A11 * b1 + A12 * b2;
        const real_t qz = A20 * b0 + A21 * b1 + A22 * b2;

        // Compute standard basis functions
        const real_t f0 = 1.0 - qx - qy - qz;
        // f1 = qx, f2 = qy, f3 = qz (no need to store)

        // Compute dual basis functions directly
        const real_t tet4_f0 = 4.0 * f0 - qx - qy - qz;
        const real_t tet4_f1 = -f0 + 4.0 * qx - qy - qz;
        const real_t tet4_f2 = -f0 - qx + 4.0 * qy - qz;
        const real_t tet4_f3 = -f0 - qx - qy + 4.0 * qz;

        // Compute weighted sum
        *outputs[v] = tet4_f0 * wfield0 + tet4_f1 * wfield1 + tet4_f2 * wfield2 + tet4_f3 * wfield3;
    }

    return 0;
}

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

real_t                                         //
tet4_quad_volume(const real_t x0,              // Tetrahedron vertices X-coordinates
                 const real_t x1,              //
                 const real_t x2,              //
                 const real_t x3,              //
                 const real_t y0,              // Tetrahedron vertices Y-coordinates
                 const real_t y1,              //
                 const real_t y2,              //
                 const real_t y3,              //
                 const real_t z0,              // Tetrahedron vertices Z-coordinates
                 const real_t z1,              //
                 const real_t z2,              //
                 const real_t z3,              //
                 const real_t det_jacobian) {  // Tetrahedron vertices Z-coordinates

    real_t volume = 0.0;

    for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i++) {  // loop over the quadrature points

        real_t g_qx, g_qy, g_qz;

        volume += tet_qw[quad_i] * det_jacobian * (1.0 / 6.0);
    }

    return volume;  // Return the total volume of the tetrahedron
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
                                          k,
                                          &i0,   // Output indices
                                          &i1,   //
                                          &i2,   //
                                          &i3,   //
                                          &i4,   //
                                          &i5,   //
                                          &i6,   //
                                          &i7);  //

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
        const int    max_refinement_L    = 3;    // Maximum refinement level

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

        const real_t rescaled_theta_volume_main = theta_volume_main / (real_t)(L * L * L);  // Sub-tetrahedron volume

        real_t theta_volume_acc  = 0.0;  // DEBUG: theta_volume_acc
        real_t theta_quad_volume = 0.0;  // DEBUG: theta_quad_volume

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

            // Optimize by calling the all-vertices function once instead of 4 separate calls
            real_t wf0, wf1, wf2, wf3;
            int    eval_result =
                    tet4_eval_all_vertices_dual_basis_weighted_physical(fx0,
                                                                        fy0,
                                                                        fz0,  // Sub-tet vertex 0
                                                                        fx1,
                                                                        fy1,
                                                                        fz1,  // Sub-tet vertex 1
                                                                        fx2,
                                                                        fy2,
                                                                        fz2,  // Sub-tet vertex 2
                                                                        fx3,
                                                                        fy3,
                                                                        fz3,  // Sub-tet vertex 3
                                                                        x0_n,
                                                                        y0_n,
                                                                        z0_n,  // Original vertex 0
                                                                        x1_n,
                                                                        y1_n,
                                                                        z1_n,  // Original vertex 1
                                                                        x2_n,
                                                                        y2_n,
                                                                        z2_n,  // Original vertex 2
                                                                        x3_n,
                                                                        y3_n,
                                                                        z3_n,                   // Original vertex 3
                                                                        weighted_field[ev[0]],  // Weighted field at vertex 0
                                                                        weighted_field[ev[1]],  // Weighted field at vertex 1
                                                                        weighted_field[ev[2]],  // Weighted field at vertex 2
                                                                        weighted_field[ev[3]],  // Weighted field at vertex 3
                                                                        &wf0,
                                                                        &wf1,
                                                                        &wf2,
                                                                        &wf3);

            if (eval_result != 0) {
                // Handle degenerate tetrahedron case
                continue;  // or appropriate error handling
            }

            // const real_t wf0 =
            //         tet4_eval_dual_basis_weighted_physical(fx0,  // Coordinates of the vertex where the value is calculated.
            //                                                fy0,
            //                                                fz0,
            //                                                x0_n,  // Coordinates of the first tet vertex in the physical space
            //                                                y0_n,
            //                                                z0_n,
            //                                                x1_n,  // Coordinates of the second tet vertex in physical space
            //                                                y1_n,
            //                                                z1_n,
            //                                                x2_n,  // Coordinates of the third tet vertex in physical space
            //                                                y2_n,
            //                                                z2_n,
            //                                                x3_n,  // Coordinates of the fourth tet vertex in physical space
            //                                                y3_n,
            //                                                z3_n,
            //                                                weighted_field[ev[0]],  // Weighted field at the first vertex
            //                                                weighted_field[ev[1]],  // Weighted field at the second vertex
            //                                                weighted_field[ev[2]],  // Weighted field at the third vertex
            //                                                weighted_field[ev[3]],  // Weighted field at the fourth vertex
            //                                                &f0_out,
            //                                                &f1_out,
            //                                                &f2_out,
            //                                                &f3_out);  //

            // const real_t wf1 =
            //         tet4_eval_dual_basis_weighted_physical(fx1,
            //                                                fy1,
            //                                                fz1,   // Coordinates of the first vertex
            //                                                x0_n,  // Coordinates of the first tet vertex in the physical space
            //                                                y0_n,
            //                                                z0_n,
            //                                                x1_n,  // Coordinates of the second tet vertex in physical space
            //                                                y1_n,
            //                                                z1_n,
            //                                                x2_n,  // Coordinates of the third tet vertex in physical space
            //                                                y2_n,
            //                                                z2_n,
            //                                                x3_n,  // Coordinates of the fourth tet vertex in physical space
            //                                                y3_n,
            //                                                z3_n,
            //                                                weighted_field[ev[0]],  // Weighted field at the first vertex
            //                                                weighted_field[ev[1]],  // Weighted field at the second vertex
            //                                                weighted_field[ev[2]],  // Weighted field at the third vertex
            //                                                weighted_field[ev[3]],  // Weighted field at the fourth vertex
            //                                                &f0_out,
            //                                                &f1_out,
            //                                                &f2_out,
            //                                                &f3_out);  //

            // const real_t wf2 =
            //         tet4_eval_dual_basis_weighted_physical(fx2,
            //                                                fy2,
            //                                                fz2,   // Coordinates of the first vertex
            //                                                x0_n,  // Coordinates of the first tet vertex in the physical space
            //                                                y0_n,
            //                                                z0_n,
            //                                                x1_n,  // Coordinates of the second tet vertex in physical space
            //                                                y1_n,
            //                                                z1_n,
            //                                                x2_n,  // Coordinates of the third tet vertex in physical space
            //                                                y2_n,
            //                                                z2_n,
            //                                                x3_n,  // Coordinates of the fourth tet vertex in physical space
            //                                                y3_n,
            //                                                z3_n,
            //                                                weighted_field[ev[0]],  // Weighted field at the first vertex
            //                                                weighted_field[ev[1]],  // Weighted field at the second vertex
            //                                                weighted_field[ev[2]],  // Weighted field at the third vertex
            //                                                weighted_field[ev[3]],  // Weighted field at the fourth vertex
            //                                                &f0_out,
            //                                                &f1_out,
            //                                                &f2_out,
            //                                                &f3_out);  //

            // const real_t wf3 =
            //         tet4_eval_dual_basis_weighted_physical(fx3,
            //                                                fy3,
            //                                                fz3,   // Coordinates of the first vertex
            //                                                x0_n,  // Coordinates of the first tet vertex in the physical space
            //                                                y0_n,
            //                                                z0_n,
            //                                                x1_n,  // Coordinates of the second tet vertex in physical space
            //                                                y1_n,
            //                                                z1_n,
            //                                                x2_n,  // Coordinates of the third tet vertex in physical space
            //                                                y2_n,
            //                                                z2_n,
            //                                                x3_n,  // Coordinates of the fourth tet vertex in physical space
            //                                                y3_n,
            //                                                z3_n,
            //                                                weighted_field[ev[0]],  // Weighted field at the first vertex
            //                                                weighted_field[ev[1]],  // Weighted field at the second vertex
            //                                                weighted_field[ev[2]],  // Weighted field at the third vertex
            //                                                weighted_field[ev[3]],  // Weighted field at the fourth vertex
            //                                                &f0_out,
            //                                                &f1_out,
            //                                                &f2_out,
            //                                                &f3_out);  //

            // printf("Weighted field values: %g, %g, %g, %g\n", wf0, wf1, wf2, wf3);

            // real_t wf1 = tet4_eval_dual_basis_weighted_physical

            // int error_flag = 0;  //
            // const real_t det_jacobian = rescaled_theta_volume_main;  //?
            // calculate_det_Jacobian_for_category(category,      //
            // x0_n,          //
            // y0_n,          //
            // z0_n,          //
            // x1_n,          //
            // y1_n,          //
            // z1_n,          //
            // x2_n,          //
            // y2_n,          //
            // z2_n,          //
            // x3_n,          //
            // y3_n,          //
            // z3_n,          //
            // L,             //
            // tet_i,         //
            // &error_flag);  //

            // theta_volume_acc += det_jacobian * (1.0 / 6.0);  // DEBUG code

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

            tet4_resample_tetrahedron_local_adjoint(  //
                    fx0,                              // Tetrahedron vertices X-coordinates
                    fx1,                              //
                    fx2,                              //
                    fx3,                              //
                    fy0,                              // Tetrahedron vertices Y-coordinates
                    fy1,                              //
                    fy2,                              //
                    fy3,                              //
                    fz0,                              // Tetrahedron vertices Z-coordinates
                    fz1,                              //
                    fz2,                              //
                    fz3,                              //
                    rescaled_theta_volume_main,       // Determinant of the Jacobian (or volume of the tetrahedron)
                    wf0,                              // Weighted field at the vertices
                    wf1,                              //
                    wf2,                              //
                    wf3,                              //
                    ox,                               // Origin of the grid
                    oy,                               //
                    oz,                               //
                    dx,                               // Spacing of the grid
                    dy,                               //
                    dz,                               //
                    stride,                           // Stride
                    n,                                // Size of the grid
                    data);                            // Output

            /// DEBUG code: CHECK: Calculate the volume of the HyTeg tetrahedron using quadrature
            theta_quad_volume += tet4_quad_volume(fx0,                                  // X-coordinates of the vertices
                                                  fx1,                                  //
                                                  fx2,                                  //
                                                  fx3,                                  //
                                                  fy0,                                  // Y-coordinates of the vertices
                                                  fy1,                                  //
                                                  fy2,                                  //
                                                  fy3,                                  //
                                                  fz0,                                  // Z-coordinates of the vertices
                                                  fz1,                                  //
                                                  fz2,                                  //
                                                  fz3,                                  //
                                                  (rescaled_theta_volume_main * 6.0));  //

        }  // END: for (int tet_i = 0; tet_i < hteg_num_tetrahedra; tet_i++)

        // printf("Theta volume for tetrahedron %ld: %g, theta_volume_acc = %g, volume ratio = %.12e, theta_quad_volume: %1.12e, "
        //        "ratio %1.12f\n",                          //
        //        element_i,                                 //
        //        theta_volume_main,                         //
        //        theta_volume_acc,                          //
        //        (theta_volume_acc / theta_volume_main),    //
        //        theta_quad_volume,                         //
        //        (theta_quad_volume / theta_volume_main));  //
    }

    RETURN_FROM_FUNCTION(ret);
}  // END OF FUNCTION tet4_resample_field_local_refine_adjoint_hyteg

// real_t                                               //
// test_micro_tet_quad(const unsigned int  category,    //
//                     const unsigned int  L,           // Refinement level
//                     const real_t const* bc,          // transposition vector for category
//                     const real_t        J_phys[9],   // Jacobian matrix
//                     const real_t        J_ref[9],    // Jacobian matrix
//                     const real_t        del_J_phys,  // Determinant of the Jacobian matrix
//                     const real_t        fx0,         // Tetrahedron vertices X-coordinates
//                     const real_t        fy0,         // Tetrahedron vertices Y-coordinates
//                     const real_t        fz0) {              // Tetrahedron vertices Z-coordinates)

//     const real_t N_micro_tet     = pow(L, 3);          // Number of micro-tetrahedra in the HyTeg tetrahedron
//     const real_t inv_N_micro_tet = 1.0 / N_micro_tet;  // Inverse of the number of micro-tetrahedra

//     const real_t theta_volume = del_J_phys / ((real_t)(6.0));  // Volume of the mini-tetrahedron in the physical space

//     for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i++) {  // loop over the quadrature points

//         // Mapping the quadrature point from the reference space to the mini-tetrahedron local coordinate system
//         const real_t xq_mref = J_ref[0] * tet_qx[quad_i] + J_ref[1] * tet_qy[quad_i] + J_ref[2] * tet_qz[quad_i] + bc[0];
//         const real_t yq_mref = J_ref[3] * tet_qx[quad_i] + J_ref[4] * tet_qy[quad_i] + J_ref[5] * tet_qz[quad_i] + bc[1];
//         const real_t zq_mref = J_ref[6] * tet_qx[quad_i] + J_ref[7] * tet_qy[quad_i] + J_ref[8] * tet_qz[quad_i] + bc[2];

//         const real_t xq_phys = J_phys[0] * xq_mref + J_phys[1] * yq_mref + J_phys[2] * zq_mref + fx0;  // Physical X-coordinate
//         const real_t yq_phys = J_phys[3] * xq_mref + J_phys[4] * yq_mref + J_phys[5] * zq_mref + fy0;  // Physical Y-coordinate
//         const real_t zq_phys = J_phys[6] * xq_mref + J_phys[7] * yq_mref + J_phys[8] * zq_mref + fz0;  // Physical Z-coordinate
//     }
// }

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_tetrahedron_local_adjoint_category ////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
real_t                                                                               //
tet4_resample_tetrahedron_local_adjoint_category(const unsigned int     category,    //
                                                 const unsigned int     L,           // Refinement level
                                                 const real_t const*    bc,          // transposition vector for category
                                                 const real_t           J_phys[9],   // Jacobian matrix
                                                 const real_t           J_ref[9],    // Jacobian matrix
                                                 const real_t           del_J_phys,  // Determinant of the Jacobian matrix
                                                 const real_t           fx0,         // Tetrahedron vertices X-coordinates
                                                 const real_t           fy0,         // Tetrahedron vertices Y-coordinates
                                                 const real_t           fz0,         // Tetrahedron vertices Z-coordinates
                                                 const real_t           wf0,         // Weighted field at the vertices
                                                 const real_t           wf1,         //
                                                 const real_t           wf2,         //
                                                 const real_t           wf3,         //
                                                 const geom_t           ox,          // Origin of the grid
                                                 const geom_t           oy,          //
                                                 const geom_t           oz,          //
                                                 const geom_t           dx,          // Spacing of the grid
                                                 const geom_t           dy,          //
                                                 const geom_t           dz,          //
                                                 const ptrdiff_t* const stride,      // Stride
                                                 const ptrdiff_t* const n,           // Size of the grid
                                                 real_t* const          data) {               // Output

    // Jacobian matrix for the tetrahedron

    const real_t N_micro_tet     = pow(L, 3);          // Number of micro-tetrahedra in the HyTeg tetrahedron
    const real_t inv_N_micro_tet = 1.0 / N_micro_tet;  // Inverse of the number of micro-tetrahedra

    const real_t theta_volume = del_J_phys / ((real_t)(6.0));  // Volume of the mini-tetrahedron in the physical space

    real_t cumulated_dV = 0.0;  // Cumulative volume for debugging

    // printf bc
    // printf("Category %d: transposition vector bc = [%e, %e, %e]\n", category, bc[0], bc[1], bc[2]);
    // printf("Tet vertices X-coordinates: \nfx0 = %e, fy0 = %e, fz0 = %e\n", fx0, fy0, fz0);

    for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i++) {  // loop over the quadrature points
        // Mapping the quadrature point from the reference space to the mini-tetrahedron
        const real_t xq_mref = J_ref[0] * tet_qx[quad_i] + J_ref[1] * tet_qy[quad_i] + J_ref[2] * tet_qz[quad_i] + bc[0];
        const real_t yq_mref = J_ref[3] * tet_qx[quad_i] + J_ref[4] * tet_qy[quad_i] + J_ref[5] * tet_qz[quad_i] + bc[1];
        const real_t zq_mref = J_ref[6] * tet_qx[quad_i] + J_ref[7] * tet_qy[quad_i] + J_ref[8] * tet_qz[quad_i] + bc[2];

        // printf("Quadrature point in the reference space: xq_mref = %e, yq_mref = %e, zq_mref = %e\n", xq_mref, yq_mref,
        // zq_mref);

        const real_t xq_phys = J_phys[0] * xq_mref + J_phys[1] * yq_mref + J_phys[2] * zq_mref + fx0;  // Physical X-coordinate
        const real_t yq_phys = J_phys[3] * xq_mref + J_phys[4] * yq_mref + J_phys[5] * zq_mref + fy0;  // Physical Y-coordinate
        const real_t zq_phys = J_phys[6] * xq_mref + J_phys[7] * yq_mref + J_phys[8] * zq_mref + fz0;  // Physical Z-coordinate

        // printf("%.12e, %.12e, %.12e\n", xq_phys, yq_phys, zq_phys);

        const real_t grid_x = (xq_phys - ox) / dx;
        const real_t grid_y = (yq_phys - oy) / dy;
        const real_t grid_z = (zq_phys - oz) / dz;

        const ptrdiff_t i = floor(grid_x);
        const ptrdiff_t j = floor(grid_y);
        const ptrdiff_t k = floor(grid_z);

        const real_t l_x = (grid_x - (real_t)i);
        const real_t l_y = (grid_y - (real_t)j);
        const real_t l_z = (grid_z - (real_t)k);

        assert(l_x >= -1e-8);
        assert(l_y >= -1e-8);
        assert(l_z >= -1e-8);

        assert(l_x <= 1 + 1e-8);
        assert(l_y <= 1 + 1e-8);
        assert(l_z <= 1 + 1e-8);

        // Move the quadrature point to the micro-tetrahedron local coordinate system

        // DUAL basis function (Shape functions for tetrahedral elements)
        // at the quadrature point in the coordinate system of the micro-tetrahedron in
        // the reference space
        const real_t f0 = 1.0 - xq_mref - yq_mref - zq_mref;
        const real_t f1 = xq_mref;
        const real_t f2 = yq_mref;
        const real_t f3 = zq_mref;

        // Values of the shape functions at the quadrature point
        // In the local coordinate system of the tetrahedral element
        // For each vertex of the tetrahedral element
        const real_t tet4_f0 = 4.0 * f0 - f1 - f2 - f3;
        const real_t tet4_f1 = -f0 + 4.0 * f1 - f2 - f3;
        const real_t tet4_f2 = -f0 - f1 + 4.0 * f2 - f3;
        const real_t tet4_f3 = -f0 - f1 - f2 + 4.0 * f3;

        // Compute the weighted field value at the quadrature point
        const real_t    wf_quad = tet4_f0 * wf0 + tet4_f1 * wf1 + tet4_f2 * wf2 + tet4_f3 * wf3;
        const real_type dV      = theta_volume * inv_N_micro_tet * tet_qw[quad_i];
        const real_type It      = wf_quad * dV;

        cumulated_dV += dV;  // Cumulative volume for debugging

        // if (wf_quad < 0.6) printf("wf_quad = %g, dV = %g, It = %g\n", wf_quad, dV, It);

        real_type hex8_f0, hex8_f1, hex8_f2, hex8_f3, hex8_f4, hex8_f5, hex8_f6, hex8_f7;
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

        const real_type d0 = It * hex8_f0;
        const real_type d1 = It * hex8_f1;
        const real_type d2 = It * hex8_f2;
        const real_type d3 = It * hex8_f3;
        const real_type d4 = It * hex8_f4;
        const real_type d5 = It * hex8_f5;
        const real_type d6 = It * hex8_f6;
        const real_type d7 = It * hex8_f7;

        // Update the data
        data[i0] += d0;
        data[i1] += d1;
        data[i2] += d2;
        data[i3] += d3;
        data[i4] += d4;
        data[i5] += d5;
        data[i6] += d6;
        data[i7] += d7;

    }  // END: for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i++)

    return cumulated_dV;  // Return the cumulative volume for debugging
}  // END OF FUNCTION tet4_resample_tetrahedron_local_adjoint_category

/**
 * @brief Calculate the determinant of the Jacobian matrix.
 *
 * @param J 3x3 Jacobian matrix
 * @return real_t
 */
real_t                      //
det_J(const real_t J[9]) {  // Calculate the determinant of the Jacobian matrix
    // J = [x1-x0, x2-x0, x3-x0]   <- Row 0: indices 0,1,2
    //     [y1-y0, y2-y0, y3-y0]   <- Row 1: indices 3,4,5
    //     [z1-z0, z2-z0, z3-z0]   <- Row 2: indices 6,7,8

    return J[0] * (J[4] * J[8] - J[5] * J[7]) - J[1] * (J[3] * J[8] - J[5] * J[6]) + J[2] * (J[3] * J[7] - J[4] * J[6]);
}

/**
 * @brief Calculate the Jacobian matrix for a tetrahedron.
 *
 * @param fx0
 * @param fx1
 * @param fx2
 * @param fx3
 * @param fy0
 * @param fy1
 * @param fy2
 * @param fy3
 * @param fz0
 * @param fz1
 * @param fz2
 * @param fz3
 * @param J  Output Jacobian matrix in the row-major order
 * @return real_t
 */
real_t                                      //
make_Jocobian_matrix_tet(const real_t fx0,  // Tetrahedron vertices X-coordinates
                         const real_t fx1,  //
                         const real_t fx2,  //
                         const real_t fx3,  //
                         const real_t fy0,  // Tetrahedron vertices Y-coordinates
                         const real_t fy1,  //
                         const real_t fy2,  //
                         const real_t fy3,  //
                         const real_t fz0,  // Tetrahedron vertices Z-coordinates
                         const real_t fz1,  //
                         const real_t fz2,  //
                         const real_t fz3,
                         real_t       J[9]) {  // Jacobian matrix
    // Compute the Jacobian matrix for tetrahedron transformation
    // J = [x1-x0, x2-x0, x3-x0]   <- Row 0: indices 0,1,2
    //     [y1-y0, y2-y0, y3-y0]   <- Row 1: indices 3,4,5
    //     [z1-z0, z2-z0, z3-z0]   <- Row 2: indices 6,7,8

    // Row 0: x-components (indices 0,1,2)
    J[0] = fx1 - fx0;  // dx/dxi
    J[1] = fx2 - fx0;  // dx/deta
    J[2] = fx3 - fx0;  // dx/dzeta

    // Row 1: y-components (indices 3,4,5)
    J[3] = fy1 - fy0;  // dy/dxi
    J[4] = fy2 - fy0;  // dy/deta
    J[5] = fy3 - fy0;  // dy/dzeta

    // Row 2: z-components (indices 6,7,8)
    J[6] = fz1 - fz0;  // dz/dxi
    J[7] = fz2 - fz0;  // dz/deta
    J[8] = fz3 - fz0;  // dz/dzeta

    // Compute determinant of the 3x3 Jacobian matrix
    const real_t det =
            J[0] * (J[4] * J[8] - J[5] * J[7]) - J[1] * (J[3] * J[8] - J[5] * J[6]) + J[2] * (J[3] * J[7] - J[4] * J[6]);

    return det;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_refine_adjoint_hyteg ////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                                                    //
tet4_resample_field_local_refine_adjoint_hyteg_d(const ptrdiff_t                      start_element,   // Mesh
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

#define HYTEG_D_LOG_ENABLED 0

#if HYTEG_D_LOG_ENABLED
#define HYTEG_D_LOG(...) printf(__VA_ARGS__)
#else
#define HYTEG_D_LOG(...) (void)0
#endif

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

    // Unit tetrahedron vertices
    const real_t x0_unit = 0.0;
    const real_t x1_unit = 1.0;
    const real_t x2_unit = 0.0;
    const real_t x3_unit = 0.0;

    const real_t y0_unit = 0.0;
    const real_t y1_unit = 0.0;
    const real_t y2_unit = 1.0;
    const real_t y3_unit = 0.0;

    const real_t z0_unit = 0.0;
    const real_t z1_unit = 0.0;
    const real_t z2_unit = 0.0;
    const real_t z3_unit = 1.0;

    real_t J_vec_mini[6][9];  // Jacobian matrices for the 6 categories of tetrahedra for the refined and reference element
    real_t J_phy[9];          // Jacobian matrices for the 6 categories of tetrahedra for the physical current

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

        const real_t wf0 = weighted_field[ev[0]];  // Weighted field at vertex 0
        const real_t wf1 = weighted_field[ev[1]];  // Weighted field at vertex 1
        const real_t wf2 = weighted_field[ev[2]];  // Weighted field at vertex 2
        const real_t wf3 = weighted_field[ev[3]];  // Weighted field at vertex 3

        real_t det_J_phys =                     //
                fabs(make_Jocobian_matrix_tet(  //
                        x0_n,                   // Tetrahedron vertices X-coordinates
                        x1_n,                   //
                        x2_n,                   //
                        x3_n,                   //
                        y0_n,                   // Tetrahedron vertices Y-coordinates
                        y1_n,                   //
                        y2_n,                   //
                        y3_n,                   //
                        z0_n,                   // Tetrahedron vertices Z-coordinates
                        z1_n,                   //
                        z2_n,                   //
                        z3_n,                   // Vertex 3 coordinates
                        J_phy));                // Output Jacobian matrix

        real_t tet_volume = tet4_measure_v2(x0_n,  // Coordinates of the vertices
                                            x1_n,  //
                                            x2_n,  //
                                            x3_n,  //
                                            y0_n,  // Coordinates of the vertices
                                            y1_n,  //
                                            y2_n,  //
                                            y3_n,  //
                                            z0_n,  // Coordinates of the vertices
                                            z1_n,  //
                                            z2_n,  //
                                            z3_n);

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

        const real_t alpha_min_threshold = 1.7;  // Minimum threshold for alpha, Less: make less refinements.
        const real_t alpha_max_threshold = 8.0;  // Maximum threshold for alpha. Less: make more refinements.
        const int    max_refinement_L    = 2;    // Maximum refinement level

        const int L = alpha_to_hyteg_level(alpha_tet,            // // DEBUG forced to 2 refinements
                                           alpha_min_threshold,  //
                                           alpha_max_threshold,  //
                                           max_refinement_L);    //

        real_t cumulated_volume = 0.0;  // Cumulative volume for debugging

        // Calculate the Jacobian matrices for the 6 categories of tetrahedra for the reference element
        for (int cat_i = 0; cat_i < 6; cat_i++) {
            // Calculate the Jacobian matrix for the current category
            jacobian_matrix_real_t(cat_i,               // Category of the tetrahedron
                                   x0_unit,             //
                                   y0_unit,             //
                                   z0_unit,             //
                                   x1_unit,             //
                                   y1_unit,             //
                                   z1_unit,             //
                                   x2_unit,             //
                                   y2_unit,             //
                                   z2_unit,             //
                                   x3_unit,             //
                                   y3_unit,             //
                                   z3_unit,             // Vertex 3 coordinates
                                   (real_t)L,           // Refinement level
                                   J_vec_mini[cat_i]);  // Output Jacobian matrix

        }  // END of for (int cat_i = 0; cat_i < 6; cat_i++)

        const real_t h = 1.0 / (real_t)L;  // Size of the sub-tetrahedron

        HYTEG_D_LOG("Processing element %ld, alpha_tet = %g, L = %d, max_edges_length = %g, d_min = %g, tet_volume = %g\n",  //
                    element_i,
                    alpha_tet,
                    L,
                    max_edges_length,
                    d_min,
                    tet_volume);

        real_t theta_volume_main = 0.0;  // Volume of the HyTeg tetrahedron

        for (int k = 0; k < L + 1; k++) {  // Loop over the refinement levels

            const unsigned int nodes_pes_side = (L - k) + 1;
            // const unsigned int nodes_per_layer = nodes_pes_side * (nodes_pes_side + 1) / 2;

            for (int i = 0; i < nodes_pes_side - 1; i++) {          // Loop over the nodes on the first edge
                for (int j = 0; j < nodes_pes_side - i - 1; j++) {  // Loop over the nodes on the second edge

                    HYTEG_D_LOG("Processing element %ld, refinement level %d, i = %d, j = %d, k = %d\n",  //
                                element_i,
                                L,
                                i,
                                j,
                                k);

                    {  // BEGIN: Cat 0
                        const unsigned int cat0 = 0;

                        HYTEG_D_LOG("**** Processing Cat 0 for element %ld, refinement level %d, i = %d, j = %d, k = %d\n",  //
                                    element_i,
                                    L,
                                    i,
                                    j,
                                    k);

                        // Coordinates of the node on the first edge for Cat 0
                        const real_t b0[3] = {(real_t)(j)*h,   //
                                              (real_t)(i)*h,   //
                                              (real_t)(k)*h};  //

                        // Solve the case for the current Cat. 0 tetrahedron.
                        // 1. Get the Jacobian matrix for the current Cat. 0 J_0 tetrahedron
                        // 2. Use J_0 to calculate the coordinates of the sub-tetrahedron vertices in the physical space
                        // 3. Calculate the weighted field at the sub-tetrahedron vertices
                        // 4. Resample the field at the sub-tetrahedron vertices

                        // Category 0
                        cumulated_volume +=  //
                                tet4_resample_tetrahedron_local_adjoint_category(
                                        cat0,              //
                                        L,                 //
                                        b0,                // Transposition vector for category 0)
                                        J_phy,             // Jacobian matrix
                                        J_vec_mini[cat0],  // Reference Jacobian matrix
                                        det_J_phys,        // Determinant of the Jacobian matrix for physical tet
                                        x0_n,              // Tetrahedron vertices XYZ-coordinates
                                        y0_n,              // 
                                        z0_n,              // 
                                        wf0,               // Weighted field at the vertices
                                        wf1,               //
                                        wf2,               //
                                        wf3,               //
                                        ox,                // Origin of the grid
                                        oy,                //
                                        oz,                //
                                        dx,                // Spacing of the grid
                                        dy,                //
                                        dz,                //
                                        stride,            // Stride
                                        n,                 //
                                        data);             // Size of the grid

                        // theta_volume_main += det_J_vec_phy[cat0] * (1.0 / 6.0);  // Accumulate the volume of the HyTeg
                        // tetrahedron

                    }  // END: Cat 0

                    if (j >= 1) {
                        const real_t b1[3] = {(real_t)(j)*h,   //
                                              (real_t)(i)*h,   //
                                              (real_t)(k)*h};  //

                        // Solve the case for the current Cat. 1, 2, 3, 4 tetrahedra.

                        for (int cat_i = 1; cat_i <= 4; cat_i++) {
                            HYTEG_D_LOG(
                                    "**** Processing Cat %d for element %ld, refinement level %d, i = %d, j = %d, k = %d\n",  //
                                    cat_i,
                                    element_i,
                                    L,
                                    i,
                                    j,
                                    k);

                            cumulated_volume +=  //
                                    tet4_resample_tetrahedron_local_adjoint_category(
                                            cat_i,              //
                                            L,                  //
                                            b1,                 // Transposition vector for category 1, 2, 3, 4
                                            J_phy,              // Jacobian matrix
                                            J_vec_mini[cat_i],  // Reference Jacobian matrix
                                            det_J_phys,         // Determinant of the Jacobian matrix
                                            x0_n,               // Tetrahedron vertices X-coordinates
                                            y0_n,               // Tetrahedron vertices Y-coordinates
                                            z0_n,               // Tetrahedron vertices Z-coordinates
                                            wf0,                // Weighted field at the vertices
                                            wf1,                //
                                            wf2,                //
                                            wf3,                //
                                            ox,                 // Origin of the grid
                                            oy,                 //
                                            oz,                 //
                                            dx,                 // Spacing of the grid
                                            dy,                 //
                                            dz,                 //
                                            stride,             // Stride
                                            n,                  //
                                            data);              // Size of the grid

                            // printf("J_vec_mini[%d] = \n[%.12e, %.12e, %.12e, \n%.12e, %.12e, %.12e, \n%.12e, %.12e, %.12e]\n",
                            //        cat_i,  //
                            //        J_vec_mini[cat_i][0],
                            //        J_vec_mini[cat_i][1],
                            //        J_vec_mini[cat_i][2],
                            //        J_vec_mini[cat_i][3],
                            //        J_vec_mini[cat_i][4],
                            //        J_vec_mini[cat_i][5],
                            //        J_vec_mini[cat_i][6],
                            //        J_vec_mini[cat_i][7],
                            //        J_vec_mini[cat_i][8]);

                            // theta_volume_main +=
                            //         det_J_vec_phy[cat_i] * (1.0 / 6.0);  // Accumulate the volume of the HyTeg tetrahedron
                        }
                    }  // END: if (j >= 1) for cat 1, 2, 3, 4

                    if (j >= 1 && i >= 1) {
                        const unsigned int cat5 = 5;

                        HYTEG_D_LOG("**** Processing Cat 5 for element %ld, refinement level %d, i = %d, j = %d, k = %d\n",  //
                                    element_i,
                                    L,
                                    i,
                                    j,
                                    k);

                        const real_t b5[3] = {(real_t)(j)*h,   //
                                              (real_t)(i)*h,   //
                                              (real_t)(k)*h};  //

                        cumulated_volume +=                                        //
                                tet4_resample_tetrahedron_local_adjoint_category(  //
                                        cat5,                                      //
                                        L,                                         //
                                        b5,                                        // Transposition vector for category 5
                                        J_phy,                                     // Jacobian matrix
                                        J_vec_mini[cat5],                          // Reference Jacobian matrix
                                        det_J_phys,                                // Determinant of the Jacobian matrix
                                        x0_n,                                      // Tetrahedron vertices X-coordinates
                                        y0_n,                                      //
                                        z0_n,                                      // Tetrahedron vertices Z-coordinates
                                        wf0,                                       // Weighted field at the vertices
                                        wf1,                                       //
                                        wf2,                                       //
                                        wf3,                                       //
                                        ox,                                        // Origin of the grid
                                        oy,                                        //
                                        oz,                                        //
                                        dx,                                        // Spacing of the grid
                                        dy,                                        //
                                        dz,                                        //
                                        stride,                                    // Stride
                                        n,                                         //
                                        data);                                     // Size of the grid
                        // theta_volume_main += det_J_vec_phy[cat5] * (1.0 / 6.0);  // Accumulate the volume of the HyTeg
                        // tetrahedron

                        // Solve the case for the current Cat. 5 tetrahedron.
                    }  // END: if (j >= 1 && i >= 1) for cat 5
                }      // END: for (int j = 0; j < nodes_pes_side - i - 1; j++) // Loop over the nodes on the second edge
            }          // END: for (int i = 0; i < nodes_pes_side - 1; i++) // Loop over the nodes on the first edge
        }              // END: for (int k = 0; k < L + 1; k++) // Loop over the refinement levels

        HYTEG_D_LOG("\n Element %ld: theta_volume_main = %.12e\n", element_i, theta_volume_main);
        HYTEG_D_LOG(" Element %ld: tet_volume =          %.12e\n", element_i, tet_volume);
        HYTEG_D_LOG(" Element %ld: cumulated_volume =    %.12e\n", element_i, cumulated_volume);
        HYTEG_D_LOG(" Element %ld: diff vol   =          %.12e\n", element_i, (theta_volume_main - tet_volume));

#if HYTEG_D_LOG_ENABLED == 1
        if (element_i == 33) exit(EXIT_SUCCESS);  // Exit the program if the refinement is done
#endif

        // Check if the tetrahedron is degenerated
        if (det_J_phys < 1e-8) {
            degenerated_tetrahedra_cnt++;
            HYTEG_D_LOG("Element %ld: Degenerated tetrahedron detected! Det(J) = %g\n", element_i, det_J_phys);
            continue;  // Skip the degenerated tetrahedron
        }

        // Check if the tetrahedron is uniformly refined
        if (L == 2) {
            uniform_refine_cnt++;
            HYTEG_D_LOG("Element %ld: Uniformly refined tetrahedron detected! L = %d\n", element_i, L);
        }

    }  // END: for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++)

    return ret;  // Return the result of the refinement
}
