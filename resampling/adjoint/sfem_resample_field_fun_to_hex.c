#include "sfem_resample_field.h"
#include "sfem_resample_field_adjoint_hyteg.h"
#include "sfem_resample_field_tet4_math.h"
#include "sfem_stack.h"

#include "mass.h"
// #include "read_mesh.h"
#include "matrixio_array.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

// #define real_t  real_t

#include "hyteg.h"
#include "hyteg_Jacobian_matrix_real_t.h"

#include "quadratures_rule.h"

#include <stdio.h>

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// resample_field_adjoint_tet4 /////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// check_point_in_tet /////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
void                                           //
check_point_in_tet(const int           p_cnt,  //
                   const real_t* const px,     //
                   const real_t* const py,     //
                   const real_t* const pz,     //
                   const real_t        x0,     //
                   const real_t        x1,     //
                   const real_t        x2,     //
                   const real_t        x3,     //
                   const real_t        y0,     //
                   const real_t        y1,     //
                   const real_t        y2,     //
                   const real_t        y3,     //
                   const real_t        z0,     //
                   const real_t        z1,     //
                   const real_t        z2,     //
                   const real_t        z3,     //
                   bool*               is_in) {              //
    //
    const real_t vol_tet_main = fabs(tet4_measure_v2(x0,    //
                                                     x1,    //
                                                     x2,    //
                                                     x3,    //
                                                     y0,    //
                                                     y1,    //
                                                     y2,    //
                                                     y3,    //
                                                     z0,    //
                                                     z1,    //
                                                     z2,    //
                                                     z3));  //

    for (int p_i = 0; p_i < p_cnt; p_i++) {
        // print tet vertices and the point coordinates
        // printf("Checking point %d at (%.6e, %.6e, %.6e) against tet vertices: \n", p_i, px[p_i], py[p_i], pz[p_i]);
        // printf("  Tet vertices X-coordinates: x0=%.6e, x1=%.6e, x2=%.6e, x3=%.6e \n", x0, x1, x2, x3);
        // printf("  Tet vertices Y-coordinates: y0=%.6e, y1=%.6e, y2=%.6e, y3=%.6e \n", y0, y1, y2, y3);
        // printf("  Tet vertices Z-coordinates: z0=%.6e, z1=%.6e, z2=%.6e, z3=%.  6e \n", z0, z1, z2, z3);

        const real_t vol0 = fabs(tet4_measure_v2(px[p_i], x1, x2, x3, py[p_i], y1, y2, y3, pz[p_i], z1, z2, z3));
        const real_t vol1 = fabs(tet4_measure_v2(x0, px[p_i], x2, x3, y0, py[p_i], y2, y3, z0, pz[p_i], z2, z3));
        const real_t vol2 = fabs(tet4_measure_v2(x0, x1, px[p_i], x3, y0, y1, py[p_i], y3, z0, z1, pz[p_i], z3));
        const real_t vol3 = fabs(tet4_measure_v2(x0, x1, x2, px[p_i], y0, y1, y2, py[p_i], z0, z1, z2, pz[p_i]));

        const real_t vol_sum  = vol0 + vol1 + vol2 + vol3;
        const real_t abs_diff = fabs(vol_sum - vol_tet_main);

        // Use both relative and absolute tolerance for robustness
        // Relative tolerance handles different mesh scales
        // Absolute tolerance prevents false negatives for very small tetrahedra
        const real_t rel_tolerance = 1e-2 * vol_tet_main;  // 1% relative error (more relaxed)
        const real_t abs_tolerance = 1e-10;                // Absolute tolerance for tiny volumes
        const real_t tolerance     = fmax(rel_tolerance, abs_tolerance);

        // printf("Point (%d): abs_diff = %.12e, tolerance = %.12e, vol_sum = %.12e, vol_tet_main = %.12e \n",
        //        p_i,
        //        abs_diff,
        //        tolerance,
        //        vol_sum,
        //        vol_tet_main);

        if (abs_diff > tolerance) {
            is_in[p_i] = false;
            // printf("  -> Point is OUTSIDE (abs_diff/vol_tet = %.6e)\n", abs_diff / vol_tet_main);
        } else {
            is_in[p_i] = true;
            // printf("  -> Point is INSIDE (abs_diff/vol_tet = %.6e)\n", abs_diff / vol_tet_main);
        }
    }

    return;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_tetrahedron_local_adjoint_category ////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
real_t                                                                                  //
tet4_apply_fun_tetrahedron_local_adjoint_category(const unsigned int     category,      //
                                                  const unsigned int     L,             // Refinement level
                                                  const real_t const*    bc,            // transposition vector for category
                                                  const real_t           J_phys[9],     // Jacobian matrix
                                                  const real_t           J_ref[9],      // Jacobian matrix
                                                  const real_t           det_J_phys,    // Determinant of the Jacobian matrix
                                                  const real_t           fx0,           // Tetrahedron vertices X-coordinates
                                                  const real_t           fy0,           // Tetrahedron vertices Y-coordinates
                                                  const real_t           fz0,           // Tetrahedron vertices Z-coordinates
                                                  const function_XYZ_t   fun_XYZ,       // Function to apply
                                                  const real_t* const    xyz_tet_main,  // Tetrahedron vertices coordinates
                                                  const geom_t           ox,            // Origin of the grid
                                                  const geom_t           oy,            //
                                                  const geom_t           oz,            //
                                                  const geom_t           dx,            // Spacing of the grid
                                                  const geom_t           dy,            //
                                                  const geom_t           dz,            //
                                                  const ptrdiff_t* const stride,        // Stride
                                                  const ptrdiff_t* const n,             // Size of the grid
                                                  real_t* const          hex_fun_data) {         // Output

    // Jacobian matrix for the tetrahedron

    const real_t N_micro_tet     = pow((double)(L), 3.0);  // Number of micro-tetrahedra in the HyTeg tetrahedron
    const real_t inv_N_micro_tet = 1.0 / N_micro_tet;      // Inverse of the number of micro-tetrahedra

    const real_t theta_volume = det_J_phys / ((real_t)(6.0));  // Volume of the mini-tetrahedron in the physical space

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

        const real_t xx_0 = (real_t)i * dx + ox;
        const real_t yy_0 = (real_t)j * dy + oy;
        const real_t zz_0 = (real_t)k * dz + oz;

        const real_t xx_1 = xx_0 + dx;
        const real_t yy_1 = yy_0 + dy;
        const real_t zz_1 = zz_0 + dz;

        // if (wf_quad < 0.6) printf("wf_quad = %g, dV = %g, It = %g\n", wf_quad, dV, It);

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

        const real_t p_hex_x[8] = {xx_0, xx_1, xx_1, xx_0, xx_0, xx_1, xx_1, xx_0};
        const real_t p_hex_y[8] = {yy_0, yy_0, yy_1, yy_1, yy_0, yy_0, yy_1, yy_1};
        const real_t p_hex_z[8] = {zz_0, zz_0, zz_0, zz_0, zz_1, zz_1, zz_1, zz_1};
        bool         is_in[8]   = {true, true, true, true, true, true, true, true};

        check_point_in_tet(8,                 // Number of points to check
                           p_hex_x,           // X-coordinates of the points to check
                           p_hex_y,           // Y-coordinates of the points to check
                           p_hex_z,           // Z-coordinates of the points to check
                           xyz_tet_main[0],   // Tetrahedron vertices X-coordinates
                           xyz_tet_main[1],   //
                           xyz_tet_main[2],   //
                           xyz_tet_main[3],   //
                           xyz_tet_main[4],   // Tetrahedron vertices Y-coordinates
                           xyz_tet_main[5],   //
                           xyz_tet_main[6],   //
                           xyz_tet_main[7],   //
                           xyz_tet_main[8],   // Tetrahedron vertices Z-coordinates
                           xyz_tet_main[9],   //
                           xyz_tet_main[10],  //
                           xyz_tet_main[11],  //
                           is_in);            // Output: is the point inside the tetrahedron?

        hex_fun_data[i0] = is_in[0] ? fun_XYZ(p_hex_x[0], p_hex_y[0], p_hex_z[0]) : hex_fun_data[i0];  // Update the data
        hex_fun_data[i1] = is_in[1] ? fun_XYZ(p_hex_x[1], p_hex_y[1], p_hex_z[1]) : hex_fun_data[i1];
        hex_fun_data[i2] = is_in[2] ? fun_XYZ(p_hex_x[2], p_hex_y[2], p_hex_z[2]) : hex_fun_data[i2];
        hex_fun_data[i3] = is_in[3] ? fun_XYZ(p_hex_x[3], p_hex_y[3], p_hex_z[3]) : hex_fun_data[i3];
        hex_fun_data[i4] = is_in[4] ? fun_XYZ(p_hex_x[4], p_hex_y[4], p_hex_z[4]) : hex_fun_data[i4];
        hex_fun_data[i5] = is_in[5] ? fun_XYZ(p_hex_x[5], p_hex_y[5], p_hex_z[5]) : hex_fun_data[i5];
        hex_fun_data[i6] = is_in[6] ? fun_XYZ(p_hex_x[6], p_hex_y[6], p_hex_z[6]) : hex_fun_data[i6];
        hex_fun_data[i7] = is_in[7] ? fun_XYZ(p_hex_x[7], p_hex_y[7], p_hex_z[7]) : hex_fun_data[i7];

        // Update the data

    }  // END: for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i++)

    return 0.0;  // Return the cumulative volume for debugging
}  // END OF FUNCTION tet4_resample_tetrahedron_local_adjoint_category

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_refine_adjoint_hyteg ////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                                                //
tet4_resample_field_apply_fun_to_hexa_d(const ptrdiff_t                      start_element,        // Mesh
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
                                        const function_XYZ_t                 fun_XYZ,              // Function to apply
                                        const real_t* const SFEM_RESTRICT    hex_fun_data) {          //

#define HYTEG_D_LOG_ENABLED 0

#if HYTEG_D_LOG_ENABLED
#define HYTEG_D_LOG(...) printf(__VA_ARGS__)
#else
#define HYTEG_D_LOG(...) (void)0
#endif

#define MAX_REF_L 20  // Maximum refinement level for HyTeg tetrahedra

    PRINT_CURRENT_FUNCTION;

    int64_t histo_L[MAX_REF_L + 1] = {0};  // Histogram of refinement levels

    int64_t max_L = 0;  // Maximum refinement level

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

    // #pragma omp parallel for  // Parallel loop over the elements
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

        const real_t xyz_tet_main[12] = {x0_n,
                                         x1_n,
                                         x2_n,
                                         x3_n,  //
                                         y0_n,
                                         y1_n,
                                         y2_n,
                                         y3_n,  //
                                         z0_n,
                                         z1_n,
                                         z2_n,
                                         z3_n};  //

        // const real_t wf0 = weighted_field[ev[0]];  // Weighted field at vertex 0
        // const real_t wf1 = weighted_field[ev[1]];  // Weighted field at vertex 1
        // const real_t wf2 = weighted_field[ev[2]];  // Weighted field at vertex 2
        // const real_t wf3 = weighted_field[ev[3]];  // Weighted field at vertex 3

        real_t det_J_phys =                     //
                fabs(make_Jacobian_matrix_tet(  //
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

        // const real_t alpha_min_threshold = 1.3;  // Minimum threshold for alpha, Less: more accurate.
        // const real_t alpha_max_threshold = 8.0;  // Maximum threshold for alpha. Less: make more refinements.

        // const int min_refinement_L = 1;   // Minimum refinement level
        // const int max_refinement_L = 22;  // HYTEG_MAX_REFINEMENT_LEVEL;  // Maximum refinement level

        const int L = alpha_to_hyteg_level(alpha_tet,                                // // DEBUG forced to 2 refinements
                                           mini_tet_parameters.alpha_min_threshold,  //
                                           mini_tet_parameters.alpha_max_threshold,  //
                                           mini_tet_parameters.min_refinement_L,     //
                                           1);                                       //

        histo_L[L] += 1;  // Update the histogram of refinement levels

        // if (element_i == 37078) {
        //     printf("---- Debug info (sfem_adjoint_mini_tet_kernel_gpu) ----\n");
        //     printf("Element %d / %d\n", element_i, end_element);
        //     printf("Vertex indices: %ld, %ld, %ld, %ld\n", (long)ev[0], (long)ev[1], (long)ev[2], (long)ev[3]);
        //     printf("Vertex coordinates:\n");
        //     printf("V0: (%e, %e, %e), wf0: %e\n", (double)x0_n, (double)y0_n, (double)z0_n, (double)wf0);
        //     printf("V1: (%e, %e, %e), wf1: %e\n", (double)x1_n, (double)y1_n, (double)z1_n, (double)wf1);
        //     printf("V2: (%e, %e, %e), wf2: %e\n", (double)x2_n, (double)y2_n, (double)z2_n, (double)wf2);
        //     printf("V3: (%e, %e, %e), wf3: %e\n", (double)x3_n, (double)y3_n, (double)z3_n, (double)wf3);
        //     printf("det_J_phys = %e\n", (double)det_J_phys);
        //     printf("tet J matrix = \n");
        //     printf("%e, %e, %e\n", (double)J_phy[0], (double)J_phy[1], (double)J_phy[2]);
        //     printf("%e, %e, %e\n", (double)J_phy[3], (double)J_phy[4], (double)J_phy[5]);
        //     printf("%e, %e, %e\n", (double)J_phy[6], (double)J_phy[7], (double)J_phy[8]);
        //     printf("tet_volume = %e\n", (double)tet_volume);
        //     printf("max_edges_length = %e, between vertices %d and %d\n", (double)max_edges_length, vertex_a, vertex_b);

        //     printf("---------------------------------------------------\n");
        // }

        if (L > max_L) {
            max_L = L;  // Update the maximum refinement level
            printf("New maximum refinement level: %d, alpha_tet = %g, max_edges_length = %g, d_min = %g, tet_volume = %g\n",  //
                   max_L,
                   alpha_tet,
                   max_edges_length,
                   d_min,
                   tet_volume);
        }

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
                                tet4_apply_fun_tetrahedron_local_adjoint_category(
                                        cat0,              //
                                        L,                 //
                                        b0,                // Translation vector for category 0)
                                        J_phy,             // Jacobian matrix
                                        J_vec_mini[cat0],  // Reference Jacobian matrix
                                        det_J_phys,        // Determinant of the Jacobian matrix for physical tet
                                        x0_n,              // Tetrahedron vertices XYZ-coordinates
                                        y0_n,              //
                                        z0_n,              //
                                        fun_XYZ,           // Function to apply
                                        xyz_tet_main,      // XYZ coordinates of the tetrahedron vertices
                                        ox,                // Origin of the grid
                                        oy,                //
                                        oz,                //
                                        dx,                // Spacing of the grid
                                        dy,                //
                                        dz,                //
                                        stride,            // Stride
                                        n,                 //
                                        hex_fun_data);     // Size of the grid

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
                                    tet4_apply_fun_tetrahedron_local_adjoint_category(
                                            cat_i,              //
                                            L,                  //
                                            b1,                 // Translation vector for category 1, 2, 3, 4
                                            J_phy,              // Jacobian matrix
                                            J_vec_mini[cat_i],  // Reference Jacobian matrix
                                            det_J_phys,         // Determinant of the Jacobian matrix
                                            x0_n,               // Tetrahedron vertices X-coordinates
                                            y0_n,               // Tetrahedron vertices Y-coordinates
                                            z0_n,               // Tetrahedron vertices Z-coordinates
                                            fun_XYZ,            // Function to apply
                                            xyz_tet_main,       // XYZ coordinates of the tetrahedron vertices
                                            ox,                 // Origin of the grid
                                            oy,                 //
                                            oz,                 //
                                            dx,                 // Spacing of the grid
                                            dy,                 //
                                            dz,                 //
                                            stride,             // Stride
                                            n,                  //
                                            hex_fun_data);      // Size of the grid

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

                        cumulated_volume +=                                         //
                                tet4_apply_fun_tetrahedron_local_adjoint_category(  //
                                        cat5,                                       //
                                        L,                                          //
                                        b5,                                         // Translation vector for category 5
                                        J_phy,                                      // Jacobian matrix
                                        J_vec_mini[cat5],                           // Reference Jacobian matrix
                                        det_J_phys,                                 // Determinant of the Jacobian matrix
                                        x0_n,                                       // Tetrahedron vertices X-coordinates
                                        y0_n,                                       //
                                        z0_n,                                       // Tetrahedron vertices Z-coordinates
                                        fun_XYZ,                                    // Function to apply
                                        xyz_tet_main,                               // XYZ coordinates of the tetrahedron vertices
                                        ox,                                         // Origin of the grid
                                        oy,                                         //
                                        oz,                                         //
                                        dx,                                         // Spacing of the grid
                                        dy,                                         //
                                        dz,                                         //
                                        stride,                                     // Stride
                                        n,                                          //
                                        hex_fun_data);                              // Size of the grid
                        // theta_volume_main += det_J_vec_phy[cat5] * (1.0 / 6.0);  // Accumulate the volume of the HyTeg
                        // tetrahedron

                        // Solve the case for the current Cat. 5 tetrahedron.
                    }  // END: if (j >= 1 && i >= 1) for cat 5
                }  // END: for (int j = 0; j < nodes_pes_side - i - 1; j++) // Loop over the nodes on the second edge
            }  // END: for (int i = 0; i < nodes_pes_side - 1; i++) // Loop over the nodes on the first edge
        }  // END: for (int k = 0; k < L + 1; k++) // Loop over the refinement levels

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
        }  // END: if (det_J_phys < 1e-8)

        // Check if the tetrahedron is uniformly refined
        if (L == 2) {
            uniform_refine_cnt++;
            HYTEG_D_LOG("Element %ld: Uniformly refined tetrahedron detected! L = %d\n", element_i, L);
        }  // END: if (L == 2)

    }  // END: for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++)

#if SFEM_LOG_LEVEL >= 5
    // Print the histogram of refinement levels
    printf("Histogram of refinement levels:\n");
    printf("Level,Number_of_elements\n");
    for (int l = 1; l <= MAX_REF_L; l++) {
        printf("%d, %ld\n", l, histo_L[l]);
    }

    printf("\nHistogram of refinement levels (visual):\n");

    // Find the maximum value
    int max_value = 0;
    for (int l = 1; l <= MAX_REF_L; l++) {
        if (histo_L[l] > max_value) {
            max_value = histo_L[l];
        }
    }  // END: for (int l = 1; l <= MAX_REF_L; l++)

    // Print the visual histogram with # symbols
    const int max_width = get_terminal_columns();  // Maximum number of # symbols per bar
    for (int l = 1; l <= MAX_REF_L; l++) {
        int num_symbols = max_value > 0 ? (int)((double)histo_L[l] / max_value * max_width) : 0;

        // Print the level and value
        printf("L = %3d: %e ", l, (double)(histo_L[l]));

        // Print the symbols
        for (int i = 0; i < num_symbols; i++) {
            printf("#");
        }
        printf("\n");
    }

    // Print the scale
    printf("\nScale: Each # represents approximately %e elements\n", (double)max_value / max_width);
#endif

    return ret;  // Return the result of the refinement
}
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////