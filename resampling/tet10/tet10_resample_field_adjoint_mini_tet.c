#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #define real_t  real_t

#include "hyteg.h"
#include "hyteg_Jacobian_matrix_real_t.h"
#include "mass.h"
#include "matrixio_array.h"
#include "quadratures_rule.h"
#include "sfem_base.h"
#include "sfem_resample_field.h"
#include "sfem_resample_field_adjoint_hyteg.h"
#include "sfem_resample_field_tet4_math.h"
#include "sfem_stack.h"
#include "tet10_resample_field.h"
#include "tet10_resample_field_V2.h"
#include "tet10_weno.h"

////////////////////////////////////////////////////////////////////////////////
// Hex8 to isoparametric tet10 local adjoint category
////////////////////////////////////////////////////////////////////////////////
real_t                                                                                  //
hex8_to_isoparametric_tet10_local_adjoint_category(const int             L,             //
                                                   const real_t const*   bc,            // transposition vector for category
                                                   const real_t const    J_phys[9],     // Jacobian matrix
                                                   const real_t const    J_ref[9],      // Jacobian matrix
                                                   const real_t          det_J_phys,    //
                                                   const geom_t          x[10],         // Tetrahedron vertices X-coordinates
                                                   const geom_t          y[10],         // Tetrahedron vertices Y-coordinates
                                                   const geom_t          z[10],         // Tetrahedron vertices Z-coordinates
                                                   const real_t          ox,            // Origin of the grid
                                                   const real_t          oy,            //
                                                   const real_t          oz,            //
                                                   const real_t          dx,            // Spacing of the grid
                                                   const real_t          dy,            //
                                                   const real_t          dz,            //
                                                   const real_t          wf_tet10[10],  // Weighted field at the vertices
                                                   const ptrdiff_t const stride[3],     // Stride
                                                   real_t* const         data) {                // Determinant of the Jacobian matrix
    // mini-tet parameters

    const real_t N_micro_tet = (real_t)(L) * (real_t)(L) * (real_t)(L);  // Number of micro-tetrahedra in the HyTeg tetrahedron
    const real_t inv_N_micro_tet = 1.0 / N_micro_tet;                    // Inverse of the number of micro-tetrahedra

    const real_t theta_volume = det_J_phys / ((real_t)(6.0));  // Volume of the mini-tetrahedron in the physical space

    real_t hex8_f[8];
    real_t tet10_f[10];

    const real_t fx0 = (real_t)x[0];  // Tetrahedron Origin X-coordinate
    const real_t fy0 = (real_t)y[0];  // Tetrahedron Origin Y-coordinate
    const real_t fz0 = (real_t)z[0];  // Tetrahedron Origin Z-coordinate

    for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i++) {  // loop over the quadrature points

        // Mapping the quadrature point from the reference space to the mini-tetrahedron
        const real_t xq_mref = J_ref[0] * tet_qx[quad_i] + J_ref[1] * tet_qy[quad_i] + J_ref[2] * tet_qz[quad_i] + bc[0];
        const real_t yq_mref = J_ref[3] * tet_qx[quad_i] + J_ref[4] * tet_qy[quad_i] + J_ref[5] * tet_qz[quad_i] + bc[1];
        const real_t zq_mref = J_ref[6] * tet_qx[quad_i] + J_ref[7] * tet_qy[quad_i] + J_ref[8] * tet_qz[quad_i] + bc[2];

        real_t g_qx;
        real_t g_qy;
        real_t g_qz;

        tet10_transform_real_t(x, y, z, xq_mref, yq_mref, zq_mref, &g_qx, &g_qy, &g_qz);

        // tet10_dual_basis_hrt(tet_qx[q], tet_qy[q], tet_qz[q], tet10_f);
        tet10_Lagrange_basis(xq_mref, yq_mref, zq_mref, tet10_f);

        const real_t grid_x = (g_qx - ox) / dx;
        const real_t grid_y = (g_qy - oy) / dy;
        const real_t grid_z = (g_qz - oz) / dz;

        const ptrdiff_t i = floor(grid_x);
        const ptrdiff_t j = floor(grid_y);
        const ptrdiff_t k = floor(grid_z);

        // Get the reminder [0, 1]
        real_t l_x = (grid_x - i);
        real_t l_y = (grid_y - j);
        real_t l_z = (grid_z - k);

        assert(l_x >= -1e-8);
        assert(l_y >= -1e-8);
        assert(l_z >= -1e-8);

        assert(l_x <= 1 + 1e-8);
        assert(l_y <= 1 + 1e-8);
        assert(l_z <= 1 + 1e-8);

        ptrdiff_t indices[10];
        hex_aa_8_collect_indices(stride, i, j, k, indices);

        hex_aa_8_eval_fun(l_x, l_y, l_z, hex8_f);

        const real_t measure = tet10_measure_real_t(x, y, z, xq_mref, yq_mref, zq_mref);
        const real_t dV      = measure * tet_qw[quad_i] * inv_N_micro_tet;

        const real_t It = (tet10_f[0] * wf_tet10[0] +  //
                           tet10_f[1] * wf_tet10[1] +  //
                           tet10_f[2] * wf_tet10[2] +  //
                           tet10_f[3] * wf_tet10[3] +  //
                           tet10_f[4] * wf_tet10[4] +  //
                           tet10_f[5] * wf_tet10[5] +  //
                           tet10_f[6] * wf_tet10[6] +  //
                           tet10_f[7] * wf_tet10[7] +  //
                           tet10_f[8] * wf_tet10[8] +  //
                           tet10_f[9] * wf_tet10[9]);  //

        const real_t d0 = It * hex8_f[0] * dV;
        const real_t d1 = It * hex8_f[1] * dV;
        const real_t d2 = It * hex8_f[2] * dV;
        const real_t d3 = It * hex8_f[3] * dV;
        const real_t d4 = It * hex8_f[4] * dV;
        const real_t d5 = It * hex8_f[5] * dV;
        const real_t d6 = It * hex8_f[6] * dV;
        const real_t d7 = It * hex8_f[7] * dV;

        data[indices[0]] += d0;
        data[indices[1]] += d1;
        data[indices[2]] += d2;
        data[indices[3]] += d3;
        data[indices[4]] += d4;
        data[indices[5]] += d5;
        data[indices[6]] += d6;
        data[indices[7]] += d7;
    }
}

//////////////////////////////////////////////////////////////////////////////////
// transform_tet10_to_mini_phys_tet10
//////////////////////////////////////////////////////////////////////////////////
void                                                              //
transform_tet10_to_mini_phys_tet10(const real_t const x[10],      // Tetrahedron vertices X-coordinates
                                   const real_t const y[10],      // Tetrahedron vertices Y-coordinates
                                   const real_t const z[10],      // Tetrahedron vertices Z-coordinates,    //
                                   real_t             J_phys[9],  // Jacobian matrix of the physical tetrahedron
                                   real_t             J_ref[9],   // Jacobian matrix of the reference tetrahedron
                                   real_t             x_m[10],    // Output mini-tetrahedra X-coordinates
                                   real_t             y_m[10],    // Output mini-tetrahedra Y-coordinates
                                   real_t             z_m[10]) {              // Output mini-tetrahedra Z-coordinates

#define SFEM_MAT_VEC_MUL_3x3_INDEX(mat, x_in, y_in, z_in, x_out, y_out, z_out, idx)              \
    do {                                                                                         \
        (x_out)[idx] = (mat)[0] * (x_in)[idx] + (mat)[1] * (y_in)[idx] + (mat)[2] * (z_in)[idx]; \
        (y_out)[idx] = (mat)[3] * (x_in)[idx] + (mat)[4] * (y_in)[idx] + (mat)[5] * (z_in)[idx]; \
        (z_out)[idx] = (mat)[6] * (x_in)[idx] + (mat)[7] * (y_in)[idx] + (mat)[8] * (z_in)[idx]; \
    } while (0)

    real_t J_tot[9];

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            J_tot[i * 3 + j] = 0.0;
            for (int k = 0; k < 3; k++) {
                J_tot[i * 3 + j] += J_phys[i * 3 + k] * J_ref[k * 3 + j];
            }
        }
    }

    SFEM_MAT_VEC_MUL_3x3_INDEX(J_tot, x, y, z, x_m, y_m, z_m, 0);
    SFEM_MAT_VEC_MUL_3x3_INDEX(J_tot, x, y, z, x_m, y_m, z_m, 1);
    SFEM_MAT_VEC_MUL_3x3_INDEX(J_tot, x, y, z, x_m, y_m, z_m, 2);
    SFEM_MAT_VEC_MUL_3x3_INDEX(J_tot, x, y, z, x_m, y_m, z_m, 3);
    SFEM_MAT_VEC_MUL_3x3_INDEX(J_tot, x, y, z, x_m, y_m, z_m, 4);
    SFEM_MAT_VEC_MUL_3x3_INDEX(J_tot, x, y, z, x_m, y_m, z_m, 5);
    SFEM_MAT_VEC_MUL_3x3_INDEX(J_tot, x, y, z, x_m, y_m, z_m, 6);
    SFEM_MAT_VEC_MUL_3x3_INDEX(J_tot, x, y, z, x_m, y_m, z_m, 7);
    SFEM_MAT_VEC_MUL_3x3_INDEX(J_tot, x, y, z, x_m, y_m, z_m, 8);
    SFEM_MAT_VEC_MUL_3x3_INDEX(J_tot, x, y, z, x_m, y_m, z_m, 9);
}

void                                                               //
translate_tet10_to_mini_phys_tet10(const real_t const x_m[10],     // Tetrahedron vertices X-coordinates
                                   const real_t const y_m[10],     // Tetrahedron vertices Y-coordinates
                                   const real_t const z_m[10],     // Tetrahedron vertices Z-coordinates
                                   const real_t       v0x,         // Jacobian matrix of the physical tetrahedron
                                   const real_t       v0y,         //
                                   const real_t       v0z,         //
                                   real_t             x_m_tr[10],  // Output mini-tetrahedra X-coordinates
                                   real_t             y_m_tr[10],  // Output mini-tetrahedra Y-coordinates
                                   real_t             z_m_tr[10]) {            // Output mini-tetrahedra Z-coordinates

    for (int i = 0; i < 10; i++) {
        x_m_tr[i] = x_m[i] + v0x;
        y_m_tr[i] = y_m[i] + v0y;
        z_m_tr[i] = z_m[i] + v0z;
    }
}

/**
 * @brief Resamples a field from a 10-node tetrahedral mesh back to a structured hexahedral grid with adaptive refinement.
 *
 * @param start_element
 * @param end_element
 * @param nnodes
 * @param elems
 * @param xyz
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param weighted_field
 * @param data
 * @return int
 */
int                                                                                                               //
hex8_to_isoparametric_tet10_resample_field_hyteg_mt_adjoint(const ptrdiff_t                      start_element,   // Mesh
                                                            const ptrdiff_t                      end_element,     //
                                                            const ptrdiff_t                      nnodes,          //
                                                            const idx_t** const SFEM_RESTRICT    elems,           //
                                                            const geom_t** const SFEM_RESTRICT   xyz,             //
                                                            const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                                                            const ptrdiff_t* const SFEM_RESTRICT stride,          //
                                                            const geom_t* const SFEM_RESTRICT    origin,          //
                                                            const geom_t* const SFEM_RESTRICT    delta,           //
                                                            const real_t* const SFEM_RESTRICT    weighted_field,  // Input WF
                                                            real_t* const SFEM_RESTRICT          data,            //
                                                            const mini_tet_parameters_t          mini_tet_parameters) {    // Output
    //
    PRINT_CURRENT_FUNCTION;

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

    real_t x_mv[6][10];  // Mini-tetrahedra X-coordinates for the 6 categories of tetrahedra for the refined and reference element
    real_t y_mv[6][10];  // Mini-tetrahedra Y-coordinates for the 6 categories of tetrahedra for the refined and reference element
    real_t z_mv[6][10];  // Mini-tetrahedra Z-coordinates for the 6 categories of tetrahedra for the refined and reference element

    real_t J_vec_mini[6][9];  // Jacobian matrices for the 6 categories of tetrahedra for the refined and reference element
    real_t J_phy[9];          // Jacobian matrices for the 6 categories of tetrahedra for the physical current

    real_t hex8_f[8];
    real_t tet10_f[10];

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        // ISOPARAMETRIC
        geom_t x[10], y[10], z[10];
        idx_t  ev[10];

        real_t hex8_f[8];
        real_t coeffs[8];

        real_t tet10_f[10];
        // real_t element_field[10];

        // loop over the 4 vertices of the tetrahedron
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][element_i];
        }

        // ISOPARAMETRIC
        for (int v = 0; v < 10; ++v) {
            x[v] = (geom_t)(xyz[0][ev[v]]);  // x-coordinates
            y[v] = (geom_t)(xyz[1][ev[v]]);  // y-coordinates
            z[v] = (geom_t)(xyz[2][ev[v]]);  // z-coordinates
        }

        const real_t det_J_phys =               //
                fabs(make_Jacobian_matrix_tet(  //
                        x[0],                   // Tetrahedron vertices X-coordinates
                        x[1],                   //
                        x[2],                   //
                        x[3],                   //
                        y[0],                   // Tetrahedron vertices Y-coordinates
                        y[1],                   //
                        y[2],                   //
                        y[3],                   //
                        z[0],                   // Tetrahedron vertices Z-coordinates
                        z[1],                   //
                        z[2],                   //
                        z[3],                   // Vertex 3 coordinates
                        J_phy));                // Output Jacobian matrix

        // memset(element_field, 0, 10 * sizeof(real_t));

        // set to zero the element field
        // memset(element_field, 0, 10 * sizeof(real_t));

        const real_t wf_tet10[10] = {weighted_field[ev[0]],
                                     weighted_field[ev[1]],
                                     weighted_field[ev[2]],
                                     weighted_field[ev[3]],
                                     weighted_field[ev[4]],
                                     weighted_field[ev[5]],
                                     weighted_field[ev[6]],
                                     weighted_field[ev[7]],
                                     weighted_field[ev[8]],
                                     weighted_field[ev[9]]};

        real_t edges_length[6];
        int    vertex_a, vertex_b;

        const real_t max_edge_len = tet10_edge_lengths(x,              //
                                                       y,              //
                                                       z,              //
                                                       &vertex_a,      //
                                                       &vertex_b,      //
                                                       edges_length);  //

        const real_t alpha = max_edge_len / d_min;

        const int L = alpha_to_hyteg_level(alpha,                                    //
                                           mini_tet_parameters.alpha_min_threshold,  //
                                           mini_tet_parameters.alpha_max_threshold,  //
                                           mini_tet_parameters.min_refinement_L,     //
                                           mini_tet_parameters.max_refinement_L);    //

        // printf("Element %td: max_edge_len = %e, d_min = %e, alpha = %e, L = %d, J = %e\n",
        //        element_i,
        //        max_edge_len,
        //        d_min,
        //        alpha,
        //        L,
        //        det_J_phys);

        const real_t h = 1.0 / (real_t)(L);  // Size of the mini-tetrahedra in the reference space

        real_t theta_volume_main = 0.0;  // Volume of the HyTeg tetrahedron

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
                                   (real_t)(L),         // Refinement level
                                   J_vec_mini[cat_i]);  // Output Jacobian matrix

            // Calculate the mini-tetrahedra coordinates for the current category
            transform_tet10_to_mini_phys_tet10(x,                  // Tetrahedron vertices X-coordinates
                                               y,                  // Tetrahedron vertices Y-coordinates
                                               z,                  // Tetrahedron vertices Z-coordinates
                                               J_phy,              // Jacobian matrix of the physical tetrahedron
                                               J_vec_mini[cat_i],  // Jacobian matrix of the reference tetrahedron
                                               x_mv[cat_i],        // Output mini-tetrahedra X-coordinates
                                               y_mv[cat_i],        // Output mini-tetrahedra Y-coordinates
                                               z_mv[cat_i]);       // Output mini-tetrahedra Z-coordinates

        }  // END of for (int cat_i = 0; cat_i < 6; cat_i++)

        for (int k = 0; k < L + 1; k++) {  // Loop over the refinement levels

            const unsigned int nodes_pes_side = (L - k) + 1;
            // const unsigned int nodes_per_layer = nodes_pes_side * (nodes_pes_side + 1) / 2;

            for (int i = 0; i < nodes_pes_side - 1; i++) {          // Loop over the nodes on the first edge
                for (int j = 0; j < nodes_pes_side - i - 1; j++) {  // Loop over the nodes on the second edge

                    const real_t b0[3] = {(real_t)(j)*h,   //
                                          (real_t)(i)*h,   //
                                          (real_t)(k)*h};  //

                    {  // BEGIN: Cat 0
                        const unsigned int cat0 = 0;

                        hex8_to_isoparametric_tet10_local_adjoint_category(  //
                                L,                                           //
                                b0,                                          // Translation vector for category 0
                                J_phy,                                       // Jacobian matrix
                                J_vec_mini[cat0],                            // Reference Jacobian matrix
                                det_J_phys,                                  // Determinant of the Jacobian matrix
                                x,                                           // Tetrahedron vertices X-coordinates
                                y,                                           //
                                z,                                           // Tetrahedron vertices Z-coordinates
                                ox,                                          // Origin of the grid
                                oy,                                          //
                                oz,                                          //
                                dx,                                          // Spacing of the grid
                                dy,                                          //
                                dz,                                          //
                                wf_tet10,                                    // Weighted field at the vertices
                                stride,                                      // Stride
                                data);                                       // Size of the grid
                    }

                    if (j >= 1) {
                        for (int cat_i = 1; cat_i < 5; cat_i++) {
                            hex8_to_isoparametric_tet10_local_adjoint_category(  //
                                    L,                                           //
                                    b0,                                          // Translation vector for category 0
                                    J_phy,                                       // Jacobian matrix
                                    J_vec_mini[cat_i],                           // Reference Jacobian matrix
                                    det_J_phys,                                  // Determinant of the Jacobian matrix
                                    x,                                           // Tetrahedron vertices X-coordinates
                                    y,                                           //
                                    z,                                           // Tetrahedron vertices Z-coordinates
                                    ox,                                          // Origin of the grid
                                    oy,                                          //
                                    oz,                                          //
                                    dx,                                          // Spacing of the grid
                                    dy,                                          //
                                    dz,                                          //
                                    wf_tet10,                                    // Weighted field at the vertices
                                    stride,                                      // Stride
                                    data);                                       // Size of the grid
                        }
                    }

                    {
                        const unsigned int cat5 = 5;
                        if (j >= 1 && i >= 1) {
                            hex8_to_isoparametric_tet10_local_adjoint_category(  //
                                    L,                                           //
                                    b0,                                          // Translation vector for category 5
                                    J_phy,                                       // Jacobian matrix
                                    J_vec_mini[cat5],                            // Reference Jacobian matrix
                                    det_J_phys,                                  // Determinant of the Jacobian matrix
                                    x,                                           // Tetrahedron vertices X-coordinates
                                    y,                                           //
                                    z,                                           // Tetrahedron vertices Z-coordinates
                                    ox,                                          // Origin of the grid
                                    oy,                                          //
                                    oz,                                          //
                                    dx,                                          // Spacing of the grid
                                    dy,                                          //
                                    dz,                                          //
                                    wf_tet10,                                    // Weighted field at the vertices
                                    stride,                                      // Stride
                                    data);                                       // Size of the grid
                        }
                    }
                }
            }
        }
    }

    RETURN_FROM_FUNCTION(0);
}