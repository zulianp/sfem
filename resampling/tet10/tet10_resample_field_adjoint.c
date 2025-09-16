#include "sfem_resample_field_tet4_math.h"
#include "tet10_resample_field.h"
#include "tet10_resample_field_V2.h"

#include "quadratures_rule.h"
#include "sfem_stack.h"
#include "tet10_weno.h"

#include "sfem_base.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////
// tet10_uniform_refinement
///////////////////////////////////////////////////////////////////////
int                                                             //
tet10_uniform_refinement(const real_t* const          x,        //
                         const real_t* const          y,        //
                         const real_t* const          z,        //
                         const real_t* const          w,        //
                         struct tet10_vertices* const rTets) {  //

    const int refine_pattern[8][4] = {// Corner tests
                                      {0, 4, 6, 7},
                                      {4, 1, 5, 8},
                                      {6, 5, 2, 9},
                                      {7, 8, 9, 3},
                                      // Octahedron tets
                                      {4, 5, 6, 8},
                                      {7, 4, 6, 8},
                                      {6, 5, 9, 8},
                                      {7, 6, 9, 8}};

    for (int ni = 0; ni < 8; ni++) {
        const int v_indices[4] = {refine_pattern[ni][0],   //
                                  refine_pattern[ni][1],   //
                                  refine_pattern[ni][2],   //
                                  refine_pattern[ni][3]};  //

        for (int j = 0; j < 4; j++) {
            // Assign the vertices coordinates of the refined tetrahedron
            rTets[ni].x[j] = x[v_indices[j]];
            rTets[ni].y[j] = y[v_indices[j]];
            rTets[ni].z[j] = z[v_indices[j]];
            rTets[ni].w[j] = w[v_indices[j]];
        }
    }

    // Add the mid-edge vertices and weights of the refined tetrahedron
    for (int ni = 0; ni < 8; ni++) {
        rTets[ni].x[4] = 0.5 * (rTets[ni].x[0] + rTets[ni].x[1]);
        rTets[ni].y[4] = 0.5 * (rTets[ni].y[0] + rTets[ni].y[1]);
        rTets[ni].z[4] = 0.5 * (rTets[ni].z[0] + rTets[ni].z[1]);
        rTets[ni].w[4] = 0.5 * (rTets[ni].w[0] + rTets[ni].w[1]);

        rTets[ni].x[5] = 0.5 * (rTets[ni].x[1] + rTets[ni].x[2]);
        rTets[ni].y[5] = 0.5 * (rTets[ni].y[1] + rTets[ni].y[2]);
        rTets[ni].z[5] = 0.5 * (rTets[ni].z[1] + rTets[ni].z[2]);
        rTets[ni].w[5] = 0.5 * (rTets[ni].w[1] + rTets[ni].w[2]);

        rTets[ni].x[6] = 0.5 * (rTets[ni].x[0] + rTets[ni].x[2]);
        rTets[ni].y[6] = 0.5 * (rTets[ni].y[0] + rTets[ni].y[2]);
        rTets[ni].z[6] = 0.5 * (rTets[ni].z[0] + rTets[ni].z[2]);
        rTets[ni].w[6] = 0.5 * (rTets[ni].w[0] + rTets[ni].w[2]);

        rTets[ni].x[7] = 0.5 * (rTets[ni].x[0] + rTets[ni].x[3]);
        rTets[ni].y[7] = 0.5 * (rTets[ni].y[0] + rTets[ni].y[3]);
        rTets[ni].z[7] = 0.5 * (rTets[ni].z[0] + rTets[ni].z[3]);
        rTets[ni].w[7] = 0.5 * (rTets[ni].w[0] + rTets[ni].w[3]);

        rTets[ni].x[8] = 0.5 * (rTets[ni].x[1] + rTets[ni].x[3]);
        rTets[ni].y[8] = 0.5 * (rTets[ni].y[1] + rTets[ni].y[3]);
        rTets[ni].z[8] = 0.5 * (rTets[ni].z[1] + rTets[ni].z[3]);
        rTets[ni].w[8] = 0.5 * (rTets[ni].w[1] + rTets[ni].w[3]);

        rTets[ni].x[9] = 0.5 * (rTets[ni].x[2] + rTets[ni].x[3]);
        rTets[ni].y[9] = 0.5 * (rTets[ni].y[2] + rTets[ni].y[3]);
        rTets[ni].z[9] = 0.5 * (rTets[ni].z[2] + rTets[ni].z[3]);
        rTets[ni].w[9] = 0.5 * (rTets[ni].w[2] + rTets[ni].w[3]);
    }

    return 8;
}

///////////////////////////////////////////////////////////////////////
// tet10_get_intermediate_index
///////////////////////////////////////////////////////////////////////
int                                                 //
tet10_get_intermediate_index(const int vertex_a,    //
                             const int vertex_b) {  //
                                                    //
    if (vertex_a == 0 && vertex_b == 1) return 4;
    if (vertex_a == 1 && vertex_b == 2) return 5;
    if (vertex_a == 0 && vertex_b == 2) return 6;
    if (vertex_a == 0 && vertex_b == 3) return 7;
    if (vertex_a == 1 && vertex_b == 3) return 8;
    if (vertex_a == 2 && vertex_b == 3) return 9;

    return -1;
}

/////////////////////////////////////////////////////////////////
// tet10_refine_two_edge_vertex
/////////////////////////////////////////////////////////////////
int                                                                  //
tet10_refine_two_edge_vertex(const real_t* const          x,         //
                             const real_t* const          y,         //
                             const real_t* const          z,         //
                             const real_t* const          w,         //
                             const int                    vertex_a,  //
                             const int                    vertex_b,  //
                             struct tet10_vertices* const rTets) {   //

    // Return the number of the new and refined tetrahedra
    // if 2 is OK
    // otherwise return 1

    if (vertex_b <= vertex_a) return 1;
    if (vertex_a < 0 || vertex_a > 3) return 1;
    if (vertex_b < 0 || vertex_b > 3) return 1;

    const int vertex_N = tet10_get_intermediate_index(vertex_a, vertex_b);
    if (vertex_N < 0) {
        printf("Error: vertex_N < 0\n");
        exit(1);
    }

    // if (vertex_a == 0 && vertex_b == 1) {
    //     printf("vertex_a = %d, vertex_b = %d, vertex_N = %d\n", vertex_a, vertex_b, vertex_N);
    // }

    // const real_t N1_x = 0.5 * (x[1] + x[vertex_N]);
    // const real_t N1_y = 0.5 * (y[1] + y[vertex_N]);
    // const real_t N1_z = 0.5 * (z[1] + z[vertex_N]);

    // const real_t N2_x = 0.5 * (x[2] + x[vertex_N]);
    // const real_t N2_y = 0.5 * (y[2] + y[vertex_N]);
    // const real_t N2_z = 0.5 * (z[2] + z[vertex_N]);

    // const real_t N3_x = 0.5 * (x[3] + x[vertex_N]);
    // const real_t N3_y = 0.5 * (y[3] + y[vertex_N]);
    // const real_t N3_z = 0.5 * (z[3] + z[vertex_N]);

    // const real_t N4_x = 0.5 * (x[0] + x[vertex_N]);
    // const real_t N4_y = 0.5 * (y[0] + y[vertex_N]);
    // const real_t N4_z = 0.5 * (z[0] + z[vertex_N]);

    real_t xn_1[10] = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]};
    real_t yn_1[10] = {y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]};
    real_t zn_1[10] = {z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9]};
    real_t wn_1[10] = {w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9]};

    real_t xn_2[10] = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]};
    real_t yn_2[10] = {y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]};
    real_t zn_2[10] = {z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9]};
    real_t wn_2[10] = {w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9]};

    xn_1[vertex_a] = x[vertex_N];
    yn_1[vertex_a] = y[vertex_N];
    zn_1[vertex_a] = z[vertex_N];

    xn_2[vertex_b] = x[vertex_N];
    yn_2[vertex_b] = y[vertex_N];
    zn_2[vertex_b] = z[vertex_N];

    // xn_1[vertex_N] = N1_x;
    // yn_1[vertex_N] = N1_y;
    // zn_1[vertex_N] = N1_z;

    // xn_2[vertex_N] = N4_x;
    // yn_2[vertex_N] = N4_y;
    // zn_2[vertex_N] = N4_z;

    xn_1[4] = 0.5 * (xn_1[0] + xn_1[1]);
    yn_1[4] = 0.5 * (yn_1[0] + yn_1[1]);
    zn_1[4] = 0.5 * (zn_1[0] + zn_1[1]);

    xn_1[5] = 0.5 * (xn_1[1] + xn_1[2]);
    yn_1[5] = 0.5 * (yn_1[1] + yn_1[2]);
    zn_1[5] = 0.5 * (zn_1[1] + zn_1[2]);

    xn_1[6] = 0.5 * (xn_1[0] + xn_1[2]);
    yn_1[6] = 0.5 * (yn_1[0] + yn_1[2]);
    zn_1[6] = 0.5 * (zn_1[0] + zn_1[2]);

    xn_1[7] = 0.5 * (xn_1[0] + xn_1[3]);
    yn_1[7] = 0.5 * (yn_1[0] + yn_1[3]);
    zn_1[7] = 0.5 * (zn_1[0] + zn_1[3]);

    xn_1[8] = 0.5 * (xn_1[1] + xn_1[3]);
    yn_1[8] = 0.5 * (yn_1[1] + yn_1[3]);
    zn_1[8] = 0.5 * (zn_1[1] + zn_1[3]);

    xn_1[9] = 0.5 * (xn_1[2] + xn_1[3]);
    yn_1[9] = 0.5 * (yn_1[2] + yn_1[3]);
    zn_1[9] = 0.5 * (zn_1[2] + zn_1[3]);

    wn_1[vertex_a] = w[vertex_N];
    wn_1[4]        = 0.5 * (wn_1[0] + wn_1[1]);
    wn_1[5]        = 0.5 * (wn_1[1] + wn_1[2]);
    wn_1[6]        = 0.5 * (wn_1[0] + wn_1[2]);
    wn_1[7]        = 0.5 * (wn_1[0] + wn_1[3]);
    wn_1[8]        = 0.5 * (wn_1[1] + wn_1[3]);
    wn_1[9]        = 0.5 * (wn_1[2] + wn_1[3]);

    ///////

    xn_2[4] = 0.5 * (xn_2[0] + xn_2[1]);
    yn_2[4] = 0.5 * (yn_2[0] + yn_2[1]);
    zn_2[4] = 0.5 * (zn_2[0] + zn_2[1]);

    xn_2[5] = 0.5 * (xn_2[1] + xn_2[2]);
    yn_2[5] = 0.5 * (yn_2[1] + yn_2[2]);
    zn_2[5] = 0.5 * (zn_2[1] + zn_2[2]);

    xn_2[6] = 0.5 * (xn_2[0] + xn_2[2]);
    yn_2[6] = 0.5 * (yn_2[0] + yn_2[2]);
    zn_2[6] = 0.5 * (zn_2[0] + zn_2[2]);

    xn_2[7] = 0.5 * (xn_2[0] + xn_2[3]);
    yn_2[7] = 0.5 * (yn_2[0] + yn_2[3]);
    zn_2[7] = 0.5 * (zn_2[0] + zn_2[3]);

    xn_2[8] = 0.5 * (xn_2[1] + xn_2[3]);
    yn_2[8] = 0.5 * (yn_2[1] + yn_2[3]);
    zn_2[8] = 0.5 * (zn_2[1] + zn_2[3]);

    xn_2[9] = 0.5 * (xn_2[2] + xn_2[3]);
    yn_2[9] = 0.5 * (yn_2[2] + yn_2[3]);
    zn_2[9] = 0.5 * (zn_2[2] + zn_2[3]);

    wn_2[vertex_b] = w[vertex_N];
    wn_2[4]        = 0.5 * (wn_2[0] + wn_2[1]);
    wn_2[5]        = 0.5 * (wn_2[1] + wn_2[2]);
    wn_2[6]        = 0.5 * (wn_2[0] + wn_2[2]);
    wn_2[7]        = 0.5 * (wn_2[0] + wn_2[3]);
    wn_2[8]        = 0.5 * (wn_2[1] + wn_2[3]);
    wn_2[9]        = 0.5 * (wn_2[2] + wn_2[3]);

    for (int i = 0; i < 10; i++) {
        rTets[0].x[i] = xn_1[i];
        rTets[0].y[i] = yn_1[i];
        rTets[0].z[i] = zn_1[i];
        rTets[0].w[i] = wn_1[i];

        rTets[1].x[i] = xn_2[i];
        rTets[1].y[i] = yn_2[i];
        rTets[1].z[i] = zn_2[i];
        rTets[1].w[i] = wn_2[i];
    }

    return 2;
}

///////////////////////////////////////////////////
////////////////////
// tet10_volumes
///////////////////////////////////////////////////////////////////////
real_t                                                   //
tet10_volumes(const struct tet10_vertices* const rTets,  //
              const int                          N,      //
              real_t* const                      V) {                         //

    real_t volume = 0.0;

    for (int ni = 0; ni < N; ni++) {
        const real_t volume_loc = tet4_measure(rTets[ni].x[0],  // x-coordinates
                                               rTets[ni].x[1],
                                               rTets[ni].x[2],
                                               rTets[ni].x[3],
                                               rTets[ni].y[0],  // y-coordinates
                                               rTets[ni].y[1],
                                               rTets[ni].y[2],
                                               rTets[ni].y[3],
                                               rTets[ni].z[0],  // z-coordinates
                                               rTets[ni].z[1],
                                               rTets[ni].z[2],
                                               rTets[ni].z[3]);  //

        volume += volume_loc;
        V[ni] = volume_loc;
    }

    return volume;
}

///////////////////////////////////////////////////////////////////////
// tet10_edge_lengths
///////////////////////////////////////////////////////////////////////
real_t                                            //
tet10_edge_lengths(const real_t* x,               //
                   const real_t* y,               //
                   const real_t* z,               //
                   int*          vertex_a,        //
                   int*          vertex_b,        //
                   real_t* const edge_lengths) {  //

    *vertex_a = -1;
    *vertex_b = -1;

    real_t    max_length          = 0.0;
    const int edges_pattern[6][2] = {{0, 1},   //
                                     {1, 2},   //
                                     {2, 0},   //
                                     {0, 3},   //
                                     {1, 3},   //
                                     {2, 3}};  //

    for (int i = 0; i < 6; i++) {
        const int i0 = edges_pattern[i][0];
        const int i1 = edges_pattern[i][1];

        const real_t len = sqrt((x[i0] - x[i1]) * (x[i0] - x[i1]) +  //
                                (y[i0] - y[i1]) * (y[i0] - y[i1]) +  //
                                (z[i0] - z[i1]) * (z[i0] - z[i1]));  //

        edge_lengths[i] = len;

        if (len > max_length) {
            *vertex_a  = i0;
            *vertex_b  = i1;
            max_length = len;
        }
    }

    if (*vertex_a > *vertex_b) {
        const int tmp = *vertex_a;
        *vertex_a     = *vertex_b;
        *vertex_b     = tmp;
    }

    return max_length;
}  // END tet10_edge_lengths

///////////////////////////////////////////////////////////////////////
// hex8_to_isoparametric_tet10_resample_field
///////////////////////////////////////////////////////////////////////
int                                                                                              //
hex8_to_isoparametric_tet10_resample_tet_adjoint(const real_t* const SFEM_RESTRICT    x,         //
                                                 const real_t* const SFEM_RESTRICT    y,         //
                                                 const real_t* const SFEM_RESTRICT    z,         //
                                                 const real_t* const SFEM_RESTRICT    wf_tet10,  //
                                                 const ptrdiff_t* const SFEM_RESTRICT stride,    //
                                                 const geom_t* const SFEM_RESTRICT    origin,    //
                                                 const geom_t* const SFEM_RESTRICT    delta,     //
                                                 real_t* const SFEM_RESTRICT          data) {             //

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    real_t hex8_f[8];
    real_t tet10_f[10];

    for (int q = 0; q < TET_QUAD_NQP; q++) {  // loop over the quadrature points

        real_t g_qx;
        real_t g_qy;
        real_t g_qz;

        tet10_transform_real_t(x, y, z, tet_qx[q], tet_qy[q], tet_qz[q], &g_qx, &g_qy, &g_qz);

        // tet10_dual_basis_hrt(tet_qx[q], tet_qy[q], tet_qz[q], tet10_f);
        tet10_Lagrange_basis(tet_qx[q], tet_qy[q], tet_qz[q], tet10_f);

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

        const real_t measure = tet10_measure_real_t(x, y, z, tet_qx[q], tet_qy[q], tet_qz[q]);
        const real_t dV      = measure * tet_qw[q];

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
}  // END hex8_to_isoparametric_tet10_resample_tet_adjoint

///////////////////////////////////////////////////////////////////////
// hex8_to_isoparametric_tet10_resample_field_refine_adjoint
///////////////////////////////////////////////////////////////////////
int                                                           //
hex8_to_isoparametric_tet10_resample_field_refine_adjoint(    //
        const ptrdiff_t                      start_element,   // Mesh
        const ptrdiff_t                      end_element,     //
        const ptrdiff_t                      nnodes,          //
        const idx_t** const SFEM_RESTRICT    elems,           //
        const geom_t** const SFEM_RESTRICT   xyz,             //
        const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
        const ptrdiff_t* const SFEM_RESTRICT stride,          //
        const geom_t* const SFEM_RESTRICT    origin,          //
        const geom_t* const SFEM_RESTRICT    delta,           //
        const real_t* const SFEM_RESTRICT    weighted_field,  // Input WF
        const real_t                         alpha_th,        // Threshold for alpha
        real_t* const SFEM_RESTRICT          data) {                   // Output SDF

    PRINT_CURRENT_FUNCTION;

    const real_t degenerated_tet_ratio = 2.5;  // TODO: make it a parameter

    int ret = 0;

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    const real_t hexahedron_volume = dx * dy * dz;

    struct tet10_vertices rTets[8];

    real_t alpha_max       = 0.0;
    real_t alpha_mim       = 1e9;
    int    refinements_cnt = 0;

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        idx_t ev[10];

        // ISOPARAMETRIC
        real_t x[10], y[10], z[10];

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
            x[v] = (real_t)(xyz[0][ev[v]]);  // x-coordinates
            y[v] = (real_t)(xyz[1][ev[v]]);  // y-coordinates
            z[v] = (real_t)(xyz[2][ev[v]]);  // z-coordinates
        }

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

        const real_t max_edge_len = tet10_edge_lengths(x, y, z, &vertex_a, &vertex_b, edges_length);

        const real_t alpha = max_edge_len / dx;

        alpha_max = alpha > alpha_max ? alpha : alpha_max;
        alpha_mim = alpha < alpha_mim ? alpha : alpha_mim;

        int n_tet = 0;

        real_t min_edge_length = edges_length[0];
        for (int i = 1; i < 6; i++) {
            min_edge_length = edges_length[i] < min_edge_length ? edges_length[i] : min_edge_length;
        }

        const real_t ratio_max_min   = max_edge_len / min_edge_length;
        const int    degenerated_tet = ratio_max_min > degenerated_tet_ratio ? 1 : 0;

        if (alpha > alpha_th && degenerated_tet == 0) {
            tet10_uniform_refinement(x, y, z, wf_tet10, rTets);
            n_tet = 8;
            refinements_cnt++;

        } else if (alpha > alpha_th && degenerated_tet == 1) {
            const int nt = tet10_refine_two_edge_vertex(x, y, z, wf_tet10, vertex_a, vertex_b, rTets);
            if (nt != 2) {
                fprintf(stderr, "tet10_refine_two_edge_vertex: %d != 2\n", nt);
                return -1;
            }
            n_tet = 2;
            refinements_cnt++;

        } else {
            n_tet = 1;
            for (int v = 0; v < 10; ++v) {
                rTets[0].x[v] = x[v];
                rTets[0].y[v] = y[v];
                rTets[0].z[v] = z[v];
                rTets[0].w[v] = wf_tet10[v];
            }
        }  // end if to select the refinement schema

        for (int ni = 0; ni < n_tet; ni++) {
            hex8_to_isoparametric_tet10_resample_tet_adjoint(rTets[ni].x,  //
                                                             rTets[ni].y,  //
                                                             rTets[ni].z,  //
                                                             rTets[ni].w,  //
                                                             stride,       //
                                                             origin,       //
                                                             delta,        //
                                                             data);        //
        }
    }

#if SFEM_LOG_LEVEL >= 5
    printf("============================================================\n");
    printf("alpha_max:       %g  \n", alpha_max);
    printf("alpha_mim:       %g  \n", alpha_mim);
    printf("refinements_cnt: %d  \n", refinements_cnt);
    printf("Total elements:  %ld \n", (end_element - start_element));
    printf("Refinement ratio %g  \n", (real_t)refinements_cnt / (real_t)(end_element - start_element));
    printf("============================================================\n");
#endif

    RETURN_FROM_FUNCTION(ret);
}  // END hex8_to_isoparametric_tet10_resample_field_refine_adjoint

///////////////////////////////////////////////////////////////////////
// insert_tet10_in_output_array
///////////////////////////////////////////////////////////////////////
int                                                                 //
insert_tet10_in_output_array(struct tet10_vertices*  tet10_head,    //
                             struct tet10_vertices** rTets_out,     //
                             int tets_size, size_t* tets_capacity,  //
                             const size_t tet_delta_capacity) {
    // Check if there is enough space in the output array
    // If not, allocate more space
    if (tets_size >= *tets_capacity) {
        *tets_capacity += tet_delta_capacity;
        *rTets_out = realloc(*rTets_out, sizeof(struct tet10_vertices) * (*tets_capacity));
        if (*rTets_out == NULL) {
            fprintf(stderr, "ERROR: realloc failed in insert_tet10_in_output_array: %s:%d\n", __FILE__, __LINE__);
            exit(1);
        }
    }

    // Copy the tet10_head to the output array
    memcpy(&(*rTets_out)[tets_size], tet10_head, sizeof(struct tet10_vertices));

    // Incrementa e restituisci il nuovo contatore
    return tets_size + 1;
}  // END insert_tet10_in_output_array

///////////////////////////////////////////////////////////////////////
// tet10_iterative_refinement
///////////////////////////////////////////////////////////////////////
int                                                                              //
tet10_iterative_refinement(const real_t* const           x,                      //
                           const real_t* const           y,                      //
                           const real_t* const           z,                      //
                           const real_t* const           w,                      //
                           const real_t                  dx,                     //
                           const real_t                  dy,                     //
                           const real_t                  dz,                     //
                           const real_t                  alpha_th,               //
                           const real_t                  degenerated_tet_ratio,  //
                           const int                     max_refined_tets,       // Maximum number of iterations
                           struct tet10_vertices** const rTets_out) {            //

    size_t       tets_capacity      = 8;
    const size_t tet_delta_capacity = 8;

    if (*rTets_out != NULL) {
        free(*rTets_out);
        *rTets_out = NULL;
    }

    *rTets_out = malloc(sizeof(struct tet10_vertices) * tets_capacity);

    int tets_size                  = 0;
    int flag_loop                  = 1;
    int n_tet                      = 0;
    int total_refined_tets         = 0;
    int degenerated_tetrahedra_cnt = 0;
    int refinements_cnt            = 0;

    struct sfem_stack* stack = sfem_stack_create(100);

    struct tet10_vertices* first_tet10 = malloc(sizeof(struct tet10_vertices));
    struct tet10_vertices  rTets[8];

    for (int i = 0; i < 10; i++) {
        first_tet10->x[i] = x[i];
        first_tet10->y[i] = y[i];
        first_tet10->z[i] = z[i];
        first_tet10->w[i] = w[i];
    }

    sfem_stack_push(stack, first_tet10);

    while (flag_loop == 1 && sfem_stack_size(stack) > 0) {
        struct tet10_vertices* tet10_head = sfem_stack_pop(stack);

        if (tet10_head == NULL) {
            fprintf(stderr, "tet10_iterative_refinement: tet10_head == NULL\n");
            exit(1);
            return -1;
        }

        real_type edges_length[6];

        int vertex_a = -1;
        int vertex_b = -1;

        const real_type max_edges_length =             //
                tet_edge_max_length(tet10_head->x[0],  //
                                    tet10_head->y[0],  //
                                    tet10_head->z[0],  //
                                    tet10_head->x[1],  //
                                    tet10_head->y[1],  //
                                    tet10_head->z[1],  //
                                    tet10_head->x[2],  //
                                    tet10_head->y[2],  //
                                    tet10_head->z[2],  //
                                    tet10_head->x[3],  //
                                    tet10_head->y[3],  //
                                    tet10_head->z[3],  //
                                    &vertex_a,         // Output
                                    &vertex_b,         // Output
                                    edges_length);     // Output

        const real_t alpha_tet           = max_edges_length / dx;
        const real_t max_min_edges_ratio = ratio_abs_max_min(edges_length, 6);

        int degenerated_tet = 0;

        if (max_min_edges_ratio > degenerated_tet_ratio && alpha_tet > alpha_th) {
            degenerated_tetrahedra_cnt++;
            degenerated_tet = 1;
        }

        if (alpha_tet <= alpha_th) {
            tets_size = insert_tet10_in_output_array(tet10_head,  //
                                                     rTets_out,
                                                     tets_size,
                                                     &tets_capacity,
                                                     tet_delta_capacity);

            if (tets_size >= max_refined_tets) {
                while (sfem_stack_size(stack) > 0) {
                    struct tet10_vertices* tet10_head_loc = sfem_stack_pop(stack);

                    tets_size = insert_tet10_in_output_array(tet10_head_loc,  //
                                                             rTets_out,
                                                             tets_size,
                                                             &tets_capacity,
                                                             tet_delta_capacity);
                    free(tet10_head_loc);
                }

                flag_loop = 0;
            }

        } else if (degenerated_tet == 1) {
            // The tetrahedron is degenerated and needs to be refined
            // By the bisection of the longest edge

            const int nt = tet10_refine_two_edge_vertex(tet10_head->x,  //
                                                        tet10_head->y,  //
                                                        tet10_head->z,  //
                                                        tet10_head->w,  //
                                                        vertex_a,       //
                                                        vertex_b,       //
                                                        rTets);         //

            if (nt != 2) {
                fprintf(stderr, "tet10_refine_two_edge_vertex: %d != 2, %s:%d\n", nt, __FILE__, __LINE__);
                exit(1);
                return -1;
            }

            n_tet = 2;
            refinements_cnt++;

            for (int ni = 0; ni < n_tet; ni++) {
                struct tet10_vertices* tet10_new = malloc(sizeof(struct tet10_vertices));
                memcpy(tet10_new, &rTets[ni], sizeof(struct tet10_vertices));
                sfem_stack_push(stack, tet10_new);
            }

        } else {
            // The tetrahedron is not degenerated and needs to be refined
            // By applying the uniform refinement.

            tet10_uniform_refinement(tet10_head->x,  //
                                     tet10_head->y,  //
                                     tet10_head->z,  //
                                     tet10_head->w,  //
                                     rTets);         //
            n_tet = 8;
            refinements_cnt++;

            for (int ni = 0; ni < n_tet; ni++) {
                struct tet10_vertices* tet10_new = malloc(sizeof(struct tet10_vertices));
                memcpy(tet10_new, &rTets[ni], sizeof(struct tet10_vertices));
                sfem_stack_push(stack, tet10_new);
            }
        }  // end if

        free(tet10_head);
        tet10_head = NULL;
    }  // end while

    /////////////////////////////////////////////////////
    // end and return

    sfem_stack_clear(stack);
    sfem_stack_destroy(stack);
    stack = NULL;

    return tets_size;
}  // END tet10_iterative_refinement

///////////////////////////////////////////////////////////////////////
// hex8_to_isoparametric_tet10_resample_field_iterative_ref_adjoint
///////////////////////////////////////////////////////////////////////
int                                                                                                                    //
hex8_to_isoparametric_tet10_resample_field_iterative_ref_adjoint(const ptrdiff_t                      start_element,   // Mesh
                                                                 const ptrdiff_t                      end_element,     //
                                                                 const ptrdiff_t                      nnodes,          //
                                                                 const idx_t** const SFEM_RESTRICT    elems,           //
                                                                 const geom_t** const SFEM_RESTRICT   xyz,             //
                                                                 const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                                                                 const ptrdiff_t* const SFEM_RESTRICT stride,          //
                                                                 const geom_t* const SFEM_RESTRICT    origin,          //
                                                                 const geom_t* const SFEM_RESTRICT    delta,           //
                                                                 const real_t* const SFEM_RESTRICT    weighted_field,  // Input WF
                                                                 const real_t                alpha_th,  // Threshold for alpha
                                                                 real_t* const SFEM_RESTRICT data) {    //

    PRINT_CURRENT_FUNCTION;

    const real_t degenerated_tet_ratio = 2.5;  // TODO: make it a parameter

    int ret = 0;

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    const real_t hexahedron_volume = dx * dy * dz;

    // real_t alpha_max       = 0.0;
    // real_t alpha_mim       = 1e9;
    // int    refinements_cnt = 0;

    int max_refinements_cnt = 0;

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        idx_t ev[10];

        // ISOPARAMETRIC
        real_t x[10], y[10], z[10];

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
            x[v] = (real_t)(xyz[0][ev[v]]);  // x-coordinates
            y[v] = (real_t)(xyz[1][ev[v]]);  // y-coordinates
            z[v] = (real_t)(xyz[2][ev[v]]);  // z-coordinates
        }

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

        // generate iterative refinement
        struct tet10_vertices* rTets_out = NULL;

        const int ref_tet10_cnt = tet10_iterative_refinement(x,                      //
                                                             y,                      //
                                                             z,                      //
                                                             wf_tet10,               //
                                                             dx,                     //
                                                             dy,                     //
                                                             dz,                     //
                                                             alpha_th,               //
                                                             degenerated_tet_ratio,  //
                                                             60,                     //
                                                             &rTets_out);            //

        max_refinements_cnt = ref_tet10_cnt > max_refinements_cnt ? ref_tet10_cnt : max_refinements_cnt;

        // printf("ref_tet10_cnt: %d\n", ref_tet10_cnt);

        // perform the adjoint on the refined mesh

        for (int ir = 0; ir < ref_tet10_cnt; ir++) {
            hex8_to_isoparametric_tet10_resample_tet_adjoint(rTets_out[ir].x,  //
                                                             rTets_out[ir].y,  //
                                                             rTets_out[ir].z,  //
                                                             rTets_out[ir].w,  //
                                                             stride,           //
                                                             origin,           //
                                                             delta,            //
                                                             data);            //

        }  // end for over refined tets
           //
        free(rTets_out);
        rTets_out = NULL;

    }  // end for over elements

    printf("max_refinements_cnt: %d, %s:%d\n", max_refinements_cnt, __FILE__, __LINE__);

    return 0;
}  // END hex8_to_isoparametric_tet10_resample_field_iterative_ref_adjoint