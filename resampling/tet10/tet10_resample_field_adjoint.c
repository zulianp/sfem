#include "sfem_resample_field_tet4_math.h"
#include "tet10_resample_field.h"
#include "tet10_resample_field_V2.h"

#include "quadratures_rule.h"
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

    return 0;
}

///////////////////////////////////////////////////////////////////////
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

        volume += volume_loc * V[ni];
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
                   real_t* const edge_lengths) {  //

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
        max_length      = len > max_length ? len : max_length;
    }

    return max_length;
}

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

        tet10_transform(x, y, z, tet_qx[q], tet_qy[q], tet_qz[q], &g_qx, &g_qy, &g_qz);
        tet10_dual_basis_hrt(tet_qx[q], tet_qy[q], tet_qz[q], tet10_f);

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

        const real_t measure = tet10_measure(x, y, z, tet_qx[q], tet_qy[q], tet_qz[q]);
        const real_t dV      = measure * tet_qw[q];

        // if (dV < 0) {
        //     fprintf(stderr, "warning: negative volume %g!\n", dV);
        //     continue;
        // }

        // if(wf_tet10[0] <  0) {
        //     fprintf(stderr, "warning: negative wf_tet10[0] %g!\n", wf_tet10[0]);
        //     continue;
        // }

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

        // if (It < -1e-9) {
        //     fprintf(stderr, "warning: negative integral %g!\n", It);
        //     continue;
        // }

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

    int ret = 0;

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    const real_t hexahedron_volume = dx * dy * dz;

    struct tet10_vertices rTets[8];

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

        real_t       edges_length[6];
        const real_t max_edge_len = tet10_edge_lengths(x, y, z, edges_length);
        const real_t alpha        = max_edge_len / dx;

        int n_tet = 0;

        if (alpha > alpha_th) {
            tet10_uniform_refinement(x, y, z, wf_tet10, rTets);
            n_tet = 8;
        } else {
            n_tet = 1;
            for (int v = 0; v < 10; ++v) {
                rTets[0].x[v] = x[v];
                rTets[0].y[v] = y[v];
                rTets[0].z[v] = z[v];
                rTets[0].w[v] = wf_tet10[v];
            }
        }

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

    RETURN_FROM_FUNCTION(ret);
}