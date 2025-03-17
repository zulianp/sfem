#include "sfem_resample_field.h"
#include "sfem_resample_field_tet4_math.h"

#include "mass.h"
// #include "read_mesh.h"
#include "matrixio_array.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

// #define real_t real_type

#include "quadratures_rule.h"

#define real_type real_t
#define SFEM_RESTRICT __restrict__

#define SFEM_RESAMPLE_GAP_DUAL

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_adjoint /////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                                         //
tet4_resample_tetrahedron_local_adjoint(const real_type                      x0,            // Tetrahedron vertices X-coordinates
                                        const real_type                      x1,            //
                                        const real_type                      x2,            //
                                        const real_type                      x3,            //
                                        const real_type                      y0,            // Tetrahedron vertices Y-coordinates
                                        const real_type                      y1,            //
                                        const real_type                      y2,            //
                                        const real_type                      y3,            //
                                        const real_type                      z0,            // Tetrahedron vertices Z-coordinates
                                        const real_type                      z1,            //
                                        const real_type                      z2,            //
                                        const real_type                      z3,            //
                                        const real_type                      theta_volume,  // Volume of the tetrahedron
                                        const real_type                      wf0,           // Weighted field at the vertices
                                        const real_type                      wf1,           //
                                        const real_type                      wf2,           //
                                        const real_type                      wf3,           //
                                        const real_type                      ox,            // Origin of the grid
                                        const real_type                      oy,            //
                                        const real_type                      oz,            //
                                        const real_type                      dx,            // Spacing of the grid
                                        const real_type                      dy,            //
                                        const real_type                      dz,            //
                                        const ptrdiff_t* const SFEM_RESTRICT stride,        // Stride
                                        const ptrdiff_t* const SFEM_RESTRICT n,             // Size of the grid
                                        real_type* const SFEM_RESTRICT       data) {              // Output

    for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i++) {  // loop over the quadrature points

        real_type g_qx, g_qy, g_qz;

        // Transform quadrature point to physical space
        // g_qx, g_qy, g_qz are the coordinates of the quadrature point in the physical space
        // of the tetrahedral element
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

#ifndef SFEM_RESAMPLE_GAP_DUAL
        // Standard basis function
        {
            tet4_f[0] = 1 - tet_qx[q] - tet_qy[q] - tet_qz[q];
            tet4_f[1] = tet_qx[q];
            tet4_f[2] = tet_qy[q];
            tet4_f[2] = tet_qz[q];
        }
#else

        real_type tet4_f0, tet4_f1, tet4_f2, tet4_f3;
        {
            // DUAL basis function (Shape functions for tetrahedral elements)
            // at the quadrature point
            const real_type f0 = 1.0 - tet_qx[quad_i] - tet_qy[quad_i] - tet_qz[quad_i];
            const real_type f1 = tet_qx[quad_i];
            const real_type f2 = tet_qy[quad_i];
            const real_type f3 = tet_qz[quad_i];

            // Values of the shape functions at the quadrature point
            // In the local coordinate system of the tetrahedral element
            // For each vertex of the tetrahedral element
            tet4_f0 = 4.0 * f0 - f1 - f2 - f3;
            tet4_f1 = -f0 + 4.0 * f1 - f2 - f3;
            tet4_f2 = -f0 - f1 + 4.0 * f2 - f3;
            tet4_f3 = -f0 - f1 - f2 + 4.0 * f3;
        }
#endif

        const real_type grid_x = (g_qx - ox) / dx;
        const real_type grid_y = (g_qy - oy) / dy;
        const real_type grid_z = (g_qz - oz) / dz;

        const ptrdiff_t i = floor(grid_x);
        const ptrdiff_t j = floor(grid_y);
        const ptrdiff_t k = floor(grid_z);

        // printf("i = %ld grid_x = %g\n", i, grid_x);
        // printf("j = %ld grid_y = %g\n", j, grid_y);
        // printf("k = %ld grid_z = %g\n", k, grid_z);

        // If outside the domain of the grid (i.e., the grid is not large enough)
        if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
            fprintf(stderr,
                    "WARNING: (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
                    "%ld)!\n",
                    g_qx,
                    g_qy,
                    g_qz,
                    i,
                    j,
                    k,
                    n[0],
                    n[1],
                    n[2]);
            exit(1);
        }

        // Get the reminder [0, 1]
        // The local coordinates of the quadrature point in the unit cube
        real_type l_x = (grid_x - (real_type)i);
        real_type l_y = (grid_y - (real_type)j);
        real_type l_z = (grid_z - (real_type)k);

        assert(l_x >= -1e-8);
        assert(l_y >= -1e-8);
        assert(l_z >= -1e-8);

        assert(l_x <= 1 + 1e-8);
        assert(l_y <= 1 + 1e-8);
        assert(l_z <= 1 + 1e-8);

        // Critical point
        // Compute the shape functions of the hexahedral (cubic) element
        // at the quadrature point

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

        // Integrate the values of the field at the vertices of the tetrahedral element
        const real_type dV = theta_volume * tet_qw[quad_i];
        const real_type It = (tet4_f0 * wf0 + tet4_f1 * wf1 + tet4_f2 * wf2 + tet4_f3 * wf3) * dV;

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
    }

    return 0;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_adjoint /////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                                     //
tet4_resample_field_local_adjoint(const ptrdiff_t                      start_element,   // Mesh
                                  const ptrdiff_t                      end_element,     //
                                  const ptrdiff_t                      nnodes,          //
                                  const idx_t** const SFEM_RESTRICT    elems,           //
                                  const geom_t** const SFEM_RESTRICT   xyz,             //
                                  const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                                  const ptrdiff_t* const SFEM_RESTRICT stride,          //
                                  const geom_t* const SFEM_RESTRICT    origin,          //
                                  const geom_t* const SFEM_RESTRICT    delta,           //
                                  const real_t* const SFEM_RESTRICT    weighted_field,  // Input weighted field
                                  real_t* const SFEM_RESTRICT          data) {                   // Output
                                                                                        //
    PRINT_CURRENT_FUNCTION;

    int ret = 0;

    const real_type ox = (real_type)origin[0];
    const real_type oy = (real_type)origin[1];
    const real_type oz = (real_type)origin[2];

    const real_type dx = (real_type)delta[0];
    const real_type dy = (real_type)delta[1];
    const real_type dz = (real_type)delta[2];

    const real_type hexahedron_volume = dx * dy * dz;

#if SFEM_LOG_LEVEL >= 5
    printf("============================================================\n");
    printf("Start: tet4_resample_field_local_adjoint  v2: %s:%d \n", __FILE__, __LINE__);
    printf("Heaxahedron volume = %g\n", hexahedron_volume);
    printf("============================================================\n");
#endif

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        // Vertices coordinates of the tetrahedron

        // loop over the 4 vertices of the tetrahedron
        idx_t ev[4];
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }

        // Read the coordinates of the vertices of the tetrahedron
        const real_type x0 = xyz[0][ev[0]];
        const real_type x1 = xyz[0][ev[1]];
        const real_type x2 = xyz[0][ev[2]];
        const real_type x3 = xyz[0][ev[3]];

        const real_type y0 = xyz[1][ev[0]];
        const real_type y1 = xyz[1][ev[1]];
        const real_type y2 = xyz[1][ev[2]];
        const real_type y3 = xyz[1][ev[3]];

        const real_type z0 = xyz[2][ev[0]];
        const real_type z1 = xyz[2][ev[1]];
        const real_type z2 = xyz[2][ev[2]];
        const real_type z3 = xyz[2][ev[3]];

        // Volume of the tetrahedron
        const real_type theta_volume = tet4_measure_v2(x0,
                                                       x1,
                                                       x2,
                                                       x3,
                                                       //
                                                       y0,
                                                       y1,
                                                       y2,
                                                       y3,
                                                       //
                                                       z0,
                                                       z1,
                                                       z2,
                                                       z3);

        const real_type ratio_Vth = theta_volume / hexahedron_volume;
        real_type       edges_length[6];
        tet_edge_length(x0,
                        y0,
                        z0,  //
                        x1,
                        y1,
                        z1,  //
                        x2,
                        y2,
                        z2,  //
                        x3,
                        y3,
                        z3,             //
                        edges_length);  //

        real_type max_edges_length = edges_length[0];
        for (int i = 1; i < 6; i++) {
            if (edges_length[i] > max_edges_length) {
                max_edges_length = edges_length[i];
            }
        }

        // printf("ratio_Vth: Volume of the tetrahedron / Volume of the hexahedron = %g\n", ratio_Vth);

        // printf("edges_length: %.2g, %.2g, %.2g, %.2g, %.2g, %.2g, hex side: %.2g; tet volume = %.2g, hex volume = %.2g\n",
        //        edges_length[0],
        //        edges_length[1],
        //        edges_length[2],
        //        edges_length[3],
        //        edges_length[4],
        //        edges_length[5],
        //        dx,
        //        theta_volume,
        //        hexahedron_volume);

        // printf("max_edges_length = %g, dx = %g, ratio %g (max_edges_length / dx)\n\n", max_edges_length, dx, (max_edges_length
        // / dx));

        const real_type wf0 = weighted_field[ev[0]];
        const real_type wf1 = weighted_field[ev[1]];
        const real_type wf2 = weighted_field[ev[2]];
        const real_type wf3 = weighted_field[ev[3]];

        // const real_type sampled_volume = hexahedron_volume * (real_type)(TET_QUAD_NQP);

        tet4_resample_tetrahedron_local_adjoint(x0,            // Tetrahedron vertices X-coordinates
                                                x1,            //
                                                x2,            //
                                                x3,            //
                                                y0,            // Tetrahedron vertices Y-coordinates
                                                y1,            //
                                                y2,            //
                                                y3,            //
                                                z0,            // Tetrahedron vertices Z-coordinates
                                                z1,            //
                                                z2,            //
                                                z3,            //
                                                theta_volume,  // Volume of the tetrahedron
                                                wf0,           // Weighted field at the vertices
                                                wf1,           //
                                                wf2,           //
                                                wf3,           //
                                                ox,            // Origin of the grid
                                                oy,            //
                                                oz,            //
                                                dx,            // Spacing of the grid
                                                dy,            //
                                                dz,            //
                                                stride,        // Stride
                                                n,             // Size of the grid
                                                data);         // Output

        // if (sampled_volume < 8.0 * theta_volume) {
        //     fprintf(stderr, "WARNING: sampled_volume < 8 * theta_volume: %g < %g\n", sampled_volume, 8.0 *
        // theta_volume);
        // }

    }  // end for i over elements

    RETURN_FROM_FUNCTION(ret);
}  // end tet4_resample_field_local_adjoint

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_refine_adjoint //////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                                            //
tet4_resample_field_local_refine_adjoint(const ptrdiff_t                      start_element,   // Mesh
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
                                         real_t* const SFEM_RESTRICT          data) {                   // Output
                                                                                               //
    PRINT_CURRENT_FUNCTION;

    int ret = 0;

    const real_type ox = (real_type)origin[0];
    const real_type oy = (real_type)origin[1];
    const real_type oz = (real_type)origin[2];

    const real_type dx = (real_type)delta[0];
    const real_type dy = (real_type)delta[1];
    const real_type dz = (real_type)delta[2];

    const real_type hexahedron_volume = dx * dy * dz;

#if SFEM_LOG_LEVEL >= 5
    printf("============================================================\n");
    printf("Start: %s: %s:%d \n", __FUNCTION__, __FILE__, __LINE__);
    printf("Heaxahedron volume = %g\n", hexahedron_volume);
    printf("============================================================\n");
#endif

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        // loop over the 4 vertices of the tetrahedron
        idx_t ev[4];
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }

        // Read the coordinates of the vertices of the tetrahedron
        const real_type x0 = xyz[0][ev[0]];
        const real_type x1 = xyz[0][ev[1]];
        const real_type x2 = xyz[0][ev[2]];
        const real_type x3 = xyz[0][ev[3]];

        const real_type y0 = xyz[1][ev[0]];
        const real_type y1 = xyz[1][ev[1]];
        const real_type y2 = xyz[1][ev[2]];
        const real_type y3 = xyz[1][ev[3]];

        const real_type z0 = xyz[2][ev[0]];
        const real_type z1 = xyz[2][ev[1]];
        const real_type z2 = xyz[2][ev[2]];
        const real_type z3 = xyz[2][ev[3]];

        // Compute the alpha_tet to decide if the tetrahedron is refined
        // Sides of the tetrahedron
        real_type edges_length[6];
        tet_edge_length(x0,  // Coordinates of the 1st vertex
                        y0,
                        z0,
                        x1,
                        y1,  // Coordinates of the 2nd vertex
                        z1,
                        x2,
                        y2,
                        z2,  // Coordinates of the 3rd vertex
                        x3,
                        y3,
                        z3,
                        edges_length);  // Output

        real_type max_edges_length = edges_length[0];
        for (int i = 1; i < 6; i++) {
            if (edges_length[i] > max_edges_length) {
                max_edges_length = edges_length[i];
            }
        }

        const real_type alpha_tet = max_edges_length / dx;
        //////////////////////////////////////////////

        // Volume of the tetrahedron
        const real_type theta_volume = tet4_measure_v2(x0,
                                                       x1,
                                                       x2,
                                                       x3,
                                                       //
                                                       y0,
                                                       y1,
                                                       y2,
                                                       y3,
                                                       //
                                                       z0,
                                                       z1,
                                                       z2,
                                                       z3);

        const real_type wf0 = weighted_field[ev[0]];
        const real_type wf1 = weighted_field[ev[1]];
        const real_type wf2 = weighted_field[ev[2]];
        const real_type wf3 = weighted_field[ev[3]];

        // const real_type sampled_volume = hexahedron_volume * (real_type)(TET_QUAD_NQP);

        tet4_resample_tetrahedron_local_adjoint(x0,            // Tetrahedron vertices X-coordinates
                                                x1,            //
                                                x2,            //
                                                x3,            //
                                                y0,            // Tetrahedron vertices Y-coordinates
                                                y1,            //
                                                y2,            //
                                                y3,            //
                                                z0,            // Tetrahedron vertices Z-coordinates
                                                z1,            //
                                                z2,            //
                                                z3,            //
                                                theta_volume,  // Volume of the tetrahedron
                                                wf0,           // Weighted field at the vertices
                                                wf1,           //
                                                wf2,           //
                                                wf3,           //
                                                ox,            // Origin of the grid
                                                oy,            //
                                                oz,            //
                                                dx,            // Spacing of the grid
                                                dy,            //
                                                dz,            //
                                                stride,        // Stride
                                                n,             // Size of the grid
                                                data);         // Output

        // if (sampled_volume < 8.0 * theta_volume) {
        //     fprintf(stderr, "WARNING: sampled_volume < 8 * theta_volume: %g < %g\n", sampled_volume, 8.0 *
        // theta_volume);
        // }

    }  // end for i over elements

    RETURN_FROM_FUNCTION(ret);
}  // end tet4_resample_field_local_refine_adjoint

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_adjoint /////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                          //
tet4_update_cnt_local_adjoint(const real_type                      x0,       // Tetrahedron vertices X-coordinates
                              const real_type                      x1,       //
                              const real_type                      x2,       //
                              const real_type                      x3,       //
                              const real_type                      y0,       // Tetrahedron vertices Y-coordinates
                              const real_type                      y1,       //
                              const real_type                      y2,       //
                              const real_type                      y3,       //
                              const real_type                      z0,       // Tetrahedron vertices Z-coordinates
                              const real_type                      z1,       //
                              const real_type                      z2,       //
                              const real_type                      z3,       //
                              const real_type                      ox,       // Origin of the grid
                              const real_type                      oy,       //
                              const real_type                      oz,       //
                              const real_type                      dx,       // Spacing of the grid
                              const real_type                      dy,       //
                              const real_type                      dz,       //
                              const ptrdiff_t* const SFEM_RESTRICT stride,   // Stride
                              const ptrdiff_t* const SFEM_RESTRICT n,        // Size of the grid
                              unsigned int* const SFEM_RESTRICT    data_cnt) {  // Output

    // Volume of the tetrahedron
    const real_type theta_volume = tet4_measure_v2(x0,
                                                   x1,
                                                   x2,
                                                   x3,
                                                   //
                                                   y0,
                                                   y1,
                                                   y2,
                                                   y3,
                                                   //
                                                   z0,
                                                   z1,
                                                   z2,
                                                   z3);

    for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i++) {  // loop over the quadrature points

        real_type g_qx, g_qy, g_qz;

        // Transform quadrature point to physical space
        // g_qx, g_qy, g_qz are the coordinates of the quadrature point in the physical space
        // of the tetrahedral element
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

        const real_type grid_x = (g_qx - ox) / dx;
        const real_type grid_y = (g_qy - oy) / dy;
        const real_type grid_z = (g_qz - oz) / dz;

        const ptrdiff_t i = floor(grid_x);
        const ptrdiff_t j = floor(grid_y);
        const ptrdiff_t k = floor(grid_z);

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

        // Update the data cnt
        data_cnt[i0] += 1;
        data_cnt[i1] += 1;
        data_cnt[i2] += 1;
        data_cnt[i3] += 1;
        data_cnt[i4] += 1;
        data_cnt[i5] += 1;
        data_cnt[i6] += 1;
        data_cnt[i7] += 1;
    }

    return 0;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_in_out_mesh_adjoint //////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                         //
tet4_cnt_mesh_adjoint(const ptrdiff_t                      start_element,   // Mesh
                      const ptrdiff_t                      end_element,     //
                      const ptrdiff_t                      nnodes,          //
                      const idx_t** const SFEM_RESTRICT    elems,           //
                      const geom_t** const SFEM_RESTRICT   xyz,             //
                      const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                      const ptrdiff_t* const SFEM_RESTRICT stride,          //
                      const geom_t* const SFEM_RESTRICT    origin,          //
                      const geom_t* const SFEM_RESTRICT    delta,           //
                      const real_t* const SFEM_RESTRICT    weighted_field,  // Input weighted field
                      unsigned int* const SFEM_RESTRICT    data_cnt) {         // Output
                                                                            //
    PRINT_CURRENT_FUNCTION;

    int ret = 0;

    const real_type ox = (real_type)origin[0];
    const real_type oy = (real_type)origin[1];
    const real_type oz = (real_type)origin[2];

    const real_type dx = (real_type)delta[0];
    const real_type dy = (real_type)delta[1];
    const real_type dz = (real_type)delta[2];

    const real_type hexahedron_volume = dx * dy * dz;

#if SFEM_LOG_LEVEL >= 5
    printf("============================================================\n");
    printf("Start: tet4_resample_field_local_adjoint  v2: %s:%d \n", __FILE__, __LINE__);
    printf("Heaxahedron volume = %g\n", hexahedron_volume);
    printf("============================================================\n");
#endif

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        // Vertices coordinates of the tetrahedron

        // loop over the 4 vertices of the tetrahedron
        idx_t ev[4];
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }

        // Read the coordinates of the vertices of the tetrahedron
        const real_type x0 = xyz[0][ev[0]];
        const real_type x1 = xyz[0][ev[1]];
        const real_type x2 = xyz[0][ev[2]];
        const real_type x3 = xyz[0][ev[3]];

        const real_type y0 = xyz[1][ev[0]];
        const real_type y1 = xyz[1][ev[1]];
        const real_type y2 = xyz[1][ev[2]];
        const real_type y3 = xyz[1][ev[3]];

        const real_type z0 = xyz[2][ev[0]];
        const real_type z1 = xyz[2][ev[1]];
        const real_type z2 = xyz[2][ev[2]];
        const real_type z3 = xyz[2][ev[3]];

        // Update the data cnt
        ret = tet4_update_cnt_local_adjoint(x0,         // Tetrahedron vertices X-coordinates
                                            x1,         //
                                            x2,         //
                                            x3,         //
                                            y0,         // Tetrahedron vertices Y-coordinates
                                            y1,         //
                                            y2,         //
                                            y3,         //
                                            z0,         // Tetrahedron vertices Z-coordinates
                                            z1,         //
                                            z2,         //
                                            z3,         //
                                            ox,         // Origin of the grid
                                            oy,         //
                                            oz,         //
                                            dx,         // Spacing of the grid
                                            dy,         //
                                            dz,         //
                                            stride,     // Stride
                                            n,          // Size of the grid
                                            data_cnt);  // Output

    }  // end for i over elements

    RETURN_FROM_FUNCTION(ret);
}

int                                                                           //
tet4_alpha_mesh_adjoint(const ptrdiff_t                      start_element,   // Mesh
                        const ptrdiff_t                      end_element,     //
                        const ptrdiff_t                      nnodes,          //
                        const idx_t** const SFEM_RESTRICT    elems,           //
                        const geom_t** const SFEM_RESTRICT   xyz,             //
                        const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                        const ptrdiff_t* const SFEM_RESTRICT stride,          //
                        const geom_t* const SFEM_RESTRICT    origin,          //
                        const geom_t* const SFEM_RESTRICT    delta,           //
                        const real_t* const SFEM_RESTRICT    weighted_field,  // Input weighted field
                        real_t* const SFEM_RESTRICT          alpha) {                  // Output
    PRINT_CURRENT_FUNCTION;

    int ret = 0;

    const real_type ox = (real_type)origin[0];
    const real_type oy = (real_type)origin[1];
    const real_type oz = (real_type)origin[2];

    const real_type dx = (real_type)delta[0];
    const real_type dy = (real_type)delta[1];
    const real_type dz = (real_type)delta[2];

    const real_type hexahedron_volume = dx * dy * dz;

#if SFEM_LOG_LEVEL >= 5
    printf("============================================================\n");
    printf("Start: tet4_resample_field_local_adjoint  v2: %s:%d \n", __FILE__, __LINE__);
    printf("Heaxahedron volume = %g\n", hexahedron_volume);
    printf("============================================================\n");
#endif

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        // Vertices coordinates of the tetrahedron

        // loop over the 4 vertices of the tetrahedron
        idx_t ev[4];
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }

        // Read the coordinates of the vertices of the tetrahedron
        const real_type x0 = xyz[0][ev[0]];
        const real_type x1 = xyz[0][ev[1]];
        const real_type x2 = xyz[0][ev[2]];
        const real_type x3 = xyz[0][ev[3]];

        const real_type y0 = xyz[1][ev[0]];
        const real_type y1 = xyz[1][ev[1]];
        const real_type y2 = xyz[1][ev[2]];
        const real_type y3 = xyz[1][ev[3]];

        const real_type z0 = xyz[2][ev[0]];
        const real_type z1 = xyz[2][ev[1]];
        const real_type z2 = xyz[2][ev[2]];
        const real_type z3 = xyz[2][ev[3]];

        // Sides of the tetrahedron
        real_type edges_length[6];
        tet_edge_length(x0,  // Coordinates of the 1st vertex
                        y0,
                        z0,
                        x1,
                        y1,  // Coordinates of the 2nd vertex
                        z1,
                        x2,
                        y2,
                        z2,  // Coordinates of the 3rd vertex
                        x3,
                        y3,
                        z3,
                        edges_length);  // Output

        real_type max_edges_length = edges_length[0];
        for (int i = 1; i < 6; i++) {
            if (edges_length[i] > max_edges_length) {
                max_edges_length = edges_length[i];
            }
        }

        real_type alpha_loc = max_edges_length / dx;

        for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i++) {  // loop over the quadrature points

            real_type g_qx, g_qy, g_qz;

            // Transform quadrature point to physical space
            // g_qx, g_qy, g_qz are the coordinates of the quadrature point in the physical space
            // of the tetrahedral element
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

            const real_type grid_x = (g_qx - ox) / dx;
            const real_type grid_y = (g_qy - oy) / dy;
            const real_type grid_z = (g_qz - oz) / dz;

            const ptrdiff_t i = floor(grid_x);
            const ptrdiff_t j = floor(grid_y);
            const ptrdiff_t k = floor(grid_z);

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

            // Update the data cnt
            alpha[i0] = alpha[i0] > alpha_loc ? alpha[i0] : alpha_loc;
            alpha[i1] = alpha[i1] > alpha_loc ? alpha[i1] : alpha_loc;
            alpha[i2] = alpha[i2] > alpha_loc ? alpha[i2] : alpha_loc;
            alpha[i3] = alpha[i3] > alpha_loc ? alpha[i3] : alpha_loc;
            alpha[i4] = alpha[i4] > alpha_loc ? alpha[i4] : alpha_loc;
            alpha[i5] = alpha[i5] > alpha_loc ? alpha[i5] : alpha_loc;
            alpha[i6] = alpha[i6] > alpha_loc ? alpha[i6] : alpha_loc;
            alpha[i7] = alpha[i7] > alpha_loc ? alpha[i7] : alpha_loc;
        }
    }  // end for i over elements

    RETURN_FROM_FUNCTION(ret);
}  // end tet4_alpha_mesh_adjoint

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_in_out_mesh_adjoint //////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                 //
is_same_side(const real_type px,    // Point X-coordinates
             const real_type py,    //       Y-coordinates
             const real_type pz,    //       Z-coordinates
             const real_type ax,    // Tet vertex a:  X-coordinates
             const real_type ay,    //                Y-coordinates
             const real_type az,    //                Z-coordinates
             const real_type bx,    // Tet vertex b:  X-coordinates
             const real_type by,    //                Y-coordinates
             const real_type bz,    //                Z-coordinates
             const real_type cx,    // Tet vertex c:  X-coordinates
             const real_type cy,    //                Y-coordinates
             const real_type cz,    //                Z-coordinates
             const real_type dx,    // Tet vertex d:  X-coordinates
             const real_type dy,    //                Y-coordinates
             const real_type dz) {  //                Z-coordinates

    // Compute vectors for the plane ABC
    real_type abx = bx - ax;
    real_type aby = by - ay;
    real_type abz = bz - az;
    real_type acx = cx - ax;
    real_type acy = cy - ay;
    real_type acz = cz - az;

    // Calculate cross product (normal vector of the plane)
    real_type nx = aby * acz - abz * acy;
    real_type ny = abz * acx - abx * acz;
    real_type nz = abx * acy - aby * acx;

    // Vector from A to D (opposite vertex)
    real_type adx = dx - ax;
    real_type ady = dy - ay;
    real_type adz = dz - az;

    // Dot product of normal with AD (direction from plane to D)
    real_type dotD = nx * adx + ny * ady + nz * adz;

    // Vector from A to P
    real_type apx = px - ax;
    real_type apy = py - ay;
    real_type apz = pz - az;

    // Dot product of normal with AP (direction from plane to P)
    real_type dotP = nx * apx + ny * apy + nz * apz;

    // Check if both dot products have the same sign (or zero)
    return (dotD * dotP >= 0.0);
}

// Function to check if a point is inside or on the boundary of a tetrahedron
int                                                 //
is_point_inside_tetrahedron(const real_type px,     //
                            const real_type py,     //
                            const real_type pz,     //
                            const real_type v1x,    //
                            const real_type v1y,    //
                            const real_type v1z,    //
                            const real_type v2x,    //
                            const real_type v2y,    //
                            const real_type v2z,    //
                            const real_type v3x,    //
                            const real_type v3y,    //
                            const real_type v3z,    //
                            const real_type v4x,    //
                            const real_type v4y,    //
                            const real_type v4z) {  //

    // Check against all four faces
    if (!is_same_side(px, py, pz, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z)) return 0;
    if (!is_same_side(px, py, pz, v1x, v1y, v1z, v2x, v2y, v2z, v4x, v4y, v4z, v3x, v3y, v3z)) return 0;
    if (!is_same_side(px, py, pz, v1x, v1y, v1z, v3x, v3y, v3z, v4x, v4y, v4z, v2x, v2y, v2z)) return 0;
    if (!is_same_side(px, py, pz, v2x, v2y, v2z, v3x, v3y, v3z, v4x, v4y, v4z, v1x, v1y, v1z)) return 0;

    return 1;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_field_in_out_mesh ////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                         //
tet4_field_in_out_mesh(const ptrdiff_t                      start_element,  // Mesh
                       const ptrdiff_t                      end_element,    //
                       const ptrdiff_t                      nnodes,         //
                       const idx_t** const SFEM_RESTRICT    elems,          //
                       const geom_t** const SFEM_RESTRICT   xyz,            //
                       const ptrdiff_t* const SFEM_RESTRICT n,              // SDF
                       const ptrdiff_t* const SFEM_RESTRICT stride,         //
                       const geom_t* const SFEM_RESTRICT    origin,         //
                       const geom_t* const SFEM_RESTRICT    delta,          //
                       BitArray*                            bit_array) {                               // Output

    PRINT_CURRENT_FUNCTION;

    int ret = 0;

    const real_type ox = (real_type)origin[0];
    const real_type oy = (real_type)origin[1];
    const real_type oz = (real_type)origin[2];

    const real_type dx = (real_type)delta[0];
    const real_type dy = (real_type)delta[1];
    const real_type dz = (real_type)delta[2];

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        // loop over the 4 vertices of the tetrahedron
        idx_t ev[4];
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }

        //
        const real_type x0 = xyz[0][ev[0]];
        const real_type x1 = xyz[0][ev[1]];
        const real_type x2 = xyz[0][ev[2]];
        const real_type x3 = xyz[0][ev[3]];

        const real_type y0 = xyz[1][ev[0]];
        const real_type y1 = xyz[1][ev[1]];
        const real_type y2 = xyz[1][ev[2]];
        const real_type y3 = xyz[1][ev[3]];

        const real_type z0 = xyz[2][ev[0]];
        const real_type z1 = xyz[2][ev[1]];
        const real_type z2 = xyz[2][ev[2]];
        const real_type z3 = xyz[2][ev[3]];

        // Define the bounding box of the tetrahedron
        const real_type xb_min = fmin(fmin(x0, x1), fmin(x2, x3));
        const real_type xb_max = fmax(fmax(x0, x1), fmax(x2, x3));

        const real_type yb_min = fmin(fmin(y0, y1), fmin(y2, y3));
        const real_type yb_max = fmax(fmax(y0, y1), fmax(y2, y3));

        const real_type zb_min = fmin(fmin(z0, z1), fmin(z2, z3));
        const real_type zb_max = fmax(fmax(z0, z1), fmax(z2, z3));

        const real_type grid_x_min = (xb_min - ox) / dx;
        const real_type grid_y_min = (yb_min - oy) / dy;
        const real_type grid_z_min = (zb_min - oz) / dz;

        const ptrdiff_t i_min = floor(grid_x_min);
        const ptrdiff_t j_min = floor(grid_y_min);
        const ptrdiff_t k_min = floor(grid_z_min);

        const real_type grid_x_max = (xb_max - ox) / dx;
        const real_type grid_y_max = (yb_max - oy) / dy;
        const real_type grid_z_max = (zb_max - oz) / dz;

        const ptrdiff_t i_max = ceil(grid_x_max);
        const ptrdiff_t j_max = ceil(grid_y_max);
        const ptrdiff_t k_max = ceil(grid_z_max);

        for (ptrdiff_t i = i_min; i <= i_max; i++) {
            for (ptrdiff_t j = j_min; j <= j_max; j++) {
                for (ptrdiff_t k = k_min; k <= k_max; k++) {
                    real_type x = ox + i * dx;
                    real_type y = oy + j * dy;
                    real_type z = oz + k * dz;

                    // Check if the point is inside the tetrahedron
                    if (is_point_inside_tetrahedron(x, y, z, x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3)) {
                        size_t index = i * stride[0] + j * stride[1] + k * stride[2];
                        set_bit(bit_array, index);
                    }  // end if

                }  // end for k
            }      // end for j
        }          // end for i

        //

    }  // end for i over elements

    RETURN_FROM_FUNCTION(ret);
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// in_out_field_mesh_tet4 ////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                                     //
in_out_field_mesh_tet4(const int                            mpi_size,   // MPI size
                       const int                            mpi_rank,   // MPI rank
                       const mesh_t* const SFEM_RESTRICT    mesh,       // Mesh: mesh_t struct
                       const ptrdiff_t* const SFEM_RESTRICT n,          // SDF: n[3]
                       const ptrdiff_t* const SFEM_RESTRICT stride,     // SDF: stride[3]
                       const geom_t* const SFEM_RESTRICT    origin,     // SDF: origin[3]
                       const geom_t* const SFEM_RESTRICT    delta,      // SDF: delta[3]
                       BitArray*                            bit_array,  // Output
                       sfem_resample_field_info*            info) {                // info
    PRINT_CURRENT_FUNCTION;

    int ret = 0;

    // set to zero the bit array
    to_zero(bit_array);

    ret = tet4_field_in_out_mesh(0,                // Mesh
                                 mesh->nelements,  //
                                 mesh->nnodes,     //
                                 mesh->elements,   //
                                 mesh->points,     //
                                 n,                // SDF
                                 stride,           //
                                 origin,           //
                                 delta,            //
                                 bit_array);       // Output

    RETURN_FROM_FUNCTION(ret);
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// in_out_field_mesh_tet4 ////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                         //
tet_uniform_refinement(const real_t               v1x,      //
                       const real_t               v1y,      //
                       const real_t               v1z,      //
                       const real_t               v2x,      //
                       const real_t               v2y,      //
                       const real_t               v2z,      //
                       const real_t               v3x,      //
                       const real_t               v3y,      //
                       const real_t               v3z,      //
                       const real_t               v4x,      //
                       const real_t               v4y,      //
                       const real_t               v4z,      //
                       const real_t               w1,       //
                       const real_t               w2,       //
                       const real_t               w3,       //
                       const real_t               w4,       //
                       struct tet_vertices* const rTets) {  //

    int ret = 0;

    // Using the method from the paper:
    // "Uniform Refinement of a Tetrahedron"
    //                     Elizabeth G. Ong
    //                         January 1991
    //                     CAM Report 91-01

    const real_t v5x = 0.5 * (v1x + v2x);
    const real_t v5y = 0.5 * (v1y + v2y);
    const real_t v5z = 0.5 * (v1z + v2z);
    const real_t v5w = 0.5 * (w1 + w2);

    const real_t v6x = 0.5 * (v2x + v4x);
    const real_t v6y = 0.5 * (v2y + v4y);
    const real_t v6z = 0.5 * (v2z + v4z);
    const real_t v6w = 0.5 * (w2 + w4);

    const real_t v7x = 0.5 * (v2x + v3x);
    const real_t v7y = 0.5 * (v2y + v3y);
    const real_t v7z = 0.5 * (v2z + v3z);
    const real_t v7w = 0.5 * (w2 + w3);

    const real_t v8x = 0.5 * (v1x + v4x);
    const real_t v8y = 0.5 * (v1y + v4y);
    const real_t v8z = 0.5 * (v1z + v4z);
    const real_t v8w = 0.5 * (w1 + w4);

    const real_t v9x = 0.5 * (v3x + v4x);
    const real_t v9y = 0.5 * (v3y + v4y);
    const real_t v9z = 0.5 * (v3z + v4z);
    const real_t v9w = 0.5 * (w3 + w4);

    const real_t v10x = 0.5 * (v1x + v3x);
    const real_t v10y = 0.5 * (v1y + v3y);
    const real_t v10z = 0.5 * (v1z + v3z);
    const real_t v10w = 0.5 * (w1 + w3);

    // First tetrahedron (v2, v5, v6, v7)
    rTets[0].x0 = v2x;
    rTets[0].y0 = v2y;
    rTets[0].z0 = v2z;
    rTets[0].w0 = w2;

    rTets[0].x1 = v5x;
    rTets[0].y1 = v5y;
    rTets[0].z1 = v5z;
    rTets[0].w1 = v5w;

    rTets[0].x2 = v6x;
    rTets[0].y2 = v6y;
    rTets[0].z2 = v6z;
    rTets[0].w2 = v6w;

    rTets[0].x3 = v7x;
    rTets[0].y3 = v7y;
    rTets[0].z3 = v7z;
    rTets[0].w3 = v7w;

    // Second tetrahedron (v1, v5, v8, v10)
    rTets[1].x0 = v1x;
    rTets[1].y0 = v1y;
    rTets[1].z0 = v1z;
    rTets[1].w0 = w1;

    rTets[1].x1 = v5x;
    rTets[1].y1 = v5y;
    rTets[1].z1 = v5z;
    rTets[1].w1 = v5w;

    rTets[1].x2 = v8x;
    rTets[1].y2 = v8y;
    rTets[1].z2 = v8z;
    rTets[1].w2 = v8w;

    rTets[1].x3 = v10x;
    rTets[1].y3 = v10y;
    rTets[1].z3 = v10z;
    rTets[1].w3 = v10w;

    // Third tetrahedron (v4, v6, v8, v9)
    rTets[2].x0 = v4x;
    rTets[2].y0 = v4y;
    rTets[2].z0 = v4z;
    rTets[2].w0 = w4;

    rTets[2].x1 = v6x;
    rTets[2].y1 = v6y;
    rTets[2].z1 = v6z;
    rTets[2].w1 = v6w;

    rTets[2].x2 = v8x;
    rTets[2].y2 = v8y;
    rTets[2].z2 = v8z;
    rTets[2].w2 = v8w;

    rTets[2].x3 = v9x;
    rTets[2].y3 = v9y;
    rTets[2].z3 = v9z;
    rTets[2].w3 = v9w;

    // Fourth tetrahedron (v3, v7, v9, v10)
    rTets[3].x0 = v3x;
    rTets[3].y0 = v3y;
    rTets[3].z0 = v3z;
    rTets[3].w0 = w3;

    rTets[3].x1 = v7x;
    rTets[3].y1 = v7y;
    rTets[3].z1 = v7z;
    rTets[3].w1 = v7w;

    rTets[3].x2 = v9x;
    rTets[3].y2 = v9y;
    rTets[3].z2 = v9z;
    rTets[3].w2 = v9w;

    rTets[3].x3 = v10x;
    rTets[3].y3 = v10y;
    rTets[3].z3 = v10z;
    rTets[3].w3 = v10w;

    // Fifth tetrahedron (v5, v6, v8, v9)
    rTets[4].x0 = v5x;
    rTets[4].y0 = v5y;
    rTets[4].z0 = v5z;
    rTets[4].w0 = v5w;

    rTets[4].x1 = v6x;
    rTets[4].y1 = v6y;
    rTets[4].z1 = v6z;
    rTets[4].w1 = v6w;

    rTets[4].x2 = v8x;
    rTets[4].y2 = v8y;
    rTets[4].z2 = v8z;
    rTets[4].w2 = v8w;

    rTets[4].x3 = v9x;
    rTets[4].y3 = v9y;
    rTets[4].z3 = v9z;
    rTets[4].w3 = v9w;

    // Sixth tetrahedron (v5, v6, v7, v9)
    rTets[5].x0 = v5x;
    rTets[5].y0 = v5y;
    rTets[5].z0 = v5z;
    rTets[5].w0 = v5w;

    rTets[5].x1 = v6x;
    rTets[5].y1 = v6y;
    rTets[5].z1 = v6z;
    rTets[5].w1 = v6w;

    rTets[5].x2 = v7x;
    rTets[5].y2 = v7y;
    rTets[5].z2 = v7z;
    rTets[5].w2 = v7w;

    rTets[5].x3 = v9x;
    rTets[5].y3 = v9y;
    rTets[5].z3 = v9z;
    rTets[5].w3 = v9w;

    // Seventh tetrahedron (v5, v7, v9, v10)
    rTets[6].x0 = v5x;
    rTets[6].y0 = v5y;
    rTets[6].z0 = v5z;
    rTets[6].w0 = v5w;

    rTets[6].x1 = v7x;
    rTets[6].y1 = v7y;
    rTets[6].z1 = v7z;
    rTets[6].w1 = v7w;

    rTets[6].x2 = v9x;
    rTets[6].y2 = v9y;
    rTets[6].z2 = v9z;
    rTets[6].w2 = v9w;

    rTets[6].x3 = v10x;
    rTets[6].y3 = v10y;
    rTets[6].z3 = v10z;
    rTets[6].w3 = v10w;

    // Eighth tetrahedron (v5, v7, v9, v10)
    rTets[7].x0 = v5x;
    rTets[7].y0 = v5y;
    rTets[7].z0 = v5z;
    rTets[7].w0 = v5w;

    rTets[7].x1 = v7x;
    rTets[7].y1 = v7y;
    rTets[7].z1 = v7z;
    rTets[7].w1 = v7w;

    rTets[7].x2 = v9x;
    rTets[7].y2 = v9y;
    rTets[7].z2 = v9z;
    rTets[7].w2 = v9w;

    rTets[7].x3 = v10x;
    rTets[7].y3 = v10y;
    rTets[7].z3 = v10z;
    rTets[7].w3 = v10w;

    return ret;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// volume_tet_array //////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
real_t                                                   //
volume_tet_array(const struct tet_vertices* const tets,  // Array of tetrahedra
                 const int                        n,     // Number of tetrahedra
                 real_t* const                    V) {                      // Output

    real_t tot_volume = 0.0;

    for (int tet_i = 0; tet_i < n; tet_i++) {
        const real_t volume = tet4_measure_v2(tets[tet_i].x0,
                                              tets[tet_i].x1,
                                              tets[tet_i].x2,
                                              tets[tet_i].x3,
                                              //
                                              tets[tet_i].y0,
                                              tets[tet_i].y1,
                                              tets[tet_i].y2,
                                              tets[tet_i].y3,
                                              //
                                              tets[tet_i].z0,
                                              tets[tet_i].z1,
                                              tets[tet_i].z2,
                                              tets[tet_i].z3);

        tot_volume += volume;
        V[tet_i] = volume;
    }

    return tot_volume;
}

real_t point_distance(const real_t x0,    //
                      const real_t y0,    //
                      const real_t z0,    //
                      const real_t x1,    //
                      const real_t y1,    //
                      const real_t z1) {  //
    const real_t dx = x1 - x0;
    const real_t dy = y1 - y0;
    const real_t dz = z1 - z0;

    return sqrt(dx * dx + dy * dy + dz * dz);
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet_edge_length ///////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                           //
tet_edge_length(const real_t  v0x,            //
                const real_t  v0y,            //
                const real_t  v0z,            //
                const real_t  v1x,            //
                const real_t  v1y,            //
                const real_t  v1z,            //
                const real_t  v2x,            //
                const real_t  v2y,            //
                const real_t  v2z,            //
                const real_t  v3x,            //
                const real_t  v3y,            //
                const real_t  v3z,            //
                real_t* const edge_length) {  // Output

    // Edge 0 (v0, v1)
    edge_length[0] = point_distance(v0x, v0y, v0z, v1x, v1y, v1z);

    // Edge 1 (v0, v2)
    edge_length[1] = point_distance(v0x, v0y, v0z, v2x, v2y, v2z);

    // Edge 2 (v0, v3)
    edge_length[2] = point_distance(v0x, v0y, v0z, v3x, v3y, v3z);

    // Edge 3 (v1, v2)
    edge_length[3] = point_distance(v1x, v1y, v1z, v2x, v2y, v2z);

    // Edge 4 (v1, v3)
    edge_length[4] = point_distance(v1x, v1y, v1z, v3x, v3y, v3z);

    // Edge 5 (v2, v3)
    edge_length[5] = point_distance(v2x, v2y, v2z, v3x, v3y, v3z);

    return 0;
}