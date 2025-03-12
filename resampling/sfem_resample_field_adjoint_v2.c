#include "sfem_resample_field.h"
#include "sfem_resample_field_tet4_math.h"

#include "mass.h"
// #include "read_mesh.h"
#include "matrixio_array.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

// #define real_t double

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
        tet4_transform_v2(x0,               // x-coordinates of the vertices
                          x1,               //
                          x2,               //
                          x3,               //
                          y0,               // y-coordinates of the vertices
                          y1,               //
                          y2,               //
                          y3,               //
                          z0,               // z-coordinates of the vertices
                          z1,               //
                          z2,               //
                          z3,               //
                          tet4_qx[quad_i],  // Quadrature point
                          tet4_qy[quad_i],  //
                          tet4_qz[quad_i],  //
                          &g_qx,            // Output coordinates
                          &g_qy,            //
                          &g_qz);           //

#ifndef SFEM_RESAMPLE_GAP_DUAL
        // Standard basis function
        {
            tet4_f[0] = 1 - tet4_qx[q] - tet4_qy[q] - tet4_qz[q];
            tet4_f[1] = tet4_qx[q];
            tet4_f[2] = tet4_qy[q];
            tet4_f[2] = tet4_qz[q];
        }
#else

        real_type tet4_f0, tet4_f1, tet4_f2, tet4_f3;
        {
            // DUAL basis function (Shape functions for tetrahedral elements)
            // at the quadrature point
            const real_type f0 = 1.0 - tet4_qx[quad_i] - tet4_qy[quad_i] - tet4_qz[quad_i];
            const real_type f1 = tet4_qx[quad_i];
            const real_type f2 = tet4_qy[quad_i];
            const real_type f3 = tet4_qz[quad_i];

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
        real_type l_x = (grid_x - (double)i);
        real_type l_y = (grid_y - (double)j);
        real_type l_z = (grid_z - (double)k);

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
        const real_type dV = theta_volume * tet4_qw[quad_i];
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

        const real_type wf0 = weighted_field[ev[0]];
        const real_type wf1 = weighted_field[ev[1]];
        const real_type wf2 = weighted_field[ev[2]];
        const real_type wf3 = weighted_field[ev[3]];

        // const real_type sampled_volume = hexahedron_volume * (double)(TET_QUAD_NQP);

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
        tet4_transform_v2(x0,               // x-coordinates of the vertices
                          x1,               //
                          x2,               //
                          x3,               //
                          y0,               // y-coordinates of the vertices
                          y1,               //
                          y2,               //
                          y3,               //
                          z0,               // z-coordinates of the vertices
                          z1,               //
                          z2,               //
                          z3,               //
                          tet4_qx[quad_i],  // Quadrature point
                          tet4_qy[quad_i],  //
                          tet4_qz[quad_i],  //
                          &g_qx,            // Output coordinates
                          &g_qy,            //
                          &g_qz);           //

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

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_in_out_mesh_adjoint //////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                        //
is_same_side(double px,    // Point X-coordinates
             double py,    //       Y-coordinates
             double pz,    //       Z-coordinates
             double ax,    // Tet vertex a:  X-coordinates
             double ay,    //                Y-coordinates
             double az,    //                Z-coordinates
             double bx,    // Tet vertex b:  X-coordinates
             double by,    //                Y-coordinates
             double bz,    //                Z-coordinates
             double cx,    // Tet vertex c:  X-coordinates
             double cy,    //                Y-coordinates
             double cz,    //                Z-coordinates
             double dx,    // Tet vertex d:  X-coordinates
             double dy,    //                Y-coordinates
             double dz) {  //                Z-coordinates

    // Compute vectors for the plane ABC
    double abx = bx - ax;
    double aby = by - ay;
    double abz = bz - az;
    double acx = cx - ax;
    double acy = cy - ay;
    double acz = cz - az;

    // Calculate cross product (normal vector of the plane)
    double nx = aby * acz - abz * acy;
    double ny = abz * acx - abx * acz;
    double nz = abx * acy - aby * acx;

    // Vector from A to D (opposite vertex)
    double adx = dx - ax;
    double ady = dy - ay;
    double adz = dz - az;

    // Dot product of normal with AD (direction from plane to D)
    double dotD = nx * adx + ny * ady + nz * adz;

    // Vector from A to P
    double apx = px - ax;
    double apy = py - ay;
    double apz = pz - az;

    // Dot product of normal with AP (direction from plane to P)
    double dotP = nx * apx + ny * apy + nz * apz;

    // Check if both dot products have the same sign (or zero)
    return (dotD * dotP >= 0.0);
}

// Function to check if a point is inside or on the boundary of a tetrahedron
int                                        //
is_point_inside_tetrahedron(double px,     //
                            double py,     //
                            double pz,     //
                            double v1x,    //
                            double v1y,    //
                            double v1z,    //
                            double v2x,    //
                            double v2y,    //
                            double v2z,    //
                            double v3x,    //
                            double v3y,    //
                            double v3z,    //
                            double v4x,    //
                            double v4y,    //
                            double v4z) {  //

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