#include <math.h>

#include "matrixio_array.h"
#include "sfem_resample_field.h"

#include "mass.h"
#include "mesh_aura.h"
#include "quadratures_rule.h"
#include "sfem_defs.h"
#include "sfem_inv_resample_field.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

real_t                  //
min4(const real_t a,    //
     const real_t b,    //
     const real_t c,    //
     const real_t d) {  //

    return MIN(MIN(a, b), MIN(c, d));
}

real_t                  //
max4(const real_t a,    //
     const real_t b,    //
     const real_t c,    //
     const real_t d) {  //

    return MAX(MAX(a, b), MAX(c, d));
}

real_t                        //
abs_real_t(const real_t x) {  //
    return x < 0 ? -x : x;
}

real_t                               //
shape_function(const real_t x1,      // 1st vertex X
               const real_t y1,      // 1st vertex Y
               const real_t z1,      // 1st vertex Z
               const real_t x2,      // 2nd vertex X
               const real_t y2,      // 2nd vertex Y
               const real_t z2,      // 2nd vertex Z
               const real_t x3,      // 3rd vertex X
               const real_t y3,      // 3rd vertex Y
               const real_t z3,      // 3rd vertex Z
               const real_t x4,      // 4th vertex X
               const real_t y4,      // 4th vertex Y
               const real_t z4,      // 4th vertex Z
               const real_t x,       // X Interpolation point
               const real_t y,       // Y Interpolation point
               const real_t z,       // Z Interpolation point
               const real_t val1,    // Value at 1st vertex
               const real_t val2,    // Value at 2nd vertex
               const real_t val3,    // Value at 3rd vertex
               const real_t val4) {  // Value at 4th vertex

    // 6 * Volume of the tetrahedron (Determinant of the Jacobian)
    const real_t V6 =
            (-(x2 * y3 * z1) + x2 * y4 * z1 + x1 * y3 * z2 - x1 * y4 * z2 + x2 * y1 * z3 - x1 * y2 * z3 + x1 * y4 * z3 -
             x2 * y4 * z3 + x4 * (-(y2 * z1) + y3 * z1 + y1 * z2 - y3 * z2 - y1 * z3 + y2 * z3) +
             (-(x2 * y1) + x1 * y2 - x1 * y3 + x2 * y3) * z4 + x3 * (-(y4 * z1) - y1 * z2 + y4 * z2 + y2 * (z1 - z4) + y1 * z4));

    const real_t a1 = (y4 * (-z1 + z2) + y2 * (z1 - z4) + y1 * (-z2 + z4)) / V6;
    const real_t b1 = (x4 * (z1 - z2) + x1 * (z2 - z4) + x2 * (-z1 + z4)) / V6;
    const real_t c1 = (x4 * (-y1 + y2) + x2 * (y1 - y4) + x1 * (-y2 + y4)) / V6;
    const real_t d1 = (x4 * y2 * z1 - x2 * y4 * z1 - x4 * y1 * z2 + x1 * y4 * z2 + x2 * y1 * z4 - x1 * y2 * z4) / V6;

    const real_t a2 = (y4 * (z1 - z3) + y1 * (z3 - z4) + y3 * (-z1 + z4)) / V6;
    const real_t b2 = (x4 * (-z1 + z3) + x3 * (z1 - z4) + x1 * (-z3 + z4)) / V6;
    const real_t c2 = (x4 * (y1 - y3) + x1 * (y3 - y4) + x3 * (-y1 + y4)) / V6;
    const real_t d2 = (x4 * y3 * z1 - x3 * y4 * z1 - x4 * y1 * z3 + x1 * y4 * z3 + x3 * y1 * z4 - x1 * y3 * z4) / V6;

    const real_t a3 = (y4 * (-z1 + z2) + y2 * (z1 - z4) + y1 * (-z2 + z4)) / V6;
    const real_t b3 = (x4 * (z1 - z2) + x1 * (z2 - z4) + x2 * (-z1 + z4)) / V6;
    const real_t c3 = (x4 * (-y1 + y2) + x2 * (y1 - y4) + x1 * (-y2 + y4)) / V6;
    const real_t d3 = (x4 * y2 * z1 - x2 * y4 * z1 - x4 * y1 * z2 + x1 * y4 * z2 + x2 * y1 * z4 - x1 * y2 * z4) / V6;

    const real_t a4 = (y3 * (-z1 + z2) + y2 * (z1 - z3) + y1 * (-z2 + z3)) / V6;
    const real_t b4 = (x3 * (z1 - z2) + x1 * (z2 - z3) + x2 * (-z1 + z3)) / V6;
    const real_t c4 = (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) / V6;
    const real_t d4 = (x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2 + x1 * y3 * z2 + x2 * y1 * z3 - x1 * y2 * z3) / V6;

    const real_t f1 = (a1 * x + b1 * y + c1 * z + d1) * val1;
    const real_t f2 = (a2 * x + b2 * y + c2 * z + d2) * val2;
    const real_t f3 = (a3 * x + b3 * y + c3 * z + d3) * val3;
    const real_t f4 = (a4 * x + b4 * y + c4 * z + d4) * val4;

    return f1 + f2 + f3 + f4;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// tetrahedron_volume //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
real_t                                 //
tetrahedron_volume(const real_t x0,    // 1st vertex X
                   const real_t y0,    // 1st vertex Y
                   const real_t z0,    // 1st vertex Z
                   const real_t x1,    // 2nd vertex X
                   const real_t y1,    // 2nd vertex Y
                   const real_t z1,    // 2nd vertex Z
                   const real_t x2,    // 3rd vertex X
                   const real_t y2,    // 3rd vertex Y
                   const real_t z2,    // 3rd vertex Z
                   const real_t x3,    // 4th vertex X
                   const real_t y3,    // 4th vertex Y
                   const real_t z3) {  // 4th vertex Z

    // determinant of the Jacobian
    //     [x0, y0, z0, 1]
    // M = [x1, y1, z1, 1]
    //     [x2, y2, z2, 1]
    //     [x3, y3, z3, 1]
    //
    // V = (1/6) * det(M)

    const real_t f1_6 = 1.0 / 6.0;

    return f1_6 * abs_real_t(-(x1 * y2 * z0) + x1 * y3 * z0 + x0 * y2 * z1 - x0 * y3 * z1 + x1 * y0 * z2 - x0 * y1 * z2 +
                             x0 * y3 * z2 - x1 * y3 * z2 + x3 * (-(y1 * z0) + y2 * z0 + y0 * z1 - y2 * z1 - y0 * z2 + y1 * z2) +
                             (-(x1 * y0) + x0 * y1 - x0 * y2 + x1 * y2) * z3 +
                             x2 * (-(y3 * z0) - y0 * z1 + y3 * z1 + y1 * (z0 - z3) + y0 * z3));
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// check_inside_testreadnum ///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
int                                            //
check_p_inside_tetrahedron(const real_t px,    //
                           const real_t py,    //
                           const real_t pz,    //
                           const real_t x0,    //
                           const real_t y0,    //
                           const real_t z0,    //
                           const real_t x1,    //
                           const real_t y1,    //
                           const real_t z1,    //
                           const real_t x2,    //
                           const real_t y2,    //
                           const real_t z2,    //
                           const real_t x3,    //
                           const real_t y3,    //
                           const real_t z3) {  //

    /**
     * see:
     * https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not/51733522#51733522
     *
     * This is the Mathematica manipulation of the above method.
     */

    const double den13 =
            (x1 * y2 * z0 - x1 * y3 * z0 - x0 * y2 * z1 + x0 * y3 * z1 - x1 * y0 * z2 + x0 * y1 * z2 - x0 * y3 * z2 +
             x1 * y3 * z2 + x3 * (y1 * z0 - y2 * z0 - y0 * z1 + y2 * z1 + y0 * z2 - y1 * z2) + x1 * y0 * z3 - x0 * y1 * z3 +
             x0 * y2 * z3 - x1 * y2 * z3 + x2 * (y3 * z0 + y0 * z1 - y3 * z1 - y0 * z3 + y1 * (-z0 + z3)));

    const double r1 =
            (pz * (-(x3 * y0) - x0 * y2 + x3 * y2 + x2 * (y0 - y3) + x0 * y3) + px * y2 * z0 - x3 * y2 * z0 - px * y3 * z0 +
             x2 * y3 * z0 - px * y0 * z2 + x3 * y0 * z2 + px * y3 * z2 - x0 * y3 * z2 + px * y0 * z3 - x2 * y0 * z3 -
             px * y2 * z3 + x0 * y2 * z3 + py * (-(x2 * z0) + x3 * z0 + x0 * z2 - x3 * z2 - x0 * z3 + x2 * z3)) /
            den13;

    const double r2 =
            (pz * (-(x3 * y0) - x0 * y1 + x3 * y1 + x1 * (y0 - y3) + x0 * y3) + px * y1 * z0 - x3 * y1 * z0 - px * y3 * z0 +
             x1 * y3 * z0 - px * y0 * z1 + x3 * y0 * z1 + px * y3 * z1 - x0 * y3 * z1 + px * y0 * z3 - x1 * y0 * z3 -
             px * y1 * z3 + x0 * y1 * z3 + py * (-(x1 * z0) + x3 * z0 + x0 * z1 - x3 * z1 - x0 * z3 + x1 * z3)) /
            (-(x1 * y2 * z0) + x1 * y3 * z0 + x0 * y2 * z1 - x0 * y3 * z1 + x1 * y0 * z2 - x0 * y1 * z2 + x0 * y3 * z2 -
             x1 * y3 * z2 + x3 * (-(y1 * z0) + y2 * z0 + y0 * z1 - y2 * z1 - y0 * z2 + y1 * z2) +
             (-(x1 * y0) + x0 * y1 - x0 * y2 + x1 * y2) * z3 + x2 * (-(y3 * z0) - y0 * z1 + y3 * z1 + y1 * (z0 - z3) + y0 * z3));

    const double r3 =
            (pz * (-(x2 * y0) - x0 * y1 + x2 * y1 + x1 * (y0 - y2) + x0 * y2) + px * y1 * z0 - x2 * y1 * z0 - px * y2 * z0 +
             x1 * y2 * z0 - px * y0 * z1 + x2 * y0 * z1 + px * y2 * z1 - x0 * y2 * z1 + px * y0 * z2 - x1 * y0 * z2 -
             px * y1 * z2 + x0 * y1 * z2 + py * (-(x1 * z0) + x2 * z0 + x0 * z1 - x2 * z1 - x0 * z2 + x1 * z2)) /
            den13;

    const double same_side = ((x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3) * (pz - z0) +
                              (py - y0) * (-(x2 * z1) + x3 * z1 + x1 * z2 - x3 * z2 - x1 * z3 + x2 * z3) +
                              (px - x0) * (y2 * z1 - y3 * z1 - y1 * z2 + y3 * z2 + y1 * z3 - y2 * z3)) *
                             ((x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3) * (-z0 + z2) +
                              (-y0 + y2) * (-(x2 * z1) + x3 * z1 + x1 * z2 - x3 * z2 - x1 * z3 + x2 * z3) +
                              (-x0 + x2) * (y2 * z1 - y3 * z1 - y1 * z2 + y3 * z2 + y1 * z3 - y2 * z3));

    return same_side > 0.0 && r1 >= 0.0 && r1 <= 1.0 && r2 >= 0.0 && r2 <= 1.0 && r3 >= 0.0 && r3 <= 1.0;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// tet4_inv_resample_field_local //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
int                                                                                 //
tet4_inv_resample_field_local(const ptrdiff_t                      start_element,   // Mesh
                              const ptrdiff_t                      end_element,     // Mesh
                              const ptrdiff_t                      nnodes,          // Mesh
                              const idx_t** const SFEM_RESTRICT    elems,           // Mesh
                              const geom_t** const SFEM_RESTRICT   xyz,             // Mesh
                              const real_t* const SFEM_RESTRICT    weighted_field,  // Input (weighted field)
                              const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                              const ptrdiff_t* const SFEM_RESTRICT stride,          // SDF
                              const geom_t* const SFEM_RESTRICT    origin,          // SDF
                              const geom_t* const SFEM_RESTRICT    delta,           // SDF
                              real_t* const SFEM_RESTRICT          data) {                   // SDF: Output
    PRINT_CURRENT_FUNCTION;

    int ret = 0;

    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    for (int element_id = start_element; element_id < end_element; element_id++) {
        //
        real_t x0 = 0.0, x1 = 0.0, x2 = 0.0, x3 = 0.0;
        real_t y0 = 0.0, y1 = 0.0, y2 = 0.0, y3 = 0.0;
        real_t z0 = 0.0, z1 = 0.0, z2 = 0.0, z3 = 0.0;

        real_t weighted_field_0 = 0.0;
        real_t weighted_field_1 = 0.0;
        real_t weighted_field_2 = 0.0;
        real_t weighted_field_3 = 0.0;

        {  // Collect the vertices of the tetrahedron
           // And its weighted field
            idx_t ev[4];
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][element_id];
            }

            x0 = xyz[0][ev[0]];
            x1 = xyz[0][ev[1]];
            x2 = xyz[0][ev[2]];
            x3 = xyz[0][ev[3]];

            y0 = xyz[1][ev[0]];
            y1 = xyz[1][ev[1]];
            y2 = xyz[1][ev[2]];
            y3 = xyz[1][ev[3]];

            z0 = xyz[2][ev[0]];
            z1 = xyz[2][ev[1]];
            z2 = xyz[2][ev[2]];
            z3 = xyz[2][ev[3]];

            weighted_field_0 = weighted_field[ev[0]];
            weighted_field_1 = weighted_field[ev[1]];
            weighted_field_2 = weighted_field[ev[2]];
            weighted_field_3 = weighted_field[ev[3]];
        }  // End of collecting the vertices of the tetrahedron

        // Get the bounding box of the tetrahedron in the grid
        // Min indices of the tetrahedron vertices
        real_t min_thv_x = min4(x0, x1, x2, x3);
        real_t min_thv_y = min4(y0, y1, y2, y3);
        real_t min_thv_z = min4(z0, z1, z2, z3);

        const real_t grid_min_x = (min_thv_x - ox) / dx;
        const real_t grid_min_y = (min_thv_y - oy) / dy;
        const real_t grid_min_z = (min_thv_z - oz) / dz;

        const ptrdiff_t i_min_thv = floor(grid_min_x);  // Min X index of the tetrahedron vertices
        const ptrdiff_t j_min_thv = floor(grid_min_y);  // Min Y index of the tetrahedron vertices
        const ptrdiff_t k_min_thv = floor(grid_min_z);  // Min Z index of the tetrahedron vertices

        // Max indices of the tetrahedron vertices
        real_t max_thv_x = max4(x0, x1, x2, x3);
        real_t max_thv_y = max4(y0, y1, y2, y3);
        real_t max_thv_z = max4(z0, z1, z2, z3);

        const real_t grid_max_x = (max_thv_x - ox) / dx;
        const real_t grid_max_y = (max_thv_y - oy) / dy;
        const real_t grid_max_z = (max_thv_z - oz) / dz;

        const ptrdiff_t i_max_thv = ceil(grid_max_x);  // Max X index of the tetrahedron vertices
        const ptrdiff_t j_max_thv = ceil(grid_max_y);  // Max Y index of the tetrahedron vertices
        const ptrdiff_t k_max_thv = ceil(grid_max_z);  // Max Z index of the tetrahedron vertices

        // Loop over the bounding box of the tetrahedron in the grid
        for (ptrdiff_t i = i_min_thv; i <= i_max_thv; i++) {
            for (ptrdiff_t j = j_min_thv; j <= j_max_thv; j++) {
                int in_out_status = 0;

                for (ptrdiff_t k = k_min_thv; k <= k_max_thv; k++) {
                    const real_t grid_x = (i * dx + ox);
                    const real_t grid_y = (j * dy + oy);
                    const real_t grid_z = (k * dz + oz);

                    // Check if the grid point is inside the tetrahedron
                    const int is_inside = check_p_inside_tetrahedron(grid_x,
                                                                     grid_y,
                                                                     grid_z,  //
                                                                     x0,
                                                                     y0,
                                                                     z0,
                                                                     x1,
                                                                     y1,
                                                                     z1,
                                                                     x2,
                                                                     y2,
                                                                     z2,
                                                                     x3,
                                                                     y3,
                                                                     z3);

                    if (in_out_status == 1 && !is_inside) {
                        // if the grid exit from the tetrahedron
                        // all the grid points after this point in the k loop
                        // are outside the tetrahedron
                        in_out_status = 0;
                        continue;
                    }

                    in_out_status = is_inside;

                    if (is_inside) {
                        // Interpolate the field at the grid point
                        const real_t val = shape_function(x0,
                                                          y0,
                                                          z0,
                                                          x1,
                                                          y1,
                                                          z1,
                                                          x2,
                                                          y2,
                                                          z2,
                                                          x3,
                                                          y3,
                                                          z3,
                                                          grid_x,
                                                          grid_y,
                                                          grid_z,
                                                          weighted_field_0,
                                                          weighted_field_1,
                                                          weighted_field_2,
                                                          weighted_field_3);

                        // Update the field at the grid point
                        const ptrdiff_t index = i * stride[0] + j * stride[1] + k * stride[2];
                        data[index]           = val;
                        // data[index]           = 2.2255;
                    }
                }
            }
        }
    }

    RETURN_FROM_FUNCTION(ret);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
int                                                                  //
apply_fun_to_mesh(const ptrdiff_t                    start_element,  // Mesh
                  const ptrdiff_t                    end_element,    // Mesh
                  const ptrdiff_t                    nnodes,         // Mesh
                  const idx_t** const SFEM_RESTRICT  elems,          // Mesh
                  const geom_t** const SFEM_RESTRICT xyz,            // Mesh
                  const function_XYZ_t               fun,            // Function
                  real_t* const SFEM_RESTRICT        weighted_field) {      // Output (weighted field)

    PRINT_CURRENT_FUNCTION;

    int ret = 0;

    for (int element_id = start_element; element_id < end_element; element_id++) {
        //
        real_t x0 = 0.0, x1 = 0.0, x2 = 0.0, x3 = 0.0;
        real_t y0 = 0.0, y1 = 0.0, y2 = 0.0, y3 = 0.0;
        real_t z0 = 0.0, z1 = 0.0, z2 = 0.0, z3 = 0.0;

        // And its weighted field
        idx_t ev[4];
        for (int v = 0; v < 4; v++) {
            ev[v] = elems[v][element_id];
        }

        {
            x0 = xyz[0][ev[0]];
            x1 = xyz[0][ev[1]];
            x2 = xyz[0][ev[2]];
            x3 = xyz[0][ev[3]];

            y0 = xyz[1][ev[0]];
            y1 = xyz[1][ev[1]];
            y2 = xyz[1][ev[2]];
            y3 = xyz[1][ev[3]];

            z0 = xyz[2][ev[0]];
            z1 = xyz[2][ev[1]];
            z2 = xyz[2][ev[2]];
            z3 = xyz[2][ev[3]];
        }

        weighted_field[ev[0]] = fun(x0, y0, z0);
        weighted_field[ev[1]] = fun(x1, y1, z1);
        weighted_field[ev[2]] = fun(x2, y2, z2);
        weighted_field[ev[3]] = fun(x3, y3, z3);
    }

    RETURN_FROM_FUNCTION(ret);
}
