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

    // Step 2: Convert to grid indices with respect to origin (0,0,0)
    // Using floor for minimum indices (with safety margin of -1)
    *min_grid_x = floor(x_min / (dx)) - 1;
    *min_grid_y = floor(y_min / (dy)) - 1;
    *min_grid_z = floor(z_min / (dz)) - 1;

    // Using ceil for maximum indices (with safety margin of +1)
    *max_grid_x = ceil(x_max / (dx)) + 1;
    *max_grid_y = ceil(y_max / (dy)) + 1;
    *max_grid_z = ceil(z_max / (dz)) + 1;

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
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
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
                    }
                }
            }

            return N * N * N;  // Total number of quadrature points
        }

        default:
            return -1;  // Unknown rule
    }
}

typedef struct {
    real_t x, y, z;    // Physical coordinates
    real_t weight;     // Physical weight
    bool   is_inside;  // Containment result
} quadrature_point_result_t;

static inline quadrature_point_result_t                       //
transform_and_check_quadrature_point(                         //
        const int                         q_ijk,              //
        const real_t* const SFEM_RESTRICT Q_nodes_x,          //
        const real_t* const SFEM_RESTRICT Q_nodes_y,          //
        const real_t* const SFEM_RESTRICT Q_nodes_z,          //
        const real_t* const SFEM_RESTRICT Q_weights,          //
        const geom_t* const SFEM_RESTRICT origin,             //
        const geom_t* const SFEM_RESTRICT delta,              //
        const real_t                      tet_vertices_x[4],  //
        const real_t                      tet_vertices_y[4],  //
        const real_t                      tet_vertices_z[4]) {                     //

    quadrature_point_result_t result;

    // Transform to physical coordinates
    result.x = Q_nodes_x[q_ijk] * delta[0] + origin[0];
    result.y = Q_nodes_y[q_ijk] * delta[1] + origin[1];
    result.z = Q_nodes_z[q_ijk] * delta[2] + origin[2];

    // Compute physical weight
    const real_t w = Q_weights[q_ijk];
    result.weight  = w * w * w * delta[0] * delta[1] * delta[2];

    // Check containment
    check_point_in_tet(1,
                       &result.x,
                       &result.y,
                       &result.z,
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
}

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

        ptrdiff_t min_grid_x, max_grid_x;
        ptrdiff_t min_grid_y, max_grid_y;
        ptrdiff_t min_grid_z, max_grid_z;

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

        const int N_midpoint = 4;
        const int dim_quad   = N_midpoint * N_midpoint * N_midpoint;
        real_t    Q_nodes_x[dim_quad];
        real_t    Q_nodes_y[dim_quad];
        real_t    Q_nodes_z[dim_quad];
        real_t    Q_weights[dim_quad];

        sfem_quad_rule_3D(TET_QUAD_MIDPOINT_NQP, N_midpoint, Q_nodes_x, Q_nodes_y, Q_nodes_z, Q_weights);

        for (int i_grid_x = min_grid_x; i_grid_x < max_grid_x; i_grid_x++) {
            for (int j_grid_y = min_grid_y; j_grid_y < max_grid_y; j_grid_y++) {
                for (int k_grid_z = min_grid_z; k_grid_z < max_grid_z; k_grid_z++) {
                    const int i = i_grid_x - min_grid_x;
                    const int j = j_grid_y - min_grid_y;
                    const int k = k_grid_z - min_grid_z;

                    // Midpoint quadrature rule in 3D

                    for (int q_ijk = 0; q_ijk < dim_quad; q_ijk++) {
                        quadrature_point_result_t result =                                                  //
                                transform_and_check_quadrature_point(q_ijk,                                 //
                                                                     Q_nodes_x,                             //
                                                                     Q_nodes_y,                             //
                                                                     Q_nodes_z,                             //
                                                                     Q_weights,                             //
                                                                     origin,                                //
                                                                     delta,                                 //
                                                                     (real_t[4]){x0_n, x1_n, x2_n, x3_n},   //
                                                                     (real_t[4]){y0_n, y1_n, y2_n, y3_n},   //
                                                                     (real_t[4]){z0_n, z1_n, z2_n, z3_n});  //

                        if (result.is_inside) {
                            // TODO: Transfer the contribution to the grid point
                        }
                    }
                }
            }
        }
    }
}  // End of tet4_resample_field_local_refine_adjoint_hyteg