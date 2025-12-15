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

#include "sfem_resample_field_adjoint_hex_quad.h"

#define out_real_t real_t

#define N_QUADRATURE_POINTS_MAX 256

static int    dim_quad_cube_p                      = 1;  // Midpoint quadrature in 3D
static real_t Q_nodes_x_p[N_QUADRATURE_POINTS_MAX] = {0.5};
static real_t Q_nodes_y_p[N_QUADRATURE_POINTS_MAX] = {0.5};
static real_t Q_nodes_z_p[N_QUADRATURE_POINTS_MAX] = {0.5};
static real_t Q_weights_p[N_QUADRATURE_POINTS_MAX] = {1.0};

int init_quad_points_hex_qtet(const int dim_quad) {
    const int dim_quad_cube = dim_quad * dim_quad * dim_quad;

    if (dim_quad_cube <= N_QUADRATURE_POINTS_MAX) {
        dim_quad_cube_p = dim_quad_cube;
        sfem_quad_rule_3D(dim_quad_cube, dim_quad_cube_p, Q_nodes_x_p, Q_nodes_y_p, Q_nodes_z_p, Q_weights_p);
    } else {
        return -1;  // Unsupported quadrature
    }

    return 0;  // Success
}

static int                   //
get_dim_qad() {              //
    return dim_quad_cube_p;  //
}  // END: get_dim_qad

int                                                                         //
compute_tet_bounding_box_norm(const real_t                   x0,            //
                              const real_t                   x1,            //
                              const real_t                   x2,            //
                              const real_t                   x3,            //
                              const real_t                   y0,            //
                              const real_t                   y1,            //
                              const real_t                   y2,            //
                              const real_t                   y3,            //
                              const real_t                   z0,            //
                              const real_t                   z1,            //
                              const real_t                   z2,            //
                              const real_t                   z3,            //
                              const ptrdiff_t                stride0,       //
                              const ptrdiff_t                stride1,       //
                              ptrdiff_t* const SFEM_RESTRICT min_grid_x,    //
                              ptrdiff_t* const SFEM_RESTRICT max_grid_x,    //
                              ptrdiff_t* const SFEM_RESTRICT min_grid_y,    //
                              ptrdiff_t* const SFEM_RESTRICT max_grid_y,    //
                              ptrdiff_t* const SFEM_RESTRICT min_grid_z,    //
                              ptrdiff_t* const SFEM_RESTRICT max_grid_z) {  //

    const real_t x_min = fmin(fmin(x0, x1), fmin(x2, x3));
    const real_t x_max = fmax(fmax(x0, x1), fmax(x2, x3));

    const real_t y_min = fmin(fmin(y0, y1), fmin(y2, y3));
    const real_t y_max = fmax(fmax(y0, y1), fmax(y2, y3));

    const real_t z_min = fmin(fmin(z0, z1), fmin(z2, z3));
    const real_t z_max = fmax(fmax(z0, z1), fmax(z2, z3));

    // const real_t dx = delta0;
    // const real_t dy = delta1;
    // const real_t dz = delta2;

    // const real_t ox = origin0;
    // const real_t oy = origin1;
    // const real_t oz = origin2;

    // Step 2: Convert to grid indices accounting for the origin
    // Formula: grid_index = (physical_coord - origin) / delta
    // Using floor for minimum indices (with safety margin of -1)
    *min_grid_x = floor(x_min) - 1;
    *min_grid_y = floor(y_min) - 1;
    *min_grid_z = floor(z_min) - 1;

    // Using ceil for maximum indices (with safety margin of +1)
    *max_grid_x = ceil(x_max) + 1;
    *max_grid_y = ceil(y_max) + 1;
    *max_grid_z = ceil(z_max) + 1;

    return 0;  // Success
}

/////////////////////////////////////////////////////////
// transform_quadrature_point_n /////////////////////////
/////////////////////////////////////////////////////////
quadrature_point_result_t                                                //
transform_quadrature_point_norm(const int q_ijk,                         //
                                const real_t* const restrict Q_nodes_x,  //
                                const real_t* const restrict Q_nodes_y,  //
                                const real_t* const restrict Q_nodes_z,  //
                                const real_t* const restrict Q_weights,  //
                                const ptrdiff_t i_grid,                  //
                                const ptrdiff_t j_grid,                  //
                                const ptrdiff_t k_grid) {                //

    quadrature_point_result_t result;
    result.is_inside = true;

    result.x = ((real_t)i_grid + Q_nodes_x[q_ijk]);
    result.y = ((real_t)j_grid + Q_nodes_y[q_ijk]);
    result.z = ((real_t)k_grid + Q_nodes_z[q_ijk]);

    result.weight = Q_weights[q_ijk];  // Could be removed since we use normalized weights

    return result;
}  // END: transform_and_check_quadrature_point

/////////////////////////////////////////////////////////
// transfer_weighted_field_tet4_to_hex //////////////////
/////////////////////////////////////////////////////////
ijk_index_t                                                                                //
transfer_weighted_field_tet4_to_hex_norm(const real_t                wf0,                  //
                                         const real_t                wf1,                  //
                                         const real_t                wf2,                  //
                                         const real_t                wf3,                  //
                                         const real_t                q_phys_x,             //
                                         const real_t                q_phys_y,             //
                                         const real_t                q_phys_z,             //
                                         const real_t                q_ref_x,              //
                                         const real_t                q_ref_y,              //
                                         const real_t                q_ref_z,              //
                                         const real_t                QW_phys_hex,          //
                                         real_t* const SFEM_RESTRICT hex_element_field) {  //

    // Compute the weighted contribution from the tetrahedron
    // Using linear shape functions for tetrahedron

    // Check if the reference coordinates are valid
    // If they are outside the tetrahedron, skip the contribution
    // Here we check if the ref coords are below the x-z, y-z, and x-y planes.
    // The others check in a previous step.
    // Check if the reference coordinates are valid (all 4 tet constraints)
    if (q_ref_x < 0.0 || q_ref_y < 0.0 || q_ref_z < 0.0 || (q_ref_x + q_ref_y + q_ref_z) > 1.0) {
        return (ijk_index_t){-1, -1, -1, false};
    }  // END if (outside tet)

    const real_t grid_x = q_phys_x;
    const real_t grid_y = q_phys_y;
    const real_t grid_z = q_phys_z;

    const ptrdiff_t i = floor(grid_x);
    const ptrdiff_t j = floor(grid_y);
    const ptrdiff_t k = floor(grid_z);

    const real_t l_x = (grid_x - (real_t)i);
    const real_t l_y = (grid_y - (real_t)j);
    const real_t l_z = (grid_z - (real_t)k);

    const real_t f0 = 1.0 - q_ref_x - q_ref_y - q_ref_z;
    const real_t f1 = q_ref_x;
    const real_t f2 = q_ref_y;
    const real_t f3 = q_ref_z;

    const real_t wf_quad = f0 * wf0 + f1 * wf1 + f2 * wf2 + f3 * wf3;

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

    const real_t wf_quad_QW = wf_quad * QW_phys_hex;

    hex_element_field[0] += wf_quad_QW * hex8_f0;
    hex_element_field[1] += wf_quad_QW * hex8_f1;
    hex_element_field[2] += wf_quad_QW * hex8_f2;
    hex_element_field[3] += wf_quad_QW * hex8_f3;
    hex_element_field[4] += wf_quad_QW * hex8_f4;
    hex_element_field[5] += wf_quad_QW * hex8_f5;
    hex_element_field[6] += wf_quad_QW * hex8_f6;
    hex_element_field[7] += wf_quad_QW * hex8_f7;

    return (ijk_index_t){i, j, k, true};
}  // END transfer_weighted_field_tet4_to_hex

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_adjoint_tet_quad_d ////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int                                                              //
tet4_resample_field_adjoint_tet_qtet_d(const real_t    x0_n,     // Tet vertices //
                                       const real_t    x1_n,     //
                                       const real_t    x2_n,     //
                                       const real_t    x3_n,     //
                                       const real_t    y0_n,     //
                                       const real_t    y1_n,     //
                                       const real_t    y2_n,     //
                                       const real_t    y3_n,     //
                                       const real_t    z0_n,     //
                                       const real_t    z1_n,     //
                                       const real_t    z2_n,     //
                                       const real_t    z3_n,     //
                                       const real_t    wf0,      // Weighted field at tet vertices
                                       const real_t    wf1,      //
                                       const real_t    wf2,      //
                                       const real_t    wf3,      //
                                       const ptrdiff_t stride0,  // Stride of hex grid
                                       const ptrdiff_t stride1,  //
                                       //    const int                   dim_quad,   // Number of quadrature points
                                       //    const real_t* const         Q_nodes_x,  // Quadrature nodes and weights
                                       //    const real_t* const         Q_nodes_y,  //
                                       //    const real_t* const         Q_nodes_z,  //
                                       //    const real_t* const         Q_weights,  //
                                       out_real_t* const SFEM_RESTRICT data) {  // Outut data array HEX
                                                                                // Placeholder implementation

    const int off0 = 0;
    const int off1 = stride0;
    const int off2 = stride0 + stride1;
    const int off3 = stride1;
    const int off4 = 0;
    const int off5 = stride0;
    const int off6 = stride0 + stride1;
    const int off7 = stride1;

    const int           dim_quad  = get_dim_qad();
    const real_t* const Q_nodes_x = Q_nodes_x_p;
    const real_t* const Q_nodes_y = Q_nodes_y_p;
    const real_t* const Q_nodes_z = Q_nodes_z_p;
    const real_t* const Q_weights = Q_weights_p;

    const real_t inv_dx = 1.0;
    const real_t inv_dy = 1.0;
    const real_t inv_dz = 1.0;

    real_t    inv_J_tet[9];
    ptrdiff_t min_grid_x, max_grid_x;
    ptrdiff_t min_grid_y, max_grid_y;
    ptrdiff_t min_grid_z, max_grid_z;

    tet4_inv_Jacobian(x0_n,        //
                      x1_n,        //
                      x2_n,        //
                      x3_n,        //
                      y0_n,        //
                      y1_n,        //
                      y2_n,        //
                      y3_n,        //
                      z0_n,        //
                      z1_n,        //
                      z2_n,        //
                      z3_n,        //
                      inv_J_tet);  //

    compute_tet_bounding_box_norm(x0_n,          //
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
                                  stride0,       //
                                  stride1,       //
                                  &min_grid_x,   //
                                  &max_grid_x,   //
                                  &min_grid_y,   //
                                  &max_grid_y,   //
                                  &min_grid_z,   //
                                  &max_grid_z);  //

    real_t hex_element_field[8] = {0.0};

    for (int k_grid_z = min_grid_z; k_grid_z < max_grid_z; k_grid_z++) {
        const real_t z_hex_min = ((real_t)k_grid_z);
        const real_t z_hex_max = z_hex_min + 1.0;

        for (int j_grid_y = min_grid_y; j_grid_y < max_grid_y; j_grid_y++) {
            const real_t y_hex_min = ((real_t)j_grid_y);
            const real_t y_hex_max = y_hex_min + 1.0;

            for (int i_grid_x = min_grid_x; i_grid_x < max_grid_x; i_grid_x++) {
                const real_t x_hex_min         = ((real_t)i_grid_x);
                const real_t x_hex_max         = x_hex_min + 1.0;
                const real_t hex_vertices_x[8] = {x_hex_min,
                                                  x_hex_max,
                                                  x_hex_max,
                                                  x_hex_min,  //
                                                  x_hex_min,
                                                  x_hex_max,
                                                  x_hex_max,
                                                  x_hex_min};

                const real_t hex_vertices_y[8] = {y_hex_min,
                                                  y_hex_min,
                                                  y_hex_max,
                                                  y_hex_max,  //
                                                  y_hex_min,
                                                  y_hex_min,
                                                  y_hex_max,
                                                  y_hex_max};

                const real_t hex_vertices_z[8] = {z_hex_min,
                                                  z_hex_min,
                                                  z_hex_min,
                                                  z_hex_min,  //
                                                  z_hex_max,
                                                  z_hex_max,
                                                  z_hex_max,
                                                  z_hex_max};

                const bool is_out_of_tet = is_hex_out_of_tet(inv_J_tet,        //
                                                             x0_n,             //
                                                             y0_n,             //
                                                             z0_n,             //
                                                             hex_vertices_x,   //
                                                             hex_vertices_y,   //
                                                             hex_vertices_z);  //

                // printf("Is out of tet: %d \n", is_out_of_tet);

                if (is_out_of_tet) continue;  // c Skip this hex cell

                // Midpoint quadrature rule in 3D

                memset(hex_element_field, 0, 8 * sizeof(real_t));

                for (int q_ijk = 0; q_ijk < dim_quad; q_ijk++) {
                    quadrature_point_result_t Qpoint_phys =             //
                            transform_quadrature_point_norm(q_ijk,      //
                                                            Q_nodes_x,  //
                                                            Q_nodes_y,  //
                                                            Q_nodes_z,  //
                                                            Q_weights,  //
                                                            i_grid_x,   //
                                                            j_grid_y,   //
                                                            k_grid_z);  //

                    real_t Q_ref_x, Q_ref_y, Q_ref_z;

                    tet4_inv_transform_J(inv_J_tet,      // Inverse Jacobian matrix
                                         Qpoint_phys.x,  // Physical coordinates of the quadrature point
                                         Qpoint_phys.y,  //
                                         Qpoint_phys.z,  //
                                         x0_n,           //
                                         y0_n,           //
                                         z0_n,           //
                                         &Q_ref_x,       // Reference coordinates of the quadrature point
                                         &Q_ref_y,       //
                                         &Q_ref_z);      //

                    // for (int v = 0; v < 8; v++) hex_element_field[v] = 0.0;

                    ijk_index_t ijk_indices =                                            //
                            transfer_weighted_field_tet4_to_hex_ckp(wf0,                 //
                                                                    wf1,                 //
                                                                    wf2,                 //
                                                                    wf3,                 //
                                                                    Qpoint_phys.x,       //
                                                                    Qpoint_phys.y,       //
                                                                    Qpoint_phys.z,       //
                                                                    Q_ref_x,             //
                                                                    Q_ref_y,             //
                                                                    Q_ref_z,             //
                                                                    Qpoint_phys.weight,  //
                                                                    0.0,                 //
                                                                    0.0,                 //
                                                                    0.0,                 //
                                                                    1.0,                 //
                                                                    1.0,                 //
                                                                    1.0,                 //
                                                                    hex_element_field);  //

                }  // END: for q_ijk

                const ptrdiff_t base_index = i_grid_x * stride0 +  //
                                             j_grid_y * stride1 +  //
                                             k_grid_z;             //

                data[base_index + off0] += hex_element_field[0];  //
                data[base_index + off1] += hex_element_field[1];  //
                data[base_index + off2] += hex_element_field[2];  //
                data[base_index + off3] += hex_element_field[3];  //
                data[base_index + off4] += hex_element_field[4];  //
                data[base_index + off5] += hex_element_field[5];  //
                data[base_index + off6] += hex_element_field[6];  //
                data[base_index + off7] += hex_element_field[7];  //

            }  // END: for k_grid_z
        }  // END: for i_grid_y
    }  // END: for j_grid_y

    RETURN_FROM_FUNCTION(0);
}  // END: Function: tet4_resample_field_adjoint_tet_quad_d