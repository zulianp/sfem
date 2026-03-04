#ifdef _OPENMP
#include <omp.h>
#endif

#include "cell_list_3d_map_mesh.h"
#include "sfem_resample_field_adjoint_cell.h"
#include "sfem_resample_field_adjoint_hex_quad.h"

#define MAX_Z_SIZE 64  // Define a maximum size for the z array to prevent overflow, adjust as needed

//////////////////////////////////////////////
// build_bounding_boxes_mesh_geom
//////////////////////////////////////////////
int                                                            //
build_bounding_boxes_mesh_geom(const mesh_t     *mesh,         //
                               boxes_t         **boxes,        //
                               mesh_tet_geom_t **mesh_geom) {  //
                                                               //
    int fb_error = 0;

    if (boxes == NULL || mesh_geom == NULL) {
        fprintf(stderr, "Error: Invalid pointer for boxes or mesh geometry\n");
        return EXIT_FAILURE;
    }

    if (*boxes != NULL) {
        fprintf(stderr, "Error: boxes pointer is not NULL\n");
        return EXIT_FAILURE;
    }

    if (*mesh_geom != NULL) {
        fprintf(stderr, "Error: mesh geometry pointer is not NULL\n");
        return EXIT_FAILURE;
    }

    fb_error =                                                    //
            make_mesh_tets_boxes(0,                               //
                                 mesh->nelements,                 //
                                 mesh->nnodes,                    //
                                 (const idx_t **)mesh->elements,  //
                                 (const geom_t **)mesh->points,   //
                                 boxes);                          //

    *mesh_geom = mesh_tet_geometry_alloc(mesh);
    mesh_tet_geometry_compute_inv_Jacobian(*mesh_geom);

    return 0;
}  // END Function: build_bounding_boxes_mesh_geom

//////////////////////////////////////////////
// build_bounding_box_statistics
//////////////////////////////////////////////
int                                                                     //
build_bounding_box_statistics(const boxes_t              *boxes,        //
                              const int                   bins,         //
                              bounding_box_statistics_t **stats,        //
                              side_length_histograms_t  **histograms) {  //
                                                                        //
    if (boxes == NULL || stats == NULL || histograms == NULL) {
        fprintf(stderr, "Error: Invalid pointer for boxes, stats, or histograms\n");
        return EXIT_FAILURE;
    }

    if (*stats != NULL) {
        fprintf(stderr, "Error: stats pointer is not NULL\n");
        return EXIT_FAILURE;
    }

    if (*histograms != NULL) {
        fprintf(stderr, "Error: histograms pointer is not NULL\n");
        return EXIT_FAILURE;
    }

    *stats      = (bounding_box_statistics_t *)malloc(sizeof(bounding_box_statistics_t));
    *histograms = (side_length_histograms_t *)malloc(sizeof(side_length_histograms_t));

    **stats      = calculate_bounding_box_statistics(boxes);
    **histograms = calculate_side_length_histograms(boxes, *stats, bins);

    return 0;
}  // END Function: build_bounding_box_statistics

//////////////////////////////////////////////
// update_hex_field
//////////////////////////////////////////////
int                                                                        //
update_hex_quad_node(const real_t                         x,               // Physical x coordinate of the quadrature point
                     const real_t                         y,               // Physical y coordinate of the quadrature point
                     const real_t                         z,               // Physical z coordinate of the quadrature point
                     const real_t                         phys_w,          // Quadrature weight for the quadrature point
                     const ptrdiff_t                      index_tet,       // The index of the tet containing the quadrature point
                     const mesh_t *const SFEM_RESTRICT    mesh,            // Mesh: mesh_t struct
                     mesh_tet_geom_t                     *mesh_geom,       // Mesh geometry data structure
                     const ptrdiff_t *const SFEM_RESTRICT stride,          // SDF: stride[3]
                     const geom_t *const SFEM_RESTRICT    origin,          // SDF: origin[3]
                     const geom_t *const SFEM_RESTRICT    delta,           // SDF: delta[3]
                     const real_t *const SFEM_RESTRICT    weighted_field,  // Weighted field
                     real_t *const SFEM_RESTRICT          hex_element_field) {      // Output field values for the 8 hex nodes

    const ptrdiff_t stride0 = stride[0];
    const ptrdiff_t stride1 = stride[1];
    const ptrdiff_t stride2 = stride[2];

    const ptrdiff_t off0 = 0;
    const ptrdiff_t off1 = stride0;
    const ptrdiff_t off2 = stride0 + stride1;
    const ptrdiff_t off3 = stride1;
    const ptrdiff_t off4 = stride2;
    const ptrdiff_t off5 = stride0 + stride2;
    const ptrdiff_t off6 = stride0 + stride1 + stride2;
    const ptrdiff_t off7 = stride1 + stride2;

    const real_t ox = origin[0];
    const real_t oy = origin[1];
    const real_t oz = origin[2];

    const real_t inv_dx = 1.0 / delta[0];
    const real_t inv_dy = 1.0 / delta[1];
    const real_t inv_dz = 1.0 / delta[2];

    const real_t grid_x = (x - ox) * inv_dx;
    const real_t grid_y = (y - oy) * inv_dy;
    const real_t grid_z = (z - oz) * inv_dz;

    const ptrdiff_t i = floor(grid_x);
    const ptrdiff_t j = floor(grid_y);
    const ptrdiff_t k = floor(grid_z);

    const real_t l_x = (grid_x - (real_t)i);
    const real_t l_y = (grid_y - (real_t)j);
    const real_t l_z = (grid_z - (real_t)k);

    const ptrdiff_t base_index = i * stride0 +  //
                                 j * stride1 +  //
                                 k * stride2;   //

    const idx_t ev0 = mesh->elements[0][index_tet];
    const idx_t ev1 = mesh->elements[1][index_tet];
    const idx_t ev2 = mesh->elements[2][index_tet];
    const idx_t ev3 = mesh->elements[3][index_tet];

    const real_t wf0 = weighted_field[ev0];  // Weighted field at vertex 0
    const real_t wf1 = weighted_field[ev1];  // Weighted field at vertex 1
    const real_t wf2 = weighted_field[ev2];  // Weighted field at vertex 2
    const real_t wf3 = weighted_field[ev3];  // Weighted field at vertex 3

    const real_t *inv_J_tet = &(mesh_geom->inv_Jacobian[index_tet * 9]);  // Inverse Jacobian for the current tet

    const real_t x0_n = mesh_geom->vetices_zero[index_tet * 3 + 0];  // x coordinate of vertex 0
    const real_t y0_n = mesh_geom->vetices_zero[index_tet * 3 + 1];  // y coordinate of vertex 0
    const real_t z0_n = mesh_geom->vetices_zero[index_tet * 3 + 2];  // z coordinate of vertex 0

    // Compute the coordinates of the quadrature point in the reference tetrahedron using the inverse Jacobian transformation.
    const real_t x_o = x - x0_n;
    const real_t y_o = y - y0_n;
    const real_t z_o = z - z0_n;

    const real_t inv_J_00 = inv_J_tet[0];
    const real_t inv_J_01 = inv_J_tet[1];
    const real_t inv_J_02 = inv_J_tet[2];
    const real_t inv_J_10 = inv_J_tet[3];
    const real_t inv_J_11 = inv_J_tet[4];
    const real_t inv_J_12 = inv_J_tet[5];
    const real_t inv_J_20 = inv_J_tet[6];
    const real_t inv_J_21 = inv_J_tet[7];
    const real_t inv_J_22 = inv_J_tet[8];

    const real_t x_ref = inv_J_00 * x_o + inv_J_01 * y_o + inv_J_02 * z_o;
    const real_t y_ref = inv_J_10 * x_o + inv_J_11 * y_o + inv_J_12 * z_o;
    const real_t z_ref = inv_J_20 * x_o + inv_J_21 * y_o + inv_J_22 * z_o;

    const real_t f0 = 1.0 - x_ref - y_ref - z_ref;
    const real_t f1 = x_ref;
    const real_t f2 = y_ref;
    const real_t f3 = z_ref;

    const real_t wf_quad = f0 * wf0 + f1 * wf1 + f2 * wf2 + f3 * wf3;

    const real_t one_minus_lx = (1.0 - l_x);
    const real_t one_minus_ly = (1.0 - l_y);
    const real_t one_minus_lz = (1.0 - l_z);

    const real_t c0 = one_minus_lx * one_minus_ly;
    const real_t c1 = l_x * one_minus_ly;
    const real_t c2 = l_x * l_y;
    const real_t c3 = one_minus_lx * l_y;

    const real_t wf_quad_QW = wf_quad * phys_w;

    const real_t w_c0 = wf_quad_QW * c0;
    const real_t w_c1 = wf_quad_QW * c1;
    const real_t w_c2 = wf_quad_QW * c2;
    const real_t w_c3 = wf_quad_QW * c3;

    real_t *const SFEM_RESTRICT out = &hex_element_field[base_index];

    out[off0] += w_c0 * one_minus_lz;
    out[off1] += w_c1 * one_minus_lz;
    out[off2] += w_c2 * one_minus_lz;
    out[off3] += w_c3 * one_minus_lz;
    out[off4] += w_c0 * l_z;
    out[off5] += w_c1 * l_z;
    out[off6] += w_c2 * l_z;
    out[off7] += w_c3 * l_z;

    return 0;
}  // END Function: update_hex_quad_node

//////////////////////////////////////////////
// update_hex_field
//////////////////////////////////////////////
int                                                                      //
update_hex_quad_node_vz(const real_t                         x,          // Physical x coordinate of the quadrature point
                        const real_t                         y,          // Physical y coordinate of the quadrature point
                        const real_t                        *z,          // Physical z coordinate of the quadrature points
                        const ptrdiff_t                      z_size,     // Size of the z array (number of z values to process)
                        const real_t                         phys_w,     // Quadrature weight for the quadrature point
                        const ptrdiff_t                      index_tet,  // The index of the tet containing the quadrature point
                        const mesh_t *const SFEM_RESTRICT    mesh,       // Mesh: mesh_t struct
                        mesh_tet_geom_t                     *mesh_geom,  // Mesh geometry data structure
                        const ptrdiff_t *const SFEM_RESTRICT stride,     // SDF: stride[3]
                        const geom_t *const SFEM_RESTRICT    origin,     // SDF: origin[3]
                        const geom_t *const SFEM_RESTRICT    delta,      // SDF: delta[3]
                        const real_t *const SFEM_RESTRICT    weighted_field,  // Weighted field
                        real_t *const SFEM_RESTRICT          hex_element_field) {      // Output field values for the 8 hex nodes

    // In this function, we assumes that the z array contains multiple z values that lies in the same tet, and we want to update
    // the hex field for each of these z values. The x and y coordinates are the same for all z values, and the index_tet is also
    // the same.

    if (z_size > MAX_Z_SIZE) {
        fprintf(stderr, "Error: z_size exceeds maximum allowed size of %d\n", MAX_Z_SIZE);
        return EXIT_FAILURE;
    }

    real_t    grid_z_buffer[MAX_Z_SIZE];      // Buffer to hold grid z values for processing
    ptrdiff_t k_buffer[MAX_Z_SIZE];           // Buffer to hold k indices for processing
    real_t    l_z_buffer[MAX_Z_SIZE];         // Buffer to hold local z coordinates for processing
    ptrdiff_t base_index_buffer[MAX_Z_SIZE];  // Buffer to hold base indices for processing
    real_t    x_ref_buffer[MAX_Z_SIZE];       // Buffer to hold x reference coordinates for processing
    real_t    y_ref_buffer[MAX_Z_SIZE];       // Buffer to hold y reference coordinates for processing
    real_t    z_ref_buffer[MAX_Z_SIZE];       // Buffer to hold z reference coordinates for processing

    const int off0 = 0;
    const int off1 = stride[0];
    const int off2 = stride[0] + stride[1];
    const int off3 = stride[1];
    const int off4 = stride[2];
    const int off5 = stride[0] + stride[2];
    const int off6 = stride[0] + stride[1] + stride[2];
    const int off7 = stride[1] + stride[2];

    const real_t ox = origin[0];
    const real_t oy = origin[1];
    const real_t oz = origin[2];

    const real_t dx     = delta[0];
    const real_t dy     = delta[1];
    const real_t dz     = delta[2];
    const real_t inv_dz = 1.0 / dz;  // Precompute inverse to avoid repeated division

    const real_t grid_x = (x - ox) / dx;
    const real_t grid_y = (y - oy) / dy;

    const ptrdiff_t i = floor(grid_x);
    const ptrdiff_t j = floor(grid_y);

    const real_t l_x = (grid_x - (real_t)i);
    const real_t l_y = (grid_y - (real_t)j);

    for (ptrdiff_t idx = 0; idx < z_size; ++idx) {
        grid_z_buffer[idx] = (z[idx] - oz) * inv_dz;  // Use multiplication instead of division

        k_buffer[idx] = floor(grid_z_buffer[idx]);

        l_z_buffer[idx] = (grid_z_buffer[idx] - (real_t)k_buffer[idx]);

        base_index_buffer[idx] = i * stride[0] +                        //
                                 j * stride[1] +                        //
                                 (ptrdiff_t)k_buffer[idx] * stride[2];  //
    }

    idx_t ev[4];

    for (int v = 0; v < 4; ++v) {
        ev[v] = mesh->elements[v][index_tet];
    }  // END: for vq

    const real_t wf0 = weighted_field[ev[0]];  // Weighted field at vertex 0
    const real_t wf1 = weighted_field[ev[1]];  // Weighted field at vertex 1
    const real_t wf2 = weighted_field[ev[2]];  // Weighted field at vertex 2
    const real_t wf3 = weighted_field[ev[3]];  // Weighted field at vertex 3

    const real_t *inv_J_tet = &(mesh_geom->inv_Jacobian[index_tet * 9]);  // Inverse Jacobian for the current tet

    const real_t x0_n = mesh_geom->vetices_zero[index_tet * 3 + 0];  // x coordinate of vertex 0
    const real_t y0_n = mesh_geom->vetices_zero[index_tet * 3 + 1];  // y coordinate of vertex 0
    const real_t z0_n = mesh_geom->vetices_zero[index_tet * 3 + 2];  // z coordinate of vertex 0

    // Compute the coordinates of the quadrature point in the reference tetrahedron using the inverse Jacobian transformation.
    const real_t x_o = x - x0_n;
    const real_t y_o = y - y0_n;

    // Compute reference coordinates efficiently in a single loop
    const real_t inv_J_00 = inv_J_tet[0];
    const real_t inv_J_01 = inv_J_tet[1];
    const real_t inv_J_02 = inv_J_tet[2];
    const real_t inv_J_10 = inv_J_tet[3];
    const real_t inv_J_11 = inv_J_tet[4];
    const real_t inv_J_12 = inv_J_tet[5];
    const real_t inv_J_20 = inv_J_tet[6];
    const real_t inv_J_21 = inv_J_tet[7];
    const real_t inv_J_22 = inv_J_tet[8];

    for (ptrdiff_t idx = 0; idx < z_size; ++idx) {
        const real_t z_o  = z[idx] - z0_n;  // Compute inline instead of using z_o_buffer
        x_ref_buffer[idx] = inv_J_00 * x_o + inv_J_01 * y_o + inv_J_02 * z_o;
        y_ref_buffer[idx] = inv_J_10 * x_o + inv_J_11 * y_o + inv_J_12 * z_o;
        z_ref_buffer[idx] = inv_J_20 * x_o + inv_J_21 * y_o + inv_J_22 * z_o;
    }  // END for (idx)

    // Precalculate factors that are independent from idx
    const real_t one_minus_lx = (1.0 - l_x);
    const real_t one_minus_ly = (1.0 - l_y);

    // Precalculate all x-y dependent factors outside the loop
    const real_t c0 = one_minus_lx * one_minus_ly;  // (1-l_x) * (1-l_y)
    const real_t c1 = l_x * one_minus_ly;           // l_x * (1-l_y)
    const real_t c2 = l_x * l_y;                    // l_x * l_y
    const real_t c3 = one_minus_lx * l_y;           // (1-l_x) * l_y

    // Direct accumulation loop - eliminates hex_element_field_local buffer and final accumulation loop
    for (ptrdiff_t idx = 0; idx < z_size; ++idx) {
        const real_t x_ref = x_ref_buffer[idx];
        const real_t y_ref = y_ref_buffer[idx];
        const real_t z_ref = z_ref_buffer[idx];

        const real_t f0 = 1.0 - x_ref - y_ref - z_ref;
        const real_t f1 = x_ref;
        const real_t f2 = y_ref;
        const real_t f3 = z_ref;

        const real_t wf_quad = f0 * wf0 + f1 * wf1 + f2 * wf2 + f3 * wf3;

        // Compute z-dependent factors for this iteration
        const real_t one_minus_lz = (1.0 - l_z_buffer[idx]);
        const real_t l_z          = l_z_buffer[idx];

        const real_t hex8_f0 = c0 * one_minus_lz;
        const real_t hex8_f1 = c1 * one_minus_lz;
        const real_t hex8_f2 = c2 * one_minus_lz;
        const real_t hex8_f3 = c3 * one_minus_lz;
        const real_t hex8_f4 = c0 * l_z;
        const real_t hex8_f5 = c1 * l_z;
        const real_t hex8_f6 = c2 * l_z;
        const real_t hex8_f7 = c3 * l_z;

        const real_t    wf_quad_QW = wf_quad * phys_w;
        const ptrdiff_t base_index = base_index_buffer[idx];

        hex_element_field[base_index + off0] += wf_quad_QW * hex8_f0;
        hex_element_field[base_index + off1] += wf_quad_QW * hex8_f1;
        hex_element_field[base_index + off2] += wf_quad_QW * hex8_f2;
        hex_element_field[base_index + off3] += wf_quad_QW * hex8_f3;
        hex_element_field[base_index + off4] += wf_quad_QW * hex8_f4;
        hex_element_field[base_index + off5] += wf_quad_QW * hex8_f5;
        hex_element_field[base_index + off6] += wf_quad_QW * hex8_f6;
        hex_element_field[base_index + off7] += wf_quad_QW * hex8_f7;
    }  // END for (idx)

    return 0;
}  // END Function: update_hex_quad_node_vz

//////////////////////////////////////////////
// compress_and_reorder
//////////////////////////////////////////////
int                                     //
compress_and_reorder(int    *keyArray,  //
                     real_t *valArray,  //
                     int     n) {           //
    int writeIndex = 0;

    for (int readIndex = 0; readIndex < n; readIndex++) {
        if (keyArray[readIndex] != -1) {
            keyArray[writeIndex] = keyArray[readIndex];
            valArray[writeIndex] = valArray[readIndex];

            writeIndex++;
        }
    }

    int newCount = writeIndex;
    while (writeIndex < n) {
        keyArray[writeIndex] = -1;
        valArray[writeIndex] = 0;
        writeIndex++;
    }

    // Return the count of valid items
    return newCount;
}  // END Function: compress_and_reorder

//////////////////////////////////////////////
// update_hex_field
//////////////////////////////////////////////
int                                                                        //
update_hex_field(cell_list_split_3d_2d_map_t         *split_map,           // Cell list split map data structure
                 boxes_t                             *boxes,               // Boxes data structure
                 mesh_tet_geom_t                     *mesh_geom,           // Mesh geometry data structure
                 const ptrdiff_t                      i_grid,              // The i index of the grid point in the hex mesh
                 const ptrdiff_t                      j_grid,              // The j index of the grid point in the hex mesh
                 const mesh_t *const SFEM_RESTRICT    mesh,                // Mesh: mesh_t struct
                 const ptrdiff_t *const SFEM_RESTRICT n,                   // SDF: n[3]
                 const ptrdiff_t *const SFEM_RESTRICT stride,              // SDF: stride[3]
                 const geom_t *const SFEM_RESTRICT    origin,              // SDF: origin[3]
                 const geom_t *const SFEM_RESTRICT    delta,               // SDF: delta[3]
                 const real_t *const SFEM_RESTRICT    weighted_field,      // Weighted field
                 real_t                              *z_array_buffer,      // Buffer to hold z values for processing
                 int                                 *tet_indices_buffer,  // Buffer to hold tet indices for processing
                 real_t *const SFEM_RESTRICT          hex_field) {                  // Output field for the hex cell containing (x,y,z)

    // get the physical coordinates of the grid point (i_grid, j_grid) in the hex mesh.
    const real_t grid_x = origin[0] + i_grid * delta[0];
    const real_t grid_y = origin[1] + j_grid * delta[1];
    const int    z_size = n[2];

    const real_t *quad_x = get_Q_nodes_x_p();
    const real_t *quad_y = get_Q_nodes_y_p();
    const real_t *quad_z = get_Q_nodes_z_p();
    const real_t *quad_w = get_Q_weights_p();
    const int     dim_q  = get_dim_qad();

    const real_t hex_volume = delta[0] * delta[1] * delta[2];

    // real_t *z_array     = malloc(z_size * sizeof(real_t));
    // int    *tet_indices = malloc(z_size * sizeof(int));
    real_t *z_array     = z_array_buffer;
    int    *tet_indices = tet_indices_buffer;

    for (int q_ijk = 0; q_ijk < dim_q; q_ijk++) {
        const real_t q_x = quad_x[q_ijk];
        const real_t q_y = quad_y[q_ijk];
        const real_t q_z = quad_z[q_ijk];
        const real_t q_w = quad_w[q_ijk];

        const real_t phys_x      = grid_x + q_x * delta[0];
        const real_t phys_y      = grid_y + q_y * delta[1];
        const real_t phys_z_base = origin[2] + q_z * delta[2];

        const real_t phys_w = q_w * hex_volume;

        // Populate z_array with the physical z coordinates for the current quadrature point
        const real_t delta_z = delta[2];
        for (ptrdiff_t k = 0; k < z_size; k++) {
            z_array[k] = phys_z_base + (real_t)k * delta_z;
        }

        query_cell_list_3d_2d_split_map_mesh_given_xy_tets_v(split_map,     //
                                                             boxes,         //
                                                             mesh_geom,     //
                                                             phys_x,        //
                                                             phys_y,        //
                                                             z_array,       //
                                                             z_size,        //
                                                             tet_indices);  //

        compress_and_reorder(tet_indices, z_array, z_size);

        for (ptrdiff_t k = 0; k < z_size; k++) {
            // If compress and reorder is called it is ensured that after the first -1 there is no others tets.
            if (tet_indices[k] < 0) break;

            update_hex_quad_node(phys_x,          //
                                 phys_y,          //
                                 z_array[k],      //
                                 phys_w,          //
                                 tet_indices[k],  //
                                 mesh,            //
                                 mesh_geom,       //
                                 stride,          //
                                 origin,          //
                                 delta,           //
                                 weighted_field,  //
                                 hex_field);      //

            // Now we have the tet index for the current quadrature point (phys_x, phys_y, z_array[k])
            // We can use this tet index to get the corresponding value from tet_g and accumulate it.
        }
    }

    return 0;
}  // END Function: update_hex_field

//////////////////////////////////////////////
// update_hex_field_vz
//////////////////////////////////////////////
int                                                                       //
update_hex_field_vz(cell_list_split_3d_2d_map_t         *split_map,       // Cell list split map data structure
                    boxes_t                             *boxes,           // Boxes data structure
                    mesh_tet_geom_t                     *mesh_geom,       // Mesh geometry data structure
                    const ptrdiff_t                      i_grid,          // The i index of the grid point in the hex mesh
                    const ptrdiff_t                      j_grid,          // The j index of the grid point in the hex mesh
                    const mesh_t *const SFEM_RESTRICT    mesh,            // Mesh: mesh_t struct
                    const ptrdiff_t *const SFEM_RESTRICT n,               // SDF: n[3]
                    const ptrdiff_t *const SFEM_RESTRICT stride,          // SDF: stride[3]
                    const geom_t *const SFEM_RESTRICT    origin,          // SDF: origin[3]
                    const geom_t *const SFEM_RESTRICT    delta,           // SDF: delta[3]
                    const real_t *const SFEM_RESTRICT    weighted_field,  // Weighted field
                    real_t *const SFEM_RESTRICT          hex_field) {              // Output field for the hex cell containing (x,y,z)

    // get the physical coordinates of the grid point (i_grid, j_grid) in the hex mesh.
    const real_t grid_x = origin[0] + i_grid * delta[0];
    const real_t grid_y = origin[1] + j_grid * delta[1];
    const int    z_size = n[2];

    const real_t *quad_x = get_Q_nodes_x_p();
    const real_t *quad_y = get_Q_nodes_y_p();
    const real_t *quad_z = get_Q_nodes_z_p();
    const real_t *quad_w = get_Q_weights_p();
    const int     dim_q  = get_dim_qad();

    const real_t hex_volume = delta[0] * delta[1] * delta[2];

    real_t *z_array     = malloc(z_size * sizeof(real_t));
    int    *tet_indices = malloc(z_size * sizeof(int));

    for (int q_ijk = 0; q_ijk < dim_q; q_ijk++) {
        const real_t q_x = quad_x[q_ijk];
        const real_t q_y = quad_y[q_ijk];
        const real_t q_z = quad_z[q_ijk];
        const real_t q_w = quad_w[q_ijk];

        const real_t phys_x      = grid_x + q_x * delta[0];
        const real_t phys_y      = grid_y + q_y * delta[1];
        const real_t phys_z_base = origin[2] + q_z * delta[2];

        // Populate z_array with the physical z coordinates for the current quadrature point
        const real_t delta_z = delta[2];
        for (ptrdiff_t k = 0; k < z_size; k++) {
            z_array[k] = phys_z_base + (real_t)k * delta_z;
        }

        query_cell_list_3d_2d_split_map_mesh_given_xy_tets_v(split_map,     //
                                                             boxes,         //
                                                             mesh_geom,     //
                                                             phys_x,        //
                                                             phys_y,        //
                                                             z_array,       //
                                                             z_size,        //
                                                             tet_indices);  //

        compress_and_reorder(tet_indices, z_array, z_size);

        real_t vz[MAX_Z_SIZE] = {0};  // Buffer to hold vz values - INITIALIZED TO ZERO

        real_t phys_w = q_w * hex_volume;

        ptrdiff_t k = 0;
        while (k < z_size) {
            const ptrdiff_t tet_index = tet_indices[k];

            // Skip invalid tet indices (marked as -1 by compress_and_reorder)
            if (tet_index < 0) {
                k++;
                continue;
            }  // END if (tet_index < 0)

            ptrdiff_t vz_index_loc = 0;

            // Collect all consecutive z values with the same valid tet_index
            while (k < z_size && tet_index == tet_indices[k] && vz_index_loc < MAX_Z_SIZE) {
                vz[vz_index_loc] = z_array[k];
                vz_index_loc += 1;
                k += 1;
            }  // END while (k < z_size && tet_index == tet_indices[k])

            // Only process if we have valid z values
            if (vz_index_loc > 0) {
                update_hex_quad_node_vz(phys_x,          //
                                        phys_y,          //
                                        vz,              //
                                        vz_index_loc,    //
                                        phys_w,          //
                                        tet_index,       //
                                        mesh,            //
                                        mesh_geom,       //
                                        stride,          //
                                        origin,          //
                                        delta,           //
                                        weighted_field,  //
                                        hex_field);      //
            }  // END if (vz_index_loc > 0)
        }  // END while (k < z_size)
    }
    // END for (int q_ijk)

    // Free allocated arrays
    free(z_array);
    z_array = NULL;

    free(tet_indices);
    tet_indices = NULL;

    return 0;
}

//////////////////////////////////////////////
// transfer_to_hex_field
//////////////////////////////////////////////
int                                                                                   //
transfer_to_hex_field_cell_tet4(cell_list_split_3d_2d_map_t         *split_map,       // Cell list split map data structure
                                boxes_t                             *boxes,           // Boxes data structure
                                mesh_tet_geom_t                     *mesh_geom,       // Mesh geometry data structure
                                const mesh_t *const SFEM_RESTRICT    mesh,            // Mesh: mesh_t struct
                                const ptrdiff_t *const SFEM_RESTRICT n,               // SDF: n[3]
                                const ptrdiff_t *const SFEM_RESTRICT stride,          // SDF: stride[3]
                                const geom_t *const SFEM_RESTRICT    origin,          // SDF: origin[3]
                                const geom_t *const SFEM_RESTRICT    delta,           // SDF: delta[3]
                                const real_t *const SFEM_RESTRICT    weighted_field,  // Weighted field
                                real_t *const SFEM_RESTRICT          hex_field) {              //

    PRINT_CURRENT_FUNCTION;

    const ptrdiff_t x_size = n[0];
    const ptrdiff_t y_size = n[1];
    const ptrdiff_t z_size = n[2];

    real_t *z_array_buffer     = malloc(z_size * sizeof(real_t));
    int    *tet_indices_buffer = malloc(z_size * sizeof(int));

    if (!z_array_buffer || !tet_indices_buffer) {
        free(z_array_buffer);
        free(tet_indices_buffer);
        RETURN_FROM_FUNCTION(-1);
    }  // END if (!z_array_buffer || !tet_indices_buffer)

    for (ptrdiff_t i_grid = 0; i_grid < x_size; i_grid++) {
        for (ptrdiff_t j_grid = 0; j_grid < y_size; j_grid++) {
            update_hex_field(split_map,           //
                             boxes,               //
                             mesh_geom,           //
                             i_grid,              //
                             j_grid,              //
                             mesh,                //
                             n,                   //
                             stride,              //
                             origin,              //
                             delta,               //
                             weighted_field,      //
                             z_array_buffer,      //
                             tet_indices_buffer,  //
                             hex_field);          //
        }
    }

    free(z_array_buffer);
    z_array_buffer = NULL;

    free(tet_indices_buffer);
    tet_indices_buffer = NULL;

    RETURN_FROM_FUNCTION(0);
}  // END Function: transfer_to_hex_field_cell_tet4

//////////////////////////////////////////////
// transfer_to_hex_field
//////////////////////////////////////////////
int                                                                                        //
transfer_to_hex_field_cell_split_par_tet4(cell_list_split_3d_2d_map_t         *split_map,  // Cell list split map data structure
                                          boxes_t                             *boxes,      // Boxes data structure
                                          mesh_tet_geom_t                     *mesh_geom,  // Mesh geometry data structure
                                          const mesh_t *const SFEM_RESTRICT    mesh,       // Mesh: mesh_t struct
                                          const ptrdiff_t *const SFEM_RESTRICT n,          // SDF: n[3]
                                          const ptrdiff_t *const SFEM_RESTRICT stride,     // SDF: stride[3]
                                          const geom_t *const SFEM_RESTRICT    origin,     // SDF: origin[3]
                                          const geom_t *const SFEM_RESTRICT    delta,      // SDF: delta[3]
                                          const real_t *const SFEM_RESTRICT    weighted_field,  // Weighted field
                                          real_t *const SFEM_RESTRICT          hex_field) {              //
    PRINT_CURRENT_FUNCTION;

#ifdef _OPENMP
    int num_procs = omp_get_num_procs();
    omp_set_dynamic(0);
    omp_set_num_threads(num_procs);
#endif

    const ptrdiff_t x_size = n[0];
    const ptrdiff_t y_size = n[1];

#pragma omp parallel
    {
        const ptrdiff_t z_size = n[2];

        real_t *z_array_buffer     = malloc(z_size * sizeof(real_t));
        int    *tet_indices_buffer = malloc(z_size * sizeof(int));

        if (z_array_buffer && tet_indices_buffer) {
            for (ptrdiff_t start_i = 0; start_i < 3; start_i++) {
                for (ptrdiff_t start_j = 0; start_j < 3; start_j++) {
                    ptrdiff_t loop_count_i = (x_size > start_i) ? ((x_size - start_i + 2) / 3) : 0;
                    ptrdiff_t loop_count_j = (y_size > start_j) ? ((y_size - start_j + 2) / 3) : 0;

#pragma omp for collapse(2) schedule(guided)
                    for (ptrdiff_t k = 0; k < loop_count_i; k++) {
                        for (ptrdiff_t m = 0; m < loop_count_j; m++) {
                            ptrdiff_t i_grid = start_i + (k * 3);
                            ptrdiff_t j_grid = start_j + (m * 3);

                            update_hex_field(split_map,           //
                                             boxes,               //
                                             mesh_geom,           //
                                             i_grid,              //
                                             j_grid,              //
                                             mesh,                //
                                             n,                   //
                                             stride,              //
                                             origin,              //
                                             delta,               //
                                             weighted_field,      //
                                             z_array_buffer,      //
                                             tet_indices_buffer,  //
                                             hex_field);          //
                        }
                    }
                }
            }
        }  // END if (z_array_buffer && tet_indices_buffer)

        free(z_array_buffer);
        z_array_buffer = NULL;

        free(tet_indices_buffer);
        tet_indices_buffer = NULL;
    }

    RETURN_FROM_FUNCTION(0);
}

/////////////////////////////////////////////////
// tet4_resample_field_adjoint_hex_quad_norm
/////////////////////////////////////////////////
int                                                                                              //
tet4_resample_field_adjoint_cell_quad(const ptrdiff_t                      start_element,        // Mesh
                                      const ptrdiff_t                      end_element,          //
                                      const mesh_t                        *mesh,                 //
                                      const ptrdiff_t *const SFEM_RESTRICT n,                    // SDF
                                      const ptrdiff_t *const SFEM_RESTRICT stride,               //
                                      const geom_t *const SFEM_RESTRICT    origin,               //
                                      const geom_t *const SFEM_RESTRICT    delta,                //
                                      const real_t *const SFEM_RESTRICT    weighted_field,       // Input weighted field
                                      const mini_tet_parameters_t          mini_tet_parameters,  //
                                      real_t *const SFEM_RESTRICT          data) {                        // SDF: data (output)

    PRINT_CURRENT_FUNCTION;

    boxes_t  *bounding_boxes_ptr = NULL;                          //
    const int fb_error           =                                //
            make_mesh_tets_boxes(0,                               //
                                 end_element,                     //
                                 mesh->nnodes,                    //
                                 (const idx_t **)mesh->elements,  //
                                 (const geom_t **)mesh->points,   //
                                 &bounding_boxes_ptr);            //

    bounding_box_statistics_t stats = calculate_bounding_box_statistics(bounding_boxes_ptr);
    print_bounding_box_statistics(&stats);

    side_length_histograms_t histograms =                         //
            calculate_side_length_histograms(bounding_boxes_ptr,  //
                                             &stats,              //
                                             50);                 //
    print_side_length_histograms(&histograms);

    side_length_cdf_thresholds_t thresholds =                         //
            calculate_cdf_thresholds(&histograms, 0.96, 0.96, 0.96);  //

    mesh_tet_geom_t *geom = mesh_tet_geometry_alloc(mesh);

    mesh_tet_geometry_compute_inv_Jacobian(geom);

    const real_t min_grid_x = origin[0];
    const real_t min_grid_y = origin[1];
    const real_t min_grid_z = origin[2];

    const real_t max_grid_x = origin[0] + delta[0] * n[0];
    const real_t max_grid_y = origin[1] + delta[1] * n[1];
    const real_t max_grid_z = origin[2] + delta[2] * n[2];

    cell_list_split_3d_2d_map_t *split_map = NULL;

    build_cell_list_3d_2d_split_map(&split_map,                     //
                                    thresholds.threshold_x,         //
                                    thresholds.threshold_y,         //
                                    bounding_boxes_ptr->min_x,      //
                                    bounding_boxes_ptr->min_y,      //
                                    bounding_boxes_ptr->min_z,      //
                                    bounding_boxes_ptr->max_x,      //
                                    bounding_boxes_ptr->max_y,      //
                                    bounding_boxes_ptr->max_z,      //
                                    bounding_boxes_ptr->num_boxes,  //
                                    min_grid_x,                     //
                                    max_grid_x,                     //
                                    min_grid_y,                     //
                                    max_grid_y,                     //
                                    min_grid_z,                     //
                                    max_grid_z);                    //

    transfer_to_hex_field_cell_tet4(split_map,           //
                                    bounding_boxes_ptr,  //
                                    geom,                //
                                    mesh,                //
                                    n,                   //
                                    stride,              //
                                    origin,              //
                                    delta,               //
                                    weighted_field,      //
                                    data);               //

    ///////// FREE RESOURCES /////////
    free_cell_list_split_3d_2d_map(split_map);
    split_map = NULL;

    mesh_tet_geometry_free(geom);
    geom = NULL;

    free_boxes_t(bounding_boxes_ptr);
    bounding_boxes_ptr = NULL;

    free_side_length_histograms(&histograms);

    RETURN_FROM_FUNCTION(0);
}  // END Function: tet4_resample_field_adjoint_cell_quad