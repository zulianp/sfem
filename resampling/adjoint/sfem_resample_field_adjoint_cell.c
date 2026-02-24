#include "sfem_resample_field_adjoint_cell.h"
#include "cell_list_3d_map_mesh.h"
#include "sfem_resample_field_adjoint_hex_quad.h"

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
update_hex_quad_node(const int                            mpi_size,        // MPI size
                     const int                            mpi_rank,        // MPI rank
                     const real_t                         x,               // Physical x coordinate of the quadrature point
                     const real_t                         y,               // Physical y coordinate of the quadrature point
                     const real_t                         z,               // Physical z coordinate of the quadrature point
                     const real_t                         phys_w,          // Quadrature weight for the quadrature point
                     const ptrdiff_t                      index_tet,       // The index of the tet containing the quadrature point
                     const mesh_t *const SFEM_RESTRICT    mesh,            // Mesh: mesh_t struct
                     mesh_tet_geom_t                     *mesh_geom,       // Mesh geometry data structure
                     const ptrdiff_t *const SFEM_RESTRICT n,               // SDF: n[3]
                     const ptrdiff_t *const SFEM_RESTRICT stride,          // SDF: stride[3]
                     const geom_t *const SFEM_RESTRICT    origin,          // SDF: origin[3]
                     const geom_t *const SFEM_RESTRICT    delta,           // SDF: delta[3]
                     const real_t *const SFEM_RESTRICT    weighted_field,  // Weighted field
                     real_t *const SFEM_RESTRICT          hex_element_field) {      // Output field values for the 8 hex nodes

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

    const real_t dx = delta[0];
    const real_t dy = delta[1];
    const real_t dz = delta[2];

    const real_t grid_x = (x - ox) / dx;
    const real_t grid_y = (y - oy) / dy;
    const real_t grid_z = (z - oz) / dz;

    const ptrdiff_t i = floor(grid_x);
    const ptrdiff_t j = floor(grid_y);
    const ptrdiff_t k = floor(grid_z);

    const real_t l_x = (grid_x - (real_t)i);
    const real_t l_y = (grid_y - (real_t)j);
    const real_t l_z = (grid_z - (real_t)k);

    const ptrdiff_t base_index = i * 1 +          //
                                 j * stride[1] +  //
                                 k * stride[2];   //

    idx_t ev[4];

    for (int v = 0; v < 4; ++v) {
        ev[v] = mesh->elements[v][index_tet];
    }  // END: for vq

    const real_t wf0 = weighted_field[ev[0]];  // Weighted field at vertex 0
    const real_t wf1 = weighted_field[ev[1]];  // Weighted field at vertex 1
    const real_t wf2 = weighted_field[ev[2]];  // Weighted field at vertex 2
    const real_t wf3 = weighted_field[ev[3]];  // Weighted field at vertex 3

    const real_t *inv_J_tet = &(mesh_geom->inv_Jacobian[index_tet * 9]);  // Inverse Jacobian for the current tet

    const real_t x0_n = mesh_geom->vetices_zero[ev[0] * 3 + 0];  // x coordinate of vertex 0
    const real_t y0_n = mesh_geom->vetices_zero[ev[0] * 3 + 1];  // y coordinate of vertex 0
    const real_t z0_n = mesh_geom->vetices_zero[ev[0] * 3 + 2];  // z coordinate of vertex 0

    // Compute the coordinates of the quadrature point in the reference tetrahedron using the inverse Jacobian transformation.
    const real_t x_o = x - x0_n;
    const real_t y_o = y - y0_n;
    const real_t z_o = z - z0_n;

    const real_t x_ref = inv_J_tet[0] * x_o + inv_J_tet[1] * y_o + inv_J_tet[2] * z_o;
    const real_t y_ref = inv_J_tet[3] * x_o + inv_J_tet[4] * y_o + inv_J_tet[5] * z_o;
    const real_t z_ref = inv_J_tet[6] * x_o + inv_J_tet[7] * y_o + inv_J_tet[8] * z_o;

    const real_t f0 = 1.0 - x_ref - y_ref - z_ref;
    const real_t f1 = x_ref;
    const real_t f2 = y_ref;
    const real_t f3 = z_ref;

    const real_t wf_quad = f0 * wf0 + f1 * wf1 + f2 * wf2 + f3 * wf3;

    const real_t hex8_f0 = (1.0 - l_x) * (1.0 - l_y) * (1.0 - l_z);
    const real_t hex8_f1 = l_x * (1.0 - l_y) * (1.0 - l_z);
    const real_t hex8_f2 = l_x * l_y * (1.0 - l_z);
    const real_t hex8_f3 = (1.0 - l_x) * l_y * (1.0 - l_z);
    const real_t hex8_f4 = (1.0 - l_x) * (1.0 - l_y) * l_z;
    const real_t hex8_f5 = l_x * (1.0 - l_y) * l_z;
    const real_t hex8_f6 = l_x * l_y * l_z;
    const real_t hex8_f7 = (1.0 - l_x) * l_y * l_z;

    const real_t wf_quad_QW = wf_quad * phys_w;

    hex_element_field[off0] += wf_quad_QW * hex8_f0;
    hex_element_field[off1] += wf_quad_QW * hex8_f1;
    hex_element_field[off2] += wf_quad_QW * hex8_f2;
    hex_element_field[off3] += wf_quad_QW * hex8_f3;
    hex_element_field[off4] += wf_quad_QW * hex8_f4;
    hex_element_field[off5] += wf_quad_QW * hex8_f5;
    hex_element_field[off6] += wf_quad_QW * hex8_f6;
    hex_element_field[off7] += wf_quad_QW * hex8_f7;
}

//////////////////////////////////////////////
// update_hex_field
//////////////////////////////////////////////
int                                                                    //
update_hex_field(const int                            mpi_size,        // MPI size
                 const int                            mpi_rank,        // MPI rank
                 cell_list_split_3d_2d_map_t         *split_map,       // Cell list split map data structure
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

        for (ptrdiff_t k = 0; k < z_size; k++) {
            if (tet_indices[k] < 0) {
                continue;  // No tet found for this z level, skip to the next
            }

            update_hex_quad_node(mpi_size,        //
                                 mpi_rank,        //
                                 phys_x,          //
                                 phys_y,          //
                                 z_array[k],      //
                                 phys_w,          //
                                 tet_indices[k],  //
                                 mesh,            //
                                 mesh_geom,       //
                                 n,               //
                                 stride,          //
                                 origin,          //
                                 delta,           //
                                 weighted_field,  //
                                 hex_field);      //

            // Now we have the tet index for the current quadrature point (phys_x, phys_y, z_array[k])
            // We can use this tet index to get the corresponding value from tet_g and accumulate it.
        }
    }

    // Free allocated arrays
    free(z_array);
    z_array = NULL;

    free(tet_indices);
    tet_indices = NULL;

    return 0;
}