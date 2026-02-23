#include "sfem_resample_field_adjoint_cell.h"
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

// const int                            mpi_size,      // MPI size
//                             const int                            mpi_rank,      // MPI rank
//                             const mesh_t* const SFEM_RESTRICT    mesh,          // Mesh: mesh_t struct
//                             const ptrdiff_t* const SFEM_RESTRICT n,             // SDF: n[3]
//                             const ptrdiff_t* const SFEM_RESTRICT stride,        // SDF: stride[3]
//                             const geom_t* const SFEM_RESTRICT    origin,        // SDF: origin[3]
//                             const geom_t* const SFEM_RESTRICT    delta,         // SDF: delta[3]
//                             const real_t* const SFEM_RESTRICT    g,             // Weighted field
//                             const function_XYZ_t                 fun_XYZ,       // Function to apply
//                             real_t* const SFEM_RESTRICT          data,          // SDF: data (output)
//                             unsigned int*                        data_cnt,      // SDF: data count (output)
//                             real_t const*                        alpha,         // SDF: tet alpha
//                             real_t const*                        volume,        // SDF: tet volume
//                             real_t const*                        data_fun_XYZ,  // SDF: data for fun_XYZ
//                             sfem_resample_field_info*            info,          // Info struct with options and flags
//                             const mini_tet_parameters_t          mini_tet_parameters

//////////////////////////////////////////////
// update_hex_field
//////////////////////////////////////////////
int                                                               //
update_hex_field(const int                            mpi_size,   // MPI size
                 const int                            mpi_rank,   // MPI rank
                 cell_list_split_3d_2d_map_t         *split_map,  // Cell list split map data structure
                 const ptrdiff_t                      i_grid,     // The i index of the grid point in the hex mesh
                 const ptrdiff_t                      j_grid,     // The j index of the grid point in the hex mesh
                 const mesh_t *const SFEM_RESTRICT    mesh,       // Mesh: mesh_t struct
                 const ptrdiff_t *const SFEM_RESTRICT n,          // SDF: n[3]
                 const ptrdiff_t *const SFEM_RESTRICT stride,     // SDF: stride[3]
                 const geom_t *const SFEM_RESTRICT    origin,     // SDF: origin[3]
                 const geom_t *const SFEM_RESTRICT    delta,      // SDF: delta[3]
                 const real_t *const SFEM_RESTRICT    tet_g,      // Weighted field
                 real_t *const SFEM_RESTRICT          hex_field) {         // Output field for the hex cell containing (x,y)

    const real_t grid_x = origin[0] + i_grid * delta[0];
    const real_t grid_y = origin[1] + j_grid * delta[1];

    const real_t *quad_x = get_Q_nodes_x_p();
    const real_t *quad_y = get_Q_nodes_y_p();
    const real_t *quad_z = get_Q_nodes_z_p();
    const real_t *quad_w = get_Q_weights_p();
    const int     dim_q  = get_dim_qad();

    const real_t hex_volume = delta[0] * delta[1] * delta[2];

    real_t *z_array     = malloc(n[2] * sizeof(real_t));
    int    *tet_indices = malloc(n[2] * sizeof(int));

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
        for (ptrdiff_t k = 0; k < n[2]; k++) {
            z_array[k] = phys_z_base + (real_t)k * delta_z;
        }
    }

    free(z_array);
    z_array = NULL;

    free(tet_indices);
    tet_indices = NULL;

    return 0;
}