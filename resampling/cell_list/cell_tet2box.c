
#include "cell_tet2box.h"

#define MY_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MY_MIN(a, b) (((a) < (b)) ? (a) : (b))

///////////////////////////////////////////////////////////////////
// make_mesh_tets_boxes
///////////////////////////////////////////////////////////////////
int                                                                     //
make_mesh_tets_boxes(const ptrdiff_t                    start_element,  // Mesh
                     const ptrdiff_t                    end_element,    //
                     const ptrdiff_t                    nnodes,         //
                     const idx_t** const SFEM_RESTRICT  elems,          //
                     const geom_t** const SFEM_RESTRICT xyz,            //
                     boxes_t**                          boxes) {                                 //

    PRINT_CURRENT_FUNCTION;

    const ptrdiff_t num_elements = end_element - start_element;

    // Allocate memory for boxes
    boxes_t* const SFEM_RESTRICT boxes_loc_ptr = allocate_boxes_t((int)num_elements);

    for (ptrdiff_t element_i = start_element; element_i < end_element; element_i++) {
        idx_t ev[4];

        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][element_i];
        }  // END: for vq

#if SFEM_LOG_LEVEL >= 5
        if (element_i % 100000 == 0) {
            printf("*** Processing element %td / %td \n", element_i, end_element);
        }
#endif

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

        ptrdiff_t min_grid_x, max_grid_x;
        ptrdiff_t min_grid_y, max_grid_y;
        ptrdiff_t min_grid_z, max_grid_z;

        min_grid_x = MY_MIN(MY_MIN(x0_n, x1_n), MY_MIN(x2_n, x3_n));
        max_grid_x = MY_MAX(MY_MAX(x0_n, x1_n), MY_MAX(x2_n, x3_n));

        min_grid_y = MY_MIN(MY_MIN(y0_n, y1_n), MY_MIN(y2_n, y3_n));
        max_grid_y = MY_MAX(MY_MAX(y0_n, y1_n), MY_MAX(y2_n, y3_n));

        min_grid_z = MY_MIN(MY_MIN(z0_n, z1_n), MY_MIN(z2_n, z3_n));
        max_grid_z = MY_MAX(MY_MAX(z0_n, z1_n), MY_MAX(z2_n, z3_n));

        const ptrdiff_t box_index = element_i - start_element;

        boxes_loc_ptr->min_x[box_index] = min_grid_x;
        boxes_loc_ptr->max_x[box_index] = max_grid_x;
        boxes_loc_ptr->min_y[box_index] = min_grid_y;
        boxes_loc_ptr->max_y[box_index] = max_grid_y;
        boxes_loc_ptr->min_z[box_index] = min_grid_z;
        boxes_loc_ptr->max_z[box_index] = max_grid_z;

    }  // END: for element_i

    *boxes = (boxes_t*)boxes_loc_ptr;
    RETURN_FROM_FUNCTION(EXIT_SUCCESS);
}  // END: Function: make_mesh_tets_boxes