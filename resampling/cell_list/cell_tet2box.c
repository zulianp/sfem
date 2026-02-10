
#include "cell_tet2box.h"

#include <math.h>

#define MY_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MY_MIN(a, b) (((a) < (b)) ? (a) : (b))

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// print_bounding_box_statistics
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void print_bounding_box_statistics(const boxes_t *boxes) {
    if (boxes == NULL || boxes->num_boxes <= 0) {
        return;
    }  // END if (boxes == NULL || boxes->num_boxes <= 0)

    printf("\n== Bounding box statistics: ==\n");

    real_t max_box_side_x = 0.0;
    real_t max_box_side_y = 0.0;
    real_t max_box_side_z = 0.0;

    real_t min_box_side_x = INFINITY;
    real_t min_box_side_y = INFINITY;
    real_t min_box_side_z = INFINITY;

    real_t sum_box_side_x = 0.0;
    real_t sum_box_side_y = 0.0;
    real_t sum_box_side_z = 0.0;

    real_t max_volume = 0.0;
    real_t min_volume = INFINITY;
    real_t sum_volume = 0.0;
    int max_volume_idx = -1;
    int min_volume_idx = -1;

    real_t max_volume_side_x, max_volume_side_y, max_volume_side_z;
    real_t min_volume_side_x, min_volume_side_y, min_volume_side_z;

    for (int i = 0; i < boxes->num_boxes; i++) {
        const real_t side_x = boxes->max_x[i] - boxes->min_x[i];
        const real_t side_y = boxes->max_y[i] - boxes->min_y[i];
        const real_t side_z = boxes->max_z[i] - boxes->min_z[i];

        const real_t volume = side_x * side_y * side_z;

        max_box_side_x = MY_MAX(max_box_side_x, side_x);
        max_box_side_y = MY_MAX(max_box_side_y, side_y);
        max_box_side_z = MY_MAX(max_box_side_z, side_z);

        min_box_side_x = MY_MIN(min_box_side_x, side_x);
        min_box_side_y = MY_MIN(min_box_side_y, side_y);
        min_box_side_z = MY_MIN(min_box_side_z, side_z);

        sum_box_side_x += side_x;
        sum_box_side_y += side_y;
        sum_box_side_z += side_z;

        if (volume > max_volume) {
            max_volume = volume;
            max_volume_idx = i;
            max_volume_side_x = side_x;
            max_volume_side_y = side_y;
            max_volume_side_z = side_z;
        }  // END if (volume > max_volume)

        if (volume < min_volume) {
            min_volume = volume;
            min_volume_idx = i;
            min_volume_side_x = side_x;
            min_volume_side_y = side_y;
            min_volume_side_z = side_z;
        }  // END if (volume < min_volume)

        sum_volume += volume;
    }  // END: for i

    printf("Largest bounding box sides: dX=%g, dY=%g, dZ=%g\n",
           (double)max_box_side_x,
           (double)max_box_side_y,
           (double)max_box_side_z);

    printf("Smallest bounding box sides: dX=%g, dY=%g, dZ=%g\n",
           (double)min_box_side_x,
           (double)min_box_side_y,
           (double)min_box_side_z);

    printf("Average bounding box sides: dX=%g, dY=%g, dZ=%g\n",
           (double)(sum_box_side_x / boxes->num_boxes),
           (double)(sum_box_side_y / boxes->num_boxes),
           (double)(sum_box_side_z / boxes->num_boxes));

    printf("Largest bounding box volume: %g\n", (double)max_volume);
    printf("*  Box index: %d, Sides: dX=%g, dY=%g, dZ=%g\n", max_volume_idx, (double)max_volume_side_x, (double)max_volume_side_y, (double)max_volume_side_z);
    printf("Smallest bounding box volume: %g\n", (double)min_volume);
    printf("*  Box index: %d, Sides: dX=%g, dY=%g, dZ=%g\n", min_volume_idx, (double)min_volume_side_x, (double)min_volume_side_y, (double)min_volume_side_z);
    printf("Average bounding box volume: %g\n", (double)(sum_volume / boxes->num_boxes));
    printf("Volume ratio (max/min): %g\n", (double)(max_volume / min_volume));

    printf("\n== End of Bounding box statistics: ==\n");

}  // END Function: print_bounding_box_statistics

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
// make_mesh_tets_boxes
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
int                                                                     //
make_mesh_tets_boxes(const ptrdiff_t                    start_element,  //
                     const ptrdiff_t                    end_element,    //
                     const ptrdiff_t                    nnodes,         //
                     const idx_t** const SFEM_RESTRICT  elems,          //
                     const geom_t** const SFEM_RESTRICT xyz,            //
                     boxes_t**                          boxes) {        //

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
        if (element_i % 1000000 == 0) {
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

        const real_t min_x = MY_MIN(MY_MIN(x0_n, x1_n), MY_MIN(x2_n, x3_n));
        const real_t max_x = MY_MAX(MY_MAX(x0_n, x1_n), MY_MAX(x2_n, x3_n));

        const real_t min_y = MY_MIN(MY_MIN(y0_n, y1_n), MY_MIN(y2_n, y3_n));
        const real_t max_y = MY_MAX(MY_MAX(y0_n, y1_n), MY_MAX(y2_n, y3_n));

        const real_t min_z = MY_MIN(MY_MIN(z0_n, z1_n), MY_MIN(z2_n, z3_n));
        const real_t max_z = MY_MAX(MY_MAX(z0_n, z1_n), MY_MAX(z2_n, z3_n));

        const ptrdiff_t box_index = element_i - start_element;

        boxes_loc_ptr->min_x[box_index] = min_x;
        boxes_loc_ptr->max_x[box_index] = max_x;
        boxes_loc_ptr->min_y[box_index] = min_y;
        boxes_loc_ptr->max_y[box_index] = max_y;
        boxes_loc_ptr->min_z[box_index] = min_z;
        boxes_loc_ptr->max_z[box_index] = max_z;

    }  // END: for element_i

    *boxes = (boxes_t*)boxes_loc_ptr;

    print_bounding_box_statistics(boxes_loc_ptr);

    RETURN_FROM_FUNCTION(EXIT_SUCCESS);
}  // END: Function: make_mesh_tets_boxes