#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cell_arg_sort.h"
#include "cell_list_3d_map.h"

/////////////////////////////////////////////////
// build_cell_list_3d_2d_map
/////////////////////////////////////////////////
cell_list_split_3d_2d_map_t *make_empty_cell_list_split_3d_2d_map(void) {
    cell_list_split_3d_2d_map_t *split_map = (cell_list_split_3d_2d_map_t *)malloc(sizeof(cell_list_split_3d_2d_map_t));

    split_map->split_x   = 0.0;
    split_map->split_y   = 0.0;
    split_map->map_lower = NULL;
    split_map->map_upper = NULL;

    return split_map;
}

/////////////////////////////////////////////////
// free_cell_list_split_3d_2d_map
/////////////////////////////////////////////////
int free_cell_list_split_3d_2d_map(cell_list_split_3d_2d_map_t *split_map) {
    if (split_map == NULL) {
        return 0;
    }

    if (split_map->map_lower != NULL) {
        free_cell_list_3d_2d_map(split_map->map_lower);
        free(split_map->map_lower);
        split_map->map_lower = NULL;
    }

    if (split_map->map_upper != NULL) {
        free_cell_list_3d_2d_map(split_map->map_upper);
        free(split_map->map_upper);
        split_map->map_upper = NULL;
    }

    free(split_map);
    return 0;
}

/////////////////////////////////////////////////
// build_cell_list_3d_2d_split_map
/////////////////////////////////////////////////
int build_cell_list_3d_2d_split_map(cell_list_3d_2d_map_t *map_lower,  //
                                    cell_list_3d_2d_map_t *map_upper,  //
                                    const real_t           split_x,    //
                                    const real_t           split_y,    //
                                    const real_t          *box_min_x,  //
                                    const real_t          *box_min_y,  //
                                    const real_t          *box_min_z,  //
                                    const real_t          *box_max_x,  //
                                    const real_t          *box_max_y,  //
                                    const real_t          *box_max_z,  //
                                    const int              num_boxes,  //
                                    const real_t           x_min,      //
                                    const real_t           x_max,      //
                                    const real_t           y_min,      //
                                    const real_t           y_max,      //
                                    const real_t           z_min,      //
                                    const real_t           z_max) {              //

    real_t max_delta_lower_x = 0.0;
    real_t max_delta_lower_y = 0.0;
    real_t max_delta_lower_z = 0.0;

    map_lower->min_x = x_min;
    map_lower->min_y = y_min;
    map_lower->min_z = z_min;
    map_lower->max_x = x_max;
    map_lower->max_y = y_max;
    map_lower->max_z = z_max;

    map_upper->min_x = x_min;
    map_upper->min_y = y_min;
    map_upper->min_z = z_min;
    map_upper->max_x = x_max;
    map_upper->max_y = y_max;
    map_upper->max_z = z_max;

    for (int i = 0; i < num_boxes; i++) {
        if (box_min_x[i] < split_x && box_min_y[i] < split_y) {
            real_t dx = fabs(box_max_x[i] - box_min_x[i]);
            real_t dy = fabs(box_max_y[i] - box_min_y[i]);
            real_t dz = fabs(box_max_z[i] - box_min_z[i]);

            if (dx > max_delta_lower_x) max_delta_lower_x = dx;
            if (dy > max_delta_lower_y) max_delta_lower_y = dy;
            if (dz > max_delta_lower_z) max_delta_lower_z = dz;

        } else {
            real_t dx = fabs(box_max_x[i] - box_min_x[i]);
            real_t dy = fabs(box_max_y[i] - box_min_y[i]);
            real_t dz = fabs(box_max_z[i] - box_min_z[i]);

            if (dx > map_upper->delta_x) map_upper->delta_x = dx;
            if (dy > map_upper->delta_y) map_upper->delta_y = dy;
            if (dz > map_upper->delta_z) map_upper->delta_z = dz;
        }
    }
}