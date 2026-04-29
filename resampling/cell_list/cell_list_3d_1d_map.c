#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cell_arg_sort.h"
#include "cell_list_3d_1d_map.h"
#include "cell_list_3d_map.h"

// ////////////////////////////////////////////////
// // cmp_int
// ////////////////////////////////////////////////
// static int cmp_int(const void *a, const void *b)
// {
//     int x = *(const int *)a, y = *(const int *)b;
//     return (x > y) - (x < y);
// }

////////////////////////////////////////////////
// cmp_real_t
////////////////////////////////////////////////
static int cmp_real_t(const void *a, const void *b) {
    real_t x = *(const real_t *)a, y = *(const real_t *)b;
    return (x > y) - (x < y);
}

////////////////////////////////////////////////
// coord_to_grid_index
////////////////////////////////////////////////
static inline int coord_to_grid_index(real_t coord, real_t origin, real_t delta) { return (int)((coord - origin) / delta); }

////////////////////////////////////////////////
// make_empty_cell_list_3d_1d_map
////////////////////////////////////////////////
cell_list_3d_1d_map_t *make_empty_cell_list_3d_1d_map(void) {
    cell_list_3d_1d_map_t *map = (cell_list_3d_1d_map_t *)malloc(sizeof(cell_list_3d_1d_map_t));

    map->cell_ptr       = NULL;
    map->cell_dict      = NULL;
    map->lower_bounds_y = NULL;
    map->upper_bounds_y = NULL;

    map->total_num_dict_entries = 0;

    map->delta_x     = 0.0;
    map->delta_y     = 0.0;
    map->delta_z     = 0.0;
    map->num_cells_x = 0;
    // map->num_cells_y = 0;
    // map->num_cells_z = 0;

    map->min_x = 0.0;
    map->min_y = 0.0;
    map->min_z = 0.0;
    map->max_x = 0.0;
    map->max_y = 0.0;
    map->max_z = 0.0;

    return map;
}

////////////////////////////////////////////////
// free_cell_list_3d_1d_map
////////////////////////////////////////////////
void free_cell_list_3d_1d_map(cell_list_3d_1d_map_t *map) {
    if (map == NULL) {
        return;
    }

    if (map->cell_ptr != NULL) {
        free(map->cell_ptr);
        map->cell_ptr = NULL;
    }

    if (map->cell_dict != NULL) {
        free(map->cell_dict);
        map->cell_dict = NULL;
    }

    if (map->lower_bounds_y != NULL) {
        free(map->lower_bounds_y);
        map->lower_bounds_y = NULL;
    }

    if (map->upper_bounds_y != NULL) {
        free(map->upper_bounds_y);
        map->upper_bounds_y = NULL;
    }

    free(map);
}

////////////////////////////////////////////////
// build_cell_list_3d_1d_map
////////////////////////////////////////////////
int                                                          //
build_cell_list_3d_1d_map(cell_list_3d_1d_map_t *map,        //
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
                          const real_t           z_max)      //
{
    real_t max_delta_x = 0.0;
    real_t max_delta_y = 0.0;
    real_t max_delta_z = 0.0;

    map->min_x = x_min;
    map->min_y = y_min;
    map->min_z = z_min;
    map->max_x = x_max;
    map->max_y = y_max;
    map->max_z = z_max;

    for (int i = 0; i < num_boxes; i++) {
        real_t dx = fabs(box_max_x[i] - box_min_x[i]);
        real_t dy = fabs(box_max_y[i] - box_min_y[i]);
        real_t dz = fabs(box_max_z[i] - box_min_z[i]);

        if (dx > max_delta_x) max_delta_x = dx;
        if (dy > max_delta_y) max_delta_y = dy;
        if (dz > max_delta_z) max_delta_z = dz;
    }

#ifdef DEBUG_OUTPUT
    printf("Max box sizes: dx = %f, dy = %f, dz = %f\n", max_delta_x, max_delta_y, max_delta_z);
#endif

    map->delta_x = max_delta_x;
    map->delta_y = max_delta_y;
    map->delta_z = max_delta_z;

    map->num_cells_x = (int)ceil((x_max - x_min) / map->delta_x);
    // map->num_cells_y = (int)ceil((y_max - y_min) / map->delta_y);
    // map->num_cells_z = (int)ceil((z_max - z_min) / map->delta_z);

    map->cell_ptr = (int *)calloc(map->num_cells_x + 1, sizeof(int));

#ifdef DEBUG_OUTPUT
    printf("Number of cells: nx = %d, ny = %d, nz = %d\n", map->num_cells_x, 0, 0);
    printf("Total number of 1D cells: %d\n", map->num_cells_x);
#endif

    for (int i = 0; i < num_boxes; i++) {
        int ix_min = coord_to_grid_index(box_min_x[i], map->min_x, map->delta_x);
        int ix_max = coord_to_grid_index(box_max_x[i], map->min_x, map->delta_x);

        if (ix_min < 0) ix_min = 0;

        if (ix_max >= map->num_cells_x) ix_max = map->num_cells_x - 1;

        for (int ix = ix_min; ix <= ix_max; ix++) {
            map->cell_ptr[ix + 1] += 1;
        }
    }

    // Accumulate counts to make the lookup pointer array
    for (int i = 1; i <= map->num_cells_x; i++) {
        map->cell_ptr[i] += map->cell_ptr[i - 1];
    }

    const int total_num_dict_entries = map->cell_ptr[map->num_cells_x];
    map->total_num_dict_entries      = total_num_dict_entries;

    map->cell_dict      = (int *)malloc(total_num_dict_entries * sizeof(int));
    map->lower_bounds_y = (real_t *)calloc(total_num_dict_entries, sizeof(real_t));
    map->upper_bounds_y = (real_t *)calloc(total_num_dict_entries, sizeof(real_t));

    int *current_count = (int *)calloc(map->num_cells_x, sizeof(int));

    // Fill the cell dictionary array
    for (int i = 0; i < num_boxes; i++) {
        int ix_min = coord_to_grid_index(box_min_x[i], map->min_x, map->delta_x);
        int ix_max = coord_to_grid_index(box_max_x[i], map->min_x, map->delta_x);

        if (ix_min < 0) ix_min = 0;

        if (ix_max >= map->num_cells_x) ix_max = map->num_cells_x - 1;

        for (int ix = ix_min; ix <= ix_max; ix++) {
            const int cell_index    = ix;
            const int index_in_dict = map->cell_ptr[cell_index] + current_count[cell_index];

            map->cell_dict[index_in_dict]      = i;
            map->lower_bounds_y[index_in_dict] = box_min_y[i];
            map->upper_bounds_y[index_in_dict] = box_max_y[i];

            current_count[cell_index] += 1;
        }
    }  // END for (num_boxes)

    free(current_count);

    int     size_arg_indices = 2024;
    int    *arg_indices      = (int *)malloc(size_arg_indices * sizeof(int));
    real_t *buffer           = (real_t *)malloc(size_arg_indices * sizeof(real_t));
    int    *buffer_int       = (int *)malloc(size_arg_indices * sizeof(int));

    for (int cell_index = 0; cell_index < map->num_cells_x; cell_index++) {
        const int start_index = map->cell_ptr[cell_index];
        const int end_index   = map->cell_ptr[cell_index + 1];

        const int size = end_index - start_index;
        if (size == 0) continue;

        if (size > size_arg_indices) {
            size_arg_indices = size;
            arg_indices      = (int *)realloc(arg_indices, size_arg_indices * sizeof(int));
            buffer           = (real_t *)realloc(buffer, size_arg_indices * sizeof(real_t));
            buffer_int       = (int *)realloc(buffer_int, size_arg_indices * sizeof(int));
        }

        for (int i = 0; i < size; i++) arg_indices[i] = i;

        real_t *lower_y = &map->lower_bounds_y[start_index];

        argsort(arg_indices, lower_y, (size_t)size, sizeof(real_t), cmp_real_t);

        for (int i = 0; i < size; i++) {
            buffer_int[i] = map->cell_dict[start_index + arg_indices[i]];
        }

        for (int i = 0; i < size; i++) {
            map->cell_dict[start_index + i] = buffer_int[i];
        }

        for (int i = 0; i < size; i++) {
            buffer[i] = map->lower_bounds_y[start_index + arg_indices[i]];
        }

        for (int i = 0; i < size; i++) {
            map->lower_bounds_y[start_index + i] = buffer[i];
        }

        for (int i = 0; i < size; i++) {
            buffer[i] = map->upper_bounds_y[start_index + arg_indices[i]];
        }

        for (int i = 0; i < size; i++) {
            map->upper_bounds_y[start_index + i] = buffer[i];
        }
    }  // END for (cell_index)

    free(arg_indices);
    free(buffer);
    free(buffer_int);

    for (int cell_index = 0; cell_index < map->num_cells_x; cell_index++) {
        const int start_index = map->cell_ptr[cell_index];
        const int end_index   = map->cell_ptr[cell_index + 1];

        const int size = end_index - start_index;
        if (size == 0) continue;

        for (int i = 1; i < size; i++) {
            if (map->upper_bounds_y[start_index + i - 1] > map->upper_bounds_y[start_index + i])
                map->upper_bounds_y[start_index + i] = map->upper_bounds_y[start_index + i - 1];
        }
    }  // END for (cell_index)

#ifdef DEBUG_OUTPUT
    printf("\nCell 3D-1D list built successfully.\n");
#endif

    return EXIT_SUCCESS;
}  // END Function: build_cell_list_3d_1d_map

////////////////////////////////////////////////
// query_cell_list_3d_1d_map
////////////////////////////////////////////////
int query_cell_list_3d_1d_map(const cell_list_3d_1d_map_t *map, const boxes_t *boxes, const real_t x, const real_t y,
                              const real_t z, int **box_indices, int *num_boxes) {
    const int ix_tmp = coord_to_grid_index(x, map->min_x, map->delta_x);
    const int ix     = (ix_tmp < 0) ? 0 : (ix_tmp >= map->num_cells_x) ? map->num_cells_x - 1 : ix_tmp;

    const int cell_index = ix;

    const int start_index = map->cell_ptr[cell_index];
    const int end_index   = map->cell_ptr[cell_index + 1];

    const int num_boxes_local = end_index - start_index;

    int boxes_found = 0;

    if (num_boxes_local > 0) {
        *box_indices = (int *)malloc(num_boxes_local * sizeof(int));

        int lower_bound_index =
                lower_bound_generic(&map->upper_bounds_y[start_index], (size_t)num_boxes_local, sizeof(real_t), &y, cmp_real_t);

        int upper_bound_index =
                upper_bound_generic(&map->lower_bounds_y[start_index], (size_t)num_boxes_local, sizeof(real_t), &y, cmp_real_t);

        lower_bound_index =
                lower_bound_index < 0 ? 0 : (lower_bound_index > num_boxes_local ? num_boxes_local : lower_bound_index);
        upper_bound_index =
                upper_bound_index < 0 ? 0 : (upper_bound_index > num_boxes_local ? num_boxes_local : upper_bound_index);

        // #ifdef DEBUG_OUTPUT
        //         printf("* Query point y = %f\n", y);
        //         printf("* Lower bound index: %d, Upper bound index: %d\n", lower_bound_index, upper_bound_index);
        //         if (lower_bound_index < num_boxes_local)
        //         {
        //             printf("* At lower_bound_index: lower_y = %f, upper_y = %f\n",
        //                    map->lower_bounds_y[start_index + lower_bound_index],
        //                    map->upper_bounds_y[start_index + lower_bound_index]);
        //         }
        //         if (upper_bound_index > 0 && upper_bound_index <= num_boxes_local)
        //         {
        //             printf("* At upper_bound_index-1: lower_y = %f, upper_y = %f\n",
        //                    map->lower_bounds_y[start_index + upper_bound_index - 1],
        //                    map->upper_bounds_y[start_index + upper_bound_index - 1]);
        //         }

        //         // Check all boxes to find which ones should contain y
        //         int count_should_check = 0;
        //         for (int i = 0; i < num_boxes_local; i++)
        //         {
        //             if (map->lower_bounds_y[start_index + i] <= y && map->upper_bounds_y[start_index + i] >= y)
        //             {
        //                 count_should_check++;
        //             }
        //         }
        //         printf("* Number of boxes with y in range [lower_y, upper_y]: %d\n", count_should_check);
        // #endif

        // for (int i = 0; i < num_boxes_local; i++)
        for (int i = lower_bound_index; i < upper_bound_index; i++) {
            const int box_index = map->cell_dict[start_index + i];

            // #ifdef DEBUG_OUTPUT
            //             printf("  - Checking box index %10d: box_min_y = %f, box_max_y = %f, for point y = %f\n",
            //                    box_index,
            //                    boxes->min_y[box_index],
            //                    boxes->max_y[box_index],
            //                    y);
            //             printf("                                   box_min_z = %f, box_max_z = %f, for point z = %f\n",
            //                    boxes->min_z[box_index],
            //                    boxes->max_z[box_index],
            //                    z);
            //             printf("                                   box_min_x = %f, box_max_x = %f, for point x = %f\n",
            //                    boxes->min_x[box_index],
            //                    boxes->max_x[box_index],
            //                    x);
            // #endif

            if (check_box_contains_pt(boxes, box_index, x, y, z)) {
                (*box_indices)[boxes_found] = box_index;
                boxes_found++;
            }
        }
        *box_indices = (int *)realloc(*box_indices, boxes_found * sizeof(int));
        *num_boxes   = boxes_found;

        // #ifdef DEBUG_OUTPUT
        //         if (boxes_found == 0)
        //         {
        //             printf("* No boxes found in cell for point (%f, %f, %f). in cell_list_3d_1d_map.c\n", x, y, z);
        //             printf("* Number of boxes in cell: %d\n", num_boxes_local);
        //         }
        //         else
        //         {
        //             printf("* Found %d boxes in cell for point (%f, %f, %f). in cell_list_3d_1d_map.c\n", boxes_found, x, y,
        //             z);
        //         }
        // #endif
    } else {
        // // #ifdef DEBUG_OUTPUT
        //         printf("No boxes found in cell for point (%f, %f, %f). in cell_list_3d_1d_map.c\n", x, y, z);
        // // #endif
        *box_indices = NULL;
        *num_boxes   = 0;
    }

    return EXIT_SUCCESS;
}

////////////////////////////////////////////////
// query_cell_list_3d_1d_map_given_xy
////////////////////////////////////////////////
int query_cell_list_3d_1d_map_given_xy(
        const cell_list_3d_1d_map_t *map, const boxes_t *boxes, const real_t x, const real_t y, const real_t *z_array,
        const int size_z,
        int    ***box_indices,  // it produces a pointer of a vector (size_z) of vector(size_boxes_local)
        int     **num_boxes) {
    const int ix_tmp = coord_to_grid_index(x, map->min_x, map->delta_x);
    const int ix     = (ix_tmp < 0) ? 0 : (ix_tmp >= map->num_cells_x) ? map->num_cells_x - 1 : ix_tmp;

    const int cell_index = ix;

    const int start_index = map->cell_ptr[cell_index];
    const int end_index   = map->cell_ptr[cell_index + 1];

    const int num_boxes_local = end_index - start_index;

    if (num_boxes_local > 0) {
        (*num_boxes)   = (int *)malloc(size_z * sizeof(int));
        (*box_indices) = (int **)malloc(size_z * sizeof(int *));

        for (int iz = 0; iz < size_z; iz++) {
            (*box_indices)[iz] = (int *)malloc(num_boxes_local * sizeof(int));
        }

        int lower_bound_index = lower_bound_float(&map->upper_bounds_y[start_index], (size_t)num_boxes_local, y);

        const int start_index_up = (lower_bound_index > 1) ? start_index + lower_bound_index - 2 : start_index;
        const int size_up        = (lower_bound_index > 1) ? num_boxes_local - (lower_bound_index - 2) : num_boxes_local;
        const int offset_up      = start_index_up - start_index;

        // int upper_bound_index = upper_bound_float(
        // &map->lower_bounds_y[start_index],
        // (size_t)num_boxes_local,
        // y);

        int upper_bound_index = upper_bound_float(     //
                &map->lower_bounds_y[start_index_up],  //
                size_up,                               //
                y);                                    //

        // Adjust upper_bound_index back to be relative to start_index
        upper_bound_index += offset_up;

        lower_bound_index =
                lower_bound_index < 0 ? 0 : (lower_bound_index > num_boxes_local ? num_boxes_local : lower_bound_index);
        upper_bound_index =
                upper_bound_index < 0 ? 0 : (upper_bound_index > num_boxes_local ? num_boxes_local : upper_bound_index);

        // const int *cell_dict_local = &map->cell_dict[start_index];
        for (int iz = 0; iz < size_z; iz++) {
            const real_t z                 = z_array[iz];
            int          boxes_found_local = 0;

            // for (int i = 0; i < num_boxes_local; i++)

            for (int i = lower_bound_index; i < upper_bound_index; i++) {
                const int box_index = map->cell_dict[start_index + i];

                if (check_box_contains_pt(boxes, box_index, x, y, z)) {
                    (*box_indices)[iz][boxes_found_local] = box_index;
                    boxes_found_local++;
                }
            }
            (*box_indices)[iz] = (int *)realloc((*box_indices)[iz], boxes_found_local * sizeof(int));
            (*num_boxes)[iz]   = boxes_found_local;
        }
    } else {
        for (int iz = 0; iz < size_z; iz++) {
            *box_indices = NULL;
            *num_boxes   = NULL;
        }
    }

    return EXIT_SUCCESS;
}

////////////////////////////////////////////////
// cell_list_3d_1d_map_bytes
////////////////////////////////////////////////
int64_t cell_list_3d_1d_map_bytes(const cell_list_3d_1d_map_t *map) {
    int64_t total_bytes = 0;

    total_bytes += (int64_t)(map->cell_ptr != NULL ? (map->num_cells_x + 1) * sizeof(int) : 0);
    total_bytes += (int64_t)(map->cell_dict != NULL ? map->total_num_dict_entries * sizeof(int) : 0);
    total_bytes += (int64_t)(map->lower_bounds_y != NULL ? map->total_num_dict_entries * sizeof(real_t) : 0);
    total_bytes += (int64_t)(map->upper_bounds_y != NULL ? map->total_num_dict_entries * sizeof(real_t) : 0);

    return total_bytes;
}