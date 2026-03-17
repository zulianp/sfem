#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cell_arg_sort.h"
#include "cell_list_3d_map.h"

////////////////////////////////////////////////
// coord_to_grid_index
////////////////////////////////////////////////
static inline int coord_to_grid_index(real_t coord, real_t origin, real_t delta) { return (int)((coord - origin) / delta); }

////////////////////////////////////////////////
// grid_to_cell_index
////////////////////////////////////////////////
static inline int grid_to_cell_index(int ix, int iy, int num_cells_x) { return ix + iy * num_cells_x; }

////////////////////////////////////////////////
// cmp_real_t
////////////////////////////////////////////////
static int cmp_real_t(const void *a, const void *b) {
    real_t x = *(const real_t *)a, y = *(const real_t *)b;
    return (x > y) - (x < y);
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// increment_cell_counts_for_box
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
static void increment_cell_counts_for_box(cell_list_3d_2d_map_t *map,          //
                                          const real_t           box_min_x,    //
                                          const real_t           box_min_y,    //
                                          const real_t           box_max_x,    //
                                          const real_t           box_max_y) {  //
                                                                               //
    int ix_min = coord_to_grid_index(box_min_x, map->min_x, map->delta_x);
    int iy_min = coord_to_grid_index(box_min_y, map->min_y, map->delta_y);
    int ix_max = coord_to_grid_index(box_max_x, map->min_x, map->delta_x);
    int iy_max = coord_to_grid_index(box_max_y, map->min_y, map->delta_y);

    if (ix_min < 0) ix_min = 0;
    if (iy_min < 0) iy_min = 0;
    if (ix_max >= map->num_cells_x) ix_max = map->num_cells_x - 1;
    if (iy_max >= map->num_cells_y) iy_max = map->num_cells_y - 1;

    for (int iy = iy_min; iy <= iy_max; iy++) {
        for (int ix = ix_min; ix <= ix_max; ix++) {
            const int cell_index = grid_to_cell_index(ix, iy, map->num_cells_x);
            map->cell_ptr[cell_index + 1] += 1;
        }  // END for (ix)
    }  // END for (iy)
}  // END Function: increment_cell_counts_for_box

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// allocate_map_dict_entries
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
static void allocate_map_dict_entries(cell_list_3d_2d_map_t *map,                   //
                                      const int              total_num_2d_cells) {  //

    const int total_num_dict_entries = map->cell_ptr[total_num_2d_cells];
    map->total_num_dict_entries      = total_num_dict_entries;

    map->cell_dict      = (int *)malloc(total_num_dict_entries * sizeof(int));
    map->lower_bounds_z = (real_t *)calloc(total_num_dict_entries, sizeof(real_t));
    map->upper_bounds_z = (real_t *)calloc(total_num_dict_entries, sizeof(real_t));
}  // END Function: allocate_map_dict_entries

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// fill_cell_dict_for_box
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
static void fill_cell_dict_for_box(cell_list_3d_2d_map_t *map,            //
                                   int                   *current_count,  //
                                   const int              box_idx,        //
                                   const real_t           box_min_x,      //
                                   const real_t           box_min_y,      //
                                   const real_t           box_min_z,      //
                                   const real_t           box_max_x,      //
                                   const real_t           box_max_y,      //
                                   const real_t           box_max_z) {    //
                                                                          //
    int ix_min = coord_to_grid_index(box_min_x, map->min_x, map->delta_x);
    int iy_min = coord_to_grid_index(box_min_y, map->min_y, map->delta_y);
    int ix_max = coord_to_grid_index(box_max_x, map->min_x, map->delta_x);
    int iy_max = coord_to_grid_index(box_max_y, map->min_y, map->delta_y);

    if (ix_min < 0) ix_min = 0;
    if (iy_min < 0) iy_min = 0;
    if (ix_max >= map->num_cells_x) ix_max = map->num_cells_x - 1;
    if (iy_max >= map->num_cells_y) iy_max = map->num_cells_y - 1;

    for (int iy = iy_min; iy <= iy_max; iy++) {
        for (int ix = ix_min; ix <= ix_max; ix++) {
            const int cell_index    = grid_to_cell_index(ix, iy, map->num_cells_x);
            const int index_in_dict = map->cell_ptr[cell_index] + current_count[cell_index];

            map->cell_dict[index_in_dict]      = box_idx;
            map->lower_bounds_z[index_in_dict] = box_min_z;
            map->upper_bounds_z[index_in_dict] = box_max_z;

            current_count[cell_index] += 1;
        }  // END for (ix)
    }  // END for (iy)
}  // END Function: fill_cell_dict_for_box

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// sort_cell_entries_by_z
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
static void sort_cell_entries_by_z(cell_list_3d_2d_map_t *map,                   //
                                   const int              total_num_2d_cells) {  //
                                                                                 //
    int     size_arg_indices = 2024;
    int    *arg_indices      = (int *)malloc(size_arg_indices * sizeof(int));
    real_t *buffer           = (real_t *)malloc(size_arg_indices * sizeof(real_t));
    int    *buffer_int       = (int *)malloc(size_arg_indices * sizeof(int));

    for (int cell_index = 0; cell_index < total_num_2d_cells; cell_index++) {
        const int start_index = map->cell_ptr[cell_index];
        const int end_index   = map->cell_ptr[cell_index + 1];

        const int size = end_index - start_index;
        if (size == 0) continue;

        if (size > size_arg_indices) {
            size_arg_indices = size;
            arg_indices      = (int *)realloc(arg_indices, size_arg_indices * sizeof(int));
            buffer           = (real_t *)realloc(buffer, size_arg_indices * sizeof(real_t));
            buffer_int       = (int *)realloc(buffer_int, size_arg_indices * sizeof(int));
        }  // END if (size > size_arg_indices)

        for (int i = 0; i < size; i++) arg_indices[i] = i;

        real_t *lower_z = &map->lower_bounds_z[start_index];

        argsort(arg_indices, lower_z, (size_t)size, sizeof(real_t), cmp_real_t);

        for (int i = 0; i < size; i++) {
            buffer_int[i] = map->cell_dict[start_index + arg_indices[i]];
        }  // END for (i)

        for (int i = 0; i < size; i++) {
            map->cell_dict[start_index + i] = buffer_int[i];
        }  // END for (i)

        for (int i = 0; i < size; i++) {
            buffer[i] = map->lower_bounds_z[start_index + arg_indices[i]];
        }  // END for (i)

        for (int i = 0; i < size; i++) {
            map->lower_bounds_z[start_index + i] = buffer[i];
        }  // END for (i)

        for (int i = 0; i < size; i++) {
            buffer[i] = map->upper_bounds_z[start_index + arg_indices[i]];
        }  // END for (i)

        for (int i = 0; i < size; i++) {
            map->upper_bounds_z[start_index + i] = buffer[i];
        }  // END for (i)
    }  // END for (cell_index)

    free(arg_indices);
    free(buffer);
    free(buffer_int);
}  // END Function: sort_cell_entries_by_z

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// ensure_upper_bounds_non_decreasing
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
static void ensure_upper_bounds_non_decreasing(cell_list_3d_2d_map_t *map,                   //
                                               const int              total_num_2d_cells) {  //
                                                                                             //
    for (int cell_index = 0; cell_index < total_num_2d_cells; cell_index++) {
        const int start_index = map->cell_ptr[cell_index];
        const int end_index   = map->cell_ptr[cell_index + 1];

        const int size = end_index - start_index;
        if (size == 0) continue;

        for (int i = 1; i < size; i++) {
            if (map->upper_bounds_z[start_index + i - 1] > map->upper_bounds_z[start_index + i]) {
                map->upper_bounds_z[start_index + i] = map->upper_bounds_z[start_index + i - 1];
            }  // END if (upper bound needs adjustment)
        }  // END for (i)
    }  // END for (cell_index)
}  // END Function: ensure_upper_bounds_non_decreasing

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
        split_map->map_lower = NULL;
    }

    if (split_map->map_upper != NULL) {
        free_cell_list_3d_2d_map(split_map->map_upper);
        split_map->map_upper = NULL;
    }

    free(split_map);
    return 0;
}

/////////////////////////////////////////////////
// fill_cell_lists_3d_2d_split_map
/////////////////////////////////////////////////
int fill_cell_lists_3d_2d_split_map(cell_list_3d_2d_map_t *map_lower,  //
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
                                    const real_t           z_max) {    //

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
        real_t dx = fabs(box_max_x[i] - box_min_x[i]);
        real_t dy = fabs(box_max_y[i] - box_min_y[i]);
        real_t dz = fabs(box_max_z[i] - box_min_z[i]);

        if (dx < split_x && dy < split_y) {
            if (dx > max_delta_lower_x) max_delta_lower_x = dx;
            if (dy > max_delta_lower_y) max_delta_lower_y = dy;
            if (dz > max_delta_lower_z) max_delta_lower_z = dz;

        } else {
            if (dx > map_upper->delta_x) map_upper->delta_x = dx;
            if (dy > map_upper->delta_y) map_upper->delta_y = dy;
            if (dz > map_upper->delta_z) map_upper->delta_z = dz;
        }
    }

    // Assign computed max deltas to map_lower
    map_lower->delta_x = max_delta_lower_x;
    map_lower->delta_y = max_delta_lower_y;
    map_lower->delta_z = max_delta_lower_z;

    map_lower->inv_delta_x = (max_delta_lower_x > 0) ? 1.0 / max_delta_lower_x : 0.0;
    map_lower->inv_delta_y = (max_delta_lower_y > 0) ? 1.0 / max_delta_lower_y : 0.0;
    map_lower->inv_delta_z = (max_delta_lower_z > 0) ? 1.0 / max_delta_lower_z : 0.0;

    map_upper->inv_delta_x = (map_upper->delta_x > 0) ? 1.0 / map_upper->delta_x : 0.0;
    map_upper->inv_delta_y = (map_upper->delta_y > 0) ? 1.0 / map_upper->delta_y : 0.0;
    map_upper->inv_delta_z = (map_upper->delta_z > 0) ? 1.0 / map_upper->delta_z : 0.0;

    map_lower->num_cells_x = (int)ceil((x_max - x_min) / map_lower->delta_x);
    map_lower->num_cells_y = (int)ceil((y_max - y_min) / map_lower->delta_y);
    map_lower->num_cells_z = (int)ceil((z_max - z_min) / map_lower->delta_z);

    map_upper->num_cells_x = (int)ceil((x_max - x_min) / map_upper->delta_x);
    map_upper->num_cells_y = (int)ceil((y_max - y_min) / map_upper->delta_y);
    map_upper->num_cells_z = (int)ceil((z_max - z_min) / map_upper->delta_z);

    const int total_num_2d_cells_lower = map_lower->num_cells_x * map_lower->num_cells_y;
    const int total_num_2d_cells_upper = map_upper->num_cells_x * map_upper->num_cells_y;

    map_lower->cell_ptr           = (int *)calloc(total_num_2d_cells_lower + 1, sizeof(int));
    map_lower->total_num_2d_cells = total_num_2d_cells_lower;

    map_upper->cell_ptr           = (int *)calloc(total_num_2d_cells_upper + 1, sizeof(int));
    map_upper->total_num_2d_cells = total_num_2d_cells_upper;

#ifdef DEBUG_OUTPUT
    printf("Max box sizes lower: dx = %f, dy = %f, dz = %f\n", max_delta_lower_x, max_delta_lower_y, max_delta_lower_z);
    printf("Max box sizes upper: dx = %f, dy = %f, dz = %f\n", map_upper->delta_x, map_upper->delta_y, map_upper->delta_z);
    printf("Number of cells lower: nx = %d, ny = %d, nz = %d\n",
           map_lower->num_cells_x,
           map_lower->num_cells_y,
           map_lower->num_cells_z);
    printf("Number of cells upper: nx = %d, ny = %d, nz = %d\n",
           map_upper->num_cells_x,
           map_upper->num_cells_y,
           map_upper->num_cells_z);
#endif

    for (int i = 0; i < num_boxes; i++) {
        real_t dx = fabs(box_max_x[i] - box_min_x[i]);
        real_t dy = fabs(box_max_y[i] - box_min_y[i]);
        real_t dz = fabs(box_max_z[i] - box_min_z[i]);

        if (dx < split_x && dy < split_y) {
            increment_cell_counts_for_box(map_lower, box_min_x[i], box_min_y[i], box_max_x[i], box_max_y[i]);

        } else {
            increment_cell_counts_for_box(map_upper, box_min_x[i], box_min_y[i], box_max_x[i], box_max_y[i]);

        }  // END if-else (split)
    }  // END for (num_boxes)

    // Accumulate counts to make the lookup pointer array lower: ....
    for (int i = 1; i <= total_num_2d_cells_lower; i++) {
        map_lower->cell_ptr[i] += map_lower->cell_ptr[i - 1];
    }  // END for (accumulate lower)

    // And accumulate counts to make the lookup pointer array upper:
    for (int i = 1; i <= total_num_2d_cells_upper; i++) {
        map_upper->cell_ptr[i] += map_upper->cell_ptr[i - 1];
    }  // END for (accumulate upper)

    allocate_map_dict_entries(map_lower, total_num_2d_cells_lower);
    allocate_map_dict_entries(map_upper, total_num_2d_cells_upper);

    int *current_count_lower = (int *)calloc(total_num_2d_cells_lower, sizeof(int));
    int *current_count_upper = (int *)calloc(total_num_2d_cells_upper, sizeof(int));

    // Fill the cell dictionary array
    for (int i = 0; i < num_boxes; i++) {
        real_t dx = fabs(box_max_x[i] - box_min_x[i]);
        real_t dy = fabs(box_max_y[i] - box_min_y[i]);
        real_t dz = fabs(box_max_z[i] - box_min_z[i]);

        if (dx < split_x && dy < split_y) {
            fill_cell_dict_for_box(map_lower,
                                   current_count_lower,
                                   i,
                                   box_min_x[i],
                                   box_min_y[i],
                                   box_min_z[i],
                                   box_max_x[i],
                                   box_max_y[i],
                                   box_max_z[i]);
        } else {
            fill_cell_dict_for_box(map_upper,
                                   current_count_upper,
                                   i,
                                   box_min_x[i],
                                   box_min_y[i],
                                   box_min_z[i],
                                   box_max_x[i],
                                   box_max_y[i],
                                   box_max_z[i]);
        }  // END if-else (split)
    }  // END for (num_boxes)

    free(current_count_lower);
    free(current_count_upper);

    // Sort the entries in each cell by their lower Z bounds
    sort_cell_entries_by_z(map_lower, total_num_2d_cells_lower);
    sort_cell_entries_by_z(map_upper, total_num_2d_cells_upper);

    // Ensure that the upper bounds are non-decreasing within each cell
    ensure_upper_bounds_non_decreasing(map_lower, total_num_2d_cells_lower);
    ensure_upper_bounds_non_decreasing(map_upper, total_num_2d_cells_upper);

    return 0;
}  // END Function: build_cell_list_3d_2d_split_map

/////////////////////////////////////////////////
// build_cell_list_3d_2d_split_map
/////////////////////////////////////////////////
int build_cell_list_3d_2d_split_map(cell_list_split_3d_2d_map_t **split_map,  //
                                    const real_t                  split_x,    //
                                    const real_t                  split_y,    //
                                    const real_t                 *box_min_x,  //
                                    const real_t                 *box_min_y,  //
                                    const real_t                 *box_min_z,  //
                                    const real_t                 *box_max_x,  //
                                    const real_t                 *box_max_y,  //
                                    const real_t                 *box_max_z,  //
                                    const int                     num_boxes,  //
                                    const real_t                  x_min,      //
                                    const real_t                  x_max,      //
                                    const real_t                  y_min,      //
                                    const real_t                  y_max,      //
                                    const real_t                  z_min,      //
                                    const real_t                  z_max) {    //

    if (split_map == NULL) {
        return -1;  // Invalid pointer
    }

    if (*split_map != NULL) {
        return -1;  // Output pointer already points to an allocated structure
    }

    *split_map            = make_empty_cell_list_split_3d_2d_map();
    (*split_map)->split_x = split_x;
    (*split_map)->split_y = split_y;

    (*split_map)->map_lower = make_empty_cell_list_3d_2d_map();
    (*split_map)->map_upper = make_empty_cell_list_3d_2d_map();

    return                                                            //
            fill_cell_lists_3d_2d_split_map((*split_map)->map_lower,  //
                                            (*split_map)->map_upper,  //
                                            split_x,                  //
                                            split_y,                  //
                                            box_min_x,                //
                                            box_min_y,                //
                                            box_min_z,                //
                                            box_max_x,                //
                                            box_max_y,                //
                                            box_max_z,                //
                                            num_boxes,                //
                                            x_min,                    //
                                            x_max,                    //
                                            y_min,                    //
                                            y_max,                    //
                                            z_min,                    //
                                            z_max);                   //
}

////////////////////////////////////////////////
// query_cell_list_3d_2d_split_map
////////////////////////////////////////////////
int query_cell_list_3d_2d_split_map(const cell_list_split_3d_2d_map_t *split_map,    //
                                    const boxes_t                     *boxes,        //
                                    const real_t                       x,            //
                                    const real_t                       y,            //
                                    const real_t                       z,            //
                                    int                              **box_indices,  //
                                    int                               *num_boxes) {  //
    if (split_map == NULL || boxes == NULL || box_indices == NULL || num_boxes == NULL) {
        return -1;  // Invalid pointer
    }

    int *box_indices_lower = NULL;
    int *box_indices_upper = NULL;

    int num_boxes_lower = -1;
    int num_boxes_upper = -1;

    int result_lower = query_cell_list_3d_2d_map(split_map->map_lower,  //
                                                 boxes,
                                                 x,
                                                 y,
                                                 z,
                                                 &box_indices_lower,
                                                 &num_boxes_lower);

    int result_upper = query_cell_list_3d_2d_map(split_map->map_upper,  //
                                                 boxes,
                                                 x,
                                                 y,
                                                 z,
                                                 &box_indices_upper,
                                                 &num_boxes_upper);

    if (result_lower != 0 || result_upper != 0) {
        if (box_indices_lower != NULL) {
            free(box_indices_lower);
        }
        if (box_indices_upper != NULL) {
            free(box_indices_upper);
        }
        return -1;
    }

    *num_boxes   = num_boxes_lower + num_boxes_upper;
    *box_indices = malloc((*num_boxes) * sizeof(int));

    // Merge the results from the lower and upper maps
    if (box_indices_lower != NULL) {
        for (int i = 0; i < num_boxes_lower; i++) {
            (*box_indices)[i] = box_indices_lower[i];
        }
        free(box_indices_lower);
    }

    if (box_indices_upper != NULL) {
        for (int i = 0; i < num_boxes_upper; i++) {
            (*box_indices)[num_boxes_lower + i] = box_indices_upper[i];
        }
        free(box_indices_upper);
    }

    return 0;
}

/////////////////////////////////////////////////
// query_cell_list_3d_2d_split_map_given_xy
////////////////////////////////////////////////
int query_cell_list_3d_2d_split_map_given_xy(
        const cell_list_split_3d_2d_map_t *split_map,    //
        const boxes_t                     *boxes,        //
        const real_t                       x,            //
        const real_t                       y,            //
        const real_t                      *z_array,      //
        const int                          size_z,       //
        int                             ***box_indices,  // it produces a pointer of a vector (size_z) of vector(size_boxes_local)
        int                              **num_boxes) {  //
                                                         //
    if (split_map == NULL || boxes == NULL || box_indices == NULL || num_boxes == NULL) {
        return -1;  // Invalid pointer
    }

    int **box_indices_lower = NULL;
    int **box_indices_upper = NULL;

    int *num_boxes_lower = NULL;
    int *num_boxes_upper = NULL;

    int result_lower =                                                //
            query_cell_list_3d_2d_map_given_xy(split_map->map_lower,  //
                                               boxes,                 //
                                               x,                     //
                                               y,                     //
                                               z_array,               //
                                               size_z,                //
                                               &box_indices_lower,    //
                                               &num_boxes_lower);     //

    int result_upper =                                                //
            query_cell_list_3d_2d_map_given_xy(split_map->map_upper,  //
                                               boxes,                 //
                                               x,                     //
                                               y,                     //
                                               z_array,               //
                                               size_z,                //
                                               &box_indices_upper,    //
                                               &num_boxes_upper);     //

    *num_boxes   = calloc(size_z, sizeof(int));
    *box_indices = malloc(size_z * sizeof(int *));

    for (int iz = 0; iz < size_z; iz++) {
        int count_lower = (box_indices_lower != NULL) ? num_boxes_lower[iz] : 0;
        int count_upper = (box_indices_upper != NULL) ? num_boxes_upper[iz] : 0;

        (*num_boxes)[iz]   = count_lower + count_upper;
        (*box_indices)[iz] = malloc((*num_boxes)[iz] * sizeof(int));

        // Merge the results from the lower and upper maps for this z value
        int idx = 0;
        if (box_indices_lower != NULL) {
            for (int i = 0; i < count_lower; i++) {
                (*box_indices)[iz][idx++] = box_indices_lower[iz][i];
            }
            free(box_indices_lower[iz]);
        }

        if (box_indices_upper != NULL) {
            for (int i = 0; i < count_upper; i++) {
                (*box_indices)[iz][idx++] = box_indices_upper[iz][i];
            }
            free(box_indices_upper[iz]);
        }
    }

    if (box_indices_lower != NULL) {
        free(box_indices_lower);
    }

    if (box_indices_upper != NULL) {
        free(box_indices_upper);
    }

    if (num_boxes_lower != NULL) {
        free(num_boxes_lower);
    }

    if (num_boxes_upper != NULL) {
        free(num_boxes_upper);
    }

    return 0;
}