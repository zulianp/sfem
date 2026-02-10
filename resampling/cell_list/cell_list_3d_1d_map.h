#ifndef __CELL_LIST_3D_1D_MAP_H__
#define __CELL_LIST_3D_1D_MAP_H__

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdbool.h>

#include "cell_list_3d_map.h"
#include "cell_list_3d_1d_map.h"

/**
 * @brief Structure for 3D to 1D cell list mapping
 */
typedef struct
{
    int *cell_ptr;
    int *cell_dict;

    real_t *lower_bounds_y;
    real_t *upper_bounds_y;

    int total_num_dict_entries;

    real_t delta_x;
    real_t delta_y;
    real_t delta_z;

    real_t min_x;
    real_t min_y;
    real_t min_z;
    real_t max_x;
    real_t max_y;
    real_t max_z;

    int num_cells_x;
    // int num_cells_y;
    // int num_cells_z;

} cell_list_3d_1d_map_t;

cell_list_3d_1d_map_t *
make_empty_cell_list_3d_1d_map(void);

void free_cell_list_3d_1d_map(cell_list_3d_1d_map_t *map);

int build_cell_list_3d_1d_map(cell_list_3d_1d_map_t *map,
                              const real_t *box_min_x,
                              const real_t *box_min_y,
                              const real_t *box_min_z,
                              const real_t *box_max_x,
                              const real_t *box_max_y,
                              const real_t *box_max_z,
                              const int num_boxes,
                              const real_t x_min,
                              const real_t x_max,
                              const real_t y_min,
                              const real_t y_max,
                              const real_t z_min,
                              const real_t z_max);

int query_cell_list_3d_1d_map(const cell_list_3d_1d_map_t *map,
                              const boxes_t *boxes,
                              const real_t x,
                              const real_t y,
                              const real_t z,
                              int **box_indices,
                              int *num_boxes);

/**
 * @brief Query the cell list for multiple z values at given (x, y)
 * @param map Pointer to cell_list_3d_1d_map_t structure
 * @param boxes Pointer to boxes_t structure
 * @param x X coordinate of the query point
 * @param y Y coordinate of the query point
 * @param z_array Array of Z coordinates of the query points
 * @param size_z Size of the Z coordinates array
 * @param box_indices Pointer to array of arrays of box indices that contain the points
 * @param num_boxes Pointer to array of number of boxes found for each Z coordinate
 */
int query_cell_list_3d_1d_map_given_xy(const cell_list_3d_1d_map_t *map,
                                       const boxes_t *boxes,
                                       const real_t x,
                                       const real_t y,
                                       const real_t *z_array,
                                       const int size_z,
                                       int ***box_indices, // it produces a pointer of a vector (size_z) of vector(size_boxes_local)
                                       int **num_boxes);

/**
 * @brief Calculate the memory usage of the cell list
 * @param map Pointer to cell_list_3d_1d_map_t structure
 * @return int64_t Memory usage in bytes
 */
int64_t cell_list_3d_1d_map_bytes(const cell_list_3d_1d_map_t *map);

#endif // __CELL_LIST_3D_1D_MAP_H__