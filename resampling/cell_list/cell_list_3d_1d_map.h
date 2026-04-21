#ifndef __CELL_LIST_3D_1D_MAP_H__
#define __CELL_LIST_3D_1D_MAP_H__

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cell_list_3d_1d_map.h"
#include "cell_list_3d_map.h"

/**
 * @brief Acceleration structure for point-in-box queries on AABBs.
 *
 * The structure partitions the domain only along the x axis (1D grid), then
 * stores per-cell candidate boxes sorted by y-interval to reduce candidate
 * checks before running the full 3D containment test.
 *
 * Memory ownership: all pointer members are allocated/freed by
 * build_cell_list_3d_1d_map() and free_cell_list_3d_1d_map().
 */
typedef struct {
    int *cell_ptr;  /**< CSR-like offsets, size num_cells_x + 1. */
    int *cell_dict; /**< Box indices for each x-cell interval. */

    real_t *lower_bounds_y; /**< Sorted lower y bounds aligned with cell_dict. */
    real_t *upper_bounds_y; /**< Monotone prefix-max upper y bounds. */

    int total_num_dict_entries; /**< Total entries in cell_dict/lower/upper arrays. */

    real_t delta_x; /**< Cell size along x (max box extent on x). */
    real_t delta_y; /**< Max box extent on y (stored for diagnostics). */
    real_t delta_z; /**< Max box extent on z (stored for diagnostics). */

    real_t min_x; /**< Domain minimum x used to build the map. */
    real_t min_y; /**< Domain minimum y used to build the map. */
    real_t min_z; /**< Domain minimum z used to build the map. */
    real_t max_x; /**< Domain maximum x used to build the map. */
    real_t max_y; /**< Domain maximum y used to build the map. */
    real_t max_z; /**< Domain maximum z used to build the map. */

    int num_cells_x; /**< Number of cells in x direction. */
    // int num_cells_y;
    // int num_cells_z;

} cell_list_3d_1d_map_t;

/**
 * @brief Allocate and initialize an empty map structure.
 * @return Pointer to a newly allocated map with all dynamic arrays set to NULL.
 */
cell_list_3d_1d_map_t *make_empty_cell_list_3d_1d_map(void);

/**
 * @brief Free all memory owned by a map.
 * @param map Map returned by make_empty_cell_list_3d_1d_map(). Accepts NULL.
 */
void free_cell_list_3d_1d_map(cell_list_3d_1d_map_t *map);

/**
 * @brief Build the x-binned map from axis-aligned boxes and domain bounds.
 *
 * The builder computes cell size from maximum box extents, assigns each box to
 * all overlapped x-cells, and prepares y-range helper arrays for faster query.
 *
 * @param map Pre-allocated map object to fill.
 * @param box_min_x Array of per-box minimum x coordinates.
 * @param box_min_y Array of per-box minimum y coordinates.
 * @param box_min_z Array of per-box minimum z coordinates.
 * @param box_max_x Array of per-box maximum x coordinates.
 * @param box_max_y Array of per-box maximum y coordinates.
 * @param box_max_z Array of per-box maximum z coordinates.
 * @param num_boxes Number of boxes in all coordinate arrays.
 * @param x_min Domain lower bound in x.
 * @param x_max Domain upper bound in x.
 * @param y_min Domain lower bound in y.
 * @param y_max Domain upper bound in y.
 * @param z_min Domain lower bound in z.
 * @param z_max Domain upper bound in z.
 * @return EXIT_SUCCESS on success.
 */
int build_cell_list_3d_1d_map(cell_list_3d_1d_map_t *map, const real_t *box_min_x, const real_t *box_min_y,
                              const real_t *box_min_z, const real_t *box_max_x, const real_t *box_max_y, const real_t *box_max_z,
                              const int num_boxes, const real_t x_min, const real_t x_max, const real_t y_min, const real_t y_max,
                              const real_t z_min, const real_t z_max);

/**
 * @brief Query candidate boxes that contain one point (x, y, z).
 *
 * On success, this function allocates *box_indices when matches are found.
 * Caller owns *box_indices and must free it. When no boxes are found,
 * *box_indices is set to NULL and *num_boxes to 0.
 *
 * @param map Pre-built map.
 * @param boxes Source box arrays used for final 3D containment checks.
 * @param x Query x coordinate.
 * @param y Query y coordinate.
 * @param z Query z coordinate.
 * @param box_indices Output pointer to an allocated array of box indices.
 * @param num_boxes Output number of valid entries in *box_indices.
 * @return EXIT_SUCCESS on success.
 */
int query_cell_list_3d_1d_map(const cell_list_3d_1d_map_t *map, const boxes_t *boxes, const real_t x, const real_t y,
                              const real_t z, int **box_indices, int *num_boxes);

/**
 * @brief Query candidate boxes for multiple z values at fixed (x, y).
 *
 * The function reuses x/y filtering once, then tests each z in z_array against
 * the filtered candidates.
 *
 * On success, this function allocates:
 * - *box_indices: array of size_z pointers.
 * - (*box_indices)[iz]: per-z index arrays (possibly zero-length after realloc).
 * - *num_boxes: array of size_z counts.
 *
 * Caller owns all allocated memory and must free each (*box_indices)[iz], then
 * *box_indices and *num_boxes.
 *
 * @param map Pre-built map.
 * @param boxes Source box arrays used for final 3D containment checks.
 * @param x Query x coordinate.
 * @param y Query y coordinate.
 * @param z_array Array of query z coordinates.
 * @param size_z Number of z values in z_array.
 * @param box_indices Output 2D jagged array of matching box indices.
 * @param num_boxes Output per-z match counts (size size_z).
 * @return EXIT_SUCCESS on success.
 */
int query_cell_list_3d_1d_map_given_xy(
        const cell_list_3d_1d_map_t *map, const boxes_t *boxes, const real_t x, const real_t y, const real_t *z_array,
        const int size_z,
        int    ***box_indices,  // it produces a pointer of a vector (size_z) of vector(size_boxes_local)
        int     **num_boxes);

/**
 * @brief Compute bytes currently owned by dynamic arrays in the map.
 *
 * This includes cell_ptr, cell_dict, lower_bounds_y and upper_bounds_y if
 * allocated. It excludes the size of the map struct itself.
 *
 * @param map Map to inspect.
 * @return Total dynamic memory footprint in bytes.
 */
int64_t cell_list_3d_1d_map_bytes(const cell_list_3d_1d_map_t *map);

#endif  // __CELL_LIST_3D_1D_MAP_H__