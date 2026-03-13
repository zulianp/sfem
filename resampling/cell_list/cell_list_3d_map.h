#ifndef __CELL_LIST_3D_MAP_H__
#define __CELL_LIST_3D_MAP_H__

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "cell_arg_sort.h"
#include "precision_types.h"

real_t                              //
random_interval(const real_t min,   //
                const real_t max);  //

//////////////////////////////////////////////////////////
// boxes_t
//////////////////////////////////////////////////////////

/**
 * @brief Structure to hold axis-aligned bounding boxes
 */
typedef struct {
    real_t *min_x;
    real_t *min_y;
    real_t *min_z;
    real_t *max_x;
    real_t *max_y;
    real_t *max_z;
    int     num_boxes;
} boxes_t;

/**
 * @brief Structure to hold axis-aligned bounding boxes in an
 * interleaved format (min_x, min_y, min_z, max_x, max_y, max_z)
 */
typedef struct {
    real_t *min_max_xyz;
    int     num_boxes;
} boxes_interleaved_t;

/**
 * @brief Create an empty boxes_interleaved_t structure
 * @return An empty boxes_interleaved_t structure
 */
boxes_interleaved_t              //
make_boxes_interleaved_t(void);  //

/**
 * @brief Initialize boxes_interleaved_t structure
 * @param boxes Pointer to boxes_interleaved_t structure to initialize
 */
void                                                   //
init_boxes_interleaved_t(boxes_interleaved_t *boxes);  //

/**
 * @brief Allocate memory for boxes_interleaved_t structure
 * @param num_boxes Number of boxes to allocate
 * @return Pointer to allocated boxes_interleaved_t structure
 */
boxes_interleaved_t *                               //
allocate_boxes_interleaved_t(const int num_boxes);  //

/**
 * @brief Free memory allocated for boxes_interleaved_t structure
 * @param boxes Pointer to boxes_interleaved_t structure to free
 */
void                                                   //
free_boxes_interleaved_t(boxes_interleaved_t *boxes);  //

/**
 * @brief Create an empty boxes_t structure
 * @return An empty boxes_t structure
 */
boxes_t              //
make_boxes_t(void);  //

/**
 * @brief Initialize boxes_t structure
 * @param boxes Pointer to boxes_t structure to initialize
 */
void                           //
init_boxes_t(boxes_t *boxes);  //

/**
 * @brief Allocate memory for boxes_t structure
 * @param num_boxes Number of boxes to allocate
 * @return Pointer to allocated boxes_t structure
 */
boxes_t *                               //
allocate_boxes_t(const int num_boxes);  //

/**
 * @brief Free memory allocated for boxes_t structure
 * @param boxes Pointer to boxes_t structure to free
 */
void                           //
free_boxes_t(boxes_t *boxes);  //

/**
 * @brief Copy boxes from boxes_t format to interleaved format
 * @param boxes Pointer to boxes_t structure to copy from
 * @param interleaved Pointer to boxes_interleaved_t structure to copy to
 * @note This function will free any existing data in interleaved->min_max_xyz before copying
 * @return void
 */
void
copy_boxes_to_interleaved(const boxes_t *boxes, boxes_interleaved_t *interleaved);

/**
 * @brief Get the bounds of a box in interleaved format
 * @param boxes Pointer to boxes_interleaved_t structure
 * @param box_index Index of the box to get bounds for
 * @param bounds Output array of size 6 to hold min_x, min_y, min_z, max_x, max_y, max_z
 */
void 
get_box_bounds_il(const boxes_interleaved_t *boxes, const int box_index, real_t bounds[6]);

/**
 * @brief Check if a box contains a given point
 * @param boxes Pointer to boxes_t structure
 * @param box_index Index of the box to check
 * @param x X coordinate of the point
 * @param y Y coordinate of the point
 * @param z Z coordinate of the point
 * @return true if the box contains the point, false otherwise
 */
bool                                             //
check_box_contains_pt(const boxes_t *boxes,      //
                      const int      box_index,  //
                      const real_t   x,          //
                      const real_t   y,          //
                      const real_t   z);           //

bool                                                  //
check_box_contains_pt_fast(const boxes_t *boxes,      //
                           const int      box_index,  //
                           const real_t   x,          //
                           const real_t   y,          //
                           const real_t   z);           //

/**
 * @brief Generate random boxes within specified bounds
 * @param boxes Pointer to boxes_t structure to fill with random boxes
 * @param x_min Minimum X coordinate
 * @param x_max Maximum X coordinate
 * @param y_min Minimum Y coordinate
 * @param y_max Maximum Y coordinate
 * @param z_min Minimum Z coordinate
 * @param z_max Maximum Z coordinate
 * @param box_size_min Minimum size of the boxes in each dimension
 * @param box_size_max Maximum size of the boxes in each dimension
 * @return int 0 on success, non-zero on failure
 */
int                                              //
make_random_boxes(boxes_t     *boxes,            //
                  const real_t x_min,            //
                  const real_t x_max,            //
                  const real_t y_min,            //
                  const real_t y_max,            //
                  const real_t z_min,            //
                  const real_t z_max,            //
                  const real_t box_size_min_x,   //
                  const real_t box_size_max_x,   //
                  const real_t box_size_min_y,   //
                  const real_t box_size_max_y,   //
                  const real_t box_size_min_z,   //
                  const real_t box_size_max_z);  //

/////////////////////////////////////////////////////////
// cell_list_3d_2d_map_t
/////////////////////////////////////////////////////////

/**
 * @brief Structure for 3D cell list with 2D mapping
 * This structure holds the data necessary to perform spatial queries
 * on a set of 3D boxes using a cell list approach with 2D mapping.
 */
typedef struct {
    int    *cell_ptr;
    int    *cell_dict;
    real_t *lower_bounds_z;
    real_t *upper_bounds_z;

    int total_num_2d_cells;
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
    int num_cells_y;
    int num_cells_z;  // TODO may not be needed

} cell_list_3d_2d_map_t;

typedef struct {
    real_t split_x;
    real_t split_y;

    cell_list_3d_2d_map_t *map_lower;
    cell_list_3d_2d_map_t *map_upper;

} cell_list_split_3d_2d_map_t;

/**
 * @brief Create an empty cell_list_3d_2d_map_t structure
 * @return Pointer to allocated cell_list_3d_2d_map_t structure
 */
cell_list_3d_2d_map_t *                //
make_empty_cell_list_3d_2d_map(void);  //

/**
 * @brief Free memory allocated for cell_list_3d_2d_map_t structure
 * @param map Pointer to cell_list_3d_2d_map_t structure to free
 */
int                                                    //
free_cell_list_3d_2d_map(cell_list_3d_2d_map_t *map);  //

/**
 * @brief Build the 3D cell list with 2D mapping
 * @param map Pointer to cell_list_3d_2d_map_t structure to build
 * @param box_min_x Array of minimum X coordinates of boxes
 * @param box_min_y Array of minimum Y coordinates of boxes
 * @param box_min_z Array of minimum Z coordinates of boxes
 * @param box_max_x Array of maximum X coordinates of boxes
 * @param box_max_y Array of maximum Y coordinates of boxes
 * @param box_max_z Array of maximum Z coordinates of boxes
 * @param num_boxes Number of boxes
 * @param x_min Minimum X coordinate of the domain
 * @param x_max Maximum X coordinate of the domain
 * @param y_min Minimum Y coordinate of the domain
 * @param y_max Maximum Y coordinate of the domain
 * @param z_min Minimum Z coordinate of the domain
 * @param z_max Maximum Z coordinate of the domain
 * @return int 0 on success, non-zero on failure
 */
int                                                          //
build_cell_list_3d_2d_map(cell_list_3d_2d_map_t *map,        //
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
                          const real_t           z_max);               //

/**
 * @brief Query the cell list to find boxes containing a given point
 * @param map Pointer to cell_list_3d_2d_map_t structure
 * @param boxes Pointer to boxes_t structure containing the boxes
 * @param x X coordinate of the query point
 * @param y Y coordinate of the query point
 * @param z Z coordinate of the query point
 * @param box_indices Pointer to array of box indices that contain the point
 * @param num_boxes Pointer to number of boxes found
 */
int                                                                  //
query_cell_list_3d_2d_map(const cell_list_3d_2d_map_t *map,          //
                          const boxes_t               *boxes,        //
                          const real_t                 x,            //
                          const real_t                 y,            //
                          const real_t                 z,            //
                          int                        **box_indices,  //
                          int                         *num_boxes);                           //

/**
 * @brief Query the cell list to find boxes containing points with given X, Y and array of Z coordinates
 * @param map Pointer to cell_list_3d_2d_map_t structure
 * @param boxes Pointer to boxes_t structure containing the boxes
 * @param x X coordinate of the query points
 * @param y Y coordinate of the query points
 * @param z_array Array of Z coordinates of the query points
 * @param size_z Size of the Z coordinates array
 * @param box_indices Pointer to array of arrays of box indices that contain the points
 * @param num_boxes Pointer to array of number of boxes found for each Z coordinate
 */
int                                                                       //
query_cell_list_3d_2d_map_given_xy(const cell_list_3d_2d_map_t *map,      //
                                   const boxes_t               *boxes,    //
                                   const real_t                 x,        //
                                   const real_t                 y,        //
                                   const real_t                *z_array,  //
                                   const int                    size_z,   //
                                   int ***box_indices,  // it produces a pointer of a vector (size_z) of vector(size_boxes_local)
                                   int  **num_boxes);    //

/**
 * @brief Linear search for boxes containing a given point
 * @param boxes Pointer to boxes_t structure containing the boxes
 * @param x X coordinate of the query point
 * @param y Y coordinate of the query point
 * @param z Z coordinate of the query point
 * @param box_indices Pointer to array of box indices that contain the point
 * @param num_boxes Pointer to number of boxes found
 */
int                                                    //
query_linear_search_boxes(const boxes_t *boxes,        //
                          const real_t   x,            //
                          const real_t   y,            //
                          const real_t   z,            //
                          int          **box_indices,  //
                          int           *num_boxes);             //

/**
 * @brief Create an empty cell_list_split_3d_2d_map_t structure
 * @return Pointer to allocated cell_list_split_3d_2d_map_t structure
 */
cell_list_split_3d_2d_map_t *                //
make_empty_cell_list_split_3d_2d_map(void);  //

/**
 * @brief Free memory allocated for cell_list_split_3d_2d_map_t structure
 * @param split_map Pointer to cell_list_split_3d_2d_map_t structure to free
 */
int                                                                      //
free_cell_list_split_3d_2d_map(cell_list_split_3d_2d_map_t *split_map);  //

/**
 * @brief Build the 3D cell list with 2D mapping using a x and y sides length split
 * @param split_map Pointer to cell_list_split_3d_2d_map_t structure to build
 * @param split_x X coordinate of the split line
 * @param split_y Y coordinate of the split line
 * @param box_min_x Array of minimum X coordinates of boxes
 * @param box_min_y Array of minimum Y coordinates of boxes
 * @param box_min_z Array of minimum Z coordinates of boxes
 * @param box_max_x Array of maximum X coordinates of boxes
 * @param box_max_y Array of maximum Y coordinates of boxes
 * @param box_max_z Array of maximum Z coordinates of boxes
 * @param num_boxes Number of boxes
 * @param x_min Minimum X coordinate of the domain
 * @param x_max Maximum X coordinate of the domain
 * @param y_min Minimum Y coordinate of the domain
 */
int                                                                //
fill_cell_lists_3d_2d_split_map(cell_list_3d_2d_map_t *map_lower,  //
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
                                const real_t           z_max);               //

/**
 * @brief Build the 3D cell list with 2D mapping using a x and y sides length split
 * @param split_map Pointer to cell_list_split_3d_2d_map_t structure to build
 * @param split_x X coordinate of the split line
 * @param split_y Y coordinate of the split line
 * @param box_min_x Array of minimum X coordinates of boxes
 * @param box_min_y Array of minimum Y coordinates of boxes
 * @param box_min_z Array of minimum Z coordinates of boxes
 * @param box_max_x Array of maximum X coordinates of boxes
 * @param box_max_y Array of maximum Y coordinates of boxes
 * @param box_max_z Array of maximum Z coordinates of boxes
 * @param num_boxes Number of boxes
 * @param x_min Minimum X coordinate of the domain
 * @param x_max Maximum X coordinate of the domain
 * @param y_min Minimum Y coordinate of the domain
 * @param y_max Maximum Y coordinate of the domain
 * @param z_min Minimum Z coordinate of the domain
 * @param z_max Maximum Z coordinate of the domain
 */
int                                                                       //
build_cell_list_3d_2d_split_map(cell_list_split_3d_2d_map_t **split_map,  //
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
                                const real_t                  z_max);                      //

/**
 * @brief Query the split cell list to find boxes containing a given point
 * @param split_map Pointer to cell_list_split_3d_2d_map_t structure
 * @param boxes Pointer to boxes_t structure containing the boxes
 * @param x X coordinate of the query point
 * @param y Y coordinate of the query point
 * @param z Z coordinate of the query point
 * @param box_indices Pointer to array of box indices that contain the point
 * @param num_boxes Pointer to number of boxes found
 */
int                                                                              //
query_cell_list_3d_2d_split_map(const cell_list_split_3d_2d_map_t *split_map,    //
                                const boxes_t                     *boxes,        //
                                const real_t                       x,            //
                                const real_t                       y,            //
                                const real_t                       z,            //
                                int                              **box_indices,  //
                                int                               *num_boxes);                                 //

/**
 * @brief Query the split cell list to find boxes containing points with given X, Y and array of Z coordinates
 * @param split_map Pointer to cell_list_split_3d_2d_map_t structure
 * @param boxes Pointer to boxes_t structure containing the boxes
 * @param x X coordinate of the query points
 * @param y Y coordinate of the query points
 * @param z_array Array of Z coordinates of the query points
 * @param size_z Size of the Z coordinates array
 * @param box_indices Pointer to array of arrays of box indices that contain the points
 * @param num_boxes Pointer to array of number of boxes found for each Z coordinate
 */
int  //
query_cell_list_3d_2d_split_map_given_xy(
        const cell_list_split_3d_2d_map_t *split_map,    //
        const boxes_t                     *boxes,        //
        const real_t                       x,            //
        const real_t                       y,            //
        const real_t                      *z_array,      //
        const int                          size_z,       //
        int                             ***box_indices,  // it produces a pointer of a vector (size_z) of vector(size_boxes_local)
        int                              **num_boxes);                                //

/**
 * @brief Calculate the memory usage of the cell list
 * @param map Pointer to cell_list_3d_2d_map_t structure
 * @return int64_t Memory usage in bytes
 */
int64_t                                                       //
cell_list_3d_2d_map_bytes(const cell_list_3d_2d_map_t *map);  //

#endif  // __CELL_LIST_3D_MAP_H__
////////////////////////////////////////////////
