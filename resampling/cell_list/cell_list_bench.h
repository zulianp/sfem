#ifndef __CELL_LIST_BENCH_H__
#define __CELL_LIST_BENCH_H__

#include "cell_list_3d_1d_map.h"
#include "cell_list_3d_map.h"
#include "sfem_mesh.h"

typedef struct {
    int num_boxes;

    real_t box_size_min;
    real_t box_size_max;

    real_t x_min;
    real_t x_max;
    real_t y_min;
    real_t y_max;
    real_t z_min;
    real_t z_max;

    int num_xy_queries;
    int num_z_queries;

} box_config_t;

void box_config_set_defaults(box_config_t *config);

void print_box_config(const box_config_t *config);

/**
 * @brief Compare function for sorting integers
 */
int my_cmp_int(const void *a, const void *b);

/**
 * @brief Benchmark the cell list query performance
 *
 * @param map The cell list data structure
 * @param boxes The boxes data structure
 * @param num_queries Number of random queries to run
 * @return int 0 on success
 */
int query_cell_list_bench(const cell_list_3d_2d_map_t *map, const boxes_t *boxes, const int num_queries);

/**
 * @brief Test the cell list query correctness against linear search
 *
 * @param map The cell list data structure
 * @param boxes The boxes data structure
 * @param num_queries Number of random queries to run
 * @return int Total number of boxes found
 */
int query_cell_list_test(const cell_list_3d_2d_map_t *map, const boxes_t *boxes, const int num_queries);

/**
 * @brief Multithreaded benchmark of the cell list query performance
 * @param map The cell list data structure
 * @param boxes The boxes data structure
 * @param num_queries Number of random queries to run
 * @return int 0 on success
 */
int query_cell_list_bench_mt(const cell_list_3d_2d_map_t *map, const boxes_t *boxes, const int num_queries);

/**
 * @brief Test the cell list query correctness for given x,y and multiple z values
 * @param map The cell list data structure
 * @param boxes The boxes data structure
 * @param num_z Number of z values to test per (x,y) pair
 * @param num_queries Total number of (x,y) queries to run
 * @return int 0 on success, EXIT_FAILURE on failure
 */
int query_cell_list_given_xy_test(const cell_list_3d_2d_map_t *map, const boxes_t *boxes, const int num_z, const int num_queries);

/**
 * @brief Test the cell list query correctness for given x,y and multiple z values
 * @param map The cell list data structure
 * @param boxes The boxes data structure
 * @param num_z Number of z values to test per (x,y) pair
 * @param num_queries Total number of (x,y) queries to run
 * @return int 0 on success, EXIT_FAILURE on failure
 */
int query_cell_list_given_xy_bench(const cell_list_3d_2d_map_t *map, const boxes_t *boxes, const int num_z,
                                   const int num_queries);

/**
 * @brief Test a single query and print results
 * @param cell_list_map The cell list data structure
 * @param boxes The boxes data structure
 * @param x_min Minimum X coordinate of the domain
 * @param x_max Maximum X coordinate of the domain
 * @param y_min Minimum Y coordinate of the domain
 * @param y_max Maximum Y coordinate of the domain
 * @param z_min Minimum Z coordinate of the domain
 * @param z_max Maximum Z coordinate of the domain
 * @param cpu_build_time_used Time taken to build the cell list
 */
void test_single_query(cell_list_3d_2d_map_t *cell_list_map,  //
                       boxes_t *boxes, real_t x_min, real_t x_max, real_t y_min, real_t y_max, real_t z_min, real_t z_max,
                       double cpu_build_time_used);

/**
 * @brief Main function for 1D-3D cell list benchmark
 * @param argc Argument count
 * @param argv Argument vector
 * @return int Exit status
 */
int main_1D_3D(int argc, char **argv, box_config_t config);

/**
 * @brief Test the 1D-3D cell list query correctness for given x,y and multiple z values
 * @param map The cell list data structure
 * @param boxes The boxes data structure
 * @param num_z Number of z values to test per (x,y) pair
 * @param num_queries Total number of (x,y) queries to run
 * @return int 0 on success, EXIT_FAILURE on failure
 */
int query_cell_list_1d_3d_given_xy_test(const cell_list_3d_1d_map_t *map,    //
                                        const boxes_t               *boxes,  //
                                        const int                    num_z,  //
                                        const int                    num_queries);              //

/**
 * @brief Benchmark the 1D-3D cell list query performance for given x,y and multiple z values
 * @param map The cell list data structure
 * @param boxes The boxes data structure
 * @param num_z Number of z values to test per (x,y) pair
 * @param num_queries Total number of (x,y) queries to run
 * @return int 0 on success, EXIT_FAILURE on failure
 */
int query_cell_list_1d_3d_given_xy_bench(const cell_list_3d_1d_map_t *map, const boxes_t *boxes, const int num_z,
                                         const int num_queries);

/**
 * @brief Test the 1D-3D cell list query correctness for given x,y and multiple z values
 * @param map The cell list data structure
 * @param boxes The boxes data structure
 * @param num_z Number of z values to test per (x,y) pair
 * @param num_queries Total number of (x,y) queries to run
 * @return int 0 on success, EXIT_FAILURE on failure
 */
int query_tet_cell_list_1d_3d_given_xy_test(const cell_list_3d_1d_map_t *map,        //
                                            const boxes_t               *boxes,      //
                                            const mesh_tet_geom_t       *mesh_geom,  //
                                            const int                    num_z,      //
                                            const int                    num_queries);                  //

/**
 * @brief Test the 2D-3D tet cell list query correctness for given x,y and multiple z values
 * @param map The cell list data structure
 * @param boxes The boxes data structure
 * @param mesh_geom The tetrahedral mesh geometry
 * @param num_z Number of z values to test per (x,y) pair
 * @param num_queries Total number of (x,y) queries to run
 * @return int Total number of boxes found
 */
int                                                                              //
query_tet_cell_list_2d_3d_given_xy_test(const cell_list_3d_2d_map_t *map,        //
                                        const boxes_t               *boxes,      //
                                        const mesh_tet_geom_t       *mesh_geom,  //
                                        const int                    num_z,      //
                                        const int                    num_queries);                  //

/**
 * @brief Benchmark the 2D-3D tet cell list query performance for given x,y and multiple z values
 * @param map The cell list data structure
 * @param boxes The boxes data structure
 * @param mesh_geom The tetrahedral mesh geometry
 * @param num_z Number of z values to test per (x,y) pair
 * @param num_queries Total number of (x,y) queries to run
 * @return int Total number of boxes found
 */
int                                                                               //
query_tet_cell_list_2d_3d_given_xy_bench(const cell_list_3d_2d_map_t *map,        //
                                         const boxes_t               *boxes,      //
                                         const mesh_tet_geom_t       *mesh_geom,  //
                                         const int                    num_z,      //
                                         const int                    num_queries);                  //

/**
 * @brief Test the cell list query correctness for given x,y and multiple z values using split map
 * @param map The split cell list data structure
 * @param boxes The boxes data structure
 * @param num_z Number of z values to test per (x,y) pair
 * @param num_queries Total number of (x,y) queries to run
 * @return int 0 on success, EXIT_FAILURE on failure
 */
int query_cell_list_given_xy_split_test(const cell_list_split_3d_2d_map_t *map,    //
                                        const boxes_t                     *boxes,  //
                                        const int                          num_z,  //
                                        const int                          num_queries);

#endif  // __CELL_LIST_BENCH_H__
/////////////////////////////////////////////////
