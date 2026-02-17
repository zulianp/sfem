#define _POSIX_C_SOURCE 199309L

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cell_list_3d_map.h"
#include "cell_list_3d_map_mesh.h"
#include "cell_list_bench.h"

////////////////////////////////////////////////
// query_cell_list_given_xy_split_test
////////////////////////////////////////////////
int query_cell_list_given_xy_split_test(const cell_list_split_3d_2d_map_t *map,    //
                                        const boxes_t                     *boxes,  //
                                        const int                          num_z,  //
                                        const int                          num_queries)                     //
{
    printf("query_cell_list_given_xy_split_test: Running %d random queries to test cell list...\n", num_queries);

    bool all_tests_passed_global = true;

    const int queries_per_xy = num_queries / num_z;

    if (queries_per_xy <= 0) {
        printf("Number of queries per xy is zero or negative, for num_queries=%d and num_z=%d.\n", num_queries, num_z);
        return EXIT_FAILURE;
    }

    real_t *z_array = (real_t *)malloc(num_z * sizeof(real_t));

    printf("Each of the %d different (x,y) points will be queried for %d different z values.\n", queries_per_xy, num_z);

    int total_boxes_found = 0;

    for (int q_xy_i = 0; q_xy_i < queries_per_xy; q_xy_i++) {
        for (int iz = 0; iz < num_z; iz++)
            z_array[iz] =
                    map->map_lower->min_z + (map->map_upper->max_z - map->map_lower->min_z) * ((real_t)rand() / (real_t)RAND_MAX);

        const real_t x =
                map->map_lower->min_x + (map->map_upper->max_x - map->map_lower->min_x) * ((real_t)rand() / (real_t)RAND_MAX);
        const real_t y =
                map->map_lower->min_y + (map->map_upper->max_y - map->map_lower->min_y) * ((real_t)rand() / (real_t)RAND_MAX);

        //  for (int q = 0; q < num_queries; q++)
        {  // Begin block of queries

            bool all_tests_passed = true;
            if (q_xy_i % 5 == 0) {
                printf("Query %d/%d...\n", q_xy_i + 1, queries_per_xy);
            }

            int **box_indices = NULL;
            int  *num_boxes   = NULL;

            query_cell_list_3d_2d_split_map_given_xy(map, boxes, x, y, z_array, num_z, &box_indices, &num_boxes);

            if (num_boxes != NULL) {
                for (int iz = 0; iz < num_z; iz++) {
                    total_boxes_found += num_boxes[iz];
                }
            }

            for (int iz = 0; iz < num_z; iz++) {
                int *box_indices_linear = NULL;
                int  num_boxes_linear   = 0;

                query_linear_search_boxes(boxes, x, y, z_array[iz], &box_indices_linear, &num_boxes_linear);

                if (box_indices != NULL) {
                    qsort(box_indices[iz], num_boxes[iz], sizeof(int), (int (*)(const void *, const void *))my_cmp_int);
                }
                qsort(box_indices_linear, num_boxes_linear, sizeof(int), (int (*)(const void *, const void *))my_cmp_int);

                if (box_indices == NULL) {
                    if (num_boxes_linear != 0) {
                        printf("== ERROR: Mismatch in number of boxes found: %d (cell list) != %d (linear search)\n",
                               0,
                               num_boxes_linear);
                        printf("** Query point: (%f, %f, %f)\n", x, y, z_array[iz]);
                        all_tests_passed = false;
                    }
                } else if (num_boxes[iz] != num_boxes_linear) {
                    printf("== ERROR: Mismatch in number of boxes found: %d (cell list) != %d (linear search)\n",
                           num_boxes[iz],
                           num_boxes_linear);
                    printf("** Query point: (%f, %f, %f)\n", x, y, z_array[iz]);

                    int max_boxes = (num_boxes[iz] > num_boxes_linear) ? num_boxes[iz] : num_boxes_linear;
                    for (int i = 0; i < max_boxes; i++) {
                        int idx_cell_list = (i < num_boxes[iz]) ? box_indices[iz][i] : -1;
                        int idx_linear    = (i < num_boxes_linear) ? box_indices_linear[i] : -1;
                        if (idx_cell_list != idx_linear) {
                            printf("== At position %d: cell list index = %d, linear search index = %d,\n** at point (%f, %f, "
                                   "%f)\n",
                                   i,
                                   idx_cell_list,
                                   idx_linear,
                                   x,
                                   y,
                                   z_array[iz]);
                            printf("** For box index %d:\n** box min (%f, %f, %f),\n** box max (%f, %f, %f)\n",
                                   (idx_cell_list != -1) ? idx_cell_list : idx_linear,
                                   boxes->min_x[(idx_cell_list != -1) ? idx_cell_list : idx_linear],
                                   boxes->min_y[(idx_cell_list != -1) ? idx_cell_list : idx_linear],
                                   boxes->min_z[(idx_cell_list != -1) ? idx_cell_list : idx_linear],
                                   boxes->max_x[(idx_cell_list != -1) ? idx_cell_list : idx_linear],
                                   boxes->max_y[(idx_cell_list != -1) ? idx_cell_list : idx_linear],
                                   boxes->max_z[(idx_cell_list != -1) ? idx_cell_list : idx_linear]);
                        }
                    }

                    all_tests_passed = false;
                }

                if (box_indices != NULL && num_boxes[iz] == num_boxes_linear) {
                    for (int i = 0; i < num_boxes[iz]; i++) {
                        if (box_indices[iz][i] != box_indices_linear[i]) {
                            printf("Mismatch in box indices at position %d: %d (cell list) != %d (linear search)\n",
                                   i,
                                   box_indices[iz][i],
                                   box_indices_linear[i]);
                            printf("** For box index %d: box min (%f, %f, %f), box max (%f, %f, %f)\n",
                                   box_indices[iz][i],
                                   boxes->min_x[box_indices[iz][i]],
                                   boxes->min_y[box_indices[iz][i]],
                                   boxes->min_z[box_indices[iz][i]],
                                   boxes->max_x[box_indices[iz][i]],
                                   boxes->max_y[box_indices[iz][i]],
                                   boxes->max_z[box_indices[iz][i]]);
                        }
                    }
                }

                if (box_indices_linear != NULL) {
                    free(box_indices_linear);
                }
            }  // end for iz

            if (box_indices != NULL) {
                for (int iz = 0; iz < num_z; iz++) {
                    if (box_indices[iz] != NULL) {
                        free(box_indices[iz]);
                    }
                }
                free(box_indices);
            }

            if (num_boxes != NULL) {
                free(num_boxes);
            }

            all_tests_passed_global = all_tests_passed_global && all_tests_passed;
        }  // end block of queries
    }  // end for q_xy_i

    free(z_array);

    if (all_tests_passed_global) {
        printf("All %d queries passed successfully!\n", num_queries);
    } else {
        printf("== ERROR: Some queries failed. Check the logs above for details.\n");
    }
    return total_boxes_found;
}

///////////////////////////////////////////////////
// query_cell_list_2d_3d_given_xy_test
///////////////////////////////////////////////////
int query_tet_split_cell_list_2d_3d_given_xy_test(const cell_list_split_3d_2d_map_t *map,        //
                                                  const boxes_t                     *boxes,      //
                                                  const mesh_tet_geom_t             *mesh_geom,  //
                                                  const int                          num_z,      //
                                                  const int                          num_queries) {                       //

    printf("query_tet_split_cell_list_2d_3d_given_xy_test: Running %d random queries to test cell list...\n", num_queries);

    bool all_tests_passed_global = true;

    const int queries_per_xy = num_queries / num_z;

    if (queries_per_xy <= 0) {
        printf("Number of queries per xy is zero or negative, for num_queries=%d and num_z=%d.\n", num_queries, num_z);
        return EXIT_FAILURE;
    }

    real_t *z_array = (real_t *)malloc(num_z * sizeof(real_t));

    printf("Each of the %d different (x,y) points will be queried for %d different z values.\n", queries_per_xy, num_z);

    int total_boxes_found = 0;

    for (int q_xy_i = 0; q_xy_i < queries_per_xy; q_xy_i++) {
        for (int iz = 0; iz < num_z; iz++)
            z_array[iz] =
                    map->map_lower->min_z + (map->map_upper->max_z - map->map_lower->min_z) * ((real_t)rand() / (real_t)RAND_MAX);

        const real_t x =
                map->map_lower->min_x + (map->map_upper->max_x - map->map_lower->min_x) * ((real_t)rand() / (real_t)RAND_MAX);
        const real_t y =
                map->map_lower->min_y + (map->map_upper->max_y - map->map_lower->min_y) * ((real_t)rand() / (real_t)RAND_MAX);

        //  for (int q = 0; q < num_queries; q++)
        {  // Begin block of queries

            bool all_tests_passed = true;
            if (q_xy_i % 5 == 0) {
                printf("tet_split_cell: Query %d/%d...\n", q_xy_i + 1, queries_per_xy);
            }

            int **box_indices = (int **)malloc(num_z * sizeof(int *));
            int  *num_boxes   = (int *)calloc((size_t)num_z, sizeof(int));

            for (int iz = 0; iz < num_z; iz++) {
                box_indices[iz] = NULL;

                const int tet_index = query_cell_list_3d_2d_split_map_mesh_given_xy(map, boxes, mesh_geom, x, y, &z_array[iz], 1);

                if (tet_index >= 0) {
                    num_boxes[iz]      = 1;
                    box_indices[iz]    = (int *)malloc(sizeof(int));
                    box_indices[iz][0] = tet_index;
                } else {
                    num_boxes[iz]   = 0;
                    box_indices[iz] = NULL;
                }  // END if (tet_index >= 0)
            }

            if (num_boxes != NULL) {
                for (int iz = 0; iz < num_z; iz++) {
                    total_boxes_found += num_boxes[iz];
                }
            }

            for (int iz = 0; iz < num_z; iz++) {
                int *box_indices_linear = NULL;
                int  num_boxes_linear   = 0;

                query_linear_search_boxes(boxes, x, y, z_array[iz], &box_indices_linear, &num_boxes_linear);

                if (box_indices != NULL) {
                    qsort(box_indices[iz], num_boxes[iz], sizeof(int), (int (*)(const void *, const void *))my_cmp_int);
                }
                qsort(box_indices_linear, num_boxes_linear, sizeof(int), (int (*)(const void *, const void *))my_cmp_int);

                if (box_indices == NULL) {
                    if (num_boxes_linear != 0) {
                        printf("== ERROR: Mismatch in number of boxes found: %d (cell list) != %d (linear search)\n",
                               0,
                               num_boxes_linear);
                        printf("** Query point: (%f, %f, %f)\n", x, y, z_array[iz]);
                        all_tests_passed = false;
                    }
                } else if (num_boxes[iz] >= 1 && num_boxes_linear <= 0) {
                    printf("== ERROR: Mismatch in number of boxes found: %d (cell list) != %d (linear search)\n",
                           num_boxes[iz],
                           num_boxes_linear);
                    printf("** Query point: (%f, %f, %f)\n", x, y, z_array[iz]);

                    int max_boxes = (num_boxes[iz] > num_boxes_linear) ? num_boxes[iz] : num_boxes_linear;
                    for (int i = 0; i < max_boxes; i++) {
                        int idx_cell_list = (i < num_boxes[iz]) ? box_indices[iz][i] : -1;
                        int idx_linear    = (i < num_boxes_linear) ? box_indices_linear[i] : -1;
                        if (idx_cell_list != idx_linear) {
                            printf("== At position %d: cell list index = %d, linear search index = %d,\n** at point (%f, %f, "
                                   "%f)\n",
                                   i,
                                   idx_cell_list,
                                   idx_linear,
                                   x,
                                   y,
                                   z_array[iz]);
                            printf("** For box index %d:\n** box min (%f, %f, %f),\n** box max (%f, %f, %f)\n",
                                   (idx_cell_list != -1) ? idx_cell_list : idx_linear,
                                   boxes->min_x[(idx_cell_list != -1) ? idx_cell_list : idx_linear],
                                   boxes->min_y[(idx_cell_list != -1) ? idx_cell_list : idx_linear],
                                   boxes->min_z[(idx_cell_list != -1) ? idx_cell_list : idx_linear],
                                   boxes->max_x[(idx_cell_list != -1) ? idx_cell_list : idx_linear],
                                   boxes->max_y[(idx_cell_list != -1) ? idx_cell_list : idx_linear],
                                   boxes->max_z[(idx_cell_list != -1) ? idx_cell_list : idx_linear]);
                        }
                    }

                    all_tests_passed = false;
                }

                if (box_indices != NULL && num_boxes[iz] >= 1 && num_boxes_linear >= 1) {
                    int matches = 0;

                    real_t point_z;

                    for (int i = 0; i < num_boxes_linear; i++) {
                        if (box_indices[iz][0] == box_indices_linear[i]) {
                            point_z = z_array[iz];
                            matches++;
                            break;
                        }
                    }

                    if (matches != 1) {
                        printf("== ERROR: No matching box index found between cell list and linear search.\n");
                        printf("** Query point: (%f, %f, %f)\n", x, y, point_z);
                        printf("** Cell list box index: %d\n", box_indices[iz][0]);
                        printf("** Linear search box indices:");
                        for (int i = 0; i < num_boxes_linear; i++) {
                            printf(" %d", box_indices_linear[i]);
                        }
                        printf("\n");
                        all_tests_passed = false;
                    }

                    if (is_point_out_of_tet(&mesh_geom->inv_Jacobian[box_indices[iz][0] * 9],  //
                                            mesh_geom->vetices_zero[box_indices[iz][0] * 3 + 0],
                                            mesh_geom->vetices_zero[box_indices[iz][0] * 3 + 1],
                                            mesh_geom->vetices_zero[box_indices[iz][0] * 3 + 2],
                                            x,
                                            y,
                                            point_z)) {  //

                        printf("== ERROR: Point is outside the tetrahedron found by cell list.\n");
                        printf("** Query point: (%f, %f, %f)\n", x, y, point_z);
                        printf("** Box index: %d\n", box_indices[iz][0]);
                        printf("** Box min: (%f, %f, %f)\n",
                               boxes->min_x[box_indices[iz][0]],
                               boxes->min_y[box_indices[iz][0]],
                               boxes->min_z[box_indices[iz][0]]);
                        printf("** Box max: (%f, %f, %f)\n",
                               boxes->max_x[box_indices[iz][0]],
                               boxes->max_y[box_indices[iz][0]],
                               boxes->max_z[box_indices[iz][0]]);

                        printf("** Tetrahedron vertices:\n");
                        for (int v = 0; v < 4; v++) {
                            idx_t  vertex_index = mesh_geom->ref_mesh->elements[v][box_indices[iz][0]];
                            real_t vertex_x     = mesh_geom->ref_mesh->points[0][vertex_index];
                            real_t vertex_y     = mesh_geom->ref_mesh->points[1][vertex_index];
                            real_t vertex_z     = mesh_geom->ref_mesh->points[2][vertex_index];
                            printf("   Vertex %d: (%f, %f, %f)\n", v, vertex_x, vertex_y, vertex_z);
                        }
                        all_tests_passed = false;
                    }
                }

                if (box_indices_linear != NULL) {
                    free(box_indices_linear);
                }
            }  // end for iz

            if (box_indices != NULL) {
                for (int iz = 0; iz < num_z; iz++) {
                    if (box_indices[iz] != NULL) {
                        free(box_indices[iz]);
                    }
                }
                free(box_indices);
            }

            if (num_boxes != NULL) {
                free(num_boxes);
            }

            all_tests_passed_global = all_tests_passed_global && all_tests_passed;
        }  // end block of queries
    }  // end for q_xy_i

    free(z_array);

    if (all_tests_passed_global) {
        printf("All %d queries passed successfully!\n", num_queries);
    } else {
        printf("== ERROR: Some queries failed. Check the logs above for details.\n");
    }
    return total_boxes_found;
}

////////////////////////////////////////////////////////////////////////////
// query_tet_cell_list_split_3d_2d_given_xy_bench
////////////////////////////////////////////////////////////////////////////
int                                                                                           //
query_tet_cell_list_split_3d_2d_given_xy_bench(const cell_list_split_3d_2d_map_t *map,        //
                                               const boxes_t                     *boxes,      //
                                               const mesh_tet_geom_t             *mesh_geom,  //
                                               const int                          num_z,      //
                                               const int                          num_queries) {
    printf("query_tet_cell_list_split_3d_2d_given_xy_bench: Running %d random queries to benchmark cell list...\n", num_queries);

    const int queries_per_xy = num_queries;

    if (queries_per_xy <= 0) {
        printf("Number of queries per xy is zero or negative, for num_queries=%d and num_z=%d.\n", num_queries, num_z);
        return EXIT_FAILURE;
    }

    printf("Each of the %d different (x,y) points will be queried for %d different z values.\n", queries_per_xy, num_z);

    real_t *z_array = (real_t *)malloc(num_z * sizeof(real_t));

    double total_time_s      = 0.0;
    int    total_boxes_found = 0;

    struct timespec start_time_global, end_time_global;
    clock_gettime(CLOCK_MONOTONIC, &start_time_global);

    for (int q_xy_i = 0; q_xy_i < queries_per_xy; q_xy_i++) {
        const real_t x =
                map->map_lower->min_x + (map->map_lower->max_x - map->map_lower->min_x) * ((real_t)rand() / (real_t)RAND_MAX);
        const real_t y =
                map->map_lower->min_y + (map->map_lower->max_y - map->map_lower->min_y) * ((real_t)rand() / (real_t)RAND_MAX);

        for (int iz = 0; iz < num_z; iz++)
            z_array[iz] =
                    map->map_lower->min_z + (map->map_lower->max_z - map->map_lower->min_z) * ((real_t)rand() / (real_t)RAND_MAX);

        // if (q_xy_i % 5 == 0) {
        //     printf("Query %d/%d...\n", q_xy_i + 1, queries_per_xy);
        // }

        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        for (int iz = 0; iz < num_z; iz++) {
            const int tet_index = query_cell_list_3d_2d_split_map_mesh_given_xy(map, boxes, mesh_geom, x, y, &z_array[iz], 1);

            clock_gettime(CLOCK_MONOTONIC, &end_time);

            if (tet_index >= 0) {
                total_boxes_found++;
            }
        }  // END for (int iz = 0; iz < num_z; iz++)
        const double elapsed_s = (end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_nsec - start_time.tv_nsec) / 1e9;
        total_time_s += elapsed_s;
    }  // END for (int q_xy_i = 0; q_xy_i < queries_per_xy; q_xy_i++)

    clock_gettime(CLOCK_MONOTONIC, &end_time_global);
    const double total_elapsed_s = (end_time_global.tv_sec - start_time_global.tv_sec) +
                                   (double)(end_time_global.tv_nsec - start_time_global.tv_nsec) / 1e9;

    free(z_array);

    const int    effective_num_queries = queries_per_xy * num_z;
    const double avg_time_per_query_us = (total_time_s / (double)(effective_num_queries)) * 1e6;

    printf("\n");
    printf("========== BENCHMARK RESULTS ==========\n");
    printf("file: %s:%d\n", __FILE__, __LINE__);
    printf("query_tet_cell_list_split_3d_2d_given_xy_bench with %d queries (each with %d z values)\n",
           effective_num_queries,
           num_z);
    printf("Total queries: %d\n", effective_num_queries);
    printf("Total boxes found: %d\n", total_boxes_found);
    printf("Total elapsed time: %.6e seconds\n", total_elapsed_s);
    printf("Cumulative query time: %.6e seconds\n", total_time_s);
    printf("Average time per query: %.6f us\n", avg_time_per_query_us);
    printf("Queries per second: %.2f K queries/s\n", (double)effective_num_queries / total_elapsed_s / 1000.0);
    printf("========================================\n\n");

    return total_boxes_found;
}  // END Function: query_tet_cell_list_split_3d_2d_given_xy_bench
