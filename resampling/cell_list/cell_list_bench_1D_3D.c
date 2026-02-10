#define _POSIX_C_SOURCE 199309L

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <pthread.h>

#include "cell_list_3d_map.h"
#include "cell_list_bench.h"
#include "cell_list_3d_1d_map.h"
#include "cell_list_3d_1d_map.h"

////////////////////////////////////////////////
// cmp_int
////////////////////////////////////////////////
static int cmp_int(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

////////////////////////////////////////////////
// main_1D_3D
////////////////////////////////////////////////
int main_1D_3D(int argc, char **argv, box_config_t config)
{

    (void)argc;
    (void)argv;

    unsigned int rand_seed;
    const char *env_seed = getenv("RAND_SEED");

    if (env_seed != NULL)
    {
        rand_seed = (unsigned int)strtoul(env_seed, NULL, 10);
    }
    else
    {
        rand_seed = (unsigned int)time(NULL);
    }
    printf("-- Random seed: %u\n", rand_seed);

    srand(rand_seed);

    const real_t box_size_min_x = config.box_size_min;
    const real_t box_size_max_x = random_interval(config.box_size_min, config.box_size_max);
    const real_t box_size_min_y = config.box_size_min;
    const real_t box_size_max_y = random_interval(config.box_size_min, config.box_size_max);
    const real_t box_size_min_z = config.box_size_min;
    const real_t box_size_max_z = random_interval(config.box_size_min, config.box_size_max);

    print_box_config(&config);

    printf("Generating %d random boxes...\n", config.num_boxes);
    boxes_t *boxes = allocate_boxes_t(config.num_boxes);
    make_random_boxes(boxes,
                      config.x_min,
                      config.x_max,
                      config.y_min,
                      config.y_max,
                      config.z_min,
                      config.z_max,
                      box_size_min_x,
                      box_size_max_x,
                      box_size_min_y,
                      box_size_max_y,
                      box_size_min_z,
                      box_size_max_z);

    cell_list_3d_1d_map_t *cell_list_map = make_empty_cell_list_3d_1d_map();

    const clock_t start = clock();

    build_cell_list_3d_1d_map(cell_list_map,
                              boxes->min_x,
                              boxes->min_y,
                              boxes->min_z,
                              boxes->max_x,
                              boxes->max_y,
                              boxes->max_z,
                              config.num_boxes,
                              config.x_min,
                              config.x_max,
                              config.y_min,
                              config.y_max,
                              config.z_min,
                              config.z_max);

    const clock_t end = clock();
    const double cpu_build_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("\nCell list build time:   %e seconds\n", cpu_build_time_used);

    int64_t cell_list_mem_bytes = cell_list_3d_1d_map_bytes(cell_list_map);
    const double cell_list_MB = ((double)cell_list_mem_bytes) / (1024.0 * 1024.0);
    //
    printf("Cell list uses %ld bytes of memory (%.2f MB).\n",
           cell_list_mem_bytes,
           cell_list_MB);

    if (1)
    {
        query_cell_list_1d_3d_given_xy_bench(cell_list_map, boxes, config.num_z_queries, config.num_xy_queries);

        free_cell_list_3d_1d_map(cell_list_map);
        free_boxes_t(boxes);

        return EXIT_SUCCESS;
    }

    const real_t xq = random_interval(config.x_min, config.x_max);
    const real_t yq = random_interval(config.y_min, config.y_max);
    const real_t zq = random_interval(config.z_min, config.z_max);

    int *box_indices = NULL;
    int num_found_boxes = 0;

    const clock_t query_start = clock();
    query_cell_list_3d_1d_map(cell_list_map,
                              boxes,
                              xq,
                              yq,
                              zq,
                              &box_indices,
                              &num_found_boxes);
    const clock_t query_end = clock();
    const double cpu_query_time_used = ((double)(query_end - query_start)) / CLOCKS_PER_SEC;

    int num_found_boxes_linear = 0;
    int *box_indices_linear = NULL;

    const clock_t linear_query_start = clock();
    query_linear_search_boxes(boxes,
                              xq,
                              yq,
                              zq,
                              &box_indices_linear,
                              &num_found_boxes_linear);
    const clock_t linear_query_end = clock();

    printf("\nSingle query at point (%f, %f, %f):\n", xq, yq, zq);
    printf("Number of boxes found: %d\n", num_found_boxes);

    qsort(box_indices, num_found_boxes, sizeof(int), cmp_int);
    printf("Number of boxes found with linear search: %d\n", num_found_boxes_linear);
    qsort(box_indices_linear, num_found_boxes_linear, sizeof(int), cmp_int);

    // for (int i = 0; i < num_found_boxes; i++)
    // {
    //     printf("Box index: %d\n", box_indices[i]);
    // }
    printf("Query time: %e seconds\n", cpu_query_time_used);

    const double cpu_linear_query_time_used = ((double)(linear_query_end - linear_query_start)) / CLOCKS_PER_SEC;
    printf("\nNumber of boxes found with linear search: %d\n", num_found_boxes_linear);
    // for (int i = 0; i < num_found_boxes_linear; i++)
    // {
    //     printf("Box index (linear search): %d\n", box_indices_linear[i]);
    // }

    if (num_found_boxes == num_found_boxes_linear)
    {
        printf("Both methods found the same number of boxes.\n");

        bool all_match = true;
        for (int i = 0; i < num_found_boxes; i++)
        {
            if (box_indices[i] != box_indices_linear[i])
            {
                all_match = false;
                break;
            }
        }
        if (all_match)
        {
            printf("Both methods found the same boxes.\n");
        }
        else
        {
            printf("Mismatch in box indices between methods.\n");
        }
    }
    else
    {
        printf("Mismatch in number of boxes found between methods.\n");
    }

    printf("Linear search query time: %e seconds\n", cpu_linear_query_time_used);

    free_boxes_t(boxes);
    free_cell_list_3d_1d_map(cell_list_map);
    if (box_indices != NULL)
    {
        free(box_indices);
    }
    if (box_indices_linear != NULL)
    {
        free(box_indices_linear);
    }

    return EXIT_SUCCESS;
}

//////////////////////////////////////////////////
// query_cell_list_1d_3d_given_xy_test
//////////////////////////////////////////////////
int query_cell_list_1d_3d_given_xy_test(const cell_list_3d_1d_map_t *map, //
                                        const boxes_t *boxes,             //
                                        const int num_z,                  //
                                        const int num_queries)
{

    printf("** Running %d random XY queries to test cell list...\n", num_queries);

    bool all_tests_passed_global = true;
    real_t *z_array = (real_t *)malloc(num_z * sizeof(real_t));

    int total_boxes_found = 0;
    for (int q_xy_i = 0; q_xy_i < num_queries; q_xy_i++)
    {

        volatile const real_t x = map->min_x + (map->max_x - map->min_x) * ((real_t)rand() / (real_t)RAND_MAX);
        volatile const real_t y = map->min_y + (map->max_y - map->min_y) * ((real_t)rand() / (real_t)RAND_MAX);

        for (int iz = 0; iz < num_z; iz++)
        {
            z_array[iz] = map->min_z + (map->max_z - map->min_z) * ((real_t)rand() / (real_t)RAND_MAX);
        }

        bool all_tests_passed = true;
        if (q_xy_i % 300 == 0)
        {
            printf("Query %d/%d...\n", q_xy_i + 1, num_queries);
        }

        int **box_indices = NULL;
        int *num_boxes = NULL;

        query_cell_list_3d_1d_map_given_xy(map,
                                           boxes,
                                           x,
                                           y,
                                           z_array,
                                           num_z,
                                           &box_indices,
                                           &num_boxes);

        if (num_boxes != NULL)
        {
            for (int iz = 0; iz < num_z; iz++)
            {
                total_boxes_found += num_boxes[iz];
            }
        }

        for (int iz = 0; iz < num_z; iz++)
        {
            int *box_indices_linear = NULL;
            int num_boxes_linear = 0;

            query_linear_search_boxes(boxes,
                                      x,
                                      y,
                                      z_array[iz],
                                      &box_indices_linear,
                                      &num_boxes_linear);

            if (box_indices != NULL)
            {
                qsort(box_indices[iz], num_boxes[iz], sizeof(int), cmp_int);
            }
            qsort(box_indices_linear, num_boxes_linear, sizeof(int), cmp_int);

            if (box_indices == NULL)
            {
                if (num_boxes_linear != 0)
                {
                    all_tests_passed = false;
                    printf("== ERROR: Query %d at iz=%d: cell list returned NULL but linear search found %d boxes.\n",
                           q_xy_i,
                           iz,
                           num_boxes_linear);
                }
            }

            else if (num_boxes[iz] != num_boxes_linear)
            {
                all_tests_passed = false;
                printf("== ERROR: Query %d at iz=%d: Mismatch in number of boxes found: %d (cell list) != %d (linear search)\n",
                       q_xy_i,
                       iz,
                       num_boxes[iz],
                       num_boxes_linear);
            }
            else
            {
                // check box indices
                for (int i = 0; i < num_boxes[iz]; i++)
                {
                    if (box_indices[iz][i] != box_indices_linear[i])
                    {
                        all_tests_passed = false;
                        printf("== ERROR: Query %d at iz=%d: Mismatch in box index at position %d: %d (cell list) != %d (linear search)\n",
                               q_xy_i,
                               iz,
                               i,
                               box_indices[iz][i],
                               box_indices_linear[i]);
                    }
                }
            }

            if (box_indices_linear != NULL)
            {
                free(box_indices_linear);
            }
        } // end for iz
        if (box_indices != NULL)
        {
            for (int iz = 0; iz < num_z; iz++)
            {
                if (box_indices[iz] != NULL)
                {
                    free(box_indices[iz]);
                }
            }
            free(box_indices);
        }

        if (num_boxes != NULL)
        {
            free(num_boxes);
        }

        all_tests_passed_global = all_tests_passed_global && all_tests_passed;

    } // end for q_xy_i

    free(z_array);

    if (all_tests_passed_global)
    {
        printf("All %d XY queries passed successfully!\n", num_queries);
    }
    else
    {
        printf("== ERROR: Some XY queries failed. Check the logs above for details.\n");
    }

    printf("Total boxes found across all queries: %d\n", total_boxes_found);
    printf("------------------------------------------------------\n\n");
    return total_boxes_found;
}

//////////////////////////////////////////////////
// query_cell_list_1d_3d_given_xy_bench
//////////////////////////////////////////////////
int query_cell_list_1d_3d_given_xy_bench(const cell_list_3d_1d_map_t *map,
                                         const boxes_t *boxes,
                                         const int num_z,
                                         const int num_queries)
{
    int total_boxes_found = 0;
    double total_time_s = 0.0;
    int total_queries = 0;

    real_t *z_array = (real_t *)malloc(num_z * sizeof(real_t));

    for (int q_xy_i = 0; q_xy_i < num_queries; q_xy_i++)
    {
        const real_t x = map->min_x + (map->max_x - map->min_x) * ((real_t)rand() / (real_t)RAND_MAX);
        const real_t y = map->min_y + (map->max_y - map->min_y) * ((real_t)rand() / (real_t)RAND_MAX);

        for (int iz = 0; iz < num_z; iz++)
        {
            z_array[iz] = map->min_z + (map->max_z - map->min_z) * ((real_t)rand() / (real_t)RAND_MAX);
        }

        int **box_indices = NULL;
        int *num_boxes = NULL;

        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        query_cell_list_3d_1d_map_given_xy(map,
                                           boxes,
                                           x,
                                           y,
                                           z_array,
                                           num_z,
                                           &box_indices,
                                           &num_boxes);

        if (num_boxes != NULL)
        {
            for (int iz = 0; iz < num_z; iz++)
            {
                total_boxes_found += num_boxes[iz];
            }
        }

        if (box_indices != NULL)
        {
            for (int iz = 0; iz < num_z; iz++)
            {
                if (box_indices[iz] != NULL)
                {
                    free(box_indices[iz]);
                }
            }
            free(box_indices);
        }

        if (num_boxes != NULL)
        {
            free(num_boxes);
        }
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        const double elapsed_time = (double)(end_time.tv_sec - start_time.tv_sec) +        //
                                    (double)(end_time.tv_nsec - start_time.tv_nsec) / 1e9; //
        total_time_s += elapsed_time;
        total_queries += num_z;
    }

    free(z_array);

    printf("\nTotal box found across %d queries: %d\n", total_queries, total_boxes_found);
    printf("Total elapsed time for %d queries: %f seconds\n", total_queries, total_time_s);
    printf("Average time per query over %d queries: %.6f us\n", total_queries, (total_time_s / (double)total_queries) * 1e6);

    return total_boxes_found;
}