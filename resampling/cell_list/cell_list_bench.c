#define _POSIX_C_SOURCE 199309L

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <pthread.h>

#include "cell_list_3d_map.h"
#include "cell_list_bench.h"
typedef struct thread_data_t
{
    const cell_list_3d_2d_map_t *map;
    const boxes_t *boxes;
    real_t *x_vec;
    real_t *y_vec;
    real_t *z_vec;
    int num_queries;
    int total_boxes_found;

} thread_data_t;

void box_config_set_defaults(box_config_t *config)
{
    if (!config)
        return;

    config->num_boxes = 9500000;

    config->box_size_min = 0.04;
    config->box_size_max = 0.05;

    config->x_min = 0.0;
    config->x_max = 100.0;

    config->y_min = 0.0;
    config->y_max = 100.0;

    config->z_min = 0.0;
    config->z_max = 100.0;

    config->num_xy_queries = 1000000;
    config->num_z_queries = 1000;
}

void print_box_config(const box_config_t *config)
{
    if (!config)
        return;

    printf("Box Configuration:\n");
    printf("  Number of boxes: %d\n", config->num_boxes);
    printf("  Box size min: %f\n", config->box_size_min);
    printf("  Box size max: %f\n", config->box_size_max);
    printf("  X range:      [%f, %f]\n", config->x_min, config->x_max);
    printf("  Y range:      [%f, %f]\n", config->y_min, config->y_max);
    printf("  Z range:      [%f, %f]\n", config->z_min, config->z_max);
}

////////////////////////////////////////////////
// my_cmp_int
////////////////////////////////////////////////
int my_cmp_int(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

////////////////////////////////////////////////
// query_cell_list_bench
////////////////////////////////////////////////
int query_cell_list_bench(const cell_list_3d_2d_map_t *map,
                          const boxes_t *boxes,
                          const int num_queries)
{

    printf("Running %d random queries to benchmark cell list...\n", num_queries);

    real_t *x_vec = (real_t *)malloc(num_queries * sizeof(real_t));
    real_t *y_vec = (real_t *)malloc(num_queries * sizeof(real_t));
    real_t *z_vec = (real_t *)malloc(num_queries * sizeof(real_t));

    for (int q = 0; q < num_queries; q++)
    {
        x_vec[q] = map->min_x + (map->max_x - map->min_x) * ((real_t)rand() / (real_t)RAND_MAX);
        y_vec[q] = map->min_y + (map->max_y - map->min_y) * ((real_t)rand() / (real_t)RAND_MAX);
        z_vec[q] = map->min_z + (map->max_z - map->min_z) * ((real_t)rand() / (real_t)RAND_MAX);
    }

    int total_boxes_found = 0;

    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // #pragma omp parallel for default(none) shared(map, boxes, num_queries) reduction(+ : total_boxes_found)
    for (int q = 0; q < num_queries; q++)
    {

        int *box_indices = NULL;
        int num_boxes = 0;

        query_cell_list_3d_2d_map(map,
                                  boxes,
                                  x_vec[q],
                                  y_vec[q],
                                  z_vec[q],
                                  &box_indices,
                                  &num_boxes);

        total_boxes_found += num_boxes;

        if (box_indices != NULL)
        {
            free(box_indices);
        }
    }
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    const double cpu_query_time_used = (end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("Total boxes found in %d queries: %d\n", num_queries, total_boxes_found);
    printf("Total elapsed time for %d queries: %f seconds\n", num_queries, cpu_query_time_used);
    printf("Average time per query: %f us\n", (cpu_query_time_used / num_queries) * 1e6);
    free(x_vec);
    free(y_vec);
    free(z_vec);

    return 0;
}

////////////////////////////////////////////////
// thread_function
////////////////////////////////////////////////
void *thread_function(void *arg)
{
    thread_data_t *data = (thread_data_t *)arg;

    data->total_boxes_found = 0;

    for (int q = 0; q < data->num_queries; q++)
    {

        int *box_indices = NULL;
        int num_boxes = 0;

        query_cell_list_3d_2d_map(data->map,
                                  data->boxes,
                                  data->x_vec[q],
                                  data->y_vec[q],
                                  data->z_vec[q],
                                  &box_indices,
                                  &num_boxes);

        data->total_boxes_found += num_boxes;

        if (box_indices != NULL)
        {
            free(box_indices);
        }
    }
    return NULL;
}

////////////////////////////////////////////////
// query_cell_list_bench_mt
////////////////////////////////////////////////
int query_cell_list_bench_mt(const cell_list_3d_2d_map_t *map,
                             const boxes_t *boxes,
                             const int num_queries)
{

    int num_threads = 18;
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    thread_data_t *thread_data_array = (thread_data_t *)malloc(num_threads * sizeof(thread_data_t));

    real_t *x_vec = (real_t *)malloc(num_queries * sizeof(real_t));
    real_t *y_vec = (real_t *)malloc(num_queries * sizeof(real_t));
    real_t *z_vec = (real_t *)malloc(num_queries * sizeof(real_t));

    for (int q = 0; q < num_queries; q++)
    {
        x_vec[q] = map->min_x + (map->max_x - map->min_x) * ((real_t)rand() / (real_t)RAND_MAX);
        y_vec[q] = map->min_y + (map->max_y - map->min_y) * ((real_t)rand() / (real_t)RAND_MAX);
        z_vec[q] = map->min_z + (map->max_z - map->min_z) * ((real_t)rand() / (real_t)RAND_MAX);
    }

    const clock_t start = clock();
    for (int t = 0; t < num_threads; t++)
    {
        thread_data_array[t].map = map;
        thread_data_array[t].boxes = boxes;
        thread_data_array[t].x_vec = x_vec;
        thread_data_array[t].y_vec = y_vec;
        thread_data_array[t].z_vec = z_vec;
        thread_data_array[t].num_queries = num_queries;
        thread_data_array[t].total_boxes_found = 0;

        pthread_create(&threads[t], NULL, thread_function, (void *)&thread_data_array[t]);
    }

    for (int t = 0; t < num_threads; t++)
    {
        pthread_join(threads[t], NULL);
    }
    const clock_t end = clock();
    const double cpu_query_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    int total_boxes_found = 0;
    for (int t = 0; t < num_threads; t++)
    {
        total_boxes_found += thread_data_array[t].total_boxes_found;
    }
    printf("Total boxes found in %d queries (multithreaded): %d\n", num_queries, total_boxes_found);
    printf("Average time per query (multithreaded): %e seconds\n", cpu_query_time_used / (num_queries * num_threads));

    free(x_vec);
    free(y_vec);
    free(z_vec);

    free(threads);
    free(thread_data_array);

    return 0;
}

////////////////////////////////////////////////
// query_cell_list_test
////////////////////////////////////////////////
int query_cell_list_test(const cell_list_3d_2d_map_t *map,
                         const boxes_t *boxes,
                         const int num_queries)
{

    printf("Running %d random queries to test cell list...\n", num_queries);

    bool all_tests_passed_global = true;

    int total_boxes_found = 0;
    for (int q = 0; q < num_queries; q++)
    {

        bool all_tests_passed = true;
        if (q % 300 == 0)
        {
            printf("Query %d/%d...\n", q + 1, num_queries);
        }

        const real_t x = map->min_x + (map->max_x - map->min_x) * ((real_t)rand() / (real_t)RAND_MAX);
        const real_t y = map->min_y + (map->max_y - map->min_y) * ((real_t)rand() / (real_t)RAND_MAX);
        const real_t z = map->min_z + (map->max_z - map->min_z) * ((real_t)rand() / (real_t)RAND_MAX);

        int *box_indices = NULL;
        int num_boxes = 0;

        query_cell_list_3d_2d_map(map,
                                  boxes,
                                  x,
                                  y,
                                  z,
                                  &box_indices,
                                  &num_boxes);

        total_boxes_found += num_boxes;

        int *box_indices_linear = NULL;
        int num_boxes_linear = 0;

        query_linear_search_boxes(boxes,
                                  x,
                                  y,
                                  z,
                                  &box_indices_linear,
                                  &num_boxes_linear);

        qsort(box_indices, num_boxes, sizeof(int), (int (*)(const void *, const void *))my_cmp_int);
        qsort(box_indices_linear, num_boxes_linear, sizeof(int), (int (*)(const void *, const void *))my_cmp_int);

        if (num_boxes != num_boxes_linear)
        {
            printf("== ERROR: Mismatch in number of boxes found: %d (cell list) != %d (linear search)\n", num_boxes, num_boxes_linear);
            printf("** Query point: (%f, %f, %f)\n", x, y, z);

            int max_boxes = (num_boxes > num_boxes_linear) ? num_boxes : num_boxes_linear;
            for (int i = 0; i < max_boxes; i++)
            {
                int idx_cell_list = (i < num_boxes) ? box_indices[i] : -1;
                int idx_linear = (i < num_boxes_linear) ? box_indices_linear[i] : -1;
                if (idx_cell_list != idx_linear)
                {
                    printf("== At position %d: cell list index = %d, linear search index = %d,\n** at point (%f, %f, %f)\n",
                           i, idx_cell_list, idx_linear, x, y, z);
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

        if (num_boxes == num_boxes_linear)
        {
            for (int i = 0; i < num_boxes; i++)
            {
                if (box_indices[i] != box_indices_linear[i])
                {
                    printf("Mismatch in box indices at position %d: %d (cell list) != %d (linear search)\n",
                           i, box_indices[i], box_indices_linear[i]);
                    printf("** For box index %d: box min (%f, %f, %f), box max (%f, %f, %f)\n",
                           box_indices[i],
                           boxes->min_x[box_indices[i]],
                           boxes->min_y[box_indices[i]],
                           boxes->min_z[box_indices[i]],
                           boxes->max_x[box_indices[i]],
                           boxes->max_y[box_indices[i]],
                           boxes->max_z[box_indices[i]]);
                }
            }
        }

        if (box_indices != NULL)
        {
            free(box_indices);
        }

        if (box_indices_linear != NULL)
        {
            free(box_indices_linear);
        }

        // if (all_tests_passed)
        // {
        //     // printf("Query %d passed: found %d boxes at point (%f, %f, %f)\n", q + 1, num_boxes, x, y, z);
        // }
        // else
        // {
        //     // printf("Query %d failed at point (%f, %f, %f)\n", q, x, y, z);
        // }

        all_tests_passed_global = all_tests_passed_global && all_tests_passed;
    }

    if (all_tests_passed_global)
    {
        printf("All %d queries passed successfully!\n", num_queries);
    }
    else
    {
        printf("== ERROR: Some queries failed. Check the logs above for details.\n");
    }
    return total_boxes_found;
}

////////////////////////////////////////////////
// query_cell_list_given_xy_test
////////////////////////////////////////////////
int query_cell_list_given_xy_test(const cell_list_3d_2d_map_t *map,
                                  const boxes_t *boxes,
                                  const int num_z,
                                  const int num_queries)
{

    printf("query_cell_list_given_xy_test: Running %d random queries to test cell list...\n", num_queries);

    bool all_tests_passed_global = true;

    const int queries_per_xy = num_queries / num_z;

    if (queries_per_xy <= 0)
    {
        printf("Number of queries per xy is zero or negative, for num_queries=%d and num_z=%d.\n", num_queries, num_z);
        return EXIT_FAILURE;
    }

    real_t *z_array = (real_t *)malloc(num_z * sizeof(real_t));

    printf("Each of the %d different (x,y) points will be queried for %d different z values.\n", queries_per_xy, num_z);

    int total_boxes_found = 0;

    for (int q_xy_i = 0; q_xy_i < queries_per_xy; q_xy_i++)
    {

        for (int iz = 0; iz < num_z; iz++)
            z_array[iz] = map->min_z + (map->max_z - map->min_z) * ((real_t)rand() / (real_t)RAND_MAX);

        const real_t x = map->min_x + (map->max_x - map->min_x) * ((real_t)rand() / (real_t)RAND_MAX);
        const real_t y = map->min_y + (map->max_y - map->min_y) * ((real_t)rand() / (real_t)RAND_MAX);

        //  for (int q = 0; q < num_queries; q++)
        { // Begin block of queries

            bool all_tests_passed = true;
            if (q_xy_i % 5 == 0)
            {
                printf("Query %d/%d...\n", q_xy_i + 1, queries_per_xy);
            }

            int **box_indices = NULL;
            int *num_boxes = NULL;

            query_cell_list_3d_2d_map_given_xy(map,
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
                    qsort(box_indices[iz], num_boxes[iz], sizeof(int), (int (*)(const void *, const void *))my_cmp_int);
                }
                qsort(box_indices_linear, num_boxes_linear, sizeof(int), (int (*)(const void *, const void *))my_cmp_int);

                if (box_indices == NULL)
                {
                    if (num_boxes_linear != 0)
                    {
                        printf("== ERROR: Mismatch in number of boxes found: %d (cell list) != %d (linear search)\n", 0, num_boxes_linear);
                        printf("** Query point: (%f, %f, %f)\n", x, y, z_array[iz]);
                        all_tests_passed = false;
                    }
                }
                else if (num_boxes[iz] != num_boxes_linear)
                {
                    printf("== ERROR: Mismatch in number of boxes found: %d (cell list) != %d (linear search)\n", num_boxes[iz], num_boxes_linear);
                    printf("** Query point: (%f, %f, %f)\n", x, y, z_array[iz]);

                    int max_boxes = (num_boxes[iz] > num_boxes_linear) ? num_boxes[iz] : num_boxes_linear;
                    for (int i = 0; i < max_boxes; i++)
                    {
                        int idx_cell_list = (i < num_boxes[iz]) ? box_indices[iz][i] : -1;
                        int idx_linear = (i < num_boxes_linear) ? box_indices_linear[i] : -1;
                        if (idx_cell_list != idx_linear)
                        {
                            printf("== At position %d: cell list index = %d, linear search index = %d,\n** at point (%f, %f, %f)\n",
                                   i, idx_cell_list, idx_linear, x, y, z_array[iz]);
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

                if (box_indices != NULL && num_boxes[iz] == num_boxes_linear)
                {
                    for (int i = 0; i < num_boxes[iz]; i++)
                    {
                        if (box_indices[iz][i] != box_indices_linear[i])
                        {
                            printf("Mismatch in box indices at position %d: %d (cell list) != %d (linear search)\n",
                                   i, box_indices[iz][i], box_indices_linear[i]);
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
        } // end block of queries
    } // end for q_xy_i

    free(z_array);

    if (all_tests_passed_global)
    {
        printf("All %d queries passed successfully!\n", num_queries);
    }
    else
    {
        printf("== ERROR: Some queries failed. Check the logs above for details.\n");
    }
    return total_boxes_found;
}

////////////////////////////////////////////////
// query_cell_list_given_xy_bench
////////////////////////////////////////////////
int query_cell_list_given_xy_bench(const cell_list_3d_2d_map_t *map,
                                   const boxes_t *boxes,
                                   const int num_z,
                                   const int num_queries)
{

    printf("query_cell_list_given_xy_bench: Running %d random queries to test cell list...\n", num_queries);

    const int queries_per_xy = num_queries;

    if (queries_per_xy <= 0)
    {
        printf("Number of queries per xy is zero or negative, for num_queries=%d and num_z=%d.\n", num_queries, num_z);
        return EXIT_FAILURE;
    }

    printf("Each of the %d different (x,y) points will be queried for %d different z values.\n", queries_per_xy, num_z);

    real_t *z_array = (real_t *)malloc(num_z * sizeof(real_t));

    double total_time_s = 0.0;

    int effective_num_queries = 0;
    int total_boxes_found = 0;

// #pragma omp parallel for default(none) shared(map, boxes, queries_per_xy, num_z, z_array) reduction(+ : total_time_s, effective_num_queries, total_boxes_found)
    for (int q_xy_i = 0; q_xy_i < queries_per_xy; q_xy_i++)
    {

        volatile const real_t x = map->min_x + (map->max_x - map->min_x) * ((real_t)rand() / (real_t)RAND_MAX);
        volatile const real_t y = map->min_y + (map->max_y - map->min_y) * ((real_t)rand() / (real_t)RAND_MAX);

        for (int iz = 0; iz < num_z; iz++)
            z_array[iz] = map->min_z + (map->max_z - map->min_z) * ((real_t)rand() / (real_t)RAND_MAX);

        int **box_indices = NULL;
        int *num_boxes = NULL;

        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        query_cell_list_3d_2d_map_given_xy(map,
                                           boxes,
                                           x,
                                           y,
                                           z_array,
                                           num_z,
                                           &box_indices,
                                           &num_boxes);

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
            for (int iz = 0; iz < num_z; iz++)
            {
                total_boxes_found += num_boxes[iz];
            }
            free(num_boxes);
        }

        clock_gettime(CLOCK_MONOTONIC, &end_time);

        const double elapsed_s = (end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_nsec - start_time.tv_nsec) / 1e9;
        total_time_s += elapsed_s;

        effective_num_queries += num_z;
    }

    free(z_array);

    const double avg_time_per_query_us = (double)total_time_s / ((double)(effective_num_queries)) * 1e6;

    printf("Total elapsed time for %d XY queries: %e seconds\n", effective_num_queries, total_time_s);
    printf("Total boxes found in %d queries: %d\n", effective_num_queries, total_boxes_found);
    printf("Average time per query: %.6f us\n", avg_time_per_query_us);

    return 0;
}

////////////////////////////////////////////////
// test_single_query
////////////////////////////////////////////////
void test_single_query(cell_list_3d_2d_map_t *cell_list_map,
                       boxes_t *boxes,
                       real_t x_min, real_t x_max,
                       real_t y_min, real_t y_max,
                       real_t z_min, real_t z_max,
                       double cpu_build_time_used)
{
    int *box_indices = NULL;
    int num_found_boxes = 0;

    const real_t qx = random_interval(x_min, x_max);
    const real_t qy = random_interval(y_min, y_max);
    const real_t qz = random_interval(z_min, z_max);

    const clock_t query_start = clock();
    query_cell_list_3d_2d_map(cell_list_map,
                              boxes,
                              qx,
                              qy,
                              qz,
                              &box_indices,
                              &num_found_boxes);
    const clock_t query_end = clock();
    const double cpu_query_time_used = ((double)(query_end - query_start)) / CLOCKS_PER_SEC;

    printf("Number of boxes found at point (%f, %f, %f): %d\n", qx, qy, qz, num_found_boxes);
    for (int i = 0; i < num_found_boxes; i++)
    {
        printf("Box index: %d\n", box_indices[i]);
    }

    int num_found_boxes_linear = 0;
    int *box_indices_linear = NULL;

    const clock_t linear_query_start = clock();
    query_linear_search_boxes(boxes,
                              qx,
                              qy,
                              qz,
                              &box_indices_linear,
                              &num_found_boxes_linear);
    const clock_t linear_query_end = clock();
    const double cpu_linear_query_time_used = ((double)(linear_query_end - linear_query_start)) / CLOCKS_PER_SEC;

    printf("Number of boxes found with linear search at point (%f, %f, %f): %d\n", qx, qy, qz, num_found_boxes_linear);
    for (int i = 0; i < num_found_boxes_linear; i++)
    {
        printf("Box index (linear search): %d\n", box_indices_linear[i]);
    }

    bool everything_correct = true;

    if (num_found_boxes_linear == num_found_boxes)
    {
        printf("Both methods found the same number of boxes.\n");

        qsort(box_indices, num_found_boxes, sizeof(int), (int (*)(const void *, const void *))my_cmp_int);
        qsort(box_indices_linear, num_found_boxes_linear, sizeof(int), (int (*)(const void *, const void *))my_cmp_int);

        for (int i = 0; i < num_found_boxes; i++)
        {
            if (box_indices[i] != box_indices_linear[i])
            {
                printf("Mismatch in box indices at position %d: %d (cell list) != %d (linear search)\n",
                       i, box_indices[i], box_indices_linear[i]);
                everything_correct = false;
            }
            else
            {
                printf("Box index match at position %d: (%d, %d)\n", i, box_indices[i], box_indices_linear[i]);
            }
        }
    }
    else
    {
        printf("== ERROR: Mismatch in number of boxes found: %d (cell list) != %d (linear search)\n",
               num_found_boxes, num_found_boxes_linear);
        everything_correct = false;
    }

    if (everything_correct)
    {
        printf("All box indices match between both methods.\n");
    }
    else
    {
        printf("There were mismatches in the box indices between both methods.\n");
    }

    if (box_indices != NULL)
    {
        free(box_indices);
    }

    if (box_indices_linear != NULL)
    {
        free(box_indices_linear);
    }

    printf("\nCell list build time:   %e seconds\n", cpu_build_time_used);
    printf("Cell list query time:     %e seconds\n", cpu_query_time_used);
    printf("Linear search query time: %e seconds\n", cpu_linear_query_time_used);
    printf("Acceleration factor:      %f\n", cpu_linear_query_time_used / cpu_query_time_used);
}
