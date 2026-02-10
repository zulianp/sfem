#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdbool.h>

#include "cell_list_3d_map.h"
#include "cell_list_bench.h"
#include "cell_list_3d_1d_map.h"

////////////////////////////////////////////////
// main
////////////////////////////////////////////////
int main_ccel_test(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    unsigned int rand_seed;
    const char *env_seed = getenv("RAND_SEED");
    const char *env_1d_3d = getenv("RUN_1D_3D");

    if (env_seed != NULL)
        rand_seed = (unsigned int)strtoul(env_seed, NULL, 10);
    else
        rand_seed = (unsigned int)time(NULL);

    printf("-- Random seed: %u\n", rand_seed);

    srand(rand_seed);

    // Boxes parameters:

    box_config_t config;
    box_config_set_defaults(&config);

    if (env_1d_3d != NULL && atoi(env_1d_3d) != 0)
        return main_1D_3D(argc, argv, config);

    const real_t box_size_min_x = config.box_size_min;
    const real_t box_size_max_x = random_interval(config.box_size_min, config.box_size_max);
    const real_t box_size_min_y = config.box_size_min;
    const real_t box_size_max_y = random_interval(config.box_size_min, config.box_size_max);
    const real_t box_size_min_z = config.box_size_min;
    const real_t box_size_max_z = random_interval(config.box_size_min, config.box_size_max);
    boxes_t *boxes = allocate_boxes_t(config.num_boxes);

    printf("Generating %d random boxes...\n", config.num_boxes);

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

    cell_list_3d_2d_map_t *cell_list_map = make_empty_cell_list_3d_2d_map();

    const clock_t start = clock();

    build_cell_list_3d_2d_map(cell_list_map,
                              boxes->min_x,
                              boxes->min_y,
                              boxes->min_z,
                              boxes->max_x,
                              boxes->max_y,
                              boxes->max_z,
                              boxes->num_boxes,
                              config.x_min,
                              config.x_max,
                              config.y_min,
                              config.y_max,
                              config.z_min,
                              config.z_max);

    const clock_t end = clock();
    const double cpu_build_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    int64_t cell_list_mem_bytes = cell_list_3d_2d_map_bytes(cell_list_map);
    const double cell_list_MB = ((double)cell_list_mem_bytes) / (1024.0 * 1024.0);

    printf("Cell list uses %ld bytes of memory (%.2f MB).\n",
           cell_list_mem_bytes,
           cell_list_MB);

    printf("\nCell list build time:   %e seconds\n", cpu_build_time_used);

    if (1) // run benchmark
    {

        printf("\nStarting cell list XY benchmark...\n");
        printf("--------------------------------\n");
        query_cell_list_given_xy_test //
        // query_cell_list_given_xy_bench //
            (cell_list_map,
             boxes,
             config.num_z_queries,
             config.num_xy_queries);

        const int N_queries = config.num_xy_queries * config.num_z_queries;

        printf("\nStarting cell list benchmark...\n");
        printf("--------------------------------\n");
        query_cell_list_bench(cell_list_map,
                              boxes,
                              N_queries);

        free_cell_list_3d_2d_map(cell_list_map);
        free_boxes_t(boxes);

        printf("\n\nBenchmark completed.\n\n");

        return 0;
    }
    else
    {

        test_single_query(cell_list_map, boxes,
                          config.x_min, config.x_max, config.y_min, config.y_max, config.z_min, config.z_max,
                          cpu_build_time_used);
    }

    free_cell_list_3d_2d_map(cell_list_map);
    free_boxes_t(boxes);

    return 0;
}
