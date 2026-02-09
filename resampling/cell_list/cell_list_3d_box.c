#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdbool.h>

#include "cell_list_3d_map.h"

real_t random_interval(real_t min, real_t max)
{
    return min + ((real_t)rand() / (real_t)RAND_MAX) * (max - min);
}

////////////////////////////////////////////////
// make_boxes_t
////////////////////////////////////////////////
boxes_t
make_boxes_t(void)
{
    boxes_t boxes;
    init_boxes_t(&boxes);
    return boxes;
}

////////////////////////////////////////////////
// init_boxes_t
////////////////////////////////////////////////
void init_boxes_t(boxes_t *boxes)
{

    boxes->min_x = NULL;
    boxes->min_y = NULL;
    boxes->min_z = NULL;
    boxes->max_x = NULL;
    boxes->max_y = NULL;
    boxes->max_z = NULL;
    boxes->num_boxes = 0;
}

////////////////////////////////////////////////
// allocate_boxes_t
////////////////////////////////////////////////
boxes_t *
allocate_boxes_t(const int num_boxes)
{
    boxes_t *boxes = (boxes_t *)malloc(sizeof(boxes_t));
    boxes->min_x = (real_t *)malloc(num_boxes * sizeof(real_t));
    boxes->min_y = (real_t *)malloc(num_boxes * sizeof(real_t));
    boxes->min_z = (real_t *)malloc(num_boxes * sizeof(real_t));
    boxes->max_x = (real_t *)malloc(num_boxes * sizeof(real_t));
    boxes->max_y = (real_t *)malloc(num_boxes * sizeof(real_t));
    boxes->max_z = (real_t *)malloc(num_boxes * sizeof(real_t));
    boxes->num_boxes = num_boxes;
    return boxes;
}

////////////////////////////////////////////////
// check_box_containment
////////////////////////////////////////////////
bool check_box_contains_pt(const boxes_t *boxes,
                           const int box_index,
                           const real_t x,
                           const real_t y,
                           const real_t z)
{
    return (x >= boxes->min_x[box_index] && x <= boxes->max_x[box_index] &&
            y >= boxes->min_y[box_index] && y <= boxes->max_y[box_index] &&
            z >= boxes->min_z[box_index] && z <= boxes->max_z[box_index]);
}

// Scalar version: branchless, better instruction scheduling
bool check_box_contains_pt_fast(const boxes_t *boxes,
                                const int box_index,
                                const real_t x,
                                const real_t y,
                                const real_t z)
{
    // Load min/max into locals to help compiler with aliasing
    const real_t min_x = boxes->min_x[box_index];
    const real_t max_x = boxes->max_x[box_index];
    const real_t min_y = boxes->min_y[box_index];
    const real_t max_y = boxes->max_y[box_index];
    const real_t min_z = boxes->min_z[box_index];
    const real_t max_z = boxes->max_z[box_index];

    // Branchless: convert bools to 0/1, multiply (AND)
    // Compiler will likely use setcc + and instructions
    return (x >= min_x) & (x <= max_x) &
           (y >= min_y) & (y <= max_y) &
           (z >= min_z) & (z <= max_z);
}

////////////////////////////////////////////////
// free_boxes_t
////////////////////////////////////////////////
void free_boxes_t(boxes_t *boxes)
{
    if (boxes != NULL)
    {
        if (boxes->min_x != NULL)
            free(boxes->min_x);
        if (boxes->min_y != NULL)
            free(boxes->min_y);
        if (boxes->min_z != NULL)
            free(boxes->min_z);
        if (boxes->max_x != NULL)
            free(boxes->max_x);
        if (boxes->max_y != NULL)
            free(boxes->max_y);
        if (boxes->max_z != NULL)
            free(boxes->max_z);
        free(boxes);
    }
}

////////////////////////////////////////////////
// make_random_boxes
////////////////////////////////////////////////
int make_random_boxes(boxes_t *boxes,
                      const real_t x_min,
                      const real_t x_max,
                      const real_t y_min,
                      const real_t y_max,
                      const real_t z_min,
                      const real_t z_max,
                      const real_t box_size_min_x,
                      const real_t box_size_max_x,
                      const real_t box_size_min_y,
                      const real_t box_size_max_y,
                      const real_t box_size_min_z,
                      const real_t box_size_max_z)
{

    if (boxes == NULL || boxes->num_boxes <= 0)
    {
        return -1;
    }

    for (int i = 0; i < boxes->num_boxes; i++)
    {

        real_t box_size_x = box_size_min_x + ((real_t)rand() / (real_t)RAND_MAX) * (box_size_max_x - box_size_min_x);
        real_t box_size_y = box_size_min_y + ((real_t)rand() / (real_t)RAND_MAX) * (box_size_max_y - box_size_min_y);
        real_t box_size_z = box_size_min_z + ((real_t)rand() / (real_t)RAND_MAX) * (box_size_max_z - box_size_min_z);

        boxes->min_x[i] = x_min + ((real_t)rand() / (real_t)RAND_MAX) * (x_max - x_min - box_size_x);
        boxes->min_y[i] = y_min + ((real_t)rand() / (real_t)RAND_MAX) * (y_max - y_min - box_size_y);
        boxes->min_z[i] = z_min + ((real_t)rand() / (real_t)RAND_MAX) * (z_max - z_min - box_size_z);

        boxes->max_x[i] = boxes->min_x[i] + box_size_x;
        boxes->max_y[i] = boxes->min_y[i] + box_size_y;
        boxes->max_z[i] = boxes->min_z[i] + box_size_z;
    }

    return 0;
}

////////////////////////////////////////////////
// query_linear_search_boxes
////////////////////////////////////////////////
int query_linear_search_boxes(const boxes_t *boxes,
                              const real_t x,
                              const real_t y,
                              const real_t z,
                              int **box_indices,
                              int *num_boxes)
{
    *num_boxes = 0;
    const int delta_num_boxes = 10;
    *box_indices = (int *)malloc(delta_num_boxes * sizeof(int));

    for (int i = 0; i < boxes->num_boxes; i++)
    {
        if (x >= boxes->min_x[i] && x <= boxes->max_x[i] &&
            y >= boxes->min_y[i] && y <= boxes->max_y[i] &&
            z >= boxes->min_z[i] && z <= boxes->max_z[i])
        {
            if (*num_boxes % delta_num_boxes == 0 && *num_boxes > 0)
            {
                *box_indices = (int *)realloc(*box_indices, (*num_boxes + delta_num_boxes) * sizeof(int));
            }
            (*box_indices)[*num_boxes] = i;
            (*num_boxes)++;
        }
    }

    return 0;
}
