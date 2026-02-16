#ifndef __CELL_TET2BOX_H__
#define __CELL_TET2BOX_H__

#include <stdbool.h>
#include <stddef.h>

#include "sfem_base.h"

#include "cell_list_3d_map.h"
#include "precision_types.h"

//////////////////////////////////////////////////////////
// bounding_box_statistics_t
//////////////////////////////////////////////////////////
typedef struct {
    real_t max_box_side_x;
    real_t max_box_side_y;
    real_t max_box_side_z;
    real_t min_box_side_x;
    real_t min_box_side_y;
    real_t min_box_side_z;
    real_t avg_box_side_x;
    real_t avg_box_side_y;
    real_t avg_box_side_z;
    real_t max_volume;
    real_t min_volume;
    real_t avg_volume;
    int max_volume_idx;
    int min_volume_idx;
    real_t max_volume_side_x;
    real_t max_volume_side_y;
    real_t max_volume_side_z;
    real_t min_volume_side_x;
    real_t min_volume_side_y;
    real_t min_volume_side_z;
    real_t volume_ratio;
} bounding_box_statistics_t;

//////////////////////////////////////////////////////////
// calculate_bounding_box_statistics
//////////////////////////////////////////////////////////
bounding_box_statistics_t calculate_bounding_box_statistics(const boxes_t *boxes);

//////////////////////////////////////////////////////////
// print_bounding_box_statistics
//////////////////////////////////////////////////////////
void print_bounding_box_statistics(const bounding_box_statistics_t *stats);

//////////////////////////////////////////////////////////
// side_length_histogram_t
//////////////////////////////////////////////////////////
/**
 * @brief Structure to hold histogram data for side lengths
 */
typedef struct {
    int num_classes;
    real_t min_value;
    real_t max_value;
    real_t bin_width;
    int *counts;  // Array of size num_classes
} side_length_histogram_t;

//////////////////////////////////////////////////////////
// side_length_histograms_t
//////////////////////////////////////////////////////////
/**
 * @brief Structure to hold histograms for all three dimensions
 */
typedef struct {
    side_length_histogram_t x_histogram;
    side_length_histogram_t y_histogram;
    side_length_histogram_t z_histogram;
} side_length_histograms_t;

//////////////////////////////////////////////////////////
// calculate_side_length_histograms
//////////////////////////////////////////////////////////
side_length_histograms_t calculate_side_length_histograms(
    const boxes_t *boxes,
    const bounding_box_statistics_t *stats,
    const int num_classes);

//////////////////////////////////////////////////////////
// print_side_length_histograms
//////////////////////////////////////////////////////////
void print_side_length_histograms(const side_length_histograms_t *histograms);

//////////////////////////////////////////////////////////
// free_side_length_histograms
//////////////////////////////////////////////////////////
void free_side_length_histograms(side_length_histograms_t *histograms);

//////////////////////////////////////////////////////////
// write_side_length_histograms
//////////////////////////////////////////////////////////
int write_side_length_histograms(const side_length_histograms_t *histograms,
                                 const char *output_dir);

int                                                                     //
make_mesh_tets_boxes(const ptrdiff_t                    start_element,  // Mesh
                     const ptrdiff_t                    end_element,    //
                     const ptrdiff_t                    nnodes,         //
                     const idx_t** const SFEM_RESTRICT  elems,          //
                     const geom_t** const SFEM_RESTRICT xyz,            //
                     boxes_t**                          boxes);

#endif  // __CELL_TET2BOX_H__