#ifndef __SFEM_RESAMPLE_FIELD_ADJOINT_CELL_H__
#define __SFEM_RESAMPLE_FIELD_ADJOINT_CELL_H__

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "cell_list_bench.h"
#include "cell_tet2box.h"
#include "resampling_utils.h"
#include "sfem_base.h"
#include "sfem_resample_field.h"

/**
 * @brief Build bounding boxes and mesh geometry for the given mesh
 * @param mesh The input mesh
 * @param boxes Pointer to store the generated boxes data structure
 * @param mesh_geom Pointer to store the generated mesh geometry data structure
 * @return int 0 on success, EXIT_FAILURE on failure
 */
int                                                           //
build_bounding_boxes_mesh_geom(const mesh_t     *mesh,        //
                               boxes_t         **boxes,       //
                               mesh_tet_geom_t **mesh_geom);  //

/**
 * @brief Build bounding box statistics and side length histograms for the given boxes
 * @param boxes The input boxes data structure
 * @param bins Number of bins to use for the side length histograms
 * @param stats Pointer to store the generated bounding box statistics
 * @param histograms Pointer to store the generated side length histograms
 * @return int 0 on success, EXIT_FAILURE on failure
 */
int                                                                    //
build_bounding_box_statistics(const boxes_t              *boxes,       //
                              const int                   bins,        //
                              bounding_box_statistics_t **stats,       //
                              side_length_histograms_t  **histograms);  //

#endif  // __SFEM_RESAMPLE_FIELD_ADJOINT_CELL_H__