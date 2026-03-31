#ifndef __CELL_LIST_RESAMPLING_GPU_H__
#define __CELL_LIST_RESAMPLING_GPU_H__

#include <stddef.h>

#include "cell_list_3d_map.h"
#include "cell_tet2box.h"
#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_mesh.h"
#include "sfem_resample_field_adjoint_hyteg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    boxes_t                     *bounding_boxes;
    boxes_interleaved_t         *bounding_boxes_interleaved;
    mesh_tet_geom_t             *geom;
    cell_list_split_3d_2d_map_t *split_map;
    side_length_histograms_t     histograms;
} tet4_resample_field_adjoint_cell_quad_gpu_cpu_data_t;

/**
 * Device-resident cell list data built on the GPU when BUILD_CELL_LIST_GPU is
 * defined.  All pointers embedded in split_map are device pointers managed by
 * CUDA and must be freed with
 * tet4_resample_field_adjoint_cell_quad_gpu_destroy_device_data().
 */
typedef struct {
    cell_list_split_3d_2d_map_t split_map; /* split map with device pointers */
} tet4_resample_field_adjoint_cell_quad_gpu_device_data_t;

/**
 * Build the split cell list entirely on the GPU.
 *
 * @param h_bounding_boxes  Host-side bounding boxes (source for the GPU upload).
 * @param split_x / split_y CDF thresholds obtained from calculate_cdf_thresholds().
 * @param x/y/z_min, x/y/z_max  Extent of the SDF grid.
 * @param gpu_data          Output struct; must be zero-initialised before the call.
 * @return 0 on success, non-zero on failure.
 */
int tet4_resample_field_adjoint_cell_quad_gpu_build_device_data(  //
        const boxes_t *h_bounding_boxes,                           //
        real_t         split_x,                                    //
        real_t         split_y,                                    //
        real_t         x_min,                                      //
        real_t         x_max,                                      //
        real_t         y_min,                                      //
        real_t         y_max,                                      //
        real_t         z_min,                                      //
        real_t         z_max,                                      //
        tet4_resample_field_adjoint_cell_quad_gpu_device_data_t *gpu_data);

/**
 * Free all device allocations owned by gpu_data.
 * Safe to call even if gpu_data was never successfully built (NULL-safe internals).
 */
void tet4_resample_field_adjoint_cell_quad_gpu_destroy_device_data(  //
        tet4_resample_field_adjoint_cell_quad_gpu_device_data_t *gpu_data);

/**
 * Launch the resampling kernel using a pre-built device-side split map.
 *
 * Compared to tet4_resample_field_adjoint_cell_quad_gpu_launch(), this variant
 * skips the CPU→GPU copy of the split map and uses gpu_data->split_map directly.
 * cpu_data must contain valid bounding_boxes_interleaved; split_map may be NULL.
 */
int tet4_resample_field_adjoint_cell_quad_gpu_launch_device_map(              //
        const tet4_resample_field_adjoint_cell_quad_gpu_cpu_data_t    *cpu_data,   //
        const tet4_resample_field_adjoint_cell_quad_gpu_device_data_t *gpu_data,   //
        const mesh_t                        *mesh,                                 //
        const ptrdiff_t *const SFEM_RESTRICT n,                                    //
        const ptrdiff_t *const SFEM_RESTRICT stride,                               //
        const geom_t *const SFEM_RESTRICT    origin,                               //
        const geom_t *const SFEM_RESTRICT    delta,                                //
        const real_t *const SFEM_RESTRICT    weighted_field,                       //
        real_t *const SFEM_RESTRICT          data);                                //

int                                                                                                  //
tet4_resample_field_adjoint_cell_quad_gpu(const ptrdiff_t                      start_element,        // Mesh
                                          const ptrdiff_t                      end_element,          //
                                          const mesh_t                        *mesh,                 //
                                          const ptrdiff_t *const SFEM_RESTRICT n,                    // SDF
                                          const ptrdiff_t *const SFEM_RESTRICT stride,               //
                                          const geom_t *const SFEM_RESTRICT    origin,               //
                                          const geom_t *const SFEM_RESTRICT    delta,                //
                                          const real_t *const SFEM_RESTRICT    weighted_field,       // Input weighted field
                                          const mini_tet_parameters_t          mini_tet_parameters,  //
                                          real_t *const SFEM_RESTRICT          data);                         // SDF: data (output)

int                                                                                                                     //
tet4_resample_field_adjoint_cell_quad_gpu_launch(const tet4_resample_field_adjoint_cell_quad_gpu_cpu_data_t *cpu_data,  //
                                                 const mesh_t                                               *mesh,      // Mesh
                                                 const ptrdiff_t *const SFEM_RESTRICT                        n,         // SDF
                                                 const ptrdiff_t *const SFEM_RESTRICT                        stride,    //
                                                 const geom_t *const SFEM_RESTRICT                           origin,    //
                                                 const geom_t *const SFEM_RESTRICT                           delta,     //
                                                 const real_t *const SFEM_RESTRICT weighted_field,  // Input weighted field
                                                 real_t *const SFEM_RESTRICT       data);                 // SDF: data (output)

#ifdef __cplusplus
}
#endif

#endif  // __CELL_LIST_RESAMPLING_GPU_H__
