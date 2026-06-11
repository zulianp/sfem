#ifndef __CELL_LIST_RASTER_GPU_H__
#define __CELL_LIST_RASTER_GPU_H__

#include "cell_list_3d_1d_map_sur_mesh.h"
#include "cell_tet2box.h"
#include "sfem_base.h"
#include "sfem_mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    boxes_t                     *bounding_boxes;
    boxes_interleaved_t         *bounding_boxes_interleaved;
    mesh_tri3_geom_t            *geom;
    cell_list_split_3d_1d_map_t *split_map;
    side_length_histograms_t     histograms;
} tri3_raster_cell_gpu_cpu_data_t;

/**
 * Device-resident datasets for the tri3 raster kernels.
 *
 * split_map.map_lower / split_map.map_upper and all arrays they reference are
 * device pointers (built on the GPU by
 * tri3_raster_mesh_cell_quad_full_gpu_build_device_data()), and
 * element_coords_device is the device copy of the tri3 AoS element coordinates
 * (nelements * 9 geom_t: x0,y0,z0,x1,y1,z1,x2,y2,z2).
 *
 * All device memory is owned by this struct and must be released with
 * tri3_raster_mesh_cell_quad_full_gpu_destroy_device_data().
 */
typedef struct {
    cell_list_split_3d_1d_map_t split_map;              /* split map with device pointers */
    geom_t                     *element_coords_device;  /* device tri3 element coords */
    ptrdiff_t                   nelements;              /* number of tri3 elements */
} tri3_raster_cell_gpu_device_data_t;

/**
 * Build the device datasets for the tri3 raster kernels: uploads the bounding
 * boxes, assembles the split 3D->1D cell list directly on the GPU, and uploads
 * the tri3 element coordinates.
 *
 * @param h_bounding_boxes Host bounding boxes (uploaded only during the build).
 * @param h_geom Host tri3 geometry; element_coords must be precomputed.
 * @param nelements Number of tri3 elements in h_geom.
 * @param split_x / split_y CDF thresholds (dx < split_x selects the lower map).
 * @param x/y/z_min, x/y/z_max Domain bounds (bounding box statistics extents).
 * @param gpu_data Output struct; must be zero-initialised before the call.
 * @return 0 on success, non-zero on failure.
 */
int                                                                                   //
tri3_raster_mesh_cell_quad_full_gpu_build_device_data(                                //
        const boxes_t                      *h_bounding_boxes,                         //
        const mesh_tri3_geom_t             *h_geom,                                   //
        const ptrdiff_t                     nelements,                                //
        const real_t                        split_x,                                  //
        const real_t                        split_y,                                  //
        const real_t                        x_min,                                    //
        const real_t                        x_max,                                    //
        const real_t                        y_min,                                    //
        const real_t                        y_max,                                    //
        const real_t                        z_min,                                    //
        const real_t                        z_max,                                    //
        tri3_raster_cell_gpu_device_data_t *gpu_data);                                //

/**
 * Free all device allocations owned by gpu_data.
 * Safe to call even if the build never ran or failed (NULL-safe internals).
 */
void                                                                          //
tri3_raster_mesh_cell_quad_full_gpu_destroy_device_data(                      //
        tri3_raster_cell_gpu_device_data_t *gpu_data);                        //

int                                                                                      //
tri3_raster_cell_quad_gpu_launch(const tri3_raster_cell_gpu_cpu_data_t *cpu_data,        //
                                 const mesh_t                          *mesh,            //
                                 const ptrdiff_t *const SFEM_RESTRICT   n,               //
                                 const ptrdiff_t *const SFEM_RESTRICT   stride,          //
                                 const geom_t *const SFEM_RESTRICT      origin,          //
                                 const geom_t *const SFEM_RESTRICT      delta,           //
                                 const real_t *const SFEM_RESTRICT      weighted_field,  //
                                 real_t *const SFEM_RESTRICT            data);           //

int                                                                                         //
tri3_raster_cell_quad_bl_gpu_launch(const tri3_raster_cell_gpu_cpu_data_t *cpu_data,        //
                                    const mesh_t                          *mesh,            //
                                    const ptrdiff_t *const SFEM_RESTRICT   n,               //
                                    const ptrdiff_t *const SFEM_RESTRICT   stride,          //
                                    const geom_t *const SFEM_RESTRICT      origin,          //
                                    const geom_t *const SFEM_RESTRICT      delta,           //
                                    const real_t *const SFEM_RESTRICT      weighted_field,  //
                                    real_t *const SFEM_RESTRICT            data);           //

/**
 * Launch the block-per-thread raster kernel using datasets that already reside
 * in GPU memory (built by tri3_raster_mesh_cell_quad_full_gpu_build_device_data()).
 * Skips the host→device copies of the split map and tri3 geometry; ownership of
 * the device data stays with gpu_data.
 */
int                                                                                            //
tri3_raster_cell_quad_bl_full_gpu_launch(const tri3_raster_cell_gpu_device_data_t *gpu_data,   //
                                         const mesh_t                             *mesh,       //
                                         const ptrdiff_t *const SFEM_RESTRICT      n,          //
                                         const ptrdiff_t *const SFEM_RESTRICT      stride,     //
                                         const geom_t *const SFEM_RESTRICT         origin,     //
                                         const geom_t *const SFEM_RESTRICT         delta,      //
                                         const real_t *const SFEM_RESTRICT         weighted_field,  //
                                         real_t *const SFEM_RESTRICT               data);      //

int                                                                                  //
tri3_raster_mesh_cell_quad_gpu(const ptrdiff_t                      start_element,   //
                               const ptrdiff_t                      end_element,     //
                               const mesh_t                        *mesh,            //
                               const ptrdiff_t *const SFEM_RESTRICT n,               //
                               const ptrdiff_t *const SFEM_RESTRICT stride,          //
                               const geom_t *const SFEM_RESTRICT    origin,          //
                               const geom_t *const SFEM_RESTRICT    delta,           //
                               const real_t *const SFEM_RESTRICT    weighted_field,  //
                               real_t *const SFEM_RESTRICT          data);           //

/**
 * Same signature and functionality as tri3_raster_mesh_cell_quad_gpu(), but the
 * cell list used for the rasterization is assembled directly on the GPU
 * (only the bounding boxes and the tri3 element coordinates are built on the
 * CPU and uploaded).
 */
int                                                                                       //
tri3_raster_mesh_cell_quad_full_gpu(const ptrdiff_t                      start_element,   //
                                    const ptrdiff_t                      end_element,     //
                                    const mesh_t                        *mesh,            //
                                    const ptrdiff_t *const SFEM_RESTRICT n,               //
                                    const ptrdiff_t *const SFEM_RESTRICT stride,          //
                                    const geom_t *const SFEM_RESTRICT    origin,          //
                                    const geom_t *const SFEM_RESTRICT    delta,           //
                                    const real_t *const SFEM_RESTRICT    weighted_field,  //
                                    real_t *const SFEM_RESTRICT          data);           //

#ifdef __cplusplus
}
#endif

#endif /* __CELL_LIST_RASTER_GPU_H__ */