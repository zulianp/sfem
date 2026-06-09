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

int                                                                                //
tri3_raster_cell_quad_gpu_launch(const tri3_raster_cell_gpu_cpu_data_t *cpu_data,  //
                                 const mesh_t                          *mesh,      //
                                 const ptrdiff_t *const SFEM_RESTRICT   n,         //
                                 const ptrdiff_t *const SFEM_RESTRICT   stride,    //
                                 const geom_t *const SFEM_RESTRICT      origin,    //
                                 const geom_t *const SFEM_RESTRICT      delta,     //
                                 const real_t *const SFEM_RESTRICT      weighted_field,  //
                                 real_t *const SFEM_RESTRICT            data);     //

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

#ifdef __cplusplus
}
#endif

#endif /* __CELL_LIST_RASTER_GPU_H__ */