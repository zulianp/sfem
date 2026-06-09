#ifndef __CELL_LIST_RASTER_GPU_H__
#define __CELL_LIST_RASTER_GPU_H__

typedef struct {
    boxes_t                     *bounding_boxes;
    boxes_interleaved_t         *bounding_boxes_interleaved;
    mesh_tri3_geom_t            *geom;
    cell_list_split_3d_1d_map_t *split_map;
    side_length_histograms_t     histograms;
} tri3_raster_cell_gpu_cpu_data_t;

#endif /* __CELL_LIST_RASTER_GPU_H__ */