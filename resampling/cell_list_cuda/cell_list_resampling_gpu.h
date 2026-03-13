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
