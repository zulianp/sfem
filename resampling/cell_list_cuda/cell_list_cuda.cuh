#ifndef CELL_LIST_CUDA_CUH
#define CELL_LIST_CUDA_CUH

#include <cuda_runtime.h>

#include "cell_list_3d_map.h"
#include "sfem_mesh.h"

/* ── device-side struct (no ref_mesh) ── */
typedef struct {
    real_t *inv_Jacobian;
    real_t *vetices_zero;
    int     nelements;
} mesh_tet_geom_device_t;

/**
 * @brief Copies a cell_list_3d_2d_map_t from host to device, including device allocations and async copies for the arrays.
 * @param h_map Pointer to the host cell list map to copy.
 * @param stream CUDA stream to use for the async copies.
 * @return A cell_list_3d_2d_map_t struct containing device pointers and scalar fields.
 */
cell_list_3d_2d_map_t                                                   //
copy_cell_list_3d_2d_map_to_device(const cell_list_3d_2d_map_t *h_map,  //
                                   cudaStream_t                 stream);

/**
 * @brief Frees the device memory allocated for the arrays in a cell_list_3d_2d_map_t and sets the pointers to NULL.
 * @param d_map Pointer to the device cell list map to free.
 * @param stream CUDA stream used for async frees.
 * Note: This function only frees the device arrays, it does not free the struct itself since it is typically returned by value.
 */
void free_cell_list_3d_2d_map_device(cell_list_3d_2d_map_t *d_map, cudaStream_t stream);

/**
 * @brief Copies a boxes_t from host to device, including async allocations and async H2D copies.
 * @param h_boxes Pointer to the host boxes struct to copy.
 * @param stream CUDA stream to use for async operations.
 * @return A boxes_t struct containing device pointers and scalar fields.
 */
boxes_t copy_boxes_to_device(const boxes_t *h_boxes, cudaStream_t stream);

/**
 * @brief Frees device memory for arrays in a boxes_t and sets pointers to NULL.
 * @param d_boxes Pointer to the device boxes struct to free.
 * @param stream CUDA stream used for async frees.
 */
void free_boxes_device(boxes_t *d_boxes, cudaStream_t stream);

/**
 * @brief Copies a cell_list_split_3d_2d_map_t from host to device.
 * @param h_split Pointer to the host split map to copy.
 * @param stream CUDA stream to use for async operations.
 * @return A cell_list_split_3d_2d_map_t struct containing device pointers and scalar fields.
 */
cell_list_split_3d_2d_map_t copy_cell_list_split_3d_2d_map_to_device(const cell_list_split_3d_2d_map_t *h_split,  //
                                                                     cudaStream_t                       stream);                        //

/**
 * @brief Frees device memory owned by a cell_list_split_3d_2d_map_t.
 * @param d_split Pointer to the device split map to free.
 * @param stream CUDA stream used for async frees.
 */
void free_cell_list_split_3d_2d_map_device(cell_list_split_3d_2d_map_t *d_split,  //
                                           cudaStream_t                 stream);                  //

/**
 * @brief Copies a mesh_tet_geom_t from host to device.
 * @param h_geom Pointer to the host mesh tet geometry to copy.
 * @param nelements Number of elements in the mesh.
 * @param stream CUDA stream to use for async operations.
 * @return A mesh_tet_geom_device_t struct containing device pointers and scalar fields.
 */
mesh_tet_geom_device_t copy_mesh_tet_geom_to_device(const mesh_tet_geom_t *h_geom,     //
                                                    int                    nelements,  //
                                                    cudaStream_t           stream);              //

/**
 * @brief Frees device memory owned by a mesh_tet_geom_device_t.
 * @param d_geom Pointer to the device mesh tet geometry to free.
 * @param stream CUDA stream used for async frees.
 */
void free_mesh_tet_geom_device(mesh_tet_geom_device_t *d_geom,  //
                               cudaStream_t            stream);            //

/**
 * @brief Copies a boxes_interleaved_t from host to device, including async allocations and async H2D copy.
 * @param h_boxes Pointer to the host boxes_interleaved struct to copy.
 * @param stream CUDA stream to use for async operations.
 * @return A boxes_interleaved_t struct containing device pointers and scalar fields.
 */
boxes_interleaved_t                                                                         //
copy_boxes_interleaved_to_device(const boxes_interleaved_t *h_boxes, cudaStream_t stream);  //

/**
 * @brief Frees device memory for arrays in a boxes_interleaved_t and sets pointers to NULL.
 * @param d_boxes Pointer to the device boxes_interleaved struct to free.
 * @param stream CUDA stream used for async frees.
 */
void                                                                               //
free_boxes_interleaved_device(boxes_interleaved_t *d_boxes, cudaStream_t stream);  //

#endif  // CELL_LIST_CUDA_CUH