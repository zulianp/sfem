#include <stddef.h>

#include "cell_list_cuda.cuh"

//////////////////////////////////////////////////
// copy_cell_list_3d_2d_map_to_device
//////////////////////////////////////////////////
cell_list_3d_2d_map_t                                                   //
copy_cell_list_3d_2d_map_to_device(const cell_list_3d_2d_map_t *h_map,  //
                                   cudaStream_t                 stream) {
    cell_list_3d_2d_map_t d_map;

    /* ── scalar fields ── */
    d_map.total_num_2d_cells     = h_map->total_num_2d_cells;
    d_map.total_num_dict_entries = h_map->total_num_dict_entries;
    d_map.delta_x                = h_map->delta_x;
    d_map.delta_y                = h_map->delta_y;
    d_map.delta_z                = h_map->delta_z;
    d_map.min_x                  = h_map->min_x;
    d_map.min_y                  = h_map->min_y;
    d_map.min_z                  = h_map->min_z;
    d_map.max_x                  = h_map->max_x;
    d_map.max_y                  = h_map->max_y;
    d_map.max_z                  = h_map->max_z;
    d_map.num_cells_x            = h_map->num_cells_x;
    d_map.num_cells_y            = h_map->num_cells_y;
    d_map.num_cells_z            = h_map->num_cells_z;

    /* ── sizes ── */
    const size_t cell_ptr_bytes = (h_map->total_num_2d_cells + 1) * sizeof(int);
    const size_t dict_bytes     = h_map->total_num_dict_entries * sizeof(int);
    const size_t bounds_bytes   = h_map->total_num_dict_entries * sizeof(real_t);

    /* ── allocate device arrays ── */
    cudaMallocAsync((void **)&d_map.cell_ptr, cell_ptr_bytes, stream);
    cudaMallocAsync((void **)&d_map.cell_dict, dict_bytes, stream);
    cudaMallocAsync((void **)&d_map.lower_bounds_z, bounds_bytes, stream);
    cudaMallocAsync((void **)&d_map.upper_bounds_z, bounds_bytes, stream);

    cudaStreamSynchronize(stream);  // ensure all allocations are complete before starting copies

    /* ── async H→D copies ── */
    cudaMemcpyAsync(d_map.cell_ptr, h_map->cell_ptr, cell_ptr_bytes, cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(d_map.cell_dict, h_map->cell_dict, dict_bytes, cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(d_map.lower_bounds_z, h_map->lower_bounds_z, bounds_bytes, cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(d_map.upper_bounds_z, h_map->upper_bounds_z, bounds_bytes, cudaMemcpyHostToDevice, stream);

    return d_map; /* struct of device pointers + scalars */
}

//////////////////////////////////////////////////
// free_cell_list_3d_2d_map_device
//////////////////////////////////////////////////
void  //
free_cell_list_3d_2d_map_device(cell_list_3d_2d_map_t *d_map, cudaStream_t stream) {
    cudaFreeAsync(d_map->cell_ptr, stream);
    cudaFreeAsync(d_map->cell_dict, stream);
    cudaFreeAsync(d_map->lower_bounds_z, stream);
    cudaFreeAsync(d_map->upper_bounds_z, stream);

    d_map->cell_ptr       = NULL;
    d_map->cell_dict      = NULL;
    d_map->lower_bounds_z = NULL;
    d_map->upper_bounds_z = NULL;
}

/////////////////////////////////////////////////
// copy_boxes_to_device
/////////////////////////////////////////////////
boxes_t copy_boxes_to_device(const boxes_t *h_boxes, cudaStream_t stream) {
    boxes_t d_boxes;

    /* ── scalar fields ── */
    d_boxes.num_boxes = h_boxes->num_boxes;

    /* ── sizes ── */
    const size_t bounds_bytes = h_boxes->num_boxes * sizeof(real_t);

    /* ── async allocate device arrays ── */
    cudaMallocAsync((void **)&d_boxes.min_x, bounds_bytes, stream);
    cudaMallocAsync((void **)&d_boxes.min_y, bounds_bytes, stream);
    cudaMallocAsync((void **)&d_boxes.min_z, bounds_bytes, stream);
    cudaMallocAsync((void **)&d_boxes.max_x, bounds_bytes, stream);
    cudaMallocAsync((void **)&d_boxes.max_y, bounds_bytes, stream);
    cudaMallocAsync((void **)&d_boxes.max_z, bounds_bytes, stream);

    /* ── wait for all allocations to complete ── */
    cudaStreamSynchronize(stream);

    /* ── async H→D copies ── */
    cudaMemcpyAsync(d_boxes.min_x, h_boxes->min_x, bounds_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_boxes.min_y, h_boxes->min_y, bounds_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_boxes.min_z, h_boxes->min_z, bounds_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_boxes.max_x, h_boxes->max_x, bounds_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_boxes.max_y, h_boxes->max_y, bounds_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_boxes.max_z, h_boxes->max_z, bounds_bytes, cudaMemcpyHostToDevice, stream);

    return d_boxes;
}

/////////////////////////////////////////////////
// free_boxes_device
/////////////////////////////////////////////////
void free_boxes_device(boxes_t *d_boxes, cudaStream_t stream) {
    cudaFreeAsync(d_boxes->min_x, stream);
    cudaFreeAsync(d_boxes->min_y, stream);
    cudaFreeAsync(d_boxes->min_z, stream);
    cudaFreeAsync(d_boxes->max_x, stream);
    cudaFreeAsync(d_boxes->max_y, stream);
    cudaFreeAsync(d_boxes->max_z, stream);

    cudaStreamSynchronize(stream);  // ensure all frees are complete before setting pointers to NULL

    d_boxes->min_x = NULL;
    d_boxes->min_y = NULL;
    d_boxes->min_z = NULL;
    d_boxes->max_x = NULL;
    d_boxes->max_y = NULL;
    d_boxes->max_z = NULL;
}

//////////////////////////////////////////////////////
// cell_list_split_3d_2d_map
//////////////////////////////////////////////////////
cell_list_split_3d_2d_map_t copy_cell_list_split_3d_2d_map_to_device(const cell_list_split_3d_2d_map_t *h_split,
                                                                     cudaStream_t                       stream) {
    // ```

    // ### Two-level copy explained
    // ```
    // Host side                          Device side
    // ─────────────────────────────      ──────────────────────────────────────
    // h_split
    //   ├── split_x / split_y  ──────►  d_split.split_x / split_y  (scalars)
    //   ├── map_lower*                   d_split.map_lower* ──► [ cell_list_3d_2d_map_t ]
    //   │     ├── cell_ptr*  ────────────────────────────────────────► [ int[] ]
    //   │     ├── cell_dict* ────────────────────────────────────────► [ int[] ]
    //   │     └── ...                                                   ...
    //   └── map_upper*                   d_split.map_upper* ──► [ cell_list_3d_2d_map_t ]
    //         └── ...

    cell_list_split_3d_2d_map_t d_split;

    /* ── scalar fields ── */
    d_split.split_x = h_split->split_x;
    d_split.split_y = h_split->split_y;

    /* ── copy inner maps to device (returns structs with device pointers) ── */
    cell_list_3d_2d_map_t d_map_lower = copy_cell_list_3d_2d_map_to_device(h_split->map_lower, stream);
    cell_list_3d_2d_map_t d_map_upper = copy_cell_list_3d_2d_map_to_device(h_split->map_upper, stream);

    /* ── allocate device memory for the map structs themselves ── */
    cudaMallocAsync((void **)&d_split.map_lower, sizeof(cell_list_3d_2d_map_t), stream);
    cudaMallocAsync((void **)&d_split.map_upper, sizeof(cell_list_3d_2d_map_t), stream);

    /* ── wait for allocations to complete ── */
    cudaStreamSynchronize(stream);

    /* ── copy the map structs (containing device pointers) to device ── */
    cudaMemcpyAsync(d_split.map_lower, &d_map_lower, sizeof(cell_list_3d_2d_map_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_split.map_upper, &d_map_upper, sizeof(cell_list_3d_2d_map_t), cudaMemcpyHostToDevice, stream);

    return d_split;
}

//////////////////////////////////////////////
// free_cell_list_split_3d_2d_map_device
//////////////////////////////////////////////
void free_cell_list_split_3d_2d_map_device(cell_list_split_3d_2d_map_t *d_split, cudaStream_t stream) {
    /* ── retrieve map structs back to host to access inner device pointers ── */
    cell_list_3d_2d_map_t d_map_lower, d_map_upper;

    cudaMemcpyAsync(&d_map_lower, d_split->map_lower, sizeof(cell_list_3d_2d_map_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&d_map_upper, d_split->map_upper, sizeof(cell_list_3d_2d_map_t), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    /* ── free inner map arrays ── */
    free_cell_list_3d_2d_map_device(&d_map_lower, stream);
    free_cell_list_3d_2d_map_device(&d_map_upper, stream);

    /* ── free the map structs themselves ── */
    cudaFreeAsync(d_split->map_lower, stream);
    cudaFreeAsync(d_split->map_upper, stream);

    d_split->map_lower = NULL;
    d_split->map_upper = NULL;
}

/* ── copy to device ── */
mesh_tet_geom_device_t copy_mesh_tet_geom_to_device(const mesh_tet_geom_t *h_geom, int nelements, cudaStream_t stream) {
    mesh_tet_geom_device_t d_geom;

    /* ── scalar fields ── */
    d_geom.nelements = nelements;

    /* ── sizes ── */
    const size_t inv_jac_bytes  = nelements * 9 * sizeof(real_t);
    const size_t vet_zero_bytes = nelements * 3 * sizeof(real_t);

    /* ── async allocate ── */
    cudaMallocAsync((void **)&d_geom.inv_Jacobian, inv_jac_bytes, stream);
    cudaMallocAsync((void **)&d_geom.vetices_zero, vet_zero_bytes, stream);

    /* ── wait for allocations to complete ── */
    cudaStreamSynchronize(stream);

    /* ── async H→D copies ── */
    cudaMemcpyAsync(d_geom.inv_Jacobian, h_geom->inv_Jacobian, inv_jac_bytes, cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(d_geom.vetices_zero, h_geom->vetices_zero, vet_zero_bytes, cudaMemcpyHostToDevice, stream);

    return d_geom;
}

/* ── companion free ── */
void free_mesh_tet_geom_device(mesh_tet_geom_device_t *d_geom, cudaStream_t stream) {
    cudaFreeAsync(d_geom->inv_Jacobian, stream);
    cudaFreeAsync(d_geom->vetices_zero, stream);

    d_geom->inv_Jacobian = NULL;
    d_geom->vetices_zero = NULL;
    d_geom->nelements    = 0;
}
