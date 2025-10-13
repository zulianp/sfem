

#ifndef __SFEM_ADJOINT_MINI_LOC_TET_CUH__
#define __SFEM_ADJOINT_MINI_LOC_TET_CUH__

#include "sfem_adjoint_mini_tet.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

// #define COLLECT_ALPHA_DATA

template <typename FloatType,  //
          typename IntType = ptrdiff_t>
class tet_properties_info_t {
public:
    // Arrays allocated on device (length = count)
    IntType* min_grid_0 = nullptr;  // delta with respect to the global grid in the global grid
    IntType* min_grid_1 = nullptr;
    IntType* min_grid_2 = nullptr;

    IntType* total_size_local = nullptr;

    IntType* stride0_local = nullptr;
    IntType* stride1_local = nullptr;
    IntType* stride2_local = nullptr;

    IntType* n0_local = nullptr;
    IntType* n1_local = nullptr;
    IntType* n2_local = nullptr;

#ifdef COLLECT_ALPHA_DATA
    FloatType* tet_alpha = nullptr;  // refinement level alpha for each tet
#endif

    // Host-side meta
    size_t count = 0;

    __device__ size_t get_tet_grid_size(const IntType tet_i) const {
        return (n0_local[tet_i] * n1_local[tet_i] * n2_local[tet_i]);
    }

    tet_properties_info_t()  = default;
    ~tet_properties_info_t() = default;

    __host__ cudaError_t alloc_async(size_t n, cudaStream_t stream) {
        count           = n;
        cudaError_t err = cudaSuccess;

        auto fail_cleanup = [&](cudaError_t e) {
            (void)free_async(stream);  // best-effort cleanup
            return e;
        };

        // ptrdiff_t arrays
        if ((err = cudaMallocAsync((void**)&min_grid_0, n * sizeof(IntType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&min_grid_1, n * sizeof(IntType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&min_grid_2, n * sizeof(IntType), stream)) != cudaSuccess) return fail_cleanup(err);

        if ((err = cudaMallocAsync((void**)&total_size_local, n * sizeof(IntType), stream)) != cudaSuccess)
            return fail_cleanup(err);

        if ((err = cudaMallocAsync((void**)&stride0_local, n * sizeof(IntType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&stride1_local, n * sizeof(IntType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&stride2_local, n * sizeof(IntType), stream)) != cudaSuccess) return fail_cleanup(err);

        if ((err = cudaMallocAsync((void**)&n0_local, n * sizeof(IntType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&n1_local, n * sizeof(IntType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&n2_local, n * sizeof(IntType), stream)) != cudaSuccess) return fail_cleanup(err);

#ifdef COLLECT_ALPHA_DATA
        if ((err = cudaMallocAsync((void**)&tet_alpha, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);
#endif

        return cudaSuccess;
    }

    __host__ cudaError_t free_async(cudaStream_t stream) {
        // Free if non-null; ignore errors to attempt freeing all
        auto free_if = [&](void* p) {
            if (p) (void)cudaFreeAsync(p, stream);
        };

        free_if(min_grid_0);
        free_if(min_grid_1);
        free_if(min_grid_2);

        free_if(total_size_local);

        free_if(stride0_local);
        free_if(stride1_local);
        free_if(stride2_local);

        free_if(n0_local);
        free_if(n1_local);
        free_if(n2_local);

#ifdef COLLECT_ALPHA_DATA
        free_if(tet_alpha);
#endif

        reset();
        // Optionally, user can cudaStreamSynchronize(stream) after this
        return cudaSuccess;
    }

private:
    __host__ void reset() {
        min_grid_0 = min_grid_1 = min_grid_2 = nullptr;
        total_size_local                     = nullptr;
        stride0_local = stride1_local = stride2_local = nullptr;
        n0_local = n1_local = n2_local = nullptr;

#ifdef COLLECT_ALPHA_DATA
        tet_alpha = nullptr;
#endif

        count = 0;
    }
};

/////////////////////////////////////////////////////////////////////////////////
// Kernel to perform adjoint mini-tetrahedron resampling
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType, typename IntType = ptrdiff_t>
__global__ void                                                                               //
sfem_make_local_data_tets_kernel_gpu(const IntType                    start_element,          // Mesh
                                     const IntType                    end_element,            //
                                     const IntType                    nnodes,                 //
                                     const elems_tet4_device          elems,                  //
                                     const xyz_tet4_device            xyz,                    //
                                     const IntType                    n0,                     // SDF
                                     const IntType                    n1,                     //
                                     const IntType                    n2,                     //
                                     const IntType                    stride0,                // Stride
                                     const IntType                    stride1,                //
                                     const IntType                    stride2,                //
                                     const geom_t                     origin0,                // Origin
                                     const geom_t                     origin1,                //
                                     const geom_t                     origin2,                //
                                     const geom_t                     dx,                     // Delta
                                     const geom_t                     dy,                     //
                                     const geom_t                     dz,                     //
                                     tet_properties_info_t<FloatType> tet_properties_info) {  //

    const int tet_id    = (blockIdx.x * blockDim.x + threadIdx.x);
    const int element_i = start_element + tet_id;  // Global element index

    if (element_i >= end_element) return;  // Out of range

    idx_t ev[4] = {0, 0, 0, 0};  // Indices of the vertices of the tetrahedron

    ev[0] = elems.elems_v0[element_i];
    ev[1] = elems.elems_v1[element_i];
    ev[2] = elems.elems_v2[element_i];
    ev[3] = elems.elems_v3[element_i];

    // Read the coordinates of the vertices of the tetrahedron
    // In the physical space
    const FloatType x0_n = FloatType(xyz.x[ev[0]]);
    const FloatType x1_n = FloatType(xyz.x[ev[1]]);
    const FloatType x2_n = FloatType(xyz.x[ev[2]]);
    const FloatType x3_n = FloatType(xyz.x[ev[3]]);

    const FloatType y0_n = FloatType(xyz.y[ev[0]]);
    const FloatType y1_n = FloatType(xyz.y[ev[1]]);
    const FloatType y2_n = FloatType(xyz.y[ev[2]]);
    const FloatType y3_n = FloatType(xyz.y[ev[3]]);

    const FloatType z0_n = FloatType(xyz.z[ev[0]]);
    const FloatType z1_n = FloatType(xyz.z[ev[1]]);
    const FloatType z2_n = FloatType(xyz.z[ev[2]]);
    const FloatType z3_n = FloatType(xyz.z[ev[3]]);

    const FloatType x_min = fast_min(fast_min(x0_n, x1_n), fast_min(x2_n, x3_n));
    const FloatType y_min = fast_min(fast_min(y0_n, y1_n), fast_min(y2_n, y3_n));
    const FloatType z_min = fast_min(fast_min(z0_n, z1_n), fast_min(z2_n, z3_n));

    const FloatType x_max = fast_max(fast_max(x0_n, x1_n), fast_max(x2_n, x3_n));
    const FloatType y_max = fast_max(fast_max(y0_n, y1_n), fast_max(y2_n, y3_n));
    const FloatType z_max = fast_max(fast_max(z0_n, z1_n), fast_max(z2_n, z3_n));

    // Step 2: Convert to grid indices with respect to origin (0,0,0)
    // Using floor for minimum indices (with safety margin of -1)
    const IntType min_grid_x = fast_floor(x_min / FloatType(dx)) - 1;
    const IntType min_grid_y = fast_floor(y_min / FloatType(dy)) - 1;
    const IntType min_grid_z = fast_floor(z_min / FloatType(dz)) - 1;

    // Using ceil for maximum indices (with safety margin of +1)
    const IntType max_grid_x = fast_ceil(x_max / FloatType(dx)) + 1;
    const IntType max_grid_y = fast_ceil(y_max / FloatType(dy)) + 1;
    const IntType max_grid_z = fast_ceil(z_max / FloatType(dz)) + 1;

    // Step 3: Calculate grid dimensions
    const IntType sizen_0 = max_grid_x - min_grid_x + 1;
    const IntType sizen_1 = max_grid_y - min_grid_y + 1;
    const IntType sizen_2 = max_grid_z - min_grid_z + 1;

    // Calculate total number of cells in the bounding box
    const IntType total_size = sizen_0 * sizen_1 * sizen_2;

    tet_properties_info.min_grid_0[element_i] = min_grid_x;
    tet_properties_info.min_grid_1[element_i] = min_grid_y;
    tet_properties_info.min_grid_2[element_i] = min_grid_z;

    tet_properties_info.n0_local[element_i] = sizen_0;
    tet_properties_info.n1_local[element_i] = sizen_1;
    tet_properties_info.n2_local[element_i] = sizen_2;

    tet_properties_info.total_size_local[element_i] = total_size;

    tet_properties_info.stride0_local[element_i] = 1;
    tet_properties_info.stride1_local[element_i] = sizen_0;
    tet_properties_info.stride2_local[element_i] = sizen_0 * sizen_1;

#ifdef COLLECT_ALPHA_DATA

    const FloatType d_min = dx < dy ? (dx < dz ? dx : dz) : (dy < dz ? dy : dz);

    FloatType       edges_length[6];
    int             vertex_a         = -1;
    int             vertex_b         = -1;
    const FloatType max_edges_length =                         //
            tet_edge_max_length_gpu<FloatType>(x0_n,           //
                                               y0_n,           //
                                               z0_n,           //
                                               x1_n,           //
                                               y1_n,           //
                                               z1_n,           //
                                               x2_n,           //
                                               y2_n,           //
                                               z2_n,           //
                                               x3_n,           //
                                               y3_n,           //
                                               z3_n,           //
                                               &vertex_a,      // Output
                                               &vertex_b,      // Output
                                               edges_length);  // Output

    const FloatType alpha_tet                = max_edges_length / d_min;
    tet_properties_info.tet_alpha[element_i] = alpha_tet;

#endif  // COLLECT_ALPHA_DATA
}

/////////////////////////////////////////////////////////////////////////////////
// Kernel to perform adjoint mini-tetrahedron resampling
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__global__ void                                                                                          //
sfem_adjoint_mini_tet_shared_loc_kernel_gpu(const ptrdiff_t                        shared_memory_size,   //
                                            const ptrdiff_t                        tets_per_block,       //
                                            const ptrdiff_t                        start_element,        // Mesh
                                            const ptrdiff_t                        end_element,          //
                                            const ptrdiff_t                        nnodes,               //
                                            const elems_tet4_device                elems,                //
                                            const xyz_tet4_device                  xyz,                  //
                                            const ptrdiff_t                        n0,                   // SDF
                                            const ptrdiff_t                        n1,                   //
                                            const ptrdiff_t                        n2,                   //
                                            const ptrdiff_t                        stride0,              // Stride
                                            const ptrdiff_t                        stride1,              //
                                            const ptrdiff_t                        stride2,              //
                                            const geom_t                           origin0,              // Origin
                                            const geom_t                           origin1,              //
                                            const geom_t                           origin2,              //
                                            const geom_t                           dx,                   // Delta
                                            const geom_t                           dy,                   //
                                            const geom_t                           dz,                   //
                                            const FloatType* const                 weighted_field,       // Input weighted field
                                            const mini_tet_parameters_t            mini_tet_parameters,  // Threshold for alpha
                                            const tet_properties_info_t<FloatType> tet_properties_info,  //
                                            FloatType* const                       data) {                                     //

    const int tet_id    = (blockIdx.x * blockDim.x + threadIdx.x) / LANES_PER_TILE;
    const int element_i = start_element + tet_id;  // Global element index
    const int warp_id   = threadIdx.x / LANES_PER_TILE;
    const int lane_id   = threadIdx.x % LANES_PER_TILE;
    // const int threading_block_size = blockDim.x;

    extern __shared__ FloatType shared_local_data[];  // Shared memory for local accumulation
    // FloatType*                  tet_buffer = &shared_local_data[tets_per_block];

#define START_BUFFER_IDX_SIZE 64
    __shared__ ptrdiff_t tet_start_buffer_idx[START_BUFFER_IDX_SIZE];

    for (int i = threadIdx.x; i < shared_memory_size; i += blockDim.x) {
        if (i < shared_memory_size) shared_local_data[i] = FloatType(0.0);
        // if (i < START_BUFFER_IDX_SIZE) tet_start_buffer_idx[i] = 0;
    }

    // __syncthreads();

    if (lane_id == 0) {
        tet_start_buffer_idx[warp_id] = tet_properties_info.total_size_local[element_i];
    }

    __syncthreads();

    if (lane_id == 0 and warp_id == 0) {
        ptrdiff_t offset = 0;
        for (int i = 0; i < tets_per_block; i++) {
            const ptrdiff_t sz      = tet_start_buffer_idx[i];
            tet_start_buffer_idx[i] = offset;
            offset += sz;
        }
    }

    __syncthreads();

    FloatType* hex_local_buffer = &shared_local_data[tet_start_buffer_idx[warp_id]];

    if (element_i >= end_element) return;  // Out of range

    // printf("Processing element %ld / %ld\n", element_i, end_element);

    const FloatType d_min             = dx < dy ? (dx < dz ? dx : dz) : (dy < dz ? dy : dz);
    const FloatType hexahedron_volume = dx * dy * dz;

    const ptrdiff_t min_grid_0 = tet_properties_info.min_grid_0[element_i];
    const ptrdiff_t min_grid_1 = tet_properties_info.min_grid_1[element_i];
    const ptrdiff_t min_grid_2 = tet_properties_info.min_grid_2[element_i];

    const ptrdiff_t stride0_local = tet_properties_info.stride0_local[element_i];
    const ptrdiff_t stride1_local = tet_properties_info.stride1_local[element_i];
    const ptrdiff_t stride2_local = tet_properties_info.stride2_local[element_i];

    const FloatType min_grid_x_coord = dx * FloatType(min_grid_0);
    const FloatType min_grid_y_coord = dy * FloatType(min_grid_1);
    const FloatType min_grid_z_coord = dz * FloatType(min_grid_2);

    const ptrdiff_t n0_local = tet_properties_info.n0_local[element_i];
    const ptrdiff_t n1_local = tet_properties_info.n1_local[element_i];
    const ptrdiff_t n2_local = tet_properties_info.n2_local[element_i];

    // if (element_i > 30000 and element_i < 30010)  //
    //     printf("Stride local: %ld, %ld, %ld, min_grid_0: %ld, min_grid_1: %ld, min_grid_2: %ld, element_i: %ld \n",
    //            stride0_local,
    //            stride1_local,
    //            stride2_local,
    //            min_grid_0,
    //            min_grid_1,
    //            min_grid_2,
    //            (long)element_i);

    // printf("Exaedre volume: %e\n", hexahedron_volume);

    idx_t ev[4] = {0, 0, 0, 0};  // Indices of the vertices of the tetrahedron

    ev[0] = elems.elems_v0[element_i];
    ev[1] = elems.elems_v1[element_i];
    ev[2] = elems.elems_v2[element_i];
    ev[3] = elems.elems_v3[element_i];

    // Read the coordinates of the vertices of the tetrahedron
    // In the physical space
    // And convert to local grid coordinates
    const FloatType x0_n = FloatType(xyz.x[ev[0]]) - min_grid_x_coord;
    const FloatType x1_n = FloatType(xyz.x[ev[1]]) - min_grid_x_coord;
    const FloatType x2_n = FloatType(xyz.x[ev[2]]) - min_grid_x_coord;
    const FloatType x3_n = FloatType(xyz.x[ev[3]]) - min_grid_x_coord;

    const FloatType y0_n = FloatType(xyz.y[ev[0]]) - min_grid_y_coord;
    const FloatType y1_n = FloatType(xyz.y[ev[1]]) - min_grid_y_coord;
    const FloatType y2_n = FloatType(xyz.y[ev[2]]) - min_grid_y_coord;
    const FloatType y3_n = FloatType(xyz.y[ev[3]]) - min_grid_y_coord;

    const FloatType z0_n = FloatType(xyz.z[ev[0]]) - min_grid_z_coord;
    const FloatType z1_n = FloatType(xyz.z[ev[1]]) - min_grid_z_coord;
    const FloatType z2_n = FloatType(xyz.z[ev[2]]) - min_grid_z_coord;
    const FloatType z3_n = FloatType(xyz.z[ev[3]]) - min_grid_z_coord;

    const FloatType wf0 = weighted_field[ev[0]];  // Weighted field at vertex 0
    const FloatType wf1 = weighted_field[ev[1]];  // Weighted field at vertex 1
    const FloatType wf2 = weighted_field[ev[2]];  // Weighted field at vertex 2
    const FloatType wf3 = weighted_field[ev[3]];  // Weighted field at vertex 3

    FloatType edges_length[6];

    int vertex_a = -1;
    int vertex_b = -1;

    const FloatType max_edges_length =                         //
            tet_edge_max_length_gpu<FloatType>(x0_n,           //
                                               y0_n,           //
                                               z0_n,           //
                                               x1_n,           //
                                               y1_n,           //
                                               z1_n,           //
                                               x2_n,           //
                                               y2_n,           //
                                               z2_n,           //
                                               x3_n,           //
                                               y3_n,           //
                                               z3_n,           //
                                               &vertex_a,      // Output
                                               &vertex_b,      // Output
                                               edges_length);  // Output

    const FloatType alpha_tet = max_edges_length / d_min;

    const int L = alpha_to_hyteg_level_gpu(alpha_tet,                                           //
                                           FloatType(mini_tet_parameters.alpha_min_threshold),  //
                                           FloatType(mini_tet_parameters.alpha_max_threshold),  //
                                           mini_tet_parameters.min_refinement_L,                //
                                           mini_tet_parameters.max_refinement_L);               //

    typename Float3<FloatType>::type Jacobian_phys[3];

    const FloatType det_J_phys = fast_abs(                            //
            make_Jacobian_matrix_tet_gpu<FloatType>(x0_n,             //
                                                    x1_n,             //
                                                    x2_n,             //
                                                    x3_n,             //
                                                    y0_n,             //
                                                    y1_n,             //
                                                    y2_n,             //
                                                    y3_n,             //
                                                    z0_n,             //
                                                    z1_n,             //
                                                    z2_n,             //
                                                    z3_n,             //
                                                    Jacobian_phys));  //

    main_tet_loop_gpu<FloatType>(L,                                          //
                                 Jacobian_phys,                              //
                                 det_J_phys,                                 //
                                 Float3<FloatType>::make(x0_n, y0_n, z0_n),  //
                                 wf0,                                        //
                                 wf1,                                        //
                                 wf2,                                        //
                                 wf3,                                        //
                                 FloatType(0.0),                             //
                                 FloatType(0.0),                             //
                                 FloatType(0.0),                             //
                                 dx,                                         //
                                 dy,                                         //
                                 dz,                                         //
                                 stride0_local,                              //
                                 stride1_local,                              //
                                 stride2_local,                              //
                                 n0_local,                                   //
                                 n1_local,                                   //
                                 n2_local,                                   //
                                 &hex_local_buffer[0],                       //
                                 0);                                         //

    // __syncwarp();

    const ptrdiff_t total_size_local = tet_properties_info.total_size_local[element_i];

    const ptrdiff_t i0_origin = fast_ceil(-origin0 / dx);
    const ptrdiff_t i1_origin = fast_ceil(-origin1 / dy);
    const ptrdiff_t i2_origin = fast_ceil(-origin2 / dz);

    for (ptrdiff_t i_total_local = lane_id; i_total_local < total_size_local; i_total_local += LANES_PER_TILE) {
        const ptrdiff_t i0 = (i_total_local % (n0_local));
        const ptrdiff_t i1 = (i_total_local / (n0_local)) % (n1_local);
        const ptrdiff_t i2 = (i_total_local / (n0_local * n1_local));

        // const int       i_total_local          = i;
        const FloatType hex_local_buffer_value = hex_local_buffer[i_total_local];

        if (hex_local_buffer_value != 0.0) {
            const ptrdiff_t gi0 = i0_origin + i0 + min_grid_0;
            const ptrdiff_t gi1 = i1_origin + i1 + min_grid_1;
            const ptrdiff_t gi2 = i2_origin + i2 + min_grid_2;

            const ptrdiff_t g_index = gi2 * stride2 + gi1 * stride1 + gi0 * stride0;

            // printf("gi0 = %ld, gi1 = %ld, gi2 = %ld, g_index = %ld, value = %e\n",
            //        gi0,
            //        gi1,
            //        gi2,
            //        g_index,
            //        hex_local_buffer_value);

            // if (gi0 >= 0 and gi0 < n0 and  //
            //     gi1 >= 0 and gi1 < n1 and  //
            //     gi2 >= 0 and gi2 < n2 and  //
            //     g_index >= 0 and           //
            //     g_index < n0 * n1 * n2)
            {  //
                // // Atomic add to global memory
                atomicAdd(&data[g_index], hex_local_buffer_value);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////
// Kernel to perform adjoint mini-tetrahedron resampling
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType,                                                                         //
          typename IntType = int>                                                                     //
__global__ void                                                                                       //
sfem_adjoint_mini_tet_buffer_cluster_loc_kernel_gpu(const IntType               buffer_size,          //
                                                    buffer_cluster_t<FloatType> buffer_cluster,       //
                                                    const IntType               tets_per_block,       //
                                                    const IntType               tet_cluster_size,     // Cluster size
                                                    const IntType               start_element,        // Mesh
                                                    const IntType               end_element,          //
                                                    const IntType               nnodes,               //
                                                    const elems_tet4_device     elems,                //
                                                    const xyz_tet4_device       xyz,                  //
                                                    const IntType               n0,                   // SDF
                                                    const IntType               n1,                   //
                                                    const IntType               n2,                   //
                                                    const IntType               stride0,              // Stride
                                                    const IntType               stride1,              //
                                                    const IntType               stride2,              //
                                                    const geom_t                origin0,              // Origin
                                                    const geom_t                origin1,              //
                                                    const geom_t                origin2,              //
                                                    const geom_t                dx,                   // Delta
                                                    const geom_t                dy,                   //
                                                    const geom_t                dz,                   //
                                                    const FloatType* const      weighted_field,       // Input weighted field
                                                    const mini_tet_parameters_t mini_tet_parameters,  // Threshold for alpha
                                                    const tet_properties_info_t<FloatType> tet_properties_info,  //
                                                    FloatType* const                       data) {                                     //

    const int tet_id_base      = blockIdx.x;
    const int warp_id_loc      = threadIdx.y;
    const int lane_id          = threadIdx.x % LANES_PER_TILE;
    const int warps_per_block  = blockDim.y;
    const int warp_id_abs      = blockIdx.x * warps_per_block + warp_id_loc;
    const int cluster_begin    = start_element + warp_id_abs * tet_cluster_size;
    const int thread_index_loc = threadIdx.x + threadIdx.y * LANES_PER_TILE;
    const int block_total_dim  = blockDim.x * blockDim.y;

    // const int cluster_end     = cluster_begin + tet_cluster_size;

    const IntType buffer_begin_idx  = tet_id_base * buffer_size;
    FloatType*    buffer_local_data = &(buffer_cluster.buffer[buffer_begin_idx]);

    // if (lane_id == 0)
    //     printf("Block %d, warp_id %d, lane_id %d, warps_per_block %d, tet_id_base %d, cluster_begin %d, cluster_end %d, "
    //            "tet_cluster_size %d, buffer_begin_idx %ld, buffer_size %ld\n",
    //            blockIdx.x,
    //            warp_id,
    //            lane_id,
    //            warps_per_block,
    //            tet_id_base,
    //            cluster_begin,
    //            cluster_end,
    //            (int)tet_cluster_size,
    //            (long)buffer_begin_idx,
    //            (long)buffer_size);

#define START_BUFFER_IDX_SIZE 64

    __shared__ IntType tet_start_buffer_idx[START_BUFFER_IDX_SIZE];

    for (int cluster_i = 0; cluster_i < tet_cluster_size; cluster_i++) {
        // Loop over elements in the cluster

        const int element_i = cluster_i + cluster_begin;  // Global element index

        for (int i = thread_index_loc; i < buffer_size; i += block_total_dim) {
            //
            // if (i + tet_id_base * buffer_size > buffer_cluster.size) {
            //     printf("Err: overflow i = %d, tet_id_base = %d, "
            //            "buffer_size = %ld, buffer_cluster.size = %ld, total_index = %ld, diff = %ld\n",
            //            i,
            //            tet_id_base,
            //            buffer_size,
            //            buffer_cluster.size,
            //            i + tet_id_base * buffer_size,
            //            (i + tet_id_base * buffer_size) - buffer_cluster.size);
            //     return;
            // }

            if (i < buffer_size) buffer_local_data[i] = FloatType(0.0);
            // if (i < START_BUFFER_IDX_SIZE) tet_start_buffer_idx[i] = 0;
        }

        // __syncthreads();

        if (element_i < end_element) {
            if (lane_id == 0) tet_start_buffer_idx[warp_id_loc] = tet_properties_info.total_size_local[element_i];
        } else {
            if (lane_id == 0) tet_start_buffer_idx[warp_id_loc] = 0;
        }

        __syncthreads();

        if (lane_id == 0 and warp_id_loc == 0) {
            IntType offset = 0;
            for (int i = 0; i < tets_per_block; i++) {
                const IntType sz        = tet_start_buffer_idx[i];
                tet_start_buffer_idx[i] = offset;
                offset += sz;
            }
        }

        __syncthreads();

        if (element_i >= end_element) {
            // Note: in this if-scope the number of __syncthreads() calls is important to maintain
            // the synchronization of the warps in the threading block.
            // !!!! It must be equal to the number of __syncthreads() calls following this if-scope.
            __syncthreads();
            __syncthreads();
            continue;  // Out of range /////////////////////////////////////
        }

        FloatType* hex_local_buffer = &buffer_local_data[tet_start_buffer_idx[warp_id_loc]];

        // return;

        // printf("Processing element %ld in range %ld - %ld\n", element_i, start_element, end_element);

        const FloatType d_min             = dx < dy ? (dx < dz ? dx : dz) : (dy < dz ? dy : dz);
        const FloatType hexahedron_volume = dx * dy * dz;

        const IntType min_grid_0 = tet_properties_info.min_grid_0[element_i];
        const IntType min_grid_1 = tet_properties_info.min_grid_1[element_i];
        const IntType min_grid_2 = tet_properties_info.min_grid_2[element_i];

        const IntType stride0_local = tet_properties_info.stride0_local[element_i];
        const IntType stride1_local = tet_properties_info.stride1_local[element_i];
        const IntType stride2_local = tet_properties_info.stride2_local[element_i];

        const FloatType min_grid_x_coord = dx * FloatType(min_grid_0);
        const FloatType min_grid_y_coord = dy * FloatType(min_grid_1);
        const FloatType min_grid_z_coord = dz * FloatType(min_grid_2);

        const IntType n0_local = tet_properties_info.n0_local[element_i];
        const IntType n1_local = tet_properties_info.n1_local[element_i];
        const IntType n2_local = tet_properties_info.n2_local[element_i];

        // if (element_i > 30000 and element_i < 30010)  //
        //     printf("Stride local: %ld, %ld, %ld, min_grid_0: %ld, min_grid_1: %ld, min_grid_2: %ld, element_i: %ld \n",
        //            stride0_local,
        //            stride1_local,
        //            stride2_local,
        //            min_grid_0,
        //            min_grid_1,
        //            min_grid_2,
        //            element_i);

        idx_t ev[4] = {0, 0, 0, 0};  // Indices of the vertices of the tetrahedron

        ev[0] = elems.elems_v0[element_i];
        ev[1] = elems.elems_v1[element_i];
        ev[2] = elems.elems_v2[element_i];
        ev[3] = elems.elems_v3[element_i];

        // Read the coordinates of the vertices of the tetrahedron
        // In the physical space
        // And convert to local grid coordinates
        const FloatType x0_n = FloatType(xyz.x[ev[0]]) - min_grid_x_coord;
        const FloatType x1_n = FloatType(xyz.x[ev[1]]) - min_grid_x_coord;
        const FloatType x2_n = FloatType(xyz.x[ev[2]]) - min_grid_x_coord;
        const FloatType x3_n = FloatType(xyz.x[ev[3]]) - min_grid_x_coord;

        const FloatType y0_n = FloatType(xyz.y[ev[0]]) - min_grid_y_coord;
        const FloatType y1_n = FloatType(xyz.y[ev[1]]) - min_grid_y_coord;
        const FloatType y2_n = FloatType(xyz.y[ev[2]]) - min_grid_y_coord;
        const FloatType y3_n = FloatType(xyz.y[ev[3]]) - min_grid_y_coord;

        const FloatType z0_n = FloatType(xyz.z[ev[0]]) - min_grid_z_coord;
        const FloatType z1_n = FloatType(xyz.z[ev[1]]) - min_grid_z_coord;
        const FloatType z2_n = FloatType(xyz.z[ev[2]]) - min_grid_z_coord;
        const FloatType z3_n = FloatType(xyz.z[ev[3]]) - min_grid_z_coord;

        const FloatType wf0 = weighted_field[ev[0]];  // Weighted field at vertex 0
        const FloatType wf1 = weighted_field[ev[1]];  // Weighted field at vertex 1
        const FloatType wf2 = weighted_field[ev[2]];  // Weighted field at vertex 2
        const FloatType wf3 = weighted_field[ev[3]];  // Weighted field at vertex 3

        FloatType edges_length[6];

        int vertex_a = -1;
        int vertex_b = -1;

        const FloatType max_edges_length =                         //
                tet_edge_max_length_gpu<FloatType>(x0_n,           //
                                                   y0_n,           //
                                                   z0_n,           //
                                                   x1_n,           //
                                                   y1_n,           //
                                                   z1_n,           //
                                                   x2_n,           //
                                                   y2_n,           //
                                                   z2_n,           //
                                                   x3_n,           //
                                                   y3_n,           //
                                                   z3_n,           //
                                                   &vertex_a,      // Output
                                                   &vertex_b,      // Output
                                                   edges_length);  // Output

        const FloatType alpha_tet = max_edges_length / d_min;

        const int L = alpha_to_hyteg_level_gpu(alpha_tet,                                           //
                                               FloatType(mini_tet_parameters.alpha_min_threshold),  //
                                               FloatType(mini_tet_parameters.alpha_max_threshold),  //
                                               mini_tet_parameters.min_refinement_L,                //
                                               mini_tet_parameters.max_refinement_L);               //

        typename Float3<FloatType>::type Jacobian_phys[3];

        const FloatType det_J_phys = fast_abs(                            //
                make_Jacobian_matrix_tet_gpu<FloatType>(x0_n,             //
                                                        x1_n,             //
                                                        x2_n,             //
                                                        x3_n,             //
                                                        y0_n,             //
                                                        y1_n,             //
                                                        y2_n,             //
                                                        y3_n,             //
                                                        z0_n,             //
                                                        z1_n,             //
                                                        z2_n,             //
                                                        z3_n,             //
                                                        Jacobian_phys));  //

        main_tet_loop_gpu<FloatType>(L,                                          //
                                     Jacobian_phys,                              //
                                     det_J_phys,                                 //
                                     Float3<FloatType>::make(x0_n, y0_n, z0_n),  //
                                     wf0,                                        //
                                     wf1,                                        //
                                     wf2,                                        //
                                     wf3,                                        //
                                     FloatType(0.0),                             //
                                     FloatType(0.0),                             //
                                     FloatType(0.0),                             //
                                     dx,                                         //
                                     dy,                                         //
                                     dz,                                         //
                                     stride0_local,                              //
                                     stride1_local,                              //
                                     stride2_local,                              //
                                     n0_local,                                   //
                                     n1_local,                                   //
                                     n2_local,                                   //
                                     &hex_local_buffer[0],                       //
                                     0);                                         //

        __syncthreads();

        const IntType total_size_local = tet_properties_info.total_size_local[element_i];

        const IntType i0_origin = fast_ceil(-origin0 / dx);
        const IntType i1_origin = fast_ceil(-origin1 / dy);
        const IntType i2_origin = fast_ceil(-origin2 / dz);

        for (IntType i_total_local = lane_id; i_total_local < total_size_local; i_total_local += LANES_PER_TILE) {
            const IntType i0 = (i_total_local % (n0_local));
            const IntType i1 = (i_total_local / (n0_local)) % (n1_local);
            const IntType i2 = (i_total_local / (n0_local * n1_local));

            // const int       i_total_local          = i;
            const FloatType hex_local_buffer_value = hex_local_buffer[i_total_local];

            if (hex_local_buffer_value != 0.0) {
                const IntType gi0 = i0_origin + i0 + min_grid_0;
                const IntType gi1 = i1_origin + i1 + min_grid_1;
                const IntType gi2 = i2_origin + i2 + min_grid_2;

                const IntType g_index = gi2 * stride2 + gi1 * stride1 + gi0 * stride0;

                // printf("gi0 = %ld, gi1 = %ld, gi2 = %ld, g_index = %ld, value = %e, i0 = %ld, i1 = %ld, i2 = %ld,
                // i0_origin = "
                //        "%ld, i1_origin = %ld, i2_origin = %ld, min_grid_0 = %ld, min_grid_1 = %ld, min_grid_2 = %ld\n",
                //        gi0,
                //        gi1,
                //        gi2,
                //        g_index,
                //        hex_local_buffer_value,
                //        i0,
                //        i1,
                //        i2,
                //        i0_origin,
                //        i1_origin,
                //        i2_origin,
                //        min_grid_0,
                //        min_grid_1,
                //        min_grid_2);

                // if (gi0 >= 0 and gi0 < n0 and  //
                //     gi1 >= 0 and gi1 < n1 and  //
                //     gi2 >= 0 and gi2 < n2 and  //
                //     g_index >= 0 and           //
                //     g_index < n0 * n1 * n2)
                {
                    // // Atomic add to global memory
                    atomicAdd(&data[g_index], hex_local_buffer_value);
                    // data[g_index] += hex_local_buffer_value;
                }
            }
        }

        __syncthreads();

    }  // End loop over elements in the cluster
}  // End kernel: sfem_adjoint_mini_tet_buffer_cluster_loc_kernel_gpu
/////////////////////////////////////////////////////////////////////////////////

#endif  // __SFEM_ADJOINT_MINI_LOC_TET_CUH__
