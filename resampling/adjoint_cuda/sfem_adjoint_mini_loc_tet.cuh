

#ifndef __SFEM_ADJOINT_MINI_LOC_TET_CUH__
#define __SFEM_ADJOINT_MINI_LOC_TET_CUH__

#include "sfem_adjoint_mini_tet.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

template <typename FloatType>
class tet_properties_info_t {
public:
    // Arrays allocated on device (length = count)
    ptrdiff_t* delta0_index = nullptr;  // delta with respect to the global grid in the global grid
    ptrdiff_t* delta1_index = nullptr;
    ptrdiff_t* delta2_index = nullptr;

    // ptrdiff_t* size0_local;
    // ptrdiff_t* size1_local;
    // ptrdiff_t* size2_local;

    ptrdiff_t* total_size_local;

    ptrdiff_t* stride0_local = nullptr;
    ptrdiff_t* stride1_local = nullptr;
    ptrdiff_t* stride2_local = nullptr;

    ptrdiff_t* n0_local = nullptr;
    ptrdiff_t* n1_local = nullptr;
    ptrdiff_t* n2_local = nullptr;

    FloatType* min_x = nullptr;
    FloatType* min_y = nullptr;
    FloatType* min_z = nullptr;

    FloatType* max_x = nullptr;
    FloatType* max_y = nullptr;
    FloatType* max_z = nullptr;

    FloatType* min_grid_x = nullptr;  // Minimum grid coordinates covered by the tet
    FloatType* min_grid_y = nullptr;
    FloatType* min_grid_z = nullptr;

    // Host-side meta
    size_t count = 0;

    __device__ size_t get_tet_grid_size(const ptrdiff_t tet_i) const {
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
        if ((err = cudaMallocAsync((void**)&delta0_index, n * sizeof(ptrdiff_t), stream)) != cudaSuccess)
            return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&delta1_index, n * sizeof(ptrdiff_t), stream)) != cudaSuccess)
            return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&delta2_index, n * sizeof(ptrdiff_t), stream)) != cudaSuccess)
            return fail_cleanup(err);

        // if ((err = cudaMallocAsync((void**)&size0_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess) return
        // fail_cleanup(err); if ((err = cudaMallocAsync((void**)&size1_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess)
        // return fail_cleanup(err); if ((err = cudaMallocAsync((void**)&size2_local, n * sizeof(ptrdiff_t), stream)) !=
        // cudaSuccess) return fail_cleanup(err);

        if ((err = cudaMallocAsync((void**)&total_size_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess)
            return fail_cleanup(err);

        if ((err = cudaMallocAsync((void**)&stride0_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess)
            return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&stride1_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess)
            return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&stride2_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess)
            return fail_cleanup(err);

        if ((err = cudaMallocAsync((void**)&n0_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&n1_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&n2_local, n * sizeof(ptrdiff_t), stream)) != cudaSuccess) return fail_cleanup(err);

        // FloatType arrays
        if ((err = cudaMallocAsync((void**)&min_x, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&min_y, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&min_z, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);

        if ((err = cudaMallocAsync((void**)&max_x, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&max_y, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&max_z, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);

        if ((err = cudaMallocAsync((void**)&min_grid_x, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&min_grid_y, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);
        if ((err = cudaMallocAsync((void**)&min_grid_z, n * sizeof(FloatType), stream)) != cudaSuccess) return fail_cleanup(err);

        return cudaSuccess;
    }

    __host__ cudaError_t free_async(cudaStream_t stream) {
        // Free if non-null; ignore errors to attempt freeing all
        auto free_if = [&](void* p) {
            if (p) (void)cudaFreeAsync(p, stream);
        };

        free_if(delta0_index);
        free_if(delta1_index);
        free_if(delta2_index);

        // free_if(size0_local);
        // free_if(size1_local);
        // free_if(size2_local);

        free_if(total_size_local);

        free_if(stride0_local);
        free_if(stride1_local);
        free_if(stride2_local);

        free_if(n0_local);
        free_if(n1_local);
        free_if(n2_local);

        free_if(min_x);
        free_if(min_y);
        free_if(min_z);

        free_if(max_x);
        free_if(max_y);
        free_if(max_z);

        free_if(min_grid_x);
        free_if(min_grid_y);
        free_if(min_grid_z);

        reset();
        // Optionally, user can cudaStreamSynchronize(stream) after this
        return cudaSuccess;
    }

private:
    __host__ void reset() {
        delta0_index = delta1_index = delta2_index = nullptr;
        // size0_local = size1_local = size2_local = nullptr;
        total_size_local = nullptr;
        stride0_local = stride1_local = stride2_local = nullptr;
        n0_local = n1_local = n2_local = nullptr;
        min_x = min_y = min_z = nullptr;
        max_x = max_y = max_z = nullptr;
        min_grid_x = min_grid_y = min_grid_z = nullptr;
        count                                = 0;
    }
};

/////////////////////////////////////////////////////////////////////////////////
// Kernel to perform adjoint mini-tetrahedron resampling
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__global__ void                                                                               //
sfem_make_local_data_tets_kernel_gpu(const ptrdiff_t                  start_element,          // Mesh
                                     const ptrdiff_t                  end_element,            //
                                     const ptrdiff_t                  nnodes,                 //
                                     const elems_tet4_device          elems,                  //
                                     const xyz_tet4_device            xyz,                    //
                                     const ptrdiff_t                  n0,                     // SDF
                                     const ptrdiff_t                  n1,                     //
                                     const ptrdiff_t                  n2,                     //
                                     const ptrdiff_t                  stride0,                // Stride
                                     const ptrdiff_t                  stride1,                //
                                     const ptrdiff_t                  stride2,                //
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

    const ptrdiff_t min_grid_x = fast_floor((x_min - FloatType(origin0)) / FloatType(dx)) - 1;
    const ptrdiff_t min_grid_y = fast_floor((y_min - FloatType(origin1)) / FloatType(dy)) - 1;
    const ptrdiff_t min_grid_z = fast_floor((z_min - FloatType(origin2)) / FloatType(dz)) - 1;

    const ptrdiff_t max_grid_x = fast_ceil((x_max - FloatType(origin0)) / FloatType(dx)) + 1;
    const ptrdiff_t max_grid_y = fast_ceil((y_max - FloatType(origin1)) / FloatType(dy)) + 1;
    const ptrdiff_t max_grid_z = fast_ceil((z_max - FloatType(origin2)) / FloatType(dz)) + 1;

    const ptrdiff_t sizen_0 = max_grid_x - min_grid_x + 1;
    const ptrdiff_t sizen_1 = max_grid_y - min_grid_y + 1;
    const ptrdiff_t sizen_2 = max_grid_z - min_grid_z + 1;

    const ptrdiff_t total_size = sizen_0 * sizen_1 * sizen_2;

    // if (total_size <= 8) {
    //     const double dx_dbl = double(dx);
    //     const double dy_dbl = double(dy);
    //     const double dz_dbl = double(dz);

    //     printf("Warning: Element %d has local grid size %ld x %ld x %ld = %ld, with dx=%e, dy=%e, dz=%e, min=(%e,%e,%e), "
    //            "max=(%e,%e,%e)\n",
    //            element_i,
    //            (long)sizen_0,
    //            (long)sizen_1,
    //            (long)sizen_2,
    //            (long)total_size,
    //            (double)dx_dbl,
    //            (double)dy_dbl,
    //            (double)dz_dbl,
    //            (double)x_min,
    //            (double)y_min,
    //            (double)z_min,
    //            (double)x_max,
    //            (double)y_max,
    //            (double)z_max);
    // }

    tet_properties_info.delta0_index[element_i] = min_grid_x;
    tet_properties_info.delta1_index[element_i] = min_grid_y;
    tet_properties_info.delta2_index[element_i] = min_grid_z;

    tet_properties_info.n0_local[element_i] = sizen_0;
    tet_properties_info.n1_local[element_i] = sizen_1;
    tet_properties_info.n2_local[element_i] = sizen_2;

    tet_properties_info.total_size_local[element_i] = total_size;

    tet_properties_info.stride0_local[element_i] = 1;
    tet_properties_info.stride1_local[element_i] = sizen_0;
    tet_properties_info.stride2_local[element_i] = sizen_0 * sizen_1;

    tet_properties_info.min_x[element_i] = x_min;
    tet_properties_info.min_y[element_i] = y_min;
    tet_properties_info.min_z[element_i] = z_min;

    tet_properties_info.max_x[element_i] = x_max;
    tet_properties_info.max_y[element_i] = y_max;
    tet_properties_info.max_z[element_i] = z_max;

    tet_properties_info.min_grid_x[element_i] = FloatType(min_grid_x);
    tet_properties_info.min_grid_y[element_i] = FloatType(min_grid_y);
    tet_properties_info.min_grid_z[element_i] = FloatType(min_grid_z);
}

/////////////////////////////////////////////////////////////////////////////////
// Kernel to perform adjoint mini-tetrahedron resampling
/////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__global__ void                                                                               //
sfem_adjoint_mini_tet_shared_loc_kernel_gpu(const ptrdiff_t             shared_memory_size,   //
                                            const ptrdiff_t             start_element,        // Mesh
                                            const ptrdiff_t             end_element,          //
                                            const ptrdiff_t             nnodes,               //
                                            const elems_tet4_device     elems,                //
                                            const xyz_tet4_device       xyz,                  //
                                            const ptrdiff_t             n0,                   // SDF
                                            const ptrdiff_t             n1,                   //
                                            const ptrdiff_t             n2,                   //
                                            const ptrdiff_t             stride0,              // Stride
                                            const ptrdiff_t             stride1,              //
                                            const ptrdiff_t             stride2,              //
                                            const geom_t                origin0,              // Origin
                                            const geom_t                origin1,              //
                                            const geom_t                origin2,              //
                                            const geom_t                dx,                   // Delta
                                            const geom_t                dy,                   //
                                            const geom_t                dz,                   //
                                            const FloatType* const      weighted_field,       // Input weighted field
                                            const mini_tet_parameters_t mini_tet_parameters,  // Threshold for alpha
                                            FloatType* const            data) {                          //

    const int tet_id               = (blockIdx.x * blockDim.x + threadIdx.x) / LANES_PER_TILE;
    const int element_i            = start_element + tet_id;  // Global element index
    const int threading_block_size = blockDim.x;

    extern __shared__ FloatType shared_local_data[];  // Shared memory for local accumulation

    for (int i = threadIdx.x; i < threading_block_size; i += blockDim.x) {
        if (i < threading_block_size) shared_local_data[i] = 0.0;
    }
    __syncthreads();

    if (element_i >= end_element) return;  // Out of range

    // printf("Processing element %ld / %ld\n", element_i, end_element);

    const FloatType d_min             = dx < dy ? (dx < dz ? dx : dz) : (dy < dz ? dy : dz);
    const FloatType hexahedron_volume = dx * dy * dz;

    // printf("Exaedre volume: %e\n", hexahedron_volume);

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

    const FloatType wf0 = weighted_field[ev[0]];  // Weighted field at vertex 0
    const FloatType wf1 = weighted_field[ev[1]];  // Weighted field at vertex 1
    const FloatType wf2 = weighted_field[ev[2]];  // Weighted field at vertex 2
    const FloatType wf3 = weighted_field[ev[3]];  // Weighted field at vertex 3

    FloatType edges_length[6];

    int vertex_a = -1;
    int vertex_b = -1;

    const FloatType max_edges_length =              //
            tet_edge_max_length_gpu(x0_n,           //
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

    const FloatType det_J_phys =                               //
            abs(make_Jacobian_matrix_tet_gpu<FloatType>(x0_n,  //
                                                        x1_n,  //
                                                        x2_n,
                                                        x3_n,
                                                        y0_n,
                                                        y1_n,
                                                        y2_n,
                                                        y3_n,
                                                        z0_n,
                                                        z1_n,
                                                        z2_n,
                                                        z3_n,
                                                        Jacobian_phys));

    main_tet_loop_gpu<FloatType>(L,                                          //
                                 Jacobian_phys,                              //
                                 det_J_phys,                                 //
                                 Float3<FloatType>::make(x0_n, y0_n, z0_n),  //
                                 wf0,                                        //
                                 wf1,                                        //
                                 wf2,                                        //
                                 wf3,                                        //
                                 origin0,                                    //
                                 origin1,                                    //
                                 origin2,                                    //
                                 dx,                                         //
                                 dy,                                         //
                                 dz,                                         //
                                 stride0,                                    //
                                 stride1,                                    //
                                 stride2,                                    //
                                 n0,                                         //
                                 n1,                                         //
                                 n2,                                         //
                                 data);                                      //
}
/////////////////////////////////////////////////////////////////////////////////

#endif  // __SFEM_ADJOINT_MINI_LOC_TET_CUH__
