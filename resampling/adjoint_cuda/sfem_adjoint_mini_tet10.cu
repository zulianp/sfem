#include <stdio.h>
#include <stdlib.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include "sfem_adjoint_mini_loc_tet.cuh"
#include "sfem_adjoint_mini_loc_tet10.cuh"
#include "sfem_adjoint_mini_tet.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

void  //                                                                                               //
call_hex8_to_isoparametric_tet10_resample_field_hyteg_mt_adjoint_kernel(  //
        const ptrdiff_t      start_element,                               // Mesh
        const ptrdiff_t      end_element,                                 //
        const ptrdiff_t      nelements,                                   //
        const ptrdiff_t      nnodes,                                      //
        const idx_t** const  elems,                                       //
        const geom_t** const xyz,                                         //
        const ptrdiff_t      n0,                                          // SDF
        const ptrdiff_t      n1,                                          //
        const ptrdiff_t      n2,                                          //
        const ptrdiff_t      stride0,                                     //
        const ptrdiff_t      stride1,                                     //
        const ptrdiff_t      stride2,                                     //
        const geom_t         ox,                                          //
        const geom_t         oy,                                          //
        const geom_t         oz,                                          //
        const geom_t         dx,                                          //
        const geom_t         dy,                                          //
        const geom_t         dz,                                          //
        const real_t* const __restrict__ weighted_field,                  // Input WF
        real_t* const __restrict__ data,                                  // Output
        const mini_tet_parameters_t mini_tet_parameters) {                //

    cudaStream_t cuda_stream_alloc = NULL;  // default stream
    cudaStreamCreate(&cuda_stream_alloc);

    real_t* data_device           = NULL;
    real_t* weighted_field_device = NULL;

    cudaMallocAsync((void**)&data_device, (n0 * n1 * n2) * sizeof(real_t), cuda_stream_alloc);
    cudaMallocAsync((void**)&weighted_field_device, nnodes * sizeof(real_t), cuda_stream_alloc);

    elems_tet10_device elems_d = make_elems_tet10_device_async(nelements, cuda_stream_alloc);
    xyz_tet10_device   xyz_d   = make_xyz_tet10_device_async(nnodes, cuda_stream_alloc);

    cudaStreamSynchronize(cuda_stream_alloc);

    cudaMemcpyAsync(weighted_field_device,    //
                    weighted_field,           //
                    nnodes * sizeof(real_t),  //
                    cudaMemcpyHostToDevice,   //
                    cuda_stream_alloc);       //

    cudaMemset((void*)data_device, 0, (n0 * n1 * n2) * sizeof(real_t));

    copy_elems_tet10_device_async(nelements,           //
                                  &elems_d,            //
                                  elems,               //
                                  cuda_stream_alloc);  //
}
