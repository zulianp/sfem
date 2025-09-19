#include <sfem_base.h>
#include <stdio.h>

// #define real_type real_t

#include "quadratures_rule_cuda.cuh"
#include "tet10_resample_field.cuh"

//////////////////////////////////////////////////////////
// make_xyz_tet10_device
//////////////////////////////////////////////////////////
xyz_tet10_device make_xyz_tet10_device(const ptrdiff_t nnodes) {  //
    //
    xyz_tet10_device xyz;

    cudaMalloc(&xyz.x, nnodes * sizeof(geom_t));
    cudaMalloc(&xyz.y, nnodes * sizeof(geom_t));
    cudaMalloc(&xyz.z, nnodes * sizeof(geom_t));
    return xyz;
}
// end make_xyz_tet10_device

xyz_tet10_device                                                            //
make_xyz_tet10_device_async(const ptrdiff_t nnodes, cudaStream_t stream) {  //
    //
    xyz_tet10_device xyz;

    cudaMallocAsync(&xyz.x, nnodes * sizeof(geom_t), stream);
    cudaMallocAsync(&xyz.y, nnodes * sizeof(geom_t), stream);
    cudaMallocAsync(&xyz.z, nnodes * sizeof(geom_t), stream);
    return xyz;
}  // end make_xyz_tet10_device_async

//////////////////////////////////////////////////////////
// copy_xyz_tet10_device
//////////////////////////////////////////////////////////
void copy_xyz_tet10_device(const ptrdiff_t   nnodes,   //
                           xyz_tet10_device* xyz,      //
                           const geom_t**    xyz_host) {  //

    cudaError_t err0 = cudaMemcpy(xyz->x, xyz_host[0], nnodes * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err1 = cudaMemcpy(xyz->y, xyz_host[1], nnodes * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err2 = cudaMemcpy(xyz->z, xyz_host[2], nnodes * sizeof(idx_t), cudaMemcpyHostToDevice);

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess) {
        printf("ERROR: copying xyz_tet10_device to device: %s\n", cudaGetErrorString(err0));
        // Handle the error or exit the program
    }
}  // end copy_xyz_tet10_device

//////////////////////////////////////////////////////////
// free_xyz_tet10_device
//////////////////////////////////////////////////////////
void free_xyz_tet10_device(xyz_tet10_device* xyz) {
    cudaFree(xyz->x);
    cudaFree(xyz->y);
    cudaFree(xyz->z);

    xyz->x = NULL;
    xyz->y = NULL;
    xyz->z = NULL;
}
// end free_xyz_tet10_device

//////////////////////////////////////////////////////////
// make_elems_tet10_device
//////////////////////////////////////////////////////////
elems_tet10_device                                     //
make_elems_tet10_managed(const ptrdiff_t nelements) {  //
    //
    elems_tet10_device elems;

    cudaError_t err0 = cudaMallocManaged(&elems.elems_v0, nelements * sizeof(idx_t));
    cudaError_t err1 = cudaMallocManaged(&elems.elems_v1, nelements * sizeof(idx_t));
    cudaError_t err2 = cudaMallocManaged(&elems.elems_v2, nelements * sizeof(idx_t));
    cudaError_t err3 = cudaMallocManaged(&elems.elems_v3, nelements * sizeof(idx_t));
    cudaError_t err4 = cudaMallocManaged(&elems.elems_v4, nelements * sizeof(idx_t));
    cudaError_t err5 = cudaMallocManaged(&elems.elems_v5, nelements * sizeof(idx_t));
    cudaError_t err6 = cudaMallocManaged(&elems.elems_v6, nelements * sizeof(idx_t));
    cudaError_t err7 = cudaMallocManaged(&elems.elems_v7, nelements * sizeof(idx_t));
    cudaError_t err8 = cudaMallocManaged(&elems.elems_v8, nelements * sizeof(idx_t));
    cudaError_t err9 = cudaMallocManaged(&elems.elems_v9, nelements * sizeof(idx_t));

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess ||
        err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess || err8 != cudaSuccess || err9 != cudaSuccess) {
        printf("ERROR: allocating memory for elems_tet10_device managed at %s:%d\n", __FILE__, __LINE__);
        // Handle error
    }

    return elems;
}

cudaError_t                                              //
copy_elems_tet10_managed(const ptrdiff_t     nelements,  //
                         elems_tet10_device* elems,      //
                         const idx_t**       elems_host) {     //
                                                         //
    cudaError_t err0 = cudaMemcpy(elems->elems_v0, elems_host[0], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err1 = cudaMemcpy(elems->elems_v1, elems_host[1], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err2 = cudaMemcpy(elems->elems_v2, elems_host[2], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err3 = cudaMemcpy(elems->elems_v3, elems_host[3], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err4 = cudaMemcpy(elems->elems_v4, elems_host[4], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err5 = cudaMemcpy(elems->elems_v5, elems_host[5], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err6 = cudaMemcpy(elems->elems_v6, elems_host[6], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err7 = cudaMemcpy(elems->elems_v7, elems_host[7], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err8 = cudaMemcpy(elems->elems_v8, elems_host[8], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err9 = cudaMemcpy(elems->elems_v9, elems_host[9], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess ||
        err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess || err8 != cudaSuccess || err9 != cudaSuccess) {
        printf("ERROR: copying elements to device: %s\n", cudaGetErrorString(cudaGetLastError()));
        return cudaGetLastError();
    }
}

void                                                   //
free_elems_tet10_managed(elems_tet10_device* elems) {  //

    cudaError_t err0 = cudaFree(elems->elems_v0);
    cudaError_t err1 = cudaFree(elems->elems_v1);
    cudaError_t err2 = cudaFree(elems->elems_v2);
    cudaError_t err3 = cudaFree(elems->elems_v3);
    cudaError_t err4 = cudaFree(elems->elems_v4);
    cudaError_t err5 = cudaFree(elems->elems_v5);
    cudaError_t err6 = cudaFree(elems->elems_v6);
    cudaError_t err7 = cudaFree(elems->elems_v7);
    cudaError_t err8 = cudaFree(elems->elems_v8);
    cudaError_t err9 = cudaFree(elems->elems_v9);

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess ||
        err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess || err8 != cudaSuccess || err9 != cudaSuccess) {
        printf("ERROR: freeing device memory for elems: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    elems->elems_v0 = NULL;
    elems->elems_v1 = NULL;
    elems->elems_v2 = NULL;
    elems->elems_v3 = NULL;
    elems->elems_v4 = NULL;
    elems->elems_v5 = NULL;
    elems->elems_v6 = NULL;
    elems->elems_v7 = NULL;
    elems->elems_v8 = NULL;
    elems->elems_v9 = NULL;
}

//////////////////////////////////////////////////////////
// make_xyz_tet10_device_unified
//////////////////////////////////////////////////////////
xyz_tet10_device                                         //
make_xyz_tet10_device_unified(const ptrdiff_t nnodes) {  //
                                                         //
    xyz_tet10_device xyz;

    xyz.x = NULL;
    xyz.y = NULL;
    xyz.z = NULL;

    return xyz;
}
// end make_xyz_tet10_device_unified

//////////////////////////////////////////////////////////
// copy_xyz_tet10_device_unified
//////////////////////////////////////////////////////////
void                                                      //
copy_xyz_tet10_device_unified(const ptrdiff_t   nnodes,   //
                              xyz_tet10_device* xyz,      //
                              const geom_t**    xyz_host) {  //
                                                          //
    xyz->x = (geom_t*)xyz_host[0];
    xyz->y = (geom_t*)xyz_host[1];
    xyz->z = (geom_t*)xyz_host[2];
}
// end copy_xyz_tet10_device_unified

//////////////////////////////////////////////////////////
// memory_hint_xyz_tet10_device_unified
//////////////////////////////////////////////////////////
void                                                                                   //
memory_hint_xyz_tet10_device_unified(const ptrdiff_t nnodes, xyz_tet10_device* xyz) {  //
                                                                                       //
    cudaError_t err0 = cudaMemAdvise(xyz->x, nnodes * sizeof(geom_t), cudaMemAdviseSetReadMostly, 0);
    cudaError_t err1 = cudaMemAdvise(xyz->y, nnodes * sizeof(geom_t), cudaMemAdviseSetReadMostly, 0);
    cudaError_t err2 = cudaMemAdvise(xyz->z, nnodes * sizeof(geom_t), cudaMemAdviseSetReadMostly, 0);

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess) {
        printf("ERROR: setting memory hint for xyz_tet10_device: %s at %s:%d\n", cudaGetErrorString(err0), __FILE__, __LINE__);
        // Handle the error or exit the program
    }

    // prefetch the data to the GPU
    cudaError_t err3 = cudaMemPrefetchAsync(xyz->x, nnodes * sizeof(geom_t), 0, 0);
    cudaError_t err4 = cudaMemPrefetchAsync(xyz->y, nnodes * sizeof(geom_t), 0, 0);
    cudaError_t err5 = cudaMemPrefetchAsync(xyz->z, nnodes * sizeof(geom_t), 0, 0);

    if (err3 != cudaSuccess || err4 != cudaSuccess || err5 != cudaSuccess) {
        printf("ERROR: prefetching data for xyz_tet10_device: %s at %s:%d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
        // Handle the error or exit the program
    }
}

//////////////////////////////////////////////////////////
// free_xyz_tet10_device_unified
//////////////////////////////////////////////////////////
void                                                    //
free_xyz_tet10_device_unified(xyz_tet10_device* xyz) {  //
                                                        //
    xyz->x = NULL;
    xyz->y = NULL;
    xyz->z = NULL;
}
// end free_xyz_tet10_device_unified

//////////////////////////////////////////////////////////
// make_xyz_tet10_managed
//////////////////////////////////////////////////////////
xyz_tet10_device make_xyz_tet10_managed(const ptrdiff_t nnodes) {
    xyz_tet10_device xyz;
    cudaMallocManaged(&xyz.x, nnodes * sizeof(geom_t));
    cudaMallocManaged(&xyz.y, nnodes * sizeof(geom_t));
    cudaMallocManaged(&xyz.z, nnodes * sizeof(geom_t));
    return xyz;
}
// end make_xyz_tet10_managed

//////////////////////////////////////////////////////////
// copy_xyz_tet10_managed
//////////////////////////////////////////////////////////
void                                               //
copy_xyz_tet10_managed(const ptrdiff_t   nnodes,   //
                       xyz_tet10_device* xyz,      //
                       const geom_t**    xyz_host) {  //

    cudaError_t err0 = cudaMemcpy(xyz->x, xyz_host[0], nnodes * sizeof(geom_t), cudaMemcpyHostToDevice);
    cudaError_t err1 = cudaMemcpy(xyz->y, xyz_host[1], nnodes * sizeof(geom_t), cudaMemcpyHostToDevice);
    cudaError_t err2 = cudaMemcpy(xyz->z, xyz_host[2], nnodes * sizeof(geom_t), cudaMemcpyHostToDevice);

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess) {
        printf("Error copying xyz_tet10_device to managed memory: %s\n", cudaGetErrorString(cudaGetLastError()));
        // Handle the error or exit the program
    }
}  // end copy_xyz_tet10_managed

//////////////////////////////////////////////////////////
// free_xyz_tet10_managed
//////////////////////////////////////////////////////////
void                                             //
free_xyz_tet10_managed(xyz_tet10_device* xyz) {  //
                                                 //
    cudaError_t err0 = cudaFree(xyz->x);
    cudaError_t err1 = cudaFree(xyz->y);
    cudaError_t err2 = cudaFree(xyz->z);

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess) {
        printf("Error freeing managed memory for xyz: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    xyz->x = NULL;
    xyz->y = NULL;
    xyz->z = NULL;
}  // end free_xyz_tet10_managed

//////////////////////////////////////////////////////////
// memory_hint_xyz_tet10_managed
//////////////////////////////////////////////////////////
void                                                     //
memory_hint_xyz_tet10_managed(const ptrdiff_t   nnodes,  //
                              xyz_tet10_device* xyz) {   //
                                                         //
    cudaMemAdvise(xyz->x, nnodes * sizeof(geom_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(xyz->y, nnodes * sizeof(geom_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(xyz->z, nnodes * sizeof(geom_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
}  // end memory_hint_xyz_tet10_managed

//////////////////////////////////////////////////////////
// make_elems_tet10_device
//////////////////////////////////////////////////////////
elems_tet10_device                                    //
make_elems_tet10_device(const ptrdiff_t nelements) {  //
                                                      //
    elems_tet10_device elems;                         //

    cudaError_t err0 = cudaMalloc(&elems.elems_v0, nelements * sizeof(idx_t));
    cudaError_t err1 = cudaMalloc(&elems.elems_v1, nelements * sizeof(idx_t));
    cudaError_t err2 = cudaMalloc(&elems.elems_v2, nelements * sizeof(idx_t));
    cudaError_t err3 = cudaMalloc(&elems.elems_v3, nelements * sizeof(idx_t));
    cudaError_t err4 = cudaMalloc(&elems.elems_v4, nelements * sizeof(idx_t));
    cudaError_t err5 = cudaMalloc(&elems.elems_v5, nelements * sizeof(idx_t));
    cudaError_t err6 = cudaMalloc(&elems.elems_v6, nelements * sizeof(idx_t));
    cudaError_t err7 = cudaMalloc(&elems.elems_v7, nelements * sizeof(idx_t));
    cudaError_t err8 = cudaMalloc(&elems.elems_v8, nelements * sizeof(idx_t));
    cudaError_t err9 = cudaMalloc(&elems.elems_v9, nelements * sizeof(idx_t));

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess ||
        err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess || err8 != cudaSuccess || err9 != cudaSuccess) {
        printf("ERROR: allocating memory for elems_tet10_device\n");
        // Handle error
    }

    return elems;
}  // end make_elems_tet10_device

elems_tet10_device                                                               //
make_elems_tet10_device_async(const ptrdiff_t nelements, cudaStream_t stream) {  //
    elems_tet10_device elems;

    cudaError_t err0 = cudaMallocAsync(&elems.elems_v0, nelements * sizeof(idx_t), stream);
    cudaError_t err1 = cudaMallocAsync(&elems.elems_v1, nelements * sizeof(idx_t), stream);
    cudaError_t err2 = cudaMallocAsync(&elems.elems_v2, nelements * sizeof(idx_t), stream);
    cudaError_t err3 = cudaMallocAsync(&elems.elems_v3, nelements * sizeof(idx_t), stream);
    cudaError_t err4 = cudaMallocAsync(&elems.elems_v4, nelements * sizeof(idx_t), stream);
    cudaError_t err5 = cudaMallocAsync(&elems.elems_v5, nelements * sizeof(idx_t), stream);
    cudaError_t err6 = cudaMallocAsync(&elems.elems_v6, nelements * sizeof(idx_t), stream);
    cudaError_t err7 = cudaMallocAsync(&elems.elems_v7, nelements * sizeof(idx_t), stream);
    cudaError_t err8 = cudaMallocAsync(&elems.elems_v8, nelements * sizeof(idx_t), stream);
    cudaError_t err9 = cudaMallocAsync(&elems.elems_v9, nelements * sizeof(idx_t), stream);

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess ||
        err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess || err8 != cudaSuccess || err9 != cudaSuccess) {
        printf("ERROR: allocating memory for elems_tet10_device async\n");
        // Handle error
    }

    return elems;
}

//////////////////////////////////////////////////////////
// copy_elems_tet10_device
//////////////////////////////////////////////////////////
cudaError_t copy_elems_tet10_device(const ptrdiff_t     nelements,  //
                                    elems_tet10_device* elems,      //
                                    const idx_t**       elems_host) {     //

    cudaError_t err0 = cudaMemcpy(elems->elems_v0, elems_host[0], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err1 = cudaMemcpy(elems->elems_v1, elems_host[1], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err2 = cudaMemcpy(elems->elems_v2, elems_host[2], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err3 = cudaMemcpy(elems->elems_v3, elems_host[3], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err4 = cudaMemcpy(elems->elems_v4, elems_host[4], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err5 = cudaMemcpy(elems->elems_v5, elems_host[5], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err6 = cudaMemcpy(elems->elems_v6, elems_host[6], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err7 = cudaMemcpy(elems->elems_v7, elems_host[7], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err8 = cudaMemcpy(elems->elems_v8, elems_host[8], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaError_t err9 = cudaMemcpy(elems->elems_v9, elems_host[9], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess ||
        err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess || err8 != cudaSuccess || err9 != cudaSuccess) {
        printf("ERROR: copying elements to device: %s\n", cudaGetErrorString(cudaGetLastError()));
        return cudaGetLastError();
    }

    return cudaSuccess;
}  // end copy_elems_tet10_device

///////////////////////////////////////////////////////////
// copy_elems_tet10_device_async
///////////////////////////////////////////////////////////
cudaError_t                                                    //
copy_elems_tet10_device_async(const ptrdiff_t     nelements,   //
                              elems_tet10_device* elems,       //
                              const idx_t**       elems_host,  //
                              cudaStream_t        stream) {           //

    cudaError_t err0 = cudaMemcpyAsync(elems->elems_v0, elems_host[0], nelements * sizeof(idx_t), cudaMemcpyHostToDevice, stream);
    cudaError_t err1 = cudaMemcpyAsync(elems->elems_v1, elems_host[1], nelements * sizeof(idx_t), cudaMemcpyHostToDevice, stream);
    cudaError_t err2 = cudaMemcpyAsync(elems->elems_v2, elems_host[2], nelements * sizeof(idx_t), cudaMemcpyHostToDevice, stream);
    cudaError_t err3 = cudaMemcpyAsync(elems->elems_v3, elems_host[3], nelements * sizeof(idx_t), cudaMemcpyHostToDevice, stream);
    cudaError_t err4 = cudaMemcpyAsync(elems->elems_v4, elems_host[4], nelements * sizeof(idx_t), cudaMemcpyHostToDevice, stream);
    cudaError_t err5 = cudaMemcpyAsync(elems->elems_v5, elems_host[5], nelements * sizeof(idx_t), cudaMemcpyHostToDevice, stream);
    cudaError_t err6 = cudaMemcpyAsync(elems->elems_v6, elems_host[6], nelements * sizeof(idx_t), cudaMemcpyHostToDevice, stream);
    cudaError_t err7 = cudaMemcpyAsync(elems->elems_v7, elems_host[7], nelements * sizeof(idx_t), cudaMemcpyHostToDevice, stream);
    cudaError_t err8 = cudaMemcpyAsync(elems->elems_v8, elems_host[8], nelements * sizeof(idx_t), cudaMemcpyHostToDevice, stream);
    cudaError_t err9 = cudaMemcpyAsync(elems->elems_v9, elems_host[9], nelements * sizeof(idx_t), cudaMemcpyHostToDevice, stream);

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess ||
        err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess || err8 != cudaSuccess || err9 != cudaSuccess) {
        printf("ERROR: copying elements to device async: %s\n", cudaGetErrorString(cudaGetLastError()));
        return cudaGetLastError();
    }

    return cudaSuccess;
}

//////////////////////////////////////////////////////////
// free_elems_tet10_device
//////////////////////////////////////////////////////////
void free_elems_tet10_device(elems_tet10_device* elems) {  //
                                                           //
    cudaError_t err0 = cudaFree(elems->elems_v0);
    cudaError_t err1 = cudaFree(elems->elems_v1);
    cudaError_t err2 = cudaFree(elems->elems_v2);
    cudaError_t err3 = cudaFree(elems->elems_v3);
    cudaError_t err4 = cudaFree(elems->elems_v4);
    cudaError_t err5 = cudaFree(elems->elems_v5);
    cudaError_t err6 = cudaFree(elems->elems_v6);
    cudaError_t err7 = cudaFree(elems->elems_v7);
    cudaError_t err8 = cudaFree(elems->elems_v8);
    cudaError_t err9 = cudaFree(elems->elems_v9);

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess ||
        err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess || err8 != cudaSuccess || err9 != cudaSuccess) {
        printf("ERROR: freeing device memory for elems: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    elems->elems_v0 = NULL;
    elems->elems_v1 = NULL;
    elems->elems_v2 = NULL;
    elems->elems_v3 = NULL;
    elems->elems_v4 = NULL;
    elems->elems_v5 = NULL;
    elems->elems_v6 = NULL;
    elems->elems_v7 = NULL;
    elems->elems_v8 = NULL;
    elems->elems_v9 = NULL;
}  // end free_elems_tet10_device

//////////////////////////////////////////////////////////
// elems_tet10_device for unified memory
//////////////////////////////////////////////////////////

/**
 * @brief
 *
 * @param nelements
 * @return elems_tet10_device
 */
elems_tet10_device                                            //
make_elems_tet10_device_unified(const ptrdiff_t nelements) {  //
                                                              //
    elems_tet10_device elems;
    elems.elems_v0 = NULL;
    elems.elems_v1 = NULL;
    elems.elems_v2 = NULL;
    elems.elems_v3 = NULL;
    elems.elems_v4 = NULL;
    elems.elems_v5 = NULL;
    elems.elems_v6 = NULL;
    elems.elems_v7 = NULL;
    elems.elems_v8 = NULL;
    elems.elems_v9 = NULL;
    return elems;
}

/**
 * @brief
 *
 * @param nelements
 * @param elems
 * @param elems_host
 * @return cudaError_t
 */
cudaError_t                                                     //
copy_elems_tet10_device_unified(const ptrdiff_t     nelements,  //
                                elems_tet10_device* elems,      //
                                const idx_t**       elems_host) {
    elems->elems_v0 = (idx_t*)elems_host[0];
    elems->elems_v1 = (idx_t*)elems_host[1];
    elems->elems_v2 = (idx_t*)elems_host[2];
    elems->elems_v3 = (idx_t*)elems_host[3];
    elems->elems_v4 = (idx_t*)elems_host[4];
    elems->elems_v5 = (idx_t*)elems_host[5];
    elems->elems_v6 = (idx_t*)elems_host[6];
    elems->elems_v7 = (idx_t*)elems_host[7];
    elems->elems_v8 = (idx_t*)elems_host[8];
    elems->elems_v9 = (idx_t*)elems_host[9];

    return cudaSuccess;
}

/**
 * @brief
 *
 * @param elems
 */
void                                                          //
free_elems_tet10_device_unified(elems_tet10_device* elems) {  //
    elems->elems_v0 = NULL;
    elems->elems_v1 = NULL;
    elems->elems_v2 = NULL;
    elems->elems_v3 = NULL;
    elems->elems_v4 = NULL;
    elems->elems_v5 = NULL;
    elems->elems_v6 = NULL;
    elems->elems_v7 = NULL;
    elems->elems_v8 = NULL;
    elems->elems_v9 = NULL;

}  //

/**
 * @brief
 *
 * @param nelements
 * @param elems
 */
void                                                                   //
memory_hint_elems_tet10_device_unified(ptrdiff_t           nelements,  //
                                       elems_tet10_device* elems) {    //

    int  device_id;
    auto error = cudaGetDevice(&device_id);

    cudaError_t err0 = cudaMemAdvise(elems->elems_v0, nelements * sizeof(idx_t), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err1 = cudaMemAdvise(elems->elems_v1, nelements * sizeof(idx_t), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err2 = cudaMemAdvise(elems->elems_v2, nelements * sizeof(idx_t), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err3 = cudaMemAdvise(elems->elems_v3, nelements * sizeof(idx_t), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err4 = cudaMemAdvise(elems->elems_v4, nelements * sizeof(idx_t), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err5 = cudaMemAdvise(elems->elems_v5, nelements * sizeof(idx_t), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err6 = cudaMemAdvise(elems->elems_v6, nelements * sizeof(idx_t), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err7 = cudaMemAdvise(elems->elems_v7, nelements * sizeof(idx_t), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err8 = cudaMemAdvise(elems->elems_v8, nelements * sizeof(idx_t), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err9 = cudaMemAdvise(elems->elems_v9, nelements * sizeof(idx_t), cudaMemAdviseSetReadMostly, device_id);

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess ||
        err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess || err8 != cudaSuccess || err9 != cudaSuccess) {
        printf("ERROR: setting memory hint for elems_tet10_device: %s at %s:%d\n", cudaGetErrorString(err0), __FILE__, __LINE__);
        // Handle the error or exit the program
    }

    // prefetch the data to the GPU
    cudaError_t err10 = cudaMemPrefetchAsync(elems->elems_v0, nelements * sizeof(idx_t), device_id);
    cudaError_t err11 = cudaMemPrefetchAsync(elems->elems_v1, nelements * sizeof(idx_t), device_id);
    cudaError_t err12 = cudaMemPrefetchAsync(elems->elems_v2, nelements * sizeof(idx_t), device_id);
    cudaError_t err13 = cudaMemPrefetchAsync(elems->elems_v3, nelements * sizeof(idx_t), device_id);
    cudaError_t err14 = cudaMemPrefetchAsync(elems->elems_v4, nelements * sizeof(idx_t), device_id);
    cudaError_t err15 = cudaMemPrefetchAsync(elems->elems_v5, nelements * sizeof(idx_t), device_id);
    cudaError_t err16 = cudaMemPrefetchAsync(elems->elems_v6, nelements * sizeof(idx_t), device_id);
    cudaError_t err17 = cudaMemPrefetchAsync(elems->elems_v7, nelements * sizeof(idx_t), device_id);
    cudaError_t err18 = cudaMemPrefetchAsync(elems->elems_v8, nelements * sizeof(idx_t), device_id);
    cudaError_t err19 = cudaMemPrefetchAsync(elems->elems_v9, nelements * sizeof(idx_t), device_id);

    if (err10 != cudaSuccess || err11 != cudaSuccess || err12 != cudaSuccess || err13 != cudaSuccess || err14 != cudaSuccess ||
        err15 != cudaSuccess || err16 != cudaSuccess || err17 != cudaSuccess || err18 != cudaSuccess || err19 != cudaSuccess) {
        printf("ERROR: prefetching data for elems_tet10_device: %s at %s:%d\n", cudaGetErrorString(err10), __FILE__, __LINE__);
        // Handle the error or exit the program
    }
}

//////////////////////////////////////////////////////////
// memory_hint_elems_tet10_device
//////////////////////////////////////////////////////////
void                                                  //
memory_hint_read_mostly(const ptrdiff_t array_size,   //
                        const ptrdiff_t sizeof_type,  //
                        void*           ptr) {
    int  device_id = 0;
    auto error     = cudaGetDevice(&device_id);

    cudaError_t err = cudaMemAdvise(ptr, array_size * sizeof_type, cudaMemAdviseSetReadMostly, device_id);

    if (err != cudaSuccess) {
        printf("ERROR: setting memory hint for elems_tet10_device: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        // Handle the error or exit the program
    }

    // prefetch the data to the GPU
    cudaError_t err2 = cudaMemPrefetchAsync(ptr, array_size * sizeof_type, device_id);

    if (err2 != cudaSuccess) {
        printf("ERROR: prefetching data for elems_tet10_device: %s at %s:%d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
        // Handle the error or exit the program
    }
}

//////////////////////////////////////////////////////////
// memory_hint_write_mostly
//////////////////////////////////////////////////////////
void                                                   //
memory_hint_write_mostly(const ptrdiff_t array_size,   //
                         const ptrdiff_t sizeof_type,  //
                         void*           ptr) {
    int  device_id = 0;
    auto error     = cudaGetDevice(&device_id);

    cudaError_t err = cudaMemAdvise(ptr, array_size * sizeof_type, cudaMemAdviseSetAccessedBy, device_id);

    if (err != cudaSuccess) {
        printf("ERROR: setting memory hint for elems_tet10_device: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        // Handle the error or exit the program
    }

    // prefetch the data to the GPU
    cudaError_t err2 = cudaMemPrefetchAsync(ptr, array_size * sizeof_type, device_id);

    if (err2 != cudaSuccess) {
        printf("ERROR: prefetching data for elems_tet10_device: %s at %s:%d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
        // Handle the error or exit the program
    }
}