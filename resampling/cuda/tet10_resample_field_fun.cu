#include <sfem_base.h>
#include <stdio.h>

// #define real_type real_t

#include "quadratures_rule_cuda.h"
#include "tet10_resample_field.cuh"

//////////////////////////////////////////////////////////
// make_xyz_tet10_device
//////////////////////////////////////////////////////////
xyz_tet10_device make_xyz_tet10_device(const ptrdiff_t nnodes) {
    xyz_tet10_device xyz;
    cudaMalloc(&xyz.x, nnodes * sizeof(float));
    cudaMalloc(&xyz.y, nnodes * sizeof(float));
    cudaMalloc(&xyz.z, nnodes * sizeof(float));
    return xyz;
}
// end make_xyz_tet10_device

//////////////////////////////////////////////////////////
// copy_xyz_tet10_device
//////////////////////////////////////////////////////////
void copy_xyz_tet10_device(const ptrdiff_t   nnodes,  //
                           xyz_tet10_device* xyz,     //
                           const float**     xyz_host) {  //
    cudaError_t err0 = cudaMemcpy(xyz->x, xyz_host[0], nnodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaError_t err1 = cudaMemcpy(xyz->y, xyz_host[1], nnodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaError_t err2 = cudaMemcpy(xyz->z, xyz_host[2], nnodes * sizeof(float), cudaMemcpyHostToDevice);

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
// make_xyz_tet10_device_unified
//////////////////////////////////////////////////////////
xyz_tet10_device make_xyz_tet10_device_unified(const ptrdiff_t nnodes) {
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
void copy_xyz_tet10_device_unified(const ptrdiff_t   nnodes,  //
                                   xyz_tet10_device* xyz,     //
                                   const float**     xyz_host) {
    xyz->x = (float*)xyz_host[0];
    xyz->y = (float*)xyz_host[1];
    xyz->z = (float*)xyz_host[2];
}
// end copy_xyz_tet10_device_unified

//////////////////////////////////////////////////////////
// memory_hint_xyz_tet10_device_unified
//////////////////////////////////////////////////////////
void                                                                                   //
memory_hint_xyz_tet10_device_unified(const ptrdiff_t nnodes, xyz_tet10_device* xyz) {  //

    cudaError_t err0 = cudaMemAdvise(xyz->x, nnodes * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaError_t err1 = cudaMemAdvise(xyz->y, nnodes * sizeof(float), cudaMemAdviseSetReadMostly, 0);
    cudaError_t err2 = cudaMemAdvise(xyz->z, nnodes * sizeof(float), cudaMemAdviseSetReadMostly, 0);

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess) {
        printf("ERROR: setting memory hint for xyz_tet10_device: %s at %s:%d\n", cudaGetErrorString(err0), __FILE__, __LINE__);
        // Handle the error or exit the program
    }

    // prefetch the data to the GPU
    cudaError_t err3 = cudaMemPrefetchAsync(xyz->x, nnodes * sizeof(float), 0, 0);
    cudaError_t err4 = cudaMemPrefetchAsync(xyz->y, nnodes * sizeof(float), 0, 0);
    cudaError_t err5 = cudaMemPrefetchAsync(xyz->z, nnodes * sizeof(float), 0, 0);

    if (err3 != cudaSuccess || err4 != cudaSuccess || err5 != cudaSuccess) {
        printf("ERROR: prefetching data for xyz_tet10_device: %s at %s:%d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
        // Handle the error or exit the program
    }
}

//////////////////////////////////////////////////////////
// free_xyz_tet10_device_unified
//////////////////////////////////////////////////////////
void free_xyz_tet10_device_unified(xyz_tet10_device* xyz) {
    xyz->x = NULL;
    xyz->y = NULL;
    xyz->z = NULL;
}
// end free_xyz_tet10_device_unified

//////////////////////////////////////////////////////////
// make_elems_tet10_device
//////////////////////////////////////////////////////////
elems_tet10_device make_elems_tet10_device(const ptrdiff_t nelements) {
    elems_tet10_device elems;

    cudaError_t err0 = cudaMalloc(&elems.elems_v0, nelements * sizeof(int));
    cudaError_t err1 = cudaMalloc(&elems.elems_v1, nelements * sizeof(int));
    cudaError_t err2 = cudaMalloc(&elems.elems_v2, nelements * sizeof(int));
    cudaError_t err3 = cudaMalloc(&elems.elems_v3, nelements * sizeof(int));
    cudaError_t err4 = cudaMalloc(&elems.elems_v4, nelements * sizeof(int));
    cudaError_t err5 = cudaMalloc(&elems.elems_v5, nelements * sizeof(int));
    cudaError_t err6 = cudaMalloc(&elems.elems_v6, nelements * sizeof(int));
    cudaError_t err7 = cudaMalloc(&elems.elems_v7, nelements * sizeof(int));
    cudaError_t err8 = cudaMalloc(&elems.elems_v8, nelements * sizeof(int));
    cudaError_t err9 = cudaMalloc(&elems.elems_v9, nelements * sizeof(int));

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess ||
        err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess || err8 != cudaSuccess || err9 != cudaSuccess) {
        printf("ERROR: allocating memory for elems_tet10_device\n");
        // Handle error
    }

    return elems;
}  // end make_elems_tet10_device

//////////////////////////////////////////////////////////
// copy_elems_tet10_device
//////////////////////////////////////////////////////////
cudaError_t copy_elems_tet10_device(const ptrdiff_t     nelements,  //
                                    elems_tet10_device* elems,      //
                                    const idx_t**       elems_host) {     //
    cudaError_t err0 = cudaMemcpy(elems->elems_v0, elems_host[0], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err1 = cudaMemcpy(elems->elems_v1, elems_host[1], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err2 = cudaMemcpy(elems->elems_v2, elems_host[2], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err3 = cudaMemcpy(elems->elems_v3, elems_host[3], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err4 = cudaMemcpy(elems->elems_v4, elems_host[4], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err5 = cudaMemcpy(elems->elems_v5, elems_host[5], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err6 = cudaMemcpy(elems->elems_v6, elems_host[6], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err7 = cudaMemcpy(elems->elems_v7, elems_host[7], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err8 = cudaMemcpy(elems->elems_v8, elems_host[8], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err9 = cudaMemcpy(elems->elems_v9, elems_host[9], nelements * sizeof(int), cudaMemcpyHostToDevice);

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess ||
        err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess || err8 != cudaSuccess || err9 != cudaSuccess) {
        printf("ERROR: copying elements to device: %s\n", cudaGetErrorString(cudaGetLastError()));
        return cudaGetLastError();
    }

    return cudaSuccess;
}  // end copy_elems_tet10_device

//////////////////////////////////////////////////////////
// free_elems_tet10_device
//////////////////////////////////////////////////////////
void free_elems_tet10_device(elems_tet10_device* elems) {
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
elems_tet10_device  //
make_elems_tet10_device_unified(const ptrdiff_t nelements) {
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
    elems->elems_v0 = (int*)elems_host[0];
    elems->elems_v1 = (int*)elems_host[1];
    elems->elems_v2 = (int*)elems_host[2];
    elems->elems_v3 = (int*)elems_host[3];
    elems->elems_v4 = (int*)elems_host[4];
    elems->elems_v5 = (int*)elems_host[5];
    elems->elems_v6 = (int*)elems_host[6];
    elems->elems_v7 = (int*)elems_host[7];
    elems->elems_v8 = (int*)elems_host[8];
    elems->elems_v9 = (int*)elems_host[9];

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
void                                                                                      //
memory_hint_elems_tet10_device_unified(ptrdiff_t nelements, elems_tet10_device* elems) {  //

    int  device_id;
    auto error = cudaGetDevice(&device_id);

    cudaError_t err0 = cudaMemAdvise(elems->elems_v0, nelements * sizeof(int), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err1 = cudaMemAdvise(elems->elems_v1, nelements * sizeof(int), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err2 = cudaMemAdvise(elems->elems_v2, nelements * sizeof(int), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err3 = cudaMemAdvise(elems->elems_v3, nelements * sizeof(int), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err4 = cudaMemAdvise(elems->elems_v4, nelements * sizeof(int), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err5 = cudaMemAdvise(elems->elems_v5, nelements * sizeof(int), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err6 = cudaMemAdvise(elems->elems_v6, nelements * sizeof(int), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err7 = cudaMemAdvise(elems->elems_v7, nelements * sizeof(int), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err8 = cudaMemAdvise(elems->elems_v8, nelements * sizeof(int), cudaMemAdviseSetReadMostly, device_id);
    cudaError_t err9 = cudaMemAdvise(elems->elems_v9, nelements * sizeof(int), cudaMemAdviseSetReadMostly, device_id);

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess ||
        err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess || err8 != cudaSuccess || err9 != cudaSuccess) {
        printf("ERROR: setting memory hint for elems_tet10_device: %s at %s:%d\n", cudaGetErrorString(err0), __FILE__, __LINE__);
        // Handle the error or exit the program
    }

    // prefetch the data to the GPU
    cudaError_t err10 = cudaMemPrefetchAsync(elems->elems_v0, nelements * sizeof(int), device_id);
    cudaError_t err11 = cudaMemPrefetchAsync(elems->elems_v1, nelements * sizeof(int), device_id);
    cudaError_t err12 = cudaMemPrefetchAsync(elems->elems_v2, nelements * sizeof(int), device_id);
    cudaError_t err13 = cudaMemPrefetchAsync(elems->elems_v3, nelements * sizeof(int), device_id);
    cudaError_t err14 = cudaMemPrefetchAsync(elems->elems_v4, nelements * sizeof(int), device_id);
    cudaError_t err15 = cudaMemPrefetchAsync(elems->elems_v5, nelements * sizeof(int), device_id);
    cudaError_t err16 = cudaMemPrefetchAsync(elems->elems_v6, nelements * sizeof(int), device_id);
    cudaError_t err17 = cudaMemPrefetchAsync(elems->elems_v7, nelements * sizeof(int), device_id);
    cudaError_t err18 = cudaMemPrefetchAsync(elems->elems_v8, nelements * sizeof(int), device_id);
    cudaError_t err19 = cudaMemPrefetchAsync(elems->elems_v9, nelements * sizeof(int), device_id);

    if (err10 != cudaSuccess || err11 != cudaSuccess || err12 != cudaSuccess || err13 != cudaSuccess || err14 != cudaSuccess ||
        err15 != cudaSuccess || err16 != cudaSuccess || err17 != cudaSuccess || err18 != cudaSuccess || err19 != cudaSuccess) {
        printf("ERROR: prefetching data for elems_tet10_device: %s at %s:%d\n", cudaGetErrorString(err10), __FILE__, __LINE__);
        // Handle the error or exit the program
    }
}

//////////////////////////////////////////////////////////
// memory_hint_elems_tet10_device
//////////////////////////////////////////////////////////
void                                                                                            //
memory_hint_read_mostly(const ptrdiff_t array_size, const ptrdiff_t element_size, void* ptr) {  //

    int  device_id = 0;
    auto error     = cudaGetDevice(&device_id);

    cudaError_t err = cudaMemAdvise(ptr, array_size * element_size, cudaMemAdviseSetReadMostly, device_id);

    if (err != cudaSuccess) {
        printf("ERROR: setting memory hint for elems_tet10_device: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        // Handle the error or exit the program
    }

    // prefetch the data to the GPU
    cudaError_t err2 = cudaMemPrefetchAsync(ptr, array_size * element_size, device_id);

    if (err2 != cudaSuccess) {
        printf("ERROR: prefetching data for elems_tet10_device: %s at %s:%d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
        // Handle the error or exit the program
    }
}

//////////////////////////////////////////////////////////
// memory_hint_write_mostly
//////////////////////////////////////////////////////////
void                                                                                             //
memory_hint_write_mostly(const ptrdiff_t array_size, const ptrdiff_t element_size, void* ptr) {  //

    int  device_id = 0;
    auto error     = cudaGetDevice(&device_id);

    cudaError_t err = cudaMemAdvise(ptr, array_size * element_size, cudaMemAdviseSetAccessedBy, device_id);

    if (err != cudaSuccess) {
        printf("ERROR: setting memory hint for elems_tet10_device: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        // Handle the error or exit the program
    }

    // prefetch the data to the GPU
    cudaError_t err2 = cudaMemPrefetchAsync(ptr, array_size * element_size, device_id);

    if (err2 != cudaSuccess) {
        printf("ERROR: prefetching data for elems_tet10_device: %s at %s:%d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
        // Handle the error or exit the program
    }
}