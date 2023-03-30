// #include "laplacian.h"

#include <cassert>
#include <cmath>
// #include <cstdio>
#include <algorithm>
#include <cstddef>



extern "C" {
#include "sfem_base.h"

#include "crs_graph.h"
#include "cuda_crs.h"
#include "sfem_base.h"
#include "sfem_vec.h"
#include "sortreduce.h"
}

#include "sfem_cuda_base.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

static inline __device__ void laplacian(const real_t *SFEM_RESTRICT jac_inv,
                                        const count_t stride,
                                        real_t *SFEM_RESTRICT element_matrix)

{
    real_t dv;
    {
        dv = 0.16666666666666666 / (jac_inv[0 * stride] * jac_inv[4 * stride] * jac_inv[8 * stride] -
                                    jac_inv[0 * stride] * jac_inv[5 * stride] * jac_inv[7 * stride] -
                                    jac_inv[1 * stride] * jac_inv[3 * stride] * jac_inv[8 * stride] +
                                    jac_inv[1 * stride] * jac_inv[5 * stride] * jac_inv[6 * stride] +
                                    jac_inv[2 * stride] * jac_inv[3 * stride] * jac_inv[7 * stride] -
                                    jac_inv[2 * stride] * jac_inv[4 * stride] * jac_inv[6 * stride]);

        assert(dv == dv);
    }

    {
        // FLOATING POINT OPS!
        //       - Result: 4*ADD + 16*ASSIGNMENT + 4*MUL + 12*POW
        //       - Subexpressions: 12*ADD + 24*MUL + 3*NEG + 6*SUB
        const real_t x0 = -jac_inv[0 * stride] - jac_inv[3 * stride] - jac_inv[6 * stride];
        const real_t x1 = -jac_inv[1 * stride] - jac_inv[4 * stride] - jac_inv[7 * stride];
        const real_t x2 = -jac_inv[2 * stride] - jac_inv[5 * stride] - jac_inv[8 * stride];
        const real_t x3 = dv * (jac_inv[0 * stride] * x0 + jac_inv[1 * stride] * x1 + jac_inv[2 * stride] * x2);
        const real_t x4 = dv * (jac_inv[3 * stride] * x0 + jac_inv[4 * stride] * x1 + jac_inv[5 * stride] * x2);
        const real_t x5 = dv * (jac_inv[6 * stride] * x0 + jac_inv[7 * stride] * x1 + jac_inv[8 * stride] * x2);
        const real_t x6 = dv * (jac_inv[0 * stride] * jac_inv[3 * stride] + jac_inv[1 * stride] * jac_inv[4 * stride] +
                                jac_inv[2 * stride] * jac_inv[5 * stride]);
        const real_t x7 = dv * (jac_inv[0 * stride] * jac_inv[6 * stride] + jac_inv[1 * stride] * jac_inv[7 * stride] +
                                jac_inv[2 * stride] * jac_inv[8 * stride]);
        const real_t x8 = dv * (jac_inv[3 * stride] * jac_inv[6 * stride] + jac_inv[4 * stride] * jac_inv[7 * stride] +
                                jac_inv[5 * stride] * jac_inv[8 * stride]);
        element_matrix[0 * stride] = dv * (pow(x0, 2) + pow(x1, 2) + pow(x2, 2));
        element_matrix[1 * stride] = x3;
        element_matrix[2 * stride] = x4;
        element_matrix[3 * stride] = x5;
        element_matrix[4 * stride] = x3;
        element_matrix[5 * stride] = dv * (pow(jac_inv[0], 2) + pow(jac_inv[1], 2) + pow(jac_inv[2], 2));
        element_matrix[6 * stride] = x6;
        element_matrix[7 * stride] = x7;
        element_matrix[8 * stride] = x4;
        element_matrix[9 * stride] = x6;
        element_matrix[10 * stride] = dv * (pow(jac_inv[3], 2) + pow(jac_inv[4], 2) + pow(jac_inv[5], 2));
        element_matrix[11 * stride] = x8;
        element_matrix[12 * stride] = x5;
        element_matrix[13 * stride] = x7;
        element_matrix[14 * stride] = x8;
    }
}

static inline __device__ __host__ int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - SFEM_VECTOR_SIZE; i += SFEM_VECTOR_SIZE) {
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }
    for (; i < size; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

static inline __device__ __host__ int find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    // if (lenrow <= 32)
    // {
    return linear_search(key, row, lenrow);

    // Using sentinel (potentially dangerous if matrix is buggy and column does not exist)
    // while (key > row[++k]) {
    //     // Hi
    // }
    // assert(k < lenrow);
    // assert(key == row[k]);
    // } else {
    //     // Use this for larger number of dofs per row
    //     return find_idx_binary_search(key, row, lenrow);
    // }
}

static inline __device__ __host__ void find_cols4(const idx_t *targets,
                                                  const idx_t *const row,
                                                  const int lenrow,
                                                  int *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 4; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(4)
        for (int d = 0; d < 4; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(4)
            for (int d = 0; d < 4; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

static inline __device__ __host__ void jacobian_inverse_micro_kernel(const real_t px0,
                                                                     const real_t px1,
                                                                     const real_t px2,
                                                                     const real_t px3,
                                                                     const real_t py0,
                                                                     const real_t py1,
                                                                     const real_t py2,
                                                                     const real_t py3,
                                                                     const real_t pz0,
                                                                     const real_t pz1,
                                                                     const real_t pz2,
                                                                     const real_t pz3,
                                                                     const count_t stride,
                                                                     real_t *jac_inv) {
    // FLOATING POINT OPS!
    //       - Result: 9*ADD + 9*ASSIGNMENT + 25*MUL
    //       - Subexpressions: 2*ADD + DIV + 12*MUL + 12*SUB
    const real_t x0 = -py0 + py2;
    const real_t x1 = -pz0 + pz3;
    const real_t x2 = x0 * x1;
    const real_t x3 = -py0 + py3;
    const real_t x4 = -pz0 + pz2;
    const real_t x5 = x3 * x4;
    const real_t x6 = -px0 + px1;
    const real_t x7 = -pz0 + pz1;
    const real_t x8 = -px0 + px2;
    const real_t x9 = x3 * x8;
    const real_t x10 = -py0 + py1;
    const real_t x11 = -px0 + px3;
    const real_t x12 = x1 * x8;
    const real_t x13 = x0 * x11;
    const real_t x14 = 1.0 / (x10 * x11 * x4 - x10 * x12 - x13 * x7 + x2 * x6 - x5 * x6 + x7 * x9);
    jac_inv[0 * stride] = x14 * (x2 - x5);
    jac_inv[1 * stride] = x14 * (x11 * x4 - x12);
    jac_inv[2 * stride] = x14 * (-x13 + x9);
    jac_inv[3 * stride] = x14 * (-x1 * x10 + x3 * x7);
    jac_inv[4 * stride] = x14 * (x1 * x6 - x11 * x7);
    jac_inv[5 * stride] = x14 * (x10 * x11 - x3 * x6);
    jac_inv[6 * stride] = x14 * (-x0 * x7 + x10 * x4);
    jac_inv[7 * stride] = x14 * (-x4 * x6 + x7 * x8);
}

__global__ void jacobian_inverse_kernel(const ptrdiff_t nelements,
                                        const geom_t *const SFEM_RESTRICT xyz,
                                        real_t *const SFEM_RESTRICT jacobian_inverse) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        ptrdiff_t xi = 0 * 4;
        ptrdiff_t yi = 1 * 4;
        ptrdiff_t zi = 2 * 4;

        jacobian_inverse_micro_kernel(
            // X-coordinates
            xyz[(xi + 0) * nelements + e],
            xyz[(xi + 1) * nelements + e],
            xyz[(xi + 2) * nelements + e],
            xyz[(xi + 3) * nelements + e],
            // Y-coordinates
            xyz[(yi + 0) * nelements + e],
            xyz[(yi + 1) * nelements + e],
            xyz[(yi + 2) * nelements + e],
            xyz[(yi + 3) * nelements + e],
            // Z-coordinates
            xyz[(zi + 0) * nelements + e],
            xyz[(zi + 1) * nelements + e],
            xyz[(zi + 2) * nelements + e],
            xyz[(zi + 3) * nelements + e],
            nelements,
            jacobian_inverse);
    }
}

__global__ void laplacian_assemble_hessian_kernel(const ptrdiff_t nelements,
                                                  const real_t *const SFEM_RESTRICT jacobian_inverse,
                                                  real_t *const SFEM_RESTRICT values) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        laplacian(jacobian_inverse, nelements, &values[e]);
    }
}

__global__ void local_to_global_kernel(const ptrdiff_t nelements,
                                       idx_t **const SFEM_RESTRICT elems,
                                       const real_t *const SFEM_RESTRICT e_matrix,
                                       const count_t *const SFEM_RESTRICT rowptr,
                                       const idx_t *const SFEM_RESTRICT colidx,
                                       real_t *const SFEM_RESTRICT values) {
    idx_t ev[4];
    idx_t ks[4];
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][e];
        }
#pragma unroll(4)
        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = elems[edof_i][e];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            const idx_t *row = &colidx[rowptr[dof_i]];

            find_cols4(ev, row, lenrow, ks);

            real_t *rowvalues = &values[rowptr[dof_i]];

#pragma unroll(4)
            for (int edof_j = 0; edof_j < 4; ++edof_j) {
                real_t v = e_matrix[(edof_i * 4 + edof_j) * nelements];
                atomicAdd(&rowvalues[ks[edof_j]], v);
            }
        }
    }
}

__global__ void print_elem_kernel(const ptrdiff_t nelements, idx_t **const elems) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nelements) return;

    printf("%d %d %d %d\n", elems[0][i], elems[1][i], elems[2][i], elems[3][i]);
}

static void local_to_global(const ptrdiff_t n,
                            const ptrdiff_t nnodes,
                            const ptrdiff_t element_offset,
                            idx_t **const SFEM_RESTRICT elems,
                            geom_t **const SFEM_RESTRICT xyz,
                            const real_t *const SFEM_RESTRICT he_matrix,
                            const count_t *const SFEM_RESTRICT rowptr,
                            const idx_t *const SFEM_RESTRICT colidx,
                            real_t *const SFEM_RESTRICT values) {
    for (int edof_i = 0; edof_i < 4; ++edof_i) {
        for (int edof_j = 0; edof_j < 4; ++edof_j) {
            ptrdiff_t offset = (edof_i * 4 + edof_j) * n;

            for (ptrdiff_t k = 0; k < n; k++) {
                const idx_t ek = element_offset + k;
                const idx_t node_i = elems[edof_i][ek];
                const count_t row_begin = rowptr[node_i];
                const count_t row_end = rowptr[node_i + 1];
                const count_t row_len = row_end - row_begin;

                ptrdiff_t crs_idx = row_begin + find_col(elems[edof_j][ek], &colidx[row_begin], row_len);
                values[crs_idx] += he_matrix[offset + k];

                assert(values[crs_idx] == values[crs_idx]);
            }
        }
    }
}

extern "C" void laplacian_assemble_hessian(const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elems,
                                           geom_t **const SFEM_RESTRICT xyz,
                                           const count_t *const SFEM_RESTRICT rowptr,
                                           const idx_t *const SFEM_RESTRICT colidx,
                                           real_t *const SFEM_RESTRICT values) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    static int block_size = 128;
    const ptrdiff_t nbatch = MIN(block_size * 1000, nelements);

    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (nbatch + block_size - 1) / block_size);

    geom_t *he_xyz = nullptr;
    SFEM_CUDA_CHECK(cudaMallocHost(&he_xyz, 3 * 4 * nbatch * sizeof(geom_t)));
    geom_t *de_xyz = nullptr;
    SFEM_CUDA_CHECK(cudaMalloc(&de_xyz, 3 * 4 * nbatch * sizeof(geom_t)));

    real_t *d_jacobian_inverse = nullptr;
    SFEM_CUDA_CHECK(cudaMalloc(&d_jacobian_inverse, 3 * 3 * nbatch * sizeof(real_t)));

    real_t *he_matrix = nullptr;
    cudaMallocHost(&he_matrix, 4 * 4 * nbatch * sizeof(real_t));
    real_t *de_matrix = nullptr;
    SFEM_CUDA_CHECK(cudaMalloc(&de_matrix, 4 * 4 * nbatch * sizeof(real_t)));

    idx_t **hd_elems[4];
    idx_t **d_elems = nullptr;

    count_t *d_rowptr = nullptr;
    idx_t *d_colidx = nullptr;
    real_t *d_values = nullptr;

    // Allocate space for indices
    for (int d = 0; d < 4; d++) {
        SFEM_CUDA_CHECK(cudaMalloc(&hd_elems[d], nbatch * sizeof(idx_t)));
    }

    SFEM_CUDA_CHECK(cudaMalloc(&d_elems, 4 * sizeof(idx_t *)));
    cudaMemcpy(d_elems, hd_elems, 4 * sizeof(idx_t *), cudaMemcpyHostToDevice);

    // Copy crs-matrix
    crs_device_create(nnodes, rowptr[nnodes], &d_rowptr, &d_colidx, &d_values);
    crs_graph_host_to_device(nnodes, rowptr[nnodes], rowptr, colidx, d_rowptr, d_colidx);

    ptrdiff_t last_n = 0;
    for (ptrdiff_t element_offset = 0; element_offset < nelements; element_offset += nbatch) {
        ptrdiff_t n = MIN(nbatch, nelements - element_offset);

        {
            // #pragma omp parallel
            {
                // #pragma omp parallel for collapse(2)
                for (int d = 0; d < 3; ++d) {
                    for (int e_node = 0; e_node < 4; e_node++) {
                        // printf("%d %d\n", d, e_node)
                        const geom_t *const x = xyz[d];
                        ptrdiff_t offset = (d * 4 + e_node) * nbatch;
                        const idx_t *const nodes = elems[e_node];

                        geom_t *buff = &he_xyz[offset];
                        // #pragma omp parallel for
                        for (ptrdiff_t k = 0; k < n; k++) {
                            buff[k] = x[nodes[k]];
                        }
                    }
                }
            }
        }

        if (last_n) {
            // Do this here to let the main kernel overlap with the packing
            local_to_global_kernel<<<n_blocks, block_size>>>(n, d_elems, de_matrix, d_rowptr, d_colidx, d_values);
        }

        SFEM_CUDA_CHECK(cudaMemcpy(de_xyz, he_xyz, 3 * 4 * n * sizeof(geom_t), cudaMemcpyHostToDevice));

        for (int e_node = 0; e_node < 4; e_node++) {
            SFEM_CUDA_CHECK(cudaMemcpy(
                hd_elems[e_node], &elems[e_node][element_offset], n * sizeof(idx_t), cudaMemcpyHostToDevice));
        }

        jacobian_inverse_kernel<<<n_blocks, block_size>>>(n, de_xyz, d_jacobian_inverse);
        laplacian_assemble_hessian_kernel<<<n_blocks, block_size>>>(n, d_jacobian_inverse, de_matrix);
        last_n = n;
    }

    if (last_n) {
        local_to_global_kernel<<<n_blocks, block_size>>>(last_n, d_elems, de_matrix, d_rowptr, d_colidx, d_values);
    }


    SFEM_CUDA_CHECK(cudaMemcpy(values, d_values, rowptr[nnodes] * sizeof(real_t), cudaMemcpyDeviceToHost));

    {  // Free resources on CPU
        cudaFreeHost(he_xyz);
        cudaFreeHost(he_matrix);
    }

    {  // Free resources on GPU
        SFEM_CUDA_CHECK(cudaFree(de_xyz));
        SFEM_CUDA_CHECK(cudaFree(de_matrix));
        SFEM_CUDA_CHECK(cudaFree(d_jacobian_inverse));

        for (int d = 0; d < 4; d++) {
            SFEM_CUDA_CHECK(cudaFree(hd_elems[d]));
        }
        SFEM_CUDA_CHECK(cudaFree(d_elems));

        crs_device_free(d_rowptr, d_colidx, d_values);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("cuda_laplacian_2.c: laplacian_assemble_hessian\t%g seconds\nloops %d\n",
           milliseconds/1000,
           int(nelements / nbatch));
}

// extern "C" void laplacian_assemble_hessian(const ptrdiff_t nelements,
//                                            const ptrdiff_t nnodes,
//                                            idx_t **const SFEM_RESTRICT elems,
//                                            geom_t **const SFEM_RESTRICT xyz,
//                                            const count_t *const SFEM_RESTRICT rowptr,
//                                            const idx_t *const SFEM_RESTRICT colidx,
//                                            real_t *const SFEM_RESTRICT values) {
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     cudaEventRecord(start);

//     static int block_size = 128;
//     const ptrdiff_t nbatch = MIN(block_size * 1000, nelements);

//     ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (nbatch + block_size - 1) / block_size);

//     geom_t *he_xyz = nullptr;
//     SFEM_CUDA_CHECK(cudaMallocHost(&he_xyz, 3 * 4 * nbatch * sizeof(geom_t)));
//     geom_t *de_xyz = nullptr;
//     SFEM_CUDA_CHECK(cudaMalloc(&de_xyz, 3 * 4 * nbatch * sizeof(geom_t)));

//     real_t *d_jacobian_inverse = nullptr;
//     SFEM_CUDA_CHECK(cudaMalloc(&d_jacobian_inverse, 3 * 3 * nbatch * sizeof(real_t)));

//     real_t *he_matrix = nullptr;
//     cudaMallocHost(&he_matrix, 4 * 4 * nbatch * sizeof(real_t));
//     real_t *de_matrix = nullptr;
//     SFEM_CUDA_CHECK(cudaMalloc(&de_matrix, 4 * 4 * nbatch * sizeof(real_t)));

//     idx_t **hd_elems[4];
//     idx_t **d_elems = nullptr;

//     count_t *d_rowptr = nullptr;
//     idx_t *d_colidx = nullptr;
//     real_t *d_values = nullptr;


//     static const int nstreams = 2;
//     cudaStream_t stream[nstreams];
//     // cudaEvent_t cu_event[nstreams];
//     for (int s = 0; s < nstreams; s++) {
//         cudaStreamCreate(&stream[s]);
//         // cudaEventCreate(&cu_event[s]);
//     }

//     // Allocate space for indices
//     for (int d = 0; d < 4; d++) {
//         SFEM_CUDA_CHECK(cudaMalloc(&hd_elems[d], nbatch * sizeof(idx_t)));
//     }

//     SFEM_CUDA_CHECK(cudaMalloc(&d_elems, 4 * sizeof(idx_t *)));
//     cudaMemcpy(d_elems, hd_elems, 4 * sizeof(idx_t *), cudaMemcpyHostToDevice);

//     // Copy crs-matrix
//     crs_device_create(nnodes, rowptr[nnodes], &d_rowptr, &d_colidx, &d_values);
//     crs_graph_host_to_device(nnodes, rowptr[nnodes], rowptr, colidx, d_rowptr, d_colidx);

//     ptrdiff_t last_n = 0;
//     ptrdiff_t last_element_offset = 0;
//     for (ptrdiff_t element_offset = 0; element_offset < nelements; element_offset += nbatch) {
//         ptrdiff_t n = MIN(nbatch, nelements - element_offset);

//         {
//             // #pragma omp parallel
//             {
//                 // #pragma omp parallel for collapse(2)
//                 for (int d = 0; d < 3; ++d) {
//                     for (int e_node = 0; e_node < 4; e_node++) {
//                         // printf("%d %d\n", d, e_node)
//                         const geom_t *const x = xyz[d];
//                         ptrdiff_t offset = (d * 4 + e_node) * nbatch;
//                         const idx_t *const nodes = elems[e_node];

//                         geom_t *buff = &he_xyz[offset];
//                         // #pragma omp parallel for
//                         for (ptrdiff_t k = 0; k < n; k++) {
//                             buff[k] = x[nodes[k]];
//                         }
//                     }
//                 }
//             }
//         }

//         for (int e_node = 0; e_node < 4; e_node++) {
//             unsigned int mem_flag = CU_MEMHOSTREGISTER_PORTABLE;
//             // unsigned int mem_flag = CU_MEMHOSTREGISTER_READ_ONLY;

//             SFEM_CUDA_CHECK(
//                 cudaHostRegister(&elems[e_node][element_offset], n * sizeof(idx_t), mem_flag));
//         }

//         if (last_n) {
//             cudaStreamSynchronize(stream[0]);
//             // Do this here to let the main kernel overlap with the packing
//             local_to_global_kernel<<<n_blocks, block_size, 0, stream[1]>>>(n, d_elems, de_matrix, d_rowptr, d_colidx, d_values);
//         }
       

//         // SFEM_CUDA_CHECK(cudaMemcpy(de_xyz, he_xyz, 3 * 4 * n * sizeof(geom_t), cudaMemcpyHostToDevice));        
//         SFEM_CUDA_CHECK(cudaMemcpyAsync(de_xyz, he_xyz, 3 * 4 * n * sizeof(geom_t), cudaMemcpyHostToDevice, stream[0]));

//         if(last_n) {
//              for (int e_node = 0; e_node < 4; e_node++) {
//                 SFEM_CUDA_CHECK(cudaHostUnregister(&elems[e_node][last_element_offset]));
//             }
//         }

//         for (int e_node = 0; e_node < 4; e_node++) {
//             // SFEM_CUDA_CHECK(cudaMemcpy(
//             //     hd_elems[e_node], &elems[e_node][element_offset], n * sizeof(idx_t), cudaMemcpyHostToDevice));

//              SFEM_CUDA_CHECK(cudaMemcpyAsync(
//                 hd_elems[e_node], &elems[e_node][element_offset], n * sizeof(idx_t), cudaMemcpyHostToDevice, stream[1]));
//         }

//         jacobian_inverse_kernel<<<n_blocks, block_size, 0, stream[0]>>>(n, de_xyz, d_jacobian_inverse);
//         laplacian_assemble_hessian_kernel<<<n_blocks, block_size, 0, stream[0]>>>(n, d_jacobian_inverse, de_matrix);
//         last_n = n;
//         last_element_offset = element_offset;
//     }

//     if (last_n) {
//         local_to_global_kernel<<<n_blocks, block_size>>>(last_n, d_elems, de_matrix, d_rowptr, d_colidx, d_values);

//         for (int e_node = 0; e_node < 4; e_node++) {
//             SFEM_CUDA_CHECK(cudaHostUnregister(&elems[e_node][last_element_offset]));
//         }
//     }

//     SFEM_CUDA_CHECK(cudaMemcpy(values, d_values, rowptr[nnodes] * sizeof(real_t), cudaMemcpyDeviceToHost));

//     {  // Free resources on CPU
//         cudaFreeHost(he_xyz);
//         cudaFreeHost(he_matrix);
//     }

//     {  // Free resources on GPU
//         SFEM_CUDA_CHECK(cudaFree(de_xyz));
//         SFEM_CUDA_CHECK(cudaFree(de_matrix));
//         SFEM_CUDA_CHECK(cudaFree(d_jacobian_inverse));

//         for (int d = 0; d < 4; d++) {
//             SFEM_CUDA_CHECK(cudaFree(hd_elems[d]));
//         }
//         SFEM_CUDA_CHECK(cudaFree(d_elems));

//         crs_device_free(d_rowptr, d_colidx, d_values);

//         for (int s = 0; s < nstreams; s++) {
//                    cudaStreamDestroy(stream[s]);
//                    // cudaEventDestroy(cu_event[s]);
//                }
//     }

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);

//     printf("cuda_laplacian_2.c: laplacian_assemble_hessian\t%g seconds\nloops %d\n",
//            milliseconds / 1000,
//            int(nelements / nbatch));
// }
