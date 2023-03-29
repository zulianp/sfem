// #include "laplacian.h"

#include <cassert>
#include <cmath>
// #include <cstdio>
#include <algorithm>
#include <cstddef>

// #include <mpi.h>

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

static inline __device__ void laplacian(const real_t x0,
                                        const real_t x1,
                                        const real_t x2,
                                        const real_t x3,
                                        const real_t y0,
                                        const real_t y1,
                                        const real_t y2,
                                        const real_t y3,
                                        const real_t z0,
                                        const real_t z1,
                                        const real_t z2,
                                        const real_t z3,
                                        const count_t stride,
                                        real_t *element_matrix) {
    // FLOATING POINT OPS!
    //    - Result: 4*ADD + 16*ASSIGNMENT + 16*MUL + 12*POW
    //    - Subexpressions: 16*ADD + 9*DIV + 56*MUL + 7*NEG + POW + 32*SUB
    const real_t x4 = z0 - z3;
    const real_t x5 = x0 - x1;
    const real_t x6 = y0 - y2;
    const real_t x7 = x5 * x6;
    const real_t x8 = z0 - z1;
    const real_t x9 = x0 - x2;
    const real_t x10 = y0 - y3;
    const real_t x11 = x10 * x9;
    const real_t x12 = z0 - z2;
    const real_t x13 = x0 - x3;
    const real_t x14 = y0 - y1;
    const real_t x15 = x13 * x14;
    const real_t x16 = x10 * x5;
    const real_t x17 = x14 * x9;
    const real_t x18 = x13 * x6;
    const real_t x19 = x11 * x8 + x12 * x15 - x12 * x16 - x17 * x4 - x18 * x8 + x4 * x7;
    const real_t x20 = 1.0 / x19;
    const real_t x21 = x11 - x18;
    const real_t x22 = -x17 + x7;
    const real_t x23 = x15 - x16 + x21 + x22;
    const real_t x24 = -x12 * x13 + x4 * x9;
    const real_t x25 = x12 * x5 - x8 * x9;
    const real_t x26 = x13 * x8;
    const real_t x27 = x4 * x5;
    const real_t x28 = x26 - x27;
    const real_t x29 = -x24 - x25 - x28;
    const real_t x30 = x10 * x8;
    const real_t x31 = x14 * x4;
    const real_t x32 = -x10 * x12 + x4 * x6;
    const real_t x33 = x12 * x14 - x6 * x8;
    const real_t x34 = x30 - x31 + x32 + x33;
    const real_t x35 = -x12;
    const real_t x36 = -x9;
    const real_t x37 = x19 * (x13 * x35 + x28 - x35 * x5 - x36 * x4 + x36 * x8);
    const real_t x38 = -x19;
    const real_t x39 = -x23;
    const real_t x40 = -x34;
    const real_t x41 = (1.0 / 6.0) / pow(x19, 2);
    const real_t x42 = x41 * (x24 * x37 + x38 * (x21 * x39 + x32 * x40));
    const real_t x43 = -x15 + x16;
    const real_t x44 = (1.0 / 6.0) * x43;
    const real_t x45 = -x26 + x27;
    const real_t x46 = -x30 + x31;
    const real_t x47 = (1.0 / 6.0) * x46;
    const real_t x48 = x20 * (-x23 * x44 + (1.0 / 6.0) * x29 * x45 - x34 * x47);
    const real_t x49 = x41 * (x25 * x37 + x38 * (x22 * x39 + x33 * x40));
    const real_t x50 = (1.0 / 6.0) * x45;
    const real_t x51 = x20 * (x21 * x44 + x24 * x50 + x32 * x47);
    const real_t x52 = x20 * (-1.0 / 6.0 * x21 * x22 - 1.0 / 6.0 * x24 * x25 - 1.0 / 6.0 * x32 * x33);
    const real_t x53 = x20 * (x22 * x44 + x25 * x50 + x33 * x47);

    element_matrix[0 * stride] = x20 * (-1.0 / 6.0 * pow(x23, 2) - 1.0 / 6.0 * pow(x29, 2) - 1.0 / 6.0 * pow(x34, 2));
    element_matrix[1 * stride] = x42;
    element_matrix[2 * stride] = x48;
    element_matrix[3 * stride] = x49;
    element_matrix[4 * stride] = x42;
    element_matrix[5 * stride] = x20 * (-1.0 / 6.0 * pow(x21, 2) - 1.0 / 6.0 * pow(x24, 2) - 1.0 / 6.0 * pow(x32, 2));
    element_matrix[6 * stride] = x51;
    element_matrix[7 * stride] = x52;
    element_matrix[8 * stride] = x48;
    element_matrix[9 * stride] = x51;
    element_matrix[10 * stride] = x20 * (-1.0 / 6.0 * pow(x43, 2) - 1.0 / 6.0 * pow(x45, 2) - 1.0 / 6.0 * pow(x46, 2));
    element_matrix[11 * stride] = x53;
    element_matrix[12 * stride] = x49;
    element_matrix[13 * stride] = x52;
    element_matrix[14 * stride] = x53;
    element_matrix[15 * stride] = x20 * (-1.0 / 6.0 * pow(x22, 2) - 1.0 / 6.0 * pow(x25, 2) - 1.0 / 6.0 * pow(x33, 2));
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

__global__ void laplacian_assemble_hessian_kernel(const ptrdiff_t nelements,
                                                  const geom_t *const SFEM_RESTRICT xyz,
                                                  real_t *const SFEM_RESTRICT values) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        real_t x[3][4];
// coalesced read
#pragma unroll(3)
        for (int d = 0; d < 3; ++d) {
#pragma unroll(4)
            for (int e_node = 0; e_node < 4; e_node++) {
                ptrdiff_t offset = (d * 4 + e_node) * nelements;
                x[d][e_node] = xyz[offset + e];
            }
        }

        laplacian(
            // X-coordinates
            x[0][0],
            x[0][1],
            x[0][2],
            x[0][3],
            // Y-coordinates
            x[1][0],
            x[1][1],
            x[1][2],
            x[1][3],
            // Z-coordinates
            x[2][0],
            x[2][1],
            x[2][2],
            x[2][3],
            nelements,
            &values[e]);
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

    const ptrdiff_t nbatch = MIN(128 * 2000, nelements);
    static int block_size = 128;
    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (nbatch + block_size - 1) / block_size);

    geom_t *he_xyz = nullptr;
    SFEM_CUDA_CHECK(cudaMallocHost(&he_xyz, 3 * 4 * nbatch * sizeof(geom_t)));
    geom_t *de_xyz = nullptr;
    SFEM_CUDA_CHECK(cudaMalloc(&de_xyz, 3 * 4 * nbatch * sizeof(geom_t)));

    real_t *he_matrix = nullptr;
    cudaMallocHost(&he_matrix, 4 * 4 * nbatch * sizeof(real_t));
    real_t *de_matrix = nullptr;
    SFEM_CUDA_CHECK(cudaMalloc(&de_matrix, 4 * 4 * nbatch * sizeof(real_t)));

    idx_t **hd_elems[4];
    idx_t **d_elems = nullptr;

    count_t *d_rowptr = nullptr;
    idx_t *d_colidx = nullptr;
    real_t *d_values = nullptr;

    int use_small = 1;

    static const int nstreams = 2;
    cudaStream_t cu_stream[nstreams];
    // cudaEvent_t cu_event[nstreams];
    for (int s = 0; s < nstreams; s++) {
        cudaStreamCreate(&cu_stream[s]);
        // cudaEventCreate(&cu_event[s]);
    }

    // Allocate space for indices
    for (int d = 0; d < 4; d++) {
        SFEM_CUDA_CHECK(cudaMalloc(&hd_elems[d], nbatch * sizeof(idx_t)));
    }

    SFEM_CUDA_CHECK(cudaMalloc(&d_elems, 4 * sizeof(idx_t *)));
    cudaMemcpy(d_elems, hd_elems, 4 * sizeof(idx_t *), cudaMemcpyHostToDevice);

    // Copy crs-matrix
    crs_device_create(nnodes, rowptr[nnodes], &d_rowptr, &d_colidx, &d_values);
    crs_graph_host_to_device(nnodes, rowptr[nnodes], rowptr, colidx, d_rowptr, d_colidx);

    ptrdiff_t last_element_offset = 0;
    ptrdiff_t n = 0;
    for (ptrdiff_t element_offset = 0; element_offset < nelements; element_offset += nbatch) {
        n = MIN(nbatch, nelements - element_offset);
        last_element_offset = element_offset;

        {
            for (int d = 0; d < 3; ++d) {
                const geom_t *const x = xyz[d];

                for (int e_node = 0; e_node < 4; e_node++) {
                    ptrdiff_t offset = (d * 4 + e_node) * nbatch;
                    const idx_t *const nodes = elems[e_node];

                    geom_t *buff = &he_xyz[offset];
                    for (ptrdiff_t k = 0; k < n; k++) {
                        buff[k] = x[nodes[k]];
                    }
                }
            }
        }

        SFEM_CUDA_CHECK(
            cudaMemcpyAsync(de_xyz, he_xyz, 3 * 4 * n * sizeof(geom_t), cudaMemcpyHostToDevice, cu_stream[0]));

        cudaStreamSynchronize(cu_stream[1]);
        for (int e_node = 0; e_node < 4; e_node++) {
            SFEM_CUDA_CHECK(cudaMemcpy(
                hd_elems[e_node], &elems[e_node][element_offset], n * sizeof(idx_t), cudaMemcpyHostToDevice));
        }

        laplacian_assemble_hessian_kernel<<<n_blocks, block_size, 0, cu_stream[0]>>>(n, de_xyz, de_matrix);
        cudaStreamSynchronize(cu_stream[0]);
        local_to_global_kernel<<<n_blocks, block_size, 0, cu_stream[1]>>>(
            n, d_elems, de_matrix, d_rowptr, d_colidx, d_values);

        //  SFEM_DEBUG_SYNCHRONIZE();
    }

    SFEM_CUDA_CHECK(cudaMemcpy(values, d_values, rowptr[nnodes] * sizeof(real_t), cudaMemcpyDeviceToHost));

    {  // Free resources on CPU
        cudaFreeHost(he_xyz);
        cudaFreeHost(he_matrix);
    }

    {  // Free resources on GPU
        SFEM_CUDA_CHECK(cudaFree(de_xyz));
        SFEM_CUDA_CHECK(cudaFree(de_matrix));

        for (int d = 0; d < 4; d++) {
            SFEM_CUDA_CHECK(cudaFree(hd_elems[d]));
        }
        SFEM_CUDA_CHECK(cudaFree(d_elems));

        crs_device_free(d_rowptr, d_colidx, d_values);

        for (int s = 0; s < nstreams; s++) {
            cudaStreamDestroy(cu_stream[s]);
            // cudaEventDestroy(cu_event[s]);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("cuda_laplacian_2.c: laplacian_assemble_hessian\t%g seconds\nloops %d\n",
              milliseconds/1000.
           int(nelements / nbatch));
}
