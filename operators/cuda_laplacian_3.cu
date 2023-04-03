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

#if 1
#include "nvToolsExt.h"
#define SFEM_RANGE_PUSH(name_) \
    do {                       \
        nvtxRangePushA(name_); \
    } while (0)
#define SFEM_RANGE_POP() \
    do {                 \
        nvtxRangePop();  \
    } while (0)
#else
#define SFEM_RANGE_PUSH(name_)
#define SFEM_RANGE_POP()
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define POW2(a) ((a) * (a))

static inline __device__ void laplacian(const real_t *SFEM_RESTRICT jac_inv,
                                        const count_t stride,
                                        real_t *SFEM_RESTRICT element_matrix)

{
    // FLOATING POINT OPS!
    //      - Result: ADD + 16*ASSIGNMENT + 3*MUL
    //      - Subexpressions: 3*NEG + 6*SUB
    const real_t x0 = -jac_inv[0 * stride] - jac_inv[1 * stride] - jac_inv[2 * stride];
    const real_t x1 = -jac_inv[1 * stride] - jac_inv[3 * stride] - jac_inv[4 * stride];
    const real_t x2 = -jac_inv[2 * stride] - jac_inv[4 * stride] - jac_inv[5 * stride];
    element_matrix[0 * stride] = jac_inv[0 * stride] + 2 * jac_inv[1 * stride] + 2 * jac_inv[2 * stride] +
                                 jac_inv[3 * stride] + 2 * jac_inv[4 * stride] + jac_inv[5 * stride];

    element_matrix[1 * stride] = x0;
    element_matrix[2 * stride] = x1;
    element_matrix[3 * stride] = x2;
    element_matrix[4 * stride] = x0;
    element_matrix[5 * stride] = jac_inv[0 * stride];
    element_matrix[6 * stride] = jac_inv[1 * stride];
    element_matrix[7 * stride] = jac_inv[2 * stride];
    element_matrix[8 * stride] = x1;
    element_matrix[9 * stride] = jac_inv[1 * stride];
    element_matrix[10 * stride] = jac_inv[3 * stride];
    element_matrix[11 * stride] = jac_inv[4 * stride];
    element_matrix[12 * stride] = x2;
    element_matrix[13 * stride] = jac_inv[2 * stride];
    element_matrix[14 * stride] = jac_inv[4 * stride];
    element_matrix[15 * stride] = jac_inv[5 * stride];
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

static inline __device__ __host__ void fff_micro_kernel(const real_t px0,
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
    //      - Result: 6*ADD + 6*ASSIGNMENT + 24*MUL + 9*POW
    //      - Subexpressions: 4*ADD + 6*DIV + 28*MUL + NEG + POW + 24*SUB
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = -pz0 + pz3;
    const real_t x3 = x1 * x2;
    const real_t x4 = x0 * x3;
    const real_t x5 = -py0 + py3;
    const real_t x6 = -pz0 + pz2;
    const real_t x7 = x5 * x6;
    const real_t x8 = x0 * x7;
    const real_t x9 = -py0 + py1;
    const real_t x10 = -px0 + px2;
    const real_t x11 = x10 * x2;
    const real_t x12 = x11 * x9;
    const real_t x13 = -pz0 + pz1;
    const real_t x14 = x10 * x5;
    const real_t x15 = x13 * x14;
    const real_t x16 = -px0 + px3;
    const real_t x17 = x16 * x6 * x9;
    const real_t x18 = x1 * x16;
    const real_t x19 = x13 * x18;
    const real_t x20 =
        -1.0 / 6.0 * x12 + (1.0 / 6.0) * x15 + (1.0 / 6.0) * x17 - 1.0 / 6.0 * x19 + (1.0 / 6.0) * x4 - 1.0 / 6.0 * x8;
    const real_t x21 = x14 - x18;
    const real_t x22 = 1./POW2(-x12 + x15 + x17 - x19 + x4 - x8);
    const real_t x23 = -x11 + x16 * x6;
    const real_t x24 = x3 - x7;
    const real_t x25 = -x0 * x5 + x16 * x9;
    const real_t x26 = x21 * x22;
    const real_t x27 = x0 * x2 - x13 * x16;
    const real_t x28 = x22 * x23;
    const real_t x29 = x13 * x5 - x2 * x9;
    const real_t x30 = x22 * x24;
    const real_t x31 = x0 * x1 - x10 * x9;
    const real_t x32 = -x0 * x6 + x10 * x13;
    const real_t x33 = -x1 * x13 + x6 * x9;
    jac_inv[0 * stride] = x20 * (POW2(x21) * x22 + x22 * POW2(x23) + x22 * POW2(x24));
    jac_inv[1 * stride] = x20 * (x25 * x26 + x27 * x28 + x29 * x30);
    jac_inv[2 * stride] = x20 * (x26 * x31 + x28 * x32 + x30 * x33);
    jac_inv[3 * stride] = x20 * (x22 * POW2(x25) + x22 * POW2(x27) + x22 * POW2(x29));
    jac_inv[4 * stride] = x20 * (x22 * x25 * x31 + x22 * x27 * x32 + x22 * x29 * x33);
    jac_inv[5 * stride] = x20 * (x22 * POW2(x31) + x22 * POW2(x32) + x22 * POW2(x33));
}

__global__ void fff_kernel(const ptrdiff_t nelements,
                                        const geom_t *const SFEM_RESTRICT xyz,
                                        real_t *const SFEM_RESTRICT fff) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        // Thy element coordinates and jacobian
        const geom_t *const this_xyz = &xyz[e];
        real_t *const this_fff = &fff[e];

        const ptrdiff_t xi = 0 * 4;
        const ptrdiff_t yi = 1 * 4;
        const ptrdiff_t zi = 2 * 4;

        fff_micro_kernel(
            // X-coordinates
            this_xyz[(xi + 0) * nelements],
            this_xyz[(xi + 1) * nelements],
            this_xyz[(xi + 2) * nelements],
            this_xyz[(xi + 3) * nelements],
            // Y-coordinates
            this_xyz[(yi + 0) * nelements],
            this_xyz[(yi + 1) * nelements],
            this_xyz[(yi + 2) * nelements],
            this_xyz[(yi + 3) * nelements],
            // Z-coordinates
            this_xyz[(zi + 0) * nelements],
            this_xyz[(zi + 1) * nelements],
            this_xyz[(zi + 2) * nelements],
            this_xyz[(zi + 3) * nelements],
            nelements,
            this_fff);
    }
}

__global__ void laplacian_assemble_hessian_kernel(const ptrdiff_t nelements,
                                                  const real_t *const SFEM_RESTRICT fff,
                                                  real_t *const SFEM_RESTRICT values) {
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        laplacian(&fff[e], nelements, &values[e]);
    }
}

__global__ void local_to_global_kernel(const ptrdiff_t nelements,
                                       idx_t **const SFEM_RESTRICT elems,
                                       const real_t *const SFEM_RESTRICT element_matrix,
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

        // offsetted array for this element
        const real_t *const this_matrix = &element_matrix[e];

        // printf("%d)\n", (int)e);

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            const idx_t *const row = &colidx[rowptr[dof_i]];

            find_cols4(ev, row, lenrow, ks);

            real_t *const rowvalues = &values[rowptr[dof_i]];

            // #pragma unroll(4)
            for (int edof_j = 0; edof_j < 4; ++edof_j) {
                ptrdiff_t idx = (edof_i * 4 + edof_j) * nelements;
                const real_t v = this_matrix[idx];
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

    SFEM_RANGE_PUSH("lapl-set-up");
    cudaEventRecord(start);

    // static int block_size = 256;
    static int block_size = 128;
    const ptrdiff_t nbatch = MIN(block_size * 500, nelements);

    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (nbatch + block_size - 1) / block_size);

    geom_t *he_xyz = nullptr;
    SFEM_CUDA_CHECK(cudaMallocHost(&he_xyz, 3 * 4 * nbatch * sizeof(geom_t)));
    geom_t *de_xyz = nullptr;
    SFEM_CUDA_CHECK(cudaMalloc(&de_xyz, 3 * 4 * nbatch * sizeof(geom_t)));

    real_t *d_fff = nullptr;
    SFEM_CUDA_CHECK(cudaMalloc(&d_fff, 6 * nbatch * sizeof(real_t)));

    real_t *he_matrix = nullptr;
    cudaMallocHost(&he_matrix, 4 * 4 * nbatch * sizeof(real_t));
    real_t *de_matrix = nullptr;
    SFEM_CUDA_CHECK(cudaMalloc(&de_matrix, 4 * 4 * nbatch * sizeof(real_t)));

    idx_t *hh_elems[4];
    for (int d = 0; d < 4; d++) {
        SFEM_CUDA_CHECK(cudaMallocHost(&hh_elems[d], nbatch * sizeof(idx_t)));
    }

    idx_t **hd_elems[4];
    idx_t **d_elems = nullptr;

    count_t *d_rowptr = nullptr;
    idx_t *d_colidx = nullptr;
    real_t *d_values = nullptr;

    static const int nstreams = 4;
    cudaStream_t stream[nstreams];
    cudaEvent_t event[nstreams];
    for (int s = 0; s < nstreams; s++) {
        cudaStreamCreate(&stream[s]);
        cudaEventCreate(&event[s]);
    }

    // Allocate space for indices
    for (int d = 0; d < 4; d++) {
        SFEM_CUDA_CHECK(cudaMalloc(&hd_elems[d], nbatch * sizeof(idx_t)));
    }

    SFEM_CUDA_CHECK(cudaMalloc(&d_elems, 4 * sizeof(idx_t *)));
    cudaMemcpy(d_elems, hd_elems, 4 * sizeof(idx_t *), cudaMemcpyHostToDevice);

    SFEM_RANGE_POP();

    SFEM_RANGE_PUSH("lapl-crs-host-to-device");
    // Copy crs-matrix
    crs_device_create(nnodes, rowptr[nnodes], &d_rowptr, &d_colidx, &d_values);
    crs_graph_host_to_device(nnodes, rowptr[nnodes], rowptr, colidx, d_rowptr, d_colidx);

    SFEM_RANGE_POP();

    // TODO CRS HtoD async

    ptrdiff_t last_n = 0;
    ptrdiff_t last_element_offset = 0;
    for (ptrdiff_t element_offset = 0; element_offset < nelements; element_offset += nbatch) {
        ptrdiff_t n = MIN(nbatch, nelements - element_offset);


        /////////////////////////////////////////////////////////
        // Packing (stream 0)
        /////////////////////////////////////////////////////////

        if (last_n) {
            cudaStreamSynchronize(stream[0]);
        }

        {
            SFEM_RANGE_PUSH("lapl-packing");
            {
                for (int d = 0; d < 3; ++d) {
                    for (int e_node = 0; e_node < 4; e_node++) {
                        const geom_t *const x = xyz[d];
                        ptrdiff_t offset = (d * 4 + e_node) * n;
                        const idx_t *const nodes = &elems[e_node][element_offset];

                        geom_t *buff = &he_xyz[offset];

#pragma omp parallel
                        {
#pragma omp for nowait
                            for (ptrdiff_t k = 0; k < n; k++) {
                                buff[k] = x[nodes[k]];
                            }
                        }
                    }
                }
            }

            SFEM_RANGE_POP();
        }


        /////////////////////////////////////////////////////////
        // Local to global (stream 3)
        /////////////////////////////////////////////////////////

        if (last_n) {
            // Make sure we have the elemental matrices and dof indices
            cudaStreamWaitEvent(stream[3], event[1]);
            cudaStreamWaitEvent(stream[3], event[2]);

            // Do this here to let the main kernel overlap with the packing
            local_to_global_kernel<<<n_blocks, block_size, 0, stream[3]>>>(
                last_n, d_elems, de_matrix, d_rowptr, d_colidx, d_values);

            cudaEventRecord(event[3], stream[3]);

            SFEM_DEBUG_SYNCHRONIZE();
        }

        /////////////////////////////////////////////////////////
        // XYZ HtoD (stream 0)
        /////////////////////////////////////////////////////////

        SFEM_CUDA_CHECK(cudaMemcpyAsync(de_xyz, he_xyz, 3 * 4 * n * sizeof(geom_t), cudaMemcpyHostToDevice, stream[0]));
        cudaEventRecord(event[0], stream[0]);

        SFEM_DEBUG_SYNCHRONIZE();
        /////////////////////////////////////////////////////////
        // Jacobian computations (stream 1)
        /////////////////////////////////////////////////////////

        // Make sure we have the new XYZ coordinates
        cudaStreamWaitEvent(stream[1], event[0]);

        fff_kernel<<<n_blocks, block_size, 0, stream[1]>>>(n, de_xyz, d_fff);

        SFEM_DEBUG_SYNCHRONIZE();
        /////////////////////////////////////////////////////////
        // DOF indices HtoD (stream 2)
        /////////////////////////////////////////////////////////

        // Ensure that previous HtoD is completed
        if(last_n) cudaStreamSynchronize(stream[2]);

        SFEM_RANGE_PUSH("lapl-copy-host-to-host");
        //  Copy elements to host-pinned memory
        for (int e_node = 0; e_node < 4; e_node++) {
            memcpy(hh_elems[e_node], &elems[e_node][element_offset], n * sizeof(idx_t));
        }

        SFEM_RANGE_POP();

        // Make sure local to global has ended
        cudaStreamWaitEvent(stream[2], event[3]);

        for (int e_node = 0; e_node < 4; e_node++) {
            SFEM_CUDA_CHECK(cudaMemcpyAsync(
                hd_elems[e_node], hh_elems[e_node], n * sizeof(idx_t), cudaMemcpyHostToDevice, stream[2]));
        }

        cudaEventRecord(event[2], stream[2]);

        SFEM_DEBUG_SYNCHRONIZE();
        /////////////////////////////////////////////////////////
        // Assemble elemental matrices (stream 1)
        /////////////////////////////////////////////////////////

        // Make sure that we have new Jacobians
        cudaStreamWaitEvent(stream[1], event[3]);

        laplacian_assemble_hessian_kernel<<<n_blocks, block_size, 0, stream[1]>>>(n, d_fff, de_matrix);
        cudaEventRecord(event[1], stream[1]);

        SFEM_DEBUG_SYNCHRONIZE();
        /////////////////////////////////////////////////////////

        last_n = n;
        last_element_offset = element_offset;
    }

    /////////////////////////////////////////////////////////
    // Local to global (stream 3)
    /////////////////////////////////////////////////////////

    if (last_n) {

        // Make sure we have the elemental matrices and dof indices
        cudaStreamWaitEvent(stream[3], event[1]);
        cudaStreamWaitEvent(stream[3], event[2]);

        // Do this here to let the main kernel overlap with the packing
        local_to_global_kernel<<<n_blocks, block_size, 0, stream[3]>>>(
            last_n, d_elems, de_matrix, d_rowptr, d_colidx, d_values);

        SFEM_DEBUG_SYNCHRONIZE();

        cudaStreamSynchronize(stream[3]);
    }

    /////////////////////////////////////////////////////////

    SFEM_RANGE_PUSH("lapl-values-device-to-host");

    SFEM_CUDA_CHECK(cudaMemcpy(values, d_values, rowptr[nnodes] * sizeof(real_t), cudaMemcpyDeviceToHost));

    SFEM_RANGE_POP();

    SFEM_RANGE_PUSH("lapl-tear-down");
    {  // Free resources on CPU
        cudaFreeHost(he_xyz);
        cudaFreeHost(he_matrix);

        for (int d = 0; d < 4; d++) {
            SFEM_CUDA_CHECK(cudaFreeHost(hh_elems[d]));
        }
    }

    {  // Free resources on GPU
        SFEM_CUDA_CHECK(cudaFree(de_xyz));
        SFEM_CUDA_CHECK(cudaFree(de_matrix));
        SFEM_CUDA_CHECK(cudaFree(d_fff));

        for (int d = 0; d < 4; d++) {
            SFEM_CUDA_CHECK(cudaFree(hd_elems[d]));
        }
        SFEM_CUDA_CHECK(cudaFree(d_elems));

        crs_device_free(d_rowptr, d_colidx, d_values);

        for (int s = 0; s < nstreams; s++) {
            cudaStreamDestroy(stream[s]);
            cudaEventDestroy(event[s]);
        }
    }

    SFEM_RANGE_POP();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("cuda_laplacian_3.c: laplacian_assemble_hessian\t%g seconds\nloops %d\n",
           milliseconds / 1000,
           int(nelements / nbatch));
}
