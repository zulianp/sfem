// CUDA smoke test for the CVFEM HEX8 kernels.
//
// The point of this file is to prove, before any real kernel design depends on it,
// that the templated CVFEM element kernels compile and run as __device__ code and
// produce the same numbers as the host. It is the first executable check of the whole
// Phase 1 premise.

#include <cstdio>

#include <cuda_runtime.h>

// The kernel header is not self-contained: it needs scalar_t and SFEM_RESTRICT in
// scope, exactly as cvfem_hex8_layout_common.hpp provides them on the host.
using scalar_t = double;
#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#include "cvfem_hex8_ns_upwind_kernels.hpp"

#include "cvfem_cuda_smoke.hpp"

#define CVFEM_CUDA_CHECK(expr)                                                        \
    do {                                                                              \
        cudaError_t _e = (expr);                                                      \
        if (_e != cudaSuccess) {                                                      \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                   \
                         cudaGetErrorString(_e));                                     \
            return 1;                                                                 \
        }                                                                             \
    } while (0)

namespace {

// One thread per element. This is the mapping the real kernels will use: the lane
// dimension of the host SIMD packs becomes threadIdx.x, which is why the scalar entry
// points -- not the _simd family -- are the ones that were made device-callable.
template <typename T>
__global__ void smoke_residual_kernel(const size_t nelements, const T rho, const T mu,
                                      const T *__restrict__ adj, const T det,
                                      const T *__restrict__ ux, const T *__restrict__ uy,
                                      const T *__restrict__ uz, const T *__restrict__ p,
                                      T *__restrict__ r_out) {
    const size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (e >= nelements) return;

    T adj_e[9];
    for (int i = 0; i < 9; ++i) adj_e[i] = adj[i];

    T r[CVFEM_HEX8_N_DOF];
    cvfem_hex8_ns_upwind_residual(rho, mu, adj_e, det, ux, uy, uz, p, r);

    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r_out[e * CVFEM_HEX8_N_DOF + i] = r[i];
}

template <typename T>
int run_residual(size_t nelements, T rho, T mu, const T *adj, T det, const T *ux,
                 const T *uy, const T *uz, const T *p, T *r_out, void *stream) {
    cudaStream_t s = stream ? *static_cast<cudaStream_t *>(stream) : cudaStream_t(0);

    T *d_adj = nullptr, *d_ux = nullptr, *d_uy = nullptr, *d_uz = nullptr;
    T *d_p = nullptr, *d_r = nullptr;
    const size_t n8 = 8 * sizeof(T);

    CVFEM_CUDA_CHECK(cudaMalloc(&d_adj, 9 * sizeof(T)));
    CVFEM_CUDA_CHECK(cudaMalloc(&d_ux, n8));
    CVFEM_CUDA_CHECK(cudaMalloc(&d_uy, n8));
    CVFEM_CUDA_CHECK(cudaMalloc(&d_uz, n8));
    CVFEM_CUDA_CHECK(cudaMalloc(&d_p, n8));
    CVFEM_CUDA_CHECK(cudaMalloc(&d_r, nelements * CVFEM_HEX8_N_DOF * sizeof(T)));

    CVFEM_CUDA_CHECK(cudaMemcpyAsync(d_adj, adj, 9 * sizeof(T), cudaMemcpyHostToDevice, s));
    CVFEM_CUDA_CHECK(cudaMemcpyAsync(d_ux, ux, n8, cudaMemcpyHostToDevice, s));
    CVFEM_CUDA_CHECK(cudaMemcpyAsync(d_uy, uy, n8, cudaMemcpyHostToDevice, s));
    CVFEM_CUDA_CHECK(cudaMemcpyAsync(d_uz, uz, n8, cudaMemcpyHostToDevice, s));
    CVFEM_CUDA_CHECK(cudaMemcpyAsync(d_p, p, n8, cudaMemcpyHostToDevice, s));

    const int block = 128;
    const int grid  = (int)((nelements + block - 1) / block);
    smoke_residual_kernel<T><<<grid, block, 0, s>>>(nelements, rho, mu, d_adj, det,
                                                    d_ux, d_uy, d_uz, d_p, d_r);
    CVFEM_CUDA_CHECK(cudaGetLastError());
    CVFEM_CUDA_CHECK(cudaMemcpyAsync(r_out, d_r,
                                     nelements * CVFEM_HEX8_N_DOF * sizeof(T),
                                     cudaMemcpyDeviceToHost, s));
    CVFEM_CUDA_CHECK(cudaStreamSynchronize(s));

    cudaFree(d_adj); cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_uz);
    cudaFree(d_p);   cudaFree(d_r);
    return 0;
}

}  // namespace

extern "C" int cvfem_cuda_smoke_device_info(int *sm_count, int *max_shmem_per_block,
                                            int *max_optin_shmem_per_block,
                                            int *warp_size) {
    int dev = 0;
    CVFEM_CUDA_CHECK(cudaGetDevice(&dev));
    CVFEM_CUDA_CHECK(cudaDeviceGetAttribute(sm_count, cudaDevAttrMultiProcessorCount, dev));
    CVFEM_CUDA_CHECK(cudaDeviceGetAttribute(max_shmem_per_block,
                                            cudaDevAttrMaxSharedMemoryPerBlock, dev));
    CVFEM_CUDA_CHECK(cudaDeviceGetAttribute(max_optin_shmem_per_block,
                                            cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    CVFEM_CUDA_CHECK(cudaDeviceGetAttribute(warp_size, cudaDevAttrWarpSize, dev));
    return 0;
}

extern "C" int cvfem_cuda_smoke_residual(size_t nelements, double rho, double mu,
                                         const double *adj, double det, const double *ux,
                                         const double *uy, const double *uz,
                                         const double *p, double *r_out, void *stream) {
    return run_residual<double>(nelements, rho, mu, adj, det, ux, uy, uz, p, r_out, stream);
}

extern "C" int cvfem_cuda_smoke_residual_f32(size_t nelements, float rho, float mu,
                                             const float *adj, float det, const float *ux,
                                             const float *uy, const float *uz,
                                             const float *p, float *r_out, void *stream) {
    return run_residual<float>(nelements, rho, mu, adj, det, ux, uy, uz, p, r_out, stream);
}
