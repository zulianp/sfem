#ifndef CVFEM_CUDA_SMOKE_HPP
#define CVFEM_CUDA_SMOKE_HPP

// C ABI for the CUDA smoke test.
//
// Deliberately nvcc-free so the .cpp driver never needs the CUDA compiler, mirroring
// bench/cuda/bench_packed_laplacian_cuda.hpp. The conventions established here are the
// ones the real kernels will follow: extern "C", int return codes, `void *stream` last
// (a cudaStream_t in disguise, so this header stays free of CUDA types).

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device properties, so the host can size shared memory without including cuda_runtime.
int cvfem_cuda_smoke_device_info(int *sm_count, int *max_shmem_per_block,
                                 int *max_optin_shmem_per_block, int *warp_size);

// Runs cvfem_hex8_ns_upwind_residual<double> on the device for `nelements` copies of a
// single element and writes the per-element residual (CVFEM_HEX8_N_DOF each) to `r_out`.
// Host arrays in, host array out; the wrapper owns all device allocation.
int cvfem_cuda_smoke_residual(size_t nelements, double rho, double mu,
                              const double *adj, double det,
                              const double *ux, const double *uy, const double *uz,
                              const double *p, double *r_out, void *stream);

// Same, instantiated on float, to prove the kernels really are scalar-type generic on
// the device and not only on the host.
int cvfem_cuda_smoke_residual_f32(size_t nelements, float rho, float mu,
                                  const float *adj, float det,
                                  const float *ux, const float *uy, const float *uz,
                                  const float *p, float *r_out, void *stream);

#ifdef __cplusplus
}
#endif

#endif  // CVFEM_CUDA_SMOKE_HPP
