#ifndef SFEM_HIP_COMPAT_CUDA_FP16_H
#define SFEM_HIP_COMPAT_CUDA_FP16_H

// HIP build only (base/hip/compat is on the include path with SFEM_ENABLE_HIP): HIP's __half has
// CUDA's names and conversions.
#include <hip/hip_fp16.h>

#include "sfem_gpu_runtime.hpp"

#if defined(__HIPCC__)
// CUDA's __double2half rounds the double once, to nearest even; through float it would round twice.
__device__ inline __half __double2half(const double x) { return __half(static_cast<_Float16>(x)); }
#endif

#endif  // SFEM_HIP_COMPAT_CUDA_FP16_H
