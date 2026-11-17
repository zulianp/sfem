#ifndef SFEM_HIP_COMPAT_CUDA_H
#define SFEM_HIP_COMPAT_CUDA_H

// HIP build only (base/hip/compat is on the include path with SFEM_ENABLE_HIP): <cuda.h> is the HIP
// runtime with SFEM's cuda* names mapped onto it.
#include "sfem_gpu_runtime.hpp"

#endif  // SFEM_HIP_COMPAT_CUDA_H
