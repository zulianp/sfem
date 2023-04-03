#ifndef SFEM_CUDA_BASE_H
#define SFEM_CUDA_BASE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdio.h>

inline void sfem_cuda_check(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda_check: %s %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define SFEM_CUDA_CHECK(ans) \
    { sfem_cuda_check((ans), __FILE__, __LINE__); }

#ifndef NDEBUG
#define SFEM_DEBUG_SYNCHRONIZE()                \
    do {                                        \
        cudaDeviceSynchronize();                \
        SFEM_CUDA_CHECK(cudaPeekAtLastError()); \
    } while (0)
#else
#define SFEM_DEBUG_SYNCHRONIZE()
#endif

#ifdef SFEM_ENABLE_NVTX
#include "nvToolsExt.h"
namespace sfem {
    namespace details {
        class Tracer {
        public:
            Tracer(const char* name) { nvtxRangePushA(name); }
            ~Tracer() { nvtxRangePop(); }
        };
    }  // namespace details
}  // namespace sfem

#define SFEM_NVTX_SCOPE(name) sfem::details::Tracer uniq_name_using_macros(name);

#else

#define SFEM_NVTX_SCOPE(name)

#endif

#endif  // SFEM_CUDA_BASE_H
