#ifndef SFEM_CUDA_BASE_H
#define SFEM_CUDA_BASE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <stdio.h>
#include <assert.h>

inline void sfem_cuda_check(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda_check: %s %s:%d\n", cudaGetErrorString(code), file, line);
        assert(!code);
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

#define SFEM_ENABLE_NVTX

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
#define SFEM_RANGE_PUSH(name_) \
    do {                       \
        nvtxRangePushA(name_); \
    } while (0)
#define SFEM_RANGE_POP() \
    do {                 \
        nvtxRangePop();  \
    } while (0)

// Launch bounds configuration for CUDA kernels
#define SFEM_LAUNCH_BOUNDS(threads_per_block, min_blocks_per_sm) \
    __launch_bounds__(threads_per_block, min_blocks_per_sm)

#else //SFEM_ENABLE_NVTX

#define SFEM_NVTX_SCOPE(name)
#define SFEM_RANGE_PUSH(name_)
#define SFEM_RANGE_POP()
 
#endif //SFEM_ENABLE_NVTX

#endif  // SFEM_CUDA_BASE_H
