#ifndef SFEM_CUDA_BASE_H
#define SFEM_CUDA_BASE_H

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <assert.h>
#include <stdio.h>

inline void sfem_cuda_check(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda_check: %s %s:%d\n", cudaGetErrorString(code), file, line);
        assert(!code);
        if (abort) exit(code);
    }
}

#define SFEM_CUDA_CHECK(ans)                        \
    {                                               \
        sfem_cuda_check((ans), __FILE__, __LINE__); \
    }

/** Lazily caches cudaDevAttrMaxGridDimX for the current device (one query per process). */
inline int sfem_cuda_max_grid_dim_x() {
    static int max_grid_x = -1;
    if (max_grid_x < 0) {
        int device_id = 0;
        cudaGetDevice(&device_id);
        cudaDeviceGetAttribute(&max_grid_x, cudaDevAttrMaxGridDimX, device_id);
    }
    return max_grid_x;
}

namespace sfem {
    /// Waits for `stream` (a cudaStream_t; null: the default device stream) and reports what failed
    /// on it, unless it is being captured into a graph, which a synchronisation would invalidate.
    inline void debug_synchronize(const cudaStream_t stream) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
            cudaGetLastError();
            return;
        }
        if (status != cudaStreamCaptureStatusNone) return;
        SFEM_CUDA_CHECK(cudaStreamSynchronize(stream));
        SFEM_CUDA_CHECK(cudaPeekAtLastError());
    }
}  // namespace sfem

#ifndef NDEBUG
#define SFEM_DEBUG_SYNCHRONIZE(stream) sfem::debug_synchronize((cudaStream_t)(stream))
#else
#define SFEM_DEBUG_SYNCHRONIZE(stream)
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
#define SFEM_LAUNCH_BOUNDS(threads_per_block, min_blocks_per_sm) __launch_bounds__(threads_per_block, min_blocks_per_sm)

#else  // SFEM_ENABLE_NVTX

#define SFEM_NVTX_SCOPE(name)
#define SFEM_RANGE_PUSH(name_)
#define SFEM_RANGE_POP()

#endif  // SFEM_ENABLE_NVTX

// A kernel's block size for HIP's register allocation. Without launch bounds HIP compiles a kernel
// for blocks of 1024 threads, which caps a thread at 128 VGPRs (CUDA allows 255), so kernels that fit
// on NVIDIA spill to scratch on AMD. Empty in CUDA builds. The kernel must not be launched with more
// threads a block.
#if defined(__HIP_PLATFORM_AMD__)
#define SFEM_HIP_LAUNCH_BOUNDS(threads_per_block) __launch_bounds__(threads_per_block)
#else
#define SFEM_HIP_LAUNCH_BOUNDS(threads_per_block)
#endif

#if defined(__HIP_PLATFORM_AMD__)
#include <cstring>
#include <initializer_list>

// AMD's FP64 matrix instruction (v_mfma_f64_16x16x4f64) exists on CDNA 2 and 3 (gfx90a, gfx94x,
// gfx950), all wave64; the kernels using it assume both. A device pass for another target compiles
// it to a trap, and sfem_hip_device_has_fp64_mfma() keeps the host from launching such a kernel there.
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
#define SFEM_HIP_FP64_MFMA_TARGET
#endif

typedef double sfem_hip_double4 __attribute__((ext_vector_type(4)));

#if defined(__HIPCC__)
__device__ __forceinline__ sfem_hip_double4 sfem_hip_mfma_f64_16x16x4(const double a, const double b, const sfem_hip_double4 c) {
#if defined(SFEM_HIP_FP64_MFMA_TARGET)
    return __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, 0);
#else
    __builtin_trap();
    return c;
#endif
}
#endif  // __HIPCC__

/** Whether the current device has the FP64 matrix instruction (its gcnArchName names one of the
 * targets above); cached per process, one GPU a rank. */
inline bool sfem_hip_device_has_fp64_mfma() {
    static const bool has = [] {
        int             device = 0;
        hipDeviceProp_t prop;
        if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&prop, device) != hipSuccess) return false;
        for (const char* arch : {"gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"}) {
            const size_t n = strlen(arch);
            if (strncmp(prop.gcnArchName, arch, n) == 0 && (prop.gcnArchName[n] == 0 || prop.gcnArchName[n] == ':')) return true;
        }
        return false;
    }();
    return has;
}
#endif  // __HIP_PLATFORM_AMD__

#endif  // SFEM_CUDA_BASE_H
