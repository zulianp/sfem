#ifndef SFEM_GPU_RUNTIME_HPP
#define SFEM_GPU_RUNTIME_HPP

// SFEM's device code under HIP, for AMD GPUs (SFEM_ENABLE_HIP).
//
// The device path is written against the CUDA runtime API, which HIP mirrors name for name. A HIP
// build puts base/hip/compat first on the include path: its cuda_runtime.h, cuda.h, cuda_fp16.h,
// ... include this header, which includes HIP and maps the cuda* names SFEM calls onto their hip*
// counterparts. The sources, generated ones included, compile unchanged; a CUDA build never sees
// this header.
//
// smesh's smesh_gpu_runtime.cuh maps the names smesh calls and is included first; only the names
// it leaves out are mapped here. SCCD, sresample and SSDF have headers like it, and a translation
// unit can include several: a macro may be redefined only with the same replacement, so a name
// mapped in more than one header must map to exactly the same tokens in each.

#include <hip/hip_runtime.h>

#include "smesh_gpu_runtime.cuh"

// The HIP runtime follows the CUDA 12 API: the branches written for older toolkits (pointer
// attributes' memoryType, cuSPARSE's pre-12 enums) are not taken.
#ifndef CUDART_VERSION
#define CUDART_VERSION 12000
#endif

#define cudaGetLastError hipGetLastError

#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMallocManaged hipMallocManaged
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemsetAsync hipMemsetAsync
// The symbol copies take the symbol's address: with the per-thread default stream
// (HIP_API_PER_THREAD_DEFAULT_STREAM) HIP's take nothing else, and without it they take it too.
#define cudaMemcpyToSymbol(symbol, ...) hipMemcpyToSymbol((const void *)&(symbol), __VA_ARGS__)
#define cudaMemcpyFromSymbol(dst, symbol, ...) hipMemcpyFromSymbol(dst, (const void *)&(symbol), __VA_ARGS__)
#define cudaMemAdvise hipMemAdvise
#define cudaMemAdviseSetReadMostly hipMemAdviseSetReadMostly
#define cudaMemAdviseSetPreferredLocation hipMemAdviseSetPreferredLocation
#define cudaMemAdviseSetAccessedBy hipMemAdviseSetAccessedBy
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#define cudaCpuDeviceId hipCpuDeviceId
#define cudaMemPool_t hipMemPool_t
#define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#define cudaMemPoolGetAttribute hipMemPoolGetAttribute
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#define cudaMemPoolAttrUsedMemHigh hipMemPoolAttrUsedMemHigh

// cuda.h's (driver API) integer types.
typedef unsigned long long cuuint64_t;
typedef unsigned int       cuuint32_t;

#define cudaStreamPerThread hipStreamPerThread
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaLaunchHostFunc hipLaunchHostFunc
// The calling convention of host callbacks, empty on Linux.
#define CUDART_CB

#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDestroy hipEventDestroy

#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
// CUDA 12's cudaGraphInstantiate(exec, graph, flags).
#define cudaGraphInstantiate hipGraphInstantiateWithFlags
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphDestroy hipGraphDestroy

#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrMaxGridDimX hipDeviceAttributeMaxGridDimX
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxSharedMemoryPerBlock hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
// CUDA's cudaFuncSetAttribute takes the kernel itself; HIP's takes its address as const void*.
template <typename Kernel>
inline hipError_t sfem_func_set_attribute(Kernel* const kernel, const hipFuncAttribute attribute, const int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attribute, value);
}
#define cudaFuncSetAttribute sfem_func_set_attribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel

#if defined(__HIPCC__)
// SFEM's warp is 32 lanes (SFEM_WARP_SIZE), and an AMD wavefront 64. The shuffles keep CUDA's
// semantics on a 32-lane warp: width 32 by default, so no lane reads across the boundary of its
// 32-lane half. The mask names lanes of a 32-lane warp, which run together in the wavefront, and
// is dropped; HIP's own *_sync builtins (HIP_ENABLE_WARP_SYNC_BUILTINS) take 64-bit masks and stay
// off. Barriers are smesh's wavefront-wide __syncwarp.
template <typename Mask, typename T>
__device__ inline T sfem_shfl_xor_sync(Mask, const T var, const int lane_mask, const int width = 32) {
    return __shfl_xor(var, lane_mask, width);
}

template <typename Mask, typename T>
__device__ inline T sfem_shfl_down_sync(Mask, const T var, const unsigned int delta, const int width = 32) {
    return __shfl_down(var, delta, width);
}

template <typename Mask, typename T>
__device__ inline T sfem_shfl_up_sync(Mask, const T var, const unsigned int delta, const int width = 32) {
    return __shfl_up(var, delta, width);
}

template <typename Mask, typename T>
__device__ inline T sfem_shfl_sync(Mask, const T var, const int src_lane, const int width = 32) {
    return __shfl(var, src_lane, width);
}

#define __shfl_xor_sync sfem_shfl_xor_sync
#define __shfl_down_sync sfem_shfl_down_sync
#define __shfl_up_sync sfem_shfl_up_sync
#define __shfl_sync sfem_shfl_sync

// __ldcg reads past the incoherent L1 what other blocks of the same kernel wrote (the last-block
// reductions): on AMD a relaxed load at device scope.
template <typename T>
__device__ inline T sfem_ldcg(const T* const p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

#define __ldcg sfem_ldcg
#endif

#endif  // SFEM_GPU_RUNTIME_HPP
