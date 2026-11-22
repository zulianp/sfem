#ifndef SFEM_CHIPSTAR_SPT_H
#define SFEM_CHIPSTAR_SPT_H

// chipStar's per-thread default stream, completed. Force-included into every HIP source of a chipStar
// build (CMAKE_HIP_PLATFORM=spirv, cmake/SFEMDependencies.cmake), smesh's and the other subprojects'
// included.
//
// With -fgpu-default-stream=per-thread clang lowers a launch to hipLaunchKernel_spt, and
// HIP_API_PER_THREAD_DEFAULT_STREAM has hip/spirv_spt.h map the calls that take no stream (hipMemcpy,
// hipMemcpyToSymbol, ...) onto their _spt versions; but no chipStar header includes spirv_spt.h, and its
// _spt declarations leave out the default arguments of the calls they replace. Both are supplied here:
// the header, and the defaults HIP declares (hip/hip_runtime_api.h), added by redeclaration.

#include <hip/hip_runtime.h>
#include <hip/spirv_spt.h>

#ifdef __cplusplus
extern "C" {
hipError_t hipMemcpyToSymbol_spt(const void *symbol, const void *src, size_t sizeBytes, size_t offset = 0,
                                 hipMemcpyKind kind = hipMemcpyHostToDevice);
hipError_t hipMemcpyFromSymbol_spt(void *dst, const void *symbol, size_t sizeBytes, size_t offset = 0,
                                   hipMemcpyKind kind = hipMemcpyDeviceToHost);
hipError_t hipMemcpyAsync_spt(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream = 0);
hipError_t hipStreamWaitEvent_spt(hipStream_t stream, hipEvent_t event, unsigned int flags = 0);
hipError_t hipEventRecord_spt(hipEvent_t event, hipStream_t stream = 0);
}
#endif

#endif  // SFEM_CHIPSTAR_SPT_H
