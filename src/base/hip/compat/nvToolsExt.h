#ifndef SFEM_HIP_COMPAT_NVTOOLSEXT_H
#define SFEM_HIP_COMPAT_NVTOOLSEXT_H

// HIP build only (base/hip/compat is on the include path with SFEM_ENABLE_HIP): NVTX ranges are
// NVIDIA's profiler annotations, and on AMD GPUs they mark nothing.
inline int nvtxRangePushA(const char*) { return 0; }
inline int nvtxRangePop() { return 0; }

#endif  // SFEM_HIP_COMPAT_NVTOOLSEXT_H
