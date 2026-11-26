#ifndef SFEM_GPU_RUNTIME_SPIRV_HPP
#define SFEM_GPU_RUNTIME_SPIRV_HPP

// SFEM's device code under chipStar: HIP compiled to SPIR-V and run through Level Zero, for Intel
// GPUs (CMAKE_HIP_PLATFORM=spirv; Aurora's PVC). Included by sfem_gpu_runtime.hpp, which maps the
// rest as for AMD. chipStar's warp is 32 lanes, SFEM_WARP_SIZE, so the width-32 shuffles there have
// CUDA's semantics unchanged. What chipStar leaves unimplemented and SFEM calls is answered here.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <mutex>
#include <utility>
#include <vector>

namespace sfem {
    namespace spirv {
        // The null stream is the calling thread's default stream (HIP_API_PER_THREAD_DEFAULT_STREAM).
        inline hipStream_t capture_key(const hipStream_t stream) { return stream ? stream : hipStreamPerThread; }

        /// The streams being captured, by any thread of the process. chipStar does not report a
        /// capture's status (hipStreamIsCapturing is not supported); SFEM begins and ends its captures
        /// through the two functions below and answers from this list. A capture begun around them
        /// (chipStar's calls made directly, or through SCCD's or smesh's runtime headers) is not in it
        /// and reads as none.
        inline std::vector<hipStream_t> &capturing_streams() {
            static std::vector<hipStream_t> streams;
            return streams;
        }

        inline std::mutex &capturing_streams_mutex() {
            static std::mutex mutex;
            return mutex;
        }

        /// chipStar captures a created stream, but capturing the per-thread default stream crashes it
        /// (chipStar 20260926, Aurora): refused here, so a caller takes its path without a graph. A
        /// graph does not run faster under chipStar, which replays the captured calls.
        inline hipError_t stream_begin_capture(const hipStream_t stream, const hipStreamCaptureMode mode) {
            if (capture_key(stream) == hipStreamPerThread) return hipErrorStreamCaptureUnsupported;
            const hipError_t err = hipStreamBeginCapture(stream, mode);
            if (err == hipSuccess) {
                std::lock_guard<std::mutex> lock(capturing_streams_mutex());
                capturing_streams().push_back(capture_key(stream));
            }
            return err;
        }

        // The capture ends whether or not a graph comes out of it.
        inline hipError_t stream_end_capture(const hipStream_t stream, hipGraph_t *const graph) {
            {
                std::lock_guard<std::mutex> lock(capturing_streams_mutex());
                auto      &streams = capturing_streams();
                const auto it      = std::find(streams.begin(), streams.end(), capture_key(stream));
                if (it != streams.end()) streams.erase(it);
            }
            return hipStreamEndCapture(stream, graph);
        }

        inline hipError_t stream_is_capturing(const hipStream_t stream, hipStreamCaptureStatus *const status) {
            std::lock_guard<std::mutex> lock(capturing_streams_mutex());
            const auto                 &streams = capturing_streams();
            *status = std::find(streams.begin(), streams.end(), capture_key(stream)) != streams.end() ? hipStreamCaptureStatusActive
                                                                                                    : hipStreamCaptureStatusNone;
            return hipSuccess;
        }

        /// The calling thread's capture mode, exchanged. chipStar does not implement it, and its
        /// failure would stay behind as the thread's last error, which the next launch's check
        /// takes for its own. chipStar captures in one mode only; the exchange keeps the value.
        inline hipError_t thread_exchange_stream_capture_mode(hipStreamCaptureMode *const mode) {
            thread_local hipStreamCaptureMode current = hipStreamCaptureModeGlobal;
            std::swap(current, *mode);
            return hipSuccess;
        }

        /// chipStar implements no memory pools. Answered here, unlike chipStar's own calls, without
        /// leaving the failure as the thread's last error (see above); the callers then do without.
        inline hipError_t device_get_default_mem_pool(hipMemPool_t *const, const int) { return hipErrorNotSupported; }

        template <typename Attribute>
        inline hipError_t mem_pool_attribute(hipMemPool_t, const Attribute, void *const) {
            return hipErrorNotSupported;
        }

        /// CUDA 12's cudaGraphInstantiate(exec, graph, flags). chipStar does not implement
        /// hipGraphInstantiateWithFlags; its hipGraphInstantiate takes the flags SFEM passes, none.
        inline hipError_t graph_instantiate(hipGraphExec_t *const exec, const hipGraph_t graph, const unsigned long long flags) {
            if (flags != 0) return hipErrorNotSupported;
            return hipGraphInstantiate(exec, graph, nullptr, nullptr, 0);
        }

        /// chipStar does not support hipFuncSetAttribute. The one attribute SFEM sets, the dynamic
        /// shared memory a launch may take, needs no opt-in there: any amount within the device's
        /// shared memory per block can be launched. The limit is the larger of the two a caller reads
        /// (base and opt-in, as frontend/ops/inexact/cuda/sfem_cuda_InexactSetup.cuh), so a request
        /// within what the device reports is accepted. chipStar (main, 2714ed9) leaves the opt-in at
        /// 0: it sets only sharedMemPerBlock, from Level Zero's maxSharedLocalMemory, and the callers
        /// then stay within the base limit and do not ask.
        template <typename Kernel>
        inline hipError_t func_set_attribute(Kernel *const, const hipFuncAttribute attribute, const int value) {
            if (attribute != hipFuncAttributeMaxDynamicSharedMemorySize) return hipErrorNotSupported;
            int device = 0, limit = 0, optin = 0;
            if (hipGetDevice(&device) != hipSuccess ||
                hipDeviceGetAttribute(&limit, hipDeviceAttributeMaxSharedMemoryPerBlock, device) != hipSuccess) {
                return hipErrorInvalidDevice;
            }
            if (hipDeviceGetAttribute(&optin, hipDeviceAttributeSharedMemPerBlockOptin, device) != hipSuccess) optin = 0;
            return value >= 0 && value <= std::max(limit, optin) ? hipSuccess : hipErrorInvalidValue;
        }
    }  // namespace spirv
}  // namespace sfem

#define cudaStreamIsCapturing sfem::spirv::stream_is_capturing
#define cudaStreamBeginCapture sfem::spirv::stream_begin_capture
#define cudaStreamEndCapture sfem::spirv::stream_end_capture
#define cudaFuncSetAttribute sfem::spirv::func_set_attribute
#define cudaGraphInstantiate sfem::spirv::graph_instantiate
#define cudaThreadExchangeStreamCaptureMode sfem::spirv::thread_exchange_stream_capture_mode
#define cudaDeviceGetDefaultMemPool sfem::spirv::device_get_default_mem_pool
#define cudaMemPoolGetAttribute sfem::spirv::mem_pool_attribute
#define cudaMemPoolSetAttribute sfem::spirv::mem_pool_attribute

#endif  // SFEM_GPU_RUNTIME_SPIRV_HPP
