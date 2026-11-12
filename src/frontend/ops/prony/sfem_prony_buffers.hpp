#pragma once

#include <cstddef>
#include <memory>

#include "sfem_aliases.hpp"

namespace sfem {

#ifdef SFEM_ENABLE_CUDA
    /// `bytes` zeroed bytes of device memory, stream-ordered in `stream`, and their return
    /// (cuda/sfem_prony_kernels_cuda.cu): neither synchronises the device.  `persistent` memory,
    /// kept across steps, comes from the default pool; the rest from a work pool of its own, which
    /// `cu_prony_pool_trim` empties.
    void *cu_prony_pool_allocate(const std::size_t bytes, void *const stream, const bool persistent);
    void  cu_prony_pool_release(void *const data, void *const stream);
    /// Wait for `stream`, then return the work pool's memory to the device, so that what a call
    /// took for its work is not held, unused, through the rest of the step.
    void  cu_prony_pool_trim(void *const stream);
    /// The work pool's high-water mark: the most a call held at once for its work.
    std::size_t cu_prony_work_pool_peak();
#endif

    namespace prony {

        /// After a call's transient buffers are dropped: on the device, return them to it
        /// (cu_prony_pool_trim); nothing on the host.
        inline void release_transients(const ExecutionSpace es, void *const stream) {
#ifdef SFEM_ENABLE_CUDA
            if (es == EXECUTION_SPACE_DEVICE) cu_prony_pool_trim(stream);
#else
            (void)es;
            (void)stream;
#endif
        }

        /// `n` zeroed values in `es`: on the device stream-ordered, returned in `stream` when the
        /// last reference is dropped, after the kernels already queued there; on the host an
        /// ordinary buffer.  `persistent`: kept across steps (a packed store's codes), else the
        /// work of one call, which `release_transients` returns to the device.
        template <typename T>
        std::shared_ptr<Buffer<T>> transient_buffer(const ptrdiff_t      n,
                                                    const ExecutionSpace es,
                                                    void *const          stream,
                                                    const bool           persistent = false) {
#ifdef SFEM_ENABLE_CUDA
            if (es == EXECUTION_SPACE_DEVICE) {
                T *const data =
                        static_cast<T *>(cu_prony_pool_allocate(static_cast<std::size_t>(n) * sizeof(T), stream, persistent));
                return std::make_shared<Buffer<T>>(
                        n, data, [stream](void *p) { cu_prony_pool_release(p, stream); }, MEMORY_SPACE_DEVICE);
            }
#else
            (void)stream;
            (void)persistent;
#endif
            return create_buffer<T>(n, es);
        }

    }  // namespace prony
}  // namespace sfem
