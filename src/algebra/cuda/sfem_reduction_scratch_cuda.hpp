#ifndef SFEM_CUDA_REDUCTION_SCRATCH_HPP
#define SFEM_CUDA_REDUCTION_SCRATCH_HPP

#include <algorithm>
#include <cstddef>

#include "sfem_cuda_base.hpp"

namespace sfem {

    /// Where a device reduction leaves its partial results and reads them back: a device array and
    /// a pinned host array of T.
    template <typename T>
    struct ReductionScratch {
        T* device;
        T* host;
    };

    /// The calling thread's reduction scratch, at least `n` entries (and at least 1024, the most
    /// block partials a reduction here produces), allocated at the first reduction that needs it
    /// and reused by every later one, so a reduction allocates nothing in steady state (each
    /// allocation also synchronises the device). A reduction completes before it returns, so one
    /// scratch per thread serves them all. Kept for the life of the process: freeing at thread exit
    /// can follow the driver's own shutdown.
    template <typename T>
    inline ReductionScratch<T> reduction_scratch(const std::ptrdiff_t n) {
        struct Slot {
            T*             device{nullptr};
            T*             host{nullptr};
            std::ptrdiff_t size{0};
        };
        thread_local Slot slot;
        if (slot.size < n) {
            if (slot.device) {
                SFEM_CUDA_CHECK(cudaFree(slot.device));
                SFEM_CUDA_CHECK(cudaFreeHost(slot.host));
            }
            const std::ptrdiff_t size = std::max<std::ptrdiff_t>(n, 1024);
            SFEM_CUDA_CHECK(cudaMalloc((void**)&slot.device, size * sizeof(T)));
            SFEM_CUDA_CHECK(cudaMallocHost((void**)&slot.host, size * sizeof(T)));
            slot.size = size;
        }
        return {slot.device, slot.host};
    }

}  // namespace sfem

#endif  // SFEM_CUDA_REDUCTION_SCRATCH_HPP
