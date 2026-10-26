#include "sfem_execution_context.hpp"

#include "sfem_config.h"

#ifdef SFEM_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace sfem {

    void* default_device_stream() {
#ifdef SFEM_ENABLE_CUDA
        return (void*)cudaStreamPerThread;
#else
        return nullptr;
#endif
    }

    void order_after(const ExecutionContext& waiting, const ExecutionContext& on) {
        if (waiting.space() != smesh::EXECUTION_SPACE_DEVICE || on.space() != smesh::EXECUTION_SPACE_DEVICE ||
            waiting.stream() == on.stream()) {
            return;
        }
#ifdef SFEM_ENABLE_CUDA
        // One event per thread: a wait takes the event as recorded when it is enqueued, so the
        // next record can reuse it.
        static thread_local cudaEvent_t event = [] {
            cudaEvent_t e = nullptr;
            cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
            return e;
        }();
        cudaEventRecord(event, static_cast<cudaStream_t>(on.stream()));
        cudaStreamWaitEvent(static_cast<cudaStream_t>(waiting.stream()), event, 0);
#endif
    }

}  // namespace sfem
