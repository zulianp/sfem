#ifndef SFEM_EXECUTION_CONTEXT_HPP
#define SFEM_EXECUTION_CONTEXT_HPP

#include "smesh_spaces.hpp"

namespace sfem {

    /// The stream device work goes to when nothing names another: the calling thread's default
    /// stream (cudaStreamPerThread). The library is built with the null stream meaning that one
    /// (`--default-stream per-thread`), so a launch, copy or thrust call without a stream joins it,
    /// and unlike the legacy stream it can be captured into a graph. Null without CUDA.
    void* default_device_stream();

    /// Where work runs and what orders it: an execution space and, on the device, the stream that
    /// orders the work (a cudaStream_t, held without CUDA types; null asks for the default device
    /// stream, which is what the context then holds). An object is given its context when it is
    /// built, not on each call.
    ///
    /// A bare ExecutionSpace converts to the context of that space on its default stream.
    class ExecutionContext {
    public:
        ExecutionContext(const smesh::ExecutionSpace space = smesh::EXECUTION_SPACE_HOST, void* const stream = nullptr)
            : space_(space),
              stream_(space == smesh::EXECUTION_SPACE_DEVICE ? (stream ? stream : default_device_stream()) : nullptr) {}

        static ExecutionContext host() { return ExecutionContext(smesh::EXECUTION_SPACE_HOST); }
        static ExecutionContext device(void* const stream = nullptr) { return ExecutionContext(smesh::EXECUTION_SPACE_DEVICE, stream); }

        smesh::ExecutionSpace space() const { return space_; }
        void*                 stream() const { return stream_; }

        bool operator==(const ExecutionContext& other) const { return space_ == other.space_ && stream_ == other.stream_; }
        bool operator!=(const ExecutionContext& other) const { return !(*this == other); }

    private:
        smesh::ExecutionSpace space_;
        void*                 stream_;
    };

    /// The work issued to `waiting` from now on runs after the work issued to `on` so far (an event
    /// between the two streams; inside a graph capture an edge). Nothing where both are one stream
    /// or either is not on the device.
    void order_after(const ExecutionContext& waiting, const ExecutionContext& on);

}  // namespace sfem

#endif  // SFEM_EXECUTION_CONTEXT_HPP
