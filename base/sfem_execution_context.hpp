#ifndef SFEM_EXECUTION_CONTEXT_HPP
#define SFEM_EXECUTION_CONTEXT_HPP

#include "smesh_spaces.hpp"

namespace sfem {

    /// Where work runs and what orders it: an execution space and, on the device, the stream that
    /// orders the work (a cudaStream_t, held without CUDA types; null is the legacy default stream).
    /// An object is given its context when it is built, not on each call.
    ///
    /// A bare ExecutionSpace converts to the context of that space on its default stream.
    class ExecutionContext {
    public:
        ExecutionContext(const smesh::ExecutionSpace space = smesh::EXECUTION_SPACE_HOST, void* const stream = nullptr)
            : space_(space), stream_(space == smesh::EXECUTION_SPACE_DEVICE ? stream : nullptr) {}

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

}  // namespace sfem

#endif  // SFEM_EXECUTION_CONTEXT_HPP
