#pragma once

#include <cstddef>
#include <cstdlib>

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#ifndef SFEM_INLINE
#define SFEM_INLINE inline
#endif

namespace sfem {
namespace codegen {

template <typename T>
struct ThreadScratchBuffer {
    T *data{nullptr};
    size_t capacity{0};

    ~ThreadScratchBuffer() { std::free(data); }

    T *ensure(const size_t size) {
        if (capacity < size) {
            std::free(data);
            data = static_cast<T *>(std::calloc(size, sizeof(T)));
            capacity = data ? size : 0;
        }
        return data;
    }
};

template <typename T>
SFEM_INLINE T *thread_scratch(const int slot, const size_t size) {
    static thread_local ThreadScratchBuffer<T> buffers[4];
    return buffers[slot].ensure(size);
}

}  // namespace codegen
}  // namespace sfem
