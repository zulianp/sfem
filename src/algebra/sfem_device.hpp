#ifndef SFEM_DEVICE_HPP
#define SFEM_DEVICE_HPP

#include "sfem_base.hpp"

// Device queries that compile in every build: the CUDA ones, or host stand-ins.
#ifdef SFEM_ENABLE_CUDA
#include "sfem_BLAS_cuda.hpp"
#else
namespace sfem {
    static inline void device_synchronize() {}
    static inline bool is_ptr_device(const void *) { return false; }
    template <typename T>
    inline T &to_host(T &ptr) {
        return ptr;
    }
}  // namespace sfem
#endif

#endif  // SFEM_DEVICE_HPP
