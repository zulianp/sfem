#ifndef SFEM_FUNCTION_INCORE_CUDA_HPP
#define SFEM_FUNCTION_INCORE_CUDA_HPP

#include <memory>
#include "sfem_Function.hpp"

#include "sfem_cuda_blas.h"

#include <cuda_runtime.h>

namespace sfem {
    void register_device_ops();
    std::shared_ptr<Constraint> to_device(const std::shared_ptr<DirichletConditions> &dc);

    template <typename T>
    std::shared_ptr<Buffer<T>> d_buffer(const std::ptrdiff_t n) {
        auto ret = std::make_shared<Buffer<T>>(
            n, (T *)d_buffer_alloc(n * sizeof(T)), &d_buffer_destroy, MEMORY_SPACE_DEVICE);
        return ret;
    }

    template <typename T>
    std::shared_ptr<Buffer<T>> to_device(const std::shared_ptr<Buffer<T>> &in) {
    	if(in->mem_space() == MEMORY_SPACE_DEVICE) {
    		return in;
    	}

        T *buff = d_buffer_alloc(in->size() * sizeof(T));
        cudaMemcpy(buff, in->data(), in->size() * sizeof(T), cudaMemcpyHostToDevice);

        return
            std::make_shared<Buffer<T>>(in->size(), buff, &d_buffer_destroy, MEMORY_SPACE_DEVICE);
    }

    template <typename T>
    std::shared_ptr<Buffer<T>> to_host(const std::shared_ptr<Buffer<T>> &in) {
    	if(in->mem_space() == MEMORY_SPACE_HOST) {
    		return in;
    	}

        T *buff = static_cast<T*>(malloc(in->size() * sizeof(T)));
        cudaMemcpy(buff, in->data(), in->size() * sizeof(T), cudaMemcpyDeviceToHost);
        return
            std::make_shared<Buffer<T>>(in->size(), buff, &free, MEMORY_SPACE_HOST);
    }

}  // namespace sfem

#endif  // SFEM_FUNCTION_INCORE_CUDA_HPP
