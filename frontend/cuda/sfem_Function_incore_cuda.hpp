#ifndef SFEM_FUNCTION_INCORE_CUDA_HPP
#define SFEM_FUNCTION_INCORE_CUDA_HPP

#include <memory>
#include "sfem_Function.hpp"

#include "sfem_cuda_blas.h"

// #include <cuda_runtime.h>

namespace sfem {
    void register_device_ops();
    std::shared_ptr<Constraint> to_device(const std::shared_ptr<DirichletConditions> &dc);

    std::shared_ptr<Buffer<idx_t>> create_device_elements(
            const std::shared_ptr<FunctionSpace> &space,
            const enum ElemType element_type);

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

        T *buff = (T*)d_buffer_alloc(in->size() * sizeof(T));
        // cudaMemcpy(buff, in->data(), in->size() * sizeof(T), cudaMemcpyHostToDevice);
        buffer_host_to_device(in->size() * sizeof(T), in->data(), buff);

        return
            std::make_shared<Buffer<T>>(in->size(), buff, &d_buffer_destroy, MEMORY_SPACE_DEVICE);
    }


    inline std::shared_ptr<CRSGraph> to_device(const std::shared_ptr<CRSGraph> &in) {
        if(in->rowptr()->mem_space() == MEMORY_SPACE_DEVICE) {
            return in;
        }

        return
            std::make_shared<CRSGraph>(to_device(in->rowptr()), to_device(in->colidx()));
    }


    template <typename T>
    std::shared_ptr<Buffer<T>> to_host(const std::shared_ptr<Buffer<T>> &in) {
    	if(in->mem_space() == MEMORY_SPACE_HOST) {
    		return in;
    	}

        T *buff = static_cast<T*>(malloc(in->size() * sizeof(T)));
        // cudaMemcpy(buff, in->data(), in->size() * sizeof(T), cudaMemcpyDeviceToHost);
        buffer_device_to_host(in->size() * sizeof(T), in->data(), buff);
        return
            std::make_shared<Buffer<T>>(in->size(), buff, &free, MEMORY_SPACE_HOST);
    }

}  // namespace sfem

#endif  // SFEM_FUNCTION_INCORE_CUDA_HPP
