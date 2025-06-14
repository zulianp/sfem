#ifndef SFEM_FUNCTION_INCORE_CUDA_HPP
#define SFEM_FUNCTION_INCORE_CUDA_HPP

#include <memory>
#include "sfem_Function.hpp"

#include "sfem_cuda_blas.h"

// #include <cuda_runtime.h>

#include "sfem_Buffer.hpp"
#include "sfem_CRSGraph.hpp"

#include <memory>

namespace sfem {
    void                        register_device_ops();
    std::shared_ptr<Constraint> to_device(const std::shared_ptr<DirichletConditions> &dc);
    std::shared_ptr<Op>         to_device(const std::shared_ptr<NeumannConditions> &nc);
    std::shared_ptr<Sideset>    to_device(const std::shared_ptr<Sideset> &ss);

    std::shared_ptr<Buffer<idx_t *>> create_device_elements(const std::shared_ptr<FunctionSpace> &space,
                                                          const enum ElemType                   element_type);

    template <typename T>
    std::shared_ptr<Buffer<T>> create_device_buffer(const std::ptrdiff_t n) {
        auto ret = std::make_shared<Buffer<T>>(n, (T *)d_buffer_alloc(n * sizeof(T)), &d_buffer_destroy, MEMORY_SPACE_DEVICE);
        return ret;
    }

    template <typename T>
    std::shared_ptr<Buffer<T>> to_device(const std::shared_ptr<Buffer<T>> &in) {
        if (in->mem_space() == MEMORY_SPACE_DEVICE) {
            return in;
        }

        T *buff = (T *)d_buffer_alloc(in->size() * sizeof(T));
        // cudaMemcpy(buff, in->data(), in->size() * sizeof(T), cudaMemcpyHostToDevice);
        buffer_host_to_device(in->size() * sizeof(T), in->data(), buff);

        return std::make_shared<Buffer<T>>(in->size(), buff, &d_buffer_destroy, MEMORY_SPACE_DEVICE);
    }

    template <typename T>
    std::shared_ptr<Buffer<T *>> to_device(const std::shared_ptr<Buffer<T *>> &in) {
        if (in->mem_space() == MEMORY_SPACE_DEVICE) {
            return in;
        }

        size_t n0 = in->extent(0);
        size_t n1 = in->extent(1);

        T **dev_buff0 = (T **)d_buffer_alloc(n0 * sizeof(T *));

        std::vector<T *> host_dev_ptrs(n0);
        for (size_t i0 = 0; i0 < n0; i0++) {
            host_dev_ptrs[i0] = (T *)d_buffer_alloc(n1 * sizeof(T));
            buffer_host_to_device(n1 * sizeof(T), in->data()[i0], host_dev_ptrs[i0]);
        }

        buffer_host_to_device(n0 * sizeof(T *), host_dev_ptrs.data(), dev_buff0);

        return std::make_shared<Buffer<T *>>(
                n0,
                n1,
                dev_buff0,
                [n0, host_dev_ptrs](int n, void **ptr) {
                    for (size_t i0 = 0; i0 < n0; i0++) {
                        d_buffer_destroy(host_dev_ptrs[i0]);
                    }

                    d_buffer_destroy(ptr);
                },
                MEMORY_SPACE_DEVICE);
    }

    inline std::shared_ptr<CRSGraph> to_device(const std::shared_ptr<CRSGraph> &in) {
        if (in->rowptr()->mem_space() == MEMORY_SPACE_DEVICE) {
            return in;
        }

        return std::make_shared<CRSGraph>(to_device(in->rowptr()), to_device(in->colidx()));
    }

    template <typename T>
    std::shared_ptr<Buffer<T>> to_host(const std::shared_ptr<Buffer<T>> &in) {
        if (in->mem_space() == MEMORY_SPACE_HOST) {
            return in;
        }

        using NonConstT = typename std::remove_const<T>::type;

        NonConstT *buff = static_cast<NonConstT *>(malloc(in->size() * sizeof(T)));
        // cudaMemcpy(buff, in->data(), in->size() * sizeof(T), cudaMemcpyDeviceToHost);
        buffer_device_to_host(in->size() * sizeof(NonConstT), in->data(), buff);
        return std::make_shared<Buffer<T>>(in->size(), buff, &free, MEMORY_SPACE_HOST);
    }

    template <typename T>
    std::shared_ptr<Buffer<T *>> to_host(const std::shared_ptr<Buffer<T *>> &in) {
        if (in->mem_space() == MEMORY_SPACE_HOST) {
            return in;
        }

        const ptrdiff_t n0 = in->extent(0);
        const ptrdiff_t n1 = in->extent(1);

        auto buffer = create_host_buffer<T>(n0, n1);

        T **dev_addr = static_cast<T **>(malloc(n0 * sizeof(T *)));

        buffer_device_to_host(n0 * sizeof(T *), in->data(), dev_addr);

        for (ptrdiff_t i = 0; i < n0; i++) {
            buffer_device_to_host(n1 * sizeof(T), dev_addr[i], buffer->data()[i]);
        }

        free(dev_addr);

        return buffer;
    }

    template <typename T>
    std::shared_ptr<SparseBlockVector<T>> to_device(const std::shared_ptr<SparseBlockVector<T>> &in) {
        if (in->mem_space() == MEMORY_SPACE_DEVICE) {
            return in;
        }

        auto ret         = std::make_shared<SparseBlockVector<T>>();
        ret->block_size_ = in->block_size_;
        ret->idx_        = sfem::to_device(in->idx_);
        ret->data_       = sfem::to_device(in->data_);
        return ret;
    }

    template <typename T>
    std::shared_ptr<SparseBlockVector<T>> to_host(const std::shared_ptr<SparseBlockVector<T>> &in) {
        if (in->mem_space() == MEMORY_SPACE_HOST) {
            return in;
        }

        auto ret         = std::make_shared<SparseBlockVector<T>>();
        ret->block_size_ = in->block_size_;
        ret->idx_        = sfem::to_host(in->idx_);
        ret->data_       = sfem::to_host(in->data_);
        return ret;
    }

}  // namespace sfem

#endif  // SFEM_FUNCTION_INCORE_CUDA_HPP
