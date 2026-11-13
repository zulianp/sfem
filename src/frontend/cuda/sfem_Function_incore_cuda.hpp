#ifndef SFEM_FUNCTION_INCORE_CUDA_HPP
#define SFEM_FUNCTION_INCORE_CUDA_HPP

#include <memory>
#include "sfem_Function.hpp"

#include "sfem_aliases.hpp"
#include "sfem_BLAS_cuda.hpp"

#include "sfem_MatrixFreeLinearSolver.hpp"

#include <memory>

#include "smesh_device_buffer.hpp"

namespace sfem {
    void register_device_ops();
    // std::shared_ptr<Constraint> to_device(const std::shared_ptr<DirichletConditions> &dc);
    // std::shared_ptr<Op>         to_device(const std::shared_ptr<NeumannConditions> &nc);

    template <typename T>
    std::shared_ptr<SparseBlockVector<T>> to_device(const std::shared_ptr<SparseBlockVector<T>> &in) {
        // Each array on its own: mem_space() is the blocks', and indices left on the host with
        // device blocks would be read by the device kernels as they are.
        if (in->idx_->mem_space() == MEMORY_SPACE_DEVICE && in->data_->mem_space() == MEMORY_SPACE_DEVICE) {
            return in;
        }

        auto ret         = std::make_shared<SparseBlockVector<T>>();
        ret->block_size_ = in->block_size_;
        ret->idx_        = in->idx_->mem_space() == MEMORY_SPACE_DEVICE ? in->idx_ : smesh::to_device(in->idx_);
        ret->data_       = in->data_->mem_space() == MEMORY_SPACE_DEVICE ? in->data_ : smesh::to_device(in->data_);
        return ret;
    }

    template <typename T>
    std::shared_ptr<SparseBlockVector<T>> to_host(const std::shared_ptr<SparseBlockVector<T>> &in) {
        if (in->idx_->mem_space() == MEMORY_SPACE_HOST && in->data_->mem_space() == MEMORY_SPACE_HOST) {
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
