#ifndef SFEM_FUNCTION_INCORE_CUDA_HPP
#define SFEM_FUNCTION_INCORE_CUDA_HPP

#include <memory>
#include "sfem_Function.hpp"

#include "sfem_aliases.hpp"
#include "sfem_cuda_blas.hpp"

#include "sfem_MatrixFreeLinearSolver.hpp"

#include <memory>

#include "smesh_device_buffer.hpp"

namespace sfem {
    void                        register_device_ops();
    std::shared_ptr<Constraint> to_device(const std::shared_ptr<DirichletConditions> &dc);
    std::shared_ptr<Op>         to_device(const std::shared_ptr<NeumannConditions> &nc);

    SharedBuffer<idx_t *> create_device_elements(const std::shared_ptr<FunctionSpace> &space, const smesh::ElemType element_type);

    template <typename T>
    std::shared_ptr<SparseBlockVector<T>> to_device(const std::shared_ptr<SparseBlockVector<T>> &in) {
        if (in->mem_space() == MEMORY_SPACE_DEVICE) {
            return in;
        }

        auto ret         = std::make_shared<SparseBlockVector<T>>();
        ret->block_size_ = in->block_size_;
        ret->idx_        = smesh::to_device(in->idx_);
        ret->data_       = smesh::to_device(in->data_);
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
