#ifndef SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP
#define SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP

#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>

#include "sfem_Buffer.hpp"
#include "sfem_base.h"

namespace sfem {

    template <typename T>
    class Operator {
    public:
        virtual ~Operator()                                        = default;
        virtual int            apply(const T* const x, T* const y) = 0;
        virtual std::ptrdiff_t rows() const                        = 0;
        virtual std::ptrdiff_t cols() const                        = 0;
        virtual ExecutionSpace execution_space() const             = 0;
    };

    template <typename T>
    class SparseBlockVector {
    public:
        int                            block_size_{0};
        std::shared_ptr<Buffer<idx_t>> idx_;
        std::shared_ptr<Buffer<T>>     data_;

        inline int                            block_size() const { return block_size_; }
        const std::shared_ptr<Buffer<idx_t>>& idx() const { return idx_; }
        const std::shared_ptr<Buffer<T>>&     data() const { return data_; }
    };

    template <typename T>
    class ShiftableOperator : public Operator<T> {
    public:
        virtual ~ShiftableOperator()                              = default;
        virtual int shift(const std::shared_ptr<Buffer<T>>& diag) = 0;
        virtual int shift(const std::shared_ptr<SparseBlockVector<T>> block_diag, const std::shared_ptr<Buffer<T>>& scaling) 
        {
            assert(false);
            SFEM_ERROR("[Error] ShiftableOperator::shift(block_diag, scaling) not implemented!\n");
            return SFEM_FAILURE;
        }
    };

    template <typename T>
    std::shared_ptr<SparseBlockVector<T>> create_sparse_block_vector(const std::shared_ptr<Buffer<idx_t>>& idx,
                                                                     const std::shared_ptr<Buffer<T>>&     data) {
        auto ret         = std::make_shared<SparseBlockVector<T>>();
        ret->block_size_ = data->size() / idx->size();
        ret->idx_        = idx;
        ret->data_       = data;
        return ret;
    }

    template <typename T>
    class LambdaOperator final : public Operator<T> {
    public:
        std::ptrdiff_t                                rows_{0};
        std::ptrdiff_t                                cols_{0};
        std::function<void(const T* const, T* const)> apply_;
        ExecutionSpace                                execution_space_;

        LambdaOperator(const std::ptrdiff_t                          rows,
                       const std::ptrdiff_t                          cols,
                       std::function<void(const T* const, T* const)> apply,
                       const ExecutionSpace                          es)
            : rows_(rows), cols_(cols), apply_(apply), execution_space_(es) {}

        inline std::ptrdiff_t rows() const override { return rows_; }
        inline std::ptrdiff_t cols() const override { return cols_; }
        inline ExecutionSpace execution_space() const override { return execution_space_; }

        int apply(const T* const x, T* const y) override {
            apply_(x, y);
            return 0;
        }
    };

    template <typename T>
    inline std::shared_ptr<Operator<T>> make_op(const std::ptrdiff_t                          rows,
                                                const std::ptrdiff_t                          cols,
                                                std::function<void(const T* const, T* const)> op,
                                                const ExecutionSpace                          es) {
        return std::make_shared<LambdaOperator<T>>(rows, cols, op, es);
    }

    template <typename T>
    inline std::shared_ptr<Operator<T>> operator+(const std::shared_ptr<Operator<T>>& left,
                                                  const std::shared_ptr<Operator<T>>& right) {
        return std::make_shared<LambdaOperator<T>>(
                left->rows(),
                left->cols(),
                [left, right](const T* const x, T* const y) {
                    right->apply(x, y);
                    left->apply(x, y);
                },
                left->execution_space());
    }

    template <typename T>
    class MatrixFreeLinearSolver : public Operator<T> {
    public:
        virtual ~MatrixFreeLinearSolver()                                          = default;
        virtual void set_op(const std::shared_ptr<Operator<T>>& op)                = 0;
        virtual void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) = 0;
        virtual void set_max_it(const int it)                                      = 0;
        virtual void set_n_dofs(const ptrdiff_t n)                                 = 0;
        virtual void set_initial_guess_zero(const bool /*val*/) {}
        virtual int  iterations() const = 0;
        virtual int  set_op_and_diag_shift(const std::shared_ptr<Operator<T>>& op, const std::shared_ptr<Buffer<T>>& diag) {
            fprintf(stderr,
                    "set_op_and_diag_shift: not implemented for subclass of "
                     "MatrixFreeLinearSolver!\n");
            assert(false);
            return SFEM_FAILURE;
        }
    };
}  // namespace sfem

#endif  // SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP
