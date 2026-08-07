#ifndef SFEM_PARALLEL_OPERATOR_HPP
#define SFEM_PARALLEL_OPERATOR_HPP

#include "sfem_Operator.hpp"
#include "sfem_aliases.hpp"

#include <cassert>
#include <functional>
#include <memory>

namespace sfem {

    template <typename T>
    class ParallelOperator : public Operator<T> {
    public:
        ~ParallelOperator() override = default;

        virtual std::shared_ptr<Communicator> comm() const = 0;

        /// Buffer capacity for range vectors (y), including ghosts/aura.
        virtual std::ptrdiff_t row_allocation_size() const = 0;

        /// Buffer capacity for domain vectors (x), including ghosts/aura.
        virtual std::ptrdiff_t col_allocation_size() const = 0;
    };

    template <typename T>
    class LambdaParallelOperator final : public ParallelOperator<T> {
    public:
        LambdaParallelOperator(const std::shared_ptr<Communicator>&           comm,
                               const std::ptrdiff_t                           rows,
                               const std::ptrdiff_t                           cols,
                               const std::ptrdiff_t                           row_allocation_size,
                               const std::ptrdiff_t                           col_allocation_size,
                               std::function<void(const T* const, T* const)>  apply,
                               const ExecutionSpace                           es)
            : comm_(comm),
              rows_(rows),
              cols_(cols),
              row_allocation_size_(row_allocation_size),
              col_allocation_size_(col_allocation_size),
              apply_(std::move(apply)),
              execution_space_(es) {
            assert(comm_);
            assert(rows_ >= 0);
            assert(cols_ >= 0);
            assert(row_allocation_size_ >= rows_);
            assert(col_allocation_size_ >= cols_);
        }

        int apply(const T* const x, T* const y) override {
            apply_(x, y);
            return SFEM_SUCCESS;
        }

        std::ptrdiff_t rows() const override { return rows_; }
        std::ptrdiff_t cols() const override { return cols_; }
        ExecutionSpace execution_space() const override { return execution_space_; }

        std::shared_ptr<Communicator> comm() const override { return comm_; }
        std::ptrdiff_t                row_allocation_size() const override { return row_allocation_size_; }
        std::ptrdiff_t                col_allocation_size() const override { return col_allocation_size_; }

    private:
        std::shared_ptr<Communicator>          comm_;
        std::ptrdiff_t                         rows_{0};
        std::ptrdiff_t                         cols_{0};
        std::ptrdiff_t                         row_allocation_size_{0};
        std::ptrdiff_t                         col_allocation_size_{0};
        std::function<void(const T* const, T* const)> apply_;
        ExecutionSpace                         execution_space_{EXECUTION_SPACE_INVALID};
    };

    template <typename T>
    inline std::shared_ptr<ParallelOperator<T>> make_parallel_op(
            const std::shared_ptr<Communicator>&          comm,
            const std::ptrdiff_t                          rows,
            const std::ptrdiff_t                          cols,
            std::function<void(const T* const, T* const)> op,
            const ExecutionSpace                          es) {
        return std::make_shared<LambdaParallelOperator<T>>(comm, rows, cols, rows, cols, std::move(op), es);
    }

    template <typename T>
    inline std::shared_ptr<ParallelOperator<T>> make_parallel_op(
            const std::shared_ptr<Communicator>&          comm,
            const std::ptrdiff_t                          rows,
            const std::ptrdiff_t                          cols,
            const std::ptrdiff_t                          row_allocation_size,
            const std::ptrdiff_t                          col_allocation_size,
            std::function<void(const T* const, T* const)> op,
            const ExecutionSpace                          es) {
        return std::make_shared<LambdaParallelOperator<T>>(
                comm, rows, cols, row_allocation_size, col_allocation_size, std::move(op), es);
    }

}  // namespace sfem

#endif  // SFEM_PARALLEL_OPERATOR_HPP
