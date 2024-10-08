#ifndef SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP
#define SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP

#include <cstddef>
#include <functional>
#include <memory>

#include "sfem_Buffer.hpp"

namespace sfem {

    template <typename T>
    class Operator {
    public:
        virtual ~Operator() = default;
        virtual int apply(const T* const x, T* const y) = 0;
        virtual std::ptrdiff_t rows() const = 0;
        virtual std::ptrdiff_t cols() const = 0;
        virtual ExecutionSpace execution_space() const = 0;
    };

    template <typename T>
    class LambdaOperator final : public Operator<T> {
    public:
        std::ptrdiff_t rows_{0};
        std::ptrdiff_t cols_{0};
        std::function<void(const T* const, T* const)> apply_;
        ExecutionSpace execution_space_;

        LambdaOperator(const std::ptrdiff_t rows,
                       const std::ptrdiff_t cols,
                       std::function<void(const T* const, T* const)> apply,
                       const ExecutionSpace es)
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
    inline std::shared_ptr<Operator<T>> make_op(const std::ptrdiff_t rows,
                                                const std::ptrdiff_t cols,
                                                std::function<void(const T* const, T* const)> op,
                                                const ExecutionSpace es) {
        return std::make_shared<LambdaOperator<T>>(rows, cols, op, es);
    }

    template <typename T>
    class MatrixFreeLinearSolver : public Operator<T> {
    public:
        virtual ~MatrixFreeLinearSolver() = default;
        virtual void set_op(const std::shared_ptr<Operator<T>>& op) = 0;
        virtual void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) = 0;
        virtual void set_max_it(const int it) = 0;
        virtual void set_n_dofs(const ptrdiff_t n) = 0;
        virtual void set_initial_guess_zero(const bool /*val*/) {}
    };
}  // namespace sfem

#endif  // SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP
