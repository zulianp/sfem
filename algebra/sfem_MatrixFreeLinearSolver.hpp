#ifndef SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP
#define SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP

#include <cstddef>
#include <functional>
#include <memory>

namespace sfem {

    template <typename T>
    class Operator {
    public:
        virtual ~Operator() = default;
        virtual int apply(const T* const x, T* const y) = 0;
        virtual std::ptrdiff_t rows() const = 0;
        virtual std::ptrdiff_t cols() const = 0;
    };

    template <typename T>
    class LambdaOperator final : public Operator<T> {
    public:
        std::ptrdiff_t rows_{0};
        std::ptrdiff_t cols_{0};
        std::function<void(const T* const, T* const)> apply_;

        LambdaOperator(const std::ptrdiff_t rows,
                       const std::ptrdiff_t cols,
                       std::function<void(const T* const, T* const)> apply)
            : rows_(rows), cols_(cols), apply_(apply) {}

        inline std::ptrdiff_t rows() const override { return rows_; }
        inline std::ptrdiff_t cols() const override { return cols_; }

        int apply(const T* const x, T* const y) override {
            apply_(x, y);
            return 0;
        }
    };

    template <typename T>
    inline std::shared_ptr<Operator<T>> make_op(const std::ptrdiff_t rows,
                                                const std::ptrdiff_t cols,
                                                std::function<void(const T* const, T* const)> op) {
        return std::make_shared<LambdaOperator<T>>(rows, cols, op);
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
