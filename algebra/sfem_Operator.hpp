#ifndef SFEM_OPERATOR_HPP
#define SFEM_OPERATOR_HPP

// C includes
#include "sfem_base.hpp"

// C++ includes
#include "sfem_aliases.hpp"
#include "sfem_execution_context.hpp"

namespace sfem {

    template <typename T>
    class Operator {
    public:
        virtual ~Operator()                             = default;
        virtual int apply(const T* const x, T* const y) = 0;
        virtual int multi_apply(const int vec_size, const T* const* const x, T* const* const y) {
            if (execution_space() == EXECUTION_SPACE_DEVICE) {
                SFEM_ERROR(
                        "apply(vec_size, x, y) is not supported for device execution space! Override this method in the "
                        "derived class.");
            }

            for (int b = 0; b < vec_size; b++) {
                apply(x[b], y[b]);
            }
            return SFEM_SUCCESS;
        }

        virtual std::ptrdiff_t   rows() const              = 0;
        virtual std::ptrdiff_t   cols() const              = 0;
        virtual ExecutionContext execution_context() const = 0;
        ExecutionSpace           execution_space() const { return execution_context().space(); }
    };

    template <typename T>
    class InPlaceOperator {
    public:
        virtual ~InPlaceOperator()                       = default;
        virtual int              apply(T* const y)         = 0;
        virtual std::ptrdiff_t   size() const              = 0;
        virtual ExecutionContext execution_context() const = 0;
        ExecutionSpace           execution_space() const { return execution_context().space(); }
    };

    template <typename T>
    using SharedOperator = std::shared_ptr<sfem::Operator<T>>;

    template <typename T, typename L = Operator<T>, typename R = Operator<T>>
    class OperatorAdd final : public Operator<T> {
    public:
        OperatorAdd(const std::shared_ptr<L>& l, const std::shared_ptr<R>& r) : left_(l), right_(r) {
            // TODO: check that the operands are actually additive
            assert(left_->rows() == right_->rows());
            assert(left_->cols() == right_->cols());
            assert(left_->execution_space() == right_->execution_space());
        }

        int apply(const T* const x, T* const y) override {
            // ATTENTION !!! Change in OperatorAdd
            // Left first: matrix-free ops zero y, then the additive term (e.g. contact SBV) accumulates.
            int err = left_->apply(x, y);
            err |= right_->apply(x, y);
            return err;
        }

        std::ptrdiff_t rows() const override { return left_->rows(); }
        std::ptrdiff_t cols() const override { return left_->cols(); }
        ExecutionContext execution_context() const override { return left_->execution_context(); }

    private:
        std::shared_ptr<L> left_;
        std::shared_ptr<R> right_;
    };

    template <typename T>
    using SharedInPlaceOperator = std::shared_ptr<sfem::InPlaceOperator<T>>;

    template <typename T, typename L = Operator<T>, typename R = Operator<T>>
    class OperatorCompose final : public Operator<T> {
    public:
        OperatorCompose(const std::shared_ptr<L>& l, const std::shared_ptr<R>& r) : left_(l), right_(r) {
            assert(left_->cols() == right_->rows());
            assert(left_->execution_space() == right_->execution_space());
        }

        int apply(const T* const x, T* const y) override {
            int err = right_->apply(x, y);
            err |= left_->apply(x, y);
            return err;
        }
        std::ptrdiff_t rows() const override { return left_->rows(); }
        std::ptrdiff_t cols() const override { return right_->cols(); }
        ExecutionContext execution_context() const override { return left_->execution_context(); }

    private:
        std::shared_ptr<L> left_;
        std::shared_ptr<R> right_;
    };

    template <typename T>
    class OperatorCompose<InPlaceOperator<T>, Operator<T>> final : public Operator<T> {
        OperatorCompose(const SharedInPlaceOperator<T>& l, const SharedOperator<T>& r) : left_(l), right_(r) {
            assert(left_->size() == right_->rows());
            assert(left_->execution_space() == right_->execution_space());
        }
        int apply(const T* const x, T* const y) override {
            int err = right_->apply(x, y);
            err |= left_->apply(y);
            return err;
        }
        std::ptrdiff_t rows() const override { return left_->size(); }
        std::ptrdiff_t cols() const override { return right_->cols(); }
        ExecutionContext execution_context() const override { return left_->execution_context(); }

    private:
        SharedInPlaceOperator<T> left_;
        const SharedOperator<T>  right_;
    };

    template <typename T>
    class LambdaOperator final : public Operator<T> {
    public:
        std::ptrdiff_t                                rows_{0};
        std::ptrdiff_t                                cols_{0};
        std::function<void(const T* const, T* const)> apply_;
        ExecutionContext                              context_;

        LambdaOperator(const std::ptrdiff_t                          rows,
                       const std::ptrdiff_t                          cols,
                       std::function<void(const T* const, T* const)> apply,
                       const ExecutionContext&                       context)
            : rows_(rows), cols_(cols), apply_(apply), context_(context) {}

        inline std::ptrdiff_t   rows() const override { return rows_; }
        inline std::ptrdiff_t   cols() const override { return cols_; }
        inline ExecutionContext execution_context() const override { return context_; }

        int apply(const T* const x, T* const y) override {
            apply_(x, y);
            return SFEM_SUCCESS;
        }
    };

    template <typename T>
    inline std::shared_ptr<Operator<T>> make_op(const std::ptrdiff_t                          rows,
                                                const std::ptrdiff_t                          cols,
                                                std::function<void(const T* const, T* const)> op,
                                                const ExecutionContext&                       context) {
        return std::make_shared<LambdaOperator<T>>(rows, cols, op, context);
    }

    template <typename T>
    class InPlaceLambdaOperator final : public InPlaceOperator<T> {
    public:
        std::ptrdiff_t                size_{0};
        std::function<void(T* const)> apply_;
        ExecutionContext              context_;

        InPlaceLambdaOperator(const std::ptrdiff_t size, std::function<void(T* const)> apply, const ExecutionContext& context)
            : size_(size), apply_(apply), context_(context) {}

        inline std::ptrdiff_t   size() const override { return size_; }
        inline ExecutionContext execution_context() const override { return context_; }

        int apply(T* const x) override {
            apply_(x);
            return SFEM_SUCCESS;
        }
    };

    template <typename T>
    inline std::shared_ptr<InPlaceOperator<T>> make_in_place_op(const std::ptrdiff_t          size,
                                                                std::function<void(T* const)> op,
                                                                const ExecutionContext&       context) {
        return std::make_shared<InPlaceLambdaOperator<T>>(size, op, context);
    }

    template <typename T>
    inline SharedOperator<T> operator+(const SharedOperator<T>& left, const SharedOperator<T>& right) {
        return std::make_shared<OperatorAdd<T>>(left, right);
    }

    template <typename T>
    inline SharedOperator<T> operator*(const SharedOperator<T>& left, const SharedOperator<T>& right) {
        return std::make_shared<OperatorCompose<T>>(left, right);
    }

    template <typename T>
    inline SharedOperator<T> operator*(const SharedInPlaceOperator<T>& left, const SharedOperator<T>& right) {
        return std::make_shared<OperatorCompose<T, SharedInPlaceOperator<T>, SharedOperator<T>>>(left, right);
    }

}  // namespace sfem

#endif
