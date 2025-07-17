#ifndef SFEM_OPERATOR_HPP
#define SFEM_OPERATOR_HPP

// C includes
#include "sfem_base.h"

// C++ includes
#include "sfem_Buffer.hpp"

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
    class InPlaceOperator {
    public:
        virtual ~InPlaceOperator()                     = default;
        virtual int            apply(T* const y)       = 0;
        virtual std::ptrdiff_t size() const            = 0;
        virtual ExecutionSpace execution_space() const = 0;
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
            int err = right_->apply(x, y);
            err |= left_->apply(x, y);
            return err;
        }

        std::ptrdiff_t rows() const override { return left_->rows(); }
        std::ptrdiff_t cols() const override { return left_->cols(); }
        ExecutionSpace execution_space() const override { return left_->execution_space(); }

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
        ExecutionSpace execution_space() const override { return left_->execution_space(); }

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
        ExecutionSpace execution_space() const override { return left_->execution_space(); }

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
            return SFEM_SUCCESS;
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
    class InPlaceLambdaOperator final : public InPlaceOperator<T> {
    public:
        std::ptrdiff_t                                size_{0};
        std::function<void(T* const)> apply_;
        ExecutionSpace                                execution_space_;

        InPlaceLambdaOperator(const std::ptrdiff_t size, std::function<void(T* const)> apply, const ExecutionSpace es)
            : size_(size), apply_(apply), execution_space_(es) {}

        inline std::ptrdiff_t size() const override { return size_; }
        inline ExecutionSpace execution_space() const override { return execution_space_; }

        int apply(T* const x) override {
            apply_(x);
            return SFEM_SUCCESS;
        }
    };

    template <typename T>
    inline std::shared_ptr<InPlaceOperator<T>> make_in_place_op(const std::ptrdiff_t          size,
                                                                std::function<void(T* const)> op,
                                                                const ExecutionSpace          es) {
        return std::make_shared<InPlaceLambdaOperator<T>>(size, op, es);
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
