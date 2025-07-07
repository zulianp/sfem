#ifndef SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP
#define SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP

// C includes
#include "sfem_base.h"

// C++ includes
#include "sfem_Buffer.hpp"
#include "sfem_Tracer.hpp"

// STL includes
#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_blas.hpp"
#endif

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
    class SparseBlockVector /*: public Operator<T>*/ {
    public:
        int                            block_size_{0};
        SharedBuffer<idx_t> idx_;
        SharedBuffer<T>     data_;

        inline int                            block_size() const { return block_size_; }
        const SharedBuffer<idx_t>& idx() const { return idx_; }
        const SharedBuffer<T>&     data() const { return data_; }
        ptrdiff_t                             n_blocks() const { return idx_->size(); }

        enum MemorySpace mem_space() const {
            if (data_) {
                return data_->mem_space();
            }

            return MEMORY_SPACE_INVALID;
        }

        void print(std::ostream& os) const {
            for (ptrdiff_t i = 0; i < n_blocks(); i++) {
                os << idx_->data()[i] << ") ";

                for (int d = 0; d < block_size(); d++) {
                    os << data_->data()[i * 6 + d];
                    os << " ";
                }

                os << "\n";
            }

            os << "\n";
        }

        // TODO maybe
        // int apply(const T* const x, T* const y) override {

        //     return SFEM_SUCCESS;
        // }

        // std::ptrdiff_t rows() const override { return data_->size(); }
        // std::ptrdiff_t cols() const override { return data_->size(); }
        // ExecutionSpace execution_space() const override { return data_->mem_space(); }
    };

    template <typename T>
    std::shared_ptr<SparseBlockVector<T>> create_sparse_block_vector(const SharedBuffer<idx_t>& idx,
                                                                     const SharedBuffer<T>&     data) {
        auto ret         = std::make_shared<SparseBlockVector<T>>();
        ret->block_size_ = data->size() / idx->size();
        ret->idx_        = idx;
        ret->data_       = data;
        return ret;
    }

    template <typename T>
    class ScaledBlockVectorMult : public Operator<T> {
    public:
        std::shared_ptr<SparseBlockVector<T>>            sbv;
        SharedBuffer<T>                       scaling;
        std::function<int(const T* const x, T* const y)> apply_;

        void default_init() {
            apply_ = [this](const T* const x, T* const y) -> int {
                const ptrdiff_t    n_blocks   = sbv->n_blocks();
                const idx_t* const idx        = sbv->idx()->data();
                const T* const     dd         = sbv->data()->data();
                const T* const     s          = scaling->data();
                const int          block_size = 3;
                assert(sbv->block_size() == 6);
                // memset(y, 0, n_blocks * 3 * sizeof(T));

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_blocks; i++) {
                    auto di = &dd[i * 6];
                    auto si = s[i];

                    const ptrdiff_t b  = idx[i];
                    auto        xi = &x[b * block_size];
                    auto        yi = &y[b * block_size];

                    T buff[3] = {0, 0, 0};

                    int d_idx = 0;
                    for (int d1 = 0; d1 < block_size; d1++) {
                        const auto m = si * di[d_idx++];
                        buff[d1] += m * xi[d1];
                        for (int d2 = d1 + 1; d2 < block_size; d2++) {
                            const auto m = si * di[d_idx++];
                            buff[d1] += m * xi[d2];
                            buff[d2] += m * xi[d1];
                        }
                    }

                    yi[0] += buff[0];
                    yi[1] += buff[1];
                    yi[2] += buff[2];
                }

                return SFEM_SUCCESS;
            };
        }

#ifdef SFEM_ENABLE_CUDA
        void cuda_init() {
            assert(sbv->mem_space() == MEMORY_SPACE_DEVICE);
            apply_ = [this](const T* const x, T* const y) -> int {
                const ptrdiff_t    n_blocks = this->sbv->n_blocks();
                const idx_t* const idx      = this->sbv->idx()->data();
                const T* const     dd       = this->sbv->data()->data();
                const T* const     s        = this->scaling->data();

                return sbv_mult3<T>(n_blocks, idx, dd, s, x, y);
            };
        }
#endif

        void init() {
#ifdef SFEM_ENABLE_CUDA
            if (sbv->mem_space() == MEMORY_SPACE_DEVICE) {
                cuda_init();
                return;
            }
#endif
            default_init();
        }

        ScaledBlockVectorMult() {}

        int apply(const T* const x, T* const y) override {
            SFEM_TRACE_SCOPE("ScaledBlockVectorMult::apply");
            return apply_(x, y);
        }

        std::ptrdiff_t rows() const override { return sbv->data()->size(); }
        std::ptrdiff_t cols() const override { return sbv->data()->size(); }
        ExecutionSpace execution_space() const override { return (enum ExecutionSpace)sbv->data()->mem_space(); }
    };

    template <typename T>
    std::shared_ptr<Operator<T>> create_sparse_block_vector_mult(const std::shared_ptr<SparseBlockVector<T>>& sbv,
                                                                 const SharedBuffer<T>&            scaling) {
        auto ret     = std::make_shared<ScaledBlockVectorMult<T>>();
        ret->sbv     = sbv;
        ret->scaling = scaling;
        ret->init();
        return ret;
    }

    template <typename T>
    class ShiftableOperator : public Operator<T> {
    public:
        virtual ~ShiftableOperator()                              = default;
        virtual int shift(const SharedBuffer<T>& diag) = 0;
        virtual int shift(const std::shared_ptr<SparseBlockVector<T>>& block_diag, const SharedBuffer<T>& scaling) {
            assert(false);
            SFEM_ERROR("[Error] ShiftableOperator::shift(block_diag, scaling) not implemented!\n");
            return SFEM_FAILURE;
        }
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
        virtual int  set_op_and_diag_shift(const std::shared_ptr<Operator<T>>& op, const SharedBuffer<T>& diag) {
            SFEM_ERROR(
                    "set_op_and_diag_shift: not implemented for subclass of "
                     "MatrixFreeLinearSolver!\n");
            return SFEM_FAILURE;
        }

        virtual int set_op_and_diag_shift(const std::shared_ptr<Operator<T>>&          op,
                                          const std::shared_ptr<SparseBlockVector<T>>& sbv,
                                          const SharedBuffer<T>&            diag) {
            SFEM_ERROR(
                    "set_op_and_diag_shift: not implemented for subclass of "
                    "MatrixFreeLinearSolver!\n");
            return SFEM_FAILURE;
        }
    };
}  // namespace sfem

#endif  // SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP
