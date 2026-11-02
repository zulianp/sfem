#ifndef SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP
#define SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP

// C includes
#include "sfem_base.hpp"

// C++ includes
#include "sfem_aliases.hpp"
#include "sfem_Operator.hpp"
#include "sfem_SparseBlockVector.hpp"
#include "sfem_TeamOperator.hpp"


// STL includes
#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>

#ifdef SFEM_ENABLE_CUDA
#include "sfem_BLAS_cuda.hpp"
#endif

namespace sfem {

    /// y += sum over the blocks i of scaling[i] B_i x at node idx[i]: the symmetric block (vec_dim
    /// (vec_dim + 1) / 2 entries) of block i, its node's vec_dim rows. The body of every product.
    template <typename T>
    inline void scaled_sym_block_mult_block(const ptrdiff_t    i,
                                            const idx_t* const idx,
                                            const T* const     dd,
                                            const T* const     s,
                                            const int          sym_block_size,
                                            const T* const     x,
                                            T* const           y) {
        const int vec_dim = (sym_block_size == 6) ? 3 : 2;
        auto      di      = &dd[i * sym_block_size];
        auto      si      = s[i];

        const ptrdiff_t b  = idx[i];
        auto            xi = &x[b * vec_dim];
        auto            yi = &y[b * vec_dim];

        T buff[3] = {0, 0, 0};

        int d_idx = 0;
        for (int d1 = 0; d1 < vec_dim; d1++) {
            const auto m = si * di[d_idx++];
            buff[d1] += m * xi[d1];
            for (int d2 = d1 + 1; d2 < vec_dim; d2++) {
                const auto m = si * di[d_idx++];
                buff[d1] += m * xi[d2];
                buff[d2] += m * xi[d1];
            }
        }

        for (int d = 0; d < vec_dim; d++) {
            yi[d] += buff[d];
        }
    }

    template <typename T>
    class ScaledBlockVectorMult : public Operator<T>, public TeamOperator<T> {
    public:
        ptrdiff_t                                        n_dofs;
        std::shared_ptr<SparseBlockVector<T>>            sbv;
        SharedBuffer<T>                                  scaling;
        std::function<int(const T* const x, T* const y)> apply_;

#ifdef SFEM_ENABLE_CUDA
        void cuda_init() {
            assert(sbv->mem_space() == MEMORY_SPACE_DEVICE);
            apply_ = [this](const T* const x, T* const y) -> int {
                const ptrdiff_t n_blocks = this->sbv->n_blocks();
                if (n_blocks == 0) {
                    return SFEM_SUCCESS;
                }
                const idx_t* const idx = this->sbv->idx()->data();
                const T* const     dd  = this->sbv->data()->data();
                const T* const     s   = this->scaling->data();

                return sbv_mult3<T>(n_blocks, idx, dd, s, x, y, this->execution_context().stream());
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
        }

        ScaledBlockVectorMult() {}

        int apply(const T* const x, T* const y) override {
            SFEM_TRACE_SCOPE("ScaledBlockVectorMult::apply");
            if (team_capable()) return team_region_apply<T>(*this, x, y);
            return apply_(x, y);
        }

        /// Every block, split among the region's threads (an orphaned `omp for`: a block's rows may
        /// be any thread's), ending on the loop's barrier: the rows are complete for every thread.
        /// The blocks are at distinct nodes.
        void team_apply(const Team& /*team*/, const T* const x, T* const y) const override {
            const ptrdiff_t n_blocks = sbv->n_blocks();
            const idx_t* const idx   = sbv->idx()->data();
            const T* const     dd    = sbv->data()->data();
            const T* const     s     = scaling->data();
            const int          sbs   = sbv->block_size();
#pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < n_blocks; i++) {
                scaled_sym_block_mult_block(i, idx, dd, s, sbs, x, y);
            }
        }
        Team team() const override { return Team::current(n_dofs); }
        bool team_capable() const override { return sbv->mem_space() == MEMORY_SPACE_HOST; }
        int  prepare() override {
            const int sbs = sbv->block_size();
            if (sbv->n_blocks() > 0 && sbs != 6 && sbs != 3) SFEM_ERROR("ScaledBlockVectorMult: symmetric blocks of %d entries\n", sbs);
            return SFEM_SUCCESS;
        }

        std::ptrdiff_t rows() const override { return n_dofs; }
        std::ptrdiff_t cols() const override { return n_dofs; }
        ExecutionContext execution_context() const override { return (enum ExecutionSpace)sbv->data()->mem_space(); }
    };

    template <typename T>
    std::shared_ptr<Operator<T>> create_sparse_block_vector_mult(const ptrdiff_t                              n_dofs,
                                                                 const std::shared_ptr<SparseBlockVector<T>>& sbv,
                                                                 const SharedBuffer<T>&                       scaling) {
        auto ret     = std::make_shared<ScaledBlockVectorMult<T>>();
        ret->n_dofs  = n_dofs;
        ret->sbv     = sbv;
        ret->scaling = scaling;
        ret->init();
        return ret;
    }

    template <typename T>
    class ShiftableOperator : public Operator<T> {
    public:
        virtual ~ShiftableOperator()                   = default;
        virtual int shift(const SharedBuffer<T>& diag) = 0;
        virtual int shift(const std::shared_ptr<SparseBlockVector<T>>& /*block_diag*/, const SharedBuffer<T>& /*scaling*/) {
            assert(false);
            SFEM_ERROR("[Error] ShiftableOperator::shift(block_diag, scaling) not implemented!\n");
            return SFEM_FAILURE;
        }
        /// shift(block_diag, scaling) from inside a region, by every thread of it, ending on a
        /// barrier: where team_shift_capable() (host, the update touching only the blocks' nodes).
        virtual bool team_shift_capable() const { return false; }
        virtual void team_shift(const std::shared_ptr<SparseBlockVector<T>>& /*block_diag*/, const SharedBuffer<T>& /*scaling*/) {}
    };

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
        /// The preconditioner set_op_and_diag_shift shifts, where it is one (null otherwise).
        virtual std::shared_ptr<ShiftableOperator<T>> shiftable_preconditioner() const { return nullptr; }
        virtual int  set_op_and_diag_shift(const std::shared_ptr<Operator<T>>& /*op*/, const SharedBuffer<T>& /*diag*/) {
            SFEM_ERROR(
                    "set_op_and_diag_shift: not implemented for subclass of "
                     "MatrixFreeLinearSolver!\n");
            return SFEM_FAILURE;
        }

        virtual int set_op_and_diag_shift(const std::shared_ptr<Operator<T>>&          /*op*/,
                                          const std::shared_ptr<SparseBlockVector<T>>& /*sbv*/,
                                          const SharedBuffer<T>&                       /*diag*/) {
            SFEM_ERROR(
                    "set_op_and_diag_shift: not implemented for subclass of "
                    "MatrixFreeLinearSolver!\n");
            return SFEM_FAILURE;
        }
    };
}  // namespace sfem

#endif  // SFEM_MATRIX_FREE_LINEAR_SOLVER_HPP
