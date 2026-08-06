#ifndef SFEM_PARALLEL_MATRIX_FREE_OPERATOR_HPP
#define SFEM_PARALLEL_MATRIX_FREE_OPERATOR_HPP

#include "sfem_Function.hpp"
#include "sfem_ParallelOperator.hpp"
#include "sfem_aliases.hpp"

#include <memory>

namespace sfem {

    /// Matrix-free operator on distributed meshes.
    ///
    /// Buffer contract for @ref apply:
    /// - @p x must provide @ref col_allocation_size() entries (owned + ghosts + aura).
    ///   Owned DOFs are read; ghost/aura slots are filled by an in-place gather.
    /// - @p y must provide @ref row_allocation_size() entries. The owned prefix is the
    ///   result; ghost/aura slots are used as assembly scratch.
    ///
    /// Optional nonlinear @p state (constructor) must already have
    /// @ref col_allocation_size() entries; @ref update_state gathers into it in place.
    class ParallelMatrixFreeOperator final : public ParallelOperator<real_t> {
    public:
        ParallelMatrixFreeOperator(const std::shared_ptr<Function>       &function,
                                   const std::shared_ptr<Buffer<real_t>> &state,
                                   ExecutionSpace                         execution_space);
        ~ParallelMatrixFreeOperator() override;

        int update_state();
        int apply(const real_t *const x, real_t *const y) override;
        std::ptrdiff_t rows() const override;
        std::ptrdiff_t cols() const override;
        ExecutionSpace execution_space() const override;

        std::shared_ptr<Communicator> comm() const override;
        std::ptrdiff_t                row_allocation_size() const override;
        std::ptrdiff_t                col_allocation_size() const override;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    std::shared_ptr<ParallelOperator<real_t>> create_parallel_matrix_free_operator(
            const std::shared_ptr<Function>       &function,
            const std::shared_ptr<Buffer<real_t>> &state,
            ExecutionSpace                         execution_space);

}  // namespace sfem

#endif  // SFEM_PARALLEL_MATRIX_FREE_OPERATOR_HPP
