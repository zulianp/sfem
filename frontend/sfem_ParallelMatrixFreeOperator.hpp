#ifndef SFEM_PARALLEL_MATRIX_FREE_OPERATOR_HPP
#define SFEM_PARALLEL_MATRIX_FREE_OPERATOR_HPP

#include "sfem_Function.hpp"
#include "sfem_Operator.hpp"
#include "sfem_aliases.hpp"

#include <memory>

namespace sfem {

    class ParallelMatrixFreeOperator final : public Operator<real_t> {
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

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    std::shared_ptr<Operator<real_t>> create_parallel_matrix_free_operator(
            const std::shared_ptr<Function>       &function,
            const std::shared_ptr<Buffer<real_t>> &state,
            ExecutionSpace                         execution_space);

}  // namespace sfem

#endif  // SFEM_PARALLEL_MATRIX_FREE_OPERATOR_HPP
