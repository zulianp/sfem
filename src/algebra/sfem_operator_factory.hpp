#ifndef SFEM_OPERATOR_FACTORY_HPP
#define SFEM_OPERATOR_FACTORY_HPP

#include "sfem_Operator.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "sfem_execution_context.hpp"

#include <memory>

namespace sfem {

    /// y = diag(diagonal_scaling) x + y. Instantiated for float and double.
    template <typename T>
    std::shared_ptr<Operator<T>> diag_op(const SharedBuffer<T> &diagonal_scaling, const ExecutionContext &context);

    /// y = x / diag. Instantiated for real_t.
    template <typename T>
    std::shared_ptr<Operator<T>> create_inverse_diagonal_scaling(const SharedBuffer<T> &diag, const ExecutionContext &context);

    /// r = rhs - op x; returns |r|. Runs in the operator's context.
    real_t residual(Operator<real_t> &op, const real_t *const rhs, const real_t *const x, real_t *const r);

}  // namespace sfem

#endif  // SFEM_OPERATOR_FACTORY_HPP
