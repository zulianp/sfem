#ifndef SFEM_BSR_TANGENT_HPP
#define SFEM_BSR_TANGENT_HPP

#include <memory>

#include "sfem_Operator.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"

namespace sfem {

    class Function;

    /// The assembled block tangent of a `Function` on its space's node-to-node graph.
    ///
    /// The one place a BSR tangent is assembled: its values are allocated once and reassembled in
    /// place at every linearisation, and every other view of it -- the operator a Krylov method
    /// applies, the Jacobi diagonal, the arrays a constraint adds its own share to -- reads them.
    class BSRTangent final {
    public:
        static std::shared_ptr<BSRTangent> create(const std::shared_ptr<Function> &f, const ExecutionSpace es);
        ~BSRTangent();

        /// Zero the values and assemble the Hessian of `f` at `x` (null for operators that do not
        /// read the state). The constrained rows are what the constraints leave in them.
        int assemble(const real_t *const x);

        /// y += K x, the constrained dofs of y replaced by those of x. With `storage_bytes` below
        /// sizeof(real_t) the values are stored in half or single precision: a copy of the values
        /// as last assembled, so it is taken again after each `assemble`. Reduced storage is host
        /// only.
        std::shared_ptr<Operator<real_t>> op(const int storage_bytes = (int)sizeof(real_t)) const;

        /// The scalar diagonal, block_rows() * block_size() entries.
        int diagonal(real_t *const out) const;

        /// max_i sum_j |K_ij|, a bound on the largest eigenvalue. Host only for now.
        real_t max_abs_row_sum() const;

        /// K_dd += factor * |reference_d| on the diagonal. Host only for now.
        int shift_diagonal(const real_t factor, const real_t *const reference);

        ptrdiff_t      block_rows() const;
        int            block_size() const;
        const count_t *rowptr() const;
        const idx_t   *colidx() const;
        real_t        *values();
        ExecutionSpace execution_space() const;

    private:
        BSRTangent();

        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif  // SFEM_BSR_TANGENT_HPP
