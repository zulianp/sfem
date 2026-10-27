#ifndef SFEM_BSR_TANGENT_HPP
#define SFEM_BSR_TANGENT_HPP

#include <memory>

#include "sfem_Operator.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "smesh_types.hpp"

namespace sfem {

    class Function;

    /// The assembled block tangent of a `Function` on its space's node-to-node graph.
    ///
    /// The one place a BSR tangent is assembled: its values are allocated once and reassembled in
    /// place at every linearisation, and every other view of it -- the operator a Krylov method
    /// applies, the Jacobi diagonal, the arrays a constraint adds its own share to -- reads them.
    ///
    /// `storage_bytes` is the width of a stored value. sizeof(real_t) by default; 4 stores them in
    /// single precision, assembled so from the element matrices in real_t (each entry rounded as it
    /// is added), on host and device, which halves the product's traffic; the products, diagonal
    /// and bounds read them into real_t. 2 assembles them in real_t and the host product reads a
    /// half-precision copy taken by `op` (host only, one process).
    class BSRTangent final {
    public:
        static std::shared_ptr<BSRTangent> create(const std::shared_ptr<Function> &f,
                                                  const ExecutionContext          &context,
                                                  const int                        storage_bytes = (int)sizeof(real_t));
        ~BSRTangent();

        /// Zero the values and assemble the Hessian of `f` at `x` (null for operators that do not
        /// read the state). The constrained rows are what the constraints leave in them.
        int assemble(const real_t *const x);

        /// y += K x, its constrained rows the identity (on the host the assembled ones; on the
        /// device the constrained dofs of y replaced by those of x). With storage_bytes 2 it reads
        /// a copy of the values as last assembled, so it is taken again after each `assemble`.
        /// More generally y = scale_output y + K x: with 0, y = K x and y's old values are not read.
        std::shared_ptr<Operator<real_t>> op(const real_t scale_output = 1) const;

        /// The scalar diagonal, block_rows() * block_size() entries.
        int diagonal(real_t *const out) const;

        /// max_i sum_j |K_ij|, a bound on the largest eigenvalue.
        /// Gershgorin's bound on the largest eigenvalue, of S A S with `scale` (in the tangent's
        /// execution space).
        real_t max_abs_row_sum(const real_t *scale = nullptr) const;

        /// K_dd += factor * |reference_d| on the diagonal.
        int shift_diagonal(const real_t factor, const real_t *const reference);

        ptrdiff_t      block_rows() const;
        int            block_size() const;
        const count_t *rowptr() const;
        const idx_t   *colidx() const;
        /// The values, of `values_type()` (sfem_TangentValues.hpp).
        void                *values();
        smesh::PrimitiveType values_type() const;
        ExecutionSpace       execution_space() const;

    private:
        BSRTangent();

        template <typename V>
        std::shared_ptr<Operator<real_t>> op(const SharedBuffer<V> &stored, const real_t scale_output) const;

        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif  // SFEM_BSR_TANGENT_HPP
