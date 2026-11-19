#ifndef SFEM_SELL_TANGENT_HPP
#define SFEM_SELL_TANGENT_HPP

#include <memory>
#include <string>

#include "sfem_InexactOptions.hpp"
#include "sfem_Operator.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"

namespace sfem {

    class Function;

    /// The assembled tangent of a vector problem in SELL-C-sigma, blocks of 3x3 (algebra/sell):
    /// `solver.linear.format: SELL`, beside BSR. The operator's own tangent, without Dirichlet rows,
    /// assembled in single precision and encoded as f32 values, f16 codes or int8 codes, in the
    /// difference form (its diagonal blocks implied by its rows' sums), the Dirichlet rows applied as
    /// the stored tangent's are. Host: C = 16, sigma = 512, a symmetric tangent's pairs stored once;
    /// device: C = 32, sigma = 1024. It is the INEXACT operator's `bsr` store in the SELL layout
    /// (`options()`), so the stored tangent's every use -- contact's extra blocks, the obstacle QP,
    /// projection -- takes it too.
    class SELLTangent final {
    public:
        /// The INEXACT options this format is: store bsr, layout sell, `encoding` (f32, f16 or i8), and
        /// the time scheme's inertia beside it (`inertia`: auto and fused add a lumped one in the store's
        /// pass, separate applies it as a pass of its own).
        static InexactOptions options(const InexactOptions::Encoding encoding,
                                      const InexactOptions::Inertia  inertia = InexactOptions::Inertia::AUTO);

        static std::shared_ptr<SELLTangent> create(const std::shared_ptr<Function> &f,
                                                   const ExecutionSpace             es,
                                                   const InexactOptions::Encoding   encoding = InexactOptions::Encoding::F32,
                                                   const InexactOptions::Inertia    inertia  = InexactOptions::Inertia::AUTO);

        /// Assembles and encodes the tangent at x (the first call sets the operators' format).
        int assemble(const real_t *const x);
        /// The operator that applies it, valid across assemblies.
        std::shared_ptr<Operator<real_t>> op(const SharedBuffer<real_t> &state) const;
        /// diag += the encoded operator's diagonal (the Krylov method's Jacobi).
        int         diagonal(const real_t *const x, real_t *const diag) const;
        std::size_t bytes() const;
        std::string describe() const;

    private:
        SELLTangent() = default;
        std::shared_ptr<Function> f_;
        ExecutionSpace            es_{EXECUTION_SPACE_HOST};
        InexactOptions            options_;
        bool                      configured_{false};
    };

}  // namespace sfem

#endif  // SFEM_SELL_TANGENT_HPP
