#ifndef SFEM_PARALLEL_GRADIENT_OPERATOR_HPP
#define SFEM_PARALLEL_GRADIENT_OPERATOR_HPP

#include "sfem_Function.hpp"
#include "sfem_aliases.hpp"

#include <memory>

namespace sfem {

    /// Residual (gradient) evaluation on distributed meshes.
    ///
    /// The counterpart of ParallelMatrixFreeOperator for the nonlinear residual rather than
    /// the Jacobian action. Function::gradient has no communication of its own: it loops its
    /// operators over local elements and reads x wherever those elements point. On a
    /// distributed mesh that includes ghost and aura nodes, whose slots nothing fills, so the
    /// residual of every owned row touched by a partition-crossing element is wrong -- and
    /// wrong consistently on every rank, which is why it survives a rank-to-rank comparison
    /// and only shows up against a serial run.
    ///
    /// WHY THERE IS NO SCATTER HERE.
    ///
    /// The aura is a one-element-deep overlap: every element touching a node this rank owns is
    /// present locally, either owned or aura. So after a GhostsAndAura gather gives those
    /// elements correct nodal values, summing over ALL local elements makes every OWNED row
    /// complete on the spot. Nothing has to be sent back to an owner afterwards. The
    /// alternative -- compute owned elements only and scatter_add the remote contributions --
    /// needs a second, additive exchange and a matching reduction order; the aura buys the
    /// same answer with one gather.
    ///
    /// Buffer contract:
    /// - @p x must provide @ref col_allocation_size() entries (owned + ghosts + aura). The
    ///   owned prefix is read as given; ghost and aura slots are overwritten by the gather.
    /// - @p out must provide @ref row_allocation_size() entries. THE OWNED PREFIX IS THE
    ///   RESULT. Ghost and aura rows hold partial sums and are meaningless on their own --
    ///   they are the contributions this rank happened to compute for nodes another rank owns.
    ///   Every consumer must therefore read, reduce and test over the owned range only. A norm
    ///   summed to the local size counts those partial rows and is not a norm of anything.
    class ParallelGradientOperator final {
    public:
        ParallelGradientOperator(const std::shared_ptr<Function> &function, ExecutionSpace execution_space);
        ~ParallelGradientOperator();

        /// Gather x's ghost/aura slots, then evaluate the residual over all local elements.
        /// Constraints are applied by Function::gradient itself, because this evaluates with
        /// ElementScope::ALL.
        int gradient(real_t *const x, real_t *const out);

        std::shared_ptr<Communicator> comm() const;
        std::ptrdiff_t                owned_dofs() const;
        std::ptrdiff_t                row_allocation_size() const;
        std::ptrdiff_t                col_allocation_size() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    /// Returns a gradient evaluator for @p function.
    ///
    /// At one rank there is no decomposition and nothing to gather, so this returns an
    /// evaluator that calls Function::gradient directly -- the serial path is untouched, not
    /// merely equivalent.
    std::shared_ptr<ParallelGradientOperator> create_parallel_gradient_operator(
            const std::shared_ptr<Function> &function,
            ExecutionSpace                   execution_space);

}  // namespace sfem

#endif  // SFEM_PARALLEL_GRADIENT_OPERATOR_HPP
