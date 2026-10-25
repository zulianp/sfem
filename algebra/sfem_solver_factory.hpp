#ifndef SFEM_SOLVER_FACTORY_HPP
#define SFEM_SOLVER_FACTORY_HPP

#include "sfem_Chebyshev3.hpp"
#include "sfem_MixedPrecisionShiftableBlockSymJacobi.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_ShiftableJacobi.hpp"
#include "sfem_Stationary.hpp"
#include "sfem_aliases.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"
#include "sfem_execution_context.hpp"
#include "sfem_mprgp.hpp"
#include "sfem_parallel_cg.hpp"

#include <memory>

// Solvers and preconditioners built for an execution space. The device builds live behind the
// definitions (sfem_solver_factory.cpp), which are instantiated for real_t (and the mixed-precision
// Jacobi for <real_t, float>).
namespace sfem {

    template <typename T>
    std::shared_ptr<ConjugateGradient<T>> create_cg(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es);

    /// Parallel CG. Caller must pass @c x with @c op->col_allocation_size() entries
    /// (owned + ghosts/aura). @c b needs at least @c op->rows(); use @c row_allocation_size()
    /// if applying sideset Dirichlet constraints into @c b (ghost indices).
    template <typename T>
    std::shared_ptr<ParallelConjugateGradient<T>> create_parallel_cg(const std::shared_ptr<ParallelOperator<T>> &op);

    template <typename T>
    std::shared_ptr<ShiftableJacobi<T>> create_shiftable_jacobi(const SharedBuffer<T> &diag, const ExecutionSpace es);

    template <typename HP, typename LP>
    std::shared_ptr<MixedPrecisionShiftableBlockSymJacobi<HP, LP>> create_mixed_precision_shiftable_block_sym_jacobi(
            const int                   dim,
            const SharedBuffer<HP>     &diag,
            const SharedBuffer<mask_t> &constraints_mask,
            const ExecutionContext     &context);

    template <typename T>
    std::shared_ptr<ShiftableBlockSymJacobi<T>> create_shiftable_block_sym_jacobi(const int                   dim,
                                                                                  const SharedBuffer<T>      &diag,
                                                                                  const SharedBuffer<mask_t> &constraints_mask,
                                                                                  const ExecutionContext     &context);

    template <typename T>
    std::shared_ptr<StationaryIteration<T>> create_stationary(const std::shared_ptr<Operator<T>> &op,
                                                              const std::shared_ptr<Operator<T>> &preconditioner,
                                                              const ExecutionSpace                es);

    template <typename T>
    std::shared_ptr<BiCGStab<T>> create_bcgs(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es);

    template <typename T>
    std::shared_ptr<Chebyshev3<T>> create_cheb3(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es);

    template <typename T>
    std::shared_ptr<MPRGP<T>> create_mprgp(const std::shared_ptr<Operator<T>> &op, const ExecutionContext &context);

    template <typename T>
    std::shared_ptr<Multigrid<T>> create_mg(const ExecutionSpace es);

}  // namespace sfem

#endif  // SFEM_SOLVER_FACTORY_HPP
