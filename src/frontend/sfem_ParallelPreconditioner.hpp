#ifndef SFEM_PARALLEL_PRECONDITIONER_HPP
#define SFEM_PARALLEL_PRECONDITIONER_HPP

#include "sfem_Function.hpp"
#include "sfem_ParallelOperator.hpp"
#include "sfem_aliases.hpp"

#include <memory>

namespace sfem {

    /// Gives a node-local preconditioner a correct face on a distributed mesh.
    ///
    /// A preconditioner that only ever touches its own row -- point or block Jacobi -- needs
    /// nothing here: it reads r[i] to write y[i]. One whose unknowns are coupled over a patch
    /// does, because it reads its neighbours' entries of the residual, and on a distributed
    /// mesh some of those neighbours are ghost or aura nodes. Nothing fills those slots:
    /// Multigrid hands the smoother mem->rhs directly, and the residual that produced it was
    /// reduced over the OWNED range alone. The patch then solves against zeros where it should
    /// have had its neighbour's residual, and contributes an under-correction to every owned
    /// row it touches.
    ///
    /// This is invisible in a rank-to-rank comparison, because every rank does it. Measured on
    /// the CVFEM cavity with a Vanka smoother inside a V-cycle: 61 linear iterations at one
    /// rank, and the 4000- and 7000-iteration caps at two and four ranks, while block-Jacobi
    /// -- immune, by the argument above -- went 5, 9, 14 over the same rank counts.
    ///
    /// WHAT THIS DOES NOT DO.
    ///
    /// It does not zero @p y. A smoother ACCUMULATES its correction: StationaryIteration calls
    /// preconditioner->apply(r, x) once per sweep without clearing x in between, so zeroing the
    /// output here would discard every sweep but the last.
    ///
    /// It does not zero the ghost/aura tail of @p r before gathering, which is what
    /// ParallelMatrixFreeOperator does for its input. The gather fills precisely those slots,
    /// so the clear would buy nothing -- and it would be a write into the cycle's own rhs
    /// buffer, which Multigrid reads back over a range that reaches into the ghost rows on any
    /// level whose operator is not a ParallelOperator (level_owned falls back to a local count
    /// there). Leave the buffer alone apart from the slots the gather owns.
    ///
    /// No scatter_add is needed on the way out, for the same reason ParallelGradientOperator
    /// needs none: the aura is a one-element-deep overlap, so every patch touching an owned
    /// node is present locally and that row is complete once its inputs are right.
    std::shared_ptr<ParallelOperator<real_t>> create_parallel_preconditioner(
            const std::shared_ptr<Operator<real_t>>    &preconditioner,
            const std::shared_ptr<FunctionSpace>       &space,
            ExecutionSpace                              execution_space);

}  // namespace sfem

#endif  // SFEM_PARALLEL_PRECONDITIONER_HPP
