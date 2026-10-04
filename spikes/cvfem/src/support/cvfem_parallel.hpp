#ifndef CVFEM_PARALLEL_HPP
#define CVFEM_PARALLEL_HPP

// One way to reduce a scalar across ranks, for the whole spike.
//
// The driver computes a lot of scalars that are only correct if every rank agrees on them:
// the FGMRES dot and norm, the pressure gauge's two projections, the Newton and Armijo
// acceptance tests, the CFL maximum, the divergence norm. Each of those is a place where a
// rank can happily evaluate its own slice and never notice the slice is partial, and each
// would otherwise grow its own `if (comm->size() > 1) ...` at the call site. That is how a
// codebase ends up with six spellings of the same reduction and one of them wrong.
//
// SERIAL MUST STAY BIT-IDENTICAL. Every function here returns its argument unchanged at one
// rank, with no MPI call and no branch on anything but the size the Domain already holds, so
// the instruction sequence for a serial run is what it was before. That is not a nicety: the
// verification matrix is compared byte for byte, so a reduction that rounds differently at
// size 1 would show up as a regression in results that have nothing to do with MPI.
//
// smesh::Communicator exposes sum and max and nothing else, so min, any, all and argmin are
// built from those rather than from MPI_Allreduce directly -- one path to the communicator,
// and the spike does not reach past it to MPI.

#include "smesh_communicator.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

namespace cvfem {

/// The rank's share of a distributed problem: who to reduce with, and how much of the local
/// arrays is actually this rank's to contribute.
///
/// n_owned and n_local are both carried because the distinction is exactly what the bugs are
/// made of. A reduction must run over [0, n_owned) -- summing to n_local counts every shared
/// and ghost entry once per rank that holds it -- while a vector must be ALLOCATED to n_local,
/// because the operator writes ghost and aura slots. Loops that reduce use n_owned; loops that
/// allocate use n_local; at one rank they are equal and every such loop is unchanged.
struct Domain {
    std::shared_ptr<smesh::Communicator> comm;
    std::ptrdiff_t                       n_owned{0};
    std::ptrdiff_t                       n_local{0};

    /// True when there is anything to reduce with. Every function below short-circuits on
    /// this, so a serial run never enters MPI and never pays for the abstraction.
    inline bool distributed() const { return comm && comm->size() > 1; }

    inline int rank() const { return comm ? comm->rank() : 0; }
    inline int size() const { return comm ? comm->size() : 1; }

    /// Whether this rank should do the printing. Diagnostics are per-run, not per-rank, and a
    /// full node otherwise says everything 288 times.
    inline bool is_root() const { return rank() == 0; }
};

template <typename T>
inline T sum(const Domain &d, const T value) {
    if (!d.distributed()) return value;
    return d.comm->sum(value);
}

template <typename T>
inline T max(const Domain &d, const T value) {
    if (!d.distributed()) return value;
    return d.comm->max(value);
}

/// Built from max because the communicator has no min. Negating is exact in floating point --
/// it flips the sign bit and touches nothing else -- so this is the true minimum and not a
/// value that drifted through the detour.
template <typename T>
inline T min(const Domain &d, const T value) {
    if (!d.distributed()) return value;
    return -d.comm->max(-value);
}

/// Did ANY rank see this? The reduction is over int rather than bool because the
/// communicator's type mapping is built for arithmetic types.
inline bool any(const Domain &d, const bool value) {
    if (!d.distributed()) return value;
    return d.comm->max(value ? 1 : 0) != 0;
}

/// Did EVERY rank see this? Used for the decisions all ranks must agree on -- accepting a
/// Newton step, accepting an Armijo backtrack, declaring convergence. Disagreement there does
/// not produce a wrong answer, it produces a hang, because the ranks that continue enter a
/// collective the ranks that stopped never reach.
inline bool all(const Domain &d, const bool value) {
    if (!d.distributed()) return value;
    return !any(d, !value);
}

/// The minimum and the rank holding it, as two max calls.
///
/// Two rather than one because there is no MINLOC here: the first finds the value, the second
/// finds who has it. Ties go to the LOWEST rank -- max over (-rank) -- so the result is a
/// function of the values alone and not of which rank happened to answer first, which is what
/// makes it reproducible across runs.
template <typename T>
inline std::pair<T, int> argmin(const Domain &d, const T value, const int local_rank) {
    if (!d.distributed()) return {value, local_rank};
    const T   best = -d.comm->max(-value);
    const int who  = -d.comm->max(value == best ? -local_rank : -(1 << 30));
    return {best, who};
}

/// A sum that does not depend on how the problem was cut.
///
/// A plain sum over ranks reassociates when the partition changes, so the last bits of a
/// residual norm move with the rank count and a convergence history stops being comparable
/// between a 1-rank and a 4-rank run. This keeps the local compensated total in long double
/// and narrows only at the reduction, which does not make the sum exact but does stop the
/// local part of it from being the thing that drifts.
///
/// The local accumulation is deliberately NOT reordered relative to the serial code: the
/// caller passes the already-accumulated local value, so at one rank this is a return.
inline double sum_kahan(const Domain &d, const long double local) {
    if (!d.distributed()) return (double)local;
    return d.comm->sum((double)local);
}

}  // namespace cvfem

#endif  // CVFEM_PARALLEL_HPP
