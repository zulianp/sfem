#ifndef SFEM_TEAM_OPERATOR_HPP
#define SFEM_TEAM_OPERATOR_HPP

#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sfem_Team.hpp"
#include "sfem_base.hpp"

namespace sfem {

    /// A host operator whose product runs inside a parallel region its caller opened: each thread
    /// of the region calls team_apply with its team(), may wait for the others (team.barrier())
    /// and, on return, its share of the rows, [team.begin, team.end), is complete; an operator
    /// whose last phase writes other threads' rows ends on a barrier. Every thread reaches every
    /// barrier: team_apply does not return early, and what can fail is checked by prepare(), once,
    /// before the region. Under MPI thread 0 takes no rows: it keeps the messages moving (a
    /// rendezvous advances only while a thread is in MPI). With input_in_share() team_apply reads x
    /// beyond the calling thread's share only after a barrier of its own, so a caller that has
    /// just written its share of x need not wait for the other threads first.
    template <typename T>
    class TeamOperator {
    public:
        virtual ~TeamOperator() = default;

        /// From inside a region, every thread with its own team().
        virtual void team_apply(const Team& team, const T* const x, T* const y) const = 0;

        /// Inside a region, the calling thread's share of the rows.
        virtual Team team() const = 0;

        virtual bool input_in_share() const { return false; }

        /// Outside any region, before the products of a region: checks and scratch.
        virtual int prepare() { return SFEM_SUCCESS; }
    };

    /// y = A x from outside a region: the one way a team operator is applied there, one region
    /// around its team_apply.
    template <typename T>
    int team_region_apply(TeamOperator<T>& op, const T* const x, T* const y) {
#ifdef _OPENMP
        assert(!omp_in_parallel());
#endif
        const int err = op.prepare();
        if (err != SFEM_SUCCESS) return err;
#pragma omp parallel
        op.team_apply(op.team(), x, y);
        return SFEM_SUCCESS;
    }

}  // namespace sfem

#endif  // SFEM_TEAM_OPERATOR_HPP
