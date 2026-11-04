#ifndef SFEM_TEAM_OPERATOR_HPP
#define SFEM_TEAM_OPERATOR_HPP

#include <cassert>
#include <functional>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sfem_Operator.hpp"
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

        /// team_apply of a solver (y the solution, x the right-hand side) told per call whether y is
        /// zero on entry, which a smoother can use to skip its first product: inside a region the
        /// hint cannot be the solver's state (MatrixFreeLinearSolver::set_initial_guess_zero). An
        /// operator that cannot use it applies as usual.
        virtual void team_solve(const Team& team, const T* const x, T* const y, const bool /*zero_guess*/) const {
            team_apply(team, x, y);
        }

        /// Inside a region, the calling thread's share of the rows.
        virtual Team team() const = 0;

        virtual bool input_in_share() const { return false; }

        /// Whether team_apply may be taken now: an operator whose team form rests on another's
        /// (a smoother over its operator) has one only where that one has.
        virtual bool team_capable() const { return true; }

        /// Outside any region, before the products of a region: checks and scratch.
        virtual int prepare() { return SFEM_SUCCESS; }

        /// After the region, a solver's outcome of its last team_apply (its leader's to write), as
        /// its plain apply returns it.
        virtual int team_status() const { return SFEM_SUCCESS; }
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

    /// A host operator given by its team form, a function of the team (rows: the entries its teams
    /// split; under MPI, `reserve_first`, thread 0 takes none). Applied from outside a region, one
    /// region around it.
    template <typename T>
    class TeamLambdaOperator final : public Operator<T>, public TeamOperator<T> {
    public:
        using TeamApply = std::function<void(const Team&, const T* const, T* const)>;
        using Prepare   = std::function<int()>;

        TeamLambdaOperator(const std::ptrdiff_t rows,
                           const std::ptrdiff_t cols,
                           const bool           reserve_first,
                           TeamApply            team_apply,
                           Prepare              prepare        = {},
                           const bool           input_in_share = false)
            : rows_(rows),
              cols_(cols),
              reserve_first_(reserve_first),
              input_in_share_(input_in_share),
              team_apply_(std::move(team_apply)),
              prepare_(std::move(prepare)) {}

        int  apply(const T* const x, T* const y) override { return team_region_apply<T>(*this, x, y); }
        Team team() const override { return Team::current(rows_, reserve_first_); }
        void team_apply(const Team& team, const T* const x, T* const y) const override { team_apply_(team, x, y); }
        bool input_in_share() const override { return input_in_share_; }
        int  prepare() override { return prepare_ ? prepare_() : SFEM_SUCCESS; }

        std::ptrdiff_t   rows() const override { return rows_; }
        std::ptrdiff_t   cols() const override { return cols_; }
        ExecutionContext execution_context() const override { return EXECUTION_SPACE_HOST; }

    private:
        std::ptrdiff_t rows_{0};
        std::ptrdiff_t cols_{0};
        bool           reserve_first_{false};
        bool           input_in_share_{false};
        TeamApply      team_apply_;
        Prepare        prepare_;
    };

    template <typename T>
    inline std::shared_ptr<TeamLambdaOperator<T>> make_team_lambda_op(const std::ptrdiff_t                       rows,
                                                                      const std::ptrdiff_t                       cols,
                                                                      const bool                                 reserve_first,
                                                                      typename TeamLambdaOperator<T>::TeamApply  team_apply,
                                                                      typename TeamLambdaOperator<T>::Prepare    prepare        = {},
                                                                      const bool                                 input_in_share = false) {
        return std::make_shared<TeamLambdaOperator<T>>(
                rows, cols, reserve_first, std::move(team_apply), std::move(prepare), input_in_share);
    }

}  // namespace sfem

#endif  // SFEM_TEAM_OPERATOR_HPP
