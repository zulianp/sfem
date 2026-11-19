#ifndef SFEM_MULTIGRID_HPP
#define SFEM_MULTIGRID_HPP

#include <algorithm>
#include <math.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_ParallelOperator.hpp"
#include "sfem_SolverExec.hpp"
#include "sfem_aliases.hpp"

#include "sfem_BLAS_openmp.hpp"
#include "sfem_tpl_blas.hpp"

// https://en.wikipedia.org/wiki/Conjugate_gradient_method
namespace sfem {

    /// level 0 is the finest
    ///
    /// A cycle is written once for its executors (sfem_SolverExec.hpp), one per level: on the host,
    /// where every level's operator, smoother and transfers have a team form, the whole apply runs
    /// in one region, each level's rows split among its threads (TeamExec); otherwise, and on the
    /// device, vector by vector through the BLAS (BlasExec).
    template <typename T>
    class Multigrid final : public Operator<T>, public TeamOperator<T> {
    public:
        /// Empty, for the execution space es: levels are added after.
        static std::shared_ptr<Multigrid<T>> create(const ExecutionSpace es);
        std::shared_ptr<BLAS<T>> blas_;
        bool        verbose{true};
        bool        debug{false};

        // The value is the cycle index, the coarse-level visits per pass: V 1, W 2 (a larger index is valid).
        enum class CycleType : int { V = 1, W = 2 };

        enum CycleReturnCode { CYCLE_CONTINUE = 0, CYCLE_CONVERGED = 1, CYCLE_FAILURE = 2 };

        class Memory {
        public:
            SharedBuffer<T> rhs;
            SharedBuffer<T> solution;
            SharedBuffer<T> work;
            SharedBuffer<T> diag;
            inline ptrdiff_t size() const { return solution->size(); }
            ~Memory() {}
        };


        ExecutionContext execution_context() const override { return blas_ ? blas_->context() : ExecutionContext(EXECUTION_SPACE_INVALID); }

        int apply(const T* const rhs, T* const x) override {
            SFEM_TRACE_SCOPE("Multigrid::apply");

            ensure_init();
            if (team_capable()) return team_region_apply<T>(*this, rhs, x);

            BlasLevels levels{*this};
            run(levels, rhs, x);
            return 0;
        }

        // As a team operator (a preconditioner in a region): the finest level's rows.
        Team team() const override { return team_operator(operator_[finest_level()])->team(); }
        bool team_capable() const override {
            if (this->execution_space() != EXECUTION_SPACE_HOST || n_levels() == 0) return false;
            for (int l = 0; l < n_levels(); l++) {
                if (!capable(operator_[l]) || !capable(smoother_[l])) return false;
                if (l != coarsest_level() && !capable(restriction_[l])) return false;
                if (l != finest_level() && !capable(prolongation_[l])) return false;
            }
            return true;
        }
        int prepare() override {
            ensure_init();
            const int n = n_levels();
            team_operators_.assign(n, nullptr);
            team_smoothers_.assign(n, nullptr);
            team_restrictions_.assign(n, nullptr);
            team_prolongations_.assign(n, nullptr);
            team_sums_.resize(n);
            for (int l = 0; l < n; l++) {
                team_operators_[l] = team_operator(operator_[l]);
                team_smoothers_[l] = team_operator(smoother_[l]);
                if (l != coarsest_level()) team_restrictions_[l] = team_operator(restriction_[l]);
                if (l != finest_level()) team_prolongations_[l] = team_operator(prolongation_[l]);
                for (auto op : {team_operators_[l], team_smoothers_[l], team_restrictions_[l], team_prolongations_[l]}) {
                    if (op && op->prepare() != SFEM_SUCCESS) return SFEM_FAILURE;
                }
                team_reduction_buffer(team_sums_[l], level_owned_[l]);
            }
            return SFEM_SUCCESS;
        }
        // A run changes what it reports (iterations), the team leader's to write.
        void team_apply(const Team& team, const T* const rhs, T* const x) const override {
            TeamLevels levels{*this, team.tid, team.threads};
            const_cast<Multigrid*>(this)->run(levels, rhs, x);
        }

        void clear() {
            prolongation_.clear();
            restriction_.clear();
            smoother_.clear();
        }

        void set_cycle_type(const CycleType val) { cycle_type_ = val; }

        inline int n_levels() const { return smoother_.size(); }

        inline std::ptrdiff_t rows() const override { return operator_[finest_level()]->rows(); }
        inline std::ptrdiff_t cols() const override { return operator_[finest_level()]->cols(); }

        // Fine level prolongation has to be null
        // Coarse level restriction has to be null
        inline void add_level(const std::shared_ptr<Operator<T>>& op,
                              const std::shared_ptr<Operator<T>>& smoother_or_solver,
                              const std::shared_ptr<Operator<T>>& prolongation,
                              const std::shared_ptr<Operator<T>>& restriction) {
            operator_.push_back(op);
            smoother_.push_back(smoother_or_solver);
            prolongation_.push_back(prolongation);
            restriction_.push_back(restriction);
        }

        void default_init() {
            blas() = OpenMPBLAS<T>::create();
        }

        inline std::shared_ptr<BLAS<T>>& blas() { return blas_; }

        void set_max_it(const int val) { max_it_ = val; }

        void set_atol(const T val) { atol_ = val; }

        // The caller promises that x is zero when apply is called, as it is for a
        // preconditioner inside a Krylov method. Coarse levels need no promise: the cycle
        // zeroes their solution before descending.
        void set_initial_guess_zero(const bool val) { initial_guess_zero_ = val; }

        int test_interp() {
            ensure_init();

            auto finest_A   = operator_[0];
            int  finest_dim = finest_A->rows();
            auto finest_vec = create_host_buffer<T>(finest_dim);
            auto finest_out = create_host_buffer<T>(finest_dim);
            for (int i = 0; i < finest_dim; i++) {
                finest_vec->data()[i] = 1;
            }
            finest_A->apply(finest_vec->data(), finest_out->data());
            real_t should_be_zero = this->blas()->norm2(finest_dim, finest_out->data());
            printf("||A 1|| = %f\n", should_be_zero);

            int failure = 0;
            for (int level = 0; level < n_levels() - 1; level++) {
                auto pt         = restriction_[level];
                auto p          = prolongation_[level + 1];
                auto A          = operator_[level];
                auto Ac         = operator_[level + 1];
                int  coarse_dim = Ac->rows();
                int  fine_dim   = A->rows();

                assert(p->rows() == coarse_dim);
                assert(p->cols() == fine_dim);
                assert(pt->rows() == fine_dim);
                assert(pt->cols() == coarse_dim);

                auto coarse_vec = create_host_buffer<T>(coarse_dim);
                for (int i = 0; i < coarse_dim; i++) {
                    coarse_vec->data()[i] = 1;
                }

                auto out1 = create_host_buffer<T>(coarse_dim);
                Ac->apply(coarse_vec->data(), out1->data());
                T ac1 = this->blas()->norm2(coarse_dim, out1->data());
                printf("Level %d: ||Ac 1|| = %f\n", level, ac1);

                auto temp  = create_host_buffer<T>(fine_dim);
                auto temp2 = create_host_buffer<T>(fine_dim);
                auto out2  = create_host_buffer<T>(coarse_dim);
                p->apply(coarse_vec->data(), temp->data());
                /* only nueman
                    for (int i = 0; i < fine_dim; i++) {
                        assert(fabs(temp->data()[i] - 1) < 1e-8);
                    }
                    */

                A->apply(temp->data(), temp2->data());
                real_t should_be_zero = this->blas()->norm2(fine_dim, temp2->data());
                printf("level: %d ||A p 1|| = %f\n", level, should_be_zero);
                pt->apply(temp2->data(), out2->data());

                this->blas()->axpby(coarse_dim, 1, out1->data(), -1, out2->data());
                T err_norm = this->blas()->norm2(coarse_dim, out2->data());
                printf("Level %d: ||Ac 1 - pt A p 1|| = %f\n", level, err_norm);
                if (err_norm > 1e-8) {
                    failure++;
                }
            }
            return failure;
        }

        std::vector<std::shared_ptr<Operator<T>>>& operators() { return operator_; }
        std::vector<std::shared_ptr<Operator<T>>>& restrictions() { return restriction_; }
        std::vector<std::shared_ptr<Operator<T>>>  smoothers() { return smoother_; }


    private:
        std::vector<std::shared_ptr<Operator<T>>> operator_;
        std::vector<std::shared_ptr<Operator<T>>> smoother_;

        std::vector<std::shared_ptr<Operator<T>>> prolongation_;
        std::vector<std::shared_ptr<Operator<T>>> restriction_;

        // Internals: each level's vectors (the finest level's rhs and solution are the caller's),
        // and what a solve on its operator spans (owned rows, allocation, ranks).
        std::vector<std::shared_ptr<Memory>> memory_;
        std::vector<std::ptrdiff_t>          level_owned_;
        std::vector<std::ptrdiff_t>          level_alloc_;
        std::vector<SolveSpan<T>>            level_span_;

        // The levels in team form and their sums (prepare).
        std::vector<std::shared_ptr<TeamOperator<T>>> team_operators_;
        std::vector<std::shared_ptr<TeamOperator<T>>> team_smoothers_;
        std::vector<std::shared_ptr<TeamOperator<T>>> team_restrictions_;
        std::vector<std::shared_ptr<TeamOperator<T>>> team_prolongations_;
        mutable std::vector<std::vector<T>>           team_sums_;

        int       max_it_{10};
        int       iterations_{0};
        CycleType cycle_type_{CycleType::V};
        T         atol_{1e-10};
        bool      initial_guess_zero_{false};

        static std::shared_ptr<TeamOperator<T>> team_operator(const std::shared_ptr<Operator<T>>& op) {
            return std::dynamic_pointer_cast<TeamOperator<T>>(op);
        }
        static bool capable(const std::shared_ptr<Operator<T>>& op) {
            auto t = team_operator(op);
            return t && t->team_capable();
        }

        // Tell a smoother whether its solution is zero on entry. Only a MatrixFreeLinearSolver
        // can use the hint; any other operator is applied unchanged.
        static void hint_zero_guess(const std::shared_ptr<Operator<T>>& smoother, const bool zero) {
            if (auto s = std::dynamic_pointer_cast<MatrixFreeLinearSolver<T>>(smoother)) s->set_initial_guess_zero(zero);
        }

        // The finest level's residual norms of a run, for its report.
        struct RunState {
            int it{0};
            T   norm0{1};
            T   previous{1};
        };

        // A level's executor over the threads of the region: the level's rows split among the same
        // threads (thread 0 takes none under MPI), its sums over the ranks.
        struct TeamLevel {
            Team                team;
            TeamReduction<T, 1> sums;
            TeamExec<T>         e;

            TeamLevel(const Multigrid& mg, const int l, const int tid, const int threads)
                : team(Team::of(mg.level_owned_[l], tid, threads, mg.level_span_[l].comm != nullptr)),
                  sums(mg.team_sums_[l].data(), team_chunk_count(mg.level_owned_[l]), mg.level_span_[l].ranks()),
                  e(team, sums) {}
            TeamLevel(const TeamLevel&) = delete;
        };

        struct TeamLevels {
            const Multigrid& mg;
            int              tid, threads;

            TeamLevel         make(const int l) const { return TeamLevel(mg, l, tid, threads); }
            TeamOperator<T>&  op(const int l) const { return *mg.team_operators_[l]; }
            TeamOperator<T>&  restriction(const int l) const { return *mg.team_restrictions_[l]; }
            TeamOperator<T>&  prolongation(const int l) const { return *mg.team_prolongations_[l]; }

            void smooth(TeamExec<T>& e, const int l, const T* const b, T* const x, const bool zero_guess) const {
                auto& s = *mg.team_smoothers_[l];
                if (!s.input_in_share()) e.barrier();
                s.team_solve(e.team(), b, x, zero_guess);
            }
            int solve_coarse(TeamExec<T>& e, const int l, const T* const b, T* const x) const {
                auto& s = *mg.team_smoothers_[l];
                if (!s.input_in_share()) e.barrier();
                s.team_apply(e.team(), b, x);
                return SFEM_SUCCESS;
            }
            // Under MPI a restriction adds into its output's ghost and aura entries before it sends
            // them to their owners: they start from zero (the leader's, before the product's barrier).
            void zero_ghosts(TeamExec<T>& e, const int l, T* const y) const {
                if (!e.leader()) return;
                for (std::ptrdiff_t i = mg.level_owned_[l]; i < mg.level_alloc_[l]; i++) y[i] = 0;
            }
        };

        // A level's executor through the BLAS: its owned rows, its allocation zeroed before a product.
        struct BlasLevel {
            BlasExec<T> e;

            BlasLevel(Multigrid& mg, const int l)
                : e(*mg.blas(), mg.level_owned_[l], mg.level_alloc_[l], mg.level_span_[l].ranks()) {}
            BlasLevel(const BlasLevel&) = delete;
        };

        struct BlasLevels {
            Multigrid& mg;

            BlasLevel     make(const int l) const { return BlasLevel(mg, l); }
            Operator<T>&  op(const int l) const { return *mg.operator_[l]; }
            Operator<T>&  restriction(const int l) const { return *mg.restriction_[l]; }
            Operator<T>&  prolongation(const int l) const { return *mg.prolongation_[l]; }

            void smooth(BlasExec<T>&, const int l, const T* const b, T* const x, const bool zero_guess) const {
                hint_zero_guess(mg.smoother_[l], zero_guess);
                mg.smoother_[l]->apply(b, x);
                hint_zero_guess(mg.smoother_[l], false);
            }
            int  solve_coarse(BlasExec<T>&, const int l, const T* const b, T* const x) const { return mg.smoother_[l]->apply(b, x); }
            void zero_ghosts(BlasExec<T>&, const int, T* const) const {}
        };

        inline int finest_level() const { return 0; }

        inline int coarsest_level() const { return n_levels() - 1; }

        inline int coarser_level(int level) const { return level + 1; }

        inline int finer_level(int level) const { return level - 1; }

        void ensure_init() {
            if (memory_.empty()) {
                init();
            }
        }

        bool verbose_rank0() const { return level_span_.empty() || level_span_[finest_level()].root(); }

        int init() {
            assert(prolongation_.size() == restriction_.size());
            assert(operator_.size() == smoother_.size());

            memory_.clear();
            memory_.resize(this->n_levels());
            level_owned_.resize(this->n_levels());
            level_alloc_.resize(this->n_levels());
            level_span_.resize(this->n_levels());

            for (int l = 0; l < n_levels(); l++) {
                // Distributed (more than one rank): the operator's owned rows and its allocation.
                level_span_[l]  = SolveSpan<T>::of(*operator_[l]);
                level_owned_[l] = level_span_[l].comm ? operator_[l]->rows() : smoother_[l]->rows();
                level_alloc_[l] = level_span_[l].comm ? level_span_[l].allocation : smoother_[l]->rows();

                memory_[l] = std::make_shared<Memory>();

                const size_t n         = static_cast<size_t>(level_alloc_[l]);
                auto         blas_impl = this->blas();
                auto         own       = [&]() {
                    return Buffer<T>::own(n, blas_impl->allocate(n), [blas_impl](void* ptr) { blas_impl->destroy(ptr); });
                };
                if (l != finest_level()) {
                    memory_[l]->solution = own();
                    memory_[l]->rhs      = own();
                }
                memory_[l]->work = own();
            }

            return 0;
        }

        /// The cycles, written once for their executors.
        template <class Levels>
        void run(Levels& levels, const T* const b, T* const x) {
            RunState s;
            int      it = 0;
            for (; it < max_it_; it++) {
                s.it = it;
                if (cycle(levels, finest_level(), b, x, s) == CYCLE_CONVERGED) {
                    break;
                }
            }
            auto lv = levels.make(finest_level());
            if (lv.e.leader()) iterations_ = it;
        }

        template <class E>
        T norm2(E& e, const T* const v) {
            return std::sqrt(e.dot(v, v));
        }

        template <class Levels>
        CycleReturnCode cycle(Levels& levels, const int level, const T* const b, T* const x, RunState& s) {
            auto  lv   = levels.make(level);
            auto& e    = lv.e;
            T*    work = memory_[level]->work->data();
            const bool say = verbose && e.leader() && verbose_rank0();

            if (coarsest_level() == level) {
                e.zeros(x);
                if (levels.solve_coarse(e, level, b, x) != SFEM_SUCCESS) return CYCLE_FAILURE;
                if (debug) {
                    e.apply(levels.op(level), x, work, false);
                    e.axpby(1, b, -1, work);
                    const T r = norm2(e, work);
                    if (e.leader()) printf("|| r_H || = %g\n", (double)r);
                }
                return CYCLE_CONTINUE;
            }

            const int  coarse   = coarser_level(level);
            auto       lc       = levels.make(coarse);
            auto&      ec       = lc.e;
            T* const   b_coarse = memory_[coarse]->rhs->data();
            T* const   x_coarse = memory_[coarse]->solution->data();

            for (int k = 0; k < static_cast<int>(this->cycle_type_); k++) {
                // Pre-smoothing starts from zero on a coarse level's first pass (the parent
                // zeroed it) and on the finest level's first pass of the first cycle when the
                // caller said so; nowhere else. Post-smoothing always starts from the corrected,
                // non-zero solution.
                const bool zero_guess = k == 0 && (level != finest_level() || (initial_guess_zero_ && s.it == 0));
                levels.smooth(e, level, b, x, zero_guess);

                {
                    // Compute residual
                    e.apply(levels.op(level), x, work, false);
                    e.axpby(1, b, -1, work);

                    if (finest_level() == level) {
                        const T norm_residual = norm2(e, work);

                        if (s.it == 0) {
                            s.norm0    = norm_residual;
                            s.previous = norm_residual;

                            if (say) {
                                printf("Multigrid\n");
                                printf("iter\tabs\t\trel\t\trate\n");
                                printf("%d\t%g\t-\t\t-\n", s.it, (double)(norm_residual));
                            }
                        } else if (say) {
                            printf("%d\t%g\t%g\t%g\n",
                                   s.it,
                                   (double)(norm_residual),
                                   (double)(norm_residual / s.norm0),
                                   (double)(norm_residual / s.previous));

                            fflush(stderr);
                            fflush(stdout);
                        }

                        s.previous = norm_residual;
                        if (norm_residual < atol_) {
                            return CYCLE_CONVERGED;
                        }
                    }
                }

                // Restriction
                levels.zero_ghosts(ec, coarse, b_coarse);
                ec.apply(levels.restriction(level), work, b_coarse, false);
                ec.zeros(x_coarse);

                CycleReturnCode ret = cycle(levels, coarse, b_coarse, x_coarse, s);
                assert(ret != CYCLE_FAILURE);
                SFEM_UNUSED(ret);

                {
                    if (debug) {
                        const T c = norm2(ec, x_coarse);
                        if (e.leader()) printf("|| c_H || = %g\n", (double)c);
                    }

                    // Prolongation
                    e.apply(levels.prolongation(coarse), x_coarse, work, false);

                    if (debug) {
                        const T c = norm2(e, work);
                        if (e.leader()) printf("|| c_h || = %g\n", (double)c);
                    }

                    // Apply coarse space correction
                    e.axpby(1, work, 1, x);
                }

                if (debug) {
                    e.apply(levels.op(level), x, work, false);
                    e.axpby(1, b, -1, work);
                    const T r = norm2(e, work);
                    if (e.leader()) printf("|| r_h || = %g\n", (double)r);
                }

                levels.smooth(e, level, b, x, false);
            }

            return CYCLE_CONTINUE;
        }
    };

    template <typename T>
    std::shared_ptr<Multigrid<T>> h_mg() {
        auto mg = std::make_shared<Multigrid<T>>();
        mg->default_init();
        return mg;
    }

    template <typename T>
    std::shared_ptr<Multigrid<T>> create_mg(const ExecutionSpace es) {
        return Multigrid<T>::create(es);
    }

}  // namespace sfem

#endif  // SFEM_MULTIGRID_HPP
