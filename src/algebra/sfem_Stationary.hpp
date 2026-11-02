#ifndef SFEM_STATIONARY_HPP
#define SFEM_STATIONARY_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <cmath>
#include <memory>
#include <vector>

#include "sfem_LpSmoother.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_ParallelOperator.hpp"
#include "sfem_aliases.hpp"
#include "sfem_BLAS_openmp.hpp"
#include "sfem_SolverExec.hpp"
#include "sfem_operator_factory.hpp"

namespace sfem {

    template <typename T>
    class StationaryIteration final : public MatrixFreeLinearSolver<T>, public TeamOperator<T> {
    public:
        /// Built for the execution space es: op preconditioned by preconditioner.
        static std::shared_ptr<StationaryIteration<T>> create(const std::shared_ptr<Operator<T>> &op,
                                                              const std::shared_ptr<Operator<T>> &preconditioner,
                                                              const ExecutionSpace                es);
        ptrdiff_t                    n_dofs{SFEM_PTRDIFF_INVALID};
        ptrdiff_t                    n_alloc{SFEM_PTRDIFF_INVALID};
        int                          max_it{3};
        SharedBuffer<T>              workspace;
        std::shared_ptr<Operator<T>> op;
        std::shared_ptr<Operator<T>> preconditioner;
        bool                         verbose{false};
        bool                         use_arg_as_first_residual{false};
        std::shared_ptr<BLAS<T>> blas;

        mutable int iterations_{0};

        std::function<void(T*)> interceptor;

        int iterations() const override { return iterations_; }

        void configure_sizes_from_op() {
            n_dofs = op ? op->rows() : SFEM_PTRDIFF_INVALID;
            span_  = op ? SolveSpan<T>::of(*op) : SolveSpan<T>{};
            if (span_.comm) {
                n_alloc = span_.allocation;
            } else if (n_alloc < n_dofs) {
                n_alloc = n_dofs;
            }
        }

        void ensure_workspace() {
            const ptrdiff_t n = (n_alloc > 0) ? n_alloc : n_dofs;
            if (!workspace || workspace->size() != static_cast<size_t>(n)) {
                auto blas_impl = blas;
                workspace = Buffer<T>::own(n, blas_impl->allocate(n), [blas_impl](void* ptr) { blas_impl->destroy(ptr); });
            }
        }

        void default_init() {
            blas = OpenMPBLAS<T>::create();
        }

        /* Operator */
        int apply(const T* const b, T* const x) override {
            SFEM_TRACE_SCOPE("StationaryIteration::apply");
            ensure_workspace();
            if (team_capable()) {
                team_reduction_buffer(team_sums_, n_dofs);
                return team_region_apply<T>(*this, b, x);
            }
            BlasExec<T> e(*blas, n_dofs, workspace->size(), span_.ranks());
            run(e, *op, *preconditioner, b, x);
            return SFEM_SUCCESS;
        }

        // As a team operator (a smoother in a region): where the operator and the preconditioner
        // have team forms, on one rank of the host.
        Team team() const override { return team_op()->team(); }
        bool team_capable() const override {
            auto A = team_op();
            auto P = team_prec();
            return A && P && A->team_capable() && P->team_capable() && n_alloc <= n_dofs &&
                   this->execution_space() == EXECUTION_SPACE_HOST;
        }
        int prepare() override {
            ensure_workspace();
            if (auto A = team_op(); A && A->prepare() != SFEM_SUCCESS) return SFEM_FAILURE;
            auto P = team_prec();
            return P ? P->prepare() : SFEM_SUCCESS;
        }
        void team_apply(const Team& team, const T* const b, T* const x) const override {
            TeamReduction<T, 1> sums(team_sums_.data(), team_chunk_count(n_dofs), span_.ranks());
            TeamExec<T>         e(team, sums);
            run(e, *team_op(), *team_prec(), b, x);
        }
        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }
        ExecutionContext      execution_context() const override { return blas ? blas->context() : ExecutionContext(EXECUTION_SPACE_INVALID); }

        /* MatrixFreeLinearSolver */
        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->op = op;
            configure_sizes_from_op();
        }
        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override { this->preconditioner = op; }
        // A zero initial guess makes the first residual the right-hand side itself, which is
        // what use_arg_as_first_residual does: the preconditioner is applied to b and the
        // operator application on a zero vector is skipped. The result is the same bits --
        // b - A 0 is b. Multigrid pre-smoothing is the caller, since it always starts from zero.
        void set_initial_guess_zero(const bool val) override { use_arg_as_first_residual = val; }
        void set_max_it(const int it) override { max_it = it; }
        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }

        int set_op_and_diag_shift(const std::shared_ptr<Operator<T>>& op, const SharedBuffer<T>& diag) override {
            this->op = op;
            configure_sizes_from_op();
            auto shiftable = std::dynamic_pointer_cast<ShiftableOperator<T>>(preconditioner);
            if (shiftable) {
                return shiftable->shift(diag);
            } else {
                SFEM_ERROR("Tried to call shift on object that is not subclass of ShiftableOperator!\n");
                return SFEM_FAILURE;
            }
        }

        int set_op_and_diag_shift(const std::shared_ptr<Operator<T>>&          op,
                                  const std::shared_ptr<SparseBlockVector<T>>& sbv,
                                  const SharedBuffer<T>&                       diag) override {
            assert(sbv->n_blocks() == (ptrdiff_t)diag->size());

            this->op = op;
            configure_sizes_from_op();

            auto shiftable = std::dynamic_pointer_cast<ShiftableOperator<T>>(preconditioner);
            if (shiftable) {
                return shiftable->shift(sbv, diag);
            } else {
                SFEM_ERROR("Tried to call shift on object that is not subclass of ShiftableOperator!\n");
                return SFEM_FAILURE;
            }

            return SFEM_FAILURE;
        }

    private:
        mutable std::vector<T> team_sums_;
        SolveSpan<T>           span_;

        // The operator and the preconditioner in team form (the fields are public, set directly).
        std::shared_ptr<TeamOperator<T>> team_op() const { return std::dynamic_pointer_cast<TeamOperator<T>>(op); }
        std::shared_ptr<TeamOperator<T>> team_prec() const { return std::dynamic_pointer_cast<TeamOperator<T>>(preconditioner); }

        /// x += P (b - A x), max_it times, written once for its executor (sfem_SolverExec.hpp);
        /// P accumulates into x.
        template <class E, class Op, class Prec>
        void run(E& e, Op& A, Prec& P, const T* const b, T* const x) const {
            int it = 0;
            if (use_arg_as_first_residual) {
                it = 1;
                e.apply(P, b, x, true);
            }
            T* const r = workspace->data();
            for (; it < max_it; it++) {
                e.apply(A, x, r, false);
                e.axpby(T(1), b, T(-1), r);
                if (verbose) {
                    const T norm_residual = std::sqrt(e.dot(r, r));
                    if (e.leader() && span_.root()) printf("%d : %f\n", it, (double)norm_residual);
                }
                e.apply(P, r, x, true);
                if (interceptor) {
                    e.barrier();
                    if (e.leader()) interceptor(x);
                    e.barrier();
                }
            }
            if (e.leader()) iterations_ = it;
        }
    };

    template <typename T>
    std::shared_ptr<StationaryIteration<T>> h_stationary(const std::shared_ptr<Operator<T>>& op,
                                                         const std::shared_ptr<Operator<T>>& preconditioner) {
        auto solver            = std::make_shared<StationaryIteration<T>>();
        solver->op             = op;
        solver->preconditioner = preconditioner;
        solver->configure_sizes_from_op();
        solver->default_init();
        return solver;
    }
    template <typename T>
    std::shared_ptr<StationaryIteration<T>> create_stationary(const std::shared_ptr<Operator<T>> &op,
                                                              const std::shared_ptr<Operator<T>> &preconditioner,
                                                              const ExecutionSpace                es) {
        return StationaryIteration<T>::create(op, preconditioner, es);
    }

}  // namespace sfem

#endif  // SFEM_STATIONARY_HPP
