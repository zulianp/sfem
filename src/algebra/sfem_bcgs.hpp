#ifndef SFEM_BCGS_HPP
#define SFEM_BCGS_HPP

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_BLAS_openmp.hpp"
#include "sfem_SolverExec.hpp"

#include <memory>
#include <vector>

// https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
namespace sfem {
    template <typename T>
    class BiCGStab final : public MatrixFreeLinearSolver<T> {
    public:
        /// Built for the execution space es, on op.
        static std::shared_ptr<BiCGStab<T>> create(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es);
        // Operator
        std::function<void(const T* const, T* const)> apply_op;
        std::function<void(const T* const, T* const)> left_preconditioner_op;
        std::function<void(const T* const, T* const)> right_preconditioner_op;
        std::shared_ptr<BLAS<T>> blas;

        ptrdiff_t n_dofs{SFEM_PTRDIFF_INVALID};
        int       iterations_{0};
        bool      diverged_{false};

        bool verbose{true};


        int iterations() const override { return iterations_; }

        ExecutionContext execution_context() const override { return blas ? blas->context() : ExecutionContext(EXECUTION_SPACE_INVALID); }

        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->apply_op = [=](const T* const x, T* const y) { op->apply(x, y); };
            op_            = op;
        }

        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override {
            this->right_preconditioner_op = [=](const T* const x, T* const y) { op->apply(x, y); };
            prec_                         = op;
        }

        void set_max_it(const int it) override { max_it = it; }

        void set_preconditioner(std::function<void(const T* const, T* const)>&& in) {
            // left_preconditioner_op = in;
            right_preconditioner_op = in;
            prec_.reset();
        }

        // Solver parameters
        T   atol{1e-16};
        T   rtol{1e-10};
        int max_it{10000};
        // Growth factor at which the solve is abandoned as divergent; 0 disables the test.
        // The non-finite guards below catch a blow-up only once it has already overflowed,
        // by which point the whole iteration budget has been spent. An iteration limit is a
        // backstop, not a stopping criterion: a solve that is clearly failing should say so
        // while the caller can still act on it.
        T   dtol{0};

        void set_atol(const T val) { atol = val; }
        void set_rtol(const T val) { rtol = val; }
        void set_dtol(const T val) { dtol = val; }

        void default_init() {
            blas = OpenMPBLAS<T>::create();
        }

        bool good() const {
            assert(apply_op);
            return blas->good() && apply_op;
        }

        bool converged(const T residual, const T residual0) const {
            return residual < atol || (residual0 > 0 && residual / residual0 < rtol);
        }

        bool diverging(const T residual, const T residual0) const {
            return dtol > 0 && residual0 > 0 && residual > dtol * residual0;
        }

        // Why the last solve stopped. A caller comparing iteration counts needs this: a run
        // that stopped on max_it reports a floor, not a result, and treating it as a result
        // silently inflates any speedup measured against it.
        bool has_diverged() const { return diverged_; }
        bool hit_max_it() const { return iterations_ >= max_it; }

        void monitor(const int iter, const T residual, const T residual0) {
            if (!verbose) return;
            const T rel = (residual0 > 0) ? (residual / residual0) : residual;
            if (iter == max_it || iter % 100 == 0 || converged(residual, residual0)) {
                std::cout << iter << ": residual abs: " << residual << ", rel: " << rel << " (rtol = " << rtol
                          << ", atol = " << atol << ")\n";
            }
        }

        /// One iteration, written once for its executor (sfem_SolverExec.hpp): on the host, when
        /// the operator (and the preconditioner) have a team form, the whole run in one region;
        /// otherwise vector by vector through the BLAS.
        int apply(const ptrdiff_t n, const T* const b, T* const x) {
            SFEM_TRACE_SCOPE("BiCGStab::apply");
            iterations_ = 0;
            diverged_   = false;
            if (!good()) {
                return SFEM_FAILURE;
            }
            const bool precond = left_preconditioner_op || right_preconditioner_op;

            T* w[7];
            for (auto& v : w) v = blas->allocate(n);

            auto team_op   = std::dynamic_pointer_cast<TeamOperator<T>>(op_);
            auto team_prec = std::dynamic_pointer_cast<TeamOperator<T>>(prec_);
            int  info      = SFEM_FAILURE;
            if (team_op && team_op->team_capable() && (!precond || (team_prec && team_prec->team_capable())) &&
                this->execution_space() == EXECUTION_SPACE_HOST && team_op->prepare() == SFEM_SUCCESS &&
                (!precond || team_prec->prepare() == SFEM_SUCCESS)) {
                T* const buffer = team_reduction_buffer(team_sums_, n);
#pragma omp parallel
                {
                    TeamReduction<T, 1> sums(buffer, team_chunk_count(n));
                    const Team          team = team_op->team();
                    TeamExec<T>         e(team, sums);
                    const int r = precond ? run_precond(e, *team_op, *team_prec, b, x, w) : run_basic(e, *team_op, b, x, w);
                    if (team.leader()) info = r;
                }
            } else {
                BlasExec<T> e(*blas, n);
                info = precond ? run_precond(e, apply_op, right_preconditioner_op, b, x, w) : run_basic(e, apply_op, b, x, w);
            }

            for (auto& v : w) blas->destroy(v);
            return info;
        }

        int apply(const T* const b, T* const x) override {
            assert(n_dofs >= 0);
            if (this->n_dofs < 0) {
                std::cerr << "Error uninitiaized n_dofs. Set set_n_dofs to set the number of dofs\n";
                return 1;
            }

            return apply(this->n_dofs, b, x);
        }

        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }

    private:
        std::shared_ptr<Operator<T>> op_, prec_;
        std::vector<T>               team_sums_;

        template <class E>
        void say(E& e, const int iter, const T residual, const T residual0) {
            if (e.leader()) monitor(iter, residual, residual0);
        }

        template <class E, class Op>
        int run_basic(E& e, Op& A, const T* const b, T* const x, T* const (&w)[7]) {
            T* const r0 = w[0];
            T* const r  = w[1];
            T* const p  = w[2];
            T* const v  = w[3];
            T* const h  = w[4];
            T* const s  = w[5];
            T* const t  = w[6];

            // Residual
            e.apply(A, x, r0, false);
            e.axpby(1, b, -1, r0);

            T rho = e.dot(r0, r0);

            const T r_norm0 = sqrt(rho);
            if (converged(r_norm0, r_norm0)) {
                return SFEM_SUCCESS;
            }

            e.copy(r0, r);
            e.copy(r0, p);

            int info = SFEM_FAILURE;
            int it   = 0;
            for (; it < max_it; it++) {
                e.apply(A, p, v, false);

                const T ptv = e.dot(r0, v);

                if (ptv == 0) {
                    info = SFEM_FAILURE;
                    break;
                }

                const T alpha = rho / ptv;

                e.zaxpby(1, x, alpha, p, h);
                e.zaxpby(1, r, -alpha, v, s);

                const T sts    = e.dot(s, s);
                const T s_norm = sqrt(sts);

                if (converged(s_norm, r_norm0)) {
                    say(e, it, s_norm, r_norm0);
                    e.copy(h, x);
                    info = SFEM_SUCCESS;
                    break;
                }

                e.apply(A, s, t, false);

                const T tts = e.dot(t, s);
                const T ttt = e.dot(t, t);

                if (ttt == 0 || !std::isfinite(tts) || !std::isfinite(ttt)) {
                    e.copy(h, x);
                    info = SFEM_FAILURE;
                    break;
                }

                const T omega = tts / ttt;
                if (!std::isfinite(omega)) {
                    e.copy(h, x);
                    info = SFEM_FAILURE;
                    break;
                }

                e.zaxpby(1, h, omega, s, x);
                e.zaxpby(1, s, -omega, t, r);

                const T rtr    = e.dot(r, r);
                const T r_norm = sqrt(rtr);
                if (!std::isfinite(r_norm)) {
                    e.copy(h, x);
                    info = SFEM_FAILURE;
                    break;
                }

                if (diverging(r_norm, r_norm0)) {
                    e.copy(h, x);
                    if (e.leader()) diverged_ = true;
                    info = SFEM_FAILURE;
                    break;
                }

                say(e, it, r_norm, r_norm0);
                if (converged(r_norm, r_norm0)) {
                    info = SFEM_SUCCESS;
                    break;
                }

                const T rho_new = e.dot(r0, r);
                const T beta    = (rho_new / rho) * (alpha / omega);
                rho             = rho_new;

                e.axpby(1, r, beta, p);
                e.axpby(-omega * beta, v, 1, p);
            }
            if (e.leader()) iterations_ = it;
            return info;
        }

        template <class E, class Op, class Prec>
        int run_precond(E& e, Op& A, Prec& M, const T* const b, T* const x, T* const (&w)[7]) {
            T* const r0 = w[0];
            T* r        = w[1];
            T* const p  = w[2];
            T* const v  = w[3];
            T* const h  = w[4];
            T* const s  = w[5];
            T* const t  = w[6];

            // Residual
            e.apply(A, x, r0, false);
            e.axpby(1, b, -1, r0);

            T rho = e.dot(r0, r0);

            const T r_norm0 = sqrt(rho);
            if (converged(r_norm0, r_norm0)) {
                return SFEM_SUCCESS;
            }

            e.copy(r0, r);
            e.copy(r0, p);

            int info = SFEM_FAILURE;
            int it   = 0;
            for (; it < max_it; it++) {
                // y = M^{-1} p. t is unused until A z, so y aliases t.
                T* const y = t;
                e.apply(M, p, y, false);

                e.apply(A, y, v, false);

                const T ptv = e.dot(r0, v);
                if (ptv == 0 || !std::isfinite(ptv)) {
                    info = SFEM_FAILURE;
                    break;
                }

                const T alpha = rho / ptv;
                if (!std::isfinite(alpha)) {
                    info = SFEM_FAILURE;
                    break;
                }

                e.zaxpby(1, x, alpha, y, h);
                e.zaxpby(1, r, -alpha, v, s);

                const T sts    = e.dot(s, s);
                const T s_norm = sqrt(sts);

                if (!std::isfinite(s_norm)) {
                    info = SFEM_FAILURE;
                    break;
                }

                if (converged(s_norm, r_norm0)) {
                    say(e, it, s_norm, r_norm0);
                    e.copy(h, x);
                    info = SFEM_SUCCESS;
                    break;
                }

                // z = M^{-1} s. r is dead after s is formed; do not alias z to x
                // (zeros(x) drops the iterate if omega is later rejected).
                T* const z = r;
                e.apply(M, s, z, false);

                e.apply(A, z, t, false);

                const T tts = e.dot(t, s);
                const T ttt = e.dot(t, t);

                if (ttt == 0 || !std::isfinite(tts) || !std::isfinite(ttt)) {
                    e.copy(h, x);
                    info = SFEM_FAILURE;
                    break;
                }

                const T omega = tts / ttt;
                if (!std::isfinite(omega)) {
                    e.copy(h, x);
                    info = SFEM_FAILURE;
                    break;
                }

                e.zaxpby(1, h, omega, z, x);
                e.zaxpby(1, s, -omega, t, r);

                const T rtr    = e.dot(r, r);
                const T r_norm = sqrt(rtr);
                if (!std::isfinite(r_norm)) {
                    e.copy(h, x);
                    info = SFEM_FAILURE;
                    break;
                }

                if (diverging(r_norm, r_norm0)) {
                    e.copy(h, x);
                    if (e.leader()) diverged_ = true;
                    info = SFEM_FAILURE;
                    break;
                }

                say(e, it, r_norm, r_norm0);
                if (converged(r_norm, r_norm0)) {
                    info = SFEM_SUCCESS;
                    break;
                }

                const T rho_new = e.dot(r0, r);
                if (rho == T(0) || omega == T(0) || !std::isfinite(rho_new)) {
                    info = SFEM_FAILURE;
                    break;
                }
                const T beta = (rho_new / rho) * (alpha / omega);
                if (!std::isfinite(beta)) {
                    info = SFEM_FAILURE;
                    break;
                }
                rho = rho_new;

                e.axpby(1, r, beta, p);
                e.axpby(-omega * beta, v, 1, p);
            }
            if (e.leader()) iterations_ = it;
            return info;
        }
    };

    template <typename T>
    std::shared_ptr<BiCGStab<T>> h_bcgs() {
        auto cg = std::make_shared<BiCGStab<T>>();
        cg->default_init();
        return cg;
    }
    template <typename T>
    std::shared_ptr<BiCGStab<T>> create_bcgs(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es) {
        return BiCGStab<T>::create(op, es);
    }

}  // namespace sfem

#endif  // SFEM_BCGS_HPP
