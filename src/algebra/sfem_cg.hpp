#ifndef SFEM_CG_HPP
#define SFEM_CG_HPP

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_operator_factory.hpp"

#include "sfem_BLAS_openmp.hpp"
#include "sfem_ConjugateGradientKernels_openmp.hpp"
#include "sfem_SolverExec.hpp"

// https://en.wikipedia.org/wiki/Conjugate_gradient_method
namespace sfem {

    template <typename T>
    class ConjugateGradient final : public MatrixFreeLinearSolver<T> {
    public:
        /// Built for the execution space es, on op.
        static std::shared_ptr<ConjugateGradient<T>> create(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es);
        // Operator
        std::shared_ptr<Operator<T>> apply_op;
        std::shared_ptr<Operator<T>> preconditioner_op;

        std::shared_ptr<BLAS<T>> blas;
        ConjugateGradientKernels<T>                impl;

        std::function<void(T*)> interceptor;

        // Solver parameters
        T              rtol{1e-10};
        T              atol{1e-16};
        int            max_it{10000};
        int            check_each{1};
        ptrdiff_t      n_dofs{SFEM_PTRDIFF_INVALID};
        int            iterations_{0};
        bool           verbose{true};
        // If true, apply_op / preconditioner overwrite their output.
        // Default false: most ops accumulate (y += Ax).
        bool           apply_overwrites_output{false};
        // Use fused CG vector kernels from impl (OpenMP / CUDA). Set false for separate BLAS baseline.
        bool           fusion{true};
        // Inside Newton on an energy: stop at the first direction of non-positive curvature,
        // returning the iterate so far (a descent direction, its directions all of positive
        // curvature), or on the first direction that direction itself scaled by |curvature|
        // (Nocedal and Wright, Algorithm 7.1). Off, the iteration runs on and loses its meaning.
        bool           stop_on_negative_curvature{false};

        int iterations() const override { return iterations_; }

        /// Whether the last apply stopped on a direction of non-positive curvature.
        bool negative_curvature() const { return negative_curvature_; }

        void set_stop_on_negative_curvature(const bool val) { stop_on_negative_curvature = val; }

        ExecutionContext execution_context() const override { return blas ? blas->context() : ExecutionContext(EXECUTION_SPACE_INVALID); }

        void set_atol(const T val) { atol = val; }

        void set_rtol(const T val) { rtol = val; }

        void set_verbose(const bool val) { verbose = val; }

        void set_apply_overwrites_output(const bool val) { apply_overwrites_output = val; }

        void set_fusion(const bool val) { fusion = val; }

        /// \deprecated Use set_fusion
        void set_host_fusion(const bool val) { set_fusion(val); }

        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            apply_op = op;
            n_dofs   = op->rows();
        }

        int set_op_and_diag_shift(const std::shared_ptr<Operator<T>>& op, const SharedBuffer<T>& diag) override {
            assert(this->execution_space() == (enum ExecutionSpace)diag->mem_space());
            auto J         = op;  // + sfem::diag_op(diag, execution_space());
            this->apply_op = J;
            n_dofs         = op->rows();

            if (preconditioner_op) {
                auto shiftable = std::dynamic_pointer_cast<ShiftableOperator<T>>(preconditioner_op);
                if (shiftable) {
                    return shiftable->shift(diag);
                } else {
                    SFEM_ERROR(
                            "Tried to call shift on object that is not subclass of "
                            "ShiftableOperator!\n");
                    return SFEM_FAILURE;
                }
            }

            return SFEM_SUCCESS;
        }

        int set_op_and_diag_shift(const std::shared_ptr<Operator<T>>&          op,
                                  const std::shared_ptr<SparseBlockVector<T>>& sbv,
                                  const SharedBuffer<T>&                       diag) override {
            assert(this->execution_space() == (enum ExecutionSpace)diag->mem_space());
            this->apply_op = op;  // + sfem::create_sparse_block_vector_mult(op->rows(), sbv, diag);

            if (preconditioner_op) {
                auto shiftable = std::dynamic_pointer_cast<ShiftableOperator<T>>(preconditioner_op);

                if (shiftable) {
                    return shiftable->shift(sbv, diag);
                } else {
                    SFEM_ERROR(
                            "Tried to call shift on object that is not subclass of "
                            "ShiftableOperator!\n");
                    assert(false);
                    return SFEM_FAILURE;
                }
            }

            return SFEM_SUCCESS;
        }

        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override { preconditioner_op = op; }

        void set_max_it(const int it) override { max_it = it; }

        void set_preconditioner(std::function<void(const T* const, T* const)>&& in) { preconditioner_op = in; }

        void default_init() {
            blas             = OpenMPBLAS<T>::create();
            OpenMPConjugateGradientKernels<T>::build(impl);
        }

        bool good() const { return blas && blas->good() && apply_op && (!fusion || impl.good()); }

        void monitor(const int iter, const T residual, const T relative_residual, const T alpha) {
            if (!verbose) return;

            if (iter == max_it || iter == 0 || iter % check_each == 0 || relative_residual < rtol) {
                std::cout << iter << ": residual abs: " << residual << ", rel: " << relative_residual << " (rtol = " << rtol
                          << ", atol = " << atol << ", alpha = " << alpha << ")\n";
            }
        }

        /// One iteration, written once for its executor (sfem_SolverExec.hpp): on the host, when
        /// the operator (and the preconditioner) have a team form, the whole run in one region;
        /// otherwise vector by vector through the BLAS.
        int apply(const ptrdiff_t n, const T* const b, T* const x) {
            SFEM_TRACE_SCOPE("ConjugateGradient::apply");
            negative_curvature_ = false;
            if (!good()) {
                assert(0);
                return SFEM_FAILURE;
            }
            ensure_workspace(n, preconditioner_op != nullptr);

            auto team_op   = std::dynamic_pointer_cast<TeamOperator<T>>(apply_op);
            auto team_prec = std::dynamic_pointer_cast<TeamOperator<T>>(preconditioner_op);
            if (team_op && team_op->team_capable() && (!preconditioner_op || (team_prec && team_prec->team_capable())) &&
                this->execution_space() == EXECUTION_SPACE_HOST) {
                if (team_op->prepare() != SFEM_SUCCESS) return SFEM_FAILURE;
                if (team_prec && team_prec->prepare() != SFEM_SUCCESS) return SFEM_FAILURE;
                T* const buffer = team_reduction_buffer(team_sums_, n);
                int      info   = SFEM_FAILURE;
#pragma omp parallel
                {
                    TeamReduction<T, 1> sums(buffer, team_chunk_count(n));
                    const Team          team = team_op->team();
                    TeamExec<T>         e(team, sums);
                    const int r = preconditioner_op ? run_precond(e, *team_op, *team_prec, n, b, x) : run_basic(e, *team_op, n, b, x);
                    if (team.leader()) info = r;
                }
                return info;
            }

            BlasExec<T> e(*blas, n);
            return preconditioner_op ? run_precond(e, *apply_op, *preconditioner_op, n, b, x) : run_basic(e, *apply_op, n, b, x);
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

        ~ConjugateGradient() { destroy_workspace(); }

    private:
        bool      negative_curvature_{false};
        T*        work_r_{nullptr};
        T*        work_z_{nullptr};
        T*        work_p_{nullptr};
        T*        work_Ap_{nullptr};
        ptrdiff_t work_n_{-1};
        // The sums of a run in one region (TeamReduction).
        std::vector<T> team_sums_;

        bool use_fusion() const { return fusion && impl.good(); }

        void destroy_workspace() {
            if (!blas) {
                work_r_ = work_z_ = work_p_ = work_Ap_ = nullptr;
                work_n_                                = -1;
                return;
            }
            if (work_r_) {
                blas->destroy(work_r_);
                work_r_ = nullptr;
            }
            if (work_z_) {
                blas->destroy(work_z_);
                work_z_ = nullptr;
            }
            if (work_p_) {
                blas->destroy(work_p_);
                work_p_ = nullptr;
            }
            if (work_Ap_) {
                blas->destroy(work_Ap_);
                work_Ap_ = nullptr;
            }
            work_n_ = -1;
        }

        void ensure_workspace(const ptrdiff_t n, const bool need_z) {
            if (work_n_ == n && work_r_ && work_p_ && work_Ap_ && (!need_z || work_z_)) {
                return;
            }

            destroy_workspace();
            work_r_  = blas->allocate(n);
            work_p_  = blas->allocate(n);
            work_Ap_ = blas->allocate(n);
            if (need_z) {
                work_z_ = blas->allocate(n);
            }
            work_n_ = n;
        }

        // x += alpha p; r -= alpha Ap; optionally return (r,r)
        T update_x_r(const ptrdiff_t n,
                     const T         alpha,
                     const T* const  p,
                     const T* const  Ap,
                     T* const        x,
                     T* const        r,
                     const bool      with_rtr) const {
            if (use_fusion()) {
                if (with_rtr) {
                    return impl.update_x_r_and_rtr(n, alpha, p, Ap, x, r);
                }
                impl.update_x_r(n, alpha, p, Ap, x, r);
                return T(0);
            }

            blas->axpby(n, alpha, p, T(1), x);
            blas->axpby(n, -alpha, Ap, T(1), r);
            return with_rtr ? blas->dot(n, r, r) : T(0);
        }

        void update_p(const ptrdiff_t n, const T beta, const T* const z, T* const p) const {
            if (use_fusion()) {
                impl.update_p(n, beta, z, p);
            } else {
                blas->axpby(n, T(1), z, beta, p);
            }
        }

        // x += alpha p; r -= alpha Ap, and with_rtr (r, r): the BLAS's fused kernels or, in a
        // region, one pass over the thread's entries.
        T x_r(BlasExec<T>&, const ptrdiff_t n, const T alpha, const T* const p, const T* const Ap, T* const x, T* const r, const bool with_rtr) {
            return update_x_r(n, alpha, p, Ap, x, r, with_rtr);
        }

        T x_r(TeamExec<T>& e, const ptrdiff_t, const T alpha, const T* const p, const T* const Ap, T* const x, T* const r, const bool with_rtr) {
            if (!with_rtr) {
                const Team& team = e.team();
                for (ptrdiff_t i = team.begin; i < team.end; i++) {
                    x[i] = alpha * p[i] + x[i];
                    r[i] = -alpha * Ap[i] + r[i];
                }
                return T(0);
            }
            return e.sweep_sum([=](const ptrdiff_t i) {
                x[i] = alpha * p[i] + x[i];
                r[i] = -alpha * Ap[i] + r[i];
                return r[i] * r[i];
            });
        }

        // p <- z + beta p.
        void z_p(BlasExec<T>&, const ptrdiff_t n, const T beta, const T* const z, T* const p) { update_p(n, beta, z, p); }
        void z_p(TeamExec<T>& e, const ptrdiff_t, const T beta, const T* const z, T* const p) { e.axpby(T(1), z, beta, p); }

        // The leader's: what the iteration says and keeps.
        template <class E>
        void say(E& e, const int iter, const T residual, const T relative, const T alpha) {
            if (e.leader()) monitor(iter, residual, relative, alpha);
        }

        template <class E>
        void intercept(E& e, T* const x) {
            if (!interceptor) return;
            e.barrier();
            if (e.leader()) interceptor(x);
            e.barrier();
        }

        template <class E, class Op>
        int run_basic(E& e, Op& A, const ptrdiff_t n, const T* const b, T* const x) {
            T* r  = work_r_;
            T* p  = work_p_;
            T* Ap = work_Ap_;

            e.apply(A, x, r, apply_overwrites_output);
            e.axpby(T(1), b, T(-1), r);

            const T rtr0    = e.dot(r, r);
            const T r_norm0 = sqrt(rtr0);
            say(e, 0, r_norm0, 1, 0);

            T rtr = rtr0;
            assert(rtr0 == rtr0);

            if (rtr0 == 0) {
                return SFEM_SUCCESS;
            }

            e.copy(r, p);

            int info = SFEM_FAILURE;
            int it   = 0;
            for (; it < max_it; it++) {
                e.apply(A, p, Ap, apply_overwrites_output);

                const T ptAp = e.dot(p, Ap);
                if (stop_on_negative_curvature && !(ptAp > 0)) {
                    if (e.leader()) negative_curvature_ = true;
                    if (it == 0) e.axpby(rtr / std::abs(ptAp), p, T(1), x);
                    info = SFEM_SUCCESS;
                    break;
                }
                const T alpha = rtr / ptAp;

                assert(ptAp == ptAp);
                assert(alpha == alpha);
                assert(rtr != 0);
                assert(rtr == rtr);

                // Fuse x/r updates + (r,r) into one backend pass
                const T rtr_new = x_r(e, n, alpha, p, Ap, x, r, true);
                const T beta    = rtr_new / rtr;
                rtr             = rtr_new;
                z_p(e, n, beta, r, p);

                T r_norm = sqrt(rtr_new);
                assert(r_norm == r_norm);

                say(e, it + 1, r_norm, r_norm / r_norm0, alpha);
                if (r_norm < atol || rtr_new == 0 || r_norm / r_norm0 < rtol) {
                    info = SFEM_SUCCESS;
                    break;
                }

                intercept(e, x);
            }
            if (e.leader()) iterations_ = it;
            return info;
        }

        template <class E, class Op, class Prec>
        int run_precond(E& e, Op& A, Prec& M, const ptrdiff_t n, const T* const b, T* const x) {
            T* r  = work_r_;
            T* z  = work_z_;
            T* p  = work_p_;
            T* Ap = work_Ap_;

            e.apply(A, x, r, apply_overwrites_output);
            e.axpby(T(1), b, T(-1), r);

            const T rtr0    = e.dot(r, r);
            const T r_norm0 = sqrt(rtr0);
            say(e, 0, r_norm0, 1, 0);

            if (rtr0 == 0) {
                return SFEM_SUCCESS;
            }

            e.apply(M, r, z, apply_overwrites_output);
            e.copy(z, p);

            e.apply(A, p, Ap, apply_overwrites_output);

            const T rtz0 = e.dot(r, z);
            T       rtz  = rtz0;

            if (rtz == 0) {
                return SFEM_SUCCESS;
            }

            {
                const T ptAp = e.dot(p, Ap);
                if (stop_on_negative_curvature && !(ptAp > 0)) {
                    if (e.leader()) {
                        negative_curvature_ = true;
                        iterations_         = 0;
                    }
                    e.axpby(rtz / std::abs(ptAp), p, T(1), x);
                    return SFEM_SUCCESS;
                }
                assert(ptAp != 0);
                const T alpha = rtz / ptAp;
                x_r(e, n, alpha, p, Ap, x, r, false);
            }

            int info = SFEM_FAILURE;
            int it   = 0;
            for (; it < max_it; it++) {
                e.apply(M, r, z, apply_overwrites_output);

                const T rtz_new = e.dot(r, z);

                assert(rtz != 0);
                const T beta = rtz_new / rtz;
                rtz          = rtz_new;

                z_p(e, n, beta, z, p);

                e.apply(A, p, Ap, apply_overwrites_output);

                const T ptAp = e.dot(p, Ap);
                if (stop_on_negative_curvature && !(ptAp > 0)) {
                    if (e.leader()) negative_curvature_ = true;
                    info = SFEM_SUCCESS;
                    break;
                }
                const T alpha = rtz / ptAp;

                x_r(e, n, alpha, p, Ap, x, r, false);

                auto anorm = sqrt(rtz);
                auto rnorm = anorm / sqrt(rtz0);

                say(e, it + 1, anorm, rnorm, alpha);
                if (anorm < atol || rnorm < rtol) {
                    info = SFEM_SUCCESS;
                    break;
                }

                intercept(e, x);
            }
            if (e.leader()) iterations_ = it;
            return info;
        }
    };

    template <typename T>
    std::shared_ptr<ConjugateGradient<T>> h_cg() {
        auto cg = std::make_shared<ConjugateGradient<T>>();
        cg->default_init();
        return cg;
    }
    template <typename T>
    std::shared_ptr<ConjugateGradient<T>> create_cg(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es) {
        return ConjugateGradient<T>::create(op, es);
    }

}  // namespace sfem

#endif  // SFEM_CG_HPP
