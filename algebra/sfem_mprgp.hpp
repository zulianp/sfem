#ifndef SFEM_MPRGP_HPP
#define SFEM_MPRGP_HPP

#include <limits>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_ParallelOperator.hpp"

#include "sfem_PowerMethod.hpp"

#include "sfem_openmp_blas.hpp"
#include "sfem_openmp_mprgp_impl.hpp"

namespace sfem {
    // From Active set expansion strategies in MPRGP algorithm, Kruzik et al. 2020
    template <typename T>
    class MPRGP final : public MatrixFreeLinearSolver<T> {
    public:
        enum ExpansionType { EXPANSION_TYPE_ORGINAL = 0, EXPANSION_TYPE_PROJECTED_CG = 1 };

        ExpansionType expansion_type_{EXPANSION_TYPE_ORGINAL};
        T rtol{1e-10};
        T atol{1e-16};
        T gamma{1};  // gamma > 0
        T eps{1e-14};
        T infty{1e15};
        T eigen_solver_tol{1e-1};
        int max_it{10000};
        int check_each{10};
        ptrdiff_t n_dofs{SFEM_PTRDIFF_INVALID};
        bool verbose{true};
        T max_eig_{0};

        int iterations_{0};
        int iterations() const override { return iterations_; }

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        SharedBuffer<T> upper_bound_;
        SharedBuffer<T> lower_bound_;
        // Convergence on the projected gradient weighted entry by entry, |w g_P| (and rtol relative
        // to |w g_0|): with variables z = S^{-1} y, w = S^{-1} measures the gradient in y. The
        // steps are the iteration's own; only the test changes. Null: unweighted.
        SharedBuffer<T> norm_weight_;

        std::function<void(const T* const, T* const)> apply_op;

        std::shared_ptr<BLAS<T>> blas;
        MPRGP_Tpl<T> impl;
        // Under MPI (a ParallelOperator of more than one rank): the vectors are the owned prefix
        // of space-sized buffers, n_dofs the owned length, n_alloc the buffers'; every reduction
        // spans the ranks, so every branch is taken alike on all of them.
        std::shared_ptr<Communicator> comm_;
        ptrdiff_t n_alloc_{-1};

        T sum(const T local) const { return comm_ ? comm_->sum(local) : local; }
        T dot(const T* const x, const T* const y) const { return sum(this->blas->dot(n_dofs, x, y)); }
        T norm2(const T* const x) const { return std::sqrt(dot(x, x)); }
        ptrdiff_t allocation_size() const { return n_alloc_ > 0 ? n_alloc_ : n_dofs; }
        std::shared_ptr<PowerMethod<T>> power_method;

        ExecutionSpace execution_space() const override { return execution_space_; }
        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }

        void set_expansion_type(const enum ExpansionType et) { expansion_type_ = et; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->apply_op = [=](const T* const x, T* const y) { op->apply(x, y); };
            n_dofs = op->rows();
            comm_.reset();
            n_alloc_ = -1;
            if (auto parallel = std::dynamic_pointer_cast<ParallelOperator<T>>(op)) {
                if (parallel->comm()->size() > 1) {
                    comm_    = parallel->comm();
                    n_alloc_ = std::max(parallel->row_allocation_size(), parallel->col_allocation_size());
                }
            }
        }
        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& /*op*/) override {
            // Ignoring op!
        }

        void set_atol(const T val) { atol = val; }
        void set_rtol(const T val) { rtol = val; }
        void set_verbose(const bool val) { verbose = val; }
        void set_max_it(const int it) override { max_it = it; }
        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }

        void set_upper_bound(const SharedBuffer<T>& ub) { upper_bound_ = ub; }
        void set_lower_bound(const SharedBuffer<T>& lb) { lower_bound_ = lb; }
        void set_norm_weight(const SharedBuffer<T>& w) { norm_weight_ = w; }

        void ensure_power_method() {
            if (!power_method) {
                auto blas_impl = blas;
                power_method = std::make_shared<PowerMethod<T>>();
                auto comm = comm_;
                power_method->norm2 = [blas_impl, comm](const std::ptrdiff_t n, const T* const x) {
                    const T local = blas_impl->dot(n, x, x);
                    return std::sqrt(comm ? comm->sum(local) : local);
                };
                power_method->scal = [blas_impl](const std::ptrdiff_t n, const T alpha, T* const x) { blas_impl->scal(n, alpha, x); };
                power_method->zeros = [blas_impl](const std::size_t n, T* const x) { blas_impl->zeros(n, x); };
            }
        }

        const T* lb() const { return lower_bound_ ? lower_bound_->data() : nullptr; }
        const T* ub() const { return upper_bound_ ? upper_bound_->data() : nullptr; }

        void project(T* const x) { impl.project(n_dofs, lb(), ub(), x); }

        /// Sums the ranks' partial reductions in place, one message for all of them.
        void sum(T* const values, const int count) const {
            if (comm_) comm_->sum(values, count, smesh::TypeToEnum<T>::value());
        }
        T min(const T local) const { return comm_ ? -comm_->max(-local) : local; }

        /// Along d, with Ad = A d.
        struct Curvature {
            T dAd, dd, gd, alpha_feas;
        };
        Curvature curvature(const T* const x, const T* const g, const T* const d, const T* const Ad, T* const work) const {
            T out[4];
            impl.curvature(n_dofs, lb(), ub(), x, g, d, Ad, infty, out, work);
            sum(out, 3);
            return {out[0], out[1], out[2], min(out[3])};
        }

        /// At the iterate a step reaches: Ad.gf for the next direction, and the norms that choose
        /// the next step and test convergence.
        struct Gradients {
            T Adgf, norm_gf, norm_gc, norm_gp;
        };
        /// x -= alpha d, g -= alpha Ad (neither without d), gf <- the free gradient.
        Gradients step(const T alpha, const T* const d, const T* const Ad, T* const x, T* const g, T* const gf, T* const work) const {
            T out[4];
            impl.step(n_dofs, lb(), ub(), alpha, d, Ad, x, g, gf, eps, norm_weight_ ? norm_weight_->data() : nullptr, out, work);
            sum(out, 4);
            return {out[0], std::sqrt(out[1]), std::sqrt(out[2]), std::sqrt(out[3])};
        }

        bool good() const {
            assert(apply_op);
            assert(lower_bound_ || upper_bound_);
            return blas->good() && impl.good() && apply_op && (lower_bound_ || upper_bound_);
        }

        void default_init() {
            blas = make_openmp_blas<T>();
            OpenMP_MPRGP<T>::build_mprgp(impl);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        /// To the bound along p, then a projected step along the free gradient there (or, projected
        /// CG, the CG step past the bound, projected); p restarts from the free gradient.
        Gradients expansion_step(const T alpha_feas, const T alpha_bar, const T alpha_cg, const T* const b, T* const p,
                                 const T* const Ap, T* const g, T* const x, T* const work) {
            // A flat direction (alpha_cg infty) has no CG step past the bound to project.
            if (alpha_cg < infty && expansion_type_ == EXPANSION_TYPE_PROJECTED_CG) {
                this->blas->axpby(n_dofs, -alpha_feas, p, 1, x);
                this->blas->axpby(n_dofs, -alpha_cg, p, 1, x);
                this->project(x);
            } else {
                impl.expand(n_dofs, lb(), ub(), alpha_feas, alpha_bar, p, Ap, x, g);
            }
            this->gradient(x, b, g);
            return this->step(0, nullptr, nullptr, x, g, p, work);
        }

        /// Whether direction d, with dAd = d . A d, sees no curvature to rounding: on a singular or
        /// indefinite operator (a body free to move) the energy does not rise along it, and only a
        /// bound ends the step.
        bool no_curvature(const T dAd, const T dd) const {
            return dd > 0 && dAd <= 64 * std::numeric_limits<T>::epsilon() * max_eig_ * dd;
        }

        /// Whether no bound stops a step of alpha_feas along d (d.d = dd): none limits it
        /// (infty), or only the bounds that stand for none (infty) do.
        bool unbounded(const T alpha_feas, const T dd) const {
            return alpha_feas >= infty || alpha_feas * std::sqrt(dd) >= infty / 2;
        }

        /// The step along a direction without curvature: to the first bound, but no longer than the
        /// expansion step's alpha_bar. The energy falls along all of it; a bound far away (a free
        /// body's rigid motion barely touching one) would otherwise send the iterate as far.
        T flat_step(const T alpha_feas, const T alpha_bar) const { return std::min(alpha_feas, alpha_bar); }

        /// A step along the chopped gradient gc; p restarts from the free gradient. False where gc
        /// sees no curvature (no_curvature) and no bound stops it.
        bool proportioning_step(T* const p, T* const Agc, T* const g, T* const gc, T* const x, const T alpha_bar,
                                T* const work, Gradients* const grads) {
            impl.chopped_gradient(n_dofs, lb(), ub(), x, g, gc, eps);

            this->blas->zeros(n_dofs, Agc);
            apply_op(gc, Agc);
            const Curvature c = curvature(x, g, gc, Agc, work);
            T               alpha_cg;
            if (no_curvature(c.dAd, c.dd)) {
                // The energy falls along -gc until a bound stops it.
                if (unbounded(c.alpha_feas, c.dd)) return false;
                alpha_cg = flat_step(c.alpha_feas, alpha_bar);
            } else {
                alpha_cg = c.gd / c.dAd;
            }

            *grads = this->step(alpha_cg, gc, Agc, x, g, p, work);
            return true;
        }

        void gradient(const T* const x, const T* const b, T* const g) {
            this->blas->zeros(n_dofs, g);
            this->apply_op(x, g);
            this->blas->axpby(n_dofs, -1, b, 1, g);
        }

        void monitor(const int iter, const T residual, const T rel_residual) {
            if (iter == max_it || iter % check_each == 0 || residual < atol ||
                rel_residual < rtol) {
                std::cout << iter << ": " << residual << " " << rel_residual << "\n";
            }
        }

        void set_max_eig(const T max_eig) { max_eig_ = max_eig; }

        int apply(const T* const b, T* const x) override {
            assert(good());
            if (!good()) {
                fprintf(stderr, "[Error] MPRGP needs to be properly initialized!\n");
                return SFEM_FAILURE;
            }

            bool converged = false;

            T* g = this->blas->allocate(allocation_size());  // Gradient
            T* p = this->blas->allocate(allocation_size());
            T* Ap_or_Ag = this->blas->allocate(allocation_size());
            T* gf_or_gc = this->blas->allocate(allocation_size());  // Free Gradient or Chopped Gradient
            T* work     = impl.work_size > 0 ? this->blas->allocate(impl.work_size) : nullptr;

            T alpha_bar = 1;
            if (expansion_type_ == EXPANSION_TYPE_ORGINAL) {
                if (max_eig_ == 0) {
                    this->ensure_power_method();
                    this->blas->values(n_dofs, 1, p);
                    max_eig_ = power_method->max_eigen_value(
                            apply_op, 10000, this->eigen_solver_tol, n_dofs, p, g);

                    printf("[MPRGP] max_eig: %g\n", max_eig_);
                }
                alpha_bar = 1.95 / max_eig_;
            } else {
                printf("Skipping power method!\n");
            }

            this->project(x);  // Make iterate feasible
            this->gradient(x, b, g);

            // |w g| (Ap_or_Ag, zero from the allocation, as scratch).
            if (norm_weight_) this->blas->xypaz(n_dofs, norm_weight_->data(), g, 0, Ap_or_Ag);
            T norm_g = norm2(norm_weight_ ? Ap_or_Ag : g);
            this->monitor(0, norm_g, 1);
            T norm_gp0 = norm_g;

            // The first direction: the free gradient at the feasible start.
            Gradients grads = this->step(0, nullptr, nullptr, x, g, p, work);

            int count_cg_steps = 0;
            int count_expansion_steps = 0;
            int count_proportioning_steps = 0;

            iterations_ = 0;

            while (!converged) {
                if (grads.norm_gc <= this->gamma * grads.norm_gf) {
                    this->blas->zeros(n_dofs, Ap_or_Ag);
                    this->apply_op(p, Ap_or_Ag);
                    const Curvature c = curvature(x, g, p, Ap_or_Ag, work);

                    // Without positive curvature the energy falls along -p until a bound stops
                    // it: the expansion step, at the bound. A free body's rigid motions are such
                    // directions, held only by the bounds.
                    const bool flat = no_curvature(c.dAd, c.dd);
                    if (flat && unbounded(c.alpha_feas, c.dd)) {
                        fprintf(stderr, "[Warning][MPRGP] No positive curvature and no bound along the search direction (%g)\n", (double)c.dAd);
                        break;
                    }

                    const T alpha_cg = flat ? infty : c.gd / c.dAd;

                    if (alpha_cg <= c.alpha_feas) {
                        grads        = this->step(alpha_cg, p, Ap_or_Ag, x, g, gf_or_gc, work);
                        const T beta = grads.Adgf / c.dAd;
                        this->blas->axpby(n_dofs, 1, gf_or_gc, -beta, p);

                        count_cg_steps++;
                    } else {
                        // apply_op inside!
                        grads = this->expansion_step(flat ? flat_step(c.alpha_feas, alpha_bar) : c.alpha_feas,
                                                     alpha_bar,
                                                     alpha_cg,
                                                     b,
                                                     p,
                                                     Ap_or_Ag,
                                                     g,
                                                     x,
                                                     work);

                        count_expansion_steps++;
                    }
                } else {
                    // apply_op inside!
                    if (!this->proportioning_step(p, Ap_or_Ag, g, gf_or_gc, x, alpha_bar, work, &grads)) {
                        fprintf(stderr, "[Warning][MPRGP] No positive curvature and no bound along the chopped gradient\n");
                        break;
                    }

                    count_proportioning_steps++;
                }

                // Check for convergence
                const T norm_gp = grads.norm_gp;
                const T rel_norm_gp = norm_gp / norm_gp0;
                converged = norm_gp < atol || rel_norm_gp < rtol;

                monitor(iterations_, norm_gp, rel_norm_gp);
                if (++iterations_ >= max_it) {
                    break;
                }
            }

            if (verbose) {
                printf("#cg_steps\t\t%d\n", count_cg_steps);
                printf("#expansion_steps\t%d\n", count_expansion_steps);
                printf("#proportioning_steps\t%d\n", count_proportioning_steps);
            }

            this->blas->destroy(g);
            this->blas->destroy(p);
            this->blas->destroy(Ap_or_Ag);
            this->blas->destroy(gf_or_gc);
            if (work) this->blas->destroy(work);
            return converged ? SFEM_SUCCESS : SFEM_FAILURE;
        }
    };
}  // namespace sfem

#endif  // SFEM_MPRGP_HPP
