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
        bool debug{false};

        int iterations_{0};
        int iterations() const override { return iterations_; }

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        SharedBuffer<T> upper_bound_;
        SharedBuffer<T> lower_bound_;

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

        void project(T* const x) {
            const T* const lb = lower_bound_ ? lower_bound_->data() : nullptr;
            const T* const ub = upper_bound_ ? upper_bound_->data() : nullptr;
            impl.project(n_dofs, lb, ub, x);

            if (debug) {
                T norm_project = norm2(x);
                printf("norm_project: %g\n", (double)norm_project);
            }
        }

        T norm_projected_gradient(const T* const x, const T* const g) const {
            const T* const lb = lower_bound_ ? lower_bound_->data() : nullptr;
            const T* const ub = upper_bound_ ? upper_bound_->data() : nullptr;
            auto ret = impl.norm_projected_gradient(n_dofs, lb, ub, x, g, eps);
            if (comm_) ret = std::sqrt(comm_->sum(ret * ret));

            if (debug) {
                printf("norm_projected_gradient: %g\n", (double)ret);
            }

            return ret;
        }

        void norm_gradients(const T* const x, const T* const g, T* const norm_free_gradient,
                            T* const norm_chopped_gradient) const {
            const T* const lb = lower_bound_ ? lower_bound_->data() : nullptr;
            const T* const ub = upper_bound_ ? upper_bound_->data() : nullptr;
            impl.norm_gradients(
                    n_dofs, lb, ub, x, g, norm_free_gradient, norm_chopped_gradient, eps);
            if (comm_) {
                T squares[2] = {*norm_free_gradient * *norm_free_gradient, *norm_chopped_gradient * *norm_chopped_gradient};
                comm_->sum(squares, 2, smesh::TypeToEnum<T>::value());
                *norm_free_gradient    = std::sqrt(squares[0]);
                *norm_chopped_gradient = std::sqrt(squares[1]);
            }

            if (debug) {
                printf("norm_free_gradient: %g, norm_chopped_gradient: %g\n",
                       (double)*norm_free_gradient,
                       (double)*norm_chopped_gradient);
            }
        }

        void chopped_gradient(const T* const x, const T* const g, T* gc) const {
            const T* const lb = lower_bound_ ? lower_bound_->data() : nullptr;
            const T* const ub = upper_bound_ ? upper_bound_->data() : nullptr;
            impl.chopped_gradient(n_dofs, lb, ub, x, g, gc, eps);
        }

        void free_gradient(const T* const x, const T* const g, T* gf) const {
            const T* const lb = lower_bound_ ? lower_bound_->data() : nullptr;
            const T* const ub = upper_bound_ ? upper_bound_->data() : nullptr;
            impl.free_gradient(n_dofs, lb, ub, x, g, gf);

            if (debug) {
                T norm_fg = norm2(gf);
                printf("norm_fg: %g\n", (double)norm_fg);
            }
        }

        T max_alpha(const T* const x, const T* const p) const {
            const T* const lb = lower_bound_ ? lower_bound_->data() : nullptr;
            const T* const ub = upper_bound_ ? upper_bound_->data() : nullptr;
            auto ret = impl.max_alpha(n_dofs, lb, ub, x, p, infty);
            if (comm_) ret = -comm_->max(-ret);

            if (debug) {
                printf("max_alpha: %g\n", (double)ret);
            }

            return ret;
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

        void cg_step(const T alpha_cg, const T dot_pAp, const T* const Ap, T* const p, T* const g,
                     T* const gf, T* const x) {
            this->blas->axpby(n_dofs, -alpha_cg, p, 1, x);

            this->blas->axpby(n_dofs, -alpha_cg, Ap, 1, g);

            this->free_gradient(x, g, gf);

            const T beta = dot(Ap, gf) / dot_pAp;

            this->blas->axpby(n_dofs, 1, gf, -beta, p);
        }

        void expansion_step(const T alpha_feas, const T alpha_bar, const T alpha_cg,
                            const T dot_pAp, const T* const b, T* const p, const T* const Ap,
                            T* const g, T* const gf, T* const x) {
            this->blas->axpby(n_dofs, -alpha_feas, p, 1, x);

            this->blas->axpby(n_dofs, -alpha_feas, Ap, 1, g);

            // A flat direction (alpha_cg infty) has no CG step past the bound to project.
            switch (alpha_cg < infty ? expansion_type_ : EXPANSION_TYPE_ORGINAL) {
                case EXPANSION_TYPE_PROJECTED_CG: {
                    this->blas->axpby(n_dofs, -alpha_cg, p, 1, x);

                    if (alpha_cg <= alpha_feas) {
                        this->blas->axpby(n_dofs, -alpha_cg, Ap, 1, g);
                        this->free_gradient(x, g, gf);
                        T beta = dot(Ap, gf) / dot_pAp;
                        this->blas->axpby(n_dofs, 1, gf, -beta, p);
                    } else {
                        this->project(x);
                        this->gradient(x, b, g);
                        this->free_gradient(x, g, gf);
                        this->blas->copy(n_dofs, gf, p);
                    }
                    break;
                }
                default: {
                    // EXPANSION_TYPE_ORGINAL
                    this->free_gradient(x, g, gf);
                    this->blas->axpby(n_dofs, -alpha_bar, gf, 1, x);
                    this->project(x);
                    this->gradient(x, b, g);
                    this->free_gradient(x, g, gf);
                    this->blas->copy(n_dofs, gf, p);
                    break;
                }
            }
        }

        /// Whether direction d, with dAd = d . A d, sees no curvature to rounding: on a singular or
        /// indefinite operator (a body free to move) the energy does not rise along it, and only a
        /// bound ends the step.
        bool no_curvature(const T dAd, const T* const d) const {
            const T dd = dot(d, d);
            return dd > 0 && dAd <= 64 * std::numeric_limits<T>::epsilon() * max_eig_ * dd;
        }

        /// Whether no bound stops a step of alpha_feas along d: none limits it (max_alpha's infty),
        /// or only the bounds that stand for none (infty) do.
        bool unbounded(const T alpha_feas, const T* const d) const {
            return alpha_feas >= infty || alpha_feas * norm2(d) >= infty / 2;
        }

        /// The step along a direction without curvature: to the first bound, but no longer than the
        /// expansion step's alpha_bar. The energy falls along all of it; a bound far away (a free
        /// body's rigid motion barely touching one) would otherwise send the iterate as far.
        T flat_step(const T alpha_feas, const T alpha_bar) const { return std::min(alpha_feas, alpha_bar); }

        /// False where the chopped gradient sees no curvature (no_curvature) and no bound stops it.
        bool proportioning_step(T* const p, T* const Agc, T* const g, T* const gf_or_gc,
                                T* const x, const T alpha_bar) {
            // Attention!
            T* gc = gf_or_gc;
            T* gf = gf_or_gc;

            this->chopped_gradient(x, g, gc);

            this->blas->zeros(n_dofs, Agc);
            apply_op(gc, Agc);
            const T dot_gcAgc = dot(gc, Agc);
            T       alpha_cg;
            if (no_curvature(dot_gcAgc, gc)) {
                // The energy falls along -gc until a bound stops it.
                const T alpha_feas = this->max_alpha(x, gc);
                if (unbounded(alpha_feas, gc)) return false;
                alpha_cg = flat_step(alpha_feas, alpha_bar);
            } else {
                alpha_cg = dot(g, gc) / dot_gcAgc;
            }

            this->blas->axpby(n_dofs, -alpha_cg, gc, 1, x);

            this->blas->axpby(n_dofs, -alpha_cg, Agc, 1, g);
            this->free_gradient(x, g, gf);

            this->blas->copy(n_dofs, gf, p);
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

            T norm_gf = -1;
            T norm_gc = -1;
            T alpha_feas = -1;  // maximal feasbile step
            T alpha_cg = -1;

            T* g = this->blas->allocate(allocation_size());  // Gradient
            T* p = this->blas->allocate(allocation_size());
            T* Ap_or_Ag = this->blas->allocate(allocation_size());
            T* gf_or_gc = this->blas->allocate(allocation_size());  // Free Gradient or Chopped Gradient

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

            T norm_g = norm2(g);
            this->monitor(0, norm_g, 1);
            T norm_gp0 = norm_g;

            this->free_gradient(x, g, gf_or_gc);
            this->blas->copy(n_dofs, gf_or_gc, p);

            int count_cg_steps = 0;
            int count_expansion_steps = 0;
            int count_proportioning_steps = 0;

            iterations_ = 0;

            while (!converged) {
                norm_gradients(x, g, &norm_gf, &norm_gc);

                if (norm_gc <= this->gamma * norm_gf) {
                    alpha_feas = this->max_alpha(x, p);
                    this->blas->zeros(n_dofs, Ap_or_Ag);
                    this->apply_op(p, Ap_or_Ag);
                    const T dot_pAp = dot(p, Ap_or_Ag);

                    // Without positive curvature the energy falls along -p until a bound stops
                    // it: the expansion step, at the bound. A free body's rigid motions are such
                    // directions, held only by the bounds.
                    const bool flat = no_curvature(dot_pAp, p);
                    if (flat && unbounded(alpha_feas, p)) {
                        fprintf(stderr, "[Warning][MPRGP] No positive curvature and no bound along the search direction (%g)\n", (double)dot_pAp);
                        break;
                    }

                    alpha_cg = flat ? infty : dot(g, p) / dot_pAp;

                    if (alpha_cg <= alpha_feas) {
                        this->cg_step(alpha_cg, dot_pAp, Ap_or_Ag, p, g, gf_or_gc, x);

                        count_cg_steps++;
                    } else {
                        // apply_op inside!
                        this->expansion_step(flat ? flat_step(alpha_feas, alpha_bar) : alpha_feas,
                                             alpha_bar,
                                             alpha_cg,
                                             dot_pAp,
                                             b,
                                             p,
                                             Ap_or_Ag,
                                             g,
                                             gf_or_gc,
                                             x);

                        count_expansion_steps++;
                    }
                } else {
                    // apply_op inside!
                    if (!this->proportioning_step(p, Ap_or_Ag, g, gf_or_gc, x, alpha_bar)) {
                        fprintf(stderr, "[Warning][MPRGP] No positive curvature and no bound along the chopped gradient\n");
                        break;
                    }

                    count_proportioning_steps++;
                }

                // Check for convergence
                const T norm_gp = this->norm_projected_gradient(x, g);
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
            return converged ? SFEM_SUCCESS : SFEM_FAILURE;
        }
    };
}  // namespace sfem

#endif  // SFEM_MPRGP_HPP
