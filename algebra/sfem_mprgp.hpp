#ifndef SFEM_MPRGP_HPP
#define SFEM_MPRGP_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>

#include "sfem_MatrixFreeLinearSolver.hpp"

#include "sfem_PowerMethod.hpp"

// TODO GPU version
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
        T eigen_solver_tol{1e-5};
        int max_it{10000};
        int check_each{10};
        ptrdiff_t n_dofs{-1};
        bool verbose{true};
        T max_eig_{0};

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        std::shared_ptr<Buffer<T>> upper_bound_;
        std::shared_ptr<Buffer<T>> lower_bound_;

        std::function<void(const T* const, T* const)> apply_op;

        // Mem management
        std::function<T*(const std::size_t)> allocate;
        std::function<void(const std::size_t, T* const x)> zeros;
        std::function<void(const std::size_t, const T value, T* const x)> values;
        std::function<void(void*)> destroy;

        std::function<void(const ptrdiff_t, const T* const, T* const)> copy;

        // blas
        std::function<T(const ptrdiff_t, const T* const, const T* const)> dot;
        std::function<void(const ptrdiff_t, const T, const T* const, const T, T* const)> axpby;
        std::function<void(const std::ptrdiff_t, const T, T* const)> scal;
        std::function<T(const ptrdiff_t, const T* const)> norm2;

        std::shared_ptr<PowerMethod<T>> power_method;

        ExecutionSpace execution_space() const override { return execution_space_; }
        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }

        void set_expansion_type(const enum ExpansionType et) { expansion_type_ = et; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->apply_op = [=](const T* const x, T* const y) { op->apply(x, y); };
            n_dofs = op->rows();
        }
        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override {
            // Ignoring op!
        }

        void set_atol(const T val) { atol = val; }

        void set_rtol(const T val) { rtol = val; }

        void set_verbose(const bool val) { verbose = val; }

        void set_max_it(const int it) override { max_it = it; }
        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }

        void set_upper_bound(const std::shared_ptr<Buffer<T>>& ub) { upper_bound_ = ub; }
        void set_lower_bound(const std::shared_ptr<Buffer<T>>& lb) { lower_bound_ = lb; }

        void ensure_power_method() {
            if (!power_method) {
                power_method = std::make_shared<PowerMethod<T>>();
                power_method->norm2 = norm2;
                power_method->scal = scal;
                power_method->zeros = zeros;
            }
        }

        void project(T* const x) {
            if (lower_bound_ && upper_bound_) {
                const T* const lb = lower_bound_->data();
                const T* const ub = upper_bound_->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    x[i] = std::max(std::min(x[i], ub[i]), lb[i]);
                }
            } else if (upper_bound_) {
                const T* const ub = upper_bound_->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    x[i] = std::min(x[i], ub[i]);
                }
            } else if (lower_bound_) {
                const T* const lb = lower_bound_->data();
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    x[i] = std::max(x[i], lb[i]);
                }
            }
        }

#define MPRGP_UTOPIA_STYLE_FG

        inline T gf_lb_ub(const T lbi, const T ubi, const T xi, const T gi) const {
#ifdef MPRGP_UTOPIA_STYLE_FG
            return (xi <= lbi || xi >= ubi) ? T(0) : gi;
#else
            return ((std::abs(lbi - xi) < eps) || (std::abs(ubi - xi) < eps)) ? T(0) : gi;
#endif
        }

        inline T gf_lb(const T lbi, const T xi, const T gi) const {
#ifdef MPRGP_UTOPIA_STYLE_FG
            return (xi <= lbi) ? T(0) : gi;
#else
            return (std::abs(lbi - xi) < eps) ? T(0) : gi;
#endif
        }

        inline T gf_ub(const T ubi, const T xi, const T gi) const {
#ifdef MPRGP_UTOPIA_STYLE_FG
            return (xi >= ubi) ? T(0) : gi;
#else
            return (std::abs(ubi - xi) < eps) ? T(0) : gi;
#endif
        }

        inline T gc_lb_ub(const T lbi, const T ubi, const T xi, const T gi) const {
            return ((std::abs(lbi - xi) < eps)
                            ? std::min(T(0), gi)
                            : ((std::abs(ubi - xi) < eps) ? std::max(T(0), gi) : T(0)));
        }

        inline T gc_lb(const T lbi, const T xi, const T gi) const {
            return ((std::abs(lbi - xi) < eps) ? std::min(T(0), gi) : T(0));
        }

        inline T gc_ub(const T ubi, const T xi, const T gi) const {
            return ((std::abs(ubi - xi) < eps) ? std::max(T(0), gi) : T(0));
        }

        T norm_projected_gradient(const T* const x, const T* const g) const {
            T ret = 0;
            if (lower_bound_ && upper_bound_) {
                const T* const lb = lower_bound_->data();
                const T* const ub = upper_bound_->data();

#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T d =
                            gf_lb_ub(lb[i], ub[i], x[i], g[i]) + gc_lb_ub(lb[i], ub[i], x[i], g[i]);
                    ret += d * d;
                }
            } else if (upper_bound_) {
                const T* const ub = upper_bound_->data();

#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T d = gf_ub(ub[i], x[i], g[i]) + gc_ub(ub[i], x[i], g[i]);
                    ret += d * d;
                }
            } else if (lower_bound_) {
                const T* const lb = lower_bound_->data();

#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T d = gf_lb(lb[i], x[i], g[i]) + gc_lb(lb[i], x[i], g[i]);
                    ret += d * d;
                }
            } else {
                assert(false);
            }

            return sqrt(ret);
        }

        void norm_gradients(const T* const x,
                            const T* const g,
                            T* const norm_free_gradient,
                            T* const norm_chopped_gradient) const {
            T acc_gf = 0;
            T acc_gc = 0;

            if (lower_bound_ && upper_bound_) {
                const T* const lb = lower_bound_->data();
                const T* const ub = upper_bound_->data();

#pragma omp parallel for reduction(+ : acc_gf), reduction(+ : acc_gc)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T val_gf = gf_lb_ub(lb[i], ub[i], x[i], g[i]);
                    const T val_gc = gc_lb_ub(lb[i], ub[i], x[i], g[i]);

                    acc_gf += val_gf * val_gf;
                    acc_gc += val_gc * val_gc;
                }
            } else if (upper_bound_) {
                const T* const ub = upper_bound_->data();

#pragma omp parallel for reduction(+ : acc_gf), reduction(+ : acc_gc)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T val_gf = gf_ub(ub[i], x[i], g[i]);
                    const T val_gc = gc_ub(ub[i], x[i], g[i]);

                    acc_gf += val_gf * val_gf;
                    acc_gc += val_gc * val_gc;
                }
            } else if (lower_bound_) {
                const T* const lb = lower_bound_->data();

#pragma omp parallel for reduction(+ : acc_gf), reduction(+ : acc_gc)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T val_gf = gf_lb(lb[i], x[i], g[i]);
                    const T val_gc = gc_lb(lb[i], x[i], g[i]);

                    acc_gf += val_gf * val_gf;
                    acc_gc += val_gc * val_gc;
                }
            } else {
                assert(false);
            }

            *norm_free_gradient = sqrt(acc_gf);
            *norm_chopped_gradient = sqrt(acc_gc);
        }

        void chopped_gradient(const T* const x, const T* const g, T* gc) const {
            if (lower_bound_ && upper_bound_) {
                const T* const lb = lower_bound_->data();
                const T* const ub = upper_bound_->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gc[i] = gc_lb_ub(lb[i], ub[i], x[i], g[i]);
                }
            } else if (upper_bound_) {
                const T* const ub = upper_bound_->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gc[i] = gc_ub(ub[i], x[i], g[i]);
                }
            } else if (lower_bound_) {
                const T* const lb = lower_bound_->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gc[i] = gc_lb(lb[i], x[i], g[i]);
                }
            }
        }

        void free_gradient(const T* const x, const T* const g, T* gf) const {
            if (lower_bound_ && upper_bound_) {
                const T* const lb = lower_bound_->data();
                const T* const ub = upper_bound_->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gf[i] = gf_lb_ub(lb[i], ub[i], x[i], g[i]);
                }
            } else if (upper_bound_) {
                const T* const ub = upper_bound_->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gf[i] = gf_ub(ub[i], x[i], g[i]);
                }
            } else if (lower_bound_) {
                const T* const lb = lower_bound_->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gf[i] = gf_lb(lb[i], x[i], g[i]);
                }
            }
        }

        T max_alpha(const T* const x, const T* const p) const {
            T ret = infty;

            if (lower_bound_ && upper_bound_) {
                const T* const lb = lower_bound_->data();
                const T* const ub = upper_bound_->data();

#pragma omp parallel for reduction(min : ret)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T alpha_lb = (p[i] > 0) ? ((x[i] - lb[i]) / p[i]) : infty;
                    const T alpha_ub = (p[i] < 0) ? ((x[i] - ub[i]) / p[i]) : infty;
                    const T alpha = std::min(alpha_lb, alpha_ub);
                    ret = std::min(alpha, ret);
                }

            } else if (upper_bound_) {
                const T* const ub = upper_bound_->data();

#pragma omp parallel for reduction(min : ret)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T alpha = (p[i] < 0) ? ((x[i] - ub[i]) / p[i]) : infty;
                    ret = std::min(alpha, ret);
                }
            } else if (lower_bound_) {
                const T* const lb = lower_bound_->data();

#pragma omp parallel for reduction(min : ret)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T alpha = (p[i] > 0) ? ((x[i] - lb[i]) / p[i]) : infty;
                    ret = std::min(alpha, ret);
                }
            }

            return ret;
        }

        bool good() const {
            assert(allocate);
            assert(destroy);
            assert(copy);
            assert(zeros);
            assert(values);
            assert(dot);
            assert(norm2);
            assert(axpby);
            assert(scal);
            assert(apply_op);
            assert(lower_bound_ || upper_bound_);

            return allocate && destroy && copy && zeros && values && dot && norm2 && axpby &&
                   scal && apply_op && (lower_bound_ || upper_bound_);
        }

        void default_init() {
            allocate = [](const std::ptrdiff_t n) -> T* { return (T*)calloc(n, sizeof(T)); };

            destroy = [](void* a) { free(a); };

            copy = [](const ptrdiff_t n, const T* const src, T* const dest) {
                std::memcpy(dest, src, n * sizeof(T));
            };

            dot = [](const ptrdiff_t n, const T* const l, const T* const r) -> T {
                T ret = 0;

#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n; i++) {
                    ret += l[i] * r[i];
                }

                return ret;
            };

            norm2 = [](const ptrdiff_t n, const T* const x) -> T {
                T ret = 0;

#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n; i++) {
                    ret += x[i] * x[i];
                }

                return sqrt(ret);
            };

            axpby = [](const ptrdiff_t n,
                       const T alpha,
                       const T* const x,
                       const T beta,
                       T* const y) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    y[i] = alpha * x[i] + beta * y[i];
                }
            };

            zeros = [](const std::ptrdiff_t n, T* const x) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    x[i] = 0;
                }
            };

            values = [](const std::ptrdiff_t n, const T v, T* const x) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    x[i] = v;
                }
            };

            scal = [](const std::ptrdiff_t n, const T alpha, T* const x) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    x[i] *= alpha;
                }
            };

            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void cg_step(const T alpha_cg,
                     const T dot_pAp,
                     const T* const Ap,
                     T* const p,
                     T* const g,
                     T* const gf,
                     T* const x) {
            // x_new = x_old - alpha_CG * p_old
            this->axpby(n_dofs, -alpha_cg, p, 1, x);

            // g_new = g_old - alpha_CG * Ap_old
            this->axpby(n_dofs, -alpha_cg, Ap, 1, g);

            // gf_new
            this->free_gradient(x, g, gf);

            // beta = dot(Ap, gf_new)/dot(Ap_old, p_old)
            const T beta = this->dot(n_dofs, Ap, gf) / dot_pAp;

            // p_new = g_new - beta * p_old
            this->axpby(n_dofs, 1, gf, -beta, p);
        }

        void expansion_step(const T alpha_feas,
                            const T alpha_bar,
                            const T alpha_cg,
                            const T dot_pAp,
                            const T* const b,
                            T* const p,
                            const T* const Ap,
                            T* const g,
                            T* const gf,
                            T* const x) {
            // x_half = x_old - alpha_feas * p
            this->axpby(n_dofs, -alpha_feas, p, 1, x);

            // g_half = g_old - alpha_feas * Ap
            this->axpby(n_dofs, -alpha_feas, Ap, 1, g);

            switch (expansion_type_) {
                case EXPANSION_TYPE_PROJECTED_CG: {
                    this->axpby(n_dofs, -alpha_cg, p, 1, x);

                    if(alpha_cg <= alpha_feas) {
                        this->axpby(n_dofs, -alpha_cg, Ap, 1, g);
                        this->free_gradient(x, g, gf);
                        T beta = this->dot(n_dofs, Ap, gf) / dot_pAp;
                        this->axpby(n_dofs, 1, gf, -beta, p);
                    } else {
                        this->project(x);
                        this->gradient(x, b, g);
                        this->free_gradient(x, g, gf);
                        this->copy(n_dofs, gf, p);
                    }
                    break;
                }
                default: {  // EXPANSION_TYPE_ORGINAL
                    this->free_gradient(x, g, gf);
                    this->axpby(n_dofs, -alpha_bar, gf, 1, x);
                    this->project(x);
                    this->gradient(x, b, g);
                    this->free_gradient(x, g, gf);
                    this->copy(n_dofs, gf, p);
                    break;
                }
            }
        }

        void proportioning_step(T* const p,
                                T* const Agc,
                                T* const g,
                                T* const gf_or_gc,
                                T* const x) {
            // Attention!
            T* gc = gf_or_gc;
            T* gf = gf_or_gc;

            this->chopped_gradient(x, g, gc);

            // alpha_CG = dot(g, gc)/dot(gc, A* gc)
            this->zeros(n_dofs, Agc);
            apply_op(gc, Agc);
            T alpha_cg = this->dot(n_dofs, g, gc) / this->dot(n_dofs, gc, Agc);

            // x_new = x_old - alpha_CG * gc
            this->axpby(n_dofs, -alpha_cg, gc, 1, x);

            // g_new = g_old - alpha_CG * A * gc
            this->axpby(n_dofs, -alpha_cg, Agc, 1, g);
            this->free_gradient(x, g, gf);

            // p_new = gf_new
            this->copy(n_dofs, gf, p);
        }

        void gradient(const T* const x, const T* const b, T* const g) {
            this->zeros(n_dofs, g);
            this->apply_op(x, g);
            this->axpby(n_dofs, -1, b, 1, g);
        }

        void monitor(const int iter, const T residual) {
            if (iter == max_it || iter % check_each == 0 || residual < atol) {
                std::cout << iter << ": " << residual << "\n";
            }
        }

        void set_max_eig(const T max_eig) { max_eig_ = max_eig; }

        int apply(const T* const b, T* const x) override {
            assert(good());
            if (!good()) {
                fprintf(stderr, "[Error] MPRGP needs to be properly initialized!\n");
                return SFEM_FAILURE;
            }

            int it = 0;
            bool converged = false;

            T norm_gp = -1;
            T norm_gf = -1;
            T norm_gc = -1;
            T alpha_feas = -1;  // maximal feasbile step
            T alpha_cg = -1;

            T* g = this->allocate(n_dofs);  // Gradient
            T* p = this->allocate(n_dofs);
            T* Ap_or_Ag = this->allocate(n_dofs);
            T* gf_or_gc = this->allocate(n_dofs);  // Free Gradient or Chopped Gradient

            T alpha_bar = 1;
            if (expansion_type_ == EXPANSION_TYPE_ORGINAL) {
                if (max_eig_ == 0) {
                    this->ensure_power_method();
                    this->values(n_dofs, 1, p);
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

            T norm_g = this->norm2(n_dofs, g);
            this->monitor(0, norm_g);

            this->free_gradient(x, g, gf_or_gc);
            this->copy(n_dofs, gf_or_gc, p);

            // printf("norm_fg = %g\n", this->norm2(n_dofs, gf_or_gc));

            int count_cg_steps = 0;
            int count_expansion_steps = 0;
            int count_proportioning_steps = 0;

            while (!converged) {
                norm_gradients(x, g, &norm_gf, &norm_gc);

                // printf("A) %g < %g (norm_gc < this->gamma * norm_gf)\n", norm_gc, norm_gf);

                if (norm_gc <= this->gamma * norm_gf) {
                    alpha_feas = this->max_alpha(x, p);
                    this->zeros(n_dofs, Ap_or_Ag);
                    this->apply_op(p, Ap_or_Ag);
                    const T dot_pAp = this->dot(n_dofs, p, Ap_or_Ag);

                    if(dot_pAp < 0) {
                        fprintf(stderr, "[Warning][MPRGP] Detected negative curvature\n");
                    }

                    alpha_cg = this->dot(n_dofs, g, p) / dot_pAp;

                    // printf("B) %g < %g\n", alpha_cg, alpha_feas);
                    if (alpha_cg <= alpha_feas) {
                        this->cg_step(alpha_cg, dot_pAp, Ap_or_Ag, p, g, gf_or_gc, x);

                        count_cg_steps++;
                    } else {
                        // apply_op inside!
                        this->expansion_step(
                                alpha_feas, alpha_bar, alpha_cg, dot_pAp, b, p, Ap_or_Ag, g, gf_or_gc, x);

                        count_expansion_steps++;
                    }
                } else {
                    // apply_op inside!
                    this->proportioning_step(p, Ap_or_Ag, g, gf_or_gc, x);

                    count_proportioning_steps++;
                }

                // Check for convergence
                const T norm_gp = this->norm_projected_gradient(x, g);
                converged = norm_gp < atol;

                monitor(it, norm_gp);
                if (++it >= max_it) {
                    break;
                }
            }

            if (verbose) {
                printf("#cg_steps\t\t%d\n", count_cg_steps);
                printf("#expansion_steps\t%d\n", count_expansion_steps);
                printf("#proportioning_steps\t%d\n", count_proportioning_steps);
            }

            this->destroy(g);
            this->destroy(p);
            this->destroy(Ap_or_Ag);
            this->destroy(gf_or_gc);
            return converged ? SFEM_SUCCESS : SFEM_FAILURE;
        }
    };
}  // namespace sfem

#endif  // SFEM_MPRGP_HPP
