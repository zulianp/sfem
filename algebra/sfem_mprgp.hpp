#ifndef SFEM_MPRGP_HPP
#define SFEM_MPRGP_HPP

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <algorithm>

#include "sfem_MatrixFreeLinearSolver.hpp"

namespace sfem {
    // From Active set expansion strategies in MPRGP algorithm, Kruzik et al. 2020
    template <typename T>
    class MPRGP final : public MatrixFreeLinearSolver<T> {
    public:
        T rtol{1e-10};
        T atol{1e-16};
        T gamma{1};  // gamma > 0
        T eps{1e-14};
        T infty{1e15};
        int max_it{10000};
        int check_each{100};
        ptrdiff_t n_dofs{-1};
        bool verbose{true};
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        std::shared_ptr<Buffer<T>> upper_bound_;
        std::shared_ptr<Buffer<T>> lower_bound_;

        std::function<void(const T* const, T* const)> apply_op;
        std::function<void(const T* const, T* const)> preconditioner_op;

        // Mem management
        std::function<T*(const std::size_t)> allocate;
        std::function<void(const std::size_t, T* const x)> zeros;
        std::function<void(void*)> destroy;

        std::function<void(const ptrdiff_t, const T* const, T* const)> copy;

        // blas
        std::function<T(const ptrdiff_t, const T* const, const T* const)> dot;
        std::function<void(const ptrdiff_t, const T, const T* const, const T, T* const)> axpby;

        ExecutionSpace execution_space() const override { return execution_space_; }
        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }
        void set_op(const std::shared_ptr<Operator<T>>& op) override {}
        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override {}
        void set_max_it(const int it) override { max_it = it; }
        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }

        void set_upper_bound(const std::shared_ptr<Buffer<T>>& ub) { upper_bound_ = ub; }

        void set_lower_bound(const std::shared_ptr<Buffer<T>>& lb) { lower_bound_ = lb; }

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

        T norm_projected_gradient(const T* const x,
                            const T* const g) const {
            assert(false);
            return -1;
        }

        void norm_gradients(const T* const x,
                            const T* const g,
                            T* const norm_free_gradient,
                            T* const norm_chopped_gradient) const {
            // In the paper the free gradient is defined by checking equality
            // between x and ub/lb, how is that numerically relevant?
            // we do >/< instead

            T acc_gf = 0;
            T acc_gc = 0;

            if (lower_bound_ && upper_bound_) {
                const T* const lb = lower_bound_->data();
                const T* const ub = upper_bound_->data();

#pragma omp parallel for reduction(+ : acc_gf), reduction(+ : acc_gc)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T val_gf = (x[i] < lb[i] || x[i] > ub[i]) ? T(0) : g[i];
                    const T val_gc =
                            (std::abs(lb[i] - x[i]) < eps)
                                    ? std::min(T(0), g[i])
                                    : ((std::abs(ub[i] - x[i]) < eps) ? std::max(T(0), g[i])
                                                                      : T(0));
                    acc_gf += val_gf * val_gf;
                    acc_gc += val_gc * val_gc;
                }
            } else if (upper_bound_) {
                const T* const ub = upper_bound_->data();

#pragma omp parallel for reduction(+ : acc_gf), reduction(+ : acc_gc)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T val_gf = (x[i] > ub[i]) ? T(0) : g[i];
                    const T val_gc = (std::abs(ub[i] - x[i]) < eps) ? std::max(T(0), g[i]) : T(0);

                    acc_gf += val_gf * val_gf;
                    acc_gc += val_gc * val_gc;
                }
            } else if (lower_bound_) {
                const T* const lb = lower_bound_->data();

#pragma omp parallel for reduction(+ : acc_gf), reduction(+ : acc_gc)
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    const T val_gf = (x[i] < lb[i]) ? T(0) : g[i];
                    const T val_gc = (std::abs(lb[i] - x[i]) < eps) ? std::min(T(0), g[i]) : T(0);

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
                    gc[i] = (std::abs(lb[i] - x[i]) < eps)
                                    ? std::min(T(0), g[i])
                                    : ((std::abs(ub[i] - x[i]) < eps) ? std::max(T(0), g[i])
                                                                      : T(0));
                }
            } else if (upper_bound_) {
                const T* const ub = upper_bound_->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gc[i] = (std::abs(ub[i] - x[i]) < eps) ? std::max(T(0), g[i]) : T(0);
                }
            } else if (lower_bound_) {
                const T* const lb = lower_bound_->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gc[i] = (std::abs(lb[i] - x[i]) < eps) ? std::min(T(0), g[i]) : T(0);
                }
            }
        }

        void free_gradient(const T* const x, const T* const g, T* gf) const {
            // In the paper the free gradient is defined by checking equality
            // between x and ub/lb, how is that numerically relevant?
            // we do >/< instead

            if (lower_bound_ && upper_bound_) {
                const T* const lb = lower_bound_->data();
                const T* const ub = upper_bound_->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gf[i] = (x[i] < lb[i] || x[i] > ub[i]) ? T(0) : g[i];
                }
            } else if (upper_bound_) {
                const T* const ub = upper_bound_->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gf[i] = (x[i] > ub[i]) ? T(0) : g[i];
                }
            } else if (lower_bound_) {
                const T* const lb = lower_bound_->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_dofs; i++) {
                    gf[i] = (x[i] < lb[i]) ? T(0) : g[i];
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

        void cg_step(const T alpha_cg,
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
            const T beta = this->dot(n_dofs, Ap, gf) / this->dot(n_dofs, Ap, p);

            // p_new = g_new - beta * p_old
            this->axpby(n_dofs, 1, g, -beta, p);
        }

        void expansion_step() {
            // x_half = x_old - alpha_feas * p
            // g_half = g_old - alpha_feas * Ap
            // x_new = P(x_half - alpha_bar * d_tilde) // ?
            // g_new = A * x_new  - b
            // p_new = gf  //?
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

        void residual(const T* const x, const T* const b, T* const g) {
            this->apply_op(x, g);
            this->axpby(n_dofs, 1, b, -1, g);
        }

        int apply(const T* const b, T* const x) override {
            int it = 0;
            bool converged = false;
            T norm_gp = -1;
            T norm_gf = -1;
            T norm_gc = -1;
            T alpha_feas = -1;  // maximal feasbile step
            T alpha_cg = -1;

            T* g = allocate(n_dofs);  // Gradient
            T* p = allocate(n_dofs);
            T* Ap_or_Ag = allocate(n_dofs);
            T* gf_or_gc = allocate(n_dofs);  // Free Gradient or Chopped Gradient

            this->project(x);  // Make iterate feasible
            this->residual(x, b, g);
            this->free_gradient(x, g, gf_or_gc);
            this->copy(n_dofs, gf_or_gc, p);



            while (!converged) {
                norm_gradients(x, g, &norm_gf, &norm_gc);

                if (norm_gc < this->gamma * norm_gf) {
                    alpha_feas = this->max_alpha(x, p);
                    this->apply_op(p, Ap_or_Ag);
                    alpha_cg = this->dot(n_dofs, g, p) / this->dot(n_dofs, p, Ap_or_Ag);

                    if (alpha_cg < alpha_feas) {
                        this->cg_step(alpha_cg, Ap_or_Ag, p, g, gf_or_gc, x);
                    } else {
                        this->expansion_step();
                    }
                } else {
                    this->proportioning_step(p, Ap_or_Ag, g, gf_or_gc, x);
                }

                // Check for convergence
                const T norm_gp = this->norm_projected_gradient(x, g);
                converged = norm_gp < atol;
            }

            destroy(g);
            destroy(p);
            destroy(Ap_or_Ag);
            destroy(gf_or_gc);
            return SFEM_SUCCESS;
        }
    };
}  // namespace sfem

#endif  // SFEM_MPRGP_HPP
