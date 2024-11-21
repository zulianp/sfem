#ifndef SFEM_ShiftedPenalty_HPP
#define SFEM_ShiftedPenalty_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_openmp_blas.hpp"

namespace sfem {
    // From Active set expansion strategies in ShiftedPenalty algorithm, Kruzik et al. 2020
    template <typename T>
    class ShiftedPenalty final : public MatrixFreeLinearSolver<T> {
    public:
        enum ExpansionType { EXPANSION_TYPE_ORGINAL = 0, EXPANSION_TYPE_PROJECTED_CG = 1 };

        ExpansionType expansion_type_{EXPANSION_TYPE_ORGINAL};
        T rtol{1e-10};
        T atol{1e-16};
        T gamma{1};  // gamma > 0
        T eps{1e-14};
        T infty{1e15};
        T eigen_solver_tol{1e-1};
        int max_it{10};
        int max_inner_it{10000};
        int check_each{10};
        ptrdiff_t n_dofs{-1};
        bool verbose{true};
        bool debug{false};
        T penalty_param_{1.1};
        T max_penalty_param_{1000};

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        std::shared_ptr<Buffer<T>> upper_bound_;
        std::shared_ptr<Buffer<T>> lower_bound_;
        std::shared_ptr<Operator<T>> constraint_scaling_matrix_;
        std::shared_ptr<Operator<T>> apply_op;

        BLAS_Tpl<T> blas;
        std::shared_ptr<MatrixFreeLinearSolver<T>> linear_solver_;

        ExecutionSpace execution_space() const override { return execution_space_; }
        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->apply_op = op;
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

        bool good() const {
            assert(apply_op);
            assert(lower_bound_ || upper_bound_);
            return blas.good() && apply_op && (lower_bound_ || upper_bound_);
        }

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void monitor(const int iter, const T residual, const T rel_residual) {
            if (iter == max_it || iter % check_each == 0 || residual < atol ||
                rel_residual < rtol) {
                std::cout << iter << ": " << residual << " " << rel_residual << "\n";
            }
        }

        std::shared_ptr<Buffer<T>> make_buffer(const ptrdiff_t n) const {
            return Buffer<T>::own(
                    n, blas.allocate(n), blas.destroy, (enum MemorySpace)execution_space());
        }

        int apply(const T* const b, T* const x) override {
            assert(good());
            if (!good()) {
                fprintf(stderr, "[Error] ShiftedPenalty needs to be properly initialized!\n");
                return SFEM_FAILURE;
            }

            assert(penalty_param > 1);

            T* lb = (lower_bound_) ? lower_bound_->data() : nullptr;
            T* ub = (upper_bound_) ? upper_bound_->data() : nullptr;

            auto c = make_buffer(n_dofs);

            // Penalty term: -gradient/residual and Hessian/Jacobian
            auto r_pen = make_buffer(n_dofs);
            auto J_pen = make_buffer(n_dofs);

            // FIXME: This is not needed if lower_bound_ is not set...
            auto lagr_lb = make_buffer(n_dofs);
            auto lagr_ub = make_buffer(n_dofs);

            T penetration_norm = 0;
            T penetration_tol = 1 / (penalty_param_ * 0.1);

            auto sq_norm_ramp_p = [this](const ptrdiff_t n, const T* const x, T* const ub) -> T {
                T ret = 0;
#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n; i++) {
                    const T diff = std::max(T(0), x[i] - ub[i]);
                    ret += diff * diff;
                }
                return ret;
            };

            auto sq_norm_ramp_m = [this](const ptrdiff_t n, const T* const x, T* const lb) -> T {
                T ret = 0;
#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n; i++) {
                    const T diff = std::min(T(0), x[i] - lb[i]);
                    ret += diff * diff;
                }
                return ret;
            };

            // Adds to negative gradient (i.e., residual)
            auto ramp_p = [](const ptrdiff_t n,
                             const T penalty_param,
                             const T* const x,
                             const T* const ub,
                                 const T* const lagr_ub,
                                 T* const out) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    out[i] -= penalty_param *
                              std::max(T(0), x[i] - ub[i] + lagr_ub[i] / penalty_param);
                }
            };

            auto ramp_m = [](const ptrdiff_t n,
                             const T penalty_param,
                             const T* const x,
                             const T* const lb,
                                 const T* const lagr_lb,
                                 T* const out) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    out[i] -= penalty_param *
                              std::min(T(0), x[i] - lb[i] + lagr_lb[i] / penalty_param);
                }
            };

            auto update_lagr_p = [](const ptrdiff_t n,
                                    const T penalty_param,
                                    const T* const x,
                                    const T* const ub,
                                        T* const lagr_ub) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    lagr_ub[i] = std::max(T(0), lagr_ub[i] + penalty_param * (x[i] - ub[i]));
                }
            };

            auto update_lagr_m = [](const ptrdiff_t n,
                                    const T penalty_param,
                                    const T* const x,
                                    const T* const lb,
                                        T* const lagr_lb) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    lagr_lb[i] = std::min(T(0), lagr_lb[i] + penalty_param * (x[i] - lb[i]));
                }
            };

            auto calc_r_pen = [ramp_p, ramp_m](const ptrdiff_t n,
                                               T* const x,
                                               const T penalty_param,
                                               const T* const lb,
                                               const T* const ub,
                                               const T* const lagr_lb,
                                               const T* const lagr_ub,
                                               T* result) {
                // Ramp negative and positive parts
                if (lb) ramp_m(n, penalty_param, x, lb, lagr_lb, result);
                if (ub) ramp_p(n, penalty_param, x, ub, lagr_ub, result);
            };

            auto calc_J_pen = [](const ptrdiff_t n,
                                 const T* const x,
                                 const T penalty_param,
                                 const T* const lb,
                                 const T* const ub,
                                 const T* const lagr_lb,
                                 const T* const lagr_ub,
                                 T* const result) {
                if (lb) {
#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n; i++) {
                        result[i] +=
                                ((x[i] - lb[i] + lagr_lb[i] / penalty_param) <= 0) * penalty_param;
                    }
                }

                if (ub) {
#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n; i++) {
                        result[i] +=
                                ((x[i] - ub[i] + lagr_ub[i] / penalty_param) >= 0) * penalty_param;
                    }
                }
            };

            T omega = 1 / penalty_param_;
            bool converged = false;
            for (int iter = 0; iter < max_it; iter++) {
                // Inner loop
                for (int inner_iter = 0; inner_iter < max_inner_it; inner_iter++) {
                    blas.zeros(n_dofs, r_pen->data());

                    // Compute residual
                    apply_op->apply(x, r_pen->data());
                    blas.axpby(n_dofs, 1, b, -1, r_pen->data());

                    // Compute penalty residual
                    calc_r_pen(n_dofs,
                               x,
                               penalty_param_,
                               lb,
                               ub,
                               lagr_lb->data(),
                               lagr_ub->data(),
                               r_pen->data());

                    const T r_pen_norm = blas.norm2(n_dofs, r_pen->data());
                    // printf("r_pen_norm: %e\n", r_pen_norm);
                    if (r_pen_norm < std::max(atol, omega) && inner_iter != 0) {
                        converged = true;
                        break;
                    }

                    if (true) {
                        blas.axpby(n_dofs, 1e-1, r_pen->data(), 0, c->data());
                    } else {
                        blas.zeros(n_dofs, J_pen->data());

                        calc_J_pen(n_dofs,
                                   x,
                                   penalty_param_,
                                   lb,
                                   lb,
                                   lagr_lb->data(),
                                   lagr_ub->data(),
                                   J_pen->data());

                        if (this->constraint_scaling_matrix_) {
                            this->constraint_scaling_matrix_->apply(J_pen->data(), J_pen->data());
                        }

                        auto J = apply_op + sfem::diag_op(n_dofs, J_pen, execution_space_);
                        linear_solver_->set_op(J);

                        blas.zeros(n_dofs, c->data());
                        linear_solver_->apply(r_pen->data(), c->data());
                    }

                    blas.axpy(n_dofs, 1, c->data(), x);
                }

                const T e_pen = ((ub) ? sq_norm_ramp_p(n_dofs, x, ub) : T(0)) +
                                ((lb) ? sq_norm_ramp_m(n_dofs, x, lb) : T(0));

                const T norm_pen = std::sqrt(e_pen);
                const T norm_rpen = blas.norm2(n_dofs, r_pen->data());

                if (norm_pen < penetration_tol) {
                    if (ub) update_lagr_p(n_dofs, penalty_param_, x, ub, lagr_ub->data());
                    if (lb) update_lagr_m(n_dofs, penalty_param_, x, lb, lagr_lb->data());

                    penetration_tol = penetration_tol / pow(penalty_param_, 0.9);
                    omega = omega / penalty_param_;
                } else {
                    penalty_param_ = std::min(penalty_param_ * 10, max_penalty_param_);
                    penetration_tol = 1 / pow(penalty_param_, 0.1);
                    omega = 1 / penalty_param_;
                }

                if (ub) {
                    printf("lagr_ub: %e\n", blas.norm2(n_dofs, lagr_ub->data()));
                }

                printf("norm_pen %e, norm_rpen %e, penetration_tol %e, penalty_param %e\n",
                       norm_pen,
                       norm_rpen,
                       penetration_tol,
                       penalty_param_);

                if (norm_pen < atol && norm_rpen < atol) {
                    converged = true;
                    break;
                }
            }

            if (!converged) {
                printf("Not Converged!\n");
            }

            return converged ? SFEM_SUCCESS : SFEM_FAILURE;
        }
    };
}  // namespace sfem

#endif  // SFEM_ShiftedPenalty_HPP
