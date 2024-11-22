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
#include "sfem_ShiftedPenalty_impl.hpp"

namespace sfem {
    // From Active set expansion strategies in ShiftedPenalty algorithm, Kruzik et al. 2020
    template <typename T>
    class ShiftedPenalty final : public MatrixFreeLinearSolver<T> {
    public:
        T rtol{1e-8};
        T atol{1e-14};
        int max_it{10};
        int max_inner_it{10};
        int check_each{1};
        ptrdiff_t n_dofs{-1};
        bool verbose{true};
        bool debug{false};
        T penalty_param_{10};
        T max_penalty_param_{1000};
        bool use_gradient_descent{false};
        int iterations_{0};
        int iterations() const override { return iterations_; }

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        std::shared_ptr<Buffer<T>> upper_bound_;
        std::shared_ptr<Buffer<T>> lower_bound_;
        std::shared_ptr<Operator<T>> constraint_scaling_op_;
        std::shared_ptr<Operator<T>> apply_op;

        BLAS_Tpl<T> blas;
        ShiftedPenalty_Tpl<T> impl;

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
            OpenMP_ShiftedPenalty<T>::build_shifted_penalty(impl);
            execution_space_ = EXECUTION_SPACE_HOST;
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

            int count_inner_iter = 0;
            int count_linear_solver_iter = 0;

            T omega = 1 / penalty_param_;
            bool converged = false;
            for (iterations_ = 0; iterations_ < max_it; iterations_++) {
                // Inner loop
                for (int inner_iter = 0; inner_iter < max_inner_it;
                     inner_iter++, count_inner_iter++) {
                    blas.zeros(n_dofs, r_pen->data());

                    // Compute material residual
                    apply_op->apply(x, r_pen->data());
                    blas.axpby(n_dofs, 1, b, -1, r_pen->data());

                    // Compute penalty residual
                    impl.calc_r_pen(n_dofs,
                               x,
                               penalty_param_,
                               lb,
                               ub,
                               lagr_lb->data(),
                               lagr_ub->data(),
                               r_pen->data());

                    if (this->constraint_scaling_op_) {
                        // TODO: Implement this
                        assert(false);
                    }

                    const T r_pen_norm = blas.norm2(n_dofs, r_pen->data());

                    if (r_pen_norm < std::max(atol, omega) && inner_iter != 0) {
                        converged = true;
                        break;
                    }

                    if (use_gradient_descent) {
                        blas.axpby(n_dofs, 1e-1, r_pen->data(), 0, c->data());
                    } else {
                        blas.zeros(n_dofs, J_pen->data());

                        impl.calc_J_pen(n_dofs,
                                   x,
                                   penalty_param_,
                                   lb,
                                   ub,
                                   lagr_lb->data(),
                                   lagr_ub->data(),
                                   J_pen->data());

                        if (this->constraint_scaling_op_) {
                            this->constraint_scaling_op_->apply(J_pen->data(), J_pen->data());
                        }

                        auto J = apply_op + sfem::diag_op(n_dofs, J_pen, execution_space());
                        linear_solver_->set_op(J);

                        blas.zeros(n_dofs, c->data());
                        linear_solver_->apply(r_pen->data(), c->data());

                        count_linear_solver_iter += linear_solver_->iterations();
                    }

                    blas.axpy(n_dofs, 1, c->data(), x);
                }

                const T e_pen = ((ub) ? impl.sq_norm_ramp_p(n_dofs, x, ub) : T(0)) +
                                ((lb) ? impl.sq_norm_ramp_m(n_dofs, x, lb) : T(0));

                const T norm_pen = std::sqrt(e_pen);
                const T norm_rpen = blas.norm2(n_dofs, r_pen->data());

                if (norm_pen < penetration_tol) {
                    if (ub) impl.update_lagr_p(n_dofs, penalty_param_, x, ub, lagr_ub->data());
                    if (lb) impl.update_lagr_m(n_dofs, penalty_param_, x, lb, lagr_lb->data());

                    penetration_tol = penetration_tol / pow(penalty_param_, 0.9);
                    omega = omega / penalty_param_;
                } else {
                    penalty_param_ = std::min(penalty_param_ * 10, max_penalty_param_);
                    penetration_tol = 1 / pow(penalty_param_, 0.1);
                    omega = 1 / penalty_param_;
                }

                if (debug && ub) {
                    printf("lagr_ub: %e\n", blas.norm2(n_dofs, lagr_ub->data()));
                }

                if (debug && lb) {
                    printf("lagr_lb: %e\n", blas.norm2(n_dofs, lagr_lb->data()));
                }

                monitor(iterations_,
                        count_inner_iter,
                        count_linear_solver_iter,
                        norm_pen,
                        norm_rpen,
                        penetration_tol,
                        penalty_param_);

                if (norm_pen < atol && norm_rpen < atol) {
                    converged = true;
                    break;
                }
            }

            if (!converged && debug) {
                printf("Not Converged!\n");
            }

            return converged ? SFEM_SUCCESS : SFEM_FAILURE;
        }

        void monitor(const int iter, const int count_inner_iter, const int count_linear_solver_iter,
                     const T norm_pen, const T norm_rpen, const T penetration_tol,
                     const T penalty_param) {
            if (iter == max_it || iter % check_each == 0 || (norm_pen < atol && norm_rpen < atol)) {
                printf("%d|%d|%d) norm_pen %e, norm_rpen %e, penetration_tol %e, penalty_param "
                       "%e\n",
                       iter,
                       count_inner_iter,
                       count_linear_solver_iter,
                       norm_pen,
                       norm_rpen,
                       penetration_tol,
                       penalty_param);
            }
        }
    };
}  // namespace sfem

#endif  // SFEM_ShiftedPenalty_HPP
