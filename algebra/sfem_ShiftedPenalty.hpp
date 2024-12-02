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
#include "sfem_ShiftedPenalty_impl.hpp"
#include "sfem_openmp_blas.hpp"

namespace sfem {

    template <typename T>
    static std::shared_ptr<Operator<T>> diag_op(const std::shared_ptr<Buffer<T>>& diagonal_scaling,
                                                const ExecutionSpace es);

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
        bool debug{true};
        T penalty_param_{10};
        T max_penalty_param_{1000};
        T damping_{1};
        bool enable_steepest_descent_{false};
        int iterations_{0};
        int iterations() const override { return iterations_; }

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        std::shared_ptr<Buffer<T>> upper_bound_;
        std::shared_ptr<Buffer<T>> lower_bound_;
        std::shared_ptr<Operator<T>> constraints_op_;
        std::shared_ptr<Operator<T>> constraints_op_transpose_;
        std::shared_ptr<Operator<T>> apply_op;

        BLAS_Tpl<T> blas;
        ShiftedPenalty_Tpl<T> impl;

        std::shared_ptr<MatrixFreeLinearSolver<T>> linear_solver_;

        ExecutionSpace execution_space() const override { return execution_space_; }
        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }
        void set_linear_solver(const std::shared_ptr<MatrixFreeLinearSolver<T>> &solver)
        {
            linear_solver_ = solver;
        }

        void set_damping(const T damping)
        {
            damping_ = damping;
        }

        void enable_steepest_descent(const bool val) {
            enable_steepest_descent_ = val;
        }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->apply_op = op;
            n_dofs = op->rows();
        }
        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override {
            // Ignoring op!
        }

        void set_constraints_op(const std::shared_ptr<Operator<T>>& op,
                                const std::shared_ptr<Operator<T>>& op_t) {
            constraints_op_ = op;
            constraints_op_transpose_ = op_t;
        }

        void set_atol(const T val) { atol = val; }
        void set_rtol(const T val) { rtol = val; }
        void set_verbose(const bool val) { verbose = val; }
        void set_max_it(const int it) override { max_it = it; }
        void set_max_inner_it(const int it) { max_inner_it = it; }
        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }
        void set_penalty_param(const T penalty_param)
        {
            penalty_param_ = penalty_param;
        }

        void set_upper_bound(const std::shared_ptr<Buffer<T>>& ub) { upper_bound_ = ub; }
        void set_lower_bound(const std::shared_ptr<Buffer<T>>& lb) { lower_bound_ = lb; }

        bool good() const {
            assert(apply_op);
            assert(lower_bound_ || upper_bound_);
            return blas.good() && apply_op && (lower_bound_ || upper_bound_);
        }

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            OpenMP_ShiftedPenalty<T>::build(impl);
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

            assert(penalty_param_ > 1);
            assert(!constraints_op_ || (constraints_op_ && constraints_op_transpose_));

            ptrdiff_t n_constrained_dofs = n_dofs;

            if (constraints_op_) {
                n_constrained_dofs = constraints_op_->rows();
            }

            T* lb = (lower_bound_) ? lower_bound_->data() : nullptr;
            T* ub = (upper_bound_) ? upper_bound_->data() : nullptr;

            auto c = make_buffer(n_dofs);

            // Penalty term: -gradient/residual and Hessian/Jacobian
            auto r_pen = make_buffer(n_dofs);
            auto J_pen = make_buffer(n_dofs);

            std::shared_ptr<Buffer<T>> lagr_lb, lagr_ub;
            if (lb) lagr_lb = make_buffer(n_constrained_dofs);
            if (ub) lagr_ub = make_buffer(n_constrained_dofs);

            T penetration_norm = 0;
            T penetration_tol = 1 / (penalty_param_ * 0.1);

            int count_inner_iter = 0;
            int count_linear_solver_iter = 0;
            int count_lagr_mult_updates = 0;

            T omega = 1 / penalty_param_;
            bool converged = false;
            for (iterations_ = 0; iterations_ < max_it; iterations_++) {
                // Inner loop
                for (int inner_iter = 0; inner_iter < max_inner_it;
                     inner_iter++, count_inner_iter++) {
                    if (constraints_op_) {
                        // Use r_pen as temp buffer
                        blas.zeros(n_constrained_dofs, r_pen->data());
                        constraints_op_->apply(x, r_pen->data());

                        // Use c as temp buffer
                        blas.zeros(n_constrained_dofs, c->data());
                        impl.calc_r_pen(n_constrained_dofs,
                                        r_pen->data(),
                                        penalty_param_,
                                        lb,
                                        ub,
                                        lagr_lb ? lagr_lb->data() : nullptr,
                                        lagr_ub ? lagr_ub->data() : nullptr,
                                        c->data());



                        if(debug) {
                            const T norm_pen = blas.norm2(n_constrained_dofs, c->data());
                            printf("norm_pen (pre): %g\n", (double)norm_pen);

                            // printf("c\n");
                            // c->print(std::cout);
                        }

                        apply_op->apply(x, r_pen->data());
                        blas.axpby(n_dofs, 1, b, -1, r_pen->data());


                        blas.zeros(n_dofs, r_pen->data());
                        constraints_op_transpose_->apply(c->data(), r_pen->data());


                        if(debug) {
                            const T norm_pen = blas.norm2(n_dofs, r_pen->data());
                            printf("norm_pen: %g\n", (double)norm_pen);

                        //     // printf("c\n");
                        //     // c->print(std::cout);
                        }

                        
                    } else {
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
                                        lagr_lb ? lagr_lb->data() : nullptr,
                                        lagr_ub ? lagr_ub->data() : nullptr,
                                        r_pen->data());
                    }

                    const T r_pen_norm = blas.norm2(n_dofs, r_pen->data());

                    if (r_pen_norm < std::max(atol, omega) && inner_iter != 0) {
                        break;
                    }

                    if (enable_steepest_descent_) {
                        // alpha = <r, r>/<A r, r>
                        blas.zeros(n_dofs, c->data());
                        apply_op->apply(r_pen->data(), c->data());

                        const T alpha = blas.dot(n_dofs, r_pen->data(), r_pen->data()) /
                                        blas.dot(n_dofs, r_pen->data(), c->data());

                        blas.axpby(n_dofs, alpha, r_pen->data(), 0, c->data());
                    } else {
                        if (constraints_op_) {
                            // Use J_pen as temp buffer
                            blas.zeros(n_constrained_dofs, J_pen->data());

                            // TODO: check if it is worth storing this as we are computing it twice?
                            constraints_op_->apply(x, J_pen->data());

                            // Use c as temp buffer
                            blas.zeros(n_constrained_dofs, c->data());
                            impl.calc_J_pen(n_constrained_dofs,
                                            J_pen->data(),
                                            penalty_param_,
                                            lb,
                                            ub,
                                            lagr_lb ? lagr_lb->data() : nullptr,
                                            lagr_ub ? lagr_ub->data() : nullptr,
                                            c->data());


                            // printf("J_pen\n");
                            // c->print(std::cout);

                            blas.zeros(n_dofs, J_pen->data());
                            constraints_op_transpose_->apply(c->data(), J_pen->data());

                            // if(debug) {
                            //     const T norm_J_pen = blas.norm2(n_dofs, J_pen->data());
                            //     printf("norm_J_pen: %g\n", (double)norm_J_pen);
                            // }

                          
                        } else {
                            blas.zeros(n_dofs, J_pen->data());

                            impl.calc_J_pen(n_dofs,
                                            x,
                                            penalty_param_,
                                            lb,
                                            ub,
                                            lagr_lb ? lagr_lb->data() : nullptr,
                                            lagr_ub ? lagr_ub->data() : nullptr,
                                            J_pen->data());
                        }

                        auto J = apply_op + sfem::diag_op(J_pen, execution_space());
                        linear_solver_->set_op(J);

                        blas.zeros(n_dofs, c->data());
                        linear_solver_->apply(r_pen->data(), c->data());

                        count_linear_solver_iter += linear_solver_->iterations();
                    }

                    blas.axpy(n_dofs, damping_, c->data(), x);
                }

                auto Tx = x;

                if(constraints_op_) {
                    blas.zeros(n_constrained_dofs, c->data());
                    constraints_op_->apply(x, c->data());
                    Tx = c->data();
                }

                const T e_pen = ((ub) ? impl.sq_norm_ramp_p(n_constrained_dofs, Tx, ub) : T(0)) +
                                ((lb) ? impl.sq_norm_ramp_m(n_constrained_dofs, Tx, lb) : T(0));

                const T norm_pen = std::sqrt(e_pen);
                const T norm_rpen = blas.norm2(n_dofs, r_pen->data());

                if (norm_pen < penetration_tol) {
                    if (ub) impl.update_lagr_p(n_constrained_dofs, penalty_param_, Tx, ub, lagr_ub->data());
                    if (lb) impl.update_lagr_m(n_constrained_dofs, penalty_param_, Tx, lb, lagr_lb->data());

                    penetration_tol = penetration_tol / pow(penalty_param_, 0.9);
                    omega = omega / penalty_param_;

                    count_lagr_mult_updates++;

                    if (debug && ub) {
                        printf("lagr_ub: %e\n", blas.norm2(n_constrained_dofs, lagr_ub->data()));
                    }

                    if (debug && lb) {
                        printf("lagr_lb: %e\n", blas.norm2(n_constrained_dofs, lagr_lb->data()));
                    }

                } else {
                    penalty_param_ = std::min(penalty_param_ * 10, max_penalty_param_);
                    penetration_tol = 1 / pow(penalty_param_, 0.1);
                    omega = 1 / penalty_param_;
                }

         
                monitor(iterations_,
                        count_inner_iter,
                        count_linear_solver_iter,
                        count_lagr_mult_updates,
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
                     const int count_lagr_mult_updates, const T norm_pen, const T norm_rpen,
                     const T penetration_tol, const T penalty_param) {
            if (iter == max_it || iter % check_each == 0 || (norm_pen < atol && norm_rpen < atol)) {
                printf("%d|%d|%d) [lagr++ %d] norm_pen %e, norm_rpen %e, penetration_tol %e, "
                       "penalty_param "
                       "%e\n",
                       iter,
                       count_inner_iter,
                       count_linear_solver_iter,
                       count_lagr_mult_updates,
                       norm_pen,
                       norm_rpen,
                       penetration_tol,
                       penalty_param);
            }
        }
    };
}  // namespace sfem

#endif  // SFEM_ShiftedPenalty_HPP
