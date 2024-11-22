#ifndef SFEM_SHIFTED_PENALTY_MULTIGRID_HPP
#define SFEM_SHIFTED_PENALTY_MULTIGRID_HPP

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_ShiftedPenalty_impl.hpp"

#include "sfem_Buffer.hpp"

// https://en.wikipedia.org/wiki/Conjugate_gradient_method
namespace sfem {

    /// level 0 is the finest
    template <typename T>
    class ShiftedPenaltyMultigrid final : public Operator<T> {
    public:
        BLAS_Tpl<T> blas;
        ShiftedPenalty_Tpl<T> impl;
        bool verbose{true};
        bool debug{false};

        void set_upper_bound(const std::shared_ptr<Buffer<T>>& ub) { upper_bound_ = ub; }
        void set_lower_bound(const std::shared_ptr<Buffer<T>>& lb) { lower_bound_ = lb; }

        std::shared_ptr<Buffer<T>> make_buffer(const ptrdiff_t n) const {
            return Buffer<T>::own(
                    n, blas.allocate(n), blas.destroy, (enum MemorySpace)execution_space());
        }

        std::shared_ptr<Buffer<T>> upper_bound_;
        std::shared_ptr<Buffer<T>> lower_bound_;
        std::shared_ptr<Buffer<T>> correction, lagr_lb, lagr_ub;

        T penalty_param_{10};  // mu
        T max_penalty_param_{1000};
        int nlsmooth_steps{1};
        int max_inner_it{10};


        void set_penalty_parameter(const T val) { penalty_param_ = val; }

        enum CycleType {
            V_CYCLE = 1,
            W_CYCLE = 2,
        };

        enum CycleReturnCode { CYCLE_CONTINUE = 0, CYCLE_CONVERGED = 1, CYCLE_FAILURE = 2 };

        class Memory {
        public:
            std::shared_ptr<Buffer<T>> rhs;
            std::shared_ptr<Buffer<T>> solution;
            std::shared_ptr<Buffer<T>> work;
            std::shared_ptr<Buffer<T>> diag;
            inline ptrdiff_t size() const { return solution->size(); }
            ~Memory() {}
        };

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        ExecutionSpace execution_space() const override { return execution_space_; }

        int apply(const T* const rhs, T* const x) override {
            ensure_init();

            // Wrap input arrays into fine level of mg
            if (wrap_input_) {
                memory_[finest_level()]->solution =
                        Buffer<T>::wrap(smoother_[finest_level()]->rows(), x);

                memory_[finest_level()]->rhs =
                        Buffer<T>::wrap(smoother_[finest_level()]->rows(), (T*)rhs);
            }

            const int level = finest_level();
            auto mem = memory_[level];
            auto smoother = smoother_[level];
            auto op = operator_[level];

            const ptrdiff_t n_dofs = op->rows();

            T* lb = (lower_bound_) ? lower_bound_->data() : nullptr;
            T* ub = (upper_bound_) ? upper_bound_->data() : nullptr;
            lagr_lb = lb? make_buffer(n_dofs) : nullptr;
            lagr_ub = ub? make_buffer(n_dofs) : nullptr;

            T penetration_norm = 0;
            T penetration_tol = 1 / (penalty_param_ * 0.1);

            int count_inner_iter = 0;
            int count_linear_solver_iter = 0;
            int count_lagr_mult_updates = 0;
            T omega = 1 / penalty_param_;

            bool converged = false;
            for (iterations_ = 0; iterations_ < max_it_; iterations_++) {
                for (int inner_iter = 0; inner_iter < max_inner_it; inner_iter++) {
                    CycleReturnCode ret = nonlinear_cycle();
                    if (ret == CYCLE_CONVERGED) {
                        break;
                    }

                    blas.zeros(n_dofs, mem->work->data());

                    // Compute material residual
                    op->apply(x, mem->work->data());
                    blas.axpby(n_dofs, 1, mem->rhs->data(), -1, mem->work->data());

                    // Compute penalty residual
                    impl.calc_r_pen(n_dofs,
                                    mem->solution->data(),
                                    penalty_param_,
                                    lb,
                                    ub,
                                    lagr_lb->data(),
                                    lagr_ub->data(),
                                    mem->work->data());

                    const T r_pen_norm = blas.norm2(n_dofs, mem->work->data());

                    if (r_pen_norm < std::max(atol_, omega) && inner_iter != 0) {
                        converged = true;
                        break;
                    }
                }

                const T e_pen =
                        ((ub) ? impl.sq_norm_ramp_p(n_dofs, mem->solution->data(), ub) : T(0)) +
                        ((lb) ? impl.sq_norm_ramp_m(n_dofs, mem->solution->data(), lb) : T(0));

                const T norm_pen = std::sqrt(e_pen);
                const T norm_rpen = blas.norm2(n_dofs, mem->work->data());

                if (norm_pen < penetration_tol) {
                    if (ub)
                        impl.update_lagr_p(
                                n_dofs, penalty_param_, mem->solution->data(), ub, lagr_ub->data());
                    if (lb)
                        impl.update_lagr_m(
                                n_dofs, penalty_param_, mem->solution->data(), lb, lagr_lb->data());

                    penetration_tol = penetration_tol / pow(penalty_param_, 0.9);
                    omega = omega / penalty_param_;

                    count_lagr_mult_updates++;
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
                        count_lagr_mult_updates,
                        norm_pen,
                        norm_rpen,
                        penetration_tol,
                        penalty_param_);

                if (norm_pen < atol_ && norm_rpen < atol_) {
                    converged = true;
                    break;
                }
            }

            return 0;
        }

        void monitor(const int iter, const int count_inner_iter, const int count_linear_solver_iter,
                     const int count_lagr_mult_updates, const T norm_pen, const T norm_rpen,
                     const T penetration_tol, const T penalty_param) {
            if (iter == max_it_ || (norm_pen < atol_ && norm_rpen < atol_)) {
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

        void clear() {
            prolongation_.clear();
            restriction_.clear();
            smoother_.clear();
        }

        inline int n_levels() const { return smoother_.size(); }

        inline std::ptrdiff_t rows() const override { return operator_[finest_level()]->rows(); }
        inline std::ptrdiff_t cols() const override { return operator_[finest_level()]->cols(); }

        // Fine level prolongation has to be null
        // Coarse level restriction has to be null
        inline void add_level(const std::shared_ptr<Operator<T>>& op,
                              const std::shared_ptr<MatrixFreeLinearSolver<T>>& smoother_or_solver,
                              const std::shared_ptr<Operator<T>>& prolongation,
                              const std::shared_ptr<Operator<T>>& restriction) {
            operator_.push_back(op);
            smoother_.push_back(smoother_or_solver);
            prolongation_.push_back(prolongation);
            restriction_.push_back(restriction);
        }

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            OpenMP_ShiftedPenalty<T>::build_shifted_penalty(impl);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void set_max_it(const int val) { max_it_ = val; }

        void set_atol(const T val) { atol_ = val; }

    private:
        std::vector<std::shared_ptr<Operator<T>>> operator_;
        std::vector<std::shared_ptr<MatrixFreeLinearSolver<T>>> smoother_;

        std::vector<std::shared_ptr<Operator<T>>> prolongation_;
        std::vector<std::shared_ptr<Operator<T>>> restriction_;

        // Internals
        std::vector<std::shared_ptr<Memory>> memory_;
        bool wrap_input_{true};

        int max_it_{10};
        int iterations_{0};
        int cycle_type_{V_CYCLE};
        T atol_{1e-10};

        T norm_residual_0{1};
        T norm_residual_previous{1};

        inline int finest_level() const { return 0; }

        inline int coarsest_level() const { return n_levels() - 1; }

        inline int coarser_level(int level) const { return level + 1; }

        inline int finer_level(int level) const { return level - 1; }

        void ensure_init() {
            if (memory_.empty()) {
                init();
            }
        }

        int init() {
            assert(prolongation_.size() == restriction_.size());
            assert(operator_.size() == smoother_.size());

            memory_.clear();
            memory_.resize(this->n_levels());

            for (int l = 0; l < n_levels(); l++) {
                memory_[l] = std::make_shared<Memory>();

                size_t n = smoother_[l]->rows();
                if (l != finest_level() || !wrap_input_) {
                    auto x = blas.allocate(n);
                    memory_[l]->solution = Buffer<T>::own(n, x, blas.destroy);

                    auto r = blas.allocate(n);
                    memory_[l]->rhs = Buffer<T>::own(n, r, blas.destroy);
                }

                auto w = blas.allocate(n);
                memory_[l]->work = Buffer<T>::own(n, w, blas.destroy);
                memory_[l]->diag = Buffer<T>::own(n, w, blas.destroy);
            }

            return 0;
        }

        void eval_residual_and_jacobian() {
            const int level = finest_level();
            auto mem = memory_[level];
            auto smoother = smoother_[level];
            auto op = operator_[level];

            const ptrdiff_t n_dofs = op->rows();

            T* lb = (lower_bound_) ? lower_bound_->data() : nullptr;
            T* ub = (upper_bound_) ? upper_bound_->data() : nullptr;

            // ---

            blas.zeros(n_dofs, mem->work->data());

            // Compute material residual
            op->apply(mem->solution->data(), mem->work->data());
            blas.axpby(n_dofs, 1, mem->rhs->data(), -1, mem->work->data());

            // Compute penalty residual
            impl.calc_r_pen(n_dofs,
                            mem->solution->data(),
                            penalty_param_,
                            lb,
                            ub,
                            lagr_lb->data(),
                            lagr_ub->data(),
                            mem->work->data());

            blas.zeros(n_dofs, mem->diag->data());
            impl.calc_J_pen(n_dofs,
                            mem->solution->data(),
                            penalty_param_,
                            lb,
                            ub,
                            lagr_lb->data(),
                            lagr_ub->data(),
                            mem->diag->data());
        }

        void nonlinear_smooth() {
            const int level = finest_level();
            auto mem = memory_[level];
            auto smoother = smoother_[level];
            auto op = operator_[level];

            const ptrdiff_t n_dofs = op->rows();
            for (int ns = 0; ns < nlsmooth_steps; ns++) {
                eval_residual_and_jacobian();

                auto J = op + sfem::diag_op(n_dofs, mem->diag, execution_space());
                smoother->set_op(J);

                blas.zeros(n_dofs, correction->data());
                // TODO Is there a way to remove correction?
                smoother->apply(mem->work->data(), correction->data());

                blas.axpy(n_dofs, 1, correction->data(), mem->solution->data());
            }
        }

        CycleReturnCode nonlinear_cycle() {
            const int level = finest_level();
            auto mem = memory_[level];
            auto smoother = smoother_[level];
            auto op = operator_[level];
            auto restriction = restriction_[level];
            auto prolongation = prolongation_[level];
            auto mem_coarse = memory_[coarser_level(level)];

            nonlinear_smooth();

            {
                // Evaluate for restriction
                eval_residual_and_jacobian();

                // Restriction
                blas.zeros(mem_coarse->rhs->size(), mem_coarse->rhs->data());
                restriction->apply(mem->work->data(), mem_coarse->rhs->data());

                blas.zeros(mem_coarse->diag->size(), mem_coarse->diag->data());
                restriction->apply(mem->diag->data(), mem_coarse->diag->data());

                blas.zeros(mem_coarse->solution->size(), mem_coarse->solution->data());
            }

            cycle(coarser_level(finest_level()));
            assert(ret != CYCLE_FAILURE);

            {
                // Prolongation
                blas.zeros(mem->work->size(), mem->work->data());
                prolongation->apply(mem_coarse->solution->data(), mem->work->data());

                // Apply coarse space correction
                blas.axpby(mem->size(), 1, mem->work->data(), 1, mem->solution->data());
            }

            nonlinear_smooth();

            return CYCLE_CONTINUE;
        }

        CycleReturnCode cycle(const int level) {
            auto mem = memory_[level];
            auto smoother = smoother_[level];
            auto op = operator_[level];

            ptrdiff_t n_dofs = op->rows();
            if (coarsest_level() == level) {
                auto J = op + sfem::diag_op(n_dofs, mem->diag, execution_space());
                smoother->set_op(J);

                blas.zeros(mem->solution->size(), mem->solution->data());
                if (!smoother->apply(mem->rhs->data(), mem->solution->data())) {
                    return CYCLE_CONTINUE;
                } else {
                    return CYCLE_FAILURE;
                }
            }

            auto restriction = restriction_[level];
            auto prolongation = prolongation_[level];
            auto mem_coarse = memory_[coarser_level(level)];

            for (int k = 0; k < this->cycle_type_; k++) {
                auto J = op + sfem::diag_op(n_dofs, mem->diag, execution_space());
                smoother->set_op(J);
                smoother->apply(mem->rhs->data(), mem->solution->data());

                {
                    // Compute residual
                    blas.zeros(mem->size(), mem->work->data());
                    J->apply(mem->solution->data(), mem->work->data());
                    blas.axpby(mem->size(), 1, mem->rhs->data(), -1, mem->work->data());
                }

                {
                    // Restriction
                    blas.zeros(mem_coarse->rhs->size(), mem_coarse->rhs->data());
                    restriction->apply(mem->work->data(), mem_coarse->rhs->data());
                    blas.zeros(mem_coarse->solution->size(), mem_coarse->solution->data());

                    blas.zeros(mem_coarse->diag->size(), mem_coarse->diag->data());
                    restriction->apply(mem->diag->data(), mem_coarse->diag->data());
                }

                CycleReturnCode ret = cycle(coarser_level(level));
                assert(ret != CYCLE_FAILURE);

                {
                    // Prolongation
                    blas.zeros(mem->work->size(), mem->work->data());
                    prolongation->apply(mem_coarse->solution->data(), mem->work->data());

                    // Apply coarse space correction
                    blas.axpby(mem->size(), 1, mem->work->data(), 1, mem->solution->data());
                }

                smoother->apply(mem->rhs->data(), mem->solution->data());
            }
            return CYCLE_CONTINUE;
        }
    };

    template <typename T>
    std::shared_ptr<ShiftedPenaltyMultigrid<T>> h_spmg() {
        auto mg = std::make_shared<ShiftedPenaltyMultigrid<T>>();
        mg->default_init();
        return mg;
    }

}  // namespace sfem

#endif  // SFEM_SHIFTED_PENALTY_MULTIGRID_HPP
