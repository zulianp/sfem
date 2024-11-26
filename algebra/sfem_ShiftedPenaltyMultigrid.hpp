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

// MATLAB version
// https://bitbucket.org/hkothari/matsci/src/ab637a0655512c4ddf299914dd45fdb563ac7b34/Solvers/%2BBoxConstraints/%40PenaltyMG/PenaltyMG.m?at=restructuring
namespace sfem {

    template <typename T>
    static std::shared_ptr<Operator<T>> diag_op(const std::shared_ptr<Buffer<T>>& diagonal_scaling,
                                                const ExecutionSpace es);

    /// level 0 is the finest
    template <typename T>
    class ShiftedPenaltyMultigrid final : public Operator<T> {
    public:
        void set_upper_bound(const std::shared_ptr<Buffer<T>>& ub) { upper_bound_ = ub; }
        void set_lower_bound(const std::shared_ptr<Buffer<T>>& lb) { lower_bound_ = lb; }
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

            count_smoothing_steps = 0;

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
            lagr_lb = lb ? make_buffer(n_dofs) : nullptr;
            lagr_ub = ub ? make_buffer(n_dofs) : nullptr;

            T penetration_norm = 0;
            T penetration_tol = 1 / (penalty_param_ * 0.1);

            int count_inner_iter = 0;
            int count_lagr_mult_updates = 0;
            T omega = 1 / penalty_param_;

            bool converged = false;
            for (iterations_ = 0; iterations_ < max_it_; iterations_++) {
                for (int inner_iter = 0; inner_iter < max_inner_it;
                     inner_iter++, count_inner_iter++) {
                    CycleReturnCode ret = nonlinear_cycle();
                    if (ret == CYCLE_CONVERGED) {
                        break;
                    }

                    blas_.zeros(n_dofs, mem->work->data());

                    // Compute material residual
                    op->apply(x, mem->work->data());
                    blas_.axpby(n_dofs, 1, mem->rhs->data(), -1, mem->work->data());

                    // Compute penalty residual
                    impl_.calc_r_pen(n_dofs,
                                     mem->solution->data(),
                                     penalty_param_,
                                     lb,
                                     ub,
                                     lagr_lb ? lagr_lb->data() : nullptr,
                                     lagr_ub ? lagr_ub->data() : nullptr,
                                     mem->work->data());

                    const T r_pen_norm = blas_.norm2(n_dofs, mem->work->data());

                    if (r_pen_norm < std::max(atol_, omega) && inner_iter != 0) {
                        break;
                    }
                }

                const T e_pen =
                        ((ub) ? impl_.sq_norm_ramp_p(n_dofs, mem->solution->data(), ub) : T(0)) +
                        ((lb) ? impl_.sq_norm_ramp_m(n_dofs, mem->solution->data(), lb) : T(0));

                const T norm_pen = std::sqrt(e_pen);
                const T norm_rpen = blas_.norm2(n_dofs, mem->work->data());

                if (norm_pen < penetration_tol) {
                    if (ub)
                        impl_.update_lagr_p(
                                n_dofs, penalty_param_, mem->solution->data(), ub, lagr_ub->data());
                    if (lb)
                        impl_.update_lagr_m(
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
                    printf("lagr_ub: %e\n", blas_.norm2(n_dofs, lagr_ub->data()));
                }

                if (debug && lb) {
                    printf("lagr_lb: %e\n", blas_.norm2(n_dofs, lagr_lb->data()));
                }

                monitor(iterations_,
                        count_inner_iter,
                        count_smoothing_steps,
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

        void monitor(const int iter, const int count_inner_iter, const int count_smoothing_steps,
                     const int count_lagr_mult_updates, const T norm_pen, const T norm_rpen,
                     const T penetration_tol, const T penalty_param) {
            // if (iter == max_it_ || (norm_pen < atol_ && norm_rpen < atol_)) {
            printf("%d|%d|%d) [lagr++ %d] norm_pen %e, norm_rpen %e, penetration_tol %e, "
                   "penalty_param "
                   "%e\n",
                   iter,
                   count_inner_iter,
                   count_smoothing_steps,
                   count_lagr_mult_updates,
                   norm_pen,
                   norm_rpen,
                   penetration_tol,
                   penalty_param);
            // }
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
            OpenMP_BLAS<T>::build_blas(blas_);
            OpenMP_ShiftedPenalty<T>::build(impl_);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void set_max_it(const int val) { max_it_ = val; }

        void set_atol(const T val) { atol_ = val; }

        void set_nlsmooth_steps(const int steps) { nlsmooth_steps = steps; }
        BLAS_Tpl<T>& blas() { return blas_; }
        ShiftedPenalty_Tpl<T>& impl() { return impl_; }

        bool debug{false};

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
        int cycle_type_{4};
        T atol_{1e-10};

        BLAS_Tpl<T> blas_;
        ShiftedPenalty_Tpl<T> impl_;
        bool verbose{true};

        std::shared_ptr<Buffer<T>> make_buffer(const ptrdiff_t n) const {
            return Buffer<T>::own(
                    n, blas_.allocate(n), blas_.destroy, (enum MemorySpace)execution_space());
        }

        std::shared_ptr<Buffer<T>> upper_bound_;
        std::shared_ptr<Buffer<T>> lower_bound_;
        std::shared_ptr<Buffer<T>> correction, lagr_lb, lagr_ub;

        T penalty_param_{10};  // mu
        T max_penalty_param_{1000};
        int nlsmooth_steps{10};
        int max_inner_it{3};

        ptrdiff_t count_smoothing_steps{0};

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

            correction = make_buffer(rows());
            for (int l = 0; l < n_levels(); l++) {
                memory_[l] = std::make_shared<Memory>();

                const ptrdiff_t n = smoother_[l]->rows();
                if (l != finest_level() || !wrap_input_) {
                    memory_[l]->solution = make_buffer(n);
                    memory_[l]->rhs = make_buffer(n);
                }

                memory_[l]->work = make_buffer(n);
                memory_[l]->diag = make_buffer(n);
            }

            return 0;
        }

        void eval_residual_and_jacobian() {
            const int level = finest_level();
            auto mem = memory_[level];
            auto smoother = smoother_[level];
            auto op = operator_[level];

            const ptrdiff_t n_dofs = op->rows();

            const T* const lb = (lower_bound_) ? lower_bound_->data() : nullptr;
            const T* const ub = (upper_bound_) ? upper_bound_->data() : nullptr;
            const T* const l_lb = lagr_lb ? lagr_lb->data() : nullptr;
            const T* const l_ub = lagr_ub ? lagr_ub->data() : nullptr;

            blas_.zeros(n_dofs, mem->work->data());

            // Compute material residual
            op->apply(mem->solution->data(), mem->work->data());
            blas_.axpby(n_dofs, 1, mem->rhs->data(), -1, mem->work->data());

            // Compute penalty residual
            impl_.calc_r_pen(n_dofs,
                             mem->solution->data(),
                             penalty_param_,
                             lb,
                             ub,
                             l_lb,
                             l_ub,
                             mem->work->data());

            blas_.zeros(n_dofs, mem->diag->data());
            impl_.calc_J_pen(n_dofs,
                             mem->solution->data(),
                             penalty_param_,
                             lb,
                             ub,
                             l_lb,
                             l_ub,
                             mem->diag->data());

            // if (debug) {
            //     printf("eval_residual_and_jacobian: ||r|| %e\n",
            //            blas_.norm2(n_dofs, mem->work->data()));
            // }
        }

        void nonlinear_smooth() {
            const int level = finest_level();
            auto mem = memory_[level];
            auto smoother = smoother_[level];
            auto op = operator_[level];

            const ptrdiff_t n_dofs = op->rows();
            for (int ns = 0; ns < nlsmooth_steps; ns++) {
                eval_residual_and_jacobian();

                smoother->set_op_and_diag_shift(op, mem->diag);

                blas_.zeros(n_dofs, correction->data());
                smoother->apply(mem->work->data(), correction->data());
                blas_.axpy(n_dofs, 1, correction->data(), mem->solution->data());

                count_smoothing_steps += smoother->iterations();
            }
        }

        CycleReturnCode nonlinear_cycle() {
            const int level = finest_level();
            auto mem = memory_[level];
            auto smoother = smoother_[level];
            auto op = operator_[level];
            auto restriction = restriction_[level];
            auto prolongation = prolongation_[coarser_level(level)];
            auto mem_coarse = memory_[coarser_level(level)];

            nonlinear_smooth();

            {
                // Evaluate for restriction
                eval_residual_and_jacobian();

                // Restriction
                blas_.zeros(mem_coarse->rhs->size(), mem_coarse->rhs->data());
                restriction->apply(mem->work->data(), mem_coarse->rhs->data());

                blas_.zeros(mem_coarse->diag->size(), mem_coarse->diag->data());
                restriction->apply(mem->diag->data(), mem_coarse->diag->data());

                blas_.zeros(mem_coarse->solution->size(), mem_coarse->solution->data());
            }

            cycle(coarser_level(finest_level()));
            assert(ret != CYCLE_FAILURE);

            {
                // Prolongation
                blas_.zeros(mem->work->size(), mem->work->data());
                prolongation->apply(mem_coarse->solution->data(), mem->work->data());

                // Apply coarse space correction
                blas_.axpby(mem->size(), 1, mem->work->data(), 1, mem->solution->data());
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
                smoother->set_op_and_diag_shift(op, mem->diag);

                blas_.zeros(mem->solution->size(), mem->solution->data());
                if (!smoother->apply(mem->rhs->data(), mem->solution->data())) {
                    return CYCLE_CONTINUE;
                } else {
                    return CYCLE_FAILURE;
                }
            }

            auto restriction = restriction_[level];
            auto prolongation = prolongation_[coarser_level(level)];
            auto mem_coarse = memory_[coarser_level(level)];

            for (int k = 0; k < this->cycle_type_; k++) {
                smoother->set_op_and_diag_shift(op, mem->diag);
                smoother->apply(mem->rhs->data(), mem->solution->data());

                {
                    // Compute residual
                    auto J = op + sfem::diag_op(mem->diag, execution_space());
                    blas_.zeros(mem->size(), mem->work->data());
                    J->apply(mem->solution->data(), mem->work->data());
                    blas_.axpby(mem->size(), 1, mem->rhs->data(), -1, mem->work->data());
                }

                {
                    // Restriction
                    blas_.zeros(mem_coarse->rhs->size(), mem_coarse->rhs->data());
                    restriction->apply(mem->work->data(), mem_coarse->rhs->data());
                    blas_.zeros(mem_coarse->solution->size(), mem_coarse->solution->data());

                    // FIXME do the restiction only after the nonlinear smoothing
                    blas_.zeros(mem_coarse->diag->size(), mem_coarse->diag->data());
                    restriction->apply(mem->diag->data(), mem_coarse->diag->data());
                }

                CycleReturnCode ret = cycle(coarser_level(level));
                assert(ret != CYCLE_FAILURE);

                {
                    // Prolongation
                    blas_.zeros(mem->work->size(), mem->work->data());
                    prolongation->apply(mem_coarse->solution->data(), mem->work->data());

                    // Apply coarse space correction
                    blas_.axpby(mem->size(), 1, mem->work->data(), 1, mem->solution->data());
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
