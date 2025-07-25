#ifndef SFEM_SHIFTED_PENALTY_MULTIGRID_HPP
#define SFEM_SHIFTED_PENALTY_MULTIGRID_HPP

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_ShiftedPenalty_impl.hpp"
#include "sfem_openmp_blas.hpp"
#include "sfem_tpl_blas.hpp"

#include "sfem_Buffer.hpp"
#include "sfem_Tracer.hpp"

// MATLAB version
// https://bitbucket.org/hkothari/matsci/src/ab637a0655512c4ddf299914dd45fdb563ac7b34/Solvers/%2BBoxConstraints/%40PenaltyMG/PenaltyMG.m?at=restructuring
namespace sfem {

    template <typename T>
    static std::shared_ptr<Operator<T>> diag_op(const SharedBuffer<T>& diagonal_scaling, const ExecutionSpace es);

    /**
     * @class ShiftedPenaltyMultigrid
     * @brief A multigrid solver with shifted penalty method for constrained optimization problems.
     *
     * This class implements a multigrid solver that incorporates a shifted penalty method
     * to handle constraints. It supports various cycle types (V-cycle, W-cycle) and provides
     * mechanisms for nonlinear smoothing, penalty parameter updates, and coarse space corrections.
     * Level 0 is the finest.
     *
     * @tparam T The data type for numerical computations (e.g., float, double).
     *
     * ### Public Enums:
     * - `CycleType`: Defines the type of multigrid cycle (V_CYCLE, W_CYCLE).
     * - `CycleReturnCode`: Return codes for the multigrid cycle (CYCLE_CONTINUE, CYCLE_CONVERGED, CYCLE_FAILURE).
     *
     * ### Nested Classes:
     * - `Memory`: Manages memory buffers for solution, right-hand side, work, and diagonal data.
     * - `Stats`: Collects and outputs statistics about the solver's performance.
     *
     * ### Public Methods:
     * - `set_upper_bound`: Sets the upper bound for the solution.
     * - `set_lower_bound`: Sets the lower bound for the solution.
     * - `set_penalty_parameter`: Sets the penalty parameter for the shifted penalty method.
     * - `set_constraints_op`: Sets the constraint operators and their transpose.
     * - `add_level_constraint_op_x_op`: Adds a level-specific constraint operator.
     * - `apply`: Applies the multigrid solver to solve the system.
     * - `clear`: Clears all levels and associated data.
     * - `n_levels`: Returns the number of levels in the multigrid hierarchy.
     * - `rows`: Returns the number of rows in the finest level operator.
     * - `cols`: Returns the number of columns in the finest level operator.
     * - `add_level`: Adds a level to the multigrid hierarchy.
     * - `add_constraints_restriction`: Adds a restriction operator for constraints.
     * - `default_init`: Initializes the solver with default settings.
     * - `set_max_it`: Sets the maximum number of iterations.
     * - `set_max_inner_it`: Sets the maximum number of inner iterations.
     * - `set_atol`: Sets the absolute tolerance for convergence.
     * - `set_nlsmooth_steps`: Sets the number of nonlinear smoothing steps.
     * - `set_cycle_type`: Sets the type of multigrid cycle.
     * - `set_project_coarse_space_correction`: Enables or disables coarse space correction projection.
     * - `set_max_penalty_param`: Sets the maximum penalty parameter.
     * - `set_penalty_param`: Sets the initial penalty parameter.
     * - `enable_line_search`: Enables or disables line search for coarse space corrections.
     * - `collect_energy_norm_correction`: Enables or disables collection of energy norm corrections.
     * - `set_debug`: Enables or disables debug mode.
     *
     * ### Private Methods:
     * - `make_buffer`: Creates a memory buffer for a given size.
     * - `ensure_init`: Ensures that the solver is initialized.
     * - `init`: Initializes the solver's memory and data structures.
     * - `shifted_op`: Constructs a shifted operator for a given level.
     * - `eval_residual_and_jacobian`: Evaluates the residual and Jacobian for the current solution.
     * - `penalty_pseudo_galerkin_assembly`: Assembles the penalty pseudo-Galerkin operator.
     * - `nonlinear_smooth`: Performs nonlinear smoothing on the finest level.
     * - `nonlinear_cycle`: Executes a nonlinear multigrid cycle.
     * - `cycle`: Executes a multigrid cycle for a given level.
     * - `collect_stats`: Collects solver statistics.
     * - `write_stats`: Writes solver statistics to a CSV file.
     * - `monitor`: Monitors and logs the solver's progress.
     *
     * ### Private Members:
     * - `execution_space_`: The execution space for computations (e.g., host or device).
     * - `operator_`: Operators for each level in the multigrid hierarchy.
     * - `smoother_`: Smoothers or solvers for each level.
     * - `prolongation_`: Prolongation operators for each level.
     * - `restriction_`: Restriction operators for each level.
     * - `constraints_op_`: Constraint operator for the finest level.
     * - `constraints_op_transpose_`: Transpose of the constraint operator.
     * - `constraints_op_x_op_`: Constraint operators for each level.
     * - `constraints_restriction_`: Restriction operators for constraints.
     * - `upper_bound_`: Upper bound for the solution.
     * - `lower_bound_`: Lower bound for the solution.
     * - `correction`, `lagr_lb`, `lagr_ub`: Buffers for corrections and Lagrange multipliers.
     * - `memory_`: Memory buffers for each level.
     * - `wrap_input_`: Flag to indicate whether to wrap input arrays.
     * - `max_it_`, `iterations_`: Maximum and current iteration counts.
     * - `cycle_type_`: Type of multigrid cycle.
     * - `atol_`: Absolute tolerance for convergence.
     * - `verbose`: Flag for verbose output.
     * - `project_coarse_space_correction_`: Flag for coarse space correction projection.
     * - `enable_line_search_`: Flag for enabling line search.
     * - `skip_coarse`: Flag to skip coarse grid corrections.
     * - `collect_energy_norm_correction_`: Flag to collect energy norm corrections.
     * - `penalty_param_`, `max_penalty_param_`: Penalty parameter and its maximum value.
     * - `nlsmooth_steps`: Number of nonlinear smoothing steps.
     * - `max_inner_it`: Maximum number of inner iterations.
     * - `count_smoothing_steps`: Counter for smoothing steps.
     * - `debug`: Flag for debug mode.
     * - `stats`: Vector to store solver statistics.
     * - `blas_`: BLAS implementation for numerical operations.
     * - `impl_`: Implementation of the shifted penalty method.
     */

    template <typename T>
    class ShiftedPenaltyMultigrid final : public Operator<T> {
    public:
        enum CycleType {
            V_CYCLE = 1,
            W_CYCLE = 2,
        };

        enum CycleReturnCode { CYCLE_CONTINUE = 0, CYCLE_CONVERGED = 1, CYCLE_FAILURE = 2 };

        class Memory {
        public:
            SharedBuffer<T>  rhs;
            SharedBuffer<T>  solution;
            SharedBuffer<T>  work;
            SharedBuffer<T>  diag;
            inline ptrdiff_t size() const { return solution->size(); }
            ~Memory() {}
        };

        struct Stats {
            int    count_iter;
            int    count_mg_cycles;
            int    count_nl_smooth;
            int    count_smooth;
            real_t norm_penetration;
            real_t norm_residual;
            real_t energy_norm_correction;
            real_t penalty_param;
            real_t omega;

            static void header(std::ostream& os) {
                os << "count_iter,";
                os << "count_mg_cycles,";
                os << "count_nl_smooth,";
                os << "count_smooth,";
                os << "norm_penetration,";
                os << "norm_residual,";
                os << "energy_norm_correction,";
                os << "penalty_param,";
                os << "omega,";
                os << "rate\n";
            }

            friend std::ostream& operator<<(std::ostream& os, const Stats& stats) {
                os << stats.count_iter << ",";
                os << stats.count_mg_cycles << ",";
                os << stats.count_nl_smooth << ",";
                os << stats.count_smooth << ",";
                os << stats.norm_penetration << ",";
                os << stats.norm_residual << ",";
                os << stats.energy_norm_correction << ",";
                os << stats.penalty_param << ",";
                os << stats.omega << ",";
                return os;
            }
        };

        void set_upper_bound(const SharedBuffer<T>& ub) { upper_bound_ = ub; }
        void set_lower_bound(const SharedBuffer<T>& lb) { lower_bound_ = lb; }
        void set_penalty_parameter(const T val) { penalty_param_ = val; }

        void set_constraints_op(const std::shared_ptr<Operator<T>>& op, const std::shared_ptr<Operator<T>>& op_t) {
            constraints_op_           = op;
            constraints_op_transpose_ = op_t;
        }

        void add_level_constraint_op_x_op(const std::shared_ptr<SparseBlockVector<T>>& constraints_op_x_op) {
            constraints_op_x_op_.push_back(constraints_op_x_op);
        }

        ExecutionSpace execution_space() const override { return execution_space_; }

        int apply(const T* const rhs, T* const x) override {
            SFEM_TRACE_SCOPE("ShiftedPenaltyMultigrid::apply");

            ensure_init();

            count_smoothing_steps = 0;

            // Wrap input arrays into fine level of mg
            if (wrap_input_) {
                memory_[finest_level()]->solution = Buffer<T>::wrap(smoother_[finest_level()]->rows(), x);

                memory_[finest_level()]->rhs = Buffer<T>::wrap(smoother_[finest_level()]->rows(), (T*)rhs);
            }

            const int level    = finest_level();
            auto      mem      = memory_[level];
            auto      smoother = smoother_[level];
            auto      op       = operator_[level];

            const ptrdiff_t n_dofs             = op->rows();
            const ptrdiff_t n_constrained_dofs = (constraints_op_ ? constraints_op_->rows() : n_dofs);

            T* lb               = (lower_bound_) ? lower_bound_->data() : nullptr;
            T* ub               = (upper_bound_) ? upper_bound_->data() : nullptr;
            lagr_lb             = lb ? make_buffer(n_constrained_dofs) : nullptr;
            lagr_ub             = ub ? make_buffer(n_constrained_dofs) : nullptr;
            const T* const l_lb = lagr_lb ? lagr_lb->data() : nullptr;
            const T* const l_ub = lagr_ub ? lagr_ub->data() : nullptr;

            T penetration_norm = 0;
            T penetration_tol  = 1 / (penalty_param_ * 0.1);

            int count_inner_iter        = 0;
            int count_lagr_mult_updates = 0;
            T   omega                   = 1 / penalty_param_;

            SharedBuffer<T> x_old;
            if (collect_energy_norm_correction_) {
                x_old = make_buffer(n_dofs);
                blas_.copy(n_dofs, x, x_old->data());
            }

            bool converged = false;
            for (iterations_ = 0; iterations_ < max_it_; iterations_++) {
                for (int inner_iter = 0; inner_iter < max_inner_it; inner_iter++) {
                    count_inner_iter++;

                    CycleReturnCode ret = nonlinear_cycle();

                    if (ret == CYCLE_CONVERGED) {
                        break;
                    }

                    if (constraints_op_) {
                        blas_.zeros(n_constrained_dofs, correction->data());

                        // Solution space to constraints space
                        constraints_op_->apply(mem->solution->data(), correction->data());

                        // Constraints space to solution space
                        blas_.zeros(n_constrained_dofs, mem->work->data());
                        impl_.calc_r_pen(
                                n_constrained_dofs, correction->data(), penalty_param_, lb, ub, l_lb, l_ub, mem->work->data());

                        blas_.zeros(n_dofs, correction->data());

                        // Constraints space to solution space
                        constraints_op_transpose_->apply(mem->work->data(), correction->data());

                        blas_.zeros(n_dofs, mem->work->data());
                        op->apply(mem->solution->data(), mem->work->data());
                        blas_.axpby(n_dofs, 1, mem->rhs->data(), -1, mem->work->data());
                        blas_.axpy(n_dofs, 1, correction->data(), mem->work->data());

                    } else {
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
                    }

                    const T r_pen_norm = blas_.norm2(n_dofs, mem->work->data());

                    if (debug) {
                        printf("%d) r_norm=%g (<%g)\n", inner_iter, (double)r_pen_norm, omega);
                    }

                    if (r_pen_norm < std::max(atol_, omega) && inner_iter != 0) {
                        break;
                    }
                }

                auto Tx = x;

                if (constraints_op_) {
                    blas_.zeros(n_constrained_dofs, correction->data());
                    constraints_op_->apply(x, correction->data());
                    Tx = correction->data();
                }

                const T e_pen = ((ub) ? impl_.sq_norm_ramp_p(n_constrained_dofs, Tx, ub) : T(0)) +
                                ((lb) ? impl_.sq_norm_ramp_m(n_constrained_dofs, Tx, lb) : T(0));

                const T norm_pen  = std::sqrt(e_pen);
                const T norm_rpen = blas_.norm2(n_dofs, mem->work->data());

                if (enable_shift) {
                    if (ub) impl_.update_lagr_p(n_constrained_dofs, penalty_param_, Tx, ub, lagr_ub->data());
                    if (lb) impl_.update_lagr_m(n_constrained_dofs, penalty_param_, Tx, lb, lagr_lb->data());
                    count_lagr_mult_updates++;
                }

                // Store it for diagonstics
                const T prev_penalty_param = penalty_param_;
                const T prev_omega         = omega;

                // I moved the previous three lines outside of the if
                if (norm_pen < penetration_tol) {
                    penetration_tol = penetration_tol / pow(penalty_param_, penetration_tol_exp);
                    omega           = std::max(atol_, omega / penalty_param_);

                } else {
                    penalty_param_  = std::min(penalty_param_ * penalty_param_increase, max_penalty_param_);
                    penetration_tol = 1 / pow(penalty_param_, (1 - penetration_tol_exp));
                    omega           = 1 / penalty_param_;
                }

                if (debug && ub) {
                    printf("lagr_ub: %e\n", blas_.norm2(n_constrained_dofs, lagr_ub->data()));
                }

                if (debug && lb) {
                    printf("lagr_lb: %e\n", blas_.norm2(n_constrained_dofs, lagr_lb->data()));
                }

                monitor(iterations_ + 1,
                        count_inner_iter,
                        count_smoothing_steps,
                        count_lagr_mult_updates,
                        norm_pen,
                        norm_rpen,
                        penetration_tol,
                        prev_penalty_param);

                real_t energy_norm_correction = -1;
                if (collect_energy_norm_correction_) {
                    SFEM_TRACE_SCOPE("collect_energy_norm_correction");

                    blas_.zaxpby(n_dofs, 1, x, -1, x_old->data(), correction->data());
                    blas_.zeros(n_dofs, x_old->data());
                    op->apply(correction->data(), x_old->data());
                    energy_norm_correction = sqrt(blas_.dot(n_dofs, x_old->data(), correction->data()));
                    blas_.copy(n_dofs, x, x_old->data());
                }

                collect_stats({.count_iter             = iterations_ + 1,
                               .count_mg_cycles        = count_inner_iter,
                               .count_nl_smooth        = 2 * (count_inner_iter * nlsmooth_steps),
                               .count_smooth           = count_smoothing_steps,
                               .norm_penetration       = norm_pen,
                               .norm_residual          = norm_rpen,
                               .energy_norm_correction = energy_norm_correction,
                               .penalty_param          = prev_penalty_param,
                               .omega                  = prev_omega});

                if (norm_pen < atol_ && norm_rpen < atol_) {
                    converged = true;
                    break;
                }
            }

            write_stats();
            return SFEM_SUCCESS;
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
        inline void add_level(const std::shared_ptr<Operator<T>>&               op,
                              const std::shared_ptr<MatrixFreeLinearSolver<T>>& smoother_or_solver,
                              const std::shared_ptr<Operator<T>>&               prolongation,
                              const std::shared_ptr<Operator<T>>&               restriction) {
            operator_.push_back(op);
            smoother_.push_back(smoother_or_solver);
            prolongation_.push_back(prolongation);
            restriction_.push_back(restriction);
        }

        inline void add_constraints_restriction(const std::shared_ptr<Operator<T>>& restict_diag) {
            constraints_restriction_.push_back(restict_diag);
        }

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas_);
            OpenMP_ShiftedPenalty<T>::build(impl_);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void set_max_it(const int val) { max_it_ = val; }
        void set_max_inner_it(const int val) { max_inner_it = val; }
        void set_atol(const T val) { atol_ = val; }
        void set_nlsmooth_steps(const int steps) { nlsmooth_steps = steps; }
        void set_cycle_type(const int val) { cycle_type_ = val; }
        void set_project_coarse_space_correction(const bool val) { project_coarse_space_correction_ = val; }
        void set_max_penalty_param(const real_t val) { max_penalty_param_ = val; }
        void set_penalty_param(const real_t val) { penalty_param_ = val; }
        void enable_line_search(const bool val) { enable_line_search_ = val; }
        void collect_energy_norm_correction(const bool val) { collect_energy_norm_correction_ = val; }
        void set_debug(const int val) { debug = val; }
        void set_execution_space(enum ExecutionSpace es) { execution_space_ = es; }
        void set_penalty_param_increase(const real_t val) { penalty_param_increase = val; }
        void set_enable_shift(const bool val) { enable_shift = val; }

        void set_update_constraints(std::function<void(const T* const)> fun) { update_constraints_ = fun; }

        BLAS_Tpl<T>&           blas() { return blas_; }
        ShiftedPenalty_Tpl<T>& impl() { return impl_; }

    private:
        SharedBuffer<T> make_buffer(const ptrdiff_t n) const {
            return Buffer<T>::own(n, blas_.allocate(n), blas_.destroy, (enum MemorySpace)execution_space());
        }

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
            SFEM_TRACE_SCOPE("ShiftedPenaltyMultigrid::init");

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
                    memory_[l]->rhs      = make_buffer(n);
                }

                memory_[l]->work = make_buffer(n);
                if (constraints_op_) {
                    memory_[l]->diag = make_buffer(constraints_op_x_op_[l]->n_blocks());
                } else {
                    memory_[l]->diag = make_buffer(n);
                }
            }

            return SFEM_SUCCESS;
        }

        std::shared_ptr<Operator<T>> shifted_op(const int level) {
            if (constraints_op_) {
                return operator_[level] + sfem::create_sparse_block_vector_mult(
                                                  operator_[level]->rows(), constraints_op_x_op_[level], memory_[level]->diag);
            } else {
                return operator_[level] + sfem::diag_op(memory_[level]->diag, execution_space());
            }
        }

        void eval_residual_and_jacobian() {
            SFEM_TRACE_SCOPE("ShiftedPenaltyMultigrid::eval_residual_and_jacobian");

            const int level    = finest_level();
            auto      mem      = memory_[level];
            auto      smoother = smoother_[level];
            auto      op       = operator_[level];

            const ptrdiff_t n_dofs             = op->rows();
            const ptrdiff_t n_constrained_dofs = (constraints_op_ ? constraints_op_->rows() : n_dofs);

            const T* const lb   = (lower_bound_) ? lower_bound_->data() : nullptr;
            const T* const ub   = (upper_bound_) ? upper_bound_->data() : nullptr;
            const T* const l_lb = lagr_lb ? lagr_lb->data() : nullptr;
            const T* const l_ub = lagr_ub ? lagr_ub->data() : nullptr;

            if(debug > 1) {
                printf("Residual: %g\n", blas_.norm2(n_dofs, mem->work->data()));
                if(ub) {
                    printf("UB: %g, LUB %g\n", blas_.norm2(n_constrained_dofs, ub), blas_.norm2(n_constrained_dofs, l_ub));
                }
                if(lb) {
                    printf("LB: %g, LLB %g\n", blas_.norm2(n_constrained_dofs, lb), blas_.norm2(n_constrained_dofs, l_lb));
                }
            }

            if (constraints_op_) {
                // Jacobian
                blas_.zeros(n_constrained_dofs, correction->data());

                // Solution space to constraints space
                constraints_op_->apply(mem->solution->data(), correction->data());
                blas_.zeros(n_constrained_dofs, mem->diag->data());

                blas_.zeros(n_constrained_dofs, mem->diag->data());
                impl_.calc_J_pen(n_constrained_dofs, correction->data(), penalty_param_, lb, ub, l_lb, l_ub, mem->diag->data());

                // Residual
                blas_.zeros(n_constrained_dofs, mem->work->data());
                impl_.calc_r_pen(n_constrained_dofs, correction->data(), penalty_param_, lb, ub, l_lb, l_ub, mem->work->data());

                blas_.zeros(n_dofs, correction->data());

                // Constraints space to solution space
                constraints_op_transpose_->apply(mem->work->data(), correction->data());

                blas_.zeros(n_dofs, mem->work->data());
                op->apply(mem->solution->data(), mem->work->data());
                blas_.axpby(n_dofs, 1, mem->rhs->data(), -1, mem->work->data());
                blas_.axpy(n_dofs, 1, correction->data(), mem->work->data());

            } else {
                blas_.zeros(n_dofs, mem->work->data());

                // Compute material residual
                op->apply(mem->solution->data(), mem->work->data());
                blas_.axpby(n_dofs, 1, mem->rhs->data(), -1, mem->work->data());

                // Compute penalty residual
                impl_.calc_r_pen(n_dofs, mem->solution->data(), penalty_param_, lb, ub, l_lb, l_ub, mem->work->data());

                blas_.zeros(n_dofs, mem->diag->data());
                impl_.calc_J_pen(n_dofs, mem->solution->data(), penalty_param_, lb, ub, l_lb, l_ub, mem->diag->data());
            }
        }

        void penalty_pseudo_galerkin_assembly() {
            SFEM_TRACE_SCOPE("ShiftedPenaltyMultigrid::penalty_pseudo_galerkin_assembly");

            for (int l = finest_level(); l != coarsest_level(); l = coarser_level(l)) {
                auto mem_coarse = memory_[coarser_level(l)];
                blas_.zeros(mem_coarse->diag->size(), mem_coarse->diag->data());

                if (constraints_op_) {
                    constraints_restriction_[l]->apply(memory_[l]->diag->data(), mem_coarse->diag->data());
                } else {
                    restriction_[l]->apply(memory_[l]->diag->data(), mem_coarse->diag->data());
                }
            }
        }

        void nonlinear_smooth() {
            SFEM_TRACE_SCOPE("ShiftedPenaltyMultigrid::nonlinear_smooth");

            const int level    = finest_level();
            auto      mem      = memory_[level];
            auto      smoother = smoother_[level];
            auto      op       = operator_[level];

            const ptrdiff_t n_dofs = op->rows();
            for (int ns = 0; ns < nlsmooth_steps; ns++) {
                eval_residual_and_jacobian();

                if (constraints_op_) {
                    smoother->set_op_and_diag_shift(op, constraints_op_x_op_[level], mem->diag);
                } else {
                    smoother->set_op_and_diag_shift(op, mem->diag);
                }

                blas_.zeros(n_dofs, correction->data());
                smoother->apply(mem->work->data(), correction->data());
                blas_.axpy(n_dofs, 1, correction->data(), mem->solution->data());

                count_smoothing_steps += smoother->iterations();
            }
        }

        CycleReturnCode nonlinear_cycle() {
            SFEM_TRACE_SCOPE("ShiftedPenaltyMultigrid::nonlinear_cycle");

            const int level        = finest_level();
            auto      mem          = memory_[level];
            auto      smoother     = smoother_[level];
            auto      sop          = shifted_op(level);
            auto      restriction  = restriction_[level];
            auto      prolongation = prolongation_[coarser_level(level)];
            auto      mem_coarse   = memory_[coarser_level(level)];

            if (update_constraints_) {
                update_constraints_(mem->solution->data());
            }

            nonlinear_smooth();

            {
                // Evaluate for restriction
                eval_residual_and_jacobian();

                // Restriction
                blas_.zeros(mem_coarse->rhs->size(), mem_coarse->rhs->data());
                restriction->apply(mem->work->data(), mem_coarse->rhs->data());
                blas_.zeros(mem_coarse->solution->size(), mem_coarse->solution->data());
            }

            if (!skip_coarse) {
                penalty_pseudo_galerkin_assembly();

                int ret = cycle(coarser_level(finest_level()));
                
                if(ret == CYCLE_FAILURE) {
                    fprintf(stderr, "Coarse level solver did not converge as desired!\n");
                }

                {
                    // Prolongation
                    blas_.zeros(correction->size(), correction->data());
                    prolongation->apply(mem_coarse->solution->data(), correction->data());

                    if (enable_line_search_) {
                        // ATTENTION to code changes and side-effects

                        //  dot(c, (b - A * x))
                        T numerator = blas_.dot(correction->size(), correction->data(), mem->work->data());
                        blas_.zeros(mem->work->size(), mem->work->data());
                        sop->apply(correction->data(), mem->work->data());

                        // dot(c, A * c)
                        T denominator = blas_.dot(correction->size(), correction->data(), mem->work->data());
                        T alpha       = numerator / (denominator == 0 ? T(1e-16) : denominator);

                        if (debug) printf("alpha = %g\n", alpha);

                        blas_.scal(correction->size(), alpha, correction->data());
                    }

                    // FIXME if we find a good reason for this add GPU support here
                    if (project_coarse_space_correction_ && execution_space() == EXECUTION_SPACE_HOST) {
                        assert(!constraints_op_);  // FIXME not supported yet!!!

                        auto            c  = correction->data();
                        auto            ub = upper_bound_->data();
                        auto            x  = mem->solution->data();
                        const ptrdiff_t n  = correction->size();

#pragma omp parallel for
                        for (ptrdiff_t i = 0; i < n; i++) {
                            x[i] = std::min(x[i] + c[i], ub[i] + std::max(T(0), x[i] - ub[i]));
                        }

                    } else {
                        // Apply coarse space correction
                        blas_.axpby(mem->size(), 1, correction->data(), 1, mem->solution->data());
                    }
                }
            }

            nonlinear_smooth();

            return CYCLE_CONTINUE;
        }

        CycleReturnCode cycle(const int level) {
            SFEM_TRACE_SCOPE_VARIANT("ShiftedPenaltyMultigrid::cycle(%d)", level);

            auto mem      = memory_[level];
            auto smoother = smoother_[level];
            auto op       = operator_[level];
            auto sop      = shifted_op(level);

            const ptrdiff_t n_dofs = op->rows();

            assert(n_dofs == mem->solution->size());
            assert(n_dofs == mem->rhs->size());

            if (coarsest_level() == level) {
                if (constraints_op_) {
                    smoother->set_op_and_diag_shift(op, constraints_op_x_op_[level], mem->diag);
                } else {
                    smoother->set_op_and_diag_shift(op, mem->diag);
                }

                blas_.zeros(mem->solution->size(), mem->solution->data());
                if (!smoother->apply(mem->rhs->data(), mem->solution->data())) {
                    return CYCLE_CONTINUE;
                } else {
                    fflush(stdout);
                    fflush(stderr);
                    fprintf(stderr, "Coarse grid solver did not reach desired tol in %d\n", smoother->iterations());
                    return CYCLE_FAILURE;
                }
            }

            auto restriction  = restriction_[level];
            auto prolongation = prolongation_[coarser_level(level)];
            auto mem_coarse   = memory_[coarser_level(level)];

            for (int k = 0; k < this->cycle_type_; k++) {
                if (constraints_op_) {
                    smoother->set_op_and_diag_shift(op, constraints_op_x_op_[level], mem->diag);
                } else {
                    smoother->set_op_and_diag_shift(op, mem->diag);
                }

                smoother->apply(mem->rhs->data(), mem->solution->data());

                {
                    // Compute residual
                    blas_.zeros(mem->size(), mem->work->data());
                    sop->apply(mem->solution->data(), mem->work->data());
                    blas_.axpby(mem->size(), 1, mem->rhs->data(), -1, mem->work->data());
                }

                {
                    // Restriction
                    blas_.zeros(mem_coarse->rhs->size(), mem_coarse->rhs->data());
                    restriction->apply(mem->work->data(), mem_coarse->rhs->data());
                    blas_.zeros(mem_coarse->solution->size(), mem_coarse->solution->data());
                }

                CycleReturnCode ret = cycle(coarser_level(level));

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

        void collect_stats(struct Stats s) { stats.push_back(s); }

        void write_stats() {
            const char* SFEM_SHIFTED_PENALTY_MULTIGRID_STATS_PATH = "./spmg_stats.csv";
            SFEM_READ_ENV(SFEM_SHIFTED_PENALTY_MULTIGRID_STATS_PATH, );

            std::ofstream os(SFEM_SHIFTED_PENALTY_MULTIGRID_STATS_PATH);
            if (!os.good()) {
                fprintf(stderr,
                        "Unable to open file SFEM_SHIFTED_PENALTY_MULTIGRID_STATS_PATH=%s\n",
                        SFEM_SHIFTED_PENALTY_MULTIGRID_STATS_PATH);
            }

            Stats::header(os);

            real_t prev_norm = stats[0].energy_norm_correction;
            for (auto& s : stats) {
                os << s;
                os << s.energy_norm_correction / prev_norm << "\n";
                prev_norm = s.energy_norm_correction;
            }

            os.close();
        }

        void monitor(const int iter,
                     const int count_inner_iter,
                     const int count_smoothing_steps,
                     const int count_lagr_mult_updates,
                     const T   norm_pen,
                     const T   norm_rpen,
                     const T   penetration_tol,
                     const T   penalty_param) {
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
        }

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        std::vector<std::shared_ptr<Operator<T>>>               operator_;
        std::vector<std::shared_ptr<MatrixFreeLinearSolver<T>>> smoother_;

        std::vector<std::shared_ptr<Operator<T>>> prolongation_;
        std::vector<std::shared_ptr<Operator<T>>> restriction_;

        std::shared_ptr<Operator<T>>                       constraints_op_;
        std::shared_ptr<Operator<T>>                       constraints_op_transpose_;
        std::vector<std::shared_ptr<SparseBlockVector<T>>> constraints_op_x_op_;
        std::vector<std::shared_ptr<Operator<T>>>          constraints_restriction_;
        std::function<void(const T* const)>                update_constraints_;

        SharedBuffer<T> upper_bound_;
        SharedBuffer<T> lower_bound_;
        SharedBuffer<T> correction, lagr_lb, lagr_ub;

        // Internals
        std::vector<std::shared_ptr<Memory>> memory_;
        bool                                 wrap_input_{true};

        bool collect_energy_norm_correction_{false};
        bool enable_line_search_{false};
        bool project_coarse_space_correction_{false};
        bool skip_coarse{false};
        bool verbose{true};

        int count_smoothing_steps{0};
        int max_inner_it{3};
        int nlsmooth_steps{3};
        int cycle_type_{V_CYCLE};
        int iterations_{0};
        int max_it_{10};

        T    max_penalty_param_{10000};
        T    penalty_param_{10};
        T    penalty_param_increase{10};
        T    atol_{1e-10};
        T    penetration_tol_exp{0.9};
        bool enable_shift{true};

        int                      debug{0};
        std::vector<struct Stats> stats;

        BLAS_Tpl<T>           blas_;
        ShiftedPenalty_Tpl<T> impl_;
    };

    template <typename T>
    std::shared_ptr<ShiftedPenaltyMultigrid<T>> h_spmg() {
        auto mg = std::make_shared<ShiftedPenaltyMultigrid<T>>();
        mg->default_init();
        return mg;
    }

}  // namespace sfem

#endif  // SFEM_SHIFTED_PENALTY_MULTIGRID_HPP
