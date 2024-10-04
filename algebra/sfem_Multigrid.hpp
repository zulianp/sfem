#ifndef SFEM_MULTIGRID_HPP
#define SFEM_MULTIGRID_HPP

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "sfem_MatrixFreeLinearSolver.hpp"

#include "sfem_Buffer.hpp"

// https://en.wikipedia.org/wiki/Conjugate_gradient_method
namespace sfem {

    /// level 0 is the finest
    template <typename T>
    class Multigrid final : public Operator<T> {
    public:
        std::function<T*(const std::size_t)> allocate;
        std::function<void(const std::size_t, T* const x)> zeros;
        std::function<void(void*)> destroy;
        // std::function<void(const ptrdiff_t, const T* const, T* const)> copy;
        std::function<void(const ptrdiff_t, const T, const T* const, const T, T* const)> axpby;
        std::function<T(const std::size_t, const T* const)> norm2;
        bool verbose{true};
        bool debug{false};

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

            for (iterations_ = 0; iterations_ < max_it_; iterations_++) {
                CycleReturnCode ret = cycle(finest_level());
                if (ret == CYCLE_CONVERGED) {
                    break;
                }
            }

            return 0;
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
                              const std::shared_ptr<Operator<T>>& smoother_or_solver,
                              const std::shared_ptr<Operator<T>>& prolongation,
                              const std::shared_ptr<Operator<T>>& restriction) {
            operator_.push_back(op);
            smoother_.push_back(smoother_or_solver);
            prolongation_.push_back(prolongation);
            restriction_.push_back(restriction);
        }

        void default_init() {
            allocate = [](const std::ptrdiff_t n) -> T* { return (T*)calloc(n, sizeof(T)); };

            destroy = [](void* a) { free(a); };

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

            zeros = [](const std::size_t n, T* const x) { memset(x, 0, n * sizeof(T)); };
            norm2 = [](const std::size_t n, const T* const x) -> T {
                T ret = 0;

#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n; i++) {
                    ret += x[i] * x[i];
                }

                return sqrt(ret);
            };

            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void set_max_it(const int val) { max_it_ = val; }

        void set_atol(const T val) { atol_ = val; }

    private:
        std::vector<std::shared_ptr<Operator<T>>> operator_;
        std::vector<std::shared_ptr<Operator<T>>> smoother_;

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
                    auto x = this->allocate(n);
                    memory_[l]->solution = Buffer<T>::own(n, x, this->destroy);

                    auto r = this->allocate(n);
                    memory_[l]->rhs = Buffer<T>::own(n, r, this->destroy);
                }

                auto w = this->allocate(n);
                memory_[l]->work = Buffer<T>::own(n, w, this->destroy);
            }

            return 0;
        }

        CycleReturnCode cycle(const int level) {
            auto mem = memory_[level];
            auto smoother = smoother_[level];

            if (coarsest_level() == level) {
                this->zeros(mem->solution->size(), mem->solution->data());
                if (!smoother->apply(mem->rhs->data(), mem->solution->data())) {
                    return CYCLE_CONTINUE;
                } else {
                    return CYCLE_FAILURE;
                }
            }

            auto op = operator_[level];
            auto restriction = restriction_[level];
            auto prolongation = prolongation_[coarser_level(level)];
            auto mem_coarse = memory_[coarser_level(level)];

            for (int k = 0; k < this->cycle_type_; k++) {
                smoother->apply(mem->rhs->data(), mem->solution->data());

                {
                    // Compute residual
                    this->zeros(mem->size(), mem->work->data());
                    op->apply(mem->solution->data(), mem->work->data());
                    this->axpby(mem->size(), 1, mem->rhs->data(), -1, mem->work->data());

                    if (finest_level() == level) {
                        T norm_residual = this->norm2(mem->work->size(), mem->work->data());

                        if (iterations_ == 0) {
                            norm_residual_0 = norm_residual;
                            norm_residual_previous = norm_residual;

                            if (verbose) {
                                printf("Multigrid\n");
                                printf("iter\tabs\t\trel\t\trate\n");
                                printf("%d\t%g\t-\t\t-\n", iterations_, (double)(norm_residual));
                            }
                        } else {
                            if (verbose) {
                                printf("%d\t%g\t%g\t%g\n",
                                       iterations_,
                                       (double)(norm_residual),
                                       (double)(norm_residual / norm_residual_0),
                                       (double)(norm_residual / norm_residual_previous));

                                fflush(stderr);
                                fflush(stdout);
                            }
                        }

                        norm_residual_previous = norm_residual;
                        if (norm_residual < atol_) {
                            return CYCLE_CONVERGED;
                        }
                    }
                }

                {
                    // Restriction
                    this->zeros(mem_coarse->rhs->size(), mem_coarse->rhs->data());
                    restriction->apply(mem->work->data(), mem_coarse->rhs->data());
                }

                CycleReturnCode ret = cycle(coarser_level(level));
                assert(ret != CYCLE_FAILURE);

                {
                    if (debug) {
                        printf("|| c_H || = %g\n",
                               (double)this->norm2(mem_coarse->solution->size(),
                                                   mem_coarse->solution->data()));
                    }

                    // Prolongation
                    this->zeros(mem->work->size(), mem->work->data());
                    prolongation->apply(mem_coarse->solution->data(), mem->work->data());

                    if (debug) {
                        printf("|| c_h || = %g\n",
                               (double)this->norm2(mem->work->size(), mem->work->data()));
                    }

                    // Apply coarse space correction
                    this->axpby(mem->size(), 1, mem->work->data(), 1, mem->solution->data());
                }


                if(debug) {
                    this->zeros(mem->size(), mem->work->data());
                    op->apply(mem->solution->data(), mem->work->data());
                    this->axpby(mem->size(), 1, mem->rhs->data(), -1, mem->work->data());
                    printf("|| r_h || = %g\n", this->norm2(mem->work->size(), mem->work->data()));
                }

                smoother->apply(mem->rhs->data(), mem->solution->data());
            }

            return CYCLE_CONTINUE;
        }
    };

    template <typename T>
    std::shared_ptr<Multigrid<T>> h_mg() {
        auto mg = std::make_shared<Multigrid<T>>();
        mg->default_init();
        return mg;
    }

}  // namespace sfem

#endif  // SFEM_MULTIGRID_HPP
