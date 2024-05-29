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

        enum CycleType {
            V_CYCLE = 1,
            W_CYCLE = 2,
        };

        class Memory {
        public:
            std::shared_ptr<Buffer<T>> solution;
            std::shared_ptr<Buffer<T>> residual;
            std::shared_ptr<Buffer<T>> work;

            inline ptrdiff_t size() const { return solution->size(); }
            ~Memory() {}
        };

        int apply(const T* const r, T* const x) override {
            ensure_init();

            // Wrap input arrays into fine level of mg
            // memory_[finest_level()]->solution =
            //     Buffer<T>::wrap(smoother_[finest_level()]->rows(), x);

            // memory_[finest_level()]->residual =
            //     Buffer<T>::wrap(smoother_[finest_level()]->rows(), (T*)r);

            for (int k = 0; k < max_it_; k++) {
                // std::cout << "iteration: " << k << ")\n";
                // operator_->apply(r, )

                cycle(finest_level());

                auto c = memory_[finest_level()]->solution
                axpby(c->size(), 1, c->data(), 1, x);


            }

            return 0;
        }

        // void set_coarse_grid_solver(const std::shared_ptr<Operator<T>>& op)
        // {
        //     coarse_grid_solver_ = op;
        // }

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

            axpby =
                [](const ptrdiff_t n, const T alpha, const T* const x, const T beta, T* const y) {
#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n; i++) {
                        y[i] = alpha * x[i] + beta * y[i];
                    }
                };

            zeros = [](const std::size_t n, T* const x) { memset(x, 0, n * sizeof(T)); };
        }

    private:
        std::vector<std::shared_ptr<Operator<T>>> operator_;
        std::vector<std::shared_ptr<Operator<T>>> smoother_;

        std::vector<std::shared_ptr<Operator<T>>> prolongation_;
        std::vector<std::shared_ptr<Operator<T>>> restriction_;

        // Internals
        std::vector<std::shared_ptr<Memory>> memory_;

        int max_it_{1};
        int cycle_type_{V_CYCLE};

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
                // if (l != finest_level()) 
                {
                    auto x = this->allocate(n);
                    memory_[l]->solution = Buffer<T>::own(n, x, this->destroy);

                    auto r = this->allocate(n);
                    memory_[l]->residual = Buffer<T>::own(n, r, this->destroy);
                }

                auto w = this->allocate(n);
                memory_[l]->work = Buffer<T>::own(n, w, this->destroy);
            }

            return 0;
        }

        int cycle(const int level) {
            auto mem = memory_[level];

            // As we are solving for the correction we start from 0
            this->zeros(mem->size(), mem->solution->data());

            if (coarsest_level() == level) {
                // std::cout << "Coarse level solve!\n";
                return smoother_[level]->apply(mem->residual->data(), mem->solution->data());
            }

            auto op = operator_[level];
            auto restriction = restriction_[level];
            auto prolongation = prolongation_[coarser_level(level)];

            for (int k = 0; k < this->cycle_type_; k++) {
                // std::cout << "Cycle " << k << " \n";

                this->zeros(solution->size(), mem->solution->data());
                smoother_[level]->apply(mem->residual->data(), mem->solution->data());

                this->zeros(mem->size(), mem->work->data());
                op->apply(mem->solution->data(), mem->work->data());

                this->axpby(mem->size(), 1, mem->residual->data(), -1, mem->work->data());
                restriction->apply(mem->work->data(),
                                   memory_[coarser_level(level)]->residual->data());

                int err = cycle(coarser_level(level));
                assert(!err);

                prolongation->apply(memory_[coarser_level(level)]->solution->data(),
                                    mem->work->data());

                // Apply correction
                this->axpby(mem->size(), 1, mem->work->data(), 1, mem->solution->data());
                smoother_[level]->apply(mem->residual->data(), mem->solution->data());
            }
            
            return 0;
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
