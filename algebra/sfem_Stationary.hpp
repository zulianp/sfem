#ifndef SFEM_STATIONARY_HPP
#define SFEM_STATIONARY_HPP

#include <cstddef>
#include <functional>
#include <memory>

#include "sfem_Buffer.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_openmp_blas.hpp"

namespace sfem {
    template <typename T>
    class StationaryIteration final : public MatrixFreeLinearSolver<T> {
    public:
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        ptrdiff_t n_dofs{-1};
        int max_it{3};
        Buffer<T> workspace;

        std::function<void(const T* const, T* const)> apply_op;
        std::function<void(const T* const, T* const)> preconditioner_op;
        BLAS_Tpl<T> blas;

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
            auto x = this->blas.allocate(n_dofs);
            workspace = Buffer<T>::own(n_dofs, x, this->blas.destroy);
        }

        /* Operator */
        int apply(const T* const b, T* const x) override {
            T* r = workspace.data();
            for (int i = 0; i < max_it; i++) {
                apply_op(x, r);
                blas.axpby(n_dofs, 1, b, -1, r);
                preconditioner_op(r, x);
            }
            return 0;
        }
        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }
        ExecutionSpace execution_space() const override { return execution_space_; }

        /* MatrixFreeLinearSolver */
        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->apply_op = [=](const T* const x, T* const y) { op->apply(x, y); };
            n_dofs = op->rows();
        }
        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override {
            this->preconditioner_op = [=](const T* const x, T* const y) { op->apply(x, y); };
        }
        void set_max_it(const int it) override { max_it = it; }
        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }
    };

    template <typename T>
    std::shared_ptr<StationaryIteration<T>> h_stationary() {
        auto solver = std::make_shared<StationaryIteration<T>>();
        solver->default_init();
        return solver;
    }
}  // namespace sfem

#endif  // SFEM_STATIONARY_HPP
