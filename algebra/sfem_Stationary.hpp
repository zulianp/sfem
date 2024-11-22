#ifndef SFEM_STATIONARY_HPP
#define SFEM_STATIONARY_HPP

#include <cstddef>
#include <functional>
#include <memory>

#include "sfem_Buffer.hpp"
#include "sfem_LpSmoother.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_openmp_blas.hpp"

namespace sfem {
    template <typename T>
    class StationaryIteration final : public MatrixFreeLinearSolver<T> {
    public:
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        ptrdiff_t n_dofs{-1};
        int max_it{3};
        std::shared_ptr<Buffer<T>> workspace;
        std::shared_ptr<Operator<T>> op;
        std::shared_ptr<Operator<T>> preconditioner;
        bool verbose{false};
        BLAS_Tpl<T> blas;

        int iterations_{0};

        int iterations() const override {
            return iterations_;
        }

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
            auto x = this->blas.allocate(n_dofs);
            workspace = Buffer<T>::own(n_dofs, x, this->blas.destroy);
        }

        /* Operator */
        int apply(const T* const b, T* const x) override {
            T* r = workspace->data();
            for (iterations_ = 0; iterations_ < max_it; iterations_++) {
                blas.zeros(workspace->size(), r);
                op->apply(x, r);
                blas.axpby(n_dofs, 1.0, b, -1.0, r);
                if (verbose) {
                    T norm_residual = this->blas.norm2(workspace->size(), r);
                    printf("%d : %f\n", iterations_, (double)norm_residual);
                }
                preconditioner->apply(r, x);
            }
            return 0;
        }
        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }
        ExecutionSpace execution_space() const override { return execution_space_; }

        /* MatrixFreeLinearSolver */
        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->op = op;
            n_dofs = op->rows();
        }
        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override {
            this->preconditioner = op;
        }
        void set_max_it(const int it) override { max_it = it; }
        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }
    };

    template <typename T>
    std::shared_ptr<StationaryIteration<T>> h_stationary(
            const std::shared_ptr<Operator<T>>& op,
            const std::shared_ptr<Operator<T>>& preconditioner) {
        auto solver = std::make_shared<StationaryIteration<T>>();
        solver->op = op;
        solver->preconditioner = preconditioner;
        solver->n_dofs = op->cols();
        solver->default_init();
        return solver;
    }
}  // namespace sfem

#endif  // SFEM_STATIONARY_HPP
