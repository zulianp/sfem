#ifndef SFEM_CHEB3_HPP
#define SFEM_CHEB3_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>

#include "sfem_PowerMethod.hpp"
#include "sfem_openmp_blas.hpp"

// https://en.wikipedia.org/wiki/Conjugate_gradient_method
namespace sfem {
    template <typename T>
    class Chebyshev3 : public MatrixFreeLinearSolver<T> {
    public:
        // Operator
        std::function<void(const T* const, T* const)> apply_op;
        std::function<void(const T* const, T* const)> preconditioner_op;

        BLAS_Tpl<T> blas;
        std::shared_ptr<PowerMethod<T>> power_method;

        std::shared_ptr<Buffer<T>> p_, temp_;

        // Solver parameters
        T atol{1e-10};
        T rtol{1e-10};
        T eigen_solver_tol{1e-6};
        int eigen_solver_max_it{1000};
        int max_it{3};

        T eig_max{0};
        T scale_eig_max{1};
        T scale_eig_min{0.06};
        ptrdiff_t n_dofs{-1};
        bool is_initial_guess_zero{false};
        bool verbose{true};

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        void set_atol(const T val) { atol = val; }
        void set_rtol(const T val) { rtol = val; }

        void set_verbose(const bool val) { verbose = val; }

        ExecutionSpace execution_space() const override { return execution_space_; }

        void set_initial_guess_zero(const bool val) override { is_initial_guess_zero = val; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            n_dofs = op->rows();
            this->apply_op = [=](const T* const x, T* const y) { op->apply(x, y); };
        }

        void set_preconditioner_op(const std::shared_ptr<Operator<T>>&) override { assert(false); }

        void set_preconditioner(std::function<void(const T* const, T* const)>&& in) {
            preconditioner_op = in;
        }

        void default_init() {
            OpenMP_BLAS<T>::build_blas(this->blas);
            ensure_power_method();
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void ensure_power_method() {
            if (!power_method) {
                power_method = std::make_shared<PowerMethod<T>>();
                power_method->norm2 = this->blas.norm2;
                power_method->scal = this->blas.scal;
                power_method->zeros = this->blas.zeros;
            }
        }

        bool good() const {
            return blas.good() && apply_op;
        }

        void monitor(const int iter, const T residual) {
            if (iter == max_it || iter % 100 == 0 || residual < atol) {
                std::cout << iter << ": " << residual << "\n";
            }
        }

        T max_eigen_value(T* const guess_eigenvector, T* const work) {
            assert(power_method);
            return power_method->max_eigen_value(
                    apply_op, eigen_solver_max_it, this->eigen_solver_tol, this->rows(), guess_eigenvector, work);
        }

        void init_with_ones() {
            T* work = blas.allocate(this->rows());
            auto ones = Buffer<T>::own(this->rows(), work, blas.destroy);
            this->blas.values(n_dofs, 1, ones->data());
            init(ones->data());
        }

        void init_with_random() 
        {
            T* work = blas.allocate(this->rows());
            auto random_vector = Buffer<T>::own(this->rows(), work, blas.destroy);
            assert(execution_space_ == EXECUTION_SPACE_HOST);

            auto v = random_vector->data();
            for(ptrdiff_t i = 0; i < this->rows(); i++) {
                v[i] = -0.5 + rand() * 1.0/RAND_MAX;;
            }   

            init(random_vector->data());
        }

        void init(const T* const guess_eigenvector) {
            T* eigenvector = blas.allocate(this->rows());
            T* work = blas.allocate(this->rows());
            blas.copy(this->rows(), guess_eigenvector, eigenvector);

            eig_max = max_eigen_value(eigenvector, work);

            // destroy(eigenvector);  // Maybe we want to keep this around?
            // destroy(work);

            p_ = Buffer<T>::own(this->rows(), work, blas.destroy);
            temp_ = Buffer<T>::own(this->rows(), eigenvector, blas.destroy);
        }

        int apply(const T* const b, T* const x) override {
            precond_apply(b, x, p_->data(), temp_->data());
            return 0;
        }

        int precond_apply(const T* const rhs,
                          T* const x,
                          // work-buffers
                          T* const p,
                          T* const temp) {
            if (!good()) {
                return -1;
            }

            const ptrdiff_t n = this->rows();

            const T eig_max = this->eig_max * scale_eig_max;
            const T eig_min = scale_eig_min * eig_max;
            const T eig_avg = (eig_min + eig_max) / 2;
            const T eig_diff = (eig_min - eig_max) / 2;

            // Iteration 0

            // Params
            T alpha = 1 / eig_avg;
            T beta = 0;
            T dea = 0;

            // Vectors
            if (is_initial_guess_zero) {
                blas.copy(n, rhs, p);
                blas.scal(n, -1, p);
            } else {
                blas.zeros(n, p);
                apply_op(x, p);
                blas.axpy(n, -1, rhs, p);
            }

            blas.axpy(n, -alpha, p, x);

            // Iteration 1
            // Params
            dea = eig_diff * alpha;
            beta = 0.5 * dea * dea;
            alpha = 1 / (eig_avg - (beta / alpha));

            // Vectors
            blas.axpby(n, -1, rhs, beta, p);

            if (temp) {
                blas.zeros(n, temp);
                apply_op(x, temp);
                blas.axpy(n, 1, temp, p);
            } else {
                // This can only be used if boundary conditions are
                // already satified or applied with a matrix
                apply_op(x, p);
            }
            blas.axpy(n, -alpha, p, x);

            // Iteration i>=2
            for (int i = 2; i < max_it; i++) {
                dea = eig_diff * alpha;
                beta = 0.25 * dea * dea;
                alpha = 1 / (eig_avg - (beta / alpha));

                blas.axpby(n, -1, rhs, beta, p);

                if (temp) {
                    blas.zeros(n, temp);
                    apply_op(x, temp);
                    blas.axpy(n, 1, temp, p);
                } else {
                    // This can only be used if boundary conditions are
                    // already satified or applied with a matrix
                    apply_op(x, p);
                }

                blas.axpy(n, -alpha, p, x);
            }

            return 0;
        }

        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }
        void set_max_it(const int it) override { max_it = it; }
        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }
    };

    template <typename T>
    std::shared_ptr<Chebyshev3<T>> h_cheb3(const std::shared_ptr<Operator<T>>& op) {
        auto ret = std::make_shared<Chebyshev3<T>>();
        ret->n_dofs = op->rows();
        ret->set_op(op);
        ret->default_init();
        return ret;
    }
}  // namespace sfem

#endif  // SFEM_CHEB3_HPP
