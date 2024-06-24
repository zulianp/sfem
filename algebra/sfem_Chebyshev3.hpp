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

// https://en.wikipedia.org/wiki/Conjugate_gradient_method
namespace sfem {
    template <typename T>
    class Chebyshev3 : public MatrixFreeLinearSolver<T> {
    public:
        // Operator
        std::function<void(const T* const, T* const)> apply_op;
        std::function<void(const T* const, T* const)> preconditioner_op;

        // Mem management
        std::function<T*(const std::size_t)> allocate;
        std::function<void(void*)> destroy;

        std::function<void(const ptrdiff_t, const T* const, T* const)> copy;
        std::function<void(const std::size_t, T* const)> zeros;

        // blas
        std::function<T(const ptrdiff_t, const T* const, const T* const)> dot;
        std::function<T(const ptrdiff_t, const T* const)> norm2;
        std::function<void(const ptrdiff_t, const T, const T* const, T* const)> axpy;
        std::function<void(const ptrdiff_t, const T, const T* const, const T, T* const)> axpby;
        std::function<void(const std::ptrdiff_t, const T, T* const)> scal;
        std::shared_ptr<PowerMethod<T>> power_method;

        std::shared_ptr<Buffer<T>> p_, temp_;

        // Solver parameters
        T tol{1e-10};
        int max_it{3};

        T eig_max{0};
        T scale_eig_max{1};
        T scale_eig_min{0.06};
        ptrdiff_t n_dofs{-1};
        bool is_initial_guess_zero{false};

        void set_initial_guess_zero(const bool val) override { is_initial_guess_zero = val; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->apply_op = [=](const T* const x, T* const y) { op->apply(x, y); };
        }

        void set_preconditioner_op(const std::shared_ptr<Operator<T>>&) override { assert(false); }

        void set_preconditioner(std::function<void(const T* const, T* const)>&& in) {
            preconditioner_op = in;
        }

        void default_init() {
            allocate = [](const std::ptrdiff_t n) -> T* { return (T*)calloc(n, sizeof(T)); };

            destroy = [](void* a) { free(a); };

            copy = [](const ptrdiff_t n, const T* const src, T* const dest) {
                std::memcpy(dest, src, n * sizeof(T));
            };

            dot = [](const ptrdiff_t n, const T* const l, const T* const r) -> T {
                T ret = 0;

#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n; i++) {
                    ret += l[i] * r[i];
                }

                return ret;
            };

            norm2 = [](const ptrdiff_t n, const T* const x) -> T {
                T ret = 0;

#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n; i++) {
                    ret += x[i] * x[i];
                }

                return sqrt(ret);
            };

            axpy = [](const ptrdiff_t n, const T alpha, const T* const x, T* const y) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    y[i] = alpha * x[i] + y[i];
                }
            };

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

            zeros = [](const std::ptrdiff_t n, T* const x) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    x[i] = 0;
                }
            };

            scal = [](const std::ptrdiff_t n, const T alpha, T* const x) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    x[i] *= alpha;
                }
            };

            ensure_power_method();
        }

        void ensure_power_method() {
            if (!power_method) {
                power_method = std::make_shared<PowerMethod<T>>();
                power_method->norm2 = norm2;
                power_method->scal = scal;
                power_method->zeros = zeros;
            }
        }

        bool good() const {
            assert(allocate);
            assert(destroy);
            assert(copy);
            assert(dot);
            assert(norm2);
            assert(axpby);
            assert(apply_op);
            assert(eig_max != 0);

            return eig_max != 0 && allocate && destroy && copy && dot && norm2 && axpby && apply_op;
        }

        void monitor(const int iter, const T residual) {
            if (iter == max_it || iter % 100 == 0 || residual < tol) {
                std::cout << iter << ": " << residual << "\n";
            }
        }

        T max_eigen_value(T* const guess_eigenvector, T* const work) {
            assert(power_method);
            return power_method->max_eigen_value(
                    apply_op, 1000, 1e-6, this->rows(), guess_eigenvector, work);
        }

        void init(const T* const guess_eigenvector) {
            T* eigenvector = allocate(this->rows());
            T* work = allocate(this->rows());
            copy(this->rows(), guess_eigenvector, eigenvector);

            eig_max = max_eigen_value(eigenvector, work);

            // destroy(eigenvector);  // Maybe we want to keep this around?
            // destroy(work);

            p_ = Buffer<T>::own(this->rows(), work, destroy);
            temp_ = Buffer<T>::own(this->rows(), eigenvector, destroy);
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
                copy(n, rhs, p);
                scal(n, -1, p);
            } else {
                zeros(n, p);
                apply_op(x, p);
                axpy(n, -1, rhs, p);
            }

            axpy(n, -alpha, p, x);

            // Iteration 1
            // Params
            dea = eig_diff * alpha;
            beta = 0.5 * dea * dea;
            alpha = 1 / (eig_avg - (beta / alpha));

            // Vectors
            axpby(n, -1, rhs, beta, p);

            if (temp) {
                zeros(n, temp);
                apply_op(x, temp);
                axpy(n, 1, temp, p);
            } else {
                // This can only be used if boundary conditions are
                // already satified or applied with a matrix
                apply_op(x, p);
            }
            axpy(n, -alpha, p, x);

            // Iteration i>=2
            for (int i = 2; i < max_it; i++) {
                dea = eig_diff * alpha;
                beta = 0.25 * dea * dea;
                alpha = 1 / (eig_avg - (beta / alpha));

                axpby(n, -1, rhs, beta, p);

                if (temp) {
                    zeros(n, temp);
                    apply_op(x, temp);
                    axpy(n, 1, temp, p);
                } else {
                    // This can only be used if boundary conditions are
                    // already satified or applied with a matrix
                    apply_op(x, p);
                }

                axpy(n, -alpha, p, x);
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
