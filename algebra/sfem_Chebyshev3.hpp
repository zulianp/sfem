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

#ifdef _OPENMP
#include <omp.h>
#endif

// 3-term Chebyshev iteration (smoother / polynomial preconditioner)
namespace sfem {

    namespace chebyshev3_host_ {

        // y ← alpha * x
        template <typename T>
        static void scaled_copy(const ptrdiff_t n, const T alpha, const T* const x, T* const y) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (ptrdiff_t i = 0; i < n; i++) {
                y[i] = alpha * x[i];
            }
        }

        // p ← β p − rhs + Ax;  x ← x − α p
        template <typename T>
        static void iteration_update(const ptrdiff_t n,
                                     const T         alpha,
                                     const T         beta,
                                     const T* const  rhs,
                                     const T* const  Ax,
                                     T* const        p,
                                     T* const        x) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (ptrdiff_t i = 0; i < n; i++) {
                const T pi = beta * p[i] - rhs[i] + Ax[i];
                p[i]       = pi;
                x[i] -= alpha * pi;
            }
        }

    }  // namespace chebyshev3_host_

    template <typename T>
    class Chebyshev3 : public MatrixFreeLinearSolver<T> {
    public:
        // Operator
        std::function<void(const T* const, T* const)> apply_op;
        std::function<void(const T* const, T* const)> preconditioner_op;

        std::shared_ptr<BLAS<T>>        blas;
        std::shared_ptr<PowerMethod<T>> power_method;

        SharedBuffer<T> p_, temp_;

        // Solver parameters
        T   atol{1e-10};
        T   rtol{1e-10};
        T   eigen_solver_tol{1e-6};
        int eigen_solver_max_it{1000};
        int max_it{3};
        int iterations_{0};

        T         eig_max{0};
        T         scale_eig_max{1};
        T         scale_eig_min{0.06};
        ptrdiff_t n_dofs{SFEM_PTRDIFF_INVALID};
        bool      is_initial_guess_zero{false};
        bool      verbose{true};
        // If true, apply_op overwrites its output (e.g. BSR SpMV with scale_output==0).
        // Default false: most ops accumulate (y += Ax); zeros() before each apply_op.
        bool apply_overwrites_output{false};

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        int iterations() const override { return iterations_; }

        void set_atol(const T val) { atol = val; }
        void set_rtol(const T val) { rtol = val; }

        void set_verbose(const bool val) { verbose = val; }
        void set_apply_overwrites_output(const bool val) { apply_overwrites_output = val; }

        ExecutionSpace execution_space() const override { return execution_space_; }

        void set_initial_guess_zero(const bool val) override { is_initial_guess_zero = val; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            n_dofs         = op->rows();
            this->apply_op = [=](const T* const x, T* const y) { op->apply(x, y); };
        }

        void set_preconditioner_op(const std::shared_ptr<Operator<T>>&) override { assert(false); }

        void set_preconditioner(std::function<void(const T* const, T* const)>&& in) { preconditioner_op = in; }

        void default_init() {
            this->blas = make_openmp_blas<T>();
            ensure_power_method();
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void ensure_power_method() {
            if (!power_method) {
                auto blas_impl      = this->blas;
                power_method        = std::make_shared<PowerMethod<T>>();
                power_method->norm2 = [blas_impl](const std::ptrdiff_t n, const T* const x) { return blas_impl->norm2(n, x); };
                power_method->scal  = [blas_impl](const std::ptrdiff_t n, const T alpha, T* const x) {
                    blas_impl->scal(n, alpha, x);
                };
                power_method->zeros = [blas_impl](const std::size_t n, T* const x) { blas_impl->zeros(n, x); };
            }
        }

        bool good() const { return blas->good() && apply_op; }

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
            SFEM_TRACE_SCOPE("Chebyshev3::init_with_ones");
            T*   work      = blas->allocate(this->rows());
            auto blas_impl = blas;
            auto ones      = Buffer<T>::own(this->rows(), work, [blas_impl](void* ptr) { blas_impl->destroy(ptr); });
            this->blas->values(n_dofs, 1, ones->data());
            init(ones->data());
        }

        void init_with_random() {
            SFEM_TRACE_SCOPE("Chebyshev3::init_with_random");
            T*   work          = blas->allocate(this->rows());
            auto blas_impl     = blas;
            auto random_vector = Buffer<T>::own(this->rows(), work, [blas_impl](void* ptr) { blas_impl->destroy(ptr); });
            assert(execution_space_ == EXECUTION_SPACE_HOST);

            auto v = random_vector->data();
            for (ptrdiff_t i = 0; i < this->rows(); i++) {
                v[i] = -0.5 + rand() * 1.0 / RAND_MAX;
            }

            init(random_vector->data());
        }

        void init(const T* const guess_eigenvector) {
            T* eigenvector = blas->allocate(this->rows());
            T* work        = blas->allocate(this->rows());
            blas->copy(this->rows(), guess_eigenvector, eigenvector);

            eig_max = max_eigen_value(eigenvector, work);

            auto blas_impl = blas;
            p_             = Buffer<T>::own(this->rows(), work, [blas_impl](void* ptr) { blas_impl->destroy(ptr); });
            temp_          = Buffer<T>::own(this->rows(), eigenvector, [blas_impl](void* ptr) { blas_impl->destroy(ptr); });
        }

        int apply(const T* const b, T* const x) override {
            SFEM_TRACE_SCOPE("Chebyshev3::apply");
            precond_apply(b, x, p_->data(), temp_->data());
            return 0;
        }

        int precond_apply(const T* const rhs,
                          T* const       x,
                          // work-buffers
                          T* const p,
                          T* const temp) {
            if (!good()) {
                return SFEM_FAILURE;
            }

            if (max_it <= 0) {
                iterations_ = 0;
                return 0;
            }

            const ptrdiff_t n = this->rows();

            const T eig_max  = this->eig_max * scale_eig_max;
            const T eig_min  = scale_eig_min * eig_max;
            const T eig_avg  = (eig_min + eig_max) / 2;
            const T eig_diff = (eig_min - eig_max) / 2;

            T alpha = T(1) / eig_avg;
            T beta  = 0;
            T dea   = 0;

            // Iteration 0: p = A x - rhs (or p = -rhs if x==0), then x -= alpha p
            if (is_initial_guess_zero) {
                scaled_copy(n, T(-1), rhs, p);
            } else {
                ensure_apply_buffer(n, p);
                apply_op(x, p);
                blas->axpy(n, T(-1), rhs, p);
            }

            blas->axpy(n, -alpha, p, x);
            iterations_ = 1;
            if (max_it == 1) {
                return 0;
            }

            // Iteration 1
            dea   = eig_diff * alpha;
            beta  = T(0.5) * dea * dea;
            alpha = T(1) / (eig_avg - (beta / alpha));
            iteration_step(n, alpha, beta, rhs, x, p, temp);
            iterations_ = 2;
            if (max_it == 2) {
                return 0;
            }

            // Iteration i >= 2
            for (; iterations_ < max_it; iterations_++) {
                dea   = eig_diff * alpha;
                beta  = T(0.25) * dea * dea;
                alpha = T(1) / (eig_avg - (beta / alpha));
                iteration_step(n, alpha, beta, rhs, x, p, temp);
            }

            return 0;
        }

        void                  set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }
        void                  set_max_it(const int it) override { max_it = it; }
        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }

    private:
        bool use_host_kernels() const { return execution_space_ == EXECUTION_SPACE_HOST; }

        void ensure_apply_buffer(const ptrdiff_t n, T* const y) const {
            if (!apply_overwrites_output) {
                blas->zeros(n, y);
            }
        }

        void scaled_copy(const ptrdiff_t n, const T alpha, const T* const x, T* const y) const {
            if (use_host_kernels()) {
                chebyshev3_host_::scaled_copy(n, alpha, x, y);
            } else {
                // Device / generic BLAS fallback
                blas->zaxpby(n, alpha, x, T(0), x, y);
            }
        }

        void iteration_update(const ptrdiff_t n,
                              const T         alpha,
                              const T         beta,
                              const T* const  rhs,
                              const T* const  Ax,
                              T* const        p,
                              T* const        x) const {
            if (use_host_kernels()) {
                chebyshev3_host_::iteration_update(n, alpha, beta, rhs, Ax, p, x);
            } else {
                blas->axpby(n, T(-1), rhs, beta, p);
                blas->axpy(n, T(1), Ax, p);
                blas->axpy(n, -alpha, p, x);
            }
        }

        void iteration_step(const ptrdiff_t n,
                            const T         alpha,
                            const T         beta,
                            const T* const  rhs,
                            T* const        x,
                            T* const        p,
                            T* const        temp) const {
            if (temp) {
                // SpMV first (independent of p), then vector update:
                //   p ← β p − rhs + A x;  x ← x − α p
                ensure_apply_buffer(n, temp);
                apply_op(x, temp);
                iteration_update(n, alpha, beta, rhs, temp, p, x);
            } else {
                // Legacy in-place path (apply overwrites p).
                blas->axpby(n, T(-1), rhs, beta, p);
                apply_op(x, p);
                blas->axpy(n, -alpha, p, x);
            }
        }
    };

    template <typename T>
    std::shared_ptr<Chebyshev3<T>> h_cheb3(const std::shared_ptr<Operator<T>>& op) {
        auto ret    = std::make_shared<Chebyshev3<T>>();
        ret->n_dofs = op->rows();
        ret->set_op(op);
        ret->default_init();
        return ret;
    }
}  // namespace sfem

#endif  // SFEM_CHEB3_HPP
