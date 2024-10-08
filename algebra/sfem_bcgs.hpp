#ifndef SFEM_BCGS_HPP
#define SFEM_BCGS_HPP

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>

#include "sfem_MatrixFreeLinearSolver.hpp"

// https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
namespace sfem {
    template <typename T>
    class BiCGStab final : public MatrixFreeLinearSolver<T> {
    public:
        // Operator
        std::function<void(const T* const, T* const)> apply_op;
        std::function<void(const T* const, T* const)> left_preconditioner_op;
        std::function<void(const T* const, T* const)> right_preconditioner_op;

        // Mem management
        std::function<T*(const std::size_t)> allocate;
        std::function<void( void*)> destroy;

        std::function<void(const std::size_t, T* const x)> zeros;

        std::function<void(const ptrdiff_t, const T* const, T* const)> copy;

        // blas
        std::function<T(const ptrdiff_t, const T* const, const T* const)> dot;
        std::function<void(const ptrdiff_t, const T, const T* const, const T, T* const)> axpby;
        std::function<
            void(const ptrdiff_t, const T, const T* const, const T, const T* const, T* const)>
            zaxpby;

        ptrdiff_t n_dofs{-1};

        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};

        ExecutionSpace execution_space() const override { return execution_space_; }

        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->apply_op = [=](const T* const x, T* const y) { op->apply(x, y); };
        }

        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override {
            this->right_preconditioner_op = [=](const T* const x, T* const y) { op->apply(x, y); };
        }

        void set_max_it(const int it) override { max_it = it; }

        void set_preconditioner(std::function<void(const T* const, T* const)>&& in) {
            // left_preconditioner_op = in;
            right_preconditioner_op = in;
        }

        // Solver parameters
        T tol{1e-10};
        int max_it{10000};

        void default_init() {
            allocate = [](const ptrdiff_t n) -> T* { return (T*)calloc(n, sizeof(T)); };

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

            axpby =
                [](const ptrdiff_t n, const T alpha, const T* const x, const T beta, T* const y) {
#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n; i++) {
                        y[i] = alpha * x[i] + beta * y[i];
                    }
                };

            zaxpby = [](const ptrdiff_t n,
                        const T alpha,
                        const T* const x,
                        const T beta,
                        const T* const y,
                        T* const z) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    z[i] = alpha * x[i] + beta * y[i];
                }
            };

            zeros = [](const std::size_t n, T* const x) {
                memset(x, 0, n*sizeof(T));
            };

            execution_space_ = EXECUTION_SPACE_HOST;
        }

        bool good() const {
            assert(allocate);
            assert(destroy);
            assert(copy);
            assert(dot);
            assert(axpby);
            assert(zaxpby);
            assert(apply_op);

            return allocate && destroy && copy && dot && axpby && zaxpby && apply_op;
        }

        void monitor(const int iter, const T residual) {
            if (iter == max_it || iter % 100 == 0 || residual < tol) {
                std::cout << iter << ": " << residual << "\n";
            }
        }

        int apply(const ptrdiff_t n, const T* const b, T* const x) {
            if (left_preconditioner_op || right_preconditioner_op) {
                return aux_apply_precond(n, b, x);
            } else {
                return aux_apply_basic(n, b, x);
            }
        }

        int apply(const T* const b, T* const x) override {
            assert(n_dofs >= 0);
            if (this->n_dofs < 0) {
                std::cerr
                    << "Error uninitiaized n_dofs. Set set_n_dofs to set the number of dofs\n";
                return 1;
            }

            return apply(this->n_dofs, b, x);
        }

        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }

    private:
        int aux_apply_basic(const ptrdiff_t n, const T* const b, T* const x) {
            if (!good()) {
                return -1;
            }

            T* r0 = allocate(n);

            // Residual
            apply_op(x, r0);
            axpby(n, 1, b, -1, r0);

            T rho = dot(n, r0, r0);

            if (sqrt(rho) < tol) {
                destroy(r0);
                return 0;
            }

            T* r = allocate(n);
            T* p = allocate(n);
            T* v = allocate(n);
            T* h = allocate(n);
            T* s = allocate(n);
            T* t = allocate(n);

            copy(n, r0, r);
            copy(n, r0, p);

            int info = -1;
            for (int k = 0; k < max_it; k++) {
                zeros(n, v);
                apply_op(p, v);

                const T ptv = dot(n, r0, v);
                const T alpha = rho / ptv;

                zaxpby(n, 1, x, alpha, p, h);
                zaxpby(n, 1, r, -alpha, v, s);

                const T sts = dot(n, s, s);

                if (sqrt(sts) < tol) {
                    monitor(k, sqrt(sts));
                    copy(n, h, x);
                    info = 0;
                    break;
                }

                zeros(n, t);
                apply_op(s, t);

                const T tts = dot(n, t, s);
                const T ttt = dot(n, t, t);
                const T omega = tts / ttt;

                zaxpby(n, 1, h, omega, s, x);
                zaxpby(n, 1, s, -omega, t, r);

                const T rtr = dot(n, r, r);

                monitor(k, sqrt(rtr));
                if (sqrt(rtr) < tol) {
                    info = 0;
                    break;
                }

                const T rho_new = dot(n, r0, r);
                const T beta = (rho_new / rho) * (alpha / omega);
                rho = rho_new;

                axpby(n, 1, r, beta, p);
                axpby(n, -omega * beta, v, 1, p);
            }

            // clean-up
            destroy(r0);
            destroy(r);
            destroy(p);
            destroy(v);
            destroy(h);
            destroy(s);
            destroy(t);
            return info;
        }

        int aux_apply_precond(const ptrdiff_t n, const T* const b, T* const x) {
            if (!good()) {
                return -1;
            }

            T* r0 = allocate(n);

            // Residual
            apply_op(x, r0);
            axpby(n, 1, b, -1, r0);

            T rho = dot(n, r0, r0);

            if (sqrt(rho) < tol) {
                destroy(r0);
                return 0;
            }

            T* r = allocate(n);
            T* p = allocate(n);
            T* v = allocate(n);
            T* h = allocate(n);
            T* s = allocate(n);
            T* t = allocate(n);

            copy(n, r0, r);
            copy(n, r0, p);

            int info = -1;
            for (int k = 0; k < max_it; k++) {
                auto y = t;  // reuse t as a temp for y
                zeros(n, y);
                right_preconditioner_op(p, y);

                zeros(n, y);
                apply_op(y, v);

                const T ptv = dot(n, r0, v);
                const T alpha = rho / ptv;

                zaxpby(n, 1, x, alpha, y, h);
                zaxpby(n, 1, r, -alpha, v, s);

                const T sts = dot(n, s, s);

                if (sqrt(sts) < tol) {
                    monitor(k, sqrt(sts));
                    copy(n, h, x);
                    info = 0;
                    break;
                }

                auto z = x;  // reuse x as a temp for z

                zeros(n, z);
                right_preconditioner_op(s, z);

                zeros(n, t);
                apply_op(z, t);

                const T tts = dot(n, t, s);
                const T ttt = dot(n, t, t);
                const T omega = tts / ttt;

                zaxpby(n, 1, h, omega, z, x);
                zaxpby(n, 1, s, -omega, t, r);

                const T rtr = dot(n, r, r);

                monitor(k, sqrt(rtr));
                if (sqrt(rtr) < tol) {
                    info = 0;
                    break;
                }

                const T rho_new = dot(n, r0, r);
                const T beta = (rho_new / rho) * (alpha / omega);
                rho = rho_new;

                axpby(n, 1, r, beta, p);
                axpby(n, -omega * beta, v, 1, p);
            }

            // clean-up
            destroy(r0);
            destroy(r);
            destroy(p);
            destroy(v);
            destroy(h);
            destroy(s);
            destroy(t);
            return info;
        }
    };

    template <typename T>
    std::shared_ptr<MatrixFreeLinearSolver<T>> h_bcgs() {
        auto cg = std::make_shared<BiCGStab<T>>();
        cg->default_init();
        return cg;
    }
}  // namespace sfem

#endif  // SFEM_BCGS_HPP
