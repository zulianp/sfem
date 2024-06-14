#ifndef SFEM_POINT_JACOBI_HPP
#define SFEM_POINT_JACOBI_HPP

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>

#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_PointJacobi.hpp"

// https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
namespace sfem {
    template <typename T>
    class PointJacobi final : public MatrixFreeLinearSolver<T> {
    public:
        // Operator
        std::function<void(const T* const, T* const)> apply_op;
        std::function<void(const T* const, T* const)> left_preconditioner_op;
        std::function<void(const T* const, T* const)> right_preconditioner_op;

        // Mem management
        std::function<T*(const std::size_t)> allocate;
        std::function<void(T*)> destroy;

        std::function<void(const std::size_t, T* const x)> zeros;

        std::function<void(const ptrdiff_t, const T* const, T* const)> copy;

        // blas
        std::function<T(const ptrdiff_t, const T* const, const T* const)> dot;
        std::function<void(const ptrdiff_t, const T, const T* const, const T, T* const)> axpby;
        std::function<
                void(const ptrdiff_t, const T, const T* const, const T, const T* const, T* const)>
                zaxpby;

        // x[i] += r[i] / d[i];
        std::function<void(const std::size_t, const T* const, T* const)> jacobi_correction_op;

        ptrdiff_t n_dofs{-1};

        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }

        void set_op(const std::shared_ptr<Operator<T>>& op) override {
            this->apply_op = [=](const T* const x, T* const y) { op->apply(x, y); };
        }

        void set_preconditioner_op(const std::shared_ptr<Operator<T>>& op) override {
            assert(false);
        }

        void set_max_it(const int it) override { max_it = it; }

        void set_preconditioner(std::function<void(const T* const, T* const)>&& in) {
            // left_preconditioner_op = in;
            right_preconditioner_op = in;
        }

        // Solver parameters
        T tol{1e-10};
        int max_it{10000};
        int check_each{100};
        bool verbose{false};

        void default_init() {
            allocate = [](const ptrdiff_t n) -> T* { return (T*)calloc(n, sizeof(T)); };

            destroy = [](T* a) { free(a); };

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

            zeros = [](const std::size_t n, T* const x) { memset(x, 0, n * sizeof(T)); };
        }

        bool good() const {
            assert(allocate);
            assert(destroy);
            assert(copy);
            assert(dot);
            assert(axpby);
            assert(zaxpby);
            assert(apply_op);
            assert(jacobi_correction_op);

            return allocate && destroy && copy && dot && axpby && zaxpby && apply_op &&
                   jacobi_correction_op;
        }

        void monitor(const int iter, const T residual) {
            if (verbose) {
                std::cout << iter << ": " << residual << "\n";
            }
        }

        int apply(const ptrdiff_t n, const T* const b, T* const x) {
            if (!good()) {
                return -1;
            }

            T* r = allocate(n);

            // Residual
            apply_op(x, r);
            axpby(n, 1, b, -1, r);

            const T norm_r0 = dot(n, r, r);
            T norm_r = norm_r0;
            if (sqrt(norm_r) < tol) {
                destroy(r);
                return 0;
            }

            int info = -1;
            int k = 1;
            for (; k < max_it; k++) {
                zeros(n, r);
                apply_op(x, r);
                axpby(n, 1, b, -1, r);
                jacobi_correction_op(n, r, x);

                if (k % check_each == 0) {
                    const T norm_r = sqrt(dot(n, r, r));
                    monitor(k, norm_r);

                    if (norm_r < tol || norm_r != norm_r) {
                        assert(norm_r == norm_r);
                        break;
                    }
                }
            }

            if (verbose) {
                const T norm_r = sqrt(dot(n, r, r));
                std::printf("Finished at iteration %d with |r| = %g, reduction %g\n",
                            k,
                            (double)norm_r,
                            (double)(norm_r / norm_r0));
            }

            // clean-up
            destroy(r);
            return info;
        }

        int apply(const T* const b, T* const x) override { return apply(n_dofs, b, x); }
        void set_n_dofs(const ptrdiff_t n) override { this->n_dofs = n; }
    };

    template <typename T>
    std::shared_ptr<PointJacobi<T>> h_pjacobi(const ptrdiff_t n,
                                              const T* const d,
                                              const T relax = T(1.0)) {
        auto jacobi = std::make_shared<PointJacobi<T>>();

        jacobi->jacobi_correction_op = [=](const std::size_t n, const T* const r, T* const x) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; i++) {
                assert(d[i] != 0);
                x[i] += relax * r[i] / d[i];
            }
        };

        jacobi->set_n_dofs(n);
        jacobi->default_init();
        return jacobi;
    }
}  // namespace sfem

#endif  // SFEM_POINT_JACOBI_HPP
