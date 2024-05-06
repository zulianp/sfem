#ifndef SFEM_CHEB3_HPP
#define SFEM_CHEB3_HPP

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>

// https://en.wikipedia.org/wiki/Conjugate_gradient_method
namespace sfem {
    template <typename T>
    class Chebyshev3 {
    public:
        // Operator
        std::function<void(const T* const, T* const)> apply_op;
        std::function<void(const T* const, T* const)> preconditioner_op;

        // Mem management
        std::function<T*(const std::size_t)> allocate;
        std::function<void(T*)> destroy;

        std::function<void(const ptrdiff_t, const T* const, T* const)> copy;

        // blas
        std::function<T(const ptrdiff_t, const T* const, const T* const)> dot;
        std::function<T(const ptrdiff_t, const T* const)> norm2;
        std::function<void(const ptrdiff_t, const T, const T* const, const T, T* const)> axpby;

        // Solver parameters
        T tol{1e-10};
        int max_it{10000};

        T eig_max{0};
        T scale_eig_min{0.1};

        void set_preconditioner(std::function<void(const T* const, T* const)> &&in)
        {
            preconditioner_op = in;
        }

        void default_init() {
            allocate = [](const std::ptrdiff_t n) -> T* { return (T*)calloc(n, sizeof(T)); };

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

            norm2 = [](const ptrdiff_t n, const T* const x) -> T {
                T ret = 0;

#pragma omp parallel for reduction(+ : ret)
                for (ptrdiff_t i = 0; i < n; i++) {
                    ret += x[i] * x[i];
                }

                return sqrt(ret);
            };

            axpby =
                [](const ptrdiff_t n, const T alpha, const T* const x, const T beta, T* const y) {
#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n; i++) {
                        y[i] = alpha * x[i] + beta * y[i];
                    }
                };
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

        T power_method_reinit(
            const ptrdiff_t n, 
            T *const eigenvector,
            T *const work) {

            T scale_factor = 1. / norm2(n, eigenvector);
            scal(n, scale_factor, eigenvector);
        
            int it = 0;
            bool converged = false;
            T eig_max = 0;
            T eig_max_old = 0;

            while (!converged) {
                apply_op(eigenvector, work);

                eig_max = norm2(work);
                copy(n, work, eigenvector);
                
                scale_factor = (1 / eig_max);
                scal(n, scale_factor, eigenvector);
                it = it + 1;

                converged = ((std::abs(eig_max_old - eig_max) < tol) || it > max_it) ? true : false;
                eig_max_old = eig_max;
            }

            return eig_max;
        }

        void init(const ptrdiff_t n, const T *const b)
        {
            T * eigenvector = allocate(n);
            T * work = allocate(n);
            copy(n, b, eigenvector);
            
            eig_max = power_method_reinit(n, eigenvector, work);

            destroy(eigenvector);
            destroy(work);
        }

        int precond_apply(
            const ptrdiff_t n,
            const int num_iterations,
             const T* const b, 
             T* const x,
             // work-buffers
             T*const r,
             T*const p,
             T*const Ap
             ) {
            if (!good()) {
                return -1;
            }

            T eig_min = eig_max * scale_eig_min;
            T avg_eig = (eig_max + eig_min) / 2;
            T diff_eig = (eig_max - eig_min) / 2;

            // Compute residual
            apply_op(x, r);
            axpby(-1, b, 1, r);
            axpby(n, -1, r, 0, p);
            T alpha = 1/avg_eig;

            axpby(alpha, p, 1, x);

            T diff_eig_alpha = diff_eig * alpha;
            T beta = 0.5 * diff_eig_alpha * diff_eig_alpha;
            alpha = 1/(avg_eig - (beta / alpha));

            axpby(n, -1, r, beta, p);

            for(int k = 0; k < num_iterations; k++) {
                diff_eig_alpha = (diff_eig * alpha);
                beta = 0.25 * diff_eig_alpha * diff_eig_alpha;
                alpha = 1.0 / (avg_eig - (beta / alpha));
                
                axpby(n, -1, r, beta, p);
                axpby(alpha, p, 1, x);
            }

            return 0;
        }
    };
}  // namespace sfem

#endif  // SFEM_CHEB3_HPP
