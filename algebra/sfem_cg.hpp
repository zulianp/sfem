#ifndef SFEM_CG_HPP
#define SFEM_CG_HPP

#include <cmath>
#include <functional>
#include <cstdlib>
#include <iostream>

// https://en.wikipedia.org/wiki/Conjugate_gradient_method
namespace sfem {
    template <typename T>
    class ConjugateGradient {
    public:
        // Operator
        std::function<void(const T* const, T* const)> apply_op;

        // Mem management
        std::function<T*(const std::size_t)> allocate;
        std::function<void(T*)> destroy;

        // blas
        std::function<T(const std::size_t, const T* const, const T* const)> dot;
        std::function<void(const std::size_t, const T, const T* const, const T, T* const)> axpby;

        // Solver parameters
        T tol{1e-10};
        int max_it{10000};

        void default_init() {
            allocate = [](const std::size_t n) -> T* { return (T*)calloc(n, sizeof(T)); };

            destroy = [](T* a) { free(a); };

            dot = [](const std::size_t n, const T* const l, const T* const r) -> T {
                T ret = 0;
                for (std::size_t i = 0; i < n; i++) {
                    ret += l[i] * r[i];
                }

                return ret;
            };

            axpby =
                [](const std::size_t n, const T alpha, const T* const x, const T beta, T* const y) {
                    for (std::size_t i = 0; i < n; i++) {
                        y[i] = alpha * x[i] + beta * y[i];
                    }
                };
        }

        bool good() const {
            assert(allocate);
            assert(destroy);
            assert(dot);
            assert(axpby);
            assert(apply_op);

            return allocate && destroy && dot && axpby && apply_op;
        }

        void monitor(const int iter, const T residual) {
            std::cout << iter << ": " << residual << "\n";
        }

        int apply(const size_t n, const T* const b, T* const x) {
            if(!good()) {
                return -1;
            }

            T* r = allocate(n);

            apply(x, r);
            axpby(1, b, -1, r);

            T rtr = dot(r, r);

            if (sqrt(rtr) < tol) {
                destroy(r);
                return 0;
            }

            T* p = allocate(n);
            T* Ap = allocate(n);

            int info = -1;
            for (int k = 0; k < max_it; k++) {
                apply(p, Ap);

                const T ptAp = dot(n, p, Ap);
                const T alpha = rtr / ptAp;

                // Opt use 2 cuda streams?
                axpby(n, alpha, p, 1, x);
                axpby(n, -alpha, Ap, 1, r);

                const T rtr_new = dot(r, r);

                monitor(k, sqrt(rtr_new));
                
                if (sqrt(rtr_new) < tol) {
                    info = 0;
                    break;
                }

                const T beta = rtr_new / rtr;
                rtr = rtr_new;
                axpby(n, 1, r, beta, p);
            }

            // clean-up
            destroy(r);
            destroy(p);
            destroy(Ap);
            return info;
        }
    };
}  // namespace sfem

#endif //SFEM_CG_HPP
