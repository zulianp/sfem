#ifndef SFEM_CG_HPP
#define SFEM_CG_HPP

#include <function>
#include <cmath>

// https://en.wikipedia.org/wiki/Conjugate_gradient_method
namespace sfem {
    template <typename T>
    class ConjugateGradient {
    public:
    	// Operator
        std::function<void(const T* const, T* const)> apply;
        
        // Mem management
        std::function<T*(const std::size_t)> allocate;
        std::function<void(const T*)> destroy;
        
        // blas
        std::function<T(const T* const, const T* const)> dot;
        std::function<void(const T, const T* const, const T, T* const)> axpby;

        // Solver parameters
        T tol{1e-10};

        int apply(const size_t n, const T* const b, T* const x) {
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

                const T ptAp = dot(p, Ap);
                const T alpha = rtr / ptAp;

                // Opt use 2 cuda streams?
                axpby(alpha, p, 1, x);
                axpby(-alpha, Ap, 1, r);

                const T rtr_new = dot(r, r);
                if (sqrt(rtr_new) < tol) {
                    info = 0;
                    break;
                }

                const T beta = rtr_new / rtr;
                rtr = rtr_new;
                axpby(1, r, beta, p);
            }

            // clean-up
            destroy(r);
            destroy(p);
            destroy(Ap);
            return info;
        }
    };
}  // namespace sfem

#endif SFEM_CG_HPP
