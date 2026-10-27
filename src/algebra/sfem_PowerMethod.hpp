#ifndef SFEM_POWER_METHOD_HPP
#define SFEM_POWER_METHOD_HPP

#include <cstddef>
#include <functional>

#include "sfem_MatrixFreeLinearSolver.hpp"

namespace sfem {

    template <typename T>
    class PowerMethod {
    public:
        std::function<T(const std::ptrdiff_t, const T* const)> norm2;
        std::function<void(const std::ptrdiff_t, const T, T* const)> scal;
        std::function<void(const std::size_t, T* const)> zeros;

        T max_eigen_value(std::function<void(const T* const, T* const)> op,
                          const int max_it,
                          const T stol,
                          const std::ptrdiff_t n,
                          T* const eigen_vector,
                          T* const work) {
            int max_it_odd = max_it + (max_it % 2 == 0);
            zeros(n, work);
            op(eigen_vector, work);
            scal(n, 1 / norm2(n, work), work);

            T* vecs[2] = {eigen_vector, work};

            T lambda = 0;
            T lambda_old = 0;
            for (int k = 0; k < max_it_odd; k++) {
                T* w = vecs[k % 2];
                T* ev = vecs[(k + 1) % 2];

                zeros(n, w);
                op(ev, w);

                lambda = norm2(n, w);
                assert(lambda == lambda);
                scal(n, 1 / lambda, w);

                if (std::abs(lambda - lambda_old) <= stol && k % 2 == 0) {
                    // We exit once the results is stored in eigen_vector
                    // If overhead is "prohibitive" then just copy
                    printf("PowerMethod (%d): lambda = %g\n", k, lambda);
                    break;
                }

                lambda_old = lambda;
            }

            return lambda;
        }
    };
}  // namespace sfem

#endif  // SFEM_POWER_METHOD_HPP
