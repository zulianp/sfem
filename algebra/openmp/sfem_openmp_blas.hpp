#ifndef SFEM_OPENMP_BLAS_HPP
#define SFEM_OPENMP_BLAS_HPP

#include "sfem_tpl_blas.hpp"

#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#else
#ifdef SFEM_ENABLE_OPENMP
#error "_OPENMP is undefined!"
#endif
#endif

namespace sfem {

    template <typename T>
    struct OpenMP_BLAS {
        static auto allocate(const std::ptrdiff_t n) -> T* { return (T*)calloc(n, sizeof(T)); }

        static void destroy(void* a) { free(a); }

        static void copy(const ptrdiff_t n, const T* const src, T* const dest) {
            std::memcpy(dest, src, n * sizeof(T));
        }

        static auto dot(const ptrdiff_t n, const T* const l, const T* const r) -> T {
            T ret = 0;

#pragma omp parallel for reduction(+ : ret)
            for (ptrdiff_t i = 0; i < n; i++) {
                ret += l[i] * r[i];
            }

            return ret;
        }

        static void axpby(const ptrdiff_t n,
                          const T alpha,
                          const T* const x,
                          const T beta,
                          T* const y) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; i++) {
                y[i] = alpha * x[i] + beta * y[i];
            }
        }

        static void zaxpby(const ptrdiff_t n,
                           const T alpha,
                           const T* const x,
                           const T beta,
                           const T* const y,
                           T* const z) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; i++) {
                z[i] = alpha * x[i] + beta * y[i];
            }
        }

        static void zeros(const std::size_t size, T* const x) {
#ifdef _OPENMP
#pragma omp parallel
            {
                size_t start, len;
                int id = omp_get_thread_num();
                int num = omp_get_num_threads();

                start = (id * size) / num;
                len = ((id + 1) * size) / num - start;

                memset(&x[start], 0, len * sizeof(T));
            }
#else
            memset(x, 0, size * sizeof(T));
#endif
        }

        static auto norm2(const ptrdiff_t n, const T* const x) -> T {
            T ret = 0;

#pragma omp parallel for reduction(+ : ret)
            for (ptrdiff_t i = 0; i < n; i++) {
                ret += x[i] * x[i];
            }

            return sqrt(ret);
        }

        static void values(const std::ptrdiff_t n, const T v, T* const x) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; i++) {
                x[i] = v;
            }
        }

        static void scal(const std::ptrdiff_t n, const T alpha, T* const x) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; i++) {
                x[i] *= alpha;
            }
        }

        static void xypaz(const std::ptrdiff_t n,
                           const T* const x,
                           const T* const y,
                           const T alpha,
                           T* const z) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; i++) {
                z[i] = x[i] * y[i] + alpha * z[i];
            }
        }

        static void build_blas(struct BLAS_Tpl<T>& tpl) {
            tpl.allocate = allocate;
            tpl.destroy = destroy;
            tpl.copy = copy;
            tpl.dot = dot;
            tpl.norm2 = norm2;
            tpl.axpy = [](const ptrdiff_t n, const T alpha, const T* const x, T* const y) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; i++) {
                    y[i] += alpha * x[i];
                }
            };

            tpl.axpby = axpby;
            tpl.zaxpby = zaxpby;
            tpl.zeros = zeros;
            tpl.values = values;
            tpl.scal = scal;
            tpl.xypaz = xypaz;
        }
    };

}  // namespace sfem

#endif  // SFEM_OPENMP_BLAS_HPP
