#ifndef SFEM_OPENMP_BLAS_HPP
#define SFEM_OPENMP_BLAS_HPP

#include "sfem_tpl_blas.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#else
#ifdef SFEM_ENABLE_OPENMP
#error "_OPENMP is undefined!"
#endif
#endif

namespace sfem {

    template <typename T>
    class OpenMP_BLAS final : public BLAS<T> {
    public:
        auto allocate(const std::size_t n) -> T* override { return (T*)calloc(n, sizeof(T)); }

        void destroy(void* a) override { free(a); }

        void copy(const ptrdiff_t n, const T* const src, T* const dest) override {
            memcpy(dest, src, n * sizeof(T));
        }

        auto dot(const ptrdiff_t n, const T* const l, const T* const r) -> T override {
            T ret = 0;

#pragma omp parallel for reduction(+ : ret)
            for (ptrdiff_t i = 0; i < n; i++) {
                ret += l[i] * r[i];
            }

            return ret;
        }

        void axpy(const ptrdiff_t n, const T alpha, const T* const x, T* const y) override {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; i++) {
                y[i] += alpha * x[i];
            }
        }

        void axpby(const ptrdiff_t n, const T alpha, const T* const x, const T beta,
                   T* const y) override {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; i++) {
                y[i] = alpha * x[i] + beta * y[i];
            }
        }

        void zaxpby(const ptrdiff_t n, const T alpha, const T* const x, const T beta,
                    const T* const y, T* const z) override {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; i++) {
                z[i] = alpha * x[i] + beta * y[i];
            }
        }

        void zeros(const std::size_t size, T* const x) override {
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

        auto norm2(const ptrdiff_t n, const T* const x) -> T override {
            T ret = 0;

#pragma omp parallel for reduction(+ : ret)
            for (ptrdiff_t i = 0; i < n; i++) {
                ret += x[i] * x[i];
            }

            return sqrt(ret);
        }

        void values(const std::size_t n, const T v, T* const x) override {
#pragma omp parallel for
            for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)n; i++) {
                x[i] = v;
            }
        }

        void scal(const std::ptrdiff_t n, const T alpha, T* const x) override {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; i++) {
                x[i] *= alpha;
            }
        }

        void reciprocal(const std::ptrdiff_t n, const T alpha, T* const x) override {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; i++) {
                if (x[i]) x[i] = alpha / x[i];
            }
        }

        void xypaz(const std::ptrdiff_t n, const T* const x, const T* const y, const T alpha,
                   T* const z) override {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; i++) {
                z[i] = x[i] * y[i] + alpha * z[i];
            }
        }
    };

    template <typename T>
    std::shared_ptr<BLAS<T>> make_openmp_blas() {
        return std::make_shared<OpenMP_BLAS<T>>();
    }

}  // namespace sfem

#endif  // SFEM_OPENMP_BLAS_HPP
