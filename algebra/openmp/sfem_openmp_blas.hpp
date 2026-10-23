#ifndef SFEM_OPENMP_BLAS_HPP
#define SFEM_OPENMP_BLAS_HPP

#include "sfem_fixed_chunks.hpp"
#include "sfem_tpl_blas.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
#ifdef SFEM_ENABLE_OPENMP
#error "_OPENMP is undefined!"
#endif
#endif

namespace sfem {

    // Deterministic reduction. On by default; SFEM_DETERMINISTIC_BLAS=0 opts out.
    //
    // OpenMP's reduction clause combines partial sums in an unspecified order over chunks
    // whose boundaries follow the thread count, so a dot product is reproducible neither
    // between runs nor between thread counts. That alone is enough to make a Krylov solve
    // irreproducible even when the operator is exact: in the CVFEM spike, a solve whose
    // operator had been made bit-deterministic still varied by about twenty percent in
    // iteration count, and the dot products were what was left.
    //
    // This splits the range into a fixed number of chunks that does not depend on the thread
    // count, sums each serially, and combines them in index order, so the result is identical
    // for any number of threads. The cost is one 256-element array and a serial combine of
    // it. Off by default, because it changes results in the last bits relative to the
    // existing path and callers may be comparing against those.
    // On by default. SFEM_DETERMINISTIC_BLAS=0 restores the OpenMP reduction clause.
    //
    // Made the default because it is better on every axis measured: reproducible across
    // thread counts, more accurate, and faster. On a CVFEM Navier-Stokes solve it cut the
    // per-iteration cost from 8509 to 7672 microseconds -- the reduction clause privatises
    // and combines per thread, where this writes a flat array and sums it once.
    inline bool blas_deterministic() {
        static const bool on = [] {
            const char *e = std::getenv("SFEM_DETERMINISTIC_BLAS");
            return !e || std::atoi(e) != 0;
        }();
        return on;
    }

    template <typename T, typename F>
    inline T blas_fixed_chunk_sum(const ptrdiff_t n, F term) {
        const ptrdiff_t NC = blas_chunk_count(n);

        if (NC == 1) {
            T acc = 0;
            for (ptrdiff_t i = 0; i < n; ++i) acc += term(i);
            return acc;
        }

        std::vector<T> part((size_t)NC);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t c = 0; c < NC; ++c) {
            ptrdiff_t b, e;
            blas_chunk_bounds(n, NC, c, &b, &e);
            T acc = 0;
            for (ptrdiff_t i = b; i < e; ++i) acc += term(i);
            part[(size_t)c] = acc;
        }
        T ret = 0;
        for (ptrdiff_t c = 0; c < NC; ++c) ret += part[(size_t)c];
        return ret;
    }


    template <typename T>
    class OpenMP_BLAS final : public BLAS<T> {
    public:
        auto allocate(const std::size_t n) -> T* override { return (T*)calloc(n, sizeof(T)); }

        void destroy(void* a) override { free(a); }

        void copy(const ptrdiff_t n, const T* const src, T* const dest) override {
            memcpy(dest, src, n * sizeof(T));
        }

        auto dot(const ptrdiff_t n, const T* const l, const T* const r) -> T override {
            if (blas_deterministic())
                return blas_fixed_chunk_sum<T>(n, [l, r](const ptrdiff_t i) { return l[i] * r[i]; });

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
            if (blas_deterministic())
                return sqrt(blas_fixed_chunk_sum<T>(n, [x](const ptrdiff_t i) { return x[i] * x[i]; }));

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
