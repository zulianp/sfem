#ifndef SFEM_OPENMP_BLAS_HPP
#define SFEM_OPENMP_BLAS_HPP

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
    struct BLAS_Tpl {
        std::function<T*(const std::size_t)> allocate;
        std::function<void(const std::size_t, T* const x)> zeros;
        std::function<void(const std::size_t, const T value, T* const x)> values;
        std::function<void(void*)> destroy;
        std::function<void(const ptrdiff_t, const T* const, T* const)> copy;
        std::function<T(const ptrdiff_t, const T* const, const T* const)> dot;
        std::function<void(const ptrdiff_t, const T, const T* const, const T, T* const)> axpby;
        std::function<void(const std::ptrdiff_t, const T, T* const)> scal;
        std::function<T(const ptrdiff_t, const T* const)> norm2;

        bool good() const {
            assert(allocate);
            assert(destroy);
            assert(copy);
            assert(zeros);
            assert(values);
            assert(dot);
            assert(norm2);
            assert(axpby);
            assert(scal);

            return allocate && destroy && copy && zeros && values && dot && norm2 && axpby && scal;
        }
    };

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

        static void build_blas(struct BLAS_Tpl<T>& tpl) {
            tpl.allocate = allocate;
            tpl.destroy = destroy;
            tpl.copy = copy;
            tpl.dot = dot;
            tpl.norm2 = norm2;
            tpl.axpby = axpby;
            tpl.zeros = zeros;
            tpl.values = values;
            tpl.scal = scal;
        }
    };

}  // namespace sfem

#endif  // SFEM_OPENMP_BLAS_HPP
