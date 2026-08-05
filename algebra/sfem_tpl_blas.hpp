#ifndef SFEM_TPL_BLAS_HPP
#define SFEM_TPL_BLAS_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace sfem {

    template <typename T>
    class BLAS {
    public:
        virtual ~BLAS() = default;

        virtual T*   allocate(const std::size_t n) = 0;
        virtual void destroy(void* a)              = 0;

        virtual void zeros(const std::size_t size, T* const x)                 = 0;
        virtual void values(const std::size_t size, const T value, T* const x) = 0;

        virtual void copy(const ptrdiff_t n, const T* const src, T* const dest) = 0;

        virtual T    dot(const ptrdiff_t n, const T* const l, const T* const r)                                             = 0;
        virtual void axpy(const ptrdiff_t n, const T alpha, const T* const x, T* const y)                                   = 0;
        virtual void axpby(const ptrdiff_t n, const T alpha, const T* const x, const T beta, T* const y)                    = 0;
        virtual void scal(const std::ptrdiff_t n, const T alpha, T* const x)                                                = 0;
        virtual void reciprocal(const std::ptrdiff_t n, const T alpha, T* const x)                                          = 0;
        virtual T    norm2(const ptrdiff_t n, const T* const x)                                                             = 0;
        virtual void zaxpby(const ptrdiff_t n, const T alpha, const T* const x, const T beta, const T* const y, T* const z) = 0;

        /// $z = x * y + \alpha * z$
        virtual void xypaz(const ptrdiff_t n, const T* const x, const T* const y, const T alpha, T* const z) = 0;

        virtual bool good() const { return true; }
    };

}  // namespace sfem

#endif
