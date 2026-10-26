#ifndef SFEM_TPL_BLAS_HPP
#define SFEM_TPL_BLAS_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "sfem_execution_context.hpp"

namespace sfem {

    /// Vector operations in one execution context: its space, and on the device the stream every
    /// operation is ordered on.
    template <typename T>
    class BLAS {
    public:
        explicit BLAS(const ExecutionContext& context) : context_(context) {}
        virtual ~BLAS() = default;

        const ExecutionContext& context() const { return context_; }
        smesh::ExecutionSpace   execution_space() const { return context_.space(); }

        virtual T*   allocate(const std::size_t n) = 0;
        virtual void destroy(void* a)              = 0;

        virtual void zeros(const std::size_t size, T* const x)                 = 0;
        virtual void values(const std::size_t size, const T value, T* const x) = 0;

        virtual void copy(const ptrdiff_t n, const T* const src, T* const dest) = 0;

        virtual T    dot(const ptrdiff_t n, const T* const l, const T* const r)                                             = 0;
        /// l . r into *result, which lives in this context's memory; ordered on the context and not
        /// waited for (on the device: no allocation and no synchronisation, so it can be captured).
        virtual void dot(const ptrdiff_t n, const T* const l, const T* const r, T* const result) = 0;
        virtual void axpy(const ptrdiff_t n, const T alpha, const T* const x, T* const y)                                   = 0;
        virtual void axpby(const ptrdiff_t n, const T alpha, const T* const x, const T beta, T* const y)                    = 0;
        virtual void scal(const std::ptrdiff_t n, const T alpha, T* const x)                                                = 0;
        virtual void reciprocal(const std::ptrdiff_t n, const T alpha, T* const x)                                          = 0;
        virtual T    norm2(const ptrdiff_t n, const T* const x)                                                             = 0;
        virtual void zaxpby(const ptrdiff_t n, const T alpha, const T* const x, const T beta, const T* const y, T* const z) = 0;

        /// $z = x * y + \alpha * z$
        virtual void xypaz(const ptrdiff_t n, const T* const x, const T* const y, const T alpha, T* const z) = 0;

        virtual bool good() const { return true; }

    private:
        ExecutionContext context_;
    };

}  // namespace sfem

#endif
