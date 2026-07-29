#ifndef SFEM_TPL_BLAS_HPP
#define SFEM_TPL_BLAS_HPP

#include <cassert>
#include <cstdint>
#include <functional>

namespace sfem {

    template <typename T>
    struct BLAS_Tpl {
        std::function<T*(const std::size_t)> allocate;
        std::function<void(void*)> destroy;

        std::function<void(const std::size_t, T* const x)> zeros;
        std::function<void(const std::size_t, const T value, T* const x)> values;

        std::function<void(const ptrdiff_t, const T* const, T* const)> copy;

        std::function<T(const ptrdiff_t, const T* const, const T* const)> dot;
        std::function<void(const ptrdiff_t, const T* const, const T* const, T* const)> dot_into;
        std::function<void(const ptrdiff_t, const T, const T* const, T* const)> axpy;
        std::function<void(const ptrdiff_t, const T, const T* const, const T, T* const)> axpby;
        std::function<void(const std::ptrdiff_t, const T, T* const)> scal;
        std::function<void(const std::ptrdiff_t, const T, T* const)> reciprocal;
        std::function<T(const ptrdiff_t, const T* const)> norm2;
        std::function<void(const ptrdiff_t, const T* const, T* const)> norm2_into;
        std::function<void(const ptrdiff_t, const T* const, T* const)> copy_scalars_to_host;
        std::function<
                void(const ptrdiff_t, const T, const T* const, const T, const T* const, T* const)>
                zaxpby;

        /// $z = x * y + \alpha * z$
        std::function<void(const ptrdiff_t, const T*const, const T*const, const T, T*const)> xypaz;

        bool good() const {
            assert(allocate);
            assert(destroy);
            assert(copy);
            assert(zeros);
            assert(values);
            assert(dot);
            assert(dot_into);
            assert(norm2);
            assert(norm2_into);
            assert(copy_scalars_to_host);
            assert(axpy);
            assert(axpby);
            assert(scal);

            return allocate && destroy && copy && zeros && values && dot && dot_into && norm2 && norm2_into && copy_scalars_to_host && axpy &&
                   axpby && zaxpby && scal && reciprocal;
        }
    };

}  // namespace sfem

#endif
