#ifndef SFEM_TPL_BLAS_HPP
#define SFEM_TPL_BLAS_HPP

#include <functional>
#include <cstdint>
#include <cassert>

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

}  // namespace sfem

#endif
