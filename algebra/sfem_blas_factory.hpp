#ifndef SFEM_BLAS_FACTORY_HPP
#define SFEM_BLAS_FACTORY_HPP

#include "sfem_execution_context.hpp"
#include "sfem_tpl_blas.hpp"

#include <memory>

namespace sfem {

    /// The BLAS of a context: CUDA on the device, ordered on the context's stream; OpenMP on the
    /// host. Instantiated for float and double.
    template <typename T>
    std::shared_ptr<BLAS<T>> blas(const ExecutionContext &context);

}  // namespace sfem

#endif  // SFEM_BLAS_FACTORY_HPP
