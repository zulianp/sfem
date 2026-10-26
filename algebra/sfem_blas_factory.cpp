#include "sfem_blas_factory.hpp"

#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "sfem_openmp_blas.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_BLAS_cuda.hpp"
#endif

namespace sfem {

    template <typename T>
    std::shared_ptr<BLAS<T>> blas(const ExecutionContext &context) {
        SFEM_UNUSED(context);
#ifdef SFEM_ENABLE_CUDA
        if (context.space() == EXECUTION_SPACE_DEVICE) {
            return GPUBLAS<T>::create(context);
        }
#endif  // SFEM_ENABLE_CUDA
        return OpenMPBLAS<T>::create();
    }

    template std::shared_ptr<BLAS<float>>  blas<float>(const ExecutionContext &);
    template std::shared_ptr<BLAS<double>> blas<double>(const ExecutionContext &);

}  // namespace sfem
