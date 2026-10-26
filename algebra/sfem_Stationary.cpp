#include "sfem_Stationary.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_blas.hpp"
#endif


namespace sfem {

    template <typename T>
    std::shared_ptr<StationaryIteration<T>> StationaryIteration<T>::create(const std::shared_ptr<Operator<T>> &op,
                                                                           const std::shared_ptr<Operator<T>> &preconditioner,
                                                                           const ExecutionSpace                es) {
        SFEM_UNUSED(es);
        auto ret            = std::make_shared<StationaryIteration<T>>();
        ret->op             = op;
        ret->preconditioner = preconditioner;
        ret->configure_sizes_from_op();

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            ret->blas = GPUBLAS<T>::create();
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            ret->default_init();
        }

        return ret;
    }

    template std::shared_ptr<StationaryIteration<real_t>> StationaryIteration<real_t>::create(const std::shared_ptr<Operator<real_t>> &,
                                                                                              const std::shared_ptr<Operator<real_t>> &,
                                                                                              const ExecutionSpace);

}  // namespace sfem
