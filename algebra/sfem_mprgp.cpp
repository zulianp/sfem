#include "sfem_mprgp.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_blas.hpp"
#include "sfem_cuda_mprgp_impl.hpp"
#endif


namespace sfem {

    template <typename T>
    std::shared_ptr<MPRGP<T>> MPRGP<T>::create(const std::shared_ptr<Operator<T>> &op, const ExecutionContext &context) {
        SFEM_UNUSED(context);
        auto mprgp = std::make_shared<sfem::MPRGP<T>>();
        mprgp->set_op(op);

#ifdef SFEM_ENABLE_CUDA
        if (context.space() == EXECUTION_SPACE_DEVICE) {
            mprgp->blas = GPUBLAS<T>::create(context);
            GPUMPRGPKernels<T>::build_mprgp(mprgp->impl, context);

        } else
#endif  // SFEM_ENABLE_CUDA
        {
            mprgp->default_init();
        }

        return mprgp;
    }

    template std::shared_ptr<MPRGP<real_t>> MPRGP<real_t>::create(const std::shared_ptr<Operator<real_t>> &, const ExecutionContext &);

}  // namespace sfem
