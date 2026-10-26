#include "sfem_ShiftableJacobi.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_ShiftableJacobi.hpp"
#include "sfem_cuda_blas.hpp"
#endif

namespace sfem {

    template <typename T>
    std::shared_ptr<ShiftableJacobi<T>> ShiftableJacobi<T>::create(const SharedBuffer<T> &diag, const ExecutionSpace es) {
        SFEM_UNUSED(es);
        auto ret = std::make_shared<sfem::ShiftableJacobi<T>>();

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            ret->blas = GPUBLAS<T>::create();
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            ret->default_init();
        }

        ret->set_diag(diag);
        return ret;
    }

    template <typename T>
    std::shared_ptr<ShiftableBlockSymJacobi<T>> ShiftableBlockSymJacobi<T>::create(const int                   dim,
                                                                                   const SharedBuffer<T>      &diag,
                                                                                   const SharedBuffer<mask_t> &constraints_mask,
                                                                                   const ExecutionContext     &context) {
        SFEM_UNUSED(context);
        auto ret = std::make_shared<sfem::ShiftableBlockSymJacobi<T>>();

#ifdef SFEM_ENABLE_CUDA
        if (context.space() == EXECUTION_SPACE_DEVICE) {
            ret->blas = GPUBLAS<T>::create(context);
            GPUShiftableBlockSymJacobiKernels<T>::build(dim, ret->impl, context);
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            ret->blas = OpenMPBLAS<T>::create();
            OpenMPShiftableBlockSymJacobiKernels<T>::build(dim, ret->impl);
        }

        ret->constraints_mask = constraints_mask;
        ret->set_diag(diag);
        return ret;
    }

    template std::shared_ptr<ShiftableJacobi<real_t>> ShiftableJacobi<real_t>::create(const SharedBuffer<real_t> &, const ExecutionSpace);
    template std::shared_ptr<ShiftableBlockSymJacobi<real_t>> ShiftableBlockSymJacobi<real_t>::create(const int,
                                                                                                      const SharedBuffer<real_t> &,
                                                                                                      const SharedBuffer<mask_t> &,
                                                                                                      const ExecutionContext &);

}  // namespace sfem
