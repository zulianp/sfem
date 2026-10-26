#include "sfem_MixedPrecisionShiftableBlockSymJacobi.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_ShiftableBlockSymJacobiKernels_cuda.hpp"
#include "sfem_BLAS_cuda.hpp"
#endif

namespace sfem {

    template <typename HP, typename LP>
    std::shared_ptr<MixedPrecisionShiftableBlockSymJacobi<HP, LP>> MixedPrecisionShiftableBlockSymJacobi<HP, LP>::create(
            const int                   dim,
            const SharedBuffer<HP>     &diag,
            const SharedBuffer<mask_t> &constraints_mask,
            const ExecutionContext     &context) {
        SFEM_UNUSED(context);
        auto ret = std::make_shared<sfem::MixedPrecisionShiftableBlockSymJacobi<HP, LP>>();

#ifdef SFEM_ENABLE_CUDA
        if (context.space() == EXECUTION_SPACE_DEVICE) {
            ret->blas = GPUBLAS<LP>::create(context);
            GPUShiftableBlockSymJacobiKernels<HP, LP>::build(dim, ret->impl, context);
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            ret->blas = OpenMPBLAS<LP>::create();
            OpenMPShiftableBlockSymJacobiKernels<HP, LP>::build(dim, ret->impl);
        }

        ret->constraints_mask = constraints_mask;
        ret->set_diag(diag);
        return ret;
    }

    template std::shared_ptr<MixedPrecisionShiftableBlockSymJacobi<real_t, float>>
    MixedPrecisionShiftableBlockSymJacobi<real_t, float>::create(const int,
                                                                 const SharedBuffer<real_t> &,
                                                                 const SharedBuffer<mask_t> &,
                                                                 const ExecutionContext &);

}  // namespace sfem
