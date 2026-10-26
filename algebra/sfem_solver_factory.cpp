#include "sfem_solver_factory.hpp"

#include "sfem_openmp_blas.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_ShiftableJacobi.hpp"
#include "sfem_cuda_blas.hpp"
#include "sfem_cuda_mprgp_impl.hpp"
#include "sfem_cuda_solver.hpp"
#endif

namespace sfem {

    template <typename T>
    std::shared_ptr<ConjugateGradient<T>> create_cg(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es) {
        SFEM_UNUSED(es);
        std::shared_ptr<ConjugateGradient<T>> cg;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            cg = sfem::d_cg<T>();
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            cg = sfem::h_cg<T>();
        }

        cg->set_n_dofs(op->rows());
        cg->set_op(op);
        return cg;
    }

    template <typename T>
    std::shared_ptr<ParallelConjugateGradient<T>> create_parallel_cg(const std::shared_ptr<ParallelOperator<T>> &op) {
        if (!op) {
            SFEM_ERROR("create_parallel_cg: null operator\n");
            return nullptr;
        }

        if (op->execution_space() != EXECUTION_SPACE_HOST) {
            SFEM_ERROR("create_parallel_cg: only EXECUTION_SPACE_HOST is supported currently\n");
            return nullptr;
        }

        auto cg = h_parallel_cg<T>();
        cg->set_n_dofs(op->rows());
        cg->set_op(op);
        return cg;
    }

    template <typename T>
    std::shared_ptr<ShiftableJacobi<T>> create_shiftable_jacobi(const SharedBuffer<T> &diag, const ExecutionSpace es) {
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

    template <typename HP, typename LP>
    std::shared_ptr<MixedPrecisionShiftableBlockSymJacobi<HP, LP>> create_mixed_precision_shiftable_block_sym_jacobi(
            const int                   dim,
            const SharedBuffer<HP>     &diag,
            const SharedBuffer<mask_t> &constraints_mask,
            const ExecutionContext     &context) {
        SFEM_UNUSED(context);
        auto ret = std::make_shared<sfem::MixedPrecisionShiftableBlockSymJacobi<HP, LP>>();

#ifdef SFEM_ENABLE_CUDA
        if (context.space() == EXECUTION_SPACE_DEVICE) {
            ret->blas = GPUBLAS<LP>::create(context);
            ShiftableBlockSymJacobi_CUDA<HP, LP>::build(dim, ret->impl, context);
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            ret->blas = OpenMPBLAS<LP>::create();
            ShiftableBlockSymJacobi_OpenMP<HP, LP>::build(dim, ret->impl);
        }

        ret->constraints_mask = constraints_mask;
        ret->set_diag(diag);
        return ret;
    }

    template <typename T>
    std::shared_ptr<ShiftableBlockSymJacobi<T>> create_shiftable_block_sym_jacobi(const int                   dim,
                                                                                  const SharedBuffer<T>      &diag,
                                                                                  const SharedBuffer<mask_t> &constraints_mask,
                                                                                  const ExecutionContext     &context) {
        SFEM_UNUSED(context);
        auto ret = std::make_shared<sfem::ShiftableBlockSymJacobi<T>>();

#ifdef SFEM_ENABLE_CUDA
        if (context.space() == EXECUTION_SPACE_DEVICE) {
            ret->blas = GPUBLAS<T>::create(context);
            ShiftableBlockSymJacobi_CUDA<T>::build(dim, ret->impl, context);
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            ret->blas = OpenMPBLAS<T>::create();
            ShiftableBlockSymJacobi_OpenMP<T>::build(dim, ret->impl);
        }

        ret->constraints_mask = constraints_mask;
        ret->set_diag(diag);
        return ret;
    }

    template <typename T>
    std::shared_ptr<StationaryIteration<T>> create_stationary(const std::shared_ptr<Operator<T>> &op,
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

    template <typename T>
    std::shared_ptr<BiCGStab<T>> create_bcgs(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es) {
        SFEM_UNUSED(es);
        std::shared_ptr<BiCGStab<T>> bcgs;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            bcgs = sfem::d_bcgs<T>();
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            bcgs = sfem::h_bcgs<T>();
        }

        bcgs->set_n_dofs(op->rows());
        bcgs->set_op(op);
        return bcgs;
    }

    template <typename T>
    std::shared_ptr<Chebyshev3<T>> create_cheb3(const std::shared_ptr<Operator<T>> &op, const ExecutionSpace es) {
        SFEM_UNUSED(es);
        std::shared_ptr<Chebyshev3<T>> cheb;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            cheb = sfem::d_cheb3<T>(op);
        } else
#endif  // SFEM_ENABLE_CUDA
        {
            cheb = sfem::h_cheb3<T>(op);
        }

        return cheb;
    }

    template <typename T>
    std::shared_ptr<MPRGP<T>> create_mprgp(const std::shared_ptr<Operator<T>> &op, const ExecutionContext &context) {
        SFEM_UNUSED(context);
        auto mprgp = std::make_shared<sfem::MPRGP<T>>();
        mprgp->set_op(op);

#ifdef SFEM_ENABLE_CUDA
        if (context.space() == EXECUTION_SPACE_DEVICE) {
            mprgp->blas = GPUBLAS<T>::create(context);
            CUDA_MPRGP<T>::build_mprgp(mprgp->impl, context);

        } else
#endif  // SFEM_ENABLE_CUDA
        {
            mprgp->default_init();
        }

        return mprgp;
    }

    template <typename T>
    std::shared_ptr<Multigrid<T>> create_mg(const ExecutionSpace es) {
        SFEM_UNUSED(es);
        std::shared_ptr<Multigrid<T>> mg;

#ifdef SFEM_ENABLE_CUDA
        if (es == EXECUTION_SPACE_DEVICE) {
            mg = sfem::d_mg<T>();

        } else
#endif  // SFEM_ENABLE_CUDA
        {
            mg = sfem::h_mg<T>();
        }

        return mg;
    }

    template std::shared_ptr<ConjugateGradient<real_t>>         create_cg<real_t>(const std::shared_ptr<Operator<real_t>> &,
                                                                          const ExecutionSpace);
    template std::shared_ptr<ParallelConjugateGradient<real_t>> create_parallel_cg<real_t>(
            const std::shared_ptr<ParallelOperator<real_t>> &);
    template std::shared_ptr<ShiftableJacobi<real_t>> create_shiftable_jacobi<real_t>(const SharedBuffer<real_t> &,
                                                                                      const ExecutionSpace);
    template std::shared_ptr<MixedPrecisionShiftableBlockSymJacobi<real_t, float>>
    create_mixed_precision_shiftable_block_sym_jacobi<real_t, float>(const int,
                                                                     const SharedBuffer<real_t> &,
                                                                     const SharedBuffer<mask_t> &,
                                                                     const ExecutionContext &);
    template std::shared_ptr<ShiftableBlockSymJacobi<real_t>> create_shiftable_block_sym_jacobi<real_t>(
            const int,
            const SharedBuffer<real_t> &,
            const SharedBuffer<mask_t> &,
            const ExecutionContext &);
    template std::shared_ptr<StationaryIteration<real_t>> create_stationary<real_t>(const std::shared_ptr<Operator<real_t>> &,
                                                                                    const std::shared_ptr<Operator<real_t>> &,
                                                                                    const ExecutionSpace);
    template std::shared_ptr<BiCGStab<real_t>>            create_bcgs<real_t>(const std::shared_ptr<Operator<real_t>> &,
                                                                   const ExecutionSpace);
    template std::shared_ptr<Chebyshev3<real_t>>          create_cheb3<real_t>(const std::shared_ptr<Operator<real_t>> &,
                                                                      const ExecutionSpace);
    template std::shared_ptr<MPRGP<real_t>>               create_mprgp<real_t>(const std::shared_ptr<Operator<real_t>> &,
                                                                 const ExecutionContext &);
    template std::shared_ptr<Multigrid<real_t>>           create_mg<real_t>(const ExecutionSpace);

}  // namespace sfem
