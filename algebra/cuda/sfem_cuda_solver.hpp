#ifndef SFEM_CUDA_SOLVER_HPP
#define SFEM_CUDA_SOLVER_HPP

#include "sfem_Chebyshev3.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"
#include "sfem_cuda_blas.hpp"
#include "sfem_cuda_cg_impl.hpp"

#include <map>
#include <memory>

namespace sfem {

    template <typename T>
    std::shared_ptr<ConjugateGradient<T>> d_cg(const ExecutionContext& context = ExecutionContext::device()) {
        auto cg              = std::make_shared<ConjugateGradient<T>>();
        cg->blas             = GPUBLAS<T>::create(context);
        GPUConjugateGradientKernels<T>::build(cg->impl, context);
        return cg;
    }

    template <typename T>
    std::shared_ptr<BiCGStab<T>> d_bcgs() {
        auto cg = std::make_shared<BiCGStab<T>>();
        cg->blas = GPUBLAS<T>::create();
        return cg;
    }

    template <typename T>
    std::shared_ptr<Chebyshev3<T>> d_cheb3(const std::shared_ptr<Operator<T>>& op) {
        auto ret = std::make_shared<Chebyshev3<T>>();
        ret->n_dofs = op->rows();
        ret->set_op(op);

        ret->blas = GPUBLAS<T>::create();
        ret->ensure_power_method();
        return ret;
    }

    template <typename T>
    std::shared_ptr<Multigrid<T>> d_mg() {
        auto mg = std::make_shared<Multigrid<T>>();
        mg->blas() = GPUBLAS<T>::create();
        return mg;
    }
    
}  // namespace sfem

#endif
