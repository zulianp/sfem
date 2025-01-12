#ifndef SFEM_CUDA_SOLVER_HPP
#define SFEM_CUDA_SOLVER_HPP

#include "sfem_Chebyshev3.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_blas.hpp"

#include <map>
#include <memory>

namespace sfem {

    template <typename T>
    std::shared_ptr<ConjugateGradient<T>> d_cg() {
        auto cg = std::make_shared<ConjugateGradient<T>>();
        CUDA_BLAS<T>::build_blas(cg->blas);
        cg->execution_space_ = EXECUTION_SPACE_DEVICE;
        return cg;
    }

    template <typename T>
    std::shared_ptr<BiCGStab<T>> d_bcgs() {
        auto cg = std::make_shared<BiCGStab<T>>();
        CUDA_BLAS<T>::build_blas(cg->blas);
        cg->execution_space_ = EXECUTION_SPACE_DEVICE;
        return cg;
    }

    template <typename T>
    std::shared_ptr<Chebyshev3<T>> d_cheb3(const std::shared_ptr<Operator<T>>& op) {
        auto ret = std::make_shared<Chebyshev3<T>>();
        ret->n_dofs = op->rows();
        ret->set_op(op);

        CUDA_BLAS<T>::build_blas(ret->blas);
        ret->ensure_power_method();
        ret->execution_space_ = EXECUTION_SPACE_DEVICE;
        return ret;
    }

    template <typename T>
    std::shared_ptr<Multigrid<T>> d_mg() {
        auto mg = std::make_shared<Multigrid<T>>();
        CUDA_BLAS<T>::build_blas(mg->blas());
        mg->execution_space_ = EXECUTION_SPACE_DEVICE;
        return mg;
    }
    
}  // namespace sfem

#endif
