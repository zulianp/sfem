#ifndef SFEM_CUDA_SOLVER_HPP
#define SFEM_CUDA_SOLVER_HPP

#include "sfem_Chebyshev3.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"
#include "sfem_cuda_blas.h"

#include <map>
#include <memory>

namespace sfem {

    template <typename T>
    std::shared_ptr<ConjugateGradient<T>> d_cg() {
        auto cg = std::make_shared<ConjugateGradient<T>>();
        cg->allocate = d_allocate;
        cg->destroy = d_destroy;
        cg->copy = d_copy;
        cg->dot = d_dot;
        cg->axpby = d_axpby;
        cg->zeros = [](const std::size_t n, T* const x) { d_memset(x, 0, n * sizeof(T)); };
        return cg;
    }

    template <typename T>
    std::shared_ptr<BiCGStab<T>> d_bcgs() {
        auto cg = std::make_shared<BiCGStab<T>>();
        cg->allocate = d_allocate;
        cg->destroy = d_destroy;
        cg->copy = d_copy;
        cg->dot = d_dot;
        cg->axpby = d_axpby;
        cg->zaxpby = d_zaxpby;
        cg->zeros = [](const std::size_t n, T* const x) { d_memset(x, 0, n * sizeof(T)); };
        return cg;
    }

    template <typename T>
    std::shared_ptr<Chebyshev3<T>> d_cheb3(const std::shared_ptr<Operator<T>>& op) {
        auto ret = std::make_shared<Chebyshev3<T>>();
        ret->n_dofs = op->rows();
        ret->set_op(op);

        ret->allocate = d_allocate;
        ret->destroy = d_destroy;
        ret->copy = d_copy;
        ret->dot = d_dot;
        ret->axpby = d_axpby;
        ret->axpy = d_axpy;
        ret->scal = d_scal;
        ret->norm2 = d_nrm2;
        ret->zeros = [](const std::size_t n, T* const x) { d_memset(x, 0, n * sizeof(T)); };
        ret->ensure_power_method();
        return ret;
    }

    template <typename T>
    std::shared_ptr<Multigrid<T>> d_mg() {
        auto mg = std::make_shared<Multigrid<T>>();
        mg->allocate = d_allocate;
        mg->destroy = d_destroy;
        mg->axpby = d_axpby;
        mg->zeros = [](const std::size_t n, T* const x) { d_memset(x, 0, n * sizeof(T)); };
        mg->norm2 = d_nrm2;
        return mg;
    }

    // template <typename T>
    // std::shared_ptr<MatrixFreeLinearSolver<T>> d_solver(const std::string& name) {
    //     using SP_t = std::shared_ptr<MatrixFreeLinearSolver<T>>;
    //     static bool initialized = false;
    //     static std::map<std::string, SP_t> factory;

    //     if (!initialized) {
    //         factory["BiCGStab"] = &d_bcgs<T>;
    //         factory["ConjugateGradient"] = &d_cg<T>;
    //         initialized = true;
    //     }

    //     auto it = factory.find(name);
    //     if (it == factory.end()) {
    //         assert(0);
    //         return d_cg<T>();
    //     }

    //     return it->second();
    // }
}  // namespace sfem

#endif
