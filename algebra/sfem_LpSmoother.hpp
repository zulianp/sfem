#ifndef SFEM_LP_SMOOTHER_HPP
#define SFEM_LP_SMOOTHER_HPP

#include <cstddef>
#include <memory>

#include "sfem_Buffer.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_openmp_blas.hpp"

namespace sfem {
    template <typename T>
    class LpSmoother final : public Operator<T> {
    public:
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        ptrdiff_t n_dofs{SFEM_PTRDIFF_INVALID};
        BLAS_Tpl<T> blas;
        SharedBuffer<T> inv_diag;

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        /* Operator */
        int apply(const T* const b, T* const x) override {
            blas.xypaz(n_dofs, inv_diag->data(), b, 1, x);
            return SFEM_SUCCESS;
        }
        inline std::ptrdiff_t rows() const override { return n_dofs; }
        inline std::ptrdiff_t cols() const override { return n_dofs; }
        ExecutionSpace execution_space() const override { return execution_space_; }
    };

    template <typename T>
    std::shared_ptr<LpSmoother<T>> h_lpsmoother(const SharedBuffer<T>& inv_diag) {
        auto ret = std::make_shared<LpSmoother<T>>();
        ret->n_dofs = inv_diag->size();
        ret->inv_diag = inv_diag;
        ret->default_init();
        return ret;
    }
}  // namespace sfem

#endif  // SFEM_LP_SMOOTHER_HPP
