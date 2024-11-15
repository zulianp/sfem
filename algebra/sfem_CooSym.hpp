#ifndef SFEM_COO_SYM_HPP
#define SFEM_COO_SYM_HPP

#include <cstddef>
#include <memory>

#include "sfem_Buffer.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_config.h"
#include "sfem_openmp_blas.hpp"
#include "sparse.h"

// This class might be better off as just a sparse matrix, but the coarsen method is an optimized
// version of the matrix triple product ptap and transposing is basically a NOP
namespace sfem {
    template <typename R, typename T>
    class CooSymSpMV final : public Operator<T> {
    public:
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        ptrdiff_t ndofs{-1};
        BLAS_Tpl<T> blas;

        std::shared_ptr<Buffer<R>> offdiag_rowidx;
        std::shared_ptr<Buffer<R>> offdiag_colidx;
        std::shared_ptr<Buffer<T>> values;
        std::shared_ptr<Buffer<T>> diag_values;

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        /* Operator */
        int apply(const T* const x, T* const y) override {
            coo_symm_spmv(x, y);
            return 0;
        }
        inline std::ptrdiff_t rows() const override { return ndofs; }
        inline std::ptrdiff_t cols() const override { return ndofs; }
        ExecutionSpace execution_space() const override { return execution_space_; }

        /* Sparse Matrix-Vector Multiplication using SymmCOO format */
        int coo_symm_spmv_(const T* const x, T* const y) {
            R* offdiag_row_indices = offdiag_rowidx->data();
            R* offdiag_col_indices = offdiag_colidx->data();
            T* offdiag_values = values->data();
            T* diag = diag_values->data();
            count_t offdiag_nnz = values->size();

#pragma omp parallel for
            for (R k = 0; k < ndofs; k++) {
                y[k] = diag[k] * x[k];
            }

            for (R k = 0; k < offdiag_nnz; k++) {
                R i = offdiag_row_indices[k];
                R j = offdiag_col_indices[k];
                T val = offdiag_values[k];
                y[i] += x[j] * val;
                y[j] += x[i] * val;
            }
        }
    };

    template <typename R, typename T>
    std::shared_ptr<CooSymSpMV<R, T>> h_coosym(const std::shared_ptr<Buffer<R>>& offdiag_rowidx,
                                               const std::shared_ptr<Buffer<R>>& offdiag_colidx,
                                               const std::shared_ptr<Buffer<T>>& values,
                                               const std::shared_ptr<Buffer<T>>& diag_values) {
        auto ret = std::make_shared<CooSymSpMV<R, T>>();
        ret->offdiag_rowidx = offdiag_rowidx;
        ret->offdiag_colidx = offdiag_colidx;
        ret->values = values;
        ret->diag_values = diag_values;
        ret->ndofs = diag_values->size();
        ret->default_init();
        return ret;
    }

}  // namespace sfem

#endif  // SFEM_COO_SYM_HPP
