#ifndef SFEM_COO_SYM_HPP
#define SFEM_COO_SYM_HPP

#include <cstddef>
#include <memory>

#include "sfem_Buffer.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_config.h"
#include "sfem_mask.h"
#include "sfem_openmp_blas.hpp"

#include "sfem_Tracer.hpp"

// This class might be better off as just a sparse matrix, but the coarsen method is an optimized
// version of the matrix triple product ptap and transposing is basically a NOP
namespace sfem {
    template <typename R, typename T>
    class CooSymSpMV final : public Operator<T> {
    public:
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        std::function<void(const T* const, T* const)> apply_;
        ptrdiff_t ndofs{SFEM_PTRDIFF_INVALID};
        BLAS_Tpl<T> blas;

        SharedBuffer<mask_t> bdy_dofs;
        SharedBuffer<R> offdiag_rowidx;
        SharedBuffer<R> offdiag_colidx;
        SharedBuffer<T> values;
        SharedBuffer<T> diag_values;

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
            apply_ = [=](const T* const x, T* const y) { return coo_sym_spmv_(x, y); };
        }

        /* Operator */
        int apply(const T* const x, T* const y) override {
            SFEM_TRACE_SCOPE("CooSymSpMV::apply");

            apply_(x, y);
            return SFEM_SUCCESS;
        }
        inline std::ptrdiff_t rows() const override { return ndofs; }
        inline std::ptrdiff_t cols() const override { return ndofs; }
        ExecutionSpace execution_space() const override { return execution_space_; }

        /* Sparse Matrix-Vector Multiplication using SymmCOO format */
        int coo_sym_spmv_(const T* const SFEM_RESTRICT x, T* const SFEM_RESTRICT y) {
            const R* const offdiag_row_indices = offdiag_rowidx->data();
            const R* const offdiag_col_indices = offdiag_colidx->data();
            const T* const offdiag_values = values->data();
            const T* const diag = diag_values->data();
            const count_t offdiag_nnz = values->size();

#pragma omp parallel for
            for (R k = 0; k < ndofs; k++) {
                y[k] += diag[k] * x[k];
            }

            // #pragma omp parallel for
            for (R k = 0; k < offdiag_nnz; k++) {
                const R i = offdiag_row_indices[k];
                const R j = offdiag_col_indices[k];
                const T val = offdiag_values[k];

                // #pragma omp atomic update
                y[j] += x[i] * val;

                // #pragma omp atomic update
                y[i] += x[j] * val;
            }

            if (bdy_dofs) {
#pragma omp parallel for
                for (R k = 0; k < ndofs; k++) {
                    if (mask_get(k, bdy_dofs->data())) {
                        y[k] = x[k];
                    }
                }
            }

            return SFEM_SUCCESS;
        }
    };

    template <typename R, typename T>
    std::shared_ptr<CooSymSpMV<R, T>> h_coosym(const SharedBuffer<mask_t>& bdy_dofs,
                                               const SharedBuffer<R>& offdiag_rowidx,
                                               const SharedBuffer<R>& offdiag_colidx,
                                               const SharedBuffer<T>& values,
                                               const SharedBuffer<T>& diag_values) {
        auto ret = std::make_shared<CooSymSpMV<R, T>>();
        ret->bdy_dofs = bdy_dofs;
        ret->offdiag_rowidx = offdiag_rowidx;
        ret->offdiag_colidx = offdiag_colidx;
        ret->values = values;
        ret->diag_values = diag_values;
        ret->ndofs = diag_values->size();
        ret->default_init();
        return ret;
    }

    /* TODO refactor this conversion code to create CooSym object.
    void csr_to_symmcoo(const ptrdiff_t nrows, const count_t nnz, const count_t *row_ptr,
                        const idx_t *col_indices, const real_t *values, SymmCOOMatrix *coo) {
        assert(nnz % 2 == 0);

        count_t nweights = (nnz - nrows) / 2;
        coo->offdiag_nnz = nweights;
        coo->dim = nrows;

        idx_t k = 0;
        for (idx_t i = 0; i < nrows; i++) {
            for (idx_t idx = row_ptr[i]; idx < row_ptr[i + 1]; idx++) {
                idx_t j = col_indices[idx];
                if (j > i) {
                    coo->offdiag_row_indices[k] = i;
                    coo->offdiag_col_indices[k] = j;
                    coo->offdiag_values[k] = values[idx];
                    k += 1;
                } else if (i == j) {
                    coo->diag[i] = values[idx];
                }
            }
        }
    }
    */
}  // namespace sfem

#endif  // SFEM_COO_SYM_HPP
