#ifndef SFEM_SHIFTABLE_JACOBI_HPP
#define SFEM_SHIFTABLE_JACOBI_HPP

#include <cstddef>
#include <memory>

#include "sfem_Buffer.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_openmp_blas.hpp"

#include "sfem_Buffer.hpp"

#include "sfem_openmp_ShiftableJacobi.hpp"

namespace sfem {
    template <typename T>
    static SharedBuffer<T> create_buffer(const std::ptrdiff_t n, const MemorySpace es);

    template <typename T>
    class ShiftableJacobi final : public ShiftableOperator<T> {
    public:
        ExecutionSpace execution_space_{EXECUTION_SPACE_INVALID};
        BLAS_Tpl<T>    blas;

        SharedBuffer<T> diag;
        SharedBuffer<T> inv_diag;
        T                          relaxation_parameter{0.3};

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void set_diag(const SharedBuffer<T>& d) {
            diag     = d;
            inv_diag = create_buffer<T>(d->size(), execution_space());
            blas.copy(diag->size(), diag->data(), inv_diag->data());
            blas.reciprocal(inv_diag->size(), relaxation_parameter, inv_diag->data());
        }

        int shift(const SharedBuffer<T>& d) override {
            assert(d->size() == diag->size());

            blas.copy(diag->size(), diag->data(), inv_diag->data());
            blas.axpy(diag->size(), 1, d->data(), inv_diag->data());
            blas.reciprocal(inv_diag->size(), relaxation_parameter, inv_diag->data());
            return SFEM_SUCCESS;
        }

        /* Operator */
        int apply(const T* const b, T* const x) override {
            SFEM_TRACE_SCOPE("ShiftableJacobi::apply");

            blas.xypaz(inv_diag->size(), inv_diag->data(), b, 1, x);
            return SFEM_SUCCESS;
        }

        inline std::ptrdiff_t rows() const override { return diag->size(); }
        inline std::ptrdiff_t cols() const override { return diag->size(); }
        ExecutionSpace        execution_space() const override { return execution_space_; }
    };

    template <typename T>
    std::shared_ptr<ShiftableJacobi<T>> h_shiftable_jacobi(const SharedBuffer<T>& diag) {
        auto ret = std::make_shared<ShiftableJacobi<T>>();
        ret->default_init();
        ret->set_diag(diag);
        return ret;
    }

    template <typename T>
    class ShiftableBlockSymJacobi final : public ShiftableOperator<T> {
    public:
        ExecutionSpace                 execution_space_{EXECUTION_SPACE_INVALID};
        BLAS_Tpl<T>                    blas;
        ShiftableBlockSymJacobi_Tpl<T> impl;

        SharedBuffer<T>      diag;
        SharedBuffer<T>      inv_diag;
        SharedBuffer<mask_t> constraints_mask;
        T                               relaxation_parameter{1./3};
        int                             block_size{3};
        bool                            is_symmetric{true};

        void default_init() {
            OpenMP_BLAS<T>::build_blas(blas);
            ShiftableBlockSymJacobi_OpenMP<T>::build(block_size, impl);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void set_diag(const SharedBuffer<T>& d) {
            SFEM_TRACE_SCOPE("ShiftableBlockSymJacobi::set_diag");

            assert(block_size == 3);
            assert(is_symmetric);
            assert(execution_space_ == (enum ExecutionSpace)d->mem_space());
            assert(constraints_mask);

            const ptrdiff_t n_blocks = d->size() / 6;
            diag                     = d;
            inv_diag                 = create_buffer<T>(n_blocks * block_size * block_size, execution_space());
            impl.sym_diag_to_diag(n_blocks, diag->data(), inv_diag->data());
            impl.inplace_invert(n_blocks, inv_diag->data());
            blas.scal(inv_diag->size(), relaxation_parameter, inv_diag->data());
        }

        int shift(const SharedBuffer<T>& d) override {
            SFEM_TRACE_SCOPE("ShiftableBlockSymJacobi::shift");

            assert(d->size() == diag->size());

            const ptrdiff_t n_blocks = d->size() / 6;
            impl.sym_diag_to_diag(n_blocks, diag->data(), inv_diag->data());
            impl.add_sym_diag_to_diag(n_blocks, d->data(), inv_diag->data());
            impl.apply_mask(n_blocks, constraints_mask->data(), inv_diag->data());
            impl.inplace_invert(n_blocks, inv_diag->data());
            blas.scal(inv_diag->size(), relaxation_parameter, inv_diag->data());
            return SFEM_SUCCESS;
        }

        int shift(const std::shared_ptr<SparseBlockVector<T>>& block_diag, const SharedBuffer<T>& scaling) override {
            SFEM_TRACE_SCOPE("ShiftableBlockSymJacobi::shift");

            const ptrdiff_t n_blocks = inv_diag->size() / 9;

            impl.sym_diag_to_diag(n_blocks, diag->data(), inv_diag->data());
            impl.add_sparse_sym_diag_to_diag(
                    block_diag->idx()->size(), block_diag->idx()->data(), block_diag->data()->data(), scaling->data(), inv_diag->data());

            impl.apply_mask(n_blocks, constraints_mask->data(), inv_diag->data());
            impl.inplace_invert(n_blocks, inv_diag->data());
            blas.scal(inv_diag->size(), relaxation_parameter, inv_diag->data());
            return SFEM_SUCCESS;
        }

        /* Operator */
        int apply(const T* const x, T* const y) override {
            SFEM_TRACE_SCOPE("ShiftableBlockSymJacobi::apply");
            const ptrdiff_t n_blocks = inv_diag->size() / (block_size * block_size);
            impl.apply(n_blocks, inv_diag->data(), x, y);
            return SFEM_SUCCESS;
        }

        inline std::ptrdiff_t rows() const override { return diag->size(); }
        inline std::ptrdiff_t cols() const override { return diag->size(); }
        ExecutionSpace        execution_space() const override { return execution_space_; }
    };

    template <typename T>
    std::shared_ptr<ShiftableBlockSymJacobi<T>> h_shiftable_block_sym_jacobi(
            const SharedBuffer<T>&      diag,
            const SharedBuffer<mask_t>& constraints_mask) {
        auto ret = std::make_shared<ShiftableBlockSymJacobi<T>>();
        ret->default_init();
        ret->constraints_mask = constraints_mask;
        ret->set_diag(diag);
        return ret;
    }
}  // namespace sfem

#endif  // SFEM_SHIFTABLE_JACOBI_HPP
