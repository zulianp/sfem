#ifndef SFEM_MIXED_PRECISION_SHIFTABLE_BLOCK_SYM_JACOBI_HPP
#define SFEM_MIXED_PRECISION_SHIFTABLE_BLOCK_SYM_JACOBI_HPP

#include "sfem_base.h"

#include "sfem_Buffer.hpp"
#include "sfem_Buffer.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_openmp_blas.hpp"
#include "sfem_openmp_ShiftableJacobi.hpp"
#include "sfem_ShiftableJacobi.hpp"

namespace sfem {

    template <typename HP, typename LP>
    class MixedPrecisionShiftableBlockSymJacobi final : public ShiftableOperator<HP> {
    public:
        ExecutionSpace                  execution_space_{EXECUTION_SPACE_INVALID};
        BLAS_Tpl<LP>                    blas;
        ShiftableBlockSymJacobi_Tpl<HP, LP> impl;

        SharedBuffer<HP>     diag;
        SharedBuffer<LP>     inv_diag;
        SharedBuffer<mask_t> constraints_mask;
        HP                               relaxation_parameter{1./3};
        int                             block_size{3};
        bool                            is_symmetric{true};

        void default_init() {
            OpenMP_BLAS<LP>::build_blas(blas);
            ShiftableBlockSymJacobi_OpenMP<HP, LP>::build(block_size, impl);
            execution_space_ = EXECUTION_SPACE_HOST;
        }

        void set_diag(const SharedBuffer<HP>& d) {
            SFEM_TRACE_SCOPE("MixedPrecisionShiftableBlockSymJacobi::set_diag");

            assert(block_size == 3);
            assert(is_symmetric);
            assert(execution_space_ == (enum ExecutionSpace)d->mem_space());
            assert(constraints_mask);

            const ptrdiff_t n_blocks = d->size() / 6;
            diag                     = d;
            inv_diag                 = create_buffer<LP>(n_blocks * block_size * block_size, execution_space());
            impl.sym_diag_to_diag(n_blocks, diag->data(), inv_diag->data());
            impl.inplace_invert(n_blocks, inv_diag->data());
            blas.scal(inv_diag->size(), relaxation_parameter, inv_diag->data());
        }

        int shift(const SharedBuffer<HP>& d) override {
            SFEM_TRACE_SCOPE("MixedPrecisionShiftableBlockSymJacobi::shift");

            assert(d->size() == diag->size());

            const ptrdiff_t n_blocks = d->size() / 6;
            impl.sym_diag_to_diag(n_blocks, diag->data(), inv_diag->data());
            impl.add_sym_diag_to_diag(n_blocks, d->data(), inv_diag->data());
            impl.apply_mask(n_blocks, constraints_mask->data(), inv_diag->data());
            impl.inplace_invert(n_blocks, inv_diag->data());
            blas.scal(inv_diag->size(), relaxation_parameter, inv_diag->data());
            return SFEM_SUCCESS;
        }

        int shift(const std::shared_ptr<SparseBlockVector<HP>>& block_diag, const SharedBuffer<HP>& scaling) override {
            SFEM_TRACE_SCOPE("MixedPrecisionShiftableBlockSymJacobi::shift");

            const ptrdiff_t n_blocks = inv_diag->size() / 9;

            impl.sym_diag_to_diag(n_blocks, diag->data(), inv_diag->data());
            impl.add_sparse_sym_diag_to_diag(block_diag->idx()->size(),
                                             block_diag->idx()->data(),
                                             block_diag->data()->data(),
                                             scaling->data(),
                                             inv_diag->data());

            impl.apply_mask(n_blocks, constraints_mask->data(), inv_diag->data());
            impl.inplace_invert(n_blocks, inv_diag->data());
            blas.scal(inv_diag->size(), relaxation_parameter, inv_diag->data());
            return SFEM_SUCCESS;
        }

        /* Operator */
        int apply(const HP* const x, HP* const y) override {
            SFEM_TRACE_SCOPE("MixedPrecisionShiftableBlockSymJacobi::apply");
            const ptrdiff_t n_blocks = inv_diag->size() / (block_size * block_size);
            impl.apply(n_blocks, inv_diag->data(), x, y);
            return SFEM_SUCCESS;
        }

        inline std::ptrdiff_t rows() const override { return diag->size(); }
        inline std::ptrdiff_t cols() const override { return diag->size(); }
        ExecutionSpace        execution_space() const override { return execution_space_; }
    };

    template <typename HP, typename LP>
    std::shared_ptr<MixedPrecisionShiftableBlockSymJacobi<HP, LP>> h_mixed_precision_shiftable_block_sym_jacobi(
            const SharedBuffer<HP>&     diag,
            const SharedBuffer<mask_t>& constraints_mask) {
        auto ret = std::make_shared<MixedPrecisionShiftableBlockSymJacobi<HP, LP>>();
        ret->default_init();
        ret->constraints_mask = constraints_mask;
        ret->set_diag(diag);
        return ret;
    }

}  // namespace sfem

#endif  // SFEM_MIXED_PRECISION_SHIFTABLE_BLOCK_SYM_JACOBI_HPP
