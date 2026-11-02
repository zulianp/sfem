#ifndef SFEM_SHIFTABLE_JACOBI_HPP
#define SFEM_SHIFTABLE_JACOBI_HPP

#include <cstddef>
#include <memory>

#include "sfem_aliases.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_BLAS_openmp.hpp"
#include "sfem_TeamOperator.hpp"

#include "sfem_ShiftableBlockSymJacobiKernels_openmp.hpp"
#include "smesh_env.hpp"

namespace sfem {

    template <typename T>
    class ShiftableJacobi final : public ShiftableOperator<T>, public TeamOperator<T> {
    public:
        /// For the execution space es, on the diagonal diag.
        static std::shared_ptr<ShiftableJacobi<T>> create(const SharedBuffer<T> &diag, const ExecutionSpace es);

        std::shared_ptr<BLAS<T>> blas;

        SharedBuffer<T> diag;
        SharedBuffer<T> inv_diag;
        T                          relaxation_parameter{0.3};

        void default_init() {
            blas = OpenMPBLAS<T>::create();
        }

        void set_relaxation_parameter(const T relaxation) {
            if (inv_diag && relaxation_parameter != T(0)) {
                blas->scal(inv_diag->size(), relaxation / relaxation_parameter, inv_diag->data());
            }

            relaxation_parameter = relaxation;
        }

        void set_diag(const SharedBuffer<T>& d) {
            diag     = d;
            inv_diag = create_buffer<T>(d->size(), this->execution_space());
            blas->copy(diag->size(), diag->data(), inv_diag->data());
            blas->reciprocal(inv_diag->size(), relaxation_parameter, inv_diag->data());
        }

        int shift(const SharedBuffer<T>& d) override {
            assert(d->size() == diag->size());

            blas->copy(diag->size(), diag->data(), inv_diag->data());
            blas->axpy(diag->size(), 1, d->data(), inv_diag->data());
            blas->reciprocal(inv_diag->size(), relaxation_parameter, inv_diag->data());
            return SFEM_SUCCESS;
        }

        /* Operator */
        int apply(const T* const b, T* const x) override {
            SFEM_TRACE_SCOPE("ShiftableJacobi::apply");
            if (team_capable()) return team_region_apply<T>(*this, b, x);
            blas->xypaz(inv_diag->size(), inv_diag->data(), b, 1, x);
            return SFEM_SUCCESS;
        }

        /// x += D^{-1} b on the calling thread's rows, as xypaz computes it: it reads b there only.
        void team_apply(const Team& team, const T* const b, T* const x) const override {
            const T* const d = inv_diag->data();
            for (ptrdiff_t i = team.begin; i < team.end; i++) x[i] = d[i] * b[i] + 1 * x[i];
        }
        Team team() const override { return Team::current(rows()); }
        bool input_in_share() const override { return true; }
        bool team_capable() const override { return this->execution_space() == EXECUTION_SPACE_HOST; }

        inline std::ptrdiff_t rows() const override { return diag->size(); }
        inline std::ptrdiff_t cols() const override { return diag->size(); }
        ExecutionContext      execution_context() const override { return blas ? blas->context() : ExecutionContext(EXECUTION_SPACE_INVALID); }
    };

    template <typename T>
    std::shared_ptr<ShiftableJacobi<T>> h_shiftable_jacobi(const SharedBuffer<T>& diag) {
        auto ret = std::make_shared<ShiftableJacobi<T>>();
        ret->default_init();
        ret->set_diag(diag);
        return ret;
    }

    template <typename T>
    class ShiftableBlockSymJacobi final : public ShiftableOperator<T>, public TeamOperator<T> {
    public:
        /// In the execution context, on the dim x dim symmetric diagonal blocks diag.
        static std::shared_ptr<ShiftableBlockSymJacobi<T>> create(const int                   dim,
                                                                  const SharedBuffer<T>      &diag,
                                                                  const SharedBuffer<mask_t> &constraints_mask,
                                                                  const ExecutionContext     &context);

        std::shared_ptr<BLAS<T>> blas;
        ShiftableBlockSymJacobiKernels<T> impl;

        SharedBuffer<T>      diag;
        SharedBuffer<T>      inv_diag;
        SharedBuffer<mask_t> constraints_mask;
        T                               relaxation_parameter{1./3};
        int                             block_size{3};
        bool                            is_symmetric{true};
        // SFEM_JACOBI_SPARSE_UPDATE=0 forces full inv_diag rebuild (debug / A-B vs sparse path).
        bool                            enable_sparse_update_{true};
        bool                            sparse_update_policy_logged_{false};

        void default_init() {
            blas = OpenMPBLAS<T>::create();
            OpenMPShiftableBlockSymJacobiKernels<T>::build(block_size, impl);
        }

        void set_relaxation_parameter(const T relaxation) {
            if (inv_diag && relaxation_parameter != T(0)) {
                blas->scal(inv_diag->size(), relaxation / relaxation_parameter, inv_diag->data());
            }

            relaxation_parameter = relaxation;
        }

        void read_sparse_update_policy() {
            enable_sparse_update_ = smesh::Env::read<bool>("SFEM_JACOBI_SPARSE_UPDATE", true);
            if (!sparse_update_policy_logged_) {
                static bool logged_once = false;
                if (!logged_once) {
                    printf("ShiftableBlockSymJacobi: SFEM_JACOBI_SPARSE_UPDATE=%d\n", (int)enable_sparse_update_);
                    logged_once = true;
                }
                sparse_update_policy_logged_ = true;
            }
        }

        void set_diag(const SharedBuffer<T>& d) {
            SFEM_TRACE_SCOPE("ShiftableBlockSymJacobi::set_diag");

            assert(block_size == 3);
            assert(is_symmetric);
            assert(this->execution_space() == (enum ExecutionSpace)d->mem_space());
            assert(constraints_mask);

            read_sparse_update_policy();

            const ptrdiff_t n_blocks = d->size() / 6;
            diag                     = d;
            inv_diag                 = create_buffer<T>(n_blocks * block_size * block_size, this->execution_space());
            impl.sym_diag_to_diag(n_blocks, diag->data(), inv_diag->data());
            impl.apply_mask(n_blocks, constraints_mask->data(), inv_diag->data());
            impl.inplace_invert(n_blocks, inv_diag->data());
            blas->scal(inv_diag->size(), relaxation_parameter, inv_diag->data());
        }

        int shift(const SharedBuffer<T>& d) override {
            SFEM_TRACE_SCOPE("ShiftableBlockSymJacobi::shift");

            assert(d->size() == diag->size());

            const ptrdiff_t n_blocks = d->size() / 6;
            impl.sym_diag_to_diag(n_blocks, diag->data(), inv_diag->data());
            impl.add_sym_diag_to_diag(n_blocks, d->data(), inv_diag->data());
            impl.apply_mask(n_blocks, constraints_mask->data(), inv_diag->data());
            impl.inplace_invert(n_blocks, inv_diag->data());
            blas->scal(inv_diag->size(), relaxation_parameter, inv_diag->data());
            return SFEM_SUCCESS;
        }

        int shift(const std::shared_ptr<SparseBlockVector<T>>& block_diag, const SharedBuffer<T>& scaling) override {
            SFEM_TRACE_SCOPE("ShiftableBlockSymJacobi::shift");

            const ptrdiff_t n_blocks = inv_diag->size() / 9;
            const ptrdiff_t n_sparse = block_diag->idx()->size();

            if (enable_sparse_update_ && impl.sparse_update_inv_diag) {
                impl.sparse_update_inv_diag(n_sparse,
                                            block_diag->idx()->data(),
                                            diag->data(),
                                            block_diag->data()->data(),
                                            scaling->data(),
                                            constraints_mask->data(),
                                            relaxation_parameter,
                                            inv_diag->data());
                return SFEM_SUCCESS;
            }

            // Full rebuild path (pre-28487044 behavior / SFEM_JACOBI_SPARSE_UPDATE=0).
            impl.sym_diag_to_diag(n_blocks, diag->data(), inv_diag->data());
            impl.add_sparse_sym_diag_to_diag(
                    n_sparse, block_diag->idx()->data(), block_diag->data()->data(), scaling->data(), inv_diag->data());

            impl.apply_mask(n_blocks, constraints_mask->data(), inv_diag->data());
            impl.inplace_invert(n_blocks, inv_diag->data());
            blas->scal(inv_diag->size(), relaxation_parameter, inv_diag->data());
            return SFEM_SUCCESS;
        }

        /* Operator */
        int apply(const T* const x, T* const y) override {
            SFEM_TRACE_SCOPE("ShiftableBlockSymJacobi::apply");
            if (team_capable()) return team_region_apply<T>(*this, x, y);
            const ptrdiff_t n_blocks = inv_diag->size() / (block_size * block_size);
            impl.apply(n_blocks, inv_diag->data(), x, y);
            return SFEM_SUCCESS;
        }

        /// y += D^{-1} x on the calling thread's rows, whole blocks (team chunks start at multiples
        /// of 12 entries), as the OpenMP kernel computes it: it reads x there only.
        void team_apply(const Team& team, const T* const x, T* const y) const override {
            static_assert(team_chunk_align % 3 == 0, "a team share is whole blocks of 3");
            const T* const inv = inv_diag->data();
            for (ptrdiff_t i = team.begin / 3; i < team.end / 3; i++) {
                const T* const xi = &x[i * 3];
                T* const       yi = &y[i * 3];
                const T* const di = &inv[i * 3 * 3];

                for (int d1 = 0; d1 < 3; d1++) {
                    for (int d2 = 0; d2 < 3; d2++) {
                        yi[d1] += di[d1 * 3 + d2] * xi[d2];
                    }
                }
            }
        }
        Team team() const override { return Team::current(inv_diag->size() / block_size); }
        bool input_in_share() const override { return true; }
        bool team_capable() const override { return this->execution_space() == EXECUTION_SPACE_HOST && block_size == 3; }

        inline std::ptrdiff_t rows() const override { return diag->size(); }
        inline std::ptrdiff_t cols() const override { return diag->size(); }
        ExecutionContext      execution_context() const override { return blas ? blas->context() : ExecutionContext(EXECUTION_SPACE_INVALID); }
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
    template <typename T>
    std::shared_ptr<ShiftableJacobi<T>> create_shiftable_jacobi(const SharedBuffer<T> &diag, const ExecutionSpace es) {
        return ShiftableJacobi<T>::create(diag, es);
    }

    template <typename T>
    std::shared_ptr<ShiftableBlockSymJacobi<T>> create_shiftable_block_sym_jacobi(const int                   dim,
                                                                                  const SharedBuffer<T>      &diag,
                                                                                  const SharedBuffer<mask_t> &constraints_mask,
                                                                                  const ExecutionContext     &context) {
        return ShiftableBlockSymJacobi<T>::create(dim, diag, constraints_mask, context);
    }

}  // namespace sfem

#endif  // SFEM_SHIFTABLE_JACOBI_HPP
