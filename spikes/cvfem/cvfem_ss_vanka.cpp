// Implementation of the diagonal Vanka smoother declared in cvfem_ss_galerkin_api.hpp.
// Separate translation unit for the same reason the operator has one: the CVFEM element
// headers define file-scope names a driver also defines.

#include "cvfem_ss_galerkin_api.hpp"

#include "cvfem_ss_vanka.hpp"

#include <memory>
#include <vector>

#include "smesh_env.hpp"

namespace cvfem_ss {

    std::shared_ptr<sfem::Operator<real_t>> make_diagonal_vanka(
            sfem::CVFEMNavierStokes &op, const std::shared_ptr<sfem::FunctionSpace> &space,
            const real_t *const state, const uint8_t *const constrained, const real_t omega) {
        if (state) op.update(state);
        const ::SSMeshData *const ss = op.semi_structured_data();
        if (!ss) SFEM_ERROR("make_diagonal_vanka: the operator is not semi-structured\n");

        struct State {
            const ::SSMeshData        *ss{nullptr};
            std::shared_ptr<CoarseBSR> A;
            VankaData                  v;
            scalar_t                   omega{1};
            bool                       mult{true};
            std::vector<scalar_t>      work;
        };
        auto st   = std::make_shared<State>();
        st->ss    = ss;
        st->omega = (scalar_t)omega;
        // Multiplicative by default: the additive form measured 0.883 against block-Jacobi's
        // 0.916, which is not worth having. SFEM_VANKA_MULT=0 selects additive.
        st->mult = smesh::Env::read<int>("SFEM_VANKA_MULT", 1) != 0;

        // Assemble the fine operator through the element-wise Galerkin path at q = 1, where
        // the prolongation is the identity and P^T A P is A itself. That path's identity gate
        // matches the matrix-free apply at 1.54e-16, so these are the operator's own entries.
        auto A = assemble_coarse_operator(op, space, space, nullptr, constrained);
        st->A  = A;  // keep alive; the patch factorisations were built from it

        vanka_setup(*ss, constrained, A->row_ptr->data(), A->col_idx->data(), A->values->data(), st->v);

        const ptrdiff_t ndof = ss->nnodes * N_FIELDS;
        return sfem::make_op<real_t>(
                ndof, ndof,
                [st](const real_t *const r, real_t *const y) {
                    if (st->mult) vanka_apply_mult(*st->ss, st->v, st->omega, r, y, st->work);
                    else          vanka_apply(*st->ss, st->v, st->omega, r, y, st->work);
                },
                sfem::EXECUTION_SPACE_HOST);
    }

    std::shared_ptr<sfem::Operator<real_t>> make_diagonal_vanka_from_bsr(
            sfem::CVFEMNavierStokes &op, const std::shared_ptr<CoarseBSR> &A,
            const uint8_t *const constrained, const real_t omega) {
        const ::SSMeshData *const ss = op.semi_structured_data();
        if (!ss || !A) return nullptr;

        struct State {
            const ::SSMeshData        *ss{nullptr};
            std::shared_ptr<CoarseBSR> A;
            VankaData                  v;
            scalar_t                   omega{1};
            bool                       mult{true};
            std::vector<scalar_t>      work;
        };
        auto st   = std::make_shared<State>();
        st->ss    = ss;
        st->A     = A;
        st->omega = (scalar_t)omega;
        st->mult  = smesh::Env::read<int>("SFEM_VANKA_MULT", 1) != 0;

        vanka_setup(*ss, constrained, A->row_ptr->data(), A->col_idx->data(), A->values->data(), st->v);

        const ptrdiff_t ndof = ss->nnodes * N_FIELDS;
        return sfem::make_op<real_t>(
                ndof, ndof,
                [st](const real_t *const r, real_t *const y) {
                    if (st->mult) vanka_apply_mult(*st->ss, st->v, st->omega, r, y, st->work);
                    else          vanka_apply(*st->ss, st->v, st->omega, r, y, st->work);
                },
                sfem::EXECUTION_SPACE_HOST);
    }

}  // namespace cvfem_ss
