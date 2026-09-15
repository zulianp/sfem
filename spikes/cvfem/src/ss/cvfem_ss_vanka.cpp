// Implementation of the diagonal Vanka smoother declared in cvfem_ss_galerkin_api.hpp.
// Separate translation unit for the same reason the operator has one: the CVFEM element
// headers define file-scope names a driver also defines.

#include "cvfem_ss_galerkin_api.hpp"

#include "cvfem_ss_vanka.hpp"

#include <memory>
#include <string>
#include <vector>

#include "smesh_env.hpp"

namespace cvfem_ss {

    namespace {
        // SFEM_VANKA_PRECISION: the storage type of the patch factorisations and of the values
        // the multiplicative sweep reads. single (the default) or double; computation is double
        // either way. See VankaCellT for why single is the default.
        bool vanka_single_precision() {
            static const bool single = smesh::Env::read_string("SFEM_VANKA_PRECISION", std::string("single")) != "double";
            return single;
        }

        template <typename T>
        std::shared_ptr<sfem::Operator<real_t>> make_vanka_op(const ::SSMeshData *const        ss,
                                                              const std::shared_ptr<CoarseBSR> &A,
                                                              const uint8_t *const              constrained,
                                                              const real_t                      omega) {
            struct State {
                const ::SSMeshData        *ss{nullptr};
                std::shared_ptr<CoarseBSR> A;
                VankaDataT<T>              v;
                scalar_t                   omega{1};
                bool                       mult{true};
                std::vector<scalar_t>      work;
            };
            auto st   = std::make_shared<State>();
            st->ss    = ss;
            st->A     = A;  // keep alive; the patch factorisations were built from it
            st->omega = (scalar_t)omega;
            // Multiplicative by default: the additive form measured 0.883 against block-Jacobi's
            // 0.916, which is not worth having. SFEM_VANKA_MULT=0 selects additive.
            st->mult = smesh::Env::read<int>("SFEM_VANKA_MULT", 1) != 0;

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

        std::shared_ptr<sfem::Operator<real_t>> make_vanka_op(const ::SSMeshData *const        ss,
                                                              const std::shared_ptr<CoarseBSR> &A,
                                                              const uint8_t *const              constrained,
                                                              const real_t                      omega) {
            return vanka_single_precision() ? make_vanka_op<float>(ss, A, constrained, omega)
                                            : make_vanka_op<scalar_t>(ss, A, constrained, omega);
        }
    }  // namespace

    std::shared_ptr<sfem::Operator<real_t>> make_diagonal_vanka(
            sfem::CVFEMNavierStokes &op, const std::shared_ptr<sfem::FunctionSpace> &space,
            const real_t *const state, const uint8_t *const constrained, const real_t omega) {
        if (state) op.update(state);
        const ::SSMeshData *const ss = op.semi_structured_data();
        if (!ss) SFEM_ERROR("make_diagonal_vanka: the operator is not semi-structured\n");

        // Assemble the fine operator through the element-wise Galerkin path at q = 1, where
        // the prolongation is the identity and P^T A P is A itself. That path's identity gate
        // matches the matrix-free apply at 1.54e-16, so these are the operator's own entries.
        std::shared_ptr<CoarseBSR> A;
        {
            SFEM_TRACE_SCOPE("cvfem_ss::make_diagonal_vanka::assemble_fine");
            A = assemble_coarse_operator(op, space, space, nullptr, constrained);
        }
        return make_vanka_op(ss, A, constrained, omega);
    }

    std::shared_ptr<sfem::Operator<real_t>> make_diagonal_vanka_from_bsr(
            sfem::CVFEMNavierStokes &op, const std::shared_ptr<CoarseBSR> &A,
            const uint8_t *const constrained, const real_t omega) {
        const ::SSMeshData *const ss = op.semi_structured_data();
        if (!ss || !A) return nullptr;
        return make_vanka_op(ss, A, constrained, omega);
    }

}  // namespace cvfem_ss
