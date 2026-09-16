// Implementation of the diagonal Vanka smoother declared in cvfem_ss_galerkin_api.hpp.
// Separate translation unit for the same reason the operator has one: the CVFEM element
// headers define file-scope names a driver also defines.

#include "cvfem_ss_galerkin_api.hpp"

#include "cvfem_ss_vanka.hpp"

#include <cmath>
#include <cstdio>
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
                                                              const real_t                      omega,
                                                              const std::shared_ptr<FineStencil> &sten = nullptr) {
            struct State {
                const ::SSMeshData          *ss{nullptr};
                std::shared_ptr<CoarseBSR>   A;
                std::shared_ptr<FineStencil> sten;
                VankaDataT<T>                v;
                scalar_t                     omega{1};
                bool                         mult{true};
                std::vector<scalar_t>        work;
            };
            auto st   = std::make_shared<State>();
            st->ss    = ss;
            st->A     = A;     // keep alive; the patch factorisations were built from it
            st->sten  = sten;  // the same, when the smoother was built from stencils
            st->omega = (scalar_t)omega;
            if (sten) {
                st->v.nc = sten->nc;
                if (std::is_same<T, float>::value)
                    st->v.sten = reinterpret_cast<const T *>(sten->vf.data());
                else
                    st->v.sten = reinterpret_cast<const T *>(sten->vd.data());
            }
            // Multiplicative by default: the additive form measured 0.883 against block-Jacobi's
            // 0.916, which is not worth having. SFEM_VANKA_MULT=0 selects additive.
            st->mult = smesh::Env::read<int>("SFEM_VANKA_MULT", 1) != 0;

            vanka_setup(*ss, constrained, A ? A->row_ptr->data() : nullptr, A ? A->col_idx->data() : nullptr,
                        A ? A->values->data() : nullptr, st->v);

            const ptrdiff_t ndof = ss->nnodes * N_FIELDS;
            return sfem::make_op<real_t>(
                    ndof, ndof,
                    [st](const real_t *const r, real_t *const y) {
                        if (st->mult) vanka_apply_mult(*st->ss, st->v, st->omega, r, y, st->work);
                        else          vanka_apply(*st->ss, st->v, st->omega, r, y, st->work);
                    },
                    sfem::EXECUTION_SPACE_HOST);
        }

        std::shared_ptr<sfem::Operator<real_t>> make_vanka_op(const ::SSMeshData *const           ss,
                                                              const std::shared_ptr<CoarseBSR>   &A,
                                                              const uint8_t *const                constrained,
                                                              const real_t                        omega,
                                                              const std::shared_ptr<FineStencil> &sten = nullptr) {
            return vanka_single_precision() ? make_vanka_op<float>(ss, A, constrained, omega, sten)
                                            : make_vanka_op<scalar_t>(ss, A, constrained, omega, sten);
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
        //
        // The patches, and the rows the multiplicative sweep walks, are read from the
        // macro-element lattice stencils that the assembly already produces. The sweep never
        // reads a column outside the macro-element it is sweeping, so the stencils carry
        // everything it uses and no global BSR is built for this level. Measured against one
        // that was, on the same state: identical iterations (617 linear at 116,212 dof, 1153 at
        // 893,924), assembly 30-32% faster, set-up 12-22% faster, 331 MB smaller at 893,924 dof.
        // The apply is the one place the stencils cost more -- they store a shared face once per
        // macro-element, 1.81x the nodes at L = 4 and 1.37x at L = 8 -- which is why the gain
        // grows with the internal level: -3.9% end to end at 893,924 dof, +2.7% at 116,212.
        std::shared_ptr<FineStencil> sten;
        {
            SFEM_TRACE_SCOPE("cvfem_ss::make_diagonal_vanka::assemble_fine_stencil");
            sten = assemble_fine_stencil(op, space, constrained, vanka_single_precision());
        }
        auto vk = make_vanka_op(ss, nullptr, constrained, omega, sten);

        // SFEM_VANKA_CHECK=1: the same smoother built the other way must apply the same
        // correction. Both are built here, so the comparison is on one state and one set of
        // factorisations rather than across runs.
        if (smesh::Env::read<int>("SFEM_VANKA_CHECK", 0)) {
            SFEM_TRACE_SCOPE("cvfem_ss::make_diagonal_vanka::check");
            auto            A   = assemble_coarse_operator(op, space, space, nullptr, constrained);
            auto            ref = make_vanka_op(ss, A, constrained, omega);
            const ptrdiff_t nd  = ss->nnodes * N_FIELDS;
            std::vector<real_t> x((size_t)nd), ys((size_t)nd, 0), yb((size_t)nd, 0);
            unsigned            s = 12345u;
            for (auto &e : x) {
                s = s * 1103515245u + 12345u;
                e = (real_t)((s >> 16) & 0x7fff) / (real_t)0x7fff - real_t(0.5);
            }
            vk->apply(x.data(), ys.data());
            ref->apply(x.data(), yb.data());
            real_t dn = 0, rn = 0;
            for (ptrdiff_t k = 0; k < nd; ++k) {
                const real_t dd = ys[(size_t)k] - yb[(size_t)k];
                dn += dd * dd;
                rn += yb[(size_t)k] * yb[(size_t)k];
            }
            const real_t rel = rn > 0 ? std::sqrt(dn / rn) : std::sqrt(dn);
            std::printf("vanka stencil vs bsr: rel |M_sten r - M_bsr r| = %.4e  %s\n", rel,
                        rel < 1e-6 ? "OK" : "MISMATCH");
        }
        return vk;
    }

    std::shared_ptr<sfem::Operator<real_t>> make_diagonal_vanka_from_bsr(
            sfem::CVFEMNavierStokes &op, const std::shared_ptr<CoarseBSR> &A,
            const uint8_t *const constrained, const real_t omega) {
        const ::SSMeshData *const ss = op.semi_structured_data();
        if (!ss || !A) return nullptr;
        return make_vanka_op(ss, A, constrained, omega);
    }

}  // namespace cvfem_ss
