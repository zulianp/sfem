// Implementation of cvfem_ss_galerkin_api.hpp. Kept in its own translation unit so the
// CVFEM element headers, which define file-scope names a driver also defines, stay out of
// the driver's compilation.

#include "cvfem_ss_galerkin_api.hpp"

#include "cvfem_ss_galerkin.hpp"

#include "smesh_buffer.hpp"
#include "smesh_semistructured.hpp"

#include <algorithm>

namespace cvfem_ss {

    std::shared_ptr<CoarseBSR> assemble_coarse_operator(const sfem::CVFEMNavierStokes              &op,
                                                        const std::shared_ptr<sfem::FunctionSpace> &coarse,
                                                        const std::shared_ptr<sfem::FunctionSpace> &fine,
                                                        std::vector<real_t> *const                  diag_out,
                                                        const uint8_t *const fine_constrained) {
        const ::SSMeshData *const ss = op.semi_structured_data();
        if (!ss) SFEM_ERROR("assemble_coarse_operator: the operator is not semi-structured\n");

        const int Lf = ss->level;
        const int Lc = coarse->has_semi_structured_mesh() ? smesh::semistructured_level(coarse->mesh()) : 1;
        if (Lc < 1 || Lf % Lc) SFEM_ERROR("assemble_coarse_operator: level %d does not divide %d\n", Lc, Lf);

        GalerkinLevel gl;
        galerkin_init(*ss, Lf / Lc, gl);
        galerkin_gid_from_spaces(coarse, fine, gl);
        if (fine_constrained)
            gl.fine_constrained.assign(fine_constrained, fine_constrained + fine->n_dofs());
        std::vector<sfem::count_t> rowptr;
        std::vector<sfem::idx_t>   colidx;
        galerkin_build_pattern(gl, rowptr, colidx);

        std::vector<ptrdiff_t> pos;
        galerkin_build_scatter(gl, rowptr, colidx, pos);
        const ptrdiff_t nblocks = (ptrdiff_t)colidx.size();

        // Assemble and reduce a chunk of macro-elements at a time. Chunk order is fixed and
        // each block sums its own sources, so the result is the same bits on any thread count.
        std::vector<scalar_t>  acc((size_t)nblocks * 16, scalar_t(0));
        std::vector<ptrdiff_t> iptr, iidx;
        const ptrdiff_t        step   = galerkin_chunk(gl);
        const size_t           stride = (size_t)gl.nc * 27;
        for (ptrdiff_t e0 = 0; e0 < gl.nmacro; e0 += step) {
            const ptrdiff_t e1 = std::min(gl.nmacro, e0 + step);
            galerkin_assemble(*ss, (scalar_t)op.rho, (scalar_t)op.mu, gl, e0, e1);
            galerkin_build_inverse(pos, nblocks, (size_t)e0 * stride, (size_t)e1 * stride, iptr, iidx);
            galerkin_accumulate(gl, iptr, iidx, acc.data());
        }

        auto rp = smesh::create_host_buffer<sfem::count_t>(rowptr.size());
        auto ci = smesh::create_host_buffer<sfem::idx_t>(colidx.size());
        auto va = smesh::create_host_buffer<real_t>((size_t)nblocks * 16);
        std::copy(rowptr.begin(), rowptr.end(), rp->data());
        std::copy(colidx.begin(), colidx.end(), ci->data());
        std::copy(acc.begin(), acc.end(), va->data());

        if (diag_out) {
            diag_out->assign((size_t)gl.n_coarse * 16, real_t(0));
            for (ptrdiff_t i = 0; i < gl.n_coarse; ++i)
                for (sfem::count_t a = rp->data()[i]; a < rp->data()[i + 1]; ++a)
                    if (ci->data()[a] == (sfem::idx_t)i)
                        std::copy(va->data() + (size_t)a * 16, va->data() + (size_t)a * 16 + 16,
                                  diag_out->data() + (size_t)i * 16);
        }

        return sfem::h_bsr_spmv<sfem::count_t, sfem::idx_t, real_t, real_t>(gl.n_coarse, gl.n_coarse, N_FIELDS, rp,
                                                                            ci, va, real_t(0));
    }

    namespace {
        // Assemble one level's element matrices into a BSR. Shared by both entry points.
        std::shared_ptr<CoarseBSR> level_to_bsr(const GalerkinLevel &gl) {
            std::vector<sfem::count_t> rowptr;
            std::vector<sfem::idx_t>   colidx;
            galerkin_build_pattern(gl, rowptr, colidx);

            std::vector<ptrdiff_t> pos, iptr, iidx;
            galerkin_build_scatter(gl, rowptr, colidx, pos);
            const ptrdiff_t nblocks = (ptrdiff_t)colidx.size();
            galerkin_build_inverse(pos, nblocks, 0, pos.size(), iptr, iidx);

            std::vector<scalar_t> acc((size_t)nblocks * 16, scalar_t(0));
            galerkin_accumulate(gl, iptr, iidx, acc.data());

            auto rp = smesh::create_host_buffer<sfem::count_t>(rowptr.size());
            auto ci = smesh::create_host_buffer<sfem::idx_t>(colidx.size());
            auto va = smesh::create_host_buffer<real_t>((size_t)nblocks * 16);
            std::copy(rowptr.begin(), rowptr.end(), rp->data());
            std::copy(colidx.begin(), colidx.end(), ci->data());
            std::copy(acc.begin(), acc.end(), va->data());

            return sfem::h_bsr_spmv<sfem::count_t, sfem::idx_t, real_t, real_t>(gl.n_coarse, gl.n_coarse,
                                                                                N_FIELDS, rp, ci, va, real_t(0));
        }
    }  // namespace

    CoarseHierarchy assemble_hierarchy(sfem::CVFEMNavierStokes                                 &op,
                                       const real_t *const                                      state,
                                       const std::vector<std::shared_ptr<sfem::FunctionSpace>> &spaces,
                                       const std::vector<std::vector<uint8_t>>                 &masks) {
        SFEM_TRACE_SCOPE("cvfem_ss::assemble_hierarchy");
        if (state) op.update(state);
        const ::SSMeshData *const ss = op.semi_structured_data();
        if (!ss) SFEM_ERROR("assemble_hierarchy: the operator is not semi-structured\n");

        const int nlevels = (int)spaces.size();
        CoarseHierarchy out;
        out.A.assign((size_t)nlevels, nullptr);
        if (nlevels < 2) return out;

        // Level 1, straight from the micro-cell matrices, with the fine mask on the columns.
        GalerkinLevel cur;
        {
            const int Lf = ss->level;
            const int Lc = spaces[1]->has_semi_structured_mesh()
                                   ? smesh::semistructured_level(spaces[1]->mesh())
                                   : 1;
            if (Lc < 1 || Lf % Lc) SFEM_ERROR("assemble_hierarchy: level %d does not divide %d\n", Lc, Lf);
            galerkin_init(*ss, Lf / Lc, cur);
            galerkin_gid_from_spaces(spaces[1], spaces[0], cur);
            cur.fine_constrained = masks[0];
            galerkin_assemble(*ss, (scalar_t)op.rho, (scalar_t)op.mu, cur, 0, cur.nmacro);
        }
        out.A[1] = level_to_bsr(cur);

        // Each level below is one element-local hop, masking the level above's columns.
        for (int i = 2; i < nlevels; ++i) {
            GalerkinLevel nxt;
            nxt.Lc = spaces[i]->has_semi_structured_mesh() ? smesh::semistructured_level(spaces[i]->mesh()) : 1;
            nxt.q  = cur.Lc / nxt.Lc;
            nxt.nc = (nxt.Lc + 1) * (nxt.Lc + 1) * (nxt.Lc + 1);
            nxt.nmacro = cur.nmacro;
            galerkin_gid_from_spaces(spaces[i], spaces[0], nxt);

            galerkin_hop(cur, masks[(size_t)i - 1].empty() ? nullptr : masks[(size_t)i - 1].data(), nxt);
            out.A[i] = level_to_bsr(nxt);
            cur      = std::move(nxt);
        }

        return out;
    }

    std::shared_ptr<sfem::Operator<real_t>> make_element_matrix_level(
            const sfem::CVFEMNavierStokes              &op,
            const std::shared_ptr<sfem::FunctionSpace> &coarse,
            const std::shared_ptr<sfem::FunctionSpace> &fine,
            std::vector<real_t> *const                  diag_out,
            const uint8_t *const                        fine_constrained,
            const uint8_t *const                        coarse_constrained) {
        const ::SSMeshData *const ss = op.semi_structured_data();
        if (!ss) SFEM_ERROR("make_element_matrix_level: the operator is not semi-structured\n");

        const int Lf = ss->level;
        const int Lc = coarse->has_semi_structured_mesh() ? smesh::semistructured_level(coarse->mesh()) : 1;
        if (Lc < 1 || Lf % Lc) SFEM_ERROR("make_element_matrix_level: level %d does not divide %d\n", Lc, Lf);

        // Held for the operator's lifetime rather than chunked: this IS the level's storage,
        // not a transient on the way to a matrix. The staging and temporary buffers are kept
        // with it so an apply does not reallocate n_coarse * 4 on every call, which a Krylov
        // smoother would pay hundreds of times per Newton step. That makes a single apply
        // non-reentrant -- it is parallel inside, but two concurrent applies of the same
        // level would share these buffers -- which matches how the cycle uses it.
        struct State {
            GalerkinLevel         gl;
            GalerkinReduce        red;
            std::vector<scalar_t> stage;
            std::vector<scalar_t> tmp;
            std::vector<uint8_t>  cmask;
        };
        auto st = std::make_shared<State>();

        galerkin_init(*ss, Lf / Lc, st->gl);
        galerkin_gid_from_spaces(coarse, fine, st->gl);
        if (fine_constrained)
            st->gl.fine_constrained.assign(fine_constrained, fine_constrained + fine->n_dofs());
        galerkin_assemble(*ss, (scalar_t)op.rho, (scalar_t)op.mu, st->gl, 0, st->gl.nmacro);
        galerkin_build_node_reduce(st->gl, st->red);

        const ptrdiff_t ndc = st->gl.n_coarse * N_FIELDS;
        st->tmp.assign((size_t)ndc, scalar_t(0));
        st->cmask.assign((size_t)ndc, 0);
        if (coarse_constrained) std::copy(coarse_constrained, coarse_constrained + ndc, st->cmask.begin());

        if (diag_out) {
            diag_out->assign((size_t)st->gl.n_coarse * 16, real_t(0));
            std::vector<scalar_t> d((size_t)st->gl.n_coarse * 16, scalar_t(0));
            galerkin_block_diag(st->gl, st->red, d.data());
            // A constrained row's diagonal must be the identity the smoother expects, the
            // same value patch_identity_rows leaves in the assembled form.
            for (ptrdiff_t n = 0; n < st->gl.n_coarse; ++n)
                for (int c = 0; c < N_FIELDS; ++c)
                    if (st->cmask[(size_t)n * N_FIELDS + (size_t)c])
                        for (int k = 0; k < N_FIELDS; ++k) {
                            d[(size_t)n * 16 + (size_t)(c * 4 + k)] = (k == c) ? scalar_t(1) : scalar_t(0);
                            d[(size_t)n * 16 + (size_t)(k * 4 + c)] = (k == c) ? scalar_t(1) : scalar_t(0);
                        }
            std::copy(d.begin(), d.end(), diag_out->data());
        }

        return sfem::make_op<real_t>(
                ndc, ndc,
                [st, ndc](const real_t *const x, real_t *const y) {
                    galerkin_apply(st->gl, st->red, st->stage, x, st->tmp.data());
                    for (ptrdiff_t k = 0; k < ndc; ++k)
                        y[(size_t)k] += st->cmask[(size_t)k] ? x[(size_t)k] : st->tmp[(size_t)k];
                },
                sfem::EXECUTION_SPACE_HOST);
    }

}  // namespace cvfem_ss
