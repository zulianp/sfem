// Implementation of cvfem_ss_galerkin_api.hpp. Kept in its own translation unit so the
// CVFEM element headers, which define file-scope names a driver also defines, stay out of
// the driver's compilation.

#include "cvfem_ss_galerkin_api.hpp"

#include "cvfem_ss_galerkin.hpp"

#include "smesh_buffer.hpp"
#include "smesh_exchange.hpp"
#include "smesh_semistructured.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace cvfem_ss {

    namespace {
        // A level's sparsity pattern, its scatter and the block -> slot map the chunked
        // inverse uses. All three depend on the level's connectivity alone -- gid, nc,
        // n_coarse -- and not on the state, yet they were rebuilt every Newton step for each
        // of the four levels the setup assembles: ~42 ms per step on the FDA nozzle at
        // 893,924 dof, for an answer that never changed. Kept here and reused when the
        // connectivity matches exactly. The comparison is the full gid array, so a different
        // mesh with the same sizes misses instead of reusing a stale pattern. The row pointers
        // and columns are shared by the matrices built on them; nothing downstream writes
        // either -- patch_identity_rows and mask_block_columns touch values only.
        // Which (macro-element, node, slot) triples denote one operator entry; see
        // build_fold_map below. Empty until a caller that needs stencils asks for it.
        struct FoldMap {
            std::vector<ptrdiff_t> ptr;  // groups + 1
            std::vector<ptrdiff_t> idx;  // triple indices, (e * nc + a) * 27 + s
        };

        struct GalerkinPattern {
            ptrdiff_t                                       nmacro{-1}, n_coarse{-1};
            int                                             nc{-1};
            std::vector<smesh::idx_t>                       gid;
            decltype(smesh::create_host_buffer<sfem::count_t>(0)) rp;
            decltype(smesh::create_host_buffer<sfem::idx_t>(0))   ci;
            std::vector<ptrdiff_t>                          pos;
            std::vector<ptrdiff_t>                          slot;  // all -1 between uses
            FoldMap                                         fold;
            bool                                            fold_built{false};
        };

        // The nodal exchange used to complete a level's lattice rows across ranks.
        //
        // Cached for the same reason galerkin_pattern is, and one more: Exchange::create_nodal
        // is COLLECTIVE, so building one per call would put a collective inside a function that
        // a freeze policy may skip on some future caller. Keyed on the mesh, because that is
        // all the exchange depends on.
        std::shared_ptr<smesh::Exchange> lattice_row_exchange(const std::shared_ptr<smesh::Mesh> &mesh) {
            static std::vector<std::pair<smesh::Mesh *, std::shared_ptr<smesh::Exchange>>> cache;
            for (auto &c : cache)
                if (c.first == mesh.get()) return c.second;
            auto ex = smesh::Exchange::create_nodal(mesh, smesh::Exchange::ExchangeScope::GhostsAndAura);
            if (cache.size() >= 8) cache.erase(cache.begin());
            cache.emplace_back(mesh.get(), ex);
            return ex;
        }

        GalerkinPattern &galerkin_pattern(const GalerkinLevel &gl) {
            static std::vector<std::unique_ptr<GalerkinPattern>> cache;
            for (auto &p : cache)
                if (p->nmacro == gl.nmacro && p->nc == gl.nc && p->n_coarse == gl.n_coarse && p->gid == gl.gid)
                    return *p;

            auto p      = std::make_unique<GalerkinPattern>();
            p->nmacro   = gl.nmacro;
            p->nc       = gl.nc;
            p->n_coarse = gl.n_coarse;
            p->gid      = gl.gid;
            std::vector<sfem::count_t> rowptr;
            std::vector<sfem::idx_t>   colidx;
            galerkin_build_pattern(gl, rowptr, colidx);
            galerkin_build_scatter(gl, rowptr, colidx, p->pos);
            p->rp = smesh::create_host_buffer<sfem::count_t>(rowptr.size());
            p->ci = smesh::create_host_buffer<sfem::idx_t>(colidx.size());
            std::copy(rowptr.begin(), rowptr.end(), p->rp->data());
            std::copy(colidx.begin(), colidx.end(), p->ci->data());
            p->slot.assign(colidx.size(), -1);

            // One process sees a handful of hierarchies at most; past eight the oldest goes.
            if (cache.size() >= 8) cache.erase(cache.begin());
            cache.push_back(std::move(p));
            return *cache.back();
        }
    }  // namespace

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
        GalerkinPattern &pat     = galerkin_pattern(gl);
        const ptrdiff_t  nblocks = (ptrdiff_t)pat.ci->size();

        // Assemble and reduce a chunk of macro-elements at a time. Chunk order is fixed and
        // each block sums its own sources, so the result is the same bits on any thread count.
        //
        // Accumulated straight into the returned value buffer. It comes from calloc, so it is
        // zero without a serial fill and its pages are first touched by the accumulating
        // threads; a separate zeroed accumulator and the copy out of it cost 37 + 64 ms per
        // Newton step on the FDA nozzle's fine level, 745 MB each way.
        auto                   rp = pat.rp;
        auto                   ci = pat.ci;
        auto                   va = smesh::create_host_buffer<real_t>((size_t)nblocks * 16);
        std::vector<ptrdiff_t> iptr, iidx, blocks;
        const ptrdiff_t        step   = galerkin_chunk(gl);
        const size_t           stride = (size_t)gl.nc * 27;
        for (ptrdiff_t e0 = 0; e0 < gl.nmacro; e0 += step) {
            const ptrdiff_t e1 = std::min(gl.nmacro, e0 + step);
            galerkin_assemble(*ss, (scalar_t)op.rho, (scalar_t)op.mu, gl, e0, e1);
            galerkin_build_inverse(pat.pos, (size_t)e0 * stride, (size_t)e1 * stride, pat.slot, blocks, iptr, iidx);
            galerkin_accumulate(gl, blocks, iptr, iidx, va->data());
        }

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
        // The fold: which (macro-element, node, slot) triples denote the same operator entry.
        //
        // galerkin_build_scatter already maps every triple to a position in the assembled BSR,
        // and two triples share that position exactly when they are the same node pair reached
        // from two macro-elements. Inverting the map therefore groups the triples that must be
        // summed. Only groups with more than one member are kept: an interior pair is assembled
        // once and needs no fold, and on the FDA nozzle at level 8 that is most of them.
        //
        // Derived from the cached pattern and cached with it, because it depends on the
        // connectivity alone.
        void build_fold_map(const GalerkinLevel &gl, GalerkinPattern &pat, FoldMap &fold) {
            SFEM_TRACE_SCOPE("cvfem_ss::build_fold_map");
            std::vector<ptrdiff_t> blocks, iptr, iidx;
            galerkin_build_inverse(pat.pos, 0, pat.pos.size(), pat.slot, blocks, iptr, iidx);

            fold.ptr.assign(1, 0);
            fold.idx.clear();
            for (size_t t = 0; t < blocks.size(); ++t) {
                const ptrdiff_t n = iptr[t + 1] - iptr[t];
                if (n < 2) continue;  // assembled once: the slot already holds the entry
                for (ptrdiff_t k = iptr[t]; k < iptr[t + 1]; ++k) fold.idx.push_back(iidx[(size_t)k]);
                fold.ptr.push_back((ptrdiff_t)fold.idx.size());
            }
            (void)gl;
        }

        // Assemble one level's element matrices into a BSR. Shared by both entry points.
        std::shared_ptr<CoarseBSR> level_to_bsr(const GalerkinLevel &gl) {
            GalerkinPattern       &pat     = galerkin_pattern(gl);
            const ptrdiff_t        nblocks = (ptrdiff_t)pat.ci->size();
            std::vector<ptrdiff_t> iptr, iidx, blocks;
            galerkin_build_inverse(pat.pos, 0, pat.pos.size(), pat.slot, blocks, iptr, iidx);

            // calloc'd, so accumulating straight into it is the same sums as into a zeroed copy.
            auto va = smesh::create_host_buffer<real_t>((size_t)nblocks * 16);
            galerkin_accumulate(gl, blocks, iptr, iidx, va->data());

            return sfem::h_bsr_spmv<sfem::count_t, sfem::idx_t, real_t, real_t>(gl.n_coarse, gl.n_coarse,
                                                                                N_FIELDS, pat.rp, pat.ci, va, real_t(0));
        }
        // A level's operator and block diagonal, backed by its element matrices. Constrained
        // rows act as identity, matching what patch_identity_rows leaves in an assembled level,
        // so the two forms are interchangeable to everything downstream.
        struct EMState {
            std::shared_ptr<GalerkinLevel> gl;
            GalerkinReduce                 red;
            std::vector<scalar_t>          stage;
            std::vector<scalar_t>          tmp;
            std::vector<uint8_t>           cmask;
        };

        std::shared_ptr<sfem::Operator<real_t>> make_em_operator(const std::shared_ptr<GalerkinLevel> &gl,
                                                                 const std::vector<uint8_t>           &cmask,
                                                                 std::vector<real_t> *const            diag_out) {
            auto st   = std::make_shared<EMState>();
            st->gl    = gl;
            st->cmask = cmask;
            galerkin_build_node_reduce(*gl, st->red);

            const ptrdiff_t ndc = gl->n_coarse * N_FIELDS;
            st->tmp.assign((size_t)ndc, scalar_t(0));
            if (st->cmask.size() != (size_t)ndc) st->cmask.assign((size_t)ndc, 0);

            if (diag_out) {
                std::vector<scalar_t> d((size_t)gl->n_coarse * 16, scalar_t(0));
                galerkin_block_diag(*gl, st->red, d.data());
                for (ptrdiff_t n = 0; n < gl->n_coarse; ++n)
                    for (int c = 0; c < N_FIELDS; ++c)
                        if (st->cmask[(size_t)n * N_FIELDS + (size_t)c])
                            // Rows only. patch_identity_rows replaces the constrained *row*
                            // and leaves the column alone; also clearing the column zeroes
                            // real off-diagonal entries of the diagonal block, which shows up
                            // as a 7.3e-3 disagreement with the reference diagonal.
                            for (int k = 0; k < N_FIELDS; ++k)
                                d[(size_t)n * 16 + (size_t)(c * 4 + k)] = (k == c) ? scalar_t(1) : scalar_t(0);
                diag_out->assign(d.begin(), d.end());
            }

            return sfem::make_op<real_t>(
                    ndc, ndc,
                    [st, ndc](const real_t *const x, real_t *const y) {
                        galerkin_apply(*st->gl, st->red, st->stage, x, st->tmp.data());
                        for (ptrdiff_t k = 0; k < ndc; ++k)
                            y[(size_t)k] += st->cmask[(size_t)k] ? x[(size_t)k] : st->tmp[(size_t)k];
                    },
                    sfem::EXECUTION_SPACE_HOST);
        }

        // The block diagonal of an assembled level, read after identity rows are patched in.
        void bsr_block_diag(const std::shared_ptr<CoarseBSR> &a, const std::vector<uint8_t> &cmask,
                            std::vector<real_t> &out) {
            // rows() counts degrees of freedom, not blocks -- the driver divides by N_FIELDS
            // everywhere it needs a block count, and so must this.
            const ptrdiff_t nn = a->rows() / N_FIELDS;
            out.assign((size_t)nn * 16, real_t(0));
            const sfem::count_t *const rp = a->row_ptr->data();
            const sfem::idx_t *const   ci = a->col_idx->data();
            const real_t *const        vd = a->values->data();
            for (ptrdiff_t r = 0; r < nn; ++r)
                for (sfem::count_t k = rp[r]; k < rp[r + 1]; ++k)
                    if (ci[k] == (sfem::idx_t)r)
                        std::copy(vd + (size_t)k * 16, vd + (size_t)k * 16 + 16, out.data() + (size_t)r * 16);
            for (ptrdiff_t n = 0; n < nn; ++n)
                for (int c = 0; c < N_FIELDS; ++c)
                    if (cmask[(size_t)n * N_FIELDS + (size_t)c])
                        for (int k = 0; k < N_FIELDS; ++k)  // rows only, as above
                            out[(size_t)n * 16 + (size_t)(c * 4 + k)] = (k == c) ? real_t(1) : real_t(0);
        }
    }  // namespace

    std::shared_ptr<FineStencil> assemble_fine_stencil(const sfem::CVFEMNavierStokes              &op,
                                                       const std::shared_ptr<sfem::FunctionSpace> &space,
                                                       const uint8_t *const                        fine_constrained,
                                                       const bool                                  single) {
        SFEM_TRACE_SCOPE("cvfem_ss::assemble_fine_stencil");
        const ::SSMeshData *const ss = op.semi_structured_data();
        if (!ss) SFEM_ERROR("assemble_fine_stencil: the operator is not semi-structured\n");

        // q = 1: the coarse lattice IS the fine one, so P is the identity and the assembled
        // stencils hold the operator's own entries -- the same identity the q = 1 gate checks.
        GalerkinLevel gl;
        galerkin_init(*ss, 1, gl);
        galerkin_gid_from_spaces(space, space, gl);
        if (fine_constrained)
            gl.fine_constrained.assign(fine_constrained, fine_constrained + space->n_dofs());

        GalerkinPattern &pat = galerkin_pattern(gl);
        if (!pat.fold_built) {
            build_fold_map(gl, pat, pat.fold);
            pat.fold_built = true;
        }

        auto out    = std::make_shared<FineStencil>();
        out->L      = gl.Lc;
        out->nc     = gl.nc;
        out->nmacro = gl.nmacro;
        out->gid.assign(gl.gid.begin(), gl.gid.end());

        const size_t stride = (size_t)gl.nc * 27;  // slots per macro-element
        const size_t nvals  = (size_t)gl.nmacro * stride * 16;
        if (single) out->vf.assign(nvals, 0.0f);
        else        out->vd.assign(nvals, real_t(0));
        float *const  vf = single ? out->vf.data() : nullptr;
        real_t *const vd = single ? nullptr : out->vd.data();

        // Assembled in the same chunks as the matrix path, for the same reason: the element
        // matrices of a whole mesh at once are a transient the size of the result.
        const ptrdiff_t step = galerkin_chunk(gl);
        for (ptrdiff_t e0 = 0; e0 < gl.nmacro; e0 += step) {
            const ptrdiff_t e1 = std::min(gl.nmacro, e0 + step);
            galerkin_assemble(*ss, (scalar_t)op.rho, (scalar_t)op.mu, gl, e0, e1);
            const size_t base = (size_t)e0 * stride * 16;
            const size_t n    = (size_t)(e1 - e0) * stride * 16;
#pragma omp parallel for schedule(static)
            for (ptrdiff_t k = 0; k < (ptrdiff_t)n; ++k) {
                if (single) vf[base + (size_t)k] = (float)gl.C[(size_t)k];
                else        vd[base + (size_t)k] = (real_t)gl.C[(size_t)k];
            }
        }

        // The fold. A pair on a shared face, edge or corner was assembled once per
        // macro-element that contains both of its nodes; the entry is the sum, and every
        // macro-element carrying the pair must hold it, because a sweep reads only its own
        // stencils. Members are in increasing triple order and groups are disjoint, so this is
        // the same sum on any thread count.
        const ptrdiff_t ngroups = (ptrdiff_t)pat.fold.ptr.size() - 1;
#pragma omp parallel for schedule(static)
        for (ptrdiff_t t = 0; t < ngroups; ++t) {
            const ptrdiff_t kb = pat.fold.ptr[(size_t)t], ke = pat.fold.ptr[(size_t)t + 1];
            if (single) {
                float acc[16] = {0};
                for (ptrdiff_t k = kb; k < ke; ++k) {
                    const float *const src = vf + (size_t)pat.fold.idx[(size_t)k] * 16;
                    for (int c = 0; c < 16; ++c) acc[c] += src[c];
                }
                for (ptrdiff_t k = kb; k < ke; ++k) {
                    float *const dst = vf + (size_t)pat.fold.idx[(size_t)k] * 16;
                    for (int c = 0; c < 16; ++c) dst[c] = acc[c];
                }
            } else {
                real_t acc[16] = {0};
                for (ptrdiff_t k = kb; k < ke; ++k) {
                    const real_t *const src = vd + (size_t)pat.fold.idx[(size_t)k] * 16;
                    for (int c = 0; c < 16; ++c) acc[c] += src[c];
                }
                for (ptrdiff_t k = kb; k < ke; ++k) {
                    real_t *const dst = vd + (size_t)pat.fold.idx[(size_t)k] * 16;
                    for (int c = 0; c < 16; ++c) dst[c] = acc[c];
                }
            }
        }

        // SFEM_VANKA_SLOT_CHECK=1: is the 27-slot lattice direction a GLOBAL key?
        //
        // The completion phase for the patch entries wants to exchange, per node, the 27 blocks
        // of its lattice row POSITIONALLY -- entry [n][s] meaning A[n][the neighbour one step
        // from n in direction s]. That is a well-defined global key only if every macro-element
        // containing n agrees on which node lies in direction s.
        //
        // gal_neighbour decodes s as an offset (di, dj, dk) in base three, but in the element's
        // OWN lattice coordinates, so the agreement holds exactly when macro-elements are
        // consistently oriented. The existing fold is immune to this because it resolves through
        // gid rather than through the direction; a positional buffer would not be.
        //
        // If the direction is a global key the completion is one fixed-width nodal exchange of
        // 27 x 16 reals, the same shape as the nodal gradient gather. If it is not, the column
        // identity has to travel alongside the values and the phase is materially bigger. That
        // is the whole difference, so it is measured rather than assumed.
        //
        // Serial is the decisive case: every element is present, so any disagreement must show.
        if (smesh::Env::read<int>("SFEM_VANKA_SLOT_CHECK", 0)) {
            const int                Lc = gl.Lc, ncl = gl.nc;
            std::vector<sfem::idx_t> first((size_t)gl.n_coarse * 27, (sfem::idx_t)0);
            std::vector<uint8_t>     seen((size_t)gl.n_coarse * 27, 0);
            ptrdiff_t                checked = 0, mismatch = 0;
            for (ptrdiff_t e = 0; e < gl.nmacro; ++e)
                for (int a = 0; a < ncl; ++a) {
                    const sfem::idx_t n = gl.gid[(size_t)e * ncl + a];
                    for (int s = 0; s < 27; ++s) {
                        const int b = gal_neighbour(Lc, a, s);
                        if (b < 0) continue;
                        const sfem::idx_t nb = gl.gid[(size_t)e * ncl + b];
                        const size_t      k  = (size_t)n * 27 + (size_t)s;
                        ++checked;
                        if (!seen[k]) {
                            seen[k]  = 1;
                            first[k] = nb;
                        } else if (first[k] != nb) {
                            ++mismatch;
                        }
                    }
                }
            std::printf("vankaslot: checked %td  mismatches %td  %s\n", checked, mismatch,
                        mismatch == 0 ? "direction IS a global key (positional 27x16 exchange is valid)"
                                      : "direction is NOT a global key (columns must travel)");
        }

        // Complete the patch entries across ranks.
        //
        // The fold above sums a node pair's contributions over the macro-elements THIS RANK
        // holds. For a pair involving an owned node that is the whole sum, because the aura
        // makes every element touching an owned node local -- which is why an owned node's row
        // is already complete here. For a pair between two non-owned nodes of an aura element
        // it is not, and those entries sit in the 8x8 block a patch inverts to produce the
        // correction at an owned node. Measured: the stencil values drift monotonically with
        // the cut (abs 81.72 at one rank down to 75.45 at seven) while the number of slots
        // summed stays fixed, which is a sum losing contributions rather than reordering.
        //
        // So the missing values all live on the owners, and completing them is a GATHER rather
        // than a scatter_add: there is nothing to accumulate first. The row is keyed by node
        // and by the 27 lattice directions, which gal_neighbour decodes as an offset in base
        // three. That is a global key only if adjacent macro-elements agree on what a direction
        // means, and SFEM_VANKA_SLOT_CHECK=1 above verifies exactly that -- it holds across
        // every case the spike runs (cavity, cavity_reg, step, poiseuille, couette; about two
        // million pairs, no disagreement), which is what lets this be one fixed-width exchange
        // instead of shipping column identities alongside the values.
        //
        // Assignment on both legs, not accumulation: after the fold every element carrying a
        // pair holds the same full sum, so packing a node's row from any of its elements gives
        // the same answer, and whatever is packed for a ghost is replaced by its owner's row.
        //
        // Unconditional once this function is entered. The gather is collective, and a
        // condition evaluated per rank in front of it is the deadlock this file already paid
        // for once in the probing path.
        if (ss->mesh && ss->mesh->is_distributed() && ss->mesh->comm() && ss->mesh->comm()->size() > 1) {
            const int       ncl = gl.nc;
            const ptrdiff_t nn  = gl.n_coarse;
            const size_t    W   = 27 * 16;  // one node's lattice row

            auto ex = lattice_row_exchange(ss->mesh);

            // Slot by slot, masked by the same predicate the pattern builder uses.
            //
            // A block copy of all 27 slots is wrong, and measurably so: an element carries the
            // slot for direction s only when gal_neighbour keeps it inside that element's
            // lattice. For a node on a macro-element face, edge or corner, a direction can be
            // interior to one element and out-of-lattice in another, where the slot is a
            // structural zero. Copying whole blocks therefore lets the last element touching a
            // node overwrite a real value from another element with that zero -- which showed
            // up as abs 81.7058 against serial's 81.7219, uniform across rank counts and so
            // wrong in the one way a rank-to-rank comparison cannot see.
            //
            // Masked, the union of the valid slots over all elements containing an owned node
            // covers that node's whole row, which is exactly what the aura guarantees.
            const int Lcl = gl.Lc;
            auto complete = [&](auto *const vals) {
                using V = typename std::remove_const<typename std::remove_reference<decltype(*vals)>::type>::type;
                std::vector<V> row((size_t)nn * W, V(0));

                for (ptrdiff_t e = 0; e < gl.nmacro; ++e)
                    for (int a = 0; a < ncl; ++a) {
                        const ptrdiff_t n    = (ptrdiff_t)gl.gid[(size_t)e * ncl + a];
                        const size_t    base = ((size_t)e * stride + (size_t)a * 27) * 16;
                        for (int s = 0; s < 27; ++s) {
                            if (gal_neighbour(Lcl, a, s) < 0) continue;
                            std::copy(vals + base + (size_t)s * 16, vals + base + (size_t)s * 16 + 16,
                                      row.data() + (size_t)n * W + (size_t)s * 16);
                        }
                    }

                ex->gather(row.data(), (ptrdiff_t)W);

                for (ptrdiff_t e = 0; e < gl.nmacro; ++e)
                    for (int a = 0; a < ncl; ++a) {
                        const ptrdiff_t n    = (ptrdiff_t)gl.gid[(size_t)e * ncl + a];
                        const size_t    base = ((size_t)e * stride + (size_t)a * 27) * 16;
                        for (int s = 0; s < 27; ++s) {
                            if (gal_neighbour(Lcl, a, s) < 0) continue;
                            std::copy(row.data() + (size_t)n * W + (size_t)s * 16,
                                      row.data() + (size_t)n * W + (size_t)s * 16 + 16,
                                      vals + base + (size_t)s * 16);
                        }
                    }
            };

            if (single) complete(out->vf.data());
            else        complete(out->vd.data());
        }

        // SFEM_VANKA_STENCIL_SUM=1: are the assembled patch stencils the same matrix however
        // the domain was cut?
        //
        // This is the last object in the Vanka chain that has not been measured. Eliminated
        // before it, each by measurement rather than argument: the linearisation state
        // (identical at 1/2/4/8), the patch multiplicities (1728 element and 4096 patch
        // incidences over owned nodes at every rank count), the input gather (the wrapper is
        // correct and uses GhostsAndAura), the coarse correction (the drift persists at
        // SFEM_GMG_CGC=0 and with no multigrid at all), the sweep variant (additive and
        // multiplicative both drift, and both upward), and the nodal pressure-gradient
        // reconstruction (invariant to fourteen figures under block-Jacobi, so its apparent
        // drift under Vanka is a consequence of Vanka rather than its cause).
        //
        // What makes the stencils the remaining candidate is the fold above. It knows nothing
        // about ownership -- build_fold_map discards its GalerkinLevel -- and it completes a
        // node pair by summing every macro-element THIS RANK holds that carries the pair. For a
        // pair involving an owned node that is complete, because every element touching an
        // owned node is local. But an element touching an owned node may itself be an aura
        // element, and its patch also holds entries between two non-owned nodes whose fold
        // groups can be short. Those entries enter the 8x8 patch, are inverted, and the inverse
        // is what produces the correction at the owned node.
        //
        // Summed over OWNED rows only and keyed by global id. Every rank holds every element
        // touching its owned nodes, so this multiset of (owned node, macro-element) slots is
        // the same however the mesh is cut, and the total must equal the serial one.
        if (smesh::Env::read<int>("SFEM_VANKA_STENCIL_SUM", 0) && ss->mesh) {
            const bool dist = ss->mesh->is_distributed() && ss->mesh->comm() && ss->mesh->comm()->size() > 1;
            const ptrdiff_t   nown = dist ? ss->mesh->distributed()->n_nodes_owned() : ss->nnodes;
            const auto *const l2g  = dist ? ss->mesh->distributed()->node_mapping()->data() : nullptr;

            // Restricted to owned ELEMENTS, not owned rows.
            //
            // The first version of this probe filtered on the row node being owned, and could
            // not have detected the thing it was written for. The fold completes a node pair
            // from the macro-elements this rank holds; for a pair involving an owned node that
            // is complete, because every element touching an owned node is local. The entries
            // that can be short are the ones between two NON-owned nodes of an aura element --
            // and those are exactly the rows an owned-row filter discards. Reading that version
            // as exoneration was wrong.
            //
            // Owned elements are the right restriction because every macro-element is owned by
            // exactly one rank: the driver's global gate prints "nelements 64 summed 64" over
            // the owned counts, so summing every row of every owned element visits each
            // (element, lattice node, slot) triple exactly once across the whole run, whatever
            // the cut. The spelling is the one the driver already uses, with the same fallback
            // for the serial case where the per-block owned count is zero.
            //
            // At one rank this is the same 64 x 27 = 1728 slots the previous version summed, so
            // its serial total (sum 9.9143272486398928, abs 81.721892532339552) remains the
            // reference every distributed run must reproduce.
            const ptrdiff_t n_owned_elements =
                    (dist && ss->mesh->block(0) && ss->mesh->block(0)->n_elements_owned())
                            ? ss->mesh->block(0)->n_elements_owned()
                            : gl.nmacro;

            long double sum = 0, absum = 0, wsum = 0, cnt = 0;
            for (ptrdiff_t e = 0; e < n_owned_elements && e < gl.nmacro; ++e)
                for (int a = 0; a < gl.nc; ++a) {
                    const ptrdiff_t n = (ptrdiff_t)out->gid[(size_t)e * (size_t)gl.nc + (size_t)a];
                    // Every row of an owned element, including rows this rank does not own.
                    // Those carry a global id only when they are inside the local numbering;
                    // a row outside it contributes to the value sums and is simply not weighted.
                    const long double g =
                            (n >= 0 && l2g && n < nown) ? (long double)l2g[n] : (long double)(n >= 0 ? n : 0);
                    const size_t base = ((size_t)e * stride + (size_t)a * 27) * 16;
                    for (int s = 0; s < 27 * 16; ++s) {
                        const long double v = single ? (long double)out->vf[base + (size_t)s]
                                                     : (long double)out->vd[base + (size_t)s];
                        sum += v;
                        absum += v < 0 ? -v : v;
                        wsum += v * (g + 1);
                    }
                    cnt += 1;
                }

            double gs = (double)sum, ga = (double)absum, gw = (double)wsum, gc = (double)cnt;
            int    rank = 0;
            if (dist) {
                gs   = ss->mesh->comm()->sum(gs);
                ga   = ss->mesh->comm()->sum(ga);
                gw   = ss->mesh->comm()->sum(gw);
                gc   = ss->mesh->comm()->sum(gc);
                rank = ss->mesh->comm()->rank();
            }
            if (rank == 0)
                std::printf("stencilsum: owned_slots %.0f  sum %.17g  abs %.17g  wsum %.17g\n", gc, gs, ga,
                            gw);
        }

        return out;
    }

    CoarseHierarchy assemble_hierarchy(sfem::CVFEMNavierStokes                                 &op,
                                       const real_t *const                                      state,
                                       const std::vector<std::shared_ptr<sfem::FunctionSpace>> &spaces,
                                       const std::vector<std::vector<uint8_t>>                 &masks,
                                       const bool                                               element_matrices) {
        SFEM_TRACE_SCOPE("cvfem_ss::assemble_hierarchy");
        if (state) op.update(state);
        const ::SSMeshData *const ss = op.semi_structured_data();
        if (!ss) SFEM_ERROR("assemble_hierarchy: the operator is not semi-structured\n");

        const int nlevels = (int)spaces.size();
        CoarseHierarchy out;
        out.A.assign((size_t)nlevels, nullptr);
        out.op.assign((size_t)nlevels, nullptr);
        out.diag.assign((size_t)nlevels, {});
        if (nlevels < 2) return out;

        // The chain of levels, all kept alive: an element-matrix level's operator holds its
        // own, and the next hop reads the one above it.
        std::vector<std::shared_ptr<GalerkinLevel>> lv((size_t)nlevels);

        // Level 1, straight from the micro-cell matrices, with the fine mask on the columns.
        {
            const int Lf = ss->level;
            const int Lc = spaces[1]->has_semi_structured_mesh()
                                   ? smesh::semistructured_level(spaces[1]->mesh())
                                   : 1;
            if (Lc < 1 || Lf % Lc) SFEM_ERROR("assemble_hierarchy: level %d does not divide %d\n", Lc, Lf);
            lv[1] = std::make_shared<GalerkinLevel>();
            galerkin_init(*ss, Lf / Lc, *lv[1]);
            galerkin_gid_from_spaces(spaces[1], spaces[0], *lv[1]);
            lv[1]->fine_constrained = masks[0];
            galerkin_assemble(*ss, (scalar_t)op.rho, (scalar_t)op.mu, *lv[1], 0, lv[1]->nmacro);
        }

        // Each level below is one element-local hop, masking the level above's columns with
        // that level's own constraints.
        for (int i = 2; i < nlevels; ++i) {
            auto nxt    = std::make_shared<GalerkinLevel>();
            nxt->Lc     = spaces[i]->has_semi_structured_mesh() ? smesh::semistructured_level(spaces[i]->mesh()) : 1;
            nxt->q      = lv[i - 1]->Lc / nxt->Lc;
            nxt->nc     = (nxt->Lc + 1) * (nxt->Lc + 1) * (nxt->Lc + 1);
            nxt->nmacro = lv[i - 1]->nmacro;
            galerkin_gid_from_spaces(spaces[i], spaces[0], *nxt);
            galerkin_hop(*lv[i - 1], masks[(size_t)i - 1].empty() ? nullptr : masks[(size_t)i - 1].data(), *nxt);
            lv[i] = nxt;
        }

        for (int i = 1; i < nlevels; ++i) {
            // The coarsest level is assembled even in element-matrix mode: it is the one that
            // gets factorised, and a matrix is the natural thing to hand a direct solve.
            const bool keep_em = element_matrices && i + 1 < nlevels;
            if (keep_em) {
                out.op[(size_t)i] = make_em_operator(lv[(size_t)i], masks[(size_t)i], &out.diag[(size_t)i]);
            } else {
                auto a = level_to_bsr(*lv[(size_t)i]);

                // Complete this level's rows across ranks, for the same reason the fine
                // stencils needed it.
                //
                // galerkin_accumulate sums a node pair's contributions over the macro-elements
                // THIS RANK holds. For a pair touching an owned node that is the whole sum,
                // because the one-deep aura makes every element touching an owned node local.
                // For a pair between two NON-owned nodes it is not -- and a Vanka patch built
                // from this matrix reads rowptr[n[a]] for all eight corners of its cell,
                // including rows this rank does not own. Block-Jacobi reads only a node's own
                // block and never sees it, which is why the drift is Vanka-only.
                //
                // Confirmed by disabling the coarse Vanka branch entirely: with intermediate
                // levels kept as element matrices, so make_diagonal_vanka_from_bsr never fires,
                // four levels measure 4 / 4 / 4 at 1 / 4 / 8 ranks against 3 / 5 / 14 with it.
                //
                // The missing values all live on the owners, so this is a GATHER, as at the
                // fine level. Three things differ and each matters: the exchange is built on
                // THIS level's mesh, because galerkin_gid_from_spaces puts coarse ids in the
                // coarse space's own numbering and the fine mesh would index different nodes;
                // it runs AFTER level_to_bsr, because the per-element blocks in gl.C are
                // partial by design and it is the accumulation that is rank-local; and it
                // reuses pat.pos, which already maps (e, a, s) to a BSR block, so the
                // (node, lattice direction) key needs nothing new.
                if (spaces[(size_t)i]) {
                    auto cm = spaces[(size_t)i]->mesh_ptr();
                    if (cm && cm->is_distributed() && cm->comm() && cm->comm()->size() > 1) {
                        const GalerkinLevel &gl  = *lv[(size_t)i];
                        GalerkinPattern     &pat = galerkin_pattern(gl);
                        const int            ncl = gl.nc, Lcl = gl.Lc;
                        const ptrdiff_t      nn  = gl.n_coarse;
                        const size_t         W   = 27 * 16;
                        real_t *const        va  = a->values->data();

                        auto                ex = lattice_row_exchange(cm);
                        std::vector<real_t> row((size_t)nn * W, real_t(0));

                        for (ptrdiff_t e = 0; e < gl.nmacro; ++e)
                            for (int la = 0; la < ncl; ++la) {
                                const ptrdiff_t n = (ptrdiff_t)gl.gid[(size_t)e * ncl + la];
                                for (int s = 0; s < 27; ++s) {
                                    if (gal_neighbour(Lcl, la, s) < 0) continue;
                                    const ptrdiff_t p = pat.pos[((size_t)e * ncl + la) * 27 + (size_t)s];
                                    if (p < 0) continue;
                                    std::copy(va + (size_t)p * 16, va + (size_t)p * 16 + 16,
                                              row.data() + (size_t)n * W + (size_t)s * 16);
                                }
                            }

                        ex->gather(row.data(), (ptrdiff_t)W);

                        for (ptrdiff_t e = 0; e < gl.nmacro; ++e)
                            for (int la = 0; la < ncl; ++la) {
                                const ptrdiff_t n = (ptrdiff_t)gl.gid[(size_t)e * ncl + la];
                                for (int s = 0; s < 27; ++s) {
                                    if (gal_neighbour(Lcl, la, s) < 0) continue;
                                    const ptrdiff_t p = pat.pos[((size_t)e * ncl + la) * 27 + (size_t)s];
                                    if (p < 0) continue;
                                    std::copy(row.data() + (size_t)n * W + (size_t)s * 16,
                                              row.data() + (size_t)n * W + (size_t)s * 16 + 16,
                                              va + (size_t)p * 16);
                                }
                            }
                    }
                }

                out.A[(size_t)i] = a;
                // Identity rows, then the diagonal read off the patched matrix.
                const ptrdiff_t nn = a->rows() / N_FIELDS;
                sfem::count_t *const rp = a->row_ptr->data();
                sfem::idx_t *const   ci = a->col_idx->data();
                real_t *const        vd = a->values->data();
                for (ptrdiff_t r = 0; r < nn; ++r)
                    for (int c = 0; c < N_FIELDS; ++c) {
                        if (!masks[(size_t)i][(size_t)r * N_FIELDS + (size_t)c]) continue;
                        for (sfem::count_t k = rp[r]; k < rp[r + 1]; ++k)
                            for (int t = 0; t < N_FIELDS; ++t)
                                vd[(size_t)k * 16 + (size_t)(c * 4 + t)] =
                                        (ci[k] == (sfem::idx_t)r && t == c) ? real_t(1) : real_t(0);
                    }
                bsr_block_diag(a, masks[(size_t)i], out.diag[(size_t)i]);
                out.op[(size_t)i] = a;
            }
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
