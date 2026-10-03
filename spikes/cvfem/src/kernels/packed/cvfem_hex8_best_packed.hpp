#ifndef CVFEM_HEX8_BEST_PACKED_HPP
#define CVFEM_HEX8_BEST_PACKED_HPP

// Packed layout: elements are grouped into packs, each pack accumulates into a
// thread-private buffer indexed by pack-local node ids, and the buffer is folded
// back into the global structure afterwards. Ghost rows -- nodes a pack touches
// but does not own -- are reduced in a second pass.
//
// The pack-local indexing lets the residual and Jacobian-action run 16-wide SIMD
// over elements. For assembly the local matrix is large enough that the round
// trip through it costs more than it saves; see cvfem_hex8_best_colored.hpp
// and cvfem_hex8_best_store.hpp.

#include "best/cvfem_hex8_best_common.hpp"
#include "kernels/cvfem_range.hpp"

static void build_pack_local_crs(PackedData               &p,
                                 const ptrdiff_t           nelements,
                                 const smesh::count_t     *rowptr_g,
                                 const smesh::idx_t       *colidx_g) {
    p.local_rowptr.resize((size_t)p.n_packs);
    p.local_colidx.resize((size_t)p.n_packs);
    p.local_global_slot.resize((size_t)p.n_packs);
    p.local_element_slot.assign((size_t)nelements * CVFEM_HEX8_N_NODES * CVFEM_HEX8_N_NODES, 0);
    p.max_local_nnz = 0;

    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - p.owned_nodes_ptr[pack];
        const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
        const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
        const ptrdiff_t e_start      = pack * p.n_elements_per_pack;
        const ptrdiff_t e_end        = std::min(nelements, (pack + 1) * p.n_elements_per_pack);

        std::vector<std::vector<pack_idx_t>> adj((size_t)n_pack_nodes);
        for (ptrdiff_t e = e_start; e < e_end; ++e) {
            pack_idx_t ev[CVFEM_HEX8_N_NODES];
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) ev[a] = p.elems[a][e];
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                for (int b = 0; b < CVFEM_HEX8_N_NODES; ++b) adj[(size_t)ev[a]].push_back(ev[b]);
            }
        }

        auto &rowptr = p.local_rowptr[(size_t)pack];
        auto &colidx = p.local_colidx[(size_t)pack];
        rowptr.assign((size_t)n_pack_nodes + 1, 0);
        for (ptrdiff_t i = 0; i < n_pack_nodes; ++i) {
            auto &row = adj[(size_t)i];
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            rowptr[(size_t)i + 1] = (int)row.size();
        }
        for (ptrdiff_t i = 0; i < n_pack_nodes; ++i) rowptr[(size_t)i + 1] += rowptr[(size_t)i];
        colidx.resize((size_t)rowptr[(size_t)n_pack_nodes]);
        for (ptrdiff_t i = 0; i < n_pack_nodes; ++i) {
            const auto &row = adj[(size_t)i];
            std::memcpy(colidx.data() + rowptr[(size_t)i], row.data(), row.size() * sizeof(pack_idx_t));
        }

        auto &global_slots = p.local_global_slot[(size_t)pack];
        global_slots.resize(colidx.size());
        for (ptrdiff_t i = 0; i < n_pack_nodes; ++i) {
            const smesh::idx_t grow  = pack_local_to_global(p, pack, n_contiguous, (pack_idx_t)i);
            const int          begin = rowptr[(size_t)i];
            const int          end   = rowptr[(size_t)i + 1];
            for (int t = begin; t < end; ++t) {
                const smesh::idx_t gcol = pack_local_to_global(p, pack, n_contiguous, colidx[(size_t)t]);
                global_slots[(size_t)t] = find_bsr_slot(rowptr_g, colidx_g, grow, gcol);
            }
        }
        p.max_local_nnz = std::max(p.max_local_nnz, (ptrdiff_t)colidx.size());

        for (ptrdiff_t e = e_start; e < e_end; ++e) {
            int *const slots = p.local_element_slot.data() + (size_t)e * 64;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t local_row = p.elems[a][e];
                const int        row_begin = rowptr[(size_t)local_row];
                const int        row_len   = rowptr[(size_t)local_row + 1] - row_begin;
                const pack_idx_t *row      = colidx.data() + row_begin;
                for (int b = 0; b < CVFEM_HEX8_N_NODES; ++b) {
                    slots[a * 8 + b] = row_begin + find_pack_col(p.elems[b][e], row, row_len);
                }
            }
        }
    }

    p.ghost_mat_ptr.assign((size_t)p.n_ghost_entries + 1, 0);
    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - p.owned_nodes_ptr[pack];
        const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
        const ptrdiff_t ghost_off    = p.ghost_ptr[pack];
        const auto     &rowptr       = p.local_rowptr[(size_t)pack];
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
            const ptrdiff_t local_i = n_contiguous + k;
            p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k + 1] = rowptr[(size_t)local_i + 1] - rowptr[(size_t)local_i];
        }
    }
    for (ptrdiff_t i = 0; i < p.n_ghost_entries; ++i) p.ghost_mat_ptr[(size_t)i + 1] += p.ghost_mat_ptr[(size_t)i];

    const ptrdiff_t gnnz = p.ghost_mat_ptr[(size_t)p.n_ghost_entries];
    if (getenv("CVFEM_PACK_STATS")) {
        ptrdiff_t sum_local_nnz = 0;
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) sum_local_nnz += (ptrdiff_t)p.local_colidx[(size_t)pack].size();
        std::printf("[pack-stats] sum_local_nnz=%td ghost_nnz=%td\n", sum_local_nnz, gnnz);
    }
    p.ghost_mat_slot.resize((size_t)gnnz);
    p.ghost_mat_val.assign((size_t)gnnz * 16, 0.0);

    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - p.owned_nodes_ptr[pack];
        const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
        const ptrdiff_t ghost_off    = p.ghost_ptr[pack];
        const auto     &rowptr       = p.local_rowptr[(size_t)pack];
        const auto     &colidx       = p.local_colidx[(size_t)pack];
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
            const ptrdiff_t local_i = n_contiguous + k;
            const int       begin   = rowptr[(size_t)local_i];
            const int       end     = rowptr[(size_t)local_i + 1];
            const ptrdiff_t dest    = p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k];
            const smesh::idx_t grow = p.ghost_idx[(size_t)ghost_off + (size_t)k];
            for (int t = 0; t < end - begin; ++t) {
                const smesh::idx_t gcol = pack_local_to_global(p, pack, n_contiguous, colidx[(size_t)begin + t]);
                p.ghost_mat_slot[(size_t)dest + (size_t)t] = find_bsr_slot(rowptr_g, colidx_g, grow, gcol);
            }
        }
    }
}

// The DEFERRED-CORRECTION higher-order convective flux, on the packed layout.
//
// Same two-pass shape as apply_residual_packed: stage the pack's fields, accumulate into a
// pack-private buffer with plain `+=`, write the owned rows straight out, stage the ghosts and
// close them with the reduction graph. No atomic anywhere, and the same fixed summation order,
// so the higher-order operator inherits the format's reproducibility rather than giving it up.
//
// It runs the SCALAR sum-factored kernel, not the 16-wide SIMD one, because the scalar kernel
// is the FASTER of the two for this operator -- 659 against 500 MDOF/s on Grace at 8,586,756
// dof (job 4812910). Both accept the correction; the SIMD one re-gathers each lane's 96 inputs
// inside every one of the twelve face loops, and that costs more than vectorising the flux
// around it returns. That is also the honest basis for the comparison this enables: the atomic
// higher-order sweep runs the same scalar kernel, so packed against atomic here isolates the
// LAYOUT with the kernel held fixed. It also means a packed higher-order number is not
// comparable with a packed first-order one, which is SIMD -- the paper says so rather than
// letting the two sit in one column.
//
// `ugrad` is nine interleaved components per node and is hoisted, as the solver lags the
// correction one Newton step. The reconstruction also needs the element's node coordinates, so
// the pack stages its coordinates whenever the correction is on, exactly as Rhie-Chow does.
// The pack sweep, driven by a range. The `#pragma omp parallel` is in the launcher below; see
// kernels/cvfem_range.hpp for why DESIGN.md wants it there. The packs of a range touch only
// nodes this part owns -- that is what the packed layout is for -- so the parts need no
// synchronisation between them, and the ghost rows they do share are reduced afterwards in the
// launcher, which is the second and independent parallel loop.
static SFEM_NOINLINE void apply_residual_packed_defcor_scalar_range(
        const cvfem_range packs,
        MeshData &d,
        PackedData &p,
        const scalar_t rho,
        const scalar_t mu,
        const scalar_t *const SFEM_RESTRICT ugrad,
        const int limiter,
        const scalar_t venkat_c,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        scalar_t *const SFEM_RESTRICT rc,
        const size_t scratch_n,
        const Hex8Extras & opt,
        const int with_rc) {
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(1, scratch_n);
        // Coordinates always, and the pressure gradient when Rhie-Chow is on: the same
        // six-array slot the first-order SIMD path uses, so no new scratch shape appears.
        scalar_t *const SFEM_RESTRICT pack_xyz =
                thread_scratch<scalar_t>(3, with_rc ? packed_rc_n(p) : packed_xyz_n(p));
        const ptrdiff_t xyz_n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x   = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y   = pack_xyz + xyz_n;
        scalar_t *const SFEM_RESTRICT pack_z   = pack_xyz + 2 * xyz_n;
        scalar_t *const SFEM_RESTRICT pack_pgx = with_rc ? pack_xyz + 3 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgy = with_rc ? pack_xyz + 4 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgz = with_rc ? pack_xyz + 5 * xyz_n : nullptr;

    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t                         n_pack_nodes = n_contiguous + n_ghost;
            const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const ptrdiff_t                         ghost_off    = p.ghost_ptr[pack];

            std::memset(pack_out, 0, (size_t)n_pack_nodes * (size_t)N_FIELDS * sizeof(scalar_t));
            fill_pack_fields(p, d, pack, n_contiguous, n_ghost, ghosts, pack_u);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x,
                                               pack_y, pack_z, pack_pgx, pack_pgy, pack_pgz);
            else
                fill_pack_xyz(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);

            // THE SIMD HIGHER-ORDER PATH IS CORRECT AND SLOWER, which is why this sweep runs
            // the scalar kernel. The 2.5e-05 discrepancy this comment used to record was real
            // but was not the reconstruction's geometry: the two EPS=false branches of
            // cvfem_hex8_ns_upwind_residual_sumfact_simd did not forward `ho` to
            // cvfem_hex8_conv_all_simd, so the correction was silently never applied on the
            // path the benchmark took. Found by observing that disabling the correction left
            // the discrepancy bit-identical. With `ho` forwarded the two layouts agree to
            // 1.3e-18, and the remaining reason to prefer the scalar kernel is throughput --
            // see the note on this function above.
            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                scalar_t ux_e[8], uy_e[8], uz_e[8], p_e[8], r[CVFEM_HEX8_N_DOF], g8[72];
                // Fields come from the PACK -- read once per pack node, contiguously for the
                // owned majority, which is the layout's advantage.
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)p.elems[a][e] * N_FIELDS;
                    ux_e[a] = u[0]; uy_e[a] = u[1]; uz_e[a] = u[2]; p_e[a] = u[3];
                }
                // The Rhie-Chow inputs come through the shared scratch, exactly as the atomic
                // sweep takes them, so the kernel sees identical inputs in both layouts.
                Hex8ExtraScratch ex;
                ex.load(d, opt, e);
                // Coordinates are gathered here rather than taken from `ex`, and that is not
                // redundant: Hex8ExtraScratch::load returns EARLY when neither Rhie-Chow nor
                // the boundary closure is on, leaving its x/y/z untouched. The reconstruction
                // works in physical space and needs them whether or not those terms are on, so
                // reading ex.x there gave an uninitialised buffer -- caught by the packed-vs-
                // atomic check at 2.9e-03, which is what that check is for.
                scalar_t xe[8], ye[8], ze[8];
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const smesh::idx_t g = d.elems[a][e];
                    xe[a] = scalar_t(d.points[0][g]);
                    ye[a] = scalar_t(d.points[1][g]);
                    ze[a] = scalar_t(d.points[2][g]);
                    for (int c = 0; c < 9; ++c) g8[a * 9 + c] = ugrad[(ptrdiff_t)g * 9 + c];
                }
                scalar_t adj[9], det;
                load_hex8_adj(d, e, adj, &det);
                cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, adj, det, ux_e, uy_e, uz_e, p_e, r,
                                                      ex.rc, /*ueps=*/scalar_t(0),
                                                      g8, xe, ye, ze,
                                                      limiter, venkat_c, nullptr);

                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    scalar_t *const SFEM_RESTRICT out = pack_out + (ptrdiff_t)p.elems[a][e] * N_FIELDS;
                    out[0] += r[a * 4 + 0];
                    out[1] += r[a * 4 + 1];
                    out[2] += r[a * 4 + 2];
                    out[3] += r[a * 4 + 3];
                }
            }

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + k * N_FIELDS;
                const ptrdiff_t                     g   = owned + k;
                rx[g] = out[0]; ry[g] = out[1]; rz[g] = out[2]; rc[g] = out[3];
            }
            scalar_t *const SFEM_RESTRICT gx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = p.ghost_buf.data() + 1 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = p.ghost_buf.data() + 3 * p.n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + (n_contiguous + k) * N_FIELDS;
                gx[ghost_off + k] = out[0]; gy[ghost_off + k] = out[1];
                gz[ghost_off + k] = out[2]; gc[ghost_off + k] = out[3];
            }
    }
}


static SFEM_NOINLINE void apply_residual_packed_defcor_scalar(MeshData       &d,
                                                       PackedData     &p,
                                                       const scalar_t  rho,
                                                       const scalar_t  mu,
                                                       const scalar_t *const SFEM_RESTRICT ugrad,
                                                       const int       limiter,
                                                       const scalar_t  venkat_c) {
    scalar_t *const SFEM_RESTRICT rx = d.rx.data();
    scalar_t *const SFEM_RESTRICT ry = d.ry.data();
    scalar_t *const SFEM_RESTRICT rz = d.rz.data();
    scalar_t *const SFEM_RESTRICT rc = d.rc.data();
    const size_t                  scratch_n = packed_scratch_n(p);
    const Hex8Extras              opt(d);
    const int                     with_rc = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);


#pragma omp parallel
    apply_residual_packed_defcor_scalar_range(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d, p, rho, mu, ugrad, limiter, venkat_c, rx, ry, rz, rc, scratch_n, opt, with_rc);

    scalar_t *const fields[N_FIELDS] = {d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data()};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const smesh::idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        for (int f = 0; f < N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + (ptrdiff_t)f * p.n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            fields[f][dest] += sum;
        }
    }
}

// The pack sweep, driven by a range. The `#pragma omp parallel` is in the launcher below; see
// kernels/cvfem_range.hpp for why DESIGN.md wants it there. The packs of a range touch only
// nodes this part owns -- that is what the packed layout is for -- so the parts need no
// synchronisation between them, and the ghost rows they do share are reduced afterwards in the
// launcher, which is the second and independent parallel loop.
static SFEM_NOINLINE void apply_residual_packed_defcor_range(
        const cvfem_range packs,
        MeshData &d,
        PackedData &p,
        const scalar_t rho,
        const scalar_t mu,
        const scalar_t *const SFEM_RESTRICT ugrad,
        const int limiter,
        const scalar_t venkat_c,
        const bool sympy,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        scalar_t *const SFEM_RESTRICT rc,
        const size_t scratch_n,
        const int with_rc) {
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(1, scratch_n);
        // Coordinates always, and the pressure gradient when Rhie-Chow is on: the same
        // six-array slot the first-order SIMD path uses, so no new scratch shape appears.
        scalar_t *const SFEM_RESTRICT pack_xyz =
                thread_scratch<scalar_t>(3, with_rc ? packed_rc_n(p) : packed_xyz_n(p));
        const ptrdiff_t xyz_n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x   = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y   = pack_xyz + xyz_n;
        scalar_t *const SFEM_RESTRICT pack_z   = pack_xyz + 2 * xyz_n;
        scalar_t *const SFEM_RESTRICT pack_pgx = with_rc ? pack_xyz + 3 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgy = with_rc ? pack_xyz + 4 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgz = with_rc ? pack_xyz + 5 * xyz_n : nullptr;

    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t                         n_pack_nodes = n_contiguous + n_ghost;
            const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const ptrdiff_t                         ghost_off    = p.ghost_ptr[pack];

            std::memset(pack_out, 0, (size_t)n_pack_nodes * (size_t)N_FIELDS * sizeof(scalar_t));
            fill_pack_fields(p, d, pack, n_contiguous, n_ghost, ghosts, pack_u);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x,
                                               pack_y, pack_z, pack_pgx, pack_pgy, pack_pgz);
            else
                fill_pack_xyz(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);

            alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE],
                    cof2[CVFEM_HEX8_VEC_SIZE], cof3[CVFEM_HEX8_VEC_SIZE], cof4[CVFEM_HEX8_VEC_SIZE],
                    cof5[CVFEM_HEX8_VEC_SIZE], cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE],
                    cof8[CVFEM_HEX8_VEC_SIZE], detv[CVFEM_HEX8_VEC_SIZE];
            Hex8InputPack    in;
            Hex8ResidualPack outp;
            Hex8RhieChowPack rcp;
            Hex8UGradPack    hop;
            hop.limiter  = limiter;
            hop.venkat_c = venkat_c;
            for (ptrdiff_t begin = e_start; begin < e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, e_end - begin));
                gather_hex8_simd_from_pack(p.elems, pack_u, d, begin, nlanes, in,
                                           cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, detv);
                if (with_rc) {
                    cvfem_hex8_gather_rc_from_pack(p.elems, pack_pgx,
                                                   pack_pgy, pack_pgz, begin, nlanes, rcp);
                    if (sympy) cvfem_hex8_gather_rc_coeff(d, begin, nlanes, rcp);
                }
                for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                    const ptrdiff_t e = begin + lane;
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        if (lane >= nlanes) {
                            hop.x[a][lane] = hop.y[a][lane] = hop.z[a][lane] = scalar_t(0);
                            for (int c = 0; c < 9; ++c) hop.g[a][c][lane] = scalar_t(0);
                            continue;
                        }
                        const smesh::idx_t g = d.elems[a][e];
                        hop.x[a][lane] = scalar_t(d.points[0][g]);
                        hop.y[a][lane] = scalar_t(d.points[1][g]);
                        hop.z[a][lane] = scalar_t(d.points[2][g]);
                        for (int c = 0; c < 9; ++c) hop.g[a][c][lane] = ugrad[(ptrdiff_t)g * 9 + c];
                    }
                }
                if (sympy) {
                    // Assigns rather than accumulates: the generated body is the complete element
                    // residual, so there is no zero-fill for it to add onto.
                    //
                    // Eight kernels, selected here rather than by a parameter inside them: the
                    // limiter is uniform across the sweep, and a four-way select inside the vector
                    // body at every surface is the shape of guard measured at 1.83x in this file.
                    // The macro keeps one copy of the fourteen common arguments; a transposed pack
                    // would hide in eight hand-written copies of it.
#define CVFEM_HEX8_SYMPY_HO_ARGS \
    rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, detv, in, hop
                    if (with_rc) {
                        switch (limiter) {
                            case 1: cvfem_hex8_ns_upwind_sympy_residual_defcor_rc_lim1_simd(
                                            CVFEM_HEX8_SYMPY_HO_ARGS, rcp, outp); break;
                            case 2: cvfem_hex8_ns_upwind_sympy_residual_defcor_rc_lim2_simd(
                                            CVFEM_HEX8_SYMPY_HO_ARGS, rcp, outp); break;
                            case 3: cvfem_hex8_ns_upwind_sympy_residual_defcor_rc_lim3_simd(
                                            CVFEM_HEX8_SYMPY_HO_ARGS, rcp, outp); break;
                            default: cvfem_hex8_ns_upwind_sympy_residual_defcor_rc_lim0_simd(
                                             CVFEM_HEX8_SYMPY_HO_ARGS, rcp, outp); break;
                        }
                    } else {
                        switch (limiter) {
                            case 1: cvfem_hex8_ns_upwind_sympy_residual_defcor_lim1_simd(
                                            CVFEM_HEX8_SYMPY_HO_ARGS, outp); break;
                            case 2: cvfem_hex8_ns_upwind_sympy_residual_defcor_lim2_simd(
                                            CVFEM_HEX8_SYMPY_HO_ARGS, outp); break;
                            case 3: cvfem_hex8_ns_upwind_sympy_residual_defcor_lim3_simd(
                                            CVFEM_HEX8_SYMPY_HO_ARGS, outp); break;
                            default: cvfem_hex8_ns_upwind_sympy_residual_defcor_lim0_simd(
                                             CVFEM_HEX8_SYMPY_HO_ARGS, outp); break;
                        }
                    }
#undef CVFEM_HEX8_SYMPY_HO_ARGS
                } else {
                    cvfem_hex8_ns_upwind_residual_sumfact_simd(
                            rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, detv, in,
                            outp, with_rc ? &rcp : nullptr, d.rhie_chow_scale, scalar_t(0), &hop);
                }
                scatter_hex8_simd_to_pack(p.elems, pack_out, begin, nlanes, outp);
            }
            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + k * N_FIELDS;
                const ptrdiff_t                     g   = owned + k;
                rx[g] = out[0]; ry[g] = out[1]; rz[g] = out[2]; rc[g] = out[3];
            }
            scalar_t *const SFEM_RESTRICT gx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = p.ghost_buf.data() + 1 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = p.ghost_buf.data() + 3 * p.n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + (n_contiguous + k) * N_FIELDS;
                gx[ghost_off + k] = out[0]; gy[ghost_off + k] = out[1];
                gz[ghost_off + k] = out[2]; gc[ghost_off + k] = out[3];
            }
    }
}


static SFEM_NOINLINE void apply_residual_packed_defcor(MeshData       &d,
                                                       PackedData     &p,
                                                       const scalar_t  rho,
                                                       const scalar_t  mu,
                                                       const scalar_t *const SFEM_RESTRICT ugrad,
                                                       const int       limiter,
                                                       const scalar_t  venkat_c,
                                                       // Run the GENERATED lane-blocked kernel
                                                       // instead of the hand-written one. Same
                                                       // sweep, same staging, same scatter -- only
                                                       // the kernel differs, which is what makes
                                                       // the comparison a kernel comparison.
                                                       const bool      sympy = false) {
    // Every limiter arm is generated now, with and without Rhie-Chow -- eight kernels. What is NOT
    // generated is Venkatakrishnan's eps^2 term: it is venkat_c * h^3, so carrying it would put a
    // square root at every sub-control surface, and every caller in the tree passes zero. Refused
    // rather than silently dropped, because a row that names a term it did not compute is worse
    // than a row that does not exist.
    if (sympy && venkat_c != scalar_t(0)) {
        std::fprintf(stderr,
                     "the generated higher-order kernels carry eps^2 = 0; a non-zero venkat_c "
                     "needs the hand-written kernel (--ho-scalar)\n");
        std::abort();
    }
    scalar_t *const SFEM_RESTRICT rx = d.rx.data();
    scalar_t *const SFEM_RESTRICT ry = d.ry.data();
    scalar_t *const SFEM_RESTRICT rz = d.rz.data();
    scalar_t *const SFEM_RESTRICT rc = d.rc.data();
    const size_t                  scratch_n = packed_scratch_n(p);
    const Hex8Extras              opt(d);
    const int                     with_rc = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);

    // The generated Rhie-Chow kernel reads the coefficient out of the staged table instead of
    // rebuilding it per surface, so the table has to exist. It is cached on the state stamp, so
    // this is a no-op after the first call for a given state. The hand-written kernel computes the
    // coefficient inline and needs none of it, which is why this is conditional.
    if (sympy && with_rc) cvfem_hex8_build_rc_coeff(d, rho, mu);


#pragma omp parallel
    apply_residual_packed_defcor_range(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d, p, rho, mu, ugrad, limiter, venkat_c, sympy, rx, ry, rz, rc, scratch_n, with_rc);

    scalar_t *const fields[N_FIELDS] = {d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data()};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const smesh::idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        for (int f = 0; f < N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + (ptrdiff_t)f * p.n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            fields[f][dest] += sum;
        }
    }
}

// ISO IS A TEMPLATE PARAMETER, NOT A RUNTIME ENUM. DESIGN.md asks for the affine and the
// isoparametric kernels to be "logically separated (now they are mixed in with enum and
// booleans)", and this was the enum: GeomKind arrived as an argument and was tested per pack,
// inside the sweep. It also left the geometry undecided at the point where it matters -- the
// lane loop -- which is the shape of guard this file's own notes record costing 1.83x, and which
// the vectorisation gate now refuses outright. The caller picks the instantiation.
// The pack sweep, driven by a range. The `#pragma omp parallel` is in the launcher below; see
// kernels/cvfem_range.hpp for why DESIGN.md wants it there. The packs of a range touch only
// nodes this part owns -- that is what the packed layout is for -- so the parts need no
// synchronisation between them, and the ghost rows they do share are reduced afterwards in the
// launcher, which is the second and independent parallel loop.
template <bool ISO>
static SFEM_NOINLINE void apply_residual_packed_range(
        const cvfem_range packs,
        MeshData &d,
        PackedData &p,
        const scalar_t rho,
        const scalar_t mu,
        const KernelKind kernel_kind,
        scalar_t *const SFEM_RESTRICT rx,
        scalar_t *const SFEM_RESTRICT ry,
        scalar_t *const SFEM_RESTRICT rz,
        scalar_t *const SFEM_RESTRICT rc,
        const size_t scratch_n,
        const int with_rc) {
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_xyz =
                (ISO || with_rc)
                        ? thread_scratch<scalar_t>(3, with_rc ? packed_rc_n(p) : packed_xyz_n(p))
                        : nullptr;
        const ptrdiff_t xyz_n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgx = with_rc ? pack_xyz + 3 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgy = with_rc ? pack_xyz + 4 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgz = with_rc ? pack_xyz + 5 * xyz_n : nullptr;


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t                         n_pack_nodes = n_contiguous + n_ghost;
            const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const ptrdiff_t                         ghost_off    = p.ghost_ptr[pack];

            std::memset(pack_out, 0, (size_t)n_pack_nodes * (size_t)N_FIELDS * sizeof(scalar_t));

            fill_pack_fields(p, d, pack, n_contiguous, n_ghost, ghosts, pack_u);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y,
                                               pack_z, pack_pgx, pack_pgy, pack_pgz);

            if constexpr (ISO) {
                fill_pack_xyz(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);
                Hex8InputPack    in;
                Hex8CoordPack    xyz;
                Hex8ResidualPack outp;
                for (ptrdiff_t begin = e_start; begin < e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                    const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, e_end - begin));
                    gather_hex8_isoparam_simd_from_pack(
                            p.elems, pack_u, pack_x, pack_y, pack_z, begin, nlanes, in, xyz);
                    cvfem_hex8_ns_upwind_residual_isoparam_simd(rho, mu, xyz, in, outp);
                    scatter_hex8_simd_to_pack(p.elems, pack_out, begin, nlanes, outp);
                }
            } else if (kernel_kind == KernelKind::Sumfact) {
                alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE], cof2[CVFEM_HEX8_VEC_SIZE];
                alignas(ALIGN_BYTES) scalar_t cof3[CVFEM_HEX8_VEC_SIZE], cof4[CVFEM_HEX8_VEC_SIZE], cof5[CVFEM_HEX8_VEC_SIZE];
                alignas(ALIGN_BYTES) scalar_t cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE], cof8[CVFEM_HEX8_VEC_SIZE];
                alignas(ALIGN_BYTES) scalar_t det[CVFEM_HEX8_VEC_SIZE];
                Hex8InputPack    in;
                Hex8ResidualPack outp;
                Hex8RhieChowPack rcp;
                for (ptrdiff_t begin = e_start; begin < e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                    const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, e_end - begin));
                    gather_hex8_simd_from_pack(p.elems,
                                               pack_u,
                                               d,
                                               begin,
                                               nlanes,
                                               in,
                                               cof0,
                                               cof1,
                                               cof2,
                                               cof3,
                                               cof4,
                                               cof5,
                                               cof6,
                                               cof7,
                                               cof8,
                                               det);
                    if (with_rc) {
                        cvfem_hex8_gather_rc_from_pack(p.elems, pack_pgx, pack_pgy,
                                                       pack_pgz, begin, nlanes, rcp);
                    }
                    cvfem_hex8_ns_upwind_residual_sumfact_simd(
                            rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det, in, outp,
                            with_rc ? &rcp : nullptr, d.rhie_chow_scale);
                    scatter_hex8_simd_to_pack(p.elems, pack_out, begin, nlanes, outp);
                }
            } else {
                const bool sympy = kernel_uses_sympy_residual(kernel_kind);
                for (ptrdiff_t e = e_start; e < e_end; ++e) {
                    scalar_t ux_e[8], uy_e[8], uz_e[8], p_e[8], r[CVFEM_HEX8_N_DOF];
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)p.elems[a][e] * N_FIELDS;
                        ux_e[a]                              = u[0];
                        uy_e[a]                              = u[1];
                        uz_e[a]                              = u[2];
                        p_e[a]                               = u[3];
                    }
                    scalar_t adj[9], det;
                    load_hex8_adj(d, e, adj, &det);
                    if (sympy)
                        cvfem_hex8_ns_upwind_sympy_residual(rho, mu, adj, det, ux_e, uy_e, uz_e, p_e, r);
                    else
                        cvfem_hex8_ns_upwind_residual(rho, mu, adj, det, ux_e, uy_e, uz_e, p_e, r);

                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        scalar_t *const SFEM_RESTRICT out = pack_out + (ptrdiff_t)p.elems[a][e] * N_FIELDS;
                        out[0] += r[a * 4 + 0];
                        out[1] += r[a * 4 + 1];
                        out[2] += r[a * 4 + 2];
                        out[3] += r[a * 4 + 3];
                    }
                }
            }

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + k * N_FIELDS;
                const ptrdiff_t                     g   = owned + k;
                rx[g]                                   = out[0];
                ry[g]                                   = out[1];
                rz[g]                                   = out[2];
                rc[g]                                   = out[3];
            }

            scalar_t *const SFEM_RESTRICT gx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = p.ghost_buf.data() + 1 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = p.ghost_buf.data() + 3 * p.n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + (n_contiguous + k) * N_FIELDS;
                gx[ghost_off + k]                       = out[0];
                gy[ghost_off + k]                       = out[1];
                gz[ghost_off + k]                       = out[2];
                gc[ghost_off + k]                       = out[3];
            }
    }
}


template <bool ISO>
static SFEM_NOINLINE void apply_residual_packed(MeshData        &d,
                                                PackedData      &p,
                                                const scalar_t   rho,
                                                const scalar_t   mu,
                                                const KernelKind kernel_kind) {
    const scalar_t *const SFEM_RESTRICT ux = d.ux.data();
    const scalar_t *const SFEM_RESTRICT uy = d.uy.data();
    const scalar_t *const SFEM_RESTRICT uz = d.uz.data();
    const scalar_t *const SFEM_RESTRICT pr = d.p.data();
    scalar_t *const SFEM_RESTRICT       rx = d.rx.data();
    scalar_t *const SFEM_RESTRICT       ry = d.ry.data();
    scalar_t *const SFEM_RESTRICT       rz = d.rz.data();
    scalar_t *const SFEM_RESTRICT       rc = d.rc.data();
    const size_t                        scratch_n = packed_scratch_n(p);
    // Rhie-Chow needs the coordinates and the nodal gradient staged per pack, six arrays
    // rather than three, so slot 3 is sized for six when it is on. This is the solver's
    // own arrangement (packed_rc_n, cvfem_hex8_ns_packed.hpp) and the constant already
    // lives in the shared cvfem_hex8_pack_common.hpp.
    const int                           with_rc   = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);


#pragma omp parallel
    apply_residual_packed_range<ISO>(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d, p, rho, mu, kernel_kind, rx, ry, rz, rc, scratch_n, with_rc);

    scalar_t *const fields[N_FIELDS] = {d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data()};
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const smesh::idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        for (int f = 0; f < N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + f * p.n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            fields[f][dest] += sum;
        }
    }
}


// ISO IS A TEMPLATE PARAMETER, NOT A RUNTIME ENUM. DESIGN.md asks for the affine and the
// isoparametric kernels to be "logically separated (now they are mixed in with enum and
// booleans)", and this was the enum: GeomKind arrived as an argument and was tested per pack,
// inside the sweep. It also left the geometry undecided at the point where it matters -- the
// lane loop -- which is the shape of guard this file's own notes record costing 1.83x, and which
// the vectorisation gate now refuses outright. The caller picks the instantiation.
// The pack sweep, driven by a range. The `#pragma omp parallel` is in the launcher below; see
// kernels/cvfem_range.hpp for why DESIGN.md wants it there. The packs of a range touch only
// nodes this part owns -- that is what the packed layout is for -- so the parts need no
// synchronisation between them, and the ghost rows they do share are reduced afterwards in the
// launcher, which is the second and independent parallel loop.
template <bool ISO>
static SFEM_NOINLINE void assemble_jacobian_packed_range(
        const cvfem_range packs,
        MeshData &d,
        PackedData &p,
        BSR4 &b,
        const scalar_t rho,
        const scalar_t mu,
        const KernelKind kernel_kind,
        const size_t u_n,
        const size_t bsr_n,
        const int with_rc) {
        PhaseAcc acc;
        alignas(ALIGN_BYTES) scalar_t dense_ke[64 * 16];
        std::memset(dense_ke, 0, sizeof(dense_ke));
        scalar_t *const SFEM_RESTRICT pack_u          = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals_pack = thread_scratch<scalar_t>(2, bsr_n);
        scalar_t *const SFEM_RESTRICT pack_xyz =
                (ISO || with_rc)
                        ? thread_scratch<scalar_t>(3, with_rc ? packed_rc_n(p) : packed_xyz_n(p))
                        : nullptr;
        const ptrdiff_t xyz_n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgx = with_rc ? pack_xyz + 3 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgy = with_rc ? pack_xyz + 4 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgz = with_rc ? pack_xyz + 5 * xyz_n : nullptr;


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto                             &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto                             &lslots       = p.local_global_slot[(size_t)pack];
            const int                               local_nnz    = lrowptr.empty() ? 0 : lrowptr.back();

            double _t = phase_now();
            std::memset(local_vals_pack, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));
            if (g_breakdown) { const double _n = wall_time(); acc.t[PH_LOCAL_MEMSET] += _n - _t; _t = _n; }

            fill_pack_fields(p, d, pack, n_contiguous, n_ghost, ghosts, pack_u);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z,
                                               pack_pgx, pack_pgy, pack_pgz);
            if constexpr (ISO)
                fill_pack_xyz(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);
            if (g_breakdown) { const double _n = wall_time(); acc.t[PH_GATHER] += _n - _t; _t = _n; }

            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                scalar_t ux_e[8], uy_e[8], uz_e[8], p_e[8];
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)p.elems[a][e] * N_FIELDS;
                    ux_e[a]                              = u[0];
                    uy_e[a]                              = u[1];
                    uz_e[a]                              = u[2];
                    p_e[a]                               = u[3];
                }

                const int *const SFEM_RESTRICT slots =
                        g_kernel_only ? g_identity_slots : p.local_element_slot.data() + (size_t)e * 64;
                scalar_t *const SFEM_RESTRICT local_vals = g_kernel_only ? dense_ke : local_vals_pack;
                scalar_t adj[9], det;
                if constexpr (!ISO) load_hex8_adj(d, e, adj, &det);
                // The coordinates and the nodal gradient come out of the pack; the
                // Hex8RhieChow points at these locals, so they must outlive the call, which
                // they do.
                scalar_t     rc_x[8], rc_y[8], rc_z[8], rc_pgx[8], rc_pgy[8], rc_pgz[8];
                const Hex8RcConfig rcfg = cvfem_hex8_rc_config_for(d);
                Hex8RhieChow rc{};
                if (with_rc) {
                    gather_hex8_coords_from_pack(p.elems, pack_x, pack_y, pack_z, e, rc_x, rc_y, rc_z);
                    gather_hex8_coords_from_pack(p.elems, pack_pgx, pack_pgy, pack_pgz, e, rc_pgx, rc_pgy, rc_pgz);
                    rc = Hex8RhieChow{rc_x,    rc_y,  rc_z,  rc_pgx, rc_pgy, rc_pgz, rcfg.scale,
                                      nullptr, nullptr, nullptr, ux_e, uy_e, uz_e, rcfg.tau};
                }
                const scalar_t *const rc_p = with_rc ? p_e : nullptr;
                if constexpr (ISO) {
                    scalar_t x[8], y[8], z[8];
                    gather_hex8_coords_from_pack(p.elems, pack_x, pack_y, pack_z, e, x, y, z);
                    if (kernel_kind == KernelKind::Fd) {
                        scalar_t ke[CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF];
                        cvfem_hex8_ns_upwind_jacobian_fd_isoparam(rho, mu, x, y, z, ux_e, uy_e, uz_e, p_e, ke);
                        hex8_local_slots_to_bsr4(slots, ke, local_vals);
                    } else {
                        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<false>(
                                rho, mu, x, y, z, ux_e, uy_e, uz_e, slots, local_vals, rc, rc_p);
                    }
                } else if (kernel_kind == KernelKind::Sumfact) {
                    cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                            rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals, rc, rc_p);
                } else if (kernel_kind == KernelKind::Sympy) {
                    cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots(rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals);
                } else if (kernel_kind == KernelKind::SympyBlock) {
                    cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_blockwise(
                            rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals);
                } else if (kernel_kind == KernelKind::SympyRow) {
                    cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_rowwise(
                            rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals);
                } else if (kernel_kind == KernelKind::SympyFace) {
                    cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_facewise(
                            rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals);
                } else {
                    scalar_t ke[CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF];
                    cvfem_hex8_ns_upwind_jacobian_fd(rho, mu, adj, det, ux_e, uy_e, uz_e, p_e, ke);
                    hex8_local_slots_to_bsr4(slots, ke, local_vals);
                }
            }

            if (g_breakdown) { const double _n = wall_time(); acc.t[PH_KERNEL] += _n - _t; _t = _n; }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            if (!g_kernel_only)
                for (int t = 0; t < owned_nnz; ++t)
                    bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals_pack + (ptrdiff_t)t * 16);

            const ptrdiff_t ghost_off = p.ghost_ptr[pack];
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const ptrdiff_t local_i = n_contiguous + k;
                const int       begin   = lrowptr[(size_t)local_i];
                const int       end     = lrowptr[(size_t)local_i + 1];
                const ptrdiff_t dest    = p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k];
                std::memcpy(p.ghost_mat_val.data() + dest * 16,
                            local_vals_pack + (ptrdiff_t)begin * 16,
                            (size_t)(end - begin) * 16 * sizeof(scalar_t));
            }
            if (g_breakdown) acc.t[PH_LOCAL_TO_GLOBAL] += wall_time() - _t;
    }
        acc.flush();
}


template <bool ISO>
static SFEM_NOINLINE void assemble_jacobian_packed(MeshData        &d,
                                                   PackedData      &p,
                                                   BSR4            &b,
                                                   const scalar_t   rho,
                                                   const scalar_t   mu,
                                                   const KernelKind kernel_kind) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);
    // This sweep is scalar per element, not SIMD over a pack, so Rhie-Chow enters through
    // the same Hex8RhieChow the atomic assembly builds -- the only difference is that the
    // coordinates and the nodal gradient are read out of the pack rather than out of the
    // mesh. Both hand-written kernels below take the term; the generated ones and the
    // finite-difference reference do not, which the driver refuses rather than measures.
    const int    with_rc = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);


#pragma omp parallel
    assemble_jacobian_packed_range<ISO>(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d, p, b, rho, mu, kernel_kind, u_n, bsr_n, with_rc);

    const double _tg = phase_now();
    scalar_t *const SFEM_RESTRICT gvalues = b.values->data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const ptrdiff_t begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t end   = p.ghost_reduce_ptr[row + 1];
        for (ptrdiff_t j = begin; j < end; ++j) {
            const ptrdiff_t ghost_entry = p.ghost_reduce_idx[j];
            const ptrdiff_t k0          = p.ghost_mat_ptr[(size_t)ghost_entry];
            const ptrdiff_t k1          = p.ghost_mat_ptr[(size_t)ghost_entry + 1];
            for (ptrdiff_t t = k0; t < k1; ++t) {
                bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16], p.ghost_mat_val.data() + t * 16);
            }
        }
    }
    if (g_breakdown) g_phase[PH_GHOST] += wall_time() - _tg;
}


// THE EXACT HIGHER-ORDER ACTION is this same sweep with two extra fields staged, so it is this
// same function with two extra arguments rather than a second copy of two hundred lines.
//
//   ugrad  the state's nodal velocity gradient, the field the residual's correction reads;
//   vgrad  the DIRECTION's, reconstructed by its own pass before every matvec.
//
// Both null -- the default -- is the lagged action: the correction is a constant within the
// Newton step, contributes nothing to J, and this function computes exactly what it computed
// before the arguments existed. Passing them carries the correction's derivative, which is what
// makes the action exact for a higher-order residual and is what costs the extra pass.
// ISO IS A TEMPLATE PARAMETER, NOT A RUNTIME ENUM. DESIGN.md asks for the affine and the
// isoparametric kernels to be "logically separated (now they are mixed in with enum and
// booleans)", and this was the enum: GeomKind arrived as an argument and was tested per pack,
// inside the sweep. It also left the geometry undecided at the point where it matters -- the
// lane loop -- which is the shape of guard this file's own notes record costing 1.83x, and which
// the vectorisation gate now refuses outright. The caller picks the instantiation.
// The pack sweep, driven by a range. The `#pragma omp parallel` is in the launcher below; see
// kernels/cvfem_range.hpp for why DESIGN.md wants it there. The packs of a range touch only
// nodes this part owns -- that is what the packed layout is for -- so the parts need no
// synchronisation between them, and the ghost rows they do share are reduced afterwards in the
// launcher, which is the second and independent parallel loop.
template <bool ISO>
static SFEM_NOINLINE void apply_jacobian_action_packed_range(
        const cvfem_range packs,
        MeshData &d,
        PackedData &p,
        const scalar_t rho,
        const scalar_t mu,
        const scalar_t *const dir,
        scalar_t *const jv,
        const scalar_t *const SFEM_RESTRICT ugrad,
        const scalar_t *const SFEM_RESTRICT vgrad,
        const int limiter,
        const scalar_t venkat_c,
        const bool with_ho,
        const size_t scratch_n,
        const int with_rc,
        const bool with_qg,
        const size_t slot3_n) {
        // The breakdown covered packed assembly and the colored matvec but not this one --
        // the operator the solver's Krylov loop actually applies. Without it nothing here
        // could be attributed to a phase.
        PhaseAcc                      acc;
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_dir = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(2, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_xyz =
                (ISO || with_rc) ? thread_scratch<scalar_t>(3, slot3_n) : nullptr;
        const ptrdiff_t xyz_n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgx = with_rc ? pack_xyz + 3 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgy = with_rc ? pack_xyz + 4 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgz = with_rc ? pack_xyz + 5 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_qg  = with_qg ? thread_scratch<scalar_t>(4, packed_qg_n(p)) : nullptr;
        scalar_t *const SFEM_RESTRICT pack_qgx = pack_qg;
        scalar_t *const SFEM_RESTRICT pack_qgy = with_qg ? pack_qg + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_qgz = with_qg ? pack_qg + 2 * xyz_n : nullptr;


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t                         n_pack_nodes = n_contiguous + n_ghost;
            const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const ptrdiff_t                         ghost_off    = p.ghost_ptr[pack];

            double _t = phase_now();
            std::memset(pack_out, 0, (size_t)n_pack_nodes * (size_t)N_FIELDS * sizeof(scalar_t));
            if (g_breakdown) { const double _n = wall_time(); acc.t[PH_LOCAL_MEMSET] += _n - _t; _t = _n; }

            fill_pack_fields(p, d, pack, n_contiguous, n_ghost, ghosts, pack_u);
            fill_pack_interleaved(p, pack, n_contiguous, n_ghost, ghosts, dir, pack_dir);

            Hex8InputPack    u_pack;
            Hex8InputPack    du_pack;
            Hex8ResidualPack outp;
            Hex8CoordPack    xyz;
            Hex8RhieChowPack rcp;
            // The two gradient packs, staged exactly as apply_residual_packed_defcor stages its
            // one. The limiter and eps^2 live on the state pack because that is where the
            // correction's own kernel reads them; the direction pack carries only the field.
            Hex8UGradPack    hop, hovp;
            hop.limiter  = limiter;
            hop.venkat_c = venkat_c;
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z,
                                               pack_pgx, pack_pgy, pack_pgz);
            if (with_qg)
                cvfem_hex8_fill_pack_qgrad(p, d, pack, n_contiguous, n_ghost, ghosts, pack_qgx, pack_qgy, pack_qgz);
            if constexpr (ISO)
                fill_pack_xyz(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);
            if (g_breakdown) { const double _n = wall_time(); acc.t[PH_GATHER] += _n - _t; _t = _n; }

            for (ptrdiff_t begin = e_start; begin < e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, e_end - begin));
                if constexpr (ISO) {
                    gather_hex8_isoparam_action_simd_from_pack(p.elems,
                                                               pack_u,
                                                               pack_dir,
                                                               pack_x,
                                                               pack_y,
                                                               pack_z,
                                                               begin,
                                                               nlanes,
                                                               u_pack,
                                                               du_pack,
                                                               xyz);
                    cvfem_hex8_ns_upwind_jacobian_action_isoparam_simd(rho, mu, xyz, u_pack, du_pack, outp);
                } else {
                    alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE],
                            cof2[CVFEM_HEX8_VEC_SIZE];
                    alignas(ALIGN_BYTES) scalar_t cof3[CVFEM_HEX8_VEC_SIZE], cof4[CVFEM_HEX8_VEC_SIZE],
                            cof5[CVFEM_HEX8_VEC_SIZE];
                    alignas(ALIGN_BYTES) scalar_t cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE],
                            cof8[CVFEM_HEX8_VEC_SIZE];
                    alignas(ALIGN_BYTES) scalar_t det[CVFEM_HEX8_VEC_SIZE];
                    gather_hex8_action_simd_from_pack(p.elems,
                                                      pack_u,
                                                      pack_dir,
                                                      d,
                                                      begin,
                                                      nlanes,
                                                      u_pack,
                                                      du_pack,
                                                      cof0,
                                                      cof1,
                                                      cof2,
                                                      cof3,
                                                      cof4,
                                                      cof5,
                                                      cof6,
                                                      cof7,
                                                      cof8,
                                                      det);
                    if (with_rc) {
                        cvfem_hex8_gather_rc_from_pack(p.elems, pack_pgx, pack_pgy, pack_pgz,
                                                       begin, nlanes, rcp);
                        cvfem_hex8_gather_rc_coeff(d, begin, nlanes, rcp);
                    }
                    if (with_qg)
                        cvfem_hex8_gather_qg_from_pack(p.elems, pack_qgx, pack_qgy, pack_qgz, begin, nlanes, rcp);
                    if (with_ho) {
                        for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
                            const ptrdiff_t e = begin + lane;
                            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                                if (lane >= nlanes) {
                                    hop.x[a][lane] = hop.y[a][lane] = hop.z[a][lane] = scalar_t(0);
                                    for (int c = 0; c < 9; ++c) {
                                        hop.g[a][c][lane]  = scalar_t(0);
                                        hovp.g[a][c][lane] = scalar_t(0);
                                    }
                                    continue;
                                }
                                const smesh::idx_t gn = d.elems[a][e];
                                hop.x[a][lane] = scalar_t(d.points[0][gn]);
                                hop.y[a][lane] = scalar_t(d.points[1][gn]);
                                hop.z[a][lane] = scalar_t(d.points[2][gn]);
                                for (int c = 0; c < 9; ++c) {
                                    hop.g[a][c][lane]  = ugrad[(ptrdiff_t)gn * 9 + c];
                                    hovp.g[a][c][lane] = vgrad[(ptrdiff_t)gn * 9 + c];
                                }
                            }
                        }
                    }
                    cvfem_hex8_ns_upwind_jacobian_action_simd(rho,
                                                              mu,
                                                              cof0,
                                                              cof1,
                                                              cof2,
                                                              cof3,
                                                              cof4,
                                                              cof5,
                                                              cof6,
                                                              cof7,
                                                              cof8,
                                                              det,
                                                              u_pack,
                                                              du_pack,
                                                              outp,
                                                              with_rc ? &rcp : nullptr,
                                                              d.rhie_chow_scale,
                                                              with_qg,
                                                              scalar_t(0),
                                                              with_ho ? &hop : nullptr,
                                                              with_ho ? &hovp : nullptr);
                }
                scatter_hex8_simd_to_pack(p.elems, pack_out, begin, nlanes, outp);
            }
            if (g_breakdown) { const double _n = wall_time(); acc.t[PH_KERNEL] += _n - _t; _t = _n; }

            std::memcpy(jv + owned * N_FIELDS, pack_out, (size_t)n_contiguous * (size_t)N_FIELDS * sizeof(scalar_t));

            scalar_t *const SFEM_RESTRICT gx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = p.ghost_buf.data() + 1 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = p.ghost_buf.data() + 3 * p.n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + (n_contiguous + k) * N_FIELDS;
                gx[ghost_off + k]                       = out[0];
                gy[ghost_off + k]                       = out[1];
                gz[ghost_off + k]                       = out[2];
                gc[ghost_off + k]                       = out[3];
            }
            if (g_breakdown) acc.t[PH_LOCAL_TO_GLOBAL] += wall_time() - _t;
    }
        acc.flush();
}


template <bool ISO>
static SFEM_NOINLINE void apply_jacobian_action_packed(MeshData              &d,
                                                       PackedData            &p,
                                                       const scalar_t         rho,
                                                       const scalar_t         mu,
                                                       const scalar_t *const  dir,
                                                       scalar_t *const        jv,
                                                       const scalar_t *const SFEM_RESTRICT ugrad = nullptr,
                                                       const scalar_t *const SFEM_RESTRICT vgrad = nullptr,
                                                       const int              limiter = 0,
                                                       const scalar_t         venkat_c = scalar_t(0)) {
    const bool with_ho = ugrad != nullptr && vgrad != nullptr;
    // Hoisted out of the face loops -- see Hex8RhieChowPack::coeff. Its own cache key makes
    // this free after the first call, so --warmup absorbs the build and the timed loop
    // measures what the solver's Krylov iterations measure.
    cvfem_hex8_build_rc_coeff(d, rho, mu);
    const size_t scratch_n = packed_scratch_n(p);
    // Rhie-Chow staged per pack, exactly as apply_residual_packed does it and exactly as
    // the solver's own packed Jacobian does (cvfem_hex8_ns_packed.hpp). Slot 3 grows from
    // three arrays to six when it is on, and slot 4 carries the direction's reconstructed
    // gradient -- the term that makes this the *exact* Rhie-Chow Jacobian rather than the
    // frozen-gradient one the assembled matrix keeps.
    const int    with_rc   = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);
    const bool   with_qg   = with_rc && !d.qgx.empty();
    const size_t slot3_n   = with_rc ? packed_rc_n(p) : packed_xyz_n(p);


#pragma omp parallel
    apply_jacobian_action_packed_range<ISO>(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d, p, rho, mu, dir, jv, ugrad, vgrad, limiter, venkat_c, with_ho, scratch_n, with_rc, with_qg, slot3_n);

    const double _tg = phase_now();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const smesh::idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        scalar_t *const    out   = jv + (ptrdiff_t)dest * N_FIELDS;
        for (int f = 0; f < N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + f * p.n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            out[f] += sum;
        }
    }
    if (g_breakdown) g_phase[PH_GHOST] += wall_time() - _tg;
}


// ------------------------------------------------- the partially assembled Jacobian action
//
// The same operator as apply_jacobian_action_packed, reading the state out of a store built
// once per Newton step instead of gathering and re-deriving it on every matvec. What
// disappears from the per-matvec work, in order of size:
//
//   * fill_pack_fields -- the state u,p is never staged at all;
//   * the pgx/pgy/pgz half of cvfem_hex8_fill_pack_xyz_pgrad, and with it half of the 768
//     doubles cvfem_hex8_gather_rc_from_pack moves per SIMD group. The coordinates stay,
//     because the direction's reconstructed gradient is contracted against the edge vectors;
//   * per element, twelve mass fluxes, twelve Rhie-Chow corrections and twelve upwind
//     switches.
//
// and what appears is one contiguous SoA read of sixty scalars per element.
//
// Affine only. The store holds a per-element tangent built from one adjugate at the element
// centre, which is not what the isoparametric kernels evaluate -- they build an area vector
// per sub-control surface from a trilinear Jacobian. The driver refuses the combination
// rather than measuring a store that describes a different operator.
// The pack sweep, driven by a range. The `#pragma omp parallel` is in the launcher below; see
// kernels/cvfem_range.hpp for why DESIGN.md wants it there. The packs of a range touch only
// nodes this part owns -- that is what the packed layout is for -- so the parts need no
// synchronisation between them, and the ghost rows they do share are reduced afterwards in the
// launcher, which is the second and independent parallel loop.
static SFEM_NOINLINE void apply_jacobian_action_packed_pa_range(
        const cvfem_range packs,
        MeshData &d,
        PackedData &p,
        const scalar_t rho,
        const scalar_t mu,
        const scalar_t *const dir,
        scalar_t *const jv,
        const size_t scratch_n,
        const int with_rc,
        const bool with_qg) {
        PhaseAcc                      acc;
        scalar_t *const SFEM_RESTRICT pack_dir = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(2, scratch_n);
        // Three arrays in slot 3, not six: the nodal pressure gradient is inside the store.
        scalar_t *const SFEM_RESTRICT pack_xyz = with_qg ? thread_scratch<scalar_t>(3, packed_xyz_n(p)) : nullptr;
        const ptrdiff_t               xyz_n    = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x   = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y   = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z   = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_qg  = with_qg ? thread_scratch<scalar_t>(4, packed_qg_n(p)) : nullptr;
        scalar_t *const SFEM_RESTRICT pack_qgx = pack_qg;
        scalar_t *const SFEM_RESTRICT pack_qgy = with_qg ? pack_qg + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_qgz = with_qg ? pack_qg + 2 * xyz_n : nullptr;


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const ptrdiff_t e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const smesh::idx_t *const SFEM_RESTRICT ghosts    = &p.ghost_idx[p.ghost_ptr[pack]];
            const ptrdiff_t                         ghost_off = p.ghost_ptr[pack];

            double _t = phase_now();
            std::memset(pack_out, 0, (size_t)n_pack_nodes * (size_t)N_FIELDS * sizeof(scalar_t));
            if (g_breakdown) { const double _n = wall_time(); acc.t[PH_LOCAL_MEMSET] += _n - _t; _t = _n; }

            fill_pack_interleaved(p, pack, n_contiguous, n_ghost, ghosts, dir, pack_dir);
            if (with_qg) {
                fill_pack_xyz(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);
                cvfem_hex8_fill_pack_qgrad(p, d, pack, n_contiguous, n_ghost, ghosts, pack_qgx, pack_qgy, pack_qgz);
            }
            if (g_breakdown) { const double _n = wall_time(); acc.t[PH_GATHER] += _n - _t; _t = _n; }

            Hex8InputPack    du_pack;
            Hex8ResidualPack outp;
            Hex8RhieChowPack rcp;
            for (ptrdiff_t begin = e_start; begin < e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, e_end - begin));
                alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE],
                        cof2[CVFEM_HEX8_VEC_SIZE];
                alignas(ALIGN_BYTES) scalar_t cof3[CVFEM_HEX8_VEC_SIZE], cof4[CVFEM_HEX8_VEC_SIZE],
                        cof5[CVFEM_HEX8_VEC_SIZE];
                alignas(ALIGN_BYTES) scalar_t cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE],
                        cof8[CVFEM_HEX8_VEC_SIZE];
                alignas(ALIGN_BYTES) scalar_t det[CVFEM_HEX8_VEC_SIZE];
                gather_hex8_simd_from_pack(p.elems, pack_dir, d, begin, nlanes, du_pack, cof0, cof1, cof2, cof3,
                                           cof4, cof5, cof6, cof7, cof8, det);
                if (with_rc) cvfem_hex8_gather_rc_coeff(d, begin, nlanes, rcp);
                if (with_qg) {
                    cvfem_hex8_gather_qg_from_pack(p.elems, pack_qgx, pack_qgy, pack_qgz, begin, nlanes, rcp);
                }
                cvfem_hex8_ns_upwind_jacobian_action_pa_simd(rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6,
                                                             cof7, cof8, det, du_pack,
                                                             d.pa_tangent.data() + begin, d.nelements, outp,
                                                             with_rc ? &rcp : nullptr, d.rhie_chow_scale, with_qg);
                scatter_hex8_simd_to_pack(p.elems, pack_out, begin, nlanes, outp);
            }
            if (g_breakdown) { const double _n = wall_time(); acc.t[PH_KERNEL] += _n - _t; _t = _n; }

            std::memcpy(jv + owned * N_FIELDS, pack_out, (size_t)n_contiguous * (size_t)N_FIELDS * sizeof(scalar_t));
            scalar_t *const SFEM_RESTRICT gx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = p.ghost_buf.data() + 1 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = p.ghost_buf.data() + 3 * p.n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + (n_contiguous + k) * N_FIELDS;
                gx[ghost_off + k]                       = out[0];
                gy[ghost_off + k]                       = out[1];
                gz[ghost_off + k]                       = out[2];
                gc[ghost_off + k]                       = out[3];
            }
            if (g_breakdown) acc.t[PH_LOCAL_TO_GLOBAL] += wall_time() - _t;
    }
        acc.flush();
}


static SFEM_NOINLINE void apply_jacobian_action_packed_pa(MeshData             &d,
                                                          PackedData           &p,
                                                          const scalar_t        rho,
                                                          const scalar_t        mu,
                                                          const scalar_t *const dir,
                                                          scalar_t *const       jv) {
    const size_t scratch_n = packed_scratch_n(p);
    const int    with_rc   = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);
    const bool   with_qg   = with_rc && !d.qgx.empty();


#pragma omp parallel
    apply_jacobian_action_packed_pa_range(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d, p, rho, mu, dir, jv, scratch_n, with_rc, with_qg);

    const double _tg = phase_now();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const smesh::idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        scalar_t *const    out   = jv + (ptrdiff_t)dest * N_FIELDS;
        for (int f = 0; f < N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + f * p.n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            out[f] += sum;
        }
    }
    if (g_breakdown) g_phase[PH_GHOST] += wall_time() - _tg;
}

#endif  // CVFEM_HEX8_BEST_PACKED_HPP
