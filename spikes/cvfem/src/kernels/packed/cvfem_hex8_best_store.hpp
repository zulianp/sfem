#ifndef CVFEM_HEX8_BEST_STORE_HPP
#define CVFEM_HEX8_BEST_STORE_HPP

// Store layout: a packed assembly whose pack-local matrix has its *owned* rows
// laid out in the global sparsity pattern. A pack's owned block is then a
// contiguous slice of the global BSR values and is flushed with one streaming
// memcpy, so every global block is written exactly once: no zero_bsr4 pass and no
// read-modify-write. Only the ghost rows still need a reduction.

#include "kernels/cvfem_phases.hpp"
#include "best/cvfem_hex8_best_common.hpp"
#include "kernels/cvfem_range.hpp"

// Build the "store" layout. Owned rows of a pack map 1:1 onto the contiguous
// global slice [rowptr_g[owned], rowptr_g[owned + n_contiguous]), so assembling
// a pack ends in one memcpy that writes every one of those blocks exactly once.
static void build_pack_store_crs(PackedData           &p,
                                 const ptrdiff_t       nelements,
                                 const count_t *rowptr_g,
                                 const idx_t   *colidx_g) {
    p.st_rowptr.resize((size_t)p.n_packs);
    p.st_owned_nnz.assign((size_t)p.n_packs, 0);
    p.st_local_nnz.assign((size_t)p.n_packs, 0);
    p.st_element_slot.assign((size_t)nelements * 64, 0);
    p.st_ghost_ptr.assign((size_t)p.n_ghost_entries + 1, 0);
    p.st_max_local_nnz = 0;

    std::vector<std::vector<pack_idx_t>> ghost_colidx((size_t)p.n_packs);

    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t owned        = p.owned_nodes_ptr[pack];
        const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
        const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
        const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
        const ptrdiff_t e_start      = pack * p.n_elements_per_pack;
        const ptrdiff_t e_end        = std::min(nelements, (pack + 1) * p.n_elements_per_pack);

        // compact adjacency, only needed for the ghost rows
        std::vector<std::vector<pack_idx_t>> adj((size_t)n_pack_nodes);
        for (ptrdiff_t e = e_start; e < e_end; ++e) {
            pack_idx_t ev[CVFEM_HEX8_N_NODES];
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) ev[a] = p.elems[a][e];
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                if ((ptrdiff_t)ev[a] < n_contiguous) continue;  // owned rows use the global pattern
                for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) adj[(size_t)ev[a]].push_back(ev[bnode]);
            }
        }

        auto &rowptr = p.st_rowptr[(size_t)pack];
        rowptr.assign((size_t)n_pack_nodes + 1, 0);
        for (ptrdiff_t i = 0; i < n_contiguous; ++i) {
            rowptr[(size_t)i + 1] = (int)(rowptr_g[owned + i + 1] - rowptr_g[owned + i]);
        }
        auto &gcol = ghost_colidx[(size_t)pack];
        for (ptrdiff_t i = n_contiguous; i < n_pack_nodes; ++i) {
            auto &row = adj[(size_t)i];
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            rowptr[(size_t)i + 1] = (int)row.size();
        }
        for (ptrdiff_t i = 0; i < n_pack_nodes; ++i) rowptr[(size_t)i + 1] += rowptr[(size_t)i];

        p.st_owned_nnz[(size_t)pack] = n_contiguous > 0 ? rowptr[(size_t)n_contiguous] : 0;
        p.st_local_nnz[(size_t)pack] = rowptr[(size_t)n_pack_nodes];
        p.st_max_local_nnz           = std::max(p.st_max_local_nnz, (ptrdiff_t)p.st_local_nnz[(size_t)pack]);

        gcol.resize((size_t)(p.st_local_nnz[(size_t)pack] - p.st_owned_nnz[(size_t)pack]));
        for (ptrdiff_t i = n_contiguous; i < n_pack_nodes; ++i) {
            const auto &row = adj[(size_t)i];
            std::memcpy(gcol.data() + (rowptr[(size_t)i] - p.st_owned_nnz[(size_t)pack]),
                        row.data(),
                        row.size() * sizeof(pack_idx_t));
        }

        // element -> local block id
        for (ptrdiff_t e = e_start; e < e_end; ++e) {
            int *const slots = p.st_element_slot.data() + (size_t)e * 64;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t local_row = p.elems[a][e];
                const int        row_begin = rowptr[(size_t)local_row];
                if ((ptrdiff_t)local_row < n_contiguous) {
                    const idx_t grow = (idx_t)(owned + (ptrdiff_t)local_row);
                    for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) {
                        const idx_t gcolb =
                                pack_local_to_global(p, pack, n_contiguous, p.elems[bnode][e]);
                        slots[a * 8 + bnode] =
                                row_begin + (int)(find_bsr_slot(rowptr_g, colidx_g, grow, gcolb) - rowptr_g[grow]);
                    }
                } else {
                    const int               row_len = rowptr[(size_t)local_row + 1] - row_begin;
                    const pack_idx_t *const row     = gcol.data() + (row_begin - p.st_owned_nnz[(size_t)pack]);
                    for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) {
                        slots[a * 8 + bnode] = row_begin + find_pack_col(p.elems[bnode][e], row, row_len);
                    }
                }
            }
        }
    }

    // ghost reduction table
    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - p.owned_nodes_ptr[pack];
        const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
        const ptrdiff_t ghost_off    = p.ghost_ptr[pack];
        const auto     &rowptr       = p.st_rowptr[(size_t)pack];
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
            const ptrdiff_t local_i                           = n_contiguous + k;
            p.st_ghost_ptr[(size_t)ghost_off + (size_t)k + 1] = rowptr[(size_t)local_i + 1] - rowptr[(size_t)local_i];
        }
    }
    for (ptrdiff_t i = 0; i < p.n_ghost_entries; ++i) p.st_ghost_ptr[(size_t)i + 1] += p.st_ghost_ptr[(size_t)i];

    const ptrdiff_t gnnz = p.st_ghost_ptr[(size_t)p.n_ghost_entries];
    p.st_ghost_slot.resize((size_t)gnnz);
    p.st_ghost_val.assign((size_t)gnnz * 16, 0.0);

    for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
        const ptrdiff_t n_contiguous = p.owned_nodes_ptr[pack + 1] - p.owned_nodes_ptr[pack];
        const ptrdiff_t n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
        const ptrdiff_t ghost_off    = p.ghost_ptr[pack];
        const auto     &rowptr       = p.st_rowptr[(size_t)pack];
        const auto     &gcol         = ghost_colidx[(size_t)pack];
        const int       owned_nnz    = p.st_owned_nnz[(size_t)pack];
        for (ptrdiff_t k = 0; k < n_ghost; ++k) {
            const ptrdiff_t    local_i = n_contiguous + k;
            const int          begin   = rowptr[(size_t)local_i];
            const int          end     = rowptr[(size_t)local_i + 1];
            const ptrdiff_t    dest    = p.st_ghost_ptr[(size_t)ghost_off + (size_t)k];
            const idx_t grow    = p.ghost_idx[(size_t)ghost_off + (size_t)k];
            for (int t = 0; t < end - begin; ++t) {
                const idx_t gcolb =
                        pack_local_to_global(p, pack, n_contiguous, gcol[(size_t)(begin - owned_nnz + t)]);
                p.st_ghost_slot[(size_t)dest + (size_t)t] = find_bsr_slot(rowptr_g, colidx_g, grow, gcolb);
            }
        }
    }
}

// Write-once assembly. Each pack accumulates into a cache-resident local matrix
// whose owned rows already carry the global sparsity pattern, then streams that
// block straight into the global BSR with a single memcpy. Every global block is
// written exactly once, so there is no zero_bsr4 pass and no read-modify-write.
// Only the ghost rows, which are shared between packs, need a reduction.
// ISO IS A TEMPLATE PARAMETER, NOT A RUNTIME ENUM. DESIGN.md asks for the affine and the
// isoparametric kernels to be "logically separated (now they are mixed in with enum and
// booleans)", and this was the enum: GeomKind arrived as an argument and was tested per pack,
// inside the sweep. It also left the geometry undecided at the point where it matters -- the
// lane loop -- which is the shape of guard this file's own notes record costing 1.83x, and which
// the vectorisation gate now refuses outright. The caller picks the instantiation.
// THE STORE SWEEP KEEPS ITS DYNAMIC SCHEDULE, which is why its range is one pack and its
// scratch arrives as arguments. The other pack sweeps take an equal static slice, because
// cvfem_range_split reproduces `schedule(static)` exactly; this one was written with
// `schedule(dynamic, 1)` -- it writes each matrix entry once rather than accumulating, so packs
// differ in how much work they carry and the balancing matters. A static split would have been a
// silent change to how the work is shared, so the launcher keeps the dynamic `omp for` and hands
// this kernel one pack at a time.
//
// The consequence is that the scratch cannot move in here: acquired per call it would be
// re-acquired once per pack rather than once per thread. It stays in the launcher's parallel
// region and is passed, which is what DESIGN.md's "only arguments that are actually used" asks
// for and what leaves no hidden per-thread state in the kernel.
template <bool ISO>
static SFEM_NOINLINE void assemble_jacobian_store_range(
        const cvfem_range packs,
        // The mesh and the pack are staging objects -- they own vectors and a shared_ptr to a
        // mesh -- so what this kernel reads out of them is what it takes. DESIGN.md: only
        // arguments that are actually used are passed.
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        const ptrdiff_t nelements,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t rhie_chow_scale,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        pack_idx_t **const SFEM_RESTRICT pack_elems,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const int *const SFEM_RESTRICT st_element_slot,
        const ptrdiff_t *const SFEM_RESTRICT st_ghost_ptr,
        scalar_t *const SFEM_RESTRICT st_ghost_val,
        const int *const SFEM_RESTRICT st_local_nnz,
        const int *const SFEM_RESTRICT st_owned_nnz,
        // BSR4 is a staging type (it owns a SharedBuffer and a graph); the kernel reads one
        // array out of it, so that is what it takes.
        const count_t *const SFEM_RESTRICT rowptr,
        const scalar_t   rho,
        const scalar_t   mu,
        const KernelKind kernel_kind,
        scalar_t *const SFEM_RESTRICT gvalues,
        const int                     with_rc,
        CVFEM_PHASE_ACC_PARAM
        scalar_t *const SFEM_RESTRICT pack_u,
        scalar_t *const SFEM_RESTRICT local_vals,
        scalar_t *const SFEM_RESTRICT pack_x,
        scalar_t *const SFEM_RESTRICT pack_y,
        scalar_t *const SFEM_RESTRICT pack_z,
        scalar_t *const SFEM_RESTRICT pack_pgx,
        scalar_t *const SFEM_RESTRICT pack_pgy,
        scalar_t *const SFEM_RESTRICT pack_pgz,
        // Resolved once per solve, in the launcher, not per element here. This parameter replaced
        // the cvfem_hex8_rc_config_for(d) call that used to sit in this body: that function takes
        // the mesh, which a kernel is not meant to name.
        const Hex8RcConfig &rc_cfg) {
    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const ptrdiff_t                         e_start      = pack * n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t                         owned        = owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts       = &ghost_idx[ghost_ptr[pack]];
            const int                               owned_nnz    = st_owned_nnz[(size_t)pack];
            const int                               local_nnz    = st_local_nnz[(size_t)pack];

            CVFEM_PHASE_CLOCK(_t);
            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));
            CVFEM_PHASE_MARK(acc, _t, PH_LOCAL_MEMSET);

            fill_pack_fields(owned_nodes_ptr, ux, uy, uz, pres, pack, n_contiguous, n_ghost, ghosts, pack_u);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(owned_nodes_ptr, points, pgx, pgy, pgz, with_rc, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z,
                                               pack_pgx, pack_pgy, pack_pgz);
            if constexpr (ISO)
                fill_pack_xyz(owned_nodes_ptr, points, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);
            CVFEM_PHASE_MARK(acc, _t, PH_GATHER);

            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                scalar_t ux_e[8], uy_e[8], uz_e[8], p_e[8];
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)pack_elems[a][e] * N_FIELDS;
                    ux_e[a]                               = u[0];
                    uy_e[a]                               = u[1];
                    uz_e[a]                               = u[2];
                    p_e[a]                                = u[3];
                }
                const int *const SFEM_RESTRICT slots = st_element_slot + (size_t)e * 64;

                scalar_t     rc_x[8], rc_y[8], rc_z[8], rc_pgx[8], rc_pgy[8], rc_pgz[8];
                const Hex8RcConfig rcfg = rc_cfg;
                Hex8RhieChow rc{};
                if (with_rc) {
                    gather_hex8_coords_from_pack(pack_elems, pack_x, pack_y, pack_z, e, rc_x, rc_y, rc_z);
                    gather_hex8_coords_from_pack(pack_elems, pack_pgx, pack_pgy, pack_pgz, e, rc_pgx, rc_pgy, rc_pgz);
                    rc = Hex8RhieChow{rc_x,    rc_y,    rc_z,    rc_pgx, rc_pgy, rc_pgz, rcfg.scale,
                                      nullptr, nullptr, nullptr, ux_e,   uy_e,   uz_e,   rcfg.tau};
                }
                const scalar_t *const rc_p = with_rc ? p_e : nullptr;

                if constexpr (ISO) {
                    scalar_t x[8], y[8], z[8];
                    gather_hex8_coords_from_pack(pack_elems, pack_x, pack_y, pack_z, e, x, y, z);
                    cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<false>(
                            rho, mu, x, y, z, ux_e, uy_e, uz_e, slots, local_vals, rc, rc_p);
                } else {
                    scalar_t adj[9], det;
                    load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
                    switch (kernel_kind) {
                        case KernelKind::Sympy:
                            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots(
                                    rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals);
                            break;
                        case KernelKind::SympyBlock:
                            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_blockwise(
                                    rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals);
                            break;
                        case KernelKind::SympyRow:
                            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_rowwise(
                                    rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals);
                            break;
                        case KernelKind::SympyFace:
                            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_facewise(
                                    rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals);
                            break;
                        case KernelKind::Sumfact:
                            if (g_dense_flush) {
                                alignas(ALIGN_BYTES) scalar_t ke[64 * 16] = {};
                                cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                                        rho, mu, adj, det, ux_e, uy_e, uz_e, g_identity_slots, ke, rc, rc_p);
                                hex8_blocks_to_slots(slots, ke, local_vals);
                            } else {
                                cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                                        rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals, rc, rc_p);
                            }
                            break;
                        default: {
                            scalar_t ke[CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF];
                            cvfem_hex8_ns_upwind_jacobian_fd(rho, mu, adj, det, ux_e, uy_e, uz_e, p_e, ke);
                            hex8_local_slots_to_bsr4(slots, ke, local_vals);
                            break;
                        }
                    }
                }
            }
            CVFEM_PHASE_MARK(acc, _t, PH_KERNEL);

            // owned rows: one streaming store over a contiguous global slice
            std::memcpy(gvalues + (ptrdiff_t)rowptr[owned] * 16, local_vals, (size_t)owned_nnz * 16 * sizeof(scalar_t));

            // ghost rows: park for the reduction below
            const ptrdiff_t ghost_off = ghost_ptr[pack];
            if (n_ghost > 0) {
                const ptrdiff_t dest = st_ghost_ptr[(size_t)ghost_off];
                const ptrdiff_t n    = st_ghost_ptr[(size_t)ghost_off + (size_t)n_ghost] - dest;
                std::memcpy(st_ghost_val + dest * 16,
                            local_vals + (ptrdiff_t)owned_nnz * 16,
                            (size_t)n * 16 * sizeof(scalar_t));
            }
            CVFEM_PHASE_MARK_LAST(acc, _t, PH_LOCAL_TO_GLOBAL);
    }
}


template <bool ISO>
static SFEM_NOINLINE void assemble_jacobian_store(MeshData        &d,
                                                  PackedData      &p,
                                                  BSR4            &b,
                                                  const scalar_t   rho,
                                                  const scalar_t   mu,
                                                  const KernelKind kernel_kind) {
    const size_t u_n   = packed_scratch_n(p.max_actual_nodes_per_pack);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.st_max_local_nnz, 1);

    scalar_t *const SFEM_RESTRICT gvalues = b.values->data();
    // As in assemble_jacobian_packed: this sweep is scalar per element, so Rhie-Chow enters
    // through a Hex8RhieChow built from pack data. Only the two hand-written kernels take
    // it; the generated ones are refused with the term rather than measured without it.
    const int                     with_rc = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);


#pragma omp parallel
    {
        CVFEM_PHASE_ACC(acc);
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);
        scalar_t *const SFEM_RESTRICT pack_xyz =
                (ISO || with_rc)
                        ? thread_scratch<scalar_t>(3, with_rc ? packed_rc_n(p.max_actual_nodes_per_pack) : packed_xyz_n(p.max_actual_nodes_per_pack))
                        : nullptr;
        const ptrdiff_t               xyz_n  = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgx = with_rc ? pack_xyz + 3 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgy = with_rc ? pack_xyz + 4 * xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_pgz = with_rc ? pack_xyz + 5 * xyz_n : nullptr;

#pragma omp for schedule(dynamic, 1)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack)
            assemble_jacobian_store_range<ISO>(cvfem_range{pack, pack + 1},
                                               d.adj_ptr, d.det_ptr, d.nelements, d.p.data(), d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.rhie_chow_scale, d.ux.data(), d.uy.data(), d.uz.data(), p.elems, p.ghost_idx, p.ghost_ptr, p.n_elements_per_pack, p.owned_nodes_ptr, p.st_element_slot.data(), p.st_ghost_ptr.data(), p.st_ghost_val.data(), p.st_local_nnz.data(), p.st_owned_nnz.data(), b.rowptr, rho, mu, kernel_kind, gvalues, with_rc,
                                               CVFEM_PHASE_ACC_ARG pack_u, local_vals, pack_x, pack_y, pack_z,
                                               pack_pgx, pack_pgy, pack_pgz,
            cvfem_hex8_rc_config_for(d));
        CVFEM_PHASE_FLUSH(acc);
    }


    CVFEM_PHASE_CLOCK(_tg);
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const ptrdiff_t begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t end   = p.ghost_reduce_ptr[row + 1];
        for (ptrdiff_t j = begin; j < end; ++j) {
            const ptrdiff_t ghost_entry = p.ghost_reduce_idx[j];
            const ptrdiff_t k0          = p.st_ghost_ptr[(size_t)ghost_entry];
            const ptrdiff_t k1          = p.st_ghost_ptr[(size_t)ghost_entry + 1];
            for (ptrdiff_t t = k0; t < k1; ++t) {
                bsr4_add16(&gvalues[(ptrdiff_t)p.st_ghost_slot[(size_t)t] * 16], p.st_ghost_val.data() + t * 16);
            }
        }
    }
    CVFEM_PHASE_GLOBAL(_tg, PH_GHOST);
}

#endif  // CVFEM_HEX8_BEST_STORE_HPP
