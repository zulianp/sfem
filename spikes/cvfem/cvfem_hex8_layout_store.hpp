#ifndef CVFEM_HEX8_LAYOUT_STORE_HPP
#define CVFEM_HEX8_LAYOUT_STORE_HPP

// Store layout: a packed assembly whose pack-local matrix has its *owned* rows
// laid out in the global sparsity pattern. A pack's owned block is then a
// contiguous slice of the global BSR values and is flushed with one streaming
// memcpy, so every global block is written exactly once: no zero_bsr4 pass and no
// read-modify-write. Only the ghost rows still need a reduction.

#include "cvfem_hex8_layout_common.hpp"

// Build the "store" layout. Owned rows of a pack map 1:1 onto the contiguous
// global slice [rowptr_g[owned], rowptr_g[owned + n_contiguous]), so assembling
// a pack ends in one memcpy that writes every one of those blocks exactly once.
static void build_pack_store_crs(PackedData           &p,
                                 const ptrdiff_t       nelements,
                                 const smesh::count_t *rowptr_g,
                                 const smesh::idx_t   *colidx_g) {
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
                    const smesh::idx_t grow = (smesh::idx_t)(owned + (ptrdiff_t)local_row);
                    for (int bnode = 0; bnode < CVFEM_HEX8_N_NODES; ++bnode) {
                        const smesh::idx_t gcolb =
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
            const smesh::idx_t grow    = p.ghost_idx[(size_t)ghost_off + (size_t)k];
            for (int t = 0; t < end - begin; ++t) {
                const smesh::idx_t gcolb =
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
static SFEM_NOINLINE void assemble_jacobian_store(MeshData        &d,
                                                  PackedData      &p,
                                                  BSR4            &b,
                                                  const scalar_t   rho,
                                                  const scalar_t   mu,
                                                  const KernelKind kernel_kind,
                                                  const GeomKind   geom_kind) {
    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.st_max_local_nnz, 1);

    scalar_t *const SFEM_RESTRICT gvalues = b.values->data();

#pragma omp parallel
    {
        PhaseAcc                      acc;
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);
        scalar_t *const SFEM_RESTRICT pack_xyz =
                geom_kind == GeomKind::Isoparam ? thread_scratch<scalar_t>(3, packed_xyz_n(p)) : nullptr;
        const ptrdiff_t               xyz_n  = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;

#pragma omp for schedule(dynamic, 1)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const int                               owned_nnz    = p.st_owned_nnz[(size_t)pack];
            const int                               local_nnz    = p.st_local_nnz[(size_t)pack];

            double _t = phase_now();
            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));
            if (g_breakdown) { const double _n = wall_time(); acc.t[PH_LOCAL_MEMSET] += _n - _t; _t = _n; }

            fill_pack_fields(p, d, pack, n_contiguous, n_ghost, ghosts, pack_u);
            if (geom_kind == GeomKind::Isoparam)
                fill_pack_xyz(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);
            if (g_breakdown) { const double _n = wall_time(); acc.t[PH_GATHER] += _n - _t; _t = _n; }

            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                scalar_t ux_e[8], uy_e[8], uz_e[8], p_e[8];
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)p.elems[a][e] * N_FIELDS;
                    ux_e[a]                               = u[0];
                    uy_e[a]                               = u[1];
                    uz_e[a]                               = u[2];
                    p_e[a]                                = u[3];
                }
                const int *const SFEM_RESTRICT slots = p.st_element_slot.data() + (size_t)e * 64;

                if (geom_kind == GeomKind::Isoparam) {
                    scalar_t x[8], y[8], z[8];
                    gather_hex8_coords_from_pack(p.elems, pack_x, pack_y, pack_z, e, x, y, z);
                    cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<false>(
                            rho, mu, x, y, z, ux_e, uy_e, uz_e, slots, local_vals);
                } else {
                    scalar_t adj[9], det;
                    load_hex8_adj(d, e, adj, &det);
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
                                        rho, mu, adj, det, ux_e, uy_e, uz_e, g_identity_slots, ke);
                                hex8_blocks_to_slots(slots, ke, local_vals);
                            } else {
                                cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                                        rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals);
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
            if (g_breakdown) { const double _n = wall_time(); acc.t[PH_KERNEL] += _n - _t; _t = _n; }

            // owned rows: one streaming store over a contiguous global slice
            std::memcpy(gvalues + (ptrdiff_t)b.rowptr[owned] * 16, local_vals, (size_t)owned_nnz * 16 * sizeof(scalar_t));

            // ghost rows: park for the reduction below
            const ptrdiff_t ghost_off = p.ghost_ptr[pack];
            if (n_ghost > 0) {
                const ptrdiff_t dest = p.st_ghost_ptr[(size_t)ghost_off];
                const ptrdiff_t n    = p.st_ghost_ptr[(size_t)ghost_off + (size_t)n_ghost] - dest;
                std::memcpy(p.st_ghost_val.data() + dest * 16,
                            local_vals + (ptrdiff_t)owned_nnz * 16,
                            (size_t)n * 16 * sizeof(scalar_t));
            }
            if (g_breakdown) acc.t[PH_LOCAL_TO_GLOBAL] += wall_time() - _t;
        }
        acc.flush();
    }

    const double _tg = phase_now();
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
    if (g_breakdown) g_phase[PH_GHOST] += wall_time() - _tg;
}

#endif  // CVFEM_HEX8_LAYOUT_STORE_HPP
