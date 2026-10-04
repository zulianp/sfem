#ifndef CVFEM_HEX8_STORE_LAUNCH_HPP
#define CVFEM_HEX8_STORE_LAUNCH_HPP

// THE STORE LAYOUT'S PACK BUILDER AND LAUNCHER.
//
// build_pack_store_crs computes the pack-local CRS this layout assembles into -- it reads the
// mesh and writes into PackedData, which is staging work by definition. assemble_jacobian_store
// is the launcher: it resolves the arrays, owns the parallel region, keeps the dynamic schedule
// this sweep was written with, and reduces the ghost rows afterwards.
//
// Both were in kernels/packed/ beside the kernel they serve, which is why that header included
// the bench's staging header.
#include "frontend/staging/cvfem_hex8_best_common.hpp"
#include "kernels/packed/cvfem_hex8_best_store.hpp"

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

#endif  // CVFEM_HEX8_STORE_LAUNCH_HPP
