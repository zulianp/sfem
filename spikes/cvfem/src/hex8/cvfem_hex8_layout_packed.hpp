#ifndef CVFEM_HEX8_LAYOUT_PACKED_HPP
#define CVFEM_HEX8_LAYOUT_PACKED_HPP

// Packed layout: elements are grouped into packs, each pack accumulates into a
// thread-private buffer indexed by pack-local node ids, and the buffer is folded
// back into the global structure afterwards. Ghost rows -- nodes a pack touches
// but does not own -- are reduced in a second pass.
//
// The pack-local indexing lets the residual and Jacobian-action run 16-wide SIMD
// over elements. For assembly the local matrix is large enough that the round
// trip through it costs more than it saves; see cvfem_hex8_layout_colored.hpp
// and cvfem_hex8_layout_store.hpp.

#include "cvfem_hex8_layout_common.hpp"

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

static SFEM_NOINLINE void apply_residual_packed(MeshData        &d,
                                                PackedData      &p,
                                                const scalar_t   rho,
                                                const scalar_t   mu,
                                                const KernelKind kernel_kind,
                                                const GeomKind   geom_kind) {
    const scalar_t *const SFEM_RESTRICT ux = d.ux.data();
    const scalar_t *const SFEM_RESTRICT uy = d.uy.data();
    const scalar_t *const SFEM_RESTRICT uz = d.uz.data();
    const scalar_t *const SFEM_RESTRICT pr = d.p.data();
    scalar_t *const SFEM_RESTRICT       rx = d.rx.data();
    scalar_t *const SFEM_RESTRICT       ry = d.ry.data();
    scalar_t *const SFEM_RESTRICT       rz = d.rz.data();
    scalar_t *const SFEM_RESTRICT       rc = d.rc.data();
    const size_t                        scratch_n = packed_scratch_n(p);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_xyz =
                geom_kind == GeomKind::Isoparam ? thread_scratch<scalar_t>(3, packed_xyz_n(p)) : nullptr;
        const ptrdiff_t xyz_n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
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

            if (geom_kind == GeomKind::Isoparam) {
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
                    cvfem_hex8_ns_upwind_residual_sumfact_simd(
                            rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det, in, outp);
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


static SFEM_NOINLINE void assemble_jacobian_packed(MeshData        &d,
                                                   PackedData      &p,
                                                   BSR4            &b,
                                                   const scalar_t   rho,
                                                   const scalar_t   mu,
                                                   const KernelKind kernel_kind,
                                                   const GeomKind   geom_kind) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        PhaseAcc acc;
        alignas(ALIGN_BYTES) scalar_t dense_ke[64 * 16];
        std::memset(dense_ke, 0, sizeof(dense_ke));
        scalar_t *const SFEM_RESTRICT pack_u          = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals_pack = thread_scratch<scalar_t>(2, bsr_n);
        scalar_t *const SFEM_RESTRICT pack_xyz =
                geom_kind == GeomKind::Isoparam ? thread_scratch<scalar_t>(3, packed_xyz_n(p)) : nullptr;
        const ptrdiff_t xyz_n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
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
            if (geom_kind == GeomKind::Isoparam)
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
                if (geom_kind != GeomKind::Isoparam) load_hex8_adj(d, e, adj, &det);
                if (geom_kind == GeomKind::Isoparam) {
                    scalar_t x[8], y[8], z[8];
                    gather_hex8_coords_from_pack(p.elems, pack_x, pack_y, pack_z, e, x, y, z);
                    if (kernel_kind == KernelKind::Fd) {
                        scalar_t ke[CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF];
                        cvfem_hex8_ns_upwind_jacobian_fd_isoparam(rho, mu, x, y, z, ux_e, uy_e, uz_e, p_e, ke);
                        hex8_local_slots_to_bsr4(slots, ke, local_vals);
                    } else {
                        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<false>(
                                rho, mu, x, y, z, ux_e, uy_e, uz_e, slots, local_vals);
                    }
                } else if (kernel_kind == KernelKind::Sumfact) {
                    cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                            rho, mu, adj, det, ux_e, uy_e, uz_e, slots, local_vals);
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


static SFEM_NOINLINE void apply_jacobian_action_packed(MeshData              &d,
                                                       PackedData            &p,
                                                       const scalar_t         rho,
                                                       const scalar_t         mu,
                                                       const scalar_t *const  dir,
                                                       scalar_t *const        jv,
                                                       const GeomKind         geom_kind) {
    const size_t scratch_n = packed_scratch_n(p);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_dir = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(2, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_xyz =
                geom_kind == GeomKind::Isoparam ? thread_scratch<scalar_t>(3, packed_xyz_n(p)) : nullptr;
        const ptrdiff_t xyz_n = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
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
            fill_pack_interleaved(p, pack, n_contiguous, n_ghost, ghosts, dir, pack_dir);

            Hex8InputPack    u_pack;
            Hex8InputPack    du_pack;
            Hex8ResidualPack outp;
            Hex8CoordPack    xyz;
            if (geom_kind == GeomKind::Isoparam)
                fill_pack_xyz(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);
            for (ptrdiff_t begin = e_start; begin < e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, e_end - begin));
                if (geom_kind == GeomKind::Isoparam) {
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
                    cvfem_hex8_ns_upwind_jacobian_action_simd(
                            rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det, u_pack, du_pack, outp);
                }
                scatter_hex8_simd_to_pack(p.elems, pack_out, begin, nlanes, outp);
            }

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
        }
    }

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
}

#endif  // CVFEM_HEX8_LAYOUT_PACKED_HPP
