#pragma once

// THE TET4 BENCHMARK'S RETIRED SWEEPS.
//
// DESIGN.md's correction: "the micro-kernel selector must be removed. Only the best
// micro-kernels need to be used (given the results in Grace), so there should be only one per
// kernel. The rest is moved to subpar". The TET4 benchmark had THIRTEEN arms behind its own
// --kernel flag -- a private enum, not the HEX8 one -- and perf/ held no TET4 row at all, so
// every arm was unmeasured on this hardware and so was the default. jobs/tet4_arms.sbatch
// measured all thirteen across all three operations; perf/tet4_arms_grace.txt is the record.
//
// Packed layout, n=96 (10,616,832 elements), 72 threads, best of three, MELEM/s:
//
//   residual   hand-written     1503.1   the eleven generated arrangements cluster 1347-1360
//   action     hand-written     1063.5   the generated arrangements cluster 1003-1010
//   assembly   sympy_block_simd  168.1   sympy_row_simd 167.1, current_slots 150.5,
//                                        plain hand-written 136.2
//
// So TET4 splits the opposite way from the HEX8 residual: the generated arrangements win the
// ASSEMBLY by 1.12x over the best hand-written variant and 1.23x over the plain one, and lose
// the two matrix-free operations by 1.11x and 1.05x. The default was sympy_row_simd, which is
// therefore the wrong kernel for two of the three operations -- it gives up 10% on the residual
// and 5% on the action, and on the assembly it ties the winner rather than being it.
//
// THE SPREAD THIS JOB CAN RESOLVE IS 0.6%, and it came for free: `current` and `current_slots`
// differ only in the assembly, so the residual and the action ran identical code under both
// labels and came back 0.6% and 0.4% apart. Every gap above beats that except sympy_block_simd
// against sympy_row_simd, which is a tie decided by the measured order.
//
// What survives: cvfem_tet4_ns_upwind_apply_{atomic,packed} for the residual,
// cvfem_tet4_ns_upwind_jacobian_action_{atomic,packed} for the action, assemble_bsr4_atomic for
// the atomic assembly and assemble_bsr4_packed_sympy_block_simd for the packed one.
//
// assemble_bsr4_atomic's own generated twin is here on the PACKED pair's evidence rather than
// its own: the job measured the packed layout, where the hand-written kernel wins 136.2 to
// 115.0, and the two layouts differ in the scatter and not in the element kernel. Said plainly
// rather than left implied.
//
// This header is included FROM THE DRIVER, after its MeshData, PackedData and BSR4 are
// declared, because these sweeps take those types.


static SFEM_NOINLINE void assemble_bsr4_packed(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t                         n_pack_nodes = n_contiguous + n_ghost;
            const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto                             &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto                             &lcolidx      = p.local_colidx[(size_t)pack];
            const auto                             &lslots       = p.local_global_slot[(size_t)pack];
            const int                               local_nnz    = lrowptr.empty() ? 0 : lrowptr.back();

            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + k * N_FIELDS;
                const ptrdiff_t               g   = owned + k;
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
                dst[3]                            = d.p[g];
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + (n_contiguous + k) * N_FIELDS;
                const smesh::idx_t            g   = ghosts[k];
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
                dst[3]                            = d.p[g];
            }

            alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                jacobian_element_packed(d, e, ev, pack_u, rho, mu, Ke);
                tet4_local_to_global_bsr4<false>(ev, Ke, lrowptr.data(), lcolidx.data(), local_vals);
            }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            for (int t = 0; t < owned_nnz; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
            }

            const ptrdiff_t ghost_off = p.ghost_ptr[pack];
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const ptrdiff_t local_i = n_contiguous + k;
                const int       begin   = lrowptr[(size_t)local_i];
                const int       end     = lrowptr[(size_t)local_i + 1];
                const ptrdiff_t dest    = p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k];
                std::memcpy(p.ghost_mat_val.data() + dest * 16, local_vals + (ptrdiff_t)begin * 16, (size_t)(end - begin) * 16 * sizeof(scalar_t));
            }
            (void)n_pack_nodes;
        }
    }

    scalar_t *const SFEM_RESTRICT gvalues = b.values->data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        (void)p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        for (ptrdiff_t j = begin; j < end; ++j) {
            const ptrdiff_t ghost_entry = p.ghost_reduce_idx[j];
            const ptrdiff_t k0          = p.ghost_mat_ptr[(size_t)ghost_entry];
            const ptrdiff_t k1          = p.ghost_mat_ptr[(size_t)ghost_entry + 1];
            for (ptrdiff_t t = k0; t < k1; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}


static SFEM_NOINLINE void assemble_bsr4_atomic_sympy(MeshData &d, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);
    const ptrdiff_t                    ne     = d.nelements;
    smesh::idx_t **const SFEM_RESTRICT elems  = d.elems;
    scalar_t *const SFEM_RESTRICT      values = b.values->data();

#pragma omp parallel
    {
        alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < ne; ++e) {
            const smesh::idx_t ev[4] = {elems[0][e], elems[1][e], elems[2][e], elems[3][e]};
            jacobian_element_global_sympy(d, e, ev, rho, mu, Ke);
            tet4_local_to_global_bsr4<true>(ev, Ke, b.rowptr, b.colidx, values);
        }
    }
}


static SFEM_NOINLINE void assemble_bsr4_packed_sympy(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t                         n_pack_nodes = n_contiguous + n_ghost;
            const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto                             &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto                             &lcolidx      = p.local_colidx[(size_t)pack];
            const auto                             &lslots       = p.local_global_slot[(size_t)pack];
            const int                               local_nnz    = lrowptr.empty() ? 0 : lrowptr.back();

            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + k * N_FIELDS;
                const ptrdiff_t               g   = owned + k;
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
                dst[3]                            = d.p[g];
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + (n_contiguous + k) * N_FIELDS;
                const smesh::idx_t            g   = ghosts[k];
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
                dst[3]                            = d.p[g];
            }

            alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                jacobian_element_packed_sympy(d, e, ev, pack_u, rho, mu, Ke);
                tet4_local_to_global_bsr4<false>(ev, Ke, lrowptr.data(), lcolidx.data(), local_vals);
            }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            for (int t = 0; t < owned_nnz; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
            }

            const ptrdiff_t ghost_off = p.ghost_ptr[pack];
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const ptrdiff_t local_i = n_contiguous + k;
                const int       begin   = lrowptr[(size_t)local_i];
                const int       end     = lrowptr[(size_t)local_i + 1];
                const ptrdiff_t dest    = p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k];
                std::memcpy(p.ghost_mat_val.data() + dest * 16,
                            local_vals + (ptrdiff_t)begin * 16,
                            (size_t)(end - begin) * 16 * sizeof(scalar_t));
            }
            (void)n_pack_nodes;
        }
    }

    scalar_t *const SFEM_RESTRICT gvalues = b.values->data();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        (void)p.ghost_reduce_dest[row];
        const ptrdiff_t begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t end   = p.ghost_reduce_ptr[row + 1];
        for (ptrdiff_t j = begin; j < end; ++j) {
            const ptrdiff_t ghost_entry = p.ghost_reduce_idx[j];
            const ptrdiff_t k0          = p.ghost_mat_ptr[(size_t)ghost_entry];
            const ptrdiff_t k1          = p.ghost_mat_ptr[(size_t)ghost_entry + 1];
            for (ptrdiff_t t = k0; t < k1; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}

template <bool Sympy, bool DirectToSlots, bool Blockwise, bool Facewise>
static SFEM_NOINLINE void assemble_bsr4_packed_slots_variant(MeshData       &d,
                                                             PackedData     &p,
                                                             BSR4           &b,
                                                             const scalar_t  rho,
                                                             const scalar_t  mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t                   e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                   e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                   owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                   n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                   n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const smesh::idx_t *const         ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto                       &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto                       &lslots       = p.local_global_slot[(size_t)pack];
            const int                         local_nnz    = lrowptr.empty() ? 0 : lrowptr.back();

            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + k * N_FIELDS;
                const ptrdiff_t               g   = owned + k;
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
                dst[3]                            = d.p[g];
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + (n_contiguous + k) * N_FIELDS;
                const smesh::idx_t            g   = ghosts[k];
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
                dst[3]                            = d.p[g];
            }

            if constexpr (DirectToSlots) {
                for (ptrdiff_t e = e_start; e < e_end; ++e) {
                    const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                    if constexpr (Facewise) {
                        jacobian_element_packed_sympy_add_slots_facewise(
                                d, e, ev, pack_u, rho, mu, p.local_element_slot.data() + (size_t)e * 16, local_vals);
                    } else if constexpr (Blockwise) {
                        jacobian_element_packed_sympy_add_slots_blockwise(
                                d, e, ev, pack_u, rho, mu, p.local_element_slot.data() + (size_t)e * 16, local_vals);
                    } else {
                        jacobian_element_packed_sympy_add_slots(
                                d, e, ev, pack_u, rho, mu, p.local_element_slot.data() + (size_t)e * 16, local_vals);
                    }
                }
            } else {
                alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
                for (ptrdiff_t e = e_start; e < e_end; ++e) {
                    const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                    if constexpr (Sympy) {
                        jacobian_element_packed_sympy(d, e, ev, pack_u, rho, mu, Ke);
                    } else {
                        jacobian_element_packed(d, e, ev, pack_u, rho, mu, Ke);
                    }
                    tet4_local_slots_to_bsr4(p.local_element_slot.data() + (size_t)e * 16, Ke, local_vals);
                }
            }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            for (int t = 0; t < owned_nnz; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
            }

            const ptrdiff_t ghost_off = p.ghost_ptr[pack];
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const ptrdiff_t local_i = n_contiguous + k;
                const int       begin   = lrowptr[(size_t)local_i];
                const int       end     = lrowptr[(size_t)local_i + 1];
                const ptrdiff_t dest    = p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k];
                std::memcpy(p.ghost_mat_val.data() + dest * 16,
                            local_vals + (ptrdiff_t)begin * 16,
                            (size_t)(end - begin) * 16 * sizeof(scalar_t));
            }
        }
    }

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
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}


static SFEM_NOINLINE void assemble_bsr4_packed_current_slots(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    assemble_bsr4_packed_slots_variant<false, false, false, false>(d, p, b, rho, mu);
}


static SFEM_NOINLINE void assemble_bsr4_packed_sympy_slots(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    assemble_bsr4_packed_slots_variant<true, false, false, false>(d, p, b, rho, mu);
}


static SFEM_NOINLINE void assemble_bsr4_packed_sympy_direct(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    assemble_bsr4_packed_slots_variant<true, true, false, false>(d, p, b, rho, mu);
}


static SFEM_NOINLINE void assemble_bsr4_packed_sympy_block(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    assemble_bsr4_packed_slots_variant<true, true, true, false>(d, p, b, rho, mu);
}


static SFEM_NOINLINE void assemble_bsr4_packed_sympy_face(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    assemble_bsr4_packed_slots_variant<true, true, false, true>(d, p, b, rho, mu);
}


static SFEM_NOINLINE void assemble_bsr4_packed_sympy_simd(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);
        alignas(ALIGN_BYTES) scalar_t Ke_vec[CVFEM_N_DOF * CVFEM_N_DOF * SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t a0[SIMD_SIZE], a1[SIMD_SIZE], a2[SIMD_SIZE], a3[SIMD_SIZE], a4[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t a5[SIMD_SIZE], a6[SIMD_SIZE], a7[SIMD_SIZE], a8[SIMD_SIZE], detv[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t ux0[SIMD_SIZE], ux1[SIMD_SIZE], ux2[SIMD_SIZE], ux3[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t uy0[SIMD_SIZE], uy1[SIMD_SIZE], uy2[SIMD_SIZE], uy3[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t uz0[SIMD_SIZE], uz1[SIMD_SIZE], uz2[SIMD_SIZE], uz3[SIMD_SIZE];

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t           e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t           e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t           owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t           n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t           n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const smesh::idx_t *const ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto               &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto               &lslots       = p.local_global_slot[(size_t)pack];
            const int                 local_nnz    = lrowptr.empty() ? 0 : lrowptr.back();

            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + k * N_FIELDS;
                const ptrdiff_t               g   = owned + k;
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
                dst[3]                            = d.p[g];
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + (n_contiguous + k) * N_FIELDS;
                const smesh::idx_t            g   = ghosts[k];
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
                dst[3]                            = d.p[g];
            }

            ptrdiff_t e = e_start;
            for (; e + SIMD_SIZE <= e_end; e += SIMD_SIZE) {
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    const ptrdiff_t  ee    = e + lane;
                    const pack_idx_t ev[4] = {p.elems[0][ee], p.elems[1][ee], p.elems[2][ee], p.elems[3][ee]};
                    const scalar_t *const u0 = pack_u + (ptrdiff_t)ev[0] * N_FIELDS;
                    const scalar_t *const u1 = pack_u + (ptrdiff_t)ev[1] * N_FIELDS;
                    const scalar_t *const u2 = pack_u + (ptrdiff_t)ev[2] * N_FIELDS;
                    const scalar_t *const u3 = pack_u + (ptrdiff_t)ev[3] * N_FIELDS;
                    a0[lane]                 = scalar_t(d.adj[0][ee]);
                    a1[lane]                 = scalar_t(d.adj[1][ee]);
                    a2[lane]                 = scalar_t(d.adj[2][ee]);
                    a3[lane]                 = scalar_t(d.adj[3][ee]);
                    a4[lane]                 = scalar_t(d.adj[4][ee]);
                    a5[lane]                 = scalar_t(d.adj[5][ee]);
                    a6[lane]                 = scalar_t(d.adj[6][ee]);
                    a7[lane]                 = scalar_t(d.adj[7][ee]);
                    a8[lane]                 = scalar_t(d.adj[8][ee]);
                    detv[lane]               = scalar_t(d.det[ee]);
                    ux0[lane]                = u0[0];
                    ux1[lane]                = u1[0];
                    ux2[lane]                = u2[0];
                    ux3[lane]                = u3[0];
                    uy0[lane]                = u0[1];
                    uy1[lane]                = u1[1];
                    uy2[lane]                = u2[1];
                    uy3[lane]                = u3[1];
                    uz0[lane]                = u0[2];
                    uz1[lane]                = u1[2];
                    uz2[lane]                = u2[2];
                    uz3[lane]                = u3[2];
                }
                cvfem_tet4_ns_upwind_sympy_jacobian_dense_vector(rho,
                                                                  mu,
                                                                  a0,
                                                                  a1,
                                                                  a2,
                                                                  a3,
                                                                  a4,
                                                                  a5,
                                                                  a6,
                                                                  a7,
                                                                  a8,
                                                                  detv,
                                                                  ux0,
                                                                  ux1,
                                                                  ux2,
                                                                  ux3,
                                                                  uy0,
                                                                  uy1,
                                                                  uy2,
                                                                  uy3,
                                                                  uz0,
                                                                  uz1,
                                                                  uz2,
                                                                  uz3,
                                                                  Ke_vec);
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    tet4_local_slots_to_bsr4_vec_lane(p.local_element_slot.data() + (size_t)(e + lane) * 16, Ke_vec, lane, local_vals);
                }
            }
            for (; e < e_end; ++e) {
                const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
                jacobian_element_packed_sympy(d, e, ev, pack_u, rho, mu, Ke);
                tet4_local_slots_to_bsr4(p.local_element_slot.data() + (size_t)e * 16, Ke, local_vals);
            }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            for (int t = 0; t < owned_nnz; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
            }

            const ptrdiff_t ghost_off = p.ghost_ptr[pack];
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const ptrdiff_t local_i = n_contiguous + k;
                const int       begin   = lrowptr[(size_t)local_i];
                const int       end     = lrowptr[(size_t)local_i + 1];
                const ptrdiff_t dest    = p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k];
                std::memcpy(p.ghost_mat_val.data() + dest * 16,
                            local_vals + (ptrdiff_t)begin * 16,
                            (size_t)(end - begin) * 16 * sizeof(scalar_t));
            }
        }
    }

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
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}


static SFEM_NOINLINE void assemble_bsr4_packed_sympy_simd_clean(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);
        alignas(ALIGN_BYTES) scalar_t Ke_vec[CVFEM_N_DOF * CVFEM_N_DOF * SIMD_SIZE];

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t           e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t           e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t           owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t           n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t           n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const smesh::idx_t *const ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto               &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto               &lslots       = p.local_global_slot[(size_t)pack];
            const int                 local_nnz    = lrowptr.empty() ? 0 : lrowptr.back();

            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + k * N_FIELDS;
                const ptrdiff_t               g   = owned + k;
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + (n_contiguous + k) * N_FIELDS;
                const smesh::idx_t            g   = ghosts[k];
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
            }

            ptrdiff_t e = e_start;
            for (; e + SIMD_SIZE <= e_end; e += SIMD_SIZE) {
#if CVFEM_SIMD_BYTES == 64
#define CVFEM_VEC8(X) scalar_v{X(0), X(1), X(2), X(3), X(4), X(5), X(6), X(7)}
#define CVFEM_VEC(X)  CVFEM_VEC8(X)
#elif CVFEM_SIMD_BYTES == 32
#define CVFEM_VEC4(X) scalar_v{X(0), X(1), X(2), X(3)}
#define CVFEM_VEC(X)  CVFEM_VEC4(X)
#elif CVFEM_SIMD_BYTES == 16
#define CVFEM_VEC2(X) scalar_v{X(0), X(1)}
#define CVFEM_VEC(X)  CVFEM_VEC2(X)
#else
#error "assemble_bsr4_packed_sympy_simd_clean supports 16, 32, or 64 byte SIMD vectors"
#endif
#define CVFEM_ADJ_LANE(K, L) scalar_t(d.adj[K][e + (L)])
#define CVFEM_DET_LANE(L)    scalar_t(d.det[e + (L)])
#define CVFEM_U_LANE(N, F, L) pack_u[(ptrdiff_t)p.elems[N][e + (L)] * N_FIELDS + (F)]
#define CVFEM_ADJ0(L) CVFEM_ADJ_LANE(0, L)
#define CVFEM_ADJ1(L) CVFEM_ADJ_LANE(1, L)
#define CVFEM_ADJ2(L) CVFEM_ADJ_LANE(2, L)
#define CVFEM_ADJ3(L) CVFEM_ADJ_LANE(3, L)
#define CVFEM_ADJ4(L) CVFEM_ADJ_LANE(4, L)
#define CVFEM_ADJ5(L) CVFEM_ADJ_LANE(5, L)
#define CVFEM_ADJ6(L) CVFEM_ADJ_LANE(6, L)
#define CVFEM_ADJ7(L) CVFEM_ADJ_LANE(7, L)
#define CVFEM_ADJ8(L) CVFEM_ADJ_LANE(8, L)
#define CVFEM_DET(L)  CVFEM_DET_LANE(L)
#define CVFEM_UX0(L)  CVFEM_U_LANE(0, 0, L)
#define CVFEM_UX1(L)  CVFEM_U_LANE(1, 0, L)
#define CVFEM_UX2(L)  CVFEM_U_LANE(2, 0, L)
#define CVFEM_UX3(L)  CVFEM_U_LANE(3, 0, L)
#define CVFEM_UY0(L)  CVFEM_U_LANE(0, 1, L)
#define CVFEM_UY1(L)  CVFEM_U_LANE(1, 1, L)
#define CVFEM_UY2(L)  CVFEM_U_LANE(2, 1, L)
#define CVFEM_UY3(L)  CVFEM_U_LANE(3, 1, L)
#define CVFEM_UZ0(L)  CVFEM_U_LANE(0, 2, L)
#define CVFEM_UZ1(L)  CVFEM_U_LANE(1, 2, L)
#define CVFEM_UZ2(L)  CVFEM_U_LANE(2, 2, L)
#define CVFEM_UZ3(L)  CVFEM_U_LANE(3, 2, L)
                const scalar_v a0   = CVFEM_VEC(CVFEM_ADJ0);
                const scalar_v a1   = CVFEM_VEC(CVFEM_ADJ1);
                const scalar_v a2   = CVFEM_VEC(CVFEM_ADJ2);
                const scalar_v a3   = CVFEM_VEC(CVFEM_ADJ3);
                const scalar_v a4   = CVFEM_VEC(CVFEM_ADJ4);
                const scalar_v a5   = CVFEM_VEC(CVFEM_ADJ5);
                const scalar_v a6   = CVFEM_VEC(CVFEM_ADJ6);
                const scalar_v a7   = CVFEM_VEC(CVFEM_ADJ7);
                const scalar_v a8   = CVFEM_VEC(CVFEM_ADJ8);
                const scalar_v detv = CVFEM_VEC(CVFEM_DET);
                const scalar_v ux0  = CVFEM_VEC(CVFEM_UX0);
                const scalar_v ux1  = CVFEM_VEC(CVFEM_UX1);
                const scalar_v ux2  = CVFEM_VEC(CVFEM_UX2);
                const scalar_v ux3  = CVFEM_VEC(CVFEM_UX3);
                const scalar_v uy0  = CVFEM_VEC(CVFEM_UY0);
                const scalar_v uy1  = CVFEM_VEC(CVFEM_UY1);
                const scalar_v uy2  = CVFEM_VEC(CVFEM_UY2);
                const scalar_v uy3  = CVFEM_VEC(CVFEM_UY3);
                const scalar_v uz0  = CVFEM_VEC(CVFEM_UZ0);
                const scalar_v uz1  = CVFEM_VEC(CVFEM_UZ1);
                const scalar_v uz2  = CVFEM_VEC(CVFEM_UZ2);
                const scalar_v uz3  = CVFEM_VEC(CVFEM_UZ3);
#undef CVFEM_UZ3
#undef CVFEM_UZ2
#undef CVFEM_UZ1
#undef CVFEM_UZ0
#undef CVFEM_UY3
#undef CVFEM_UY2
#undef CVFEM_UY1
#undef CVFEM_UY0
#undef CVFEM_UX3
#undef CVFEM_UX2
#undef CVFEM_UX1
#undef CVFEM_UX0
#undef CVFEM_DET
#undef CVFEM_ADJ8
#undef CVFEM_ADJ7
#undef CVFEM_ADJ6
#undef CVFEM_ADJ5
#undef CVFEM_ADJ4
#undef CVFEM_ADJ3
#undef CVFEM_ADJ2
#undef CVFEM_ADJ1
#undef CVFEM_ADJ0
#undef CVFEM_U_LANE
#undef CVFEM_DET_LANE
#undef CVFEM_ADJ_LANE
#undef CVFEM_VEC
#if CVFEM_SIMD_BYTES == 64
#undef CVFEM_VEC8
#elif CVFEM_SIMD_BYTES == 32
#undef CVFEM_VEC4
#elif CVFEM_SIMD_BYTES == 16
#undef CVFEM_VEC2
#endif
                cvfem_tet4_ns_upwind_sympy_jacobian_dense_vector_values(rho,
                                                                         mu,
                                                                         a0,
                                                                         a1,
                                                                         a2,
                                                                         a3,
                                                                         a4,
                                                                         a5,
                                                                         a6,
                                                                         a7,
                                                                         a8,
                                                                         detv,
                                                                         ux0,
                                                                         ux1,
                                                                         ux2,
                                                                         ux3,
                                                                         uy0,
                                                                         uy1,
                                                                         uy2,
                                                                         uy3,
                                                                         uz0,
                                                                         uz1,
                                                                         uz2,
                                                                         uz3,
                                                                         Ke_vec);
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    tet4_local_slots_to_bsr4_vec_lane(p.local_element_slot.data() + (size_t)(e + lane) * 16, Ke_vec, lane, local_vals);
                }
            }
            for (; e < e_end; ++e) {
                const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                alignas(ALIGN_BYTES) scalar_t Ke[CVFEM_N_DOF * CVFEM_N_DOF];
                jacobian_element_packed_sympy(d, e, ev, pack_u, rho, mu, Ke);
                tet4_local_slots_to_bsr4(p.local_element_slot.data() + (size_t)e * 16, Ke, local_vals);
            }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            for (int t = 0; t < owned_nnz; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
            }

            const ptrdiff_t ghost_off = p.ghost_ptr[pack];
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const ptrdiff_t local_i = n_contiguous + k;
                const int       begin   = lrowptr[(size_t)local_i];
                const int       end     = lrowptr[(size_t)local_i + 1];
                const ptrdiff_t dest    = p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k];
                std::memcpy(p.ghost_mat_val.data() + dest * 16,
                            local_vals + (ptrdiff_t)begin * 16,
                            (size_t)(end - begin) * 16 * sizeof(scalar_t));
            }
        }
    }

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
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}


static SFEM_INLINE void assemble_bsr4_packed_sympy_row_simd_pack(MeshData &d,
                                                                 PackedData &p,
                                                                 BSR4 &b,
                                                                 const scalar_t rho,
                                                                 const scalar_t mu,
                                                                 const ptrdiff_t pack,
                                                                 scalar_t *const SFEM_RESTRICT pack_u,
                                                                 scalar_t *const SFEM_RESTRICT local_vals) {
    const ptrdiff_t           e_start      = pack * p.n_elements_per_pack;
    const ptrdiff_t           e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
    const ptrdiff_t           owned        = p.owned_nodes_ptr[pack];
    const ptrdiff_t           n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
    const ptrdiff_t           n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
    const smesh::idx_t *const ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
    const auto               &lrowptr      = p.local_rowptr[(size_t)pack];
    const auto               &lslots       = p.local_global_slot[(size_t)pack];

    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        scalar_t *const SFEM_RESTRICT dst = pack_u + k * N_FIELDS;
        const ptrdiff_t               g   = owned + k;
        dst[0]                            = d.ux[g];
        dst[1]                            = d.uy[g];
        dst[2]                            = d.uz[g];
        dst[3]                            = d.p[g];
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        scalar_t *const SFEM_RESTRICT dst = pack_u + (n_contiguous + k) * N_FIELDS;
        const smesh::idx_t            g   = ghosts[k];
        dst[0]                            = d.ux[g];
        dst[1]                            = d.uy[g];
        dst[2]                            = d.uz[g];
        dst[3]                            = d.p[g];
    }

    ptrdiff_t e = e_start;
    for (; e + SIMD_SIZE <= e_end; e += SIMD_SIZE) {
        scalar_v adj0 = scalar_v{}, adj1 = scalar_v{}, adj2 = scalar_v{};
        scalar_v adj3 = scalar_v{}, adj4 = scalar_v{}, adj5 = scalar_v{};
        scalar_v adj6 = scalar_v{}, adj7 = scalar_v{}, adj8 = scalar_v{}, detv = scalar_v{};
        scalar_v ux0 = scalar_v{}, ux1 = scalar_v{}, ux2 = scalar_v{}, ux3 = scalar_v{};
        scalar_v uy0 = scalar_v{}, uy1 = scalar_v{}, uy2 = scalar_v{}, uy3 = scalar_v{};
        scalar_v uz0 = scalar_v{}, uz1 = scalar_v{}, uz2 = scalar_v{}, uz3 = scalar_v{};

#pragma unroll
        for (int lane = 0; lane < SIMD_SIZE; ++lane) {
            const ptrdiff_t  ee  = e + lane;
            const pack_idx_t ev0 = p.elems[0][ee];
            const pack_idx_t ev1 = p.elems[1][ee];
            const pack_idx_t ev2 = p.elems[2][ee];
            const pack_idx_t ev3 = p.elems[3][ee];

            const scalar_t *const u0 = pack_u + (ptrdiff_t)ev0 * N_FIELDS;
            const scalar_t *const u1 = pack_u + (ptrdiff_t)ev1 * N_FIELDS;
            const scalar_t *const u2 = pack_u + (ptrdiff_t)ev2 * N_FIELDS;
            const scalar_t *const u3 = pack_u + (ptrdiff_t)ev3 * N_FIELDS;

            adj0[lane] = scalar_t(d.adj[0][ee]);
            adj1[lane] = scalar_t(d.adj[1][ee]);
            adj2[lane] = scalar_t(d.adj[2][ee]);
            adj3[lane] = scalar_t(d.adj[3][ee]);
            adj4[lane] = scalar_t(d.adj[4][ee]);
            adj5[lane] = scalar_t(d.adj[5][ee]);
            adj6[lane] = scalar_t(d.adj[6][ee]);
            adj7[lane] = scalar_t(d.adj[7][ee]);
            adj8[lane] = scalar_t(d.adj[8][ee]);
            detv[lane] = scalar_t(d.det[ee]);

            ux0[lane] = u0[0];
            ux1[lane] = u1[0];
            ux2[lane] = u2[0];
            ux3[lane] = u3[0];
            uy0[lane] = u0[1];
            uy1[lane] = u1[1];
            uy2[lane] = u2[1];
            uy3[lane] = u3[1];
            uz0[lane] = u0[2];
            uz1[lane] = u1[2];
            uz2[lane] = u2[2];
            uz3[lane] = u3[2];
        }

        cvfem_tet4_ns_upwind_sympy_jacobian_add_bsr_slots_rowwise_vector_values(rho,
                                                                                mu,
                                                                                adj0,
                                                                                adj1,
                                                                                adj2,
                                                                                adj3,
                                                                                adj4,
                                                                                adj5,
                                                                                adj6,
                                                                                adj7,
                                                                                adj8,
                                                                                detv,
                                                                                ux0,
                                                                                ux1,
                                                                                ux2,
                                                                                ux3,
                                                                                uy0,
                                                                                uy1,
                                                                                uy2,
                                                                                uy3,
                                                                                uz0,
                                                                                uz1,
                                                                                uz2,
                                                                                uz3,
                                                                                p.local_element_slot.data() + (size_t)e * 16,
                                                                                local_vals);
    }
    for (; e < e_end; ++e) {
        const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
        jacobian_element_packed_sympy_add_slots_blockwise(
                d, e, ev, pack_u, rho, mu, p.local_element_slot.data() + (size_t)e * 16, local_vals);
    }

    scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
    const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
    for (int t = 0; t < owned_nnz; ++t) {
        cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
    }

    const ptrdiff_t ghost_off = p.ghost_ptr[pack];
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        const ptrdiff_t local_i = n_contiguous + k;
        const int       begin   = lrowptr[(size_t)local_i];
        const int       end     = lrowptr[(size_t)local_i + 1];
        const ptrdiff_t dest    = p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k];
        std::memcpy(p.ghost_mat_val.data() + dest * 16,
                    local_vals + (ptrdiff_t)begin * 16,
                    (size_t)(end - begin) * 16 * sizeof(scalar_t));
    }
}


static SFEM_NOINLINE void assemble_bsr4_packed_sympy_row_simd(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const auto &lrowptr   = p.local_rowptr[(size_t)pack];
            const int   local_nnz = lrowptr.empty() ? 0 : lrowptr.back();
            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));
            assemble_bsr4_packed_sympy_row_simd_pack(d, p, b, rho, mu, pack, pack_u, local_vals);
        }
    }

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
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}


static SFEM_NOINLINE void assemble_bsr4_packed_sympy_row_simd_fused(MeshData &d,
                                                                    PackedData &p,
                                                                    BSR4 &b,
                                                                    const scalar_t rho,
                                                                    const scalar_t mu) {
    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);
        scalar_t *const SFEM_RESTRICT gvalues    = b.values->data();

#pragma omp for schedule(static)
        for (ptrdiff_t i = 0; i < b.nnz * 16; ++i) {
            gvalues[i] = scalar_t(0);
        }

        scalar_t *const SFEM_RESTRICT ghost_values = p.ghost_mat_val.data();
        const ptrdiff_t               n_ghost_vals = (ptrdiff_t)p.ghost_mat_val.size();
#pragma omp for schedule(static)
        for (ptrdiff_t i = 0; i < n_ghost_vals; ++i) {
            ghost_values[i] = scalar_t(0);
        }

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const auto &lrowptr   = p.local_rowptr[(size_t)pack];
            const int   local_nnz = lrowptr.empty() ? 0 : lrowptr.back();
            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));
            assemble_bsr4_packed_sympy_row_simd_pack(d, p, b, rho, mu, pack, pack_u, local_vals);
        }

#pragma omp for schedule(static)
        for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
            const ptrdiff_t begin = p.ghost_reduce_ptr[row];
            const ptrdiff_t end   = p.ghost_reduce_ptr[row + 1];
            for (ptrdiff_t j = begin; j < end; ++j) {
                const ptrdiff_t ghost_entry = p.ghost_reduce_idx[j];
                const ptrdiff_t k0          = p.ghost_mat_ptr[(size_t)ghost_entry];
                const ptrdiff_t k1          = p.ghost_mat_ptr[(size_t)ghost_entry + 1];
                for (ptrdiff_t t = k0; t < k1; ++t) {
                    cvfem_bsr4_add16_vec(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                         ghost_values + t * 16);
                }
            }
        }
    }
}


static SFEM_NOINLINE void assemble_bsr4_packed_sympy_face_simd(MeshData &d, PackedData &p, BSR4 &b, const scalar_t rho, const scalar_t mu) {
    zero_bsr4(b);

    const size_t u_n   = packed_scratch_n(p);
    const size_t bsr_n = 16 * (size_t)std::max<ptrdiff_t>(p.max_local_nnz, 1);

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u     = thread_scratch<scalar_t>(0, u_n);
        scalar_t *const SFEM_RESTRICT local_vals = thread_scratch<scalar_t>(2, bsr_n);
        alignas(ALIGN_BYTES) scalar_t face_ke[CVFEM_TET4_NS_UPWIND_FACE_SIMD_MAX_NNZ * SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t a0[SIMD_SIZE], a1[SIMD_SIZE], a2[SIMD_SIZE], a3[SIMD_SIZE], a4[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t a5[SIMD_SIZE], a6[SIMD_SIZE], a7[SIMD_SIZE], a8[SIMD_SIZE], detv[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t ux0[SIMD_SIZE], ux1[SIMD_SIZE], ux2[SIMD_SIZE], ux3[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t uy0[SIMD_SIZE], uy1[SIMD_SIZE], uy2[SIMD_SIZE], uy3[SIMD_SIZE];
        alignas(ALIGN_BYTES) scalar_t uz0[SIMD_SIZE], uz1[SIMD_SIZE], uz2[SIMD_SIZE], uz3[SIMD_SIZE];

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < p.n_packs; ++pack) {
            const ptrdiff_t           e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t           e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t           owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t           n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t           n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const smesh::idx_t *const ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const auto               &lrowptr      = p.local_rowptr[(size_t)pack];
            const auto               &lslots       = p.local_global_slot[(size_t)pack];
            const int                 local_nnz    = lrowptr.empty() ? 0 : lrowptr.back();

            std::memset(local_vals, 0, (size_t)local_nnz * 16 * sizeof(scalar_t));

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + k * N_FIELDS;
                const ptrdiff_t               g   = owned + k;
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
                dst[3]                            = d.p[g];
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + (n_contiguous + k) * N_FIELDS;
                const smesh::idx_t            g   = ghosts[k];
                dst[0]                            = d.ux[g];
                dst[1]                            = d.uy[g];
                dst[2]                            = d.uz[g];
                dst[3]                            = d.p[g];
            }

            ptrdiff_t e = e_start;
            for (; e + SIMD_SIZE <= e_end; e += SIMD_SIZE) {
                for (int lane = 0; lane < SIMD_SIZE; ++lane) {
                    const ptrdiff_t  ee    = e + lane;
                    const pack_idx_t ev[4] = {p.elems[0][ee], p.elems[1][ee], p.elems[2][ee], p.elems[3][ee]};
                    const scalar_t *const u0 = pack_u + (ptrdiff_t)ev[0] * N_FIELDS;
                    const scalar_t *const u1 = pack_u + (ptrdiff_t)ev[1] * N_FIELDS;
                    const scalar_t *const u2 = pack_u + (ptrdiff_t)ev[2] * N_FIELDS;
                    const scalar_t *const u3 = pack_u + (ptrdiff_t)ev[3] * N_FIELDS;
                    a0[lane]                 = scalar_t(d.adj[0][ee]);
                    a1[lane]                 = scalar_t(d.adj[1][ee]);
                    a2[lane]                 = scalar_t(d.adj[2][ee]);
                    a3[lane]                 = scalar_t(d.adj[3][ee]);
                    a4[lane]                 = scalar_t(d.adj[4][ee]);
                    a5[lane]                 = scalar_t(d.adj[5][ee]);
                    a6[lane]                 = scalar_t(d.adj[6][ee]);
                    a7[lane]                 = scalar_t(d.adj[7][ee]);
                    a8[lane]                 = scalar_t(d.adj[8][ee]);
                    detv[lane]               = scalar_t(d.det[ee]);
                    ux0[lane]                = u0[0];
                    ux1[lane]                = u1[0];
                    ux2[lane]                = u2[0];
                    ux3[lane]                = u3[0];
                    uy0[lane]                = u0[1];
                    uy1[lane]                = u1[1];
                    uy2[lane]                = u2[1];
                    uy3[lane]                = u3[1];
                    uz0[lane]                = u0[2];
                    uz1[lane]                = u1[2];
                    uz2[lane]                = u2[2];
                    uz3[lane]                = u3[2];
                }

#define CVFEM_FACE_SIMD(FACE)                                                                                                      \
    do {                                                                                                                           \
        cvfem_tet4_ns_upwind_sympy_jacobian_face##FACE##_vector(                                                                   \
                rho, mu, a0, a1, a2, a3, a4, a5, a6, a7, a8, detv, ux0, ux1, ux2, ux3, uy0, uy1, uy2, uy3, uz0, uz1, uz2, uz3,    \
                face_ke);                                                                                                          \
        for (int lane = 0; lane < SIMD_SIZE; ++lane) {                                                                             \
            cvfem_tet4_ns_upwind_sympy_jacobian_face##FACE##_vector_lane_to_bsr_slots(                                             \
                    p.local_element_slot.data() + (size_t)(e + lane) * 16, face_ke, lane, local_vals);                             \
        }                                                                                                                          \
    } while (0)

                CVFEM_FACE_SIMD(0);
                CVFEM_FACE_SIMD(1);
                CVFEM_FACE_SIMD(2);
                CVFEM_FACE_SIMD(3);
                CVFEM_FACE_SIMD(4);
                CVFEM_FACE_SIMD(5);

#undef CVFEM_FACE_SIMD
            }
            for (; e < e_end; ++e) {
                const pack_idx_t ev[4] = {p.elems[0][e], p.elems[1][e], p.elems[2][e], p.elems[3][e]};
                jacobian_element_packed_sympy_add_slots_facewise(
                        d, e, ev, pack_u, rho, mu, p.local_element_slot.data() + (size_t)e * 16, local_vals);
            }

            scalar_t *const SFEM_RESTRICT gvalues   = b.values->data();
            const int                     owned_nnz = n_contiguous > 0 ? lrowptr[(size_t)n_contiguous] : 0;
            for (int t = 0; t < owned_nnz; ++t) {
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)lslots[(size_t)t] * 16], local_vals + (ptrdiff_t)t * 16);
            }

            const ptrdiff_t ghost_off = p.ghost_ptr[pack];
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const ptrdiff_t local_i = n_contiguous + k;
                const int       begin   = lrowptr[(size_t)local_i];
                const int       end     = lrowptr[(size_t)local_i + 1];
                const ptrdiff_t dest    = p.ghost_mat_ptr[(size_t)ghost_off + (size_t)k];
                std::memcpy(p.ghost_mat_val.data() + dest * 16,
                            local_vals + (ptrdiff_t)begin * 16,
                            (size_t)(end - begin) * 16 * sizeof(scalar_t));
            }
        }
    }

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
                cvfem_bsr4_add16(&gvalues[(ptrdiff_t)p.ghost_mat_slot[(size_t)t] * 16],
                                 p.ghost_mat_val.data() + t * 16);
            }
        }
    }
}


static SFEM_NOINLINE void cvfem_tet4_ns_upwind_apply_sympy_atomic(MeshData &d, const scalar_t rho, const scalar_t mu) {
    reset_residual(d);
    const ptrdiff_t ne = d.nelements;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t begin = 0; begin < ne; begin += VEC_SIZE) {
        const int        nlanes = int(std::min<ptrdiff_t>(ne - begin, VEC_SIZE));
        Tet4InputPack    in;
        Tet4ResidualPack out;
        gather_tet4_pack_global(d, begin, nlanes, in);
        run_microkernel_sympy(d, rho, mu, begin, nlanes, in, out);
        scatter_tet4_pack_global(d, begin, nlanes, out);
    }
}


static SFEM_NOINLINE void cvfem_tet4_ns_upwind_apply_sympy_packed(MeshData &d,
                                                                  PackedData &p,
                                                                  const scalar_t rho,
                                                                  const scalar_t mu) {
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

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + k * N_FIELDS;
                const ptrdiff_t               g   = owned + k;
                dst[0]                            = ux[g];
                dst[1]                            = uy[g];
                dst[2]                            = uz[g];
                dst[3]                            = pr[g];
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                scalar_t *const SFEM_RESTRICT dst = pack_u + (n_contiguous + k) * N_FIELDS;
                const smesh::idx_t            g   = ghosts[k];
                dst[0]                            = ux[g];
                dst[1]                            = uy[g];
                dst[2]                            = uz[g];
                dst[3]                            = pr[g];
            }

            for (ptrdiff_t begin = e_start; begin < e_end; begin += VEC_SIZE) {
                const int        nlanes = int(MIN((ptrdiff_t)VEC_SIZE, e_end - begin));
                Tet4InputPack    in;
                Tet4ResidualPack out;
                gather_tet4_pack_local(p.elems, pack_u, begin, nlanes, in);
                run_microkernel_sympy(d, rho, mu, begin, nlanes, in, out);
                scatter_tet4_pack_local(p.elems, pack_out, begin, nlanes, out);
            }

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                const scalar_t *const SFEM_RESTRICT po = pack_out + k * N_FIELDS;
                const ptrdiff_t                     g  = owned + k;
                rx[g]                                  = po[0];
                ry[g]                                  = po[1];
                rz[g]                                  = po[2];
                rc[g]                                  = po[3];
            }

            scalar_t *const SFEM_RESTRICT gx = p.ghost_buf.data() + 0 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gy = p.ghost_buf.data() + 1 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gz = p.ghost_buf.data() + 2 * p.n_ghost_entries;
            scalar_t *const SFEM_RESTRICT gc = p.ghost_buf.data() + 3 * p.n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT po = pack_out + (n_contiguous + k) * N_FIELDS;
                gx[ghost_off + k]                      = po[0];
                gy[ghost_off + k]                      = po[1];
                gz[ghost_off + k]                      = po[2];
                gc[ghost_off + k]                      = po[3];
            }
        }
    }

    scalar_t *const out_fields[N_FIELDS] = {d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data()};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t row = 0; row < p.n_ghost_reduce_rows; ++row) {
        const smesh::idx_t dest  = p.ghost_reduce_dest[row];
        const ptrdiff_t    begin = p.ghost_reduce_ptr[row];
        const ptrdiff_t    end   = p.ghost_reduce_ptr[row + 1];
        for (int f = 0; f < N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = p.ghost_buf.data() + f * p.n_ghost_entries;
            scalar_t                            sum   = 0.0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[p.ghost_reduce_idx[j]];
            out_fields[f][dest] += sum;
        }
    }
}
