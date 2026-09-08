#ifndef CVFEM_HEX8_LAYOUT_COLORED_HPP
#define CVFEM_HEX8_LAYOUT_COLORED_HPP

// Colored layout: the pack decomposition is colored so that two packs sharing a
// color never touch a common node. Within a color the element kernels can write
// straight into the global residual / matrix with plain non-atomic updates, so
// there is no pack-local buffer, no local->global fold and no ghost reduction --
// only a barrier between colors.
//
// Like the packed layout it stages a pack's nodal fields into a compact buffer,
// so the element kernels still run 16-wide SIMD; what it drops is the *reduction*
// on the way out.
//
// When to use which (measured on Apple M1 Max, 8 threads, n=48, interleaved A/B):
//
//                      vs packed   vs atomic
//   residual              0.61x       1.28x
//   jacobian action       0.60x       1.07x
//   jacobian assemble     1.46x       2.06x
//
// The payoff scales with how much reduction work coloring removes, while its cost
// is a fixed number of barriers -- one per color, 12-16 for an SFC pack
// decomposition. Assembly reduces a 372 MiB matrix, so coloring wins by a lot.
// The residual only reduces a 3.6 MiB vector, so the barriers cost more than the
// ghost reduce they replace: single-threaded the colored residual is in fact 1.18x
// *faster* than packed, and the whole advantage is given back to barrier waits by
// 8 threads. Coloring is still the better choice than atomics for every operation,
// and it is the useful shape where a ghost-reduction pass is awkward to express.
//
// Colors are balanced by construction (see cvfem_pack_coloring.hpp); an unbalanced
// coloring costs the residual another ~20% in barrier waits.

#include "cvfem_hex8_layout_common.hpp"
#include "cvfem_pack_coloring.hpp"

// ---------------------------------------------------------------------------
// Global-index gather / scatter
// ---------------------------------------------------------------------------

// Write a pack's accumulated output into the global arrays. Coloring makes this
// race-free without atomics and without a separate reduction pass: the owned
// nodes are a contiguous global window, the ghost nodes are scattered but few.
// The updates accumulate (+=) rather than store, because a node owned by this
// pack also receives contributions from packs that ghost it, and those may run in
// an earlier color.
static SFEM_INLINE void flush_pack_to_global_soa(const PackedData                       &p,
                                                 const ptrdiff_t                         pack,
                                                 const ptrdiff_t                         n_contiguous,
                                                 const ptrdiff_t                         n_ghost,
                                                 const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                                 const scalar_t *const SFEM_RESTRICT     pack_out,
                                                 scalar_t *const SFEM_RESTRICT           rx,
                                                 scalar_t *const SFEM_RESTRICT           ry,
                                                 scalar_t *const SFEM_RESTRICT           rz,
                                                 scalar_t *const SFEM_RESTRICT           rc) {
    const ptrdiff_t owned = p.owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        const scalar_t *const SFEM_RESTRICT src = pack_out + k * N_FIELDS;
        const ptrdiff_t                     g   = owned + k;
        rx[g] += src[0];
        ry[g] += src[1];
        rz[g] += src[2];
        rc[g] += src[3];
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        const scalar_t *const SFEM_RESTRICT src = pack_out + (n_contiguous + k) * N_FIELDS;
        const smesh::idx_t                  g   = ghosts[k];
        rx[g] += src[0];
        ry[g] += src[1];
        rz[g] += src[2];
        rc[g] += src[3];
    }
}

static SFEM_INLINE void flush_pack_to_global_interleaved(const PackedData                       &p,
                                                         const ptrdiff_t                         pack,
                                                         const ptrdiff_t                         n_contiguous,
                                                         const ptrdiff_t                         n_ghost,
                                                         const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                                         const scalar_t *const SFEM_RESTRICT     pack_out,
                                                         scalar_t *const SFEM_RESTRICT           jv) {
    const ptrdiff_t owned = p.owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        const scalar_t *const SFEM_RESTRICT src = pack_out + k * N_FIELDS;
        scalar_t *const SFEM_RESTRICT       dst = jv + (owned + k) * N_FIELDS;
        for (int f = 0; f < N_FIELDS; ++f) dst[f] += src[f];
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        const scalar_t *const SFEM_RESTRICT src = pack_out + (n_contiguous + k) * N_FIELDS;
        scalar_t *const SFEM_RESTRICT       dst = jv + (ptrdiff_t)ghosts[k] * N_FIELDS;
        for (int f = 0; f < N_FIELDS; ++f) dst[f] += src[f];
    }
}

// ---------------------------------------------------------------------------
// Residual
// ---------------------------------------------------------------------------

// Colored residual. Like the packed residual it stages the pack's nodal fields
// into a compact pack-local buffer, so the element gather stays cheap; unlike it,
// there is no pack-local *output* buffer to zero, copy out and reduce -- the
// element kernels accumulate straight into d.rx/ry/rz/rc, which coloring makes
// race-free. The sumfact and isoparam kernels run 16-wide over elements; the
// scalar SymPy/current kernels fall back to one element at a time.
static SFEM_NOINLINE void apply_residual_colored(MeshData           &d,
                                                 PackedData         &p,
                                                 const PackColoring &c,
                                                 const scalar_t      rho,
                                                 const scalar_t      mu,
                                                 const KernelKind    kernel_kind,
                                                 const GeomKind      geom_kind) {
    reset_residual(d);

    scalar_t *const SFEM_RESTRICT rx        = d.rx.data();
    scalar_t *const SFEM_RESTRICT ry        = d.ry.data();
    scalar_t *const SFEM_RESTRICT rz        = d.rz.data();
    scalar_t *const SFEM_RESTRICT rc        = d.rc.data();
    const size_t                  scratch_n = packed_scratch_n(p);

#pragma omp parallel
    {
        PhaseAcc                          acc;
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_xyz =
                geom_kind == GeomKind::Isoparam ? thread_scratch<scalar_t>(3, packed_xyz_n(p)) : nullptr;
        const ptrdiff_t               xyz_n  = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;

        for (int color = 0; color < c.n_colors; ++color) {
            const ptrdiff_t cbegin = c.color_ptr[(size_t)color];
            const ptrdiff_t cend   = c.color_ptr[(size_t)color + 1];
#pragma omp for schedule(dynamic, 1)
            for (ptrdiff_t i = cbegin; i < cend; ++i) {
                const ptrdiff_t                         pack         = c.pack_order[(size_t)i];
                const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
                const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
                const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
                const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
                const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
                const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];

                double _t = phase_now();
                std::memset(pack_out, 0, (size_t)(n_contiguous + n_ghost) * (size_t)N_FIELDS * sizeof(scalar_t));
                fill_pack_fields(p, d, pack, n_contiguous, n_ghost, ghosts, pack_u);
                if (geom_kind == GeomKind::Isoparam)
                    fill_pack_xyz(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);
                if (g_breakdown) { const double _n = wall_time(); acc.t[PH_GATHER] += _n - _t; _t = _n; }

                if (geom_kind == GeomKind::Isoparam) {
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
                    alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE],
                            cof2[CVFEM_HEX8_VEC_SIZE];
                    alignas(ALIGN_BYTES) scalar_t cof3[CVFEM_HEX8_VEC_SIZE], cof4[CVFEM_HEX8_VEC_SIZE],
                            cof5[CVFEM_HEX8_VEC_SIZE];
                    alignas(ALIGN_BYTES) scalar_t cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE],
                            cof8[CVFEM_HEX8_VEC_SIZE];
                    alignas(ALIGN_BYTES) scalar_t det[CVFEM_HEX8_VEC_SIZE];
                    Hex8InputPack                 in;
                    Hex8ResidualPack              outp;
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
                            ux_e[a]                               = u[0];
                            uy_e[a]                               = u[1];
                            uz_e[a]                               = u[2];
                            p_e[a]                                = u[3];
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
                if (g_breakdown) { const double _n = wall_time(); acc.t[PH_KERNEL] += _n - _t; _t = _n; }

                flush_pack_to_global_soa(p, pack, n_contiguous, n_ghost, ghosts, pack_out, rx, ry, rz, rc);
                if (g_breakdown) acc.t[PH_LOCAL_TO_GLOBAL] += wall_time() - _t;
            }
        }
        acc.flush();
    }
}

// ---------------------------------------------------------------------------
// Jacobian action
// ---------------------------------------------------------------------------

// The same colored sweep applied to the matrix-free Jacobian action, so --layout
// colored covers all three operations instead of silently falling back.
static SFEM_NOINLINE void apply_jacobian_action_colored(MeshData                           &d,
                                                        PackedData                         &p,
                                                        const PackColoring                 &c,
                                                        const scalar_t                      rho,
                                                        const scalar_t                      mu,
                                                        const scalar_t *const SFEM_RESTRICT dir,
                                                        scalar_t *const SFEM_RESTRICT       jv,
                                                        const GeomKind                      geom_kind) {
    cvfem_zero_scalars(jv, d.nnodes * N_FIELDS);

    const size_t scratch_n = packed_scratch_n(p);

#pragma omp parallel
    {
        PhaseAcc                          acc;
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_dir = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(2, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_xyz =
                geom_kind == GeomKind::Isoparam ? thread_scratch<scalar_t>(3, packed_xyz_n(p)) : nullptr;
        const ptrdiff_t               xyz_n  = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x = pack_xyz;
        scalar_t *const SFEM_RESTRICT pack_y = pack_xyz ? pack_xyz + xyz_n : nullptr;
        scalar_t *const SFEM_RESTRICT pack_z = pack_xyz ? pack_xyz + 2 * xyz_n : nullptr;

        for (int color = 0; color < c.n_colors; ++color) {
            const ptrdiff_t cbegin = c.color_ptr[(size_t)color];
            const ptrdiff_t cend   = c.color_ptr[(size_t)color + 1];
#pragma omp for schedule(dynamic, 1)
            for (ptrdiff_t i = cbegin; i < cend; ++i) {
                const ptrdiff_t                         pack         = c.pack_order[(size_t)i];
                const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
                const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
                const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
                const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
                const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
                const smesh::idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];

                double _t = phase_now();
                std::memset(pack_out, 0, (size_t)(n_contiguous + n_ghost) * (size_t)N_FIELDS * sizeof(scalar_t));
                fill_pack_fields(p, d, pack, n_contiguous, n_ghost, ghosts, pack_u);
                fill_pack_interleaved(p, pack, n_contiguous, n_ghost, ghosts, dir, pack_dir);
                if (geom_kind == GeomKind::Isoparam)
                    fill_pack_xyz(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z);
                if (g_breakdown) { const double _n = wall_time(); acc.t[PH_GATHER] += _n - _t; _t = _n; }

                Hex8InputPack    u_pack, du_pack;
                Hex8ResidualPack outp;
                Hex8CoordPack    xyz;
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
                if (g_breakdown) { const double _n = wall_time(); acc.t[PH_KERNEL] += _n - _t; _t = _n; }

                flush_pack_to_global_interleaved(p, pack, n_contiguous, n_ghost, ghosts, pack_out, jv);
                if (g_breakdown) acc.t[PH_LOCAL_TO_GLOBAL] += wall_time() - _t;
            }
        }
        acc.flush();
    }
}

// ---------------------------------------------------------------------------
// Jacobian assembly
// ---------------------------------------------------------------------------

// Colored assembly: elements are visited pack by pack, one color at a time, and
// the element kernel accumulates straight into the global BSR values. No local
// pack matrix, no local->global copy, no ghost reduction, no atomics.
static SFEM_NOINLINE void assemble_jacobian_colored(MeshData        &d,
                                                    PackedData      &p,
                                                    const PackColoring &c,
                                                    BSR4            &b,
                                                    const scalar_t   rho,
                                                    const scalar_t   mu,
                                                    const KernelKind kernel_kind,
                                                    const GeomKind   geom_kind) {
    zero_bsr4(b);

    scalar_t *const SFEM_RESTRICT       values = b.values->data();
    const int *const SFEM_RESTRICT      gslots = reinterpret_cast<const int *>(b.element_slots.data());

#pragma omp parallel
    {
        PhaseAcc acc;
        for (int color = 0; color < c.n_colors; ++color) {
            const ptrdiff_t cbegin = c.color_ptr[(size_t)color];
            const ptrdiff_t cend   = c.color_ptr[(size_t)color + 1];
#pragma omp for schedule(dynamic, 1)
            for (ptrdiff_t i = cbegin; i < cend; ++i) {
                const ptrdiff_t pack    = c.pack_order[(size_t)i];
                const ptrdiff_t e_start = pack * p.n_elements_per_pack;
                const ptrdiff_t e_end   = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
                const double    _t      = phase_now();

                for (ptrdiff_t e = e_start; e < e_end; ++e) {
                    scalar_t ux_e[8], uy_e[8], uz_e[8], p_e[8];
                    gather_element_fields(d, e, ux_e, uy_e, uz_e, p_e);
                    const int *const SFEM_RESTRICT slots = gslots + (size_t)e * 64;

                    if (geom_kind == GeomKind::Isoparam) {
                        scalar_t x[8], y[8], z[8];
                        gather_element_coords(d, e, x, y, z);
                        cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<false>(
                                rho, mu, x, y, z, ux_e, uy_e, uz_e, slots, values);
                    } else {
                        scalar_t adj[9], det;
                        load_hex8_adj(d, e, adj, &det);
                        switch (kernel_kind) {
                            case KernelKind::Sympy:
                                cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots(
                                        rho, mu, adj, det, ux_e, uy_e, uz_e, slots, values);
                                break;
                            case KernelKind::SympyBlock:
                                cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_blockwise(
                                        rho, mu, adj, det, ux_e, uy_e, uz_e, slots, values);
                                break;
                            case KernelKind::SympyRow:
                                cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_rowwise(
                                        rho, mu, adj, det, ux_e, uy_e, uz_e, slots, values);
                                break;
                            case KernelKind::SympyFace:
                                cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_facewise(
                                        rho, mu, adj, det, ux_e, uy_e, uz_e, slots, values);
                                break;
                            case KernelKind::Sumfact:
                                if (g_dense_flush) {
                                    alignas(ALIGN_BYTES) scalar_t ke[64 * 16] = {};
                                    cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                                            rho, mu, adj, det, ux_e, uy_e, uz_e, g_identity_slots, ke);
                                    hex8_blocks_to_slots(slots, ke, values);
                                } else {
                                    cvfem_hex8_ns_upwind_jacobian_add_slots<false>(
                                            rho, mu, adj, det, ux_e, uy_e, uz_e, slots, values);
                                }
                                break;
                            default: {
                                scalar_t ke[CVFEM_HEX8_N_DOF * CVFEM_HEX8_N_DOF];
                                cvfem_hex8_ns_upwind_jacobian_fd(rho, mu, adj, det, ux_e, uy_e, uz_e, p_e, ke);
                                hex8_local_slots_to_bsr4(slots, ke, values);
                                break;
                            }
                        }
                    }
                }
                if (g_breakdown) acc.t[PH_KERNEL] += wall_time() - _t;
            }
        }
        acc.flush();
    }
}

#endif  // CVFEM_HEX8_LAYOUT_COLORED_HPP
