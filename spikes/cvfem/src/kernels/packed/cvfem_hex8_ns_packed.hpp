#ifndef CVFEM_HEX8_NS_PACKED_HPP
#define CVFEM_HEX8_NS_PACKED_HPP

#include "smesh_packed_mesh.hpp"
#include "kernels/cvfem_range.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// PackedData and the pack helpers live in the shared header; this file used to
// carry a divergent trimmed copy of them.
#include "hex8/cvfem_hex8_pack_common.hpp"

// After the kernel headers, which steady.cpp includes before this file.
#include "hex8/cvfem_hex8_pack_helpers.hpp"






static void cvfem_hex8_precompute_affine_geometry(MeshData &d) {
    precompute_affine_geometry(d);
}

static SFEM_INLINE void cvfem_hex8_load_adj(const MeshData &d, const ptrdiff_t e, scalar_t adj[9], scalar_t *det) {
    load_hex8_adj(d.adj_ptr, d.det_ptr, e, adj, det);
}

// A thin alias for gather_hex8_adj_soa under this family's name; it takes the affine geometry
// rather than the mesh for the same reason the function it forwards to does.
static SFEM_INLINE void cvfem_hex8_gather_adj_soa(const scalar_t *const *const SFEM_RESTRICT adj_ptr,
                                                  const scalar_t *const SFEM_RESTRICT        det_ptr,
                                                  const ptrdiff_t               begin,
                                                  const int                     nlanes,
                                                  scalar_t *const SFEM_RESTRICT cof0,
                                                  scalar_t *const SFEM_RESTRICT cof1,
                                                  scalar_t *const SFEM_RESTRICT cof2,
                                                  scalar_t *const SFEM_RESTRICT cof3,
                                                  scalar_t *const SFEM_RESTRICT cof4,
                                                  scalar_t *const SFEM_RESTRICT cof5,
                                                  scalar_t *const SFEM_RESTRICT cof6,
                                                  scalar_t *const SFEM_RESTRICT cof7,
                                                  scalar_t *const SFEM_RESTRICT cof8,
                                                  scalar_t *const SFEM_RESTRICT det) {
    gather_hex8_adj_soa(adj_ptr, det_ptr, begin, nlanes, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det);
}

static SFEM_INLINE void cvfem_hex8_gather_simd_from_pack(pack_idx_t **const SFEM_RESTRICT   elems,
                                                         const scalar_t *const SFEM_RESTRICT pack_u,
                                                         const MeshData                     &d,
                                                         const ptrdiff_t                     begin,
                                                         const int                           nlanes,
                                                         Hex8InputPack                      &in,
                                                         scalar_t *const SFEM_RESTRICT       cof0,
                                                         scalar_t *const SFEM_RESTRICT       cof1,
                                                         scalar_t *const SFEM_RESTRICT       cof2,
                                                         scalar_t *const SFEM_RESTRICT       cof3,
                                                         scalar_t *const SFEM_RESTRICT       cof4,
                                                         scalar_t *const SFEM_RESTRICT       cof5,
                                                         scalar_t *const SFEM_RESTRICT       cof6,
                                                         scalar_t *const SFEM_RESTRICT       cof7,
                                                         scalar_t *const SFEM_RESTRICT       cof8,
                                                         scalar_t *const SFEM_RESTRICT       det) {
    cvfem_hex8_gather_adj_soa(d.adj_ptr, d.det_ptr, begin, nlanes, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det);
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)elems[a][e] * N_FIELDS;
                in.ux[a][lane]                        = u[0];
                in.uy[a][lane]                        = u[1];
                in.uz[a][lane]                        = u[2];
                in.p[a][lane]                         = u[3];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                in.ux[a][lane] = in.uy[a][lane] = in.uz[a][lane] = in.p[a][lane] = scalar_t(0);
            }
        }
    }
}

static SFEM_INLINE void cvfem_hex8_gather_action_simd_from_pack(pack_idx_t **const SFEM_RESTRICT   elems,
                                                                const scalar_t *const SFEM_RESTRICT pack_u,
                                                                const scalar_t *const SFEM_RESTRICT pack_dir,
                                                                const MeshData                     &d,
                                                                const ptrdiff_t                     begin,
                                                                const int                           nlanes,
                                                                Hex8InputPack                      &u,
                                                                Hex8InputPack                      &du,
                                                                scalar_t *const SFEM_RESTRICT       cof0,
                                                                scalar_t *const SFEM_RESTRICT       cof1,
                                                                scalar_t *const SFEM_RESTRICT       cof2,
                                                                scalar_t *const SFEM_RESTRICT       cof3,
                                                                scalar_t *const SFEM_RESTRICT       cof4,
                                                                scalar_t *const SFEM_RESTRICT       cof5,
                                                                scalar_t *const SFEM_RESTRICT       cof6,
                                                                scalar_t *const SFEM_RESTRICT       cof7,
                                                                scalar_t *const SFEM_RESTRICT       cof8,
                                                                scalar_t *const SFEM_RESTRICT       det) {
    cvfem_hex8_gather_simd_from_pack(
            elems, pack_u, d, begin, nlanes, u, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det);
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const scalar_t *const SFEM_RESTRICT v = pack_dir + (ptrdiff_t)elems[a][e] * N_FIELDS;
                du.ux[a][lane]                        = v[0];
                du.uy[a][lane]                        = v[1];
                du.uz[a][lane]                        = v[2];
                du.p[a][lane]                         = v[3];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                du.ux[a][lane] = du.uy[a][lane] = du.uz[a][lane] = du.p[a][lane] = scalar_t(0);
            }
        }
    }
}

// Takes the arrays, not the staging objects, for the reason its bench twin does: it is called
// from inside the pack sweeps, and naming PackedData or MeshData here is what makes
// src/kernels/ depend on them.
static SFEM_INLINE void cvfem_hex8_fill_pack_fields(const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
                                                    const scalar_t *const SFEM_RESTRICT  ux,
                                                    const scalar_t *const SFEM_RESTRICT  uy,
                                                    const scalar_t *const SFEM_RESTRICT  uz,
                                                    const scalar_t *const SFEM_RESTRICT  pr,
                                                    const ptrdiff_t   pack,
                                                    const ptrdiff_t   n_contiguous,
                                                    const ptrdiff_t   n_ghost,
                                                    const idx_t *const SFEM_RESTRICT ghosts,
                                                    scalar_t *const SFEM_RESTRICT pack_u) {
    const ptrdiff_t owned = owned_nodes_ptr[pack];
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
        const idx_t            g   = ghosts[k];
        dst[0]                            = ux[g];
        dst[1]                            = uy[g];
        dst[2]                            = uz[g];
        dst[3]                            = pr[g];
    }
}

// The ghost reduction, driven by a range. It is the second and independent parallel loop of
// every packed sweep -- the rows a pack does not own -- so DESIGN.md's threading rule applies to
// it as much as to the element pass, and it needs its own entry point rather than riding on the
// sweep's.
//
// IT TAKES ONLY WHAT IT READS, which is the other half of DESIGN.md's rule for this directory:
// "the signatures of the functions are lean-and-mean only arguments that are acually used are
// passed". PackedData is a staging object -- it owns std::vectors and a shared_ptr<smesh::Mesh> --
// so naming it in a kernel signature is what keeps src/kernels/ dependent on a library. These
// four arrays and one count are the whole of what the reduction reads; the launcher resolves them.
static SFEM_INLINE void cvfem_hex8_ghost_reduce_soa_range(const cvfem_range rows,
        const idx_t *const SFEM_RESTRICT     ghost_reduce_dest,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const ptrdiff_t                      n_ghost_entries,
        const scalar_t *const SFEM_RESTRICT  ghost_buf,
        scalar_t *const fields[N_FIELDS]) {
    for (ptrdiff_t row = rows.begin; row < rows.end; ++row) {
        const idx_t dest  = ghost_reduce_dest[row];
        const ptrdiff_t    begin = ghost_reduce_ptr[row];
        const ptrdiff_t    end   = ghost_reduce_ptr[row + 1];
        for (int f = 0; f < N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = ghost_buf + f * n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[ghost_reduce_idx[j]];
            fields[f][dest] += sum;
        }
    }
}


static SFEM_INLINE void cvfem_hex8_ghost_reduce_soa(PackedData &p, scalar_t *const fields[N_FIELDS]) {
#pragma omp parallel
    cvfem_hex8_ghost_reduce_soa_range(cvfem_range_split(0, p.n_ghost_reduce_rows, 1,
                                   cvfem_thread_index(), cvfem_n_threads()),
            p.ghost_reduce_dest, p.ghost_reduce_ptr, p.ghost_reduce_idx,
            p.n_ghost_entries, p.ghost_buf.data(), fields);
}

// The same reduction at an arbitrary width, for destinations that carry more than the four
// fields per node -- the 4x4 block diagonal is sixteen. Separate rather than a template
// parameter on the one above, because that one is on the matvec's hot path and is compiled
// for exactly one width; this one runs once per Newton step.
//
// `buf` is field-major, n_ghost_entries apart, which is the layout the packs write.
// The ghost reduction, driven by a range. It is the second and independent parallel loop of
// every packed sweep -- the rows a pack does not own -- so DESIGN.md's threading rule applies to
// it as much as to the element pass, and it needs its own entry point rather than riding on the
// sweep's.
//
// IT TAKES ONLY WHAT IT READS, which is the other half of DESIGN.md's rule for this directory:
// "the signatures of the functions are lean-and-mean only arguments that are acually used are
// passed". PackedData is a staging object -- it owns std::vectors and a shared_ptr<smesh::Mesh> --
// so naming it in a kernel signature is what keeps src/kernels/ dependent on a library. These
// four arrays and one count are the whole of what the reduction reads; the launcher resolves them.
static SFEM_INLINE void cvfem_hex8_ghost_reduce_wide_range(const cvfem_range rows,
        const idx_t *const SFEM_RESTRICT     ghost_reduce_dest,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const ptrdiff_t                      n_ghost_entries,
        const scalar_t *const SFEM_RESTRICT buf,
        const int                           width,
        scalar_t *const SFEM_RESTRICT       dst) {
    for (ptrdiff_t row = rows.begin; row < rows.end; ++row) {
        const idx_t dest  = ghost_reduce_dest[row];
        const ptrdiff_t    begin = ghost_reduce_ptr[row];
        const ptrdiff_t    end   = ghost_reduce_ptr[row + 1];
        scalar_t *const    out   = dst + (ptrdiff_t)dest * width;
        for (int f = 0; f < width; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = buf + (ptrdiff_t)f * n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[ghost_reduce_idx[j]];
            out[f] += sum;
        }
    }
}


static SFEM_INLINE void cvfem_hex8_ghost_reduce_wide(PackedData                         &p,
        const scalar_t *const SFEM_RESTRICT buf,
        const int                           width,
        scalar_t *const SFEM_RESTRICT       dst) {
#pragma omp parallel
    cvfem_hex8_ghost_reduce_wide_range(cvfem_range_split(0, p.n_ghost_reduce_rows, 1,
                                   cvfem_thread_index(), cvfem_n_threads()),
            p.ghost_reduce_dest, p.ghost_reduce_ptr, p.ghost_reduce_idx,
            p.n_ghost_entries, buf, width, dst);
}

// The ghost reduction, driven by a range. It is the second and independent parallel loop of
// every packed sweep -- the rows a pack does not own -- so DESIGN.md's threading rule applies to
// it as much as to the element pass, and it needs its own entry point rather than riding on the
// sweep's.
//
// IT TAKES ONLY WHAT IT READS, which is the other half of DESIGN.md's rule for this directory:
// "the signatures of the functions are lean-and-mean only arguments that are acually used are
// passed". PackedData is a staging object -- it owns std::vectors and a shared_ptr<smesh::Mesh> --
// so naming it in a kernel signature is what keeps src/kernels/ dependent on a library. These
// four arrays and one count are the whole of what the reduction reads; the launcher resolves them.
static SFEM_INLINE void cvfem_hex8_ghost_reduce_interleaved_range(const cvfem_range rows,
        const idx_t *const SFEM_RESTRICT     ghost_reduce_dest,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const ptrdiff_t                      n_ghost_entries,
        const scalar_t *const SFEM_RESTRICT  ghost_buf,
        scalar_t *const SFEM_RESTRICT jv) {
    for (ptrdiff_t row = rows.begin; row < rows.end; ++row) {
        const idx_t dest  = ghost_reduce_dest[row];
        const ptrdiff_t    begin = ghost_reduce_ptr[row];
        const ptrdiff_t    end   = ghost_reduce_ptr[row + 1];
        scalar_t *const    out   = jv + (ptrdiff_t)dest * N_FIELDS;
        for (int f = 0; f < N_FIELDS; ++f) {
            const scalar_t *const SFEM_RESTRICT ghost = ghost_buf + f * n_ghost_entries;
            scalar_t                            sum   = 0;
            for (ptrdiff_t j = begin; j < end; ++j) sum += ghost[ghost_reduce_idx[j]];
            out[f] += sum;
        }
    }
}


static SFEM_INLINE void cvfem_hex8_ghost_reduce_interleaved(PackedData &p, scalar_t *const SFEM_RESTRICT jv) {
#pragma omp parallel
    cvfem_hex8_ghost_reduce_interleaved_range(cvfem_range_split(0, p.n_ghost_reduce_rows, 1,
                                   cvfem_thread_index(), cvfem_n_threads()),
            p.ghost_reduce_dest, p.ghost_reduce_ptr, p.ghost_reduce_idx,
            p.n_ghost_entries, p.ghost_buf.data(), jv);
}

// The pack sweep, driven by a range; the `#pragma omp parallel` is in the launcher below. See
// kernels/cvfem_range.hpp. A pack writes only the nodes it owns, so the parts need no
// synchronisation and the shared ghost rows are reduced afterwards.
static SFEM_NOINLINE void cvfem_hex8_apply_residual_packed_range(
        const cvfem_range packs,
        MeshData &d,
        PackedData &p,
        const scalar_t rho,
        const scalar_t mu,
        const size_t scratch_n,
        const size_t rc_n,
        const int with_rc) {
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_rc  = thread_scratch<scalar_t>(3, rc_n);
        const ptrdiff_t               nmax     = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x   = pack_rc;
        scalar_t *const SFEM_RESTRICT pack_y   = pack_rc + nmax;
        scalar_t *const SFEM_RESTRICT pack_z   = pack_rc + 2 * nmax;
        scalar_t *const SFEM_RESTRICT pack_pgx = pack_rc + 3 * nmax;
        scalar_t *const SFEM_RESTRICT pack_pgy = pack_rc + 4 * nmax;
        scalar_t *const SFEM_RESTRICT pack_pgz = pack_rc + 5 * nmax;


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t                         n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const ptrdiff_t                         ghost_off    = p.ghost_ptr[pack];

            std::memset(pack_out, 0, (size_t)n_pack_nodes * (size_t)N_FIELDS * sizeof(scalar_t));
            cvfem_hex8_fill_pack_fields(p.owned_nodes_ptr, d.ux.data(), d.uy.data(), d.uz.data(), d.p.data(), pack, n_contiguous, n_ghost, ghosts, pack_u);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(p.owned_nodes_ptr, d.points, d.pgx.data(), d.pgy.data(), d.pgz.data(), !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0), pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z,
                                               pack_pgx, pack_pgy, pack_pgz);

            alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE], cof2[CVFEM_HEX8_VEC_SIZE];
            alignas(ALIGN_BYTES) scalar_t cof3[CVFEM_HEX8_VEC_SIZE], cof4[CVFEM_HEX8_VEC_SIZE], cof5[CVFEM_HEX8_VEC_SIZE];
            alignas(ALIGN_BYTES) scalar_t cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE], cof8[CVFEM_HEX8_VEC_SIZE];
            alignas(ALIGN_BYTES) scalar_t det[CVFEM_HEX8_VEC_SIZE];
            Hex8InputPack     in;
            Hex8ResidualPack  outp;
            Hex8RhieChowPack  rcp;
            for (ptrdiff_t begin = e_start; begin < e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, e_end - begin));
                cvfem_hex8_gather_simd_from_pack(p.elems,
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
                    cvfem_hex8_gather_rc_from_pack(p.elems, pack_pgx, pack_pgy, pack_pgz, begin,
                                                   nlanes, rcp);
                }
                cvfem_hex8_ns_upwind_residual_sumfact_simd(rho,
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
                                                           in,
                                                           outp,
                                                           with_rc ? &rcp : nullptr,
                                                           d.rhie_chow_scale,
                                                           d.upwind_eps);
                cvfem_hex8_scatter_simd_to_pack(p.elems, pack_out, begin, nlanes, outp);
            }

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                const scalar_t *const SFEM_RESTRICT out = pack_out + k * N_FIELDS;
                const ptrdiff_t                     g   = owned + k;
                d.rx[g]                                 = out[0];
                d.ry[g]                                 = out[1];
                d.rz[g]                                 = out[2];
                d.rc[g]                                 = out[3];
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


static SFEM_NOINLINE void cvfem_hex8_apply_residual_packed(MeshData &d, PackedData &p, const scalar_t rho, const scalar_t mu) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_residual_packed");
    const size_t scratch_n = packed_scratch_n(p.max_actual_nodes_per_pack);
    const size_t rc_n      = packed_rc_n(p.max_actual_nodes_per_pack);
    const int    with_rc   = d.rhie_chow_scale != scalar_t(0);


#pragma omp parallel
    cvfem_hex8_apply_residual_packed_range(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d, p, rho, mu, scratch_n, rc_n, with_rc);


    scalar_t *const fields[N_FIELDS] = {d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data()};
    cvfem_hex8_ghost_reduce_soa(p, fields);
}

// The pack sweep, driven by a range; the `#pragma omp parallel` is in the launcher below. See
// kernels/cvfem_range.hpp. A pack writes only the nodes it owns, so the parts need no
// synchronisation and the shared ghost rows are reduced afterwards.
static SFEM_NOINLINE void cvfem_hex8_apply_jacobian_action_packed_range(
        const cvfem_range packs,
        MeshData &d,
        PackedData &p,
        const scalar_t rho,
        const scalar_t mu,
        const scalar_t *const SFEM_RESTRICT dir,
        scalar_t *const SFEM_RESTRICT jv,
        const size_t scratch_n,
        const size_t rc_n,
        const size_t qg_n,
        const int with_rc,
        const bool with_qg,
        const bool with_ho) {
        scalar_t *const SFEM_RESTRICT pack_u   = thread_scratch<scalar_t>(0, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_dir = thread_scratch<scalar_t>(1, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(2, scratch_n);
        scalar_t *const SFEM_RESTRICT pack_rc  = thread_scratch<scalar_t>(3, rc_n);
        const ptrdiff_t               nmax     = p.max_actual_nodes_per_pack > 0 ? p.max_actual_nodes_per_pack : 1;
        scalar_t *const SFEM_RESTRICT pack_x   = pack_rc;
        scalar_t *const SFEM_RESTRICT pack_y   = pack_rc + nmax;
        scalar_t *const SFEM_RESTRICT pack_z   = pack_rc + 2 * nmax;
        scalar_t *const SFEM_RESTRICT pack_pgx = pack_rc + 3 * nmax;
        scalar_t *const SFEM_RESTRICT pack_pgy = pack_rc + 4 * nmax;
        scalar_t *const SFEM_RESTRICT pack_pgz = pack_rc + 5 * nmax;
        scalar_t *const SFEM_RESTRICT pack_qg  = with_qg ? thread_scratch<scalar_t>(4, qg_n) : nullptr;
        scalar_t *const SFEM_RESTRICT pack_qgx = pack_qg;
        scalar_t *const SFEM_RESTRICT pack_qgy = with_qg ? pack_qg + nmax : nullptr;
        scalar_t *const SFEM_RESTRICT pack_qgz = with_qg ? pack_qg + 2 * nmax : nullptr;


    for (ptrdiff_t pack = packs.begin; pack < packs.end; ++pack) {
            const ptrdiff_t                         e_start      = pack * p.n_elements_per_pack;
            const ptrdiff_t                         e_end        = MIN(d.nelements, (pack + 1) * p.n_elements_per_pack);
            const ptrdiff_t                         owned        = p.owned_nodes_ptr[pack];
            const ptrdiff_t                         n_contiguous = p.owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t                         n_ghost      = p.ghost_ptr[pack + 1] - p.ghost_ptr[pack];
            const ptrdiff_t                         n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts       = &p.ghost_idx[p.ghost_ptr[pack]];
            const ptrdiff_t                         ghost_off    = p.ghost_ptr[pack];

            std::memset(pack_out, 0, (size_t)n_pack_nodes * (size_t)N_FIELDS * sizeof(scalar_t));
            cvfem_hex8_fill_pack_fields(p.owned_nodes_ptr, d.ux.data(), d.uy.data(), d.uz.data(), d.p.data(), pack, n_contiguous, n_ghost, ghosts, pack_u);
            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                scalar_t *const SFEM_RESTRICT dstd = pack_dir + k * N_FIELDS;
                const ptrdiff_t               g    = owned + k;
                dstd[0]                            = dir[(ptrdiff_t)g * N_FIELDS + 0];
                dstd[1]                            = dir[(ptrdiff_t)g * N_FIELDS + 1];
                dstd[2]                            = dir[(ptrdiff_t)g * N_FIELDS + 2];
                dstd[3]                            = dir[(ptrdiff_t)g * N_FIELDS + 3];
            }
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                scalar_t *const SFEM_RESTRICT dstd = pack_dir + (n_contiguous + k) * N_FIELDS;
                const idx_t            g    = ghosts[k];
                dstd[0]                            = dir[(ptrdiff_t)g * N_FIELDS + 0];
                dstd[1]                            = dir[(ptrdiff_t)g * N_FIELDS + 1];
                dstd[2]                            = dir[(ptrdiff_t)g * N_FIELDS + 2];
                dstd[3]                            = dir[(ptrdiff_t)g * N_FIELDS + 3];
            }
            if (with_qg)
                cvfem_hex8_fill_pack_qgrad(p.owned_nodes_ptr, d.qgx.data(), d.qgy.data(), d.qgz.data(), pack, n_contiguous, n_ghost, ghosts, pack_qgx, pack_qgy, pack_qgz);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(p.owned_nodes_ptr, d.points, d.pgx.data(), d.pgy.data(), d.pgz.data(), !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0), pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z,
                                               pack_pgx, pack_pgy, pack_pgz);

            Hex8InputPack    u_pack;
            Hex8InputPack    du_pack;
            Hex8UGradPack    hop, hovp;
            hop.limiter  = d.conv_limiter;
            hop.venkat_c = d.conv_venkat_c;
            Hex8ResidualPack outp;
            Hex8RhieChowPack rcp;
            alignas(ALIGN_BYTES) scalar_t cof0[CVFEM_HEX8_VEC_SIZE], cof1[CVFEM_HEX8_VEC_SIZE], cof2[CVFEM_HEX8_VEC_SIZE];
            alignas(ALIGN_BYTES) scalar_t cof3[CVFEM_HEX8_VEC_SIZE], cof4[CVFEM_HEX8_VEC_SIZE], cof5[CVFEM_HEX8_VEC_SIZE];
            alignas(ALIGN_BYTES) scalar_t cof6[CVFEM_HEX8_VEC_SIZE], cof7[CVFEM_HEX8_VEC_SIZE], cof8[CVFEM_HEX8_VEC_SIZE];
            alignas(ALIGN_BYTES) scalar_t det[CVFEM_HEX8_VEC_SIZE];
            for (ptrdiff_t begin = e_start; begin < e_end; begin += CVFEM_HEX8_VEC_SIZE) {
                const int nlanes = int(MIN((ptrdiff_t)CVFEM_HEX8_VEC_SIZE, e_end - begin));
                cvfem_hex8_gather_action_simd_from_pack(p.elems,
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
                    cvfem_hex8_gather_rc_from_pack(p.elems, pack_pgx, pack_pgy, pack_pgz, begin,
                                                   nlanes, rcp);
                    cvfem_hex8_gather_rc_coeff(d.rc_coeff.data(), d.rc_w.data(), cvfem_hex8_rc_config_for(d), begin, nlanes, rcp);
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
                                    hop.g[a][c][lane] = scalar_t(0); hovp.g[a][c][lane] = scalar_t(0);
                                }
                                continue;
                            }
                            const idx_t gn = d.elems[a][e];
                            hop.x[a][lane] = scalar_t(d.points[0][gn]);
                            hop.y[a][lane] = scalar_t(d.points[1][gn]);
                            hop.z[a][lane] = scalar_t(d.points[2][gn]);
                            for (int c = 0; c < 9; ++c) {
                                hop.g[a][c][lane]  = d.ugrad[(ptrdiff_t)gn * 9 + c];
                                hovp.g[a][c][lane] = d.vgrad[(ptrdiff_t)gn * 9 + c];
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
                                                          d.upwind_eps,
                                                          with_ho ? &hop : nullptr,
                                                          with_ho ? &hovp : nullptr);
                cvfem_hex8_scatter_simd_to_pack(p.elems, pack_out, begin, nlanes, outp);
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


static SFEM_NOINLINE void cvfem_hex8_apply_jacobian_action_packed(MeshData                    &d,
                                                                  PackedData                  &p,
                                                                  const scalar_t               rho,
                                                                  const scalar_t               mu,
                                                                  const scalar_t *const SFEM_RESTRICT dir,
                                                                  scalar_t *const SFEM_RESTRICT       jv) {
    {
        // Hoisted out of the face loops -- see Hex8RhieChowPack::coeff. Guarded by its own
        // cache key, so this is a handful of comparisons on every call but the first after
        // rho, mu or the scale move; the Reynolds continuation moves mu between stages,
        // which is why it lives here and not in initialize().
        SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::build_rc_coeff");
        cvfem_hex8_build_rc_coeff(d, rho, mu);
    }
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_jacobian_action_packed");
    const size_t scratch_n = packed_scratch_n(p.max_actual_nodes_per_pack);
    const size_t rc_n      = packed_rc_n(p.max_actual_nodes_per_pack);
    const size_t qg_n      = packed_qg_n(p.max_actual_nodes_per_pack);
    const int    with_rc   = d.rhie_chow_scale != scalar_t(0);
    // The Rhie-Chow term differentiates through the nodal pressure-gradient reconstruction.
    // apply_jacobian_action_accumulate reconstructs the direction's gradient into d.qg
    // before calling this, or clears it, so a non-empty d.qgx is exactly the signal that the
    // exact form is wanted. Without this the packed Jacobian is the frozen-pg one while the
    // residual is not, and Newton is capped at a linear rate.
    const bool   with_qg   = with_rc && !d.qgx.empty();
    // The exact higher-order action, signalled the same way: a non-empty d.vgrad means
    // apply_jacobian_action_accumulate reconstructed the direction's velocity gradient for it.
    const bool   with_ho   = d.conv_ho != 0 && !d.ugrad.empty() && !d.vgrad.empty();


#pragma omp parallel
    cvfem_hex8_apply_jacobian_action_packed_range(cvfem_range_split(0, p.n_packs, 1, cvfem_thread_index(), cvfem_n_threads()),
            d, p, rho, mu, dir, jv, scratch_n, rc_n, qg_n, with_rc, with_qg, with_ho);


    cvfem_hex8_ghost_reduce_interleaved(p, jv);
}

#endif

