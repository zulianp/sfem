#ifndef CVFEM_HEX8_NS_PACKED_HPP
#define CVFEM_HEX8_NS_PACKED_HPP

#include "smesh_packed_mesh.hpp"

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
#include "cvfem_hex8_pack_common.hpp"






static void cvfem_hex8_precompute_affine_geometry(MeshData &d) {
    for (int c = 0; c < 9; ++c) d.jacobian_adjugate[c].resize((size_t)d.nelements);
    d.jacobian_determinant.resize((size_t)d.nelements);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < d.nelements; ++e) {
        scalar_t x[8], y[8], z[8], adj[9], det;
        const auto *const px = d.points[0];
        const auto *const py = d.points[1];
        const auto *const pz = d.points[2];
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const smesh::idx_t g = d.elems[a][e];
            x[a]                 = scalar_t(px[g]);
            y[a]                 = scalar_t(py[g]);
            z[a]                 = scalar_t(pz[g]);
        }
        cvfem_hex8_affine_adj(x, y, z, adj, &det);
        for (int c = 0; c < 9; ++c) d.jacobian_adjugate[c][(size_t)e] = adj[c];
        d.jacobian_determinant[(size_t)e] = det;
    }
}

static SFEM_INLINE void cvfem_hex8_load_adj(const MeshData &d, const ptrdiff_t e, scalar_t adj[9], scalar_t *det) {
    for (int c = 0; c < 9; ++c) adj[c] = d.jacobian_adjugate[c][(size_t)e];
    *det = d.jacobian_determinant[(size_t)e];
}

static SFEM_INLINE void cvfem_hex8_gather_adj_soa(const MeshData               &d,
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
    const size_t n = (size_t)nlanes * sizeof(scalar_t);
    std::memcpy(cof0, d.jacobian_adjugate[0].data() + begin, n);
    std::memcpy(cof1, d.jacobian_adjugate[1].data() + begin, n);
    std::memcpy(cof2, d.jacobian_adjugate[2].data() + begin, n);
    std::memcpy(cof3, d.jacobian_adjugate[3].data() + begin, n);
    std::memcpy(cof4, d.jacobian_adjugate[4].data() + begin, n);
    std::memcpy(cof5, d.jacobian_adjugate[5].data() + begin, n);
    std::memcpy(cof6, d.jacobian_adjugate[6].data() + begin, n);
    std::memcpy(cof7, d.jacobian_adjugate[7].data() + begin, n);
    std::memcpy(cof8, d.jacobian_adjugate[8].data() + begin, n);
    std::memcpy(det, d.jacobian_determinant.data() + begin, n);
    if (nlanes < CVFEM_HEX8_VEC_SIZE) {
        const size_t pad = (size_t)(CVFEM_HEX8_VEC_SIZE - nlanes) * sizeof(scalar_t);
        std::memset(cof0 + nlanes, 0, pad);
        std::memset(cof1 + nlanes, 0, pad);
        std::memset(cof2 + nlanes, 0, pad);
        std::memset(cof3 + nlanes, 0, pad);
        std::memset(cof4 + nlanes, 0, pad);
        std::memset(cof5 + nlanes, 0, pad);
        std::memset(cof6 + nlanes, 0, pad);
        std::memset(cof7 + nlanes, 0, pad);
        std::memset(cof8 + nlanes, 0, pad);
        for (int lane = nlanes; lane < CVFEM_HEX8_VEC_SIZE; ++lane) det[lane] = scalar_t(1);
    }
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
    cvfem_hex8_gather_adj_soa(d, begin, nlanes, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det);
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

static SFEM_INLINE void cvfem_hex8_gather_rc_from_pack(pack_idx_t **const SFEM_RESTRICT     elems,
                                                       const scalar_t *const SFEM_RESTRICT pack_x,
                                                       const scalar_t *const SFEM_RESTRICT pack_y,
                                                       const scalar_t *const SFEM_RESTRICT pack_z,
                                                       const scalar_t *const SFEM_RESTRICT pack_pgx,
                                                       const scalar_t *const SFEM_RESTRICT pack_pgy,
                                                       const scalar_t *const SFEM_RESTRICT pack_pgz,
                                                       const ptrdiff_t                     begin,
                                                       const int                           nlanes,
                                                       Hex8RhieChowPack                   &rc) {
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t loc = elems[a][e];
                rc.x[a][lane]        = pack_x[loc];
                rc.y[a][lane]        = pack_y[loc];
                rc.z[a][lane]        = pack_z[loc];
                rc.pgx[a][lane]      = pack_pgx[loc];
                rc.pgy[a][lane]      = pack_pgy[loc];
                rc.pgz[a][lane]      = pack_pgz[loc];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                rc.x[a][lane] = rc.y[a][lane] = rc.z[a][lane] = scalar_t(0);
                rc.pgx[a][lane] = rc.pgy[a][lane] = rc.pgz[a][lane] = scalar_t(0);
            }
        }
    }
}

static SFEM_INLINE void cvfem_hex8_scatter_simd_to_pack(pack_idx_t **const SFEM_RESTRICT elems,
                                                        scalar_t *const SFEM_RESTRICT    pack_out,
                                                        const ptrdiff_t                  begin,
                                                        const int                        nlanes,
                                                        const Hex8ResidualPack          &out) {
    for (int lane = 0; lane < nlanes; ++lane) {
        const ptrdiff_t e = begin + lane;
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            scalar_t *const SFEM_RESTRICT dst = pack_out + (ptrdiff_t)elems[a][e] * N_FIELDS;
            dst[0] += out.rx[a][lane];
            dst[1] += out.ry[a][lane];
            dst[2] += out.rz[a][lane];
            dst[3] += out.rc[a][lane];
        }
    }
}

static SFEM_INLINE void cvfem_hex8_fill_pack_xyz_pgrad(const PackedData                  &p,
                                                       const MeshData                    &d,
                                                       const ptrdiff_t                    pack,
                                                       const ptrdiff_t                    n_contiguous,
                                                       const ptrdiff_t                    n_ghost,
                                                       const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                                       scalar_t *const SFEM_RESTRICT      pack_x,
                                                       scalar_t *const SFEM_RESTRICT      pack_y,
                                                       scalar_t *const SFEM_RESTRICT      pack_z,
                                                       scalar_t *const SFEM_RESTRICT      pack_pgx,
                                                       scalar_t *const SFEM_RESTRICT      pack_pgy,
                                                       scalar_t *const SFEM_RESTRICT      pack_pgz) {
    const auto *const px    = d.points[0];
    const auto *const py    = d.points[1];
    const auto *const pz    = d.points[2];
    const ptrdiff_t   owned = p.owned_nodes_ptr[pack];
    const int         with_pg = !d.pgx.empty() && d.rhie_chow_scale != scalar_t(0);
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        const ptrdiff_t g = owned + k;
        pack_x[k]         = scalar_t(px[g]);
        pack_y[k]         = scalar_t(py[g]);
        pack_z[k]         = scalar_t(pz[g]);
        pack_pgx[k]       = with_pg ? d.pgx[(size_t)g] : scalar_t(0);
        pack_pgy[k]       = with_pg ? d.pgy[(size_t)g] : scalar_t(0);
        pack_pgz[k]       = with_pg ? d.pgz[(size_t)g] : scalar_t(0);
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        const smesh::idx_t g         = ghosts[k];
        pack_x[n_contiguous + k]     = scalar_t(px[g]);
        pack_y[n_contiguous + k]     = scalar_t(py[g]);
        pack_z[n_contiguous + k]     = scalar_t(pz[g]);
        pack_pgx[n_contiguous + k]   = with_pg ? d.pgx[(size_t)g] : scalar_t(0);
        pack_pgy[n_contiguous + k]   = with_pg ? d.pgy[(size_t)g] : scalar_t(0);
        pack_pgz[n_contiguous + k]   = with_pg ? d.pgz[(size_t)g] : scalar_t(0);
    }
}

static SFEM_INLINE void cvfem_hex8_fill_pack_fields(const PackedData &p,
                                                    const MeshData   &d,
                                                    const ptrdiff_t   pack,
                                                    const ptrdiff_t   n_contiguous,
                                                    const ptrdiff_t   n_ghost,
                                                    const smesh::idx_t *const SFEM_RESTRICT ghosts,
                                                    scalar_t *const SFEM_RESTRICT pack_u) {
    const ptrdiff_t owned = p.owned_nodes_ptr[pack];
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
}

static SFEM_INLINE void cvfem_hex8_ghost_reduce_soa(PackedData &p, scalar_t *const fields[N_FIELDS]) {
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

static SFEM_INLINE void cvfem_hex8_ghost_reduce_interleaved(PackedData &p, scalar_t *const SFEM_RESTRICT jv) {
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

static SFEM_NOINLINE void cvfem_hex8_apply_residual_packed(MeshData &d, PackedData &p, const scalar_t rho, const scalar_t mu) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_residual_packed");
    const size_t scratch_n = packed_scratch_n(p);
    const size_t rc_n      = packed_rc_n(p);
    const int    with_rc   = d.rhie_chow_scale != scalar_t(0);

#pragma omp parallel
    {
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
            cvfem_hex8_fill_pack_fields(p, d, pack, n_contiguous, n_ghost, ghosts, pack_u);
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z,
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
                if (with_rc)
                    cvfem_hex8_gather_rc_from_pack(p.elems, pack_x, pack_y, pack_z, pack_pgx, pack_pgy, pack_pgz, begin,
                                                   nlanes, rcp);
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
                                                           d.rhie_chow_scale);
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

    scalar_t *const fields[N_FIELDS] = {d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data()};
    cvfem_hex8_ghost_reduce_soa(p, fields);
}

static SFEM_NOINLINE void cvfem_hex8_apply_jacobian_action_packed(MeshData                    &d,
                                                                  PackedData                  &p,
                                                                  const scalar_t               rho,
                                                                  const scalar_t               mu,
                                                                  const scalar_t *const SFEM_RESTRICT dir,
                                                                  scalar_t *const SFEM_RESTRICT       jv) {
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady::apply_jacobian_action_packed");
    const size_t scratch_n = packed_scratch_n(p);
    const size_t rc_n      = packed_rc_n(p);
    const int    with_rc   = d.rhie_chow_scale != scalar_t(0);

#pragma omp parallel
    {
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
            cvfem_hex8_fill_pack_fields(p, d, pack, n_contiguous, n_ghost, ghosts, pack_u);
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
                const smesh::idx_t            g    = ghosts[k];
                dstd[0]                            = dir[(ptrdiff_t)g * N_FIELDS + 0];
                dstd[1]                            = dir[(ptrdiff_t)g * N_FIELDS + 1];
                dstd[2]                            = dir[(ptrdiff_t)g * N_FIELDS + 2];
                dstd[3]                            = dir[(ptrdiff_t)g * N_FIELDS + 3];
            }
            if (with_rc)
                cvfem_hex8_fill_pack_xyz_pgrad(p, d, pack, n_contiguous, n_ghost, ghosts, pack_x, pack_y, pack_z,
                                               pack_pgx, pack_pgy, pack_pgz);

            Hex8InputPack    u_pack;
            Hex8InputPack    du_pack;
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
                if (with_rc)
                    cvfem_hex8_gather_rc_from_pack(p.elems, pack_x, pack_y, pack_z, pack_pgx, pack_pgy, pack_pgz, begin,
                                                   nlanes, rcp);
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
                                                          d.rhie_chow_scale);
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

    cvfem_hex8_ghost_reduce_interleaved(p, jv);
}

#endif

