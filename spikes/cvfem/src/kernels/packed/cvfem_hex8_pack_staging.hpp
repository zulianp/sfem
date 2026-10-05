#ifndef CVFEM_HEX8_PACK_STAGING_HPP
#define CVFEM_HEX8_PACK_STAGING_HPP

// STAGING ONE PACK: the gathers, the fills and the scatter that every pack sweep runs around
// its element kernel.
//
// These ten functions used to live in frontend/staging/, and the pack sweeps called them anyway
// -- reached by INCLUDE ORDER rather than by an include, because a launcher pulls the staging
// header in before the kernel header. So src/kernels/ depended on ten definitions outside it
// and cvfem_kernels_self_contained could not see it: there was no include from kernels/ to
// object to. Four of them named smesh::idx_t, which that gate forbids in kernels/ code, from
// inside a call tree that kernels/ owned.
//
// They are here now, and they name no library: the including translation unit supplies idx_t,
// pack_idx_t, geom_t and scalar_t by contract, which is how every other kernel in this tree
// takes them. DESIGN.md: "header only self-contained code with templated types and localized
// macros (no library dependencies allowed)".
//
// They are templated on all four for the reason the correction gives -- "they should support
// different types for the computation, template scalar_t, geom_t, idx_t, etc..." -- and because
// a sweep cannot be templated while the staging it calls is not.

#include "kernels/microkernels/hex8/cvfem_hex8_ns_upwind_kernels.hpp"
#include "kernels/microkernels/hex8/isoparametric/cvfem_hex8_ns_upwind_isoparam.hpp"
#include "kernels/packed/cvfem_pack_scratch.hpp"
#include "kernels/cvfem_hex8_flags.hpp"


template <typename scalar_t, typename pack_idx_t>
static SFEM_INLINE void gather_hex8_coords_from_pack(pack_idx_t **const SFEM_RESTRICT     elems,
                                                     const scalar_t *const SFEM_RESTRICT pack_x,
                                                     const scalar_t *const SFEM_RESTRICT pack_y,
                                                     const scalar_t *const SFEM_RESTRICT pack_z,
                                                     const ptrdiff_t                     e,
                                                     scalar_t *const SFEM_RESTRICT       x,
                                                     scalar_t *const SFEM_RESTRICT       y,
                                                     scalar_t *const SFEM_RESTRICT       z) {
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        const pack_idx_t loc = elems[a][e];
        x[a]                 = pack_x[loc];
        y[a]                 = pack_y[loc];
        z[a]                 = pack_z[loc];
    }
}

// Copy a pack's nodal fields into an interleaved pack-local buffer. Indexing the
// element kernels through pack-local ids turns four scattered global reads per
// node into one contiguous read, which is why the packed and colored layouts both
// stage through this buffer rather than gathering from d.ux/uy/uz/p directly.
// Takes the arrays, not the staging objects: it is called from inside the pack sweeps, and a
// PackedData or MeshData parameter here is a staging dependency in src/kernels/, which DESIGN.md
// does not allow there.
template <typename scalar_t, typename idx_t>
static SFEM_INLINE void fill_pack_fields(const ptrdiff_t *const SFEM_RESTRICT    owned_nodes_ptr,
                                         const scalar_t *const SFEM_RESTRICT     ux,
                                         const scalar_t *const SFEM_RESTRICT     uy,
                                         const scalar_t *const SFEM_RESTRICT     uz,
                                         const scalar_t *const SFEM_RESTRICT     pr,
                                         const ptrdiff_t                         pack,
                                         const ptrdiff_t                         n_contiguous,
                                         const ptrdiff_t                         n_ghost,
                                         const idx_t *const SFEM_RESTRICT ghosts,
                                         scalar_t *const SFEM_RESTRICT           pack_u) {
    const ptrdiff_t                     owned = owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        scalar_t *const SFEM_RESTRICT dst = pack_u + k * CVFEM_HEX8_N_FIELDS;
        const ptrdiff_t               g   = owned + k;
        dst[0]                            = ux[g];
        dst[1]                            = uy[g];
        dst[2]                            = uz[g];
        dst[3]                            = pr[g];
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        scalar_t *const SFEM_RESTRICT dst = pack_u + (n_contiguous + k) * CVFEM_HEX8_N_FIELDS;
        const idx_t            g   = ghosts[k];
        dst[0]                            = ux[g];
        dst[1]                            = uy[g];
        dst[2]                            = uz[g];
        dst[3]                            = pr[g];
    }
}

// Takes the arrays, not the staging objects: it is called from inside the pack sweeps, and a
// PackedData or MeshData parameter here is a staging dependency in src/kernels/, which DESIGN.md
// does not allow there.
template <typename scalar_t, typename idx_t, typename geom_t>
static SFEM_INLINE void fill_pack_xyz(const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
                                      geom_t **const SFEM_RESTRICT       points,
                                      const ptrdiff_t                    pack,
                                      const ptrdiff_t                    n_contiguous,
                                      const ptrdiff_t                    n_ghost,
                                      const idx_t *const SFEM_RESTRICT ghosts,
                                      scalar_t *const SFEM_RESTRICT      pack_x,
                                      scalar_t *const SFEM_RESTRICT      pack_y,
                                      scalar_t *const SFEM_RESTRICT      pack_z) {
    const auto *const px    = points[0];
    const auto *const py    = points[1];
    const auto *const pz    = points[2];
    const ptrdiff_t   owned = owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        const ptrdiff_t g = owned + k;
        pack_x[k]         = scalar_t(px[g]);
        pack_y[k]         = scalar_t(py[g]);
        pack_z[k]         = scalar_t(pz[g]);
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        const idx_t g = ghosts[k];
        pack_x[n_contiguous + k] = scalar_t(px[g]);
        pack_y[n_contiguous + k] = scalar_t(py[g]);
        pack_z[n_contiguous + k] = scalar_t(pz[g]);
    }
}

// Takes the arrays and the flag, not the staging objects, for the reason the gathers above do:
// called from inside the pack sweeps, a PackT or MeshT parameter here is what keeps
// src/kernels/ dependent on the staging layer. with_pg is resolved by the caller, where the
// Rhie-Chow decision already lives.
template <typename scalar_t, typename idx_t, typename geom_t>
static SFEM_INLINE void cvfem_hex8_fill_pack_xyz_pgrad(const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
                                                       geom_t **const SFEM_RESTRICT       points,
                                                       const scalar_t *const SFEM_RESTRICT pgx,
                                                       const scalar_t *const SFEM_RESTRICT pgy,
                                                       const scalar_t *const SFEM_RESTRICT pgz,
                                                       const int                          with_pg,
                                                       const ptrdiff_t                    pack,
                                                       const ptrdiff_t                    n_contiguous,
                                                       const ptrdiff_t                    n_ghost,
                                                       const idx_t *const SFEM_RESTRICT ghosts,
                                                       scalar_t *const SFEM_RESTRICT      pack_x,
                                                       scalar_t *const SFEM_RESTRICT      pack_y,
                                                       scalar_t *const SFEM_RESTRICT      pack_z,
                                                       scalar_t *const SFEM_RESTRICT      pack_pgx,
                                                       scalar_t *const SFEM_RESTRICT      pack_pgy,
                                                       scalar_t *const SFEM_RESTRICT      pack_pgz) {
    const auto *const px    = points[0];
    const auto *const py    = points[1];
    const auto *const pz    = points[2];
    const ptrdiff_t   owned = owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        const ptrdiff_t g = owned + k;
        pack_x[k]         = scalar_t(px[g]);
        pack_y[k]         = scalar_t(py[g]);
        pack_z[k]         = scalar_t(pz[g]);
        pack_pgx[k]       = with_pg ? pgx[(size_t)g] : scalar_t(0);
        pack_pgy[k]       = with_pg ? pgy[(size_t)g] : scalar_t(0);
        pack_pgz[k]       = with_pg ? pgz[(size_t)g] : scalar_t(0);
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        const idx_t g         = ghosts[k];
        pack_x[n_contiguous + k]     = scalar_t(px[g]);
        pack_y[n_contiguous + k]     = scalar_t(py[g]);
        pack_z[n_contiguous + k]     = scalar_t(pz[g]);
        pack_pgx[n_contiguous + k]   = with_pg ? pgx[(size_t)g] : scalar_t(0);
        pack_pgy[n_contiguous + k]   = with_pg ? pgy[(size_t)g] : scalar_t(0);
        pack_pgz[n_contiguous + k]   = with_pg ? pgz[(size_t)g] : scalar_t(0);
    }
}

// The same staging for the DIRECTION's reconstructed gradient, which only the Jacobian
// action needs. Separate from the routine above rather than another pair of arguments on
// it: the residual and the benchmark call that one and have nothing to put here.
// Takes the arrays and the flag, not the staging objects, for the reason the gathers above do:
// called from inside the pack sweeps, a PackT or MeshT parameter here is what keeps
// src/kernels/ dependent on the staging layer.

template <typename scalar_t, typename idx_t>
static SFEM_INLINE void cvfem_hex8_fill_pack_qgrad(const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
                                                   const scalar_t *const SFEM_RESTRICT qgx,
                                                   const scalar_t *const SFEM_RESTRICT qgy,
                                                   const scalar_t *const SFEM_RESTRICT qgz,
                                                   const ptrdiff_t                    pack,
                                                   const ptrdiff_t                    n_contiguous,
                                                   const ptrdiff_t                    n_ghost,
                                                   const idx_t *const SFEM_RESTRICT ghosts,
                                                   scalar_t *const SFEM_RESTRICT      pack_qgx,
                                                   scalar_t *const SFEM_RESTRICT      pack_qgy,
                                                   scalar_t *const SFEM_RESTRICT      pack_qgz) {
    const ptrdiff_t owned = owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        const ptrdiff_t g = owned + k;
        pack_qgx[k]       = qgx[(size_t)g];
        pack_qgy[k]       = qgy[(size_t)g];
        pack_qgz[k]       = qgz[(size_t)g];
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        const idx_t g       = ghosts[k];
        pack_qgx[n_contiguous + k] = qgx[(size_t)g];
        pack_qgy[n_contiguous + k] = qgy[(size_t)g];
        pack_qgz[n_contiguous + k] = qgz[(size_t)g];
    }
}

// The SoA gather for the above, straight into the pack the face loops read.
//
// It takes the two tables rather than the mesh, because it is called from inside five pack
// sweeps and a MeshT parameter here is a staging dependency in src/kernels/, which DESIGN.md
// does not allow there. They are the only things it read.
template <typename scalar_t>
static SFEM_INLINE void cvfem_hex8_gather_rc_coeff(const scalar_t *const SFEM_RESTRICT src,
                                                   const scalar_t *const SFEM_RESTRICT srcw,
                                                   const Hex8RcConfigT<scalar_t> &cfg,
                                                   const ptrdiff_t   begin,
                                                   const int         nlanes,
                                                   Hex8RhieChowPackT<scalar_t> &rc) {
    // Lane-major out of an element-major table: one element's twelve coefficients are
    // consecutive, so this walks one stream instead of twelve. Measured neutral, not faster.
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const scalar_t *const SFEM_RESTRICT e = src + (ptrdiff_t)(begin + lane) * CVFEM_HEX8_N_SCS;
            const scalar_t *const SFEM_RESTRICT w = srcw + (ptrdiff_t)(begin + lane) * CVFEM_HEX8_N_SCS;
            for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
                rc.coeff[s][lane] = e[s];
                rc.wdu[s][lane]   = w[s];
            }
        } else {
            for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
                rc.coeff[s][lane] = scalar_t(0);
                rc.wdu[s][lane]   = scalar_t(0);
            }
        }
    }
    // What the coefficient table was built with, for its velocity sensitivity in the face loop.
    // Resolved by the caller: this function no longer sees a mesh.
    rc.scale               = cfg.scale;
    rc.tau                 = cfg.tau;
}

// Per-element scratch for the above. Declared inside the element loop; `rc` points into
// this object, so it must outlive the kernel call -- which it does, being a local.


template <typename scalar_t, typename pack_idx_t>
static SFEM_INLINE void gather_hex8_simd_from_pack(pack_idx_t **const SFEM_RESTRICT   elems,
                                                   const scalar_t *const SFEM_RESTRICT pack_u,
                                                   const scalar_t *const *const SFEM_RESTRICT adj_ptr,
                                                   const scalar_t *const SFEM_RESTRICT        det_ptr,
                                                   const ptrdiff_t                     begin,
                                                   const int                           nlanes,
                                                   Hex8InputPackT<scalar_t>                      &in,
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
    gather_hex8_adj_soa(adj_ptr, det_ptr, begin, nlanes, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det);
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const scalar_t *const SFEM_RESTRICT u = pack_u + (ptrdiff_t)elems[a][e] * CVFEM_HEX8_N_FIELDS;
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


template <typename scalar_t, typename pack_idx_t>
static SFEM_INLINE void gather_hex8_isoparam_simd_from_pack(pack_idx_t **const SFEM_RESTRICT     elems,
                                                            const scalar_t *const SFEM_RESTRICT pack_u,
                                                            const scalar_t *const SFEM_RESTRICT pack_x,
                                                            const scalar_t *const SFEM_RESTRICT pack_y,
                                                            const scalar_t *const SFEM_RESTRICT pack_z,
                                                            const ptrdiff_t                     begin,
                                                            const int                           nlanes,
                                                            Hex8InputPackT<scalar_t>                      &in,
                                                            Hex8CoordPackT<scalar_t>                      &xyz) {
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t                    loc = elems[a][e];
                const scalar_t *const SFEM_RESTRICT u   = pack_u + (ptrdiff_t)loc * CVFEM_HEX8_N_FIELDS;
                in.ux[a][lane]                          = u[0];
                in.uy[a][lane]                          = u[1];
                in.uz[a][lane]                          = u[2];
                in.p[a][lane]                           = u[3];
                xyz.x[a][lane]                          = pack_x[loc];
                xyz.y[a][lane]                          = pack_y[loc];
                xyz.z[a][lane]                          = pack_z[loc];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                in.ux[a][lane] = in.uy[a][lane] = in.uz[a][lane] = in.p[a][lane] = scalar_t(0);
                xyz.x[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][0];
                xyz.y[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][1];
                xyz.z[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][2];
            }
        }
    }
}

// ---------------------------------------------------------------- Rhie-Chow pack staging
//
// Moved here from cvfem_hex8_ns_packed.hpp so the benchmark can stage the term too. The
// gather needs nothing but raw arrays and was already family-independent; the filler is
// templated on the two container types the way the rest of this header is.
template <typename scalar_t, typename pack_idx_t>
static SFEM_INLINE void cvfem_hex8_gather_rc_from_pack(pack_idx_t **const SFEM_RESTRICT     elems,
                                                       const scalar_t *const SFEM_RESTRICT pack_pgx,
                                                       const scalar_t *const SFEM_RESTRICT pack_pgy,
                                                       const scalar_t *const SFEM_RESTRICT pack_pgz,
                                                       const ptrdiff_t                     begin,
                                                       const int                           nlanes,
                                                       Hex8RhieChowPackT<scalar_t>                   &rc) {
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t loc = elems[a][e];
                rc.pgx[a][lane]      = pack_pgx[loc];
                rc.pgy[a][lane]      = pack_pgy[loc];
                rc.pgz[a][lane]      = pack_pgz[loc];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                rc.pgx[a][lane] = rc.pgy[a][lane] = rc.pgz[a][lane] = scalar_t(0);
            }
        }
    }
}

// No mesh argument, so no template parameter to deduce -- a plain function.

// THE SAME CLASS, FOUND BY THE CALL CHECK rather than by inspection: seven more
// definitions the pack sweeps reached across the frontend/staging boundary.
// Same, for an already-interleaved global vector (a Krylov direction).
// Takes the arrays, not the staging objects: it is called from inside the pack sweeps, and a
// PackedData or MeshData parameter here is a staging dependency in src/kernels/, which DESIGN.md
// does not allow there.
template <typename scalar_t, typename idx_t>
static SFEM_INLINE void fill_pack_interleaved(const ptrdiff_t *const SFEM_RESTRICT    owned_nodes_ptr,
                                              const ptrdiff_t                         pack,
                                              const ptrdiff_t                         n_contiguous,
                                              const ptrdiff_t                         n_ghost,
                                              const idx_t *const SFEM_RESTRICT ghosts,
                                              const scalar_t *const SFEM_RESTRICT     src,
                                              scalar_t *const SFEM_RESTRICT           pack_v) {
    const ptrdiff_t owned = owned_nodes_ptr[pack];
    for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
        std::memcpy(pack_v + k * CVFEM_HEX8_N_FIELDS, src + (owned + k) * CVFEM_HEX8_N_FIELDS, CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));
    }
    for (ptrdiff_t k = 0; k < n_ghost; ++k) {
        std::memcpy(pack_v + (n_contiguous + k) * CVFEM_HEX8_N_FIELDS,
                    src + (ptrdiff_t)ghosts[k] * CVFEM_HEX8_N_FIELDS,
                    CVFEM_HEX8_N_FIELDS * sizeof(scalar_t));
    }
}



template <typename scalar_t, typename pack_idx_t>
static SFEM_INLINE void gather_hex8_action_simd_from_pack(pack_idx_t **const SFEM_RESTRICT   elems,
                                                          const scalar_t *const SFEM_RESTRICT pack_u,
                                                          const scalar_t *const SFEM_RESTRICT pack_dir,
                                                          const scalar_t *const *const SFEM_RESTRICT adj_ptr,
                                                          const scalar_t *const SFEM_RESTRICT        det_ptr,
                                                          const ptrdiff_t                     begin,
                                                          const int                           nlanes,
                                                          Hex8InputPackT<scalar_t>                      &u,
                                                          Hex8InputPackT<scalar_t>                      &du,
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
    gather_hex8_simd_from_pack(elems, pack_u, adj_ptr, det_ptr, begin, nlanes, u, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det);
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const scalar_t *const SFEM_RESTRICT dvec = pack_dir + (ptrdiff_t)elems[a][e] * CVFEM_HEX8_N_FIELDS;
                du.ux[a][lane]                           = dvec[0];
                du.uy[a][lane]                           = dvec[1];
                du.uz[a][lane]                           = dvec[2];
                du.p[a][lane]                            = dvec[3];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                du.ux[a][lane] = du.uy[a][lane] = du.uz[a][lane] = du.p[a][lane] = scalar_t(0);
            }
        }
    }
}


template <typename scalar_t, typename pack_idx_t>
static SFEM_INLINE void gather_hex8_isoparam_action_simd_from_pack(pack_idx_t **const SFEM_RESTRICT     elems,
                                                                   const scalar_t *const SFEM_RESTRICT pack_u,
                                                                   const scalar_t *const SFEM_RESTRICT pack_dir,
                                                                   const scalar_t *const SFEM_RESTRICT pack_x,
                                                                   const scalar_t *const SFEM_RESTRICT pack_y,
                                                                   const scalar_t *const SFEM_RESTRICT pack_z,
                                                                   const ptrdiff_t                     begin,
                                                                   const int                           nlanes,
                                                                   Hex8InputPackT<scalar_t>                      &u,
                                                                   Hex8InputPackT<scalar_t>                      &du,
                                                                   Hex8CoordPackT<scalar_t>                      &xyz) {
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t                    loc  = elems[a][e];
                const scalar_t *const SFEM_RESTRICT usrc = pack_u + (ptrdiff_t)loc * CVFEM_HEX8_N_FIELDS;
                const scalar_t *const SFEM_RESTRICT dsrc = pack_dir + (ptrdiff_t)loc * CVFEM_HEX8_N_FIELDS;
                u.ux[a][lane]                            = usrc[0];
                u.uy[a][lane]                            = usrc[1];
                u.uz[a][lane]                            = usrc[2];
                u.p[a][lane]                             = usrc[3];
                du.ux[a][lane]                           = dsrc[0];
                du.uy[a][lane]                           = dsrc[1];
                du.uz[a][lane]                           = dsrc[2];
                du.p[a][lane]                            = dsrc[3];
                xyz.x[a][lane]                           = pack_x[loc];
                xyz.y[a][lane]                           = pack_y[loc];
                xyz.z[a][lane]                           = pack_z[loc];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                u.ux[a][lane] = u.uy[a][lane] = u.uz[a][lane] = u.p[a][lane] = scalar_t(0);
                du.ux[a][lane] = du.uy[a][lane] = du.uz[a][lane] = du.p[a][lane] = scalar_t(0);
                xyz.x[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][0];
                xyz.y[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][1];
                xyz.z[a][lane]                                                   = CVFEM_HEX8_UNIT_CUBE[a][2];
            }
        }
    }
}

// The direction's gradient into the same pack, called straight after the routine above
// when the Jacobian action needs it. Padding lanes are zeroed here too: they multiply real
// geometry and would otherwise contribute whatever the last sweep left behind.
template <typename scalar_t, typename pack_idx_t>
static SFEM_INLINE void cvfem_hex8_gather_qg_from_pack(pack_idx_t **const SFEM_RESTRICT     elems,
                                                       const scalar_t *const SFEM_RESTRICT pack_qgx,
                                                       const scalar_t *const SFEM_RESTRICT pack_qgy,
                                                       const scalar_t *const SFEM_RESTRICT pack_qgz,
                                                       const ptrdiff_t                     begin,
                                                       const int                           nlanes,
                                                       Hex8RhieChowPackT<scalar_t>                   &rc) {
    for (int lane = 0; lane < CVFEM_HEX8_VEC_SIZE; ++lane) {
        if (lane < nlanes) {
            const ptrdiff_t e = begin + lane;
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const pack_idx_t loc = elems[a][e];
                rc.qgx[a][lane]      = pack_qgx[loc];
                rc.qgy[a][lane]      = pack_qgy[loc];
                rc.qgz[a][lane]      = pack_qgz[loc];
            }
        } else {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a)
                rc.qgx[a][lane] = rc.qgy[a][lane] = rc.qgz[a][lane] = scalar_t(0);
        }
    }
}


template <typename scalar_t, typename pack_idx_t>
static SFEM_INLINE void cvfem_hex8_scatter_simd_to_pack(pack_idx_t **const SFEM_RESTRICT elems,
                                                        scalar_t *const SFEM_RESTRICT    pack_out,
                                                        const ptrdiff_t                  begin,
                                                        const int                        nlanes,
                                                        const Hex8ResidualPackT<scalar_t>          &out) {
    scatter_hex8_simd_to_pack(elems, pack_out, begin, nlanes, out);
}

// Flush a dense block-major element matrix ke[(i*8+k)*16 + c] into the target
// matrix: 64 contiguous 16-double adds instead of ~768 scattered scalar updates.
template <typename scalar_t>
static SFEM_INLINE void hex8_blocks_to_slots(const int *const SFEM_RESTRICT      slots,
                                             const scalar_t *const SFEM_RESTRICT ke,
                                             scalar_t *const SFEM_RESTRICT       values) {
    for (int blk = 0; blk < 64; ++blk) {
        scalar_t *const SFEM_RESTRICT       dst = values + (ptrdiff_t)slots[blk] * 16;
        const scalar_t *const SFEM_RESTRICT src = ke + blk * 16;
#pragma omp simd
        for (int c = 0; c < 16; ++c) dst[c] += src[c];
    }
}








template <typename scalar_t>
static SFEM_INLINE void bsr4_add16(scalar_t *const SFEM_RESTRICT dst, const scalar_t *const SFEM_RESTRICT src) {
#pragma omp simd
    for (int i = 0; i < 16; ++i) dst[i] += src[i];
}


template <typename scalar_t, typename pack_idx_t>
static SFEM_INLINE void scatter_hex8_simd_to_pack(pack_idx_t **const SFEM_RESTRICT elems,
                                                  scalar_t *const SFEM_RESTRICT    pack_out,
                                                  const ptrdiff_t                  begin,
                                                  const int                        nlanes,
                                                  const Hex8ResidualPackT<scalar_t>          &out) {
    for (int lane = 0; lane < nlanes; ++lane) {
        const ptrdiff_t e = begin + lane;
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            scalar_t *const SFEM_RESTRICT dst = pack_out + (ptrdiff_t)elems[a][e] * CVFEM_HEX8_N_FIELDS;
            dst[0] += out.rx[a][lane];
            dst[1] += out.ry[a][lane];
            dst[2] += out.rz[a][lane];
            dst[3] += out.rc[a][lane];
        }
    }
}


// ONE PACK'S LANE LOOP, AFFINE. Called by both drains' sweeps: the contiguous packed residual
// and the pack-coloured one, which differ in nothing else. The adjugate and determinant come
// from the precomputed table, which is what makes this the affine loop.
template <typename scalar_t, typename idx_t, typename pack_idx_t>
static SFEM_INLINE void cvfem_hex8_residual_lanes_affine(
        const Hex8PackExtentT<idx_t>        &x,
        const Hex8PackCoordsT<scalar_t>     &pk,
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT  det_ptr,
        pack_idx_t **const SFEM_RESTRICT     pack_elems,
        const scalar_t *const SFEM_RESTRICT  pack_u,
        scalar_t *const SFEM_RESTRICT        pack_out,
        const scalar_t                       rho,
        const scalar_t                       mu,
        const scalar_t                       rhie_chow_scale,
        const int                            with_rc) {
    alignas(ALIGN_BYTES) scalar_t cof0[cvfem_hex8_vec_size<scalar_t>], cof1[cvfem_hex8_vec_size<scalar_t>], cof2[cvfem_hex8_vec_size<scalar_t>];
    alignas(ALIGN_BYTES) scalar_t cof3[cvfem_hex8_vec_size<scalar_t>], cof4[cvfem_hex8_vec_size<scalar_t>], cof5[cvfem_hex8_vec_size<scalar_t>];
    alignas(ALIGN_BYTES) scalar_t cof6[cvfem_hex8_vec_size<scalar_t>], cof7[cvfem_hex8_vec_size<scalar_t>], cof8[cvfem_hex8_vec_size<scalar_t>];
    alignas(ALIGN_BYTES) scalar_t det[cvfem_hex8_vec_size<scalar_t>];
    Hex8InputPackT<scalar_t>    in;
    Hex8ResidualPackT<scalar_t> outp;
    Hex8RhieChowPackT<scalar_t> rcp;
    for (ptrdiff_t begin = x.e_start; begin < x.e_end; begin += cvfem_hex8_vec_size<scalar_t>) {
        const int nlanes = int(MIN((ptrdiff_t)cvfem_hex8_vec_size<scalar_t>, x.e_end - begin));
        gather_hex8_simd_from_pack(pack_elems, pack_u, adj_ptr, det_ptr, begin, nlanes, in,
                                   cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det);
        if (with_rc) {
            cvfem_hex8_gather_rc_from_pack(pack_elems, pk.pgx, pk.pgy, pk.pgz, begin, nlanes, rcp);
        }
        cvfem_hex8_ns_upwind_residual_sumfact_simd(
                rho, mu, cof0, cof1, cof2, cof3, cof4, cof5, cof6, cof7, cof8, det, in, outp,
                with_rc ? &rcp : nullptr, rhie_chow_scale);
        scatter_hex8_simd_to_pack(pack_elems, pack_out, begin, nlanes, outp);
    }
}

// ONE PACK'S LANE LOOP, ISOPARAMETRIC. The Jacobian comes per sub-control surface from the
// coordinates the pack staged, so this takes no adjugate table and no Rhie-Chow: the
// isoparametric SIMD kernel carries no such term.
template <typename scalar_t, typename idx_t, typename pack_idx_t>
static SFEM_INLINE void cvfem_hex8_residual_lanes_isoparam(
        const Hex8PackExtentT<idx_t>       &x,
        const Hex8PackCoordsT<scalar_t>    &pk,
        pack_idx_t **const SFEM_RESTRICT    pack_elems,
        const scalar_t *const SFEM_RESTRICT pack_u,
        scalar_t *const SFEM_RESTRICT       pack_out,
        const scalar_t                      rho,
        const scalar_t                      mu) {
    Hex8InputPackT<scalar_t>    in;
    Hex8CoordPackT<scalar_t>    xyz;
    Hex8ResidualPackT<scalar_t> outp;
    for (ptrdiff_t begin = x.e_start; begin < x.e_end; begin += cvfem_hex8_vec_size<scalar_t>) {
        const int nlanes = int(MIN((ptrdiff_t)cvfem_hex8_vec_size<scalar_t>, x.e_end - begin));
        gather_hex8_isoparam_simd_from_pack(pack_elems, pack_u, pk.x, pk.y, pk.z, begin, nlanes,
                                            in, xyz);
        cvfem_hex8_ns_upwind_residual_isoparam_simd(rho, mu, xyz, in, outp);
        scatter_hex8_simd_to_pack(pack_elems, pack_out, begin, nlanes, outp);
    }
}


// ONE PACK'S LANE LOOP FOR THE JACOBIAN ACTION, AFFINE. Called by both drains' sweeps -- the
// contiguous packed action and the pack-coloured one -- which differ in nothing else. It carries
// everything the operator can: the Rhie-Chow term exact or frozen, the direction's reconstructed
// gradient, and the exact higher-order correction when `ugrad` and `vgrad` are given. The
// pack-coloured sweep passes null for those two, so that branch folds away for it.
template <typename scalar_t, typename idx_t, typename pack_idx_t, typename geom_t>
static SFEM_INLINE void cvfem_hex8_action_lanes_affine(
        const Hex8PackExtentT<idx_t>              &x,
        const Hex8PackCoordsT<scalar_t>           &pk,
        const Hex8PackQGradT<scalar_t>            &qg,
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT        det_ptr,
        idx_t **const SFEM_RESTRICT                mesh_elems,
        geom_t **const SFEM_RESTRICT               points,
        pack_idx_t **const SFEM_RESTRICT           pack_elems,
        const scalar_t *const SFEM_RESTRICT        pack_u,
        const scalar_t *const SFEM_RESTRICT        pack_dir,
        scalar_t *const SFEM_RESTRICT              pack_out,
        const scalar_t *const SFEM_RESTRICT        rc_coeff,
        const scalar_t *const SFEM_RESTRICT        rc_w,
        const scalar_t *const SFEM_RESTRICT        ugrad,
        const scalar_t *const SFEM_RESTRICT        vgrad,
        const scalar_t                             rho,
        const scalar_t                             mu,
        const scalar_t                             rhie_chow_scale,
        const int                                  with_rc,
        const bool                                 with_qg,
        const bool                                 with_ho,
        const Hex8RcConfigT<scalar_t>             &rc_cfg,
        // The limiter and its coefficient, read by the higher-order correction. They travel in
        // the gradient pack, which is where the element kernel looks for them.
        const int                                  limiter,
        const scalar_t                             venkat_c) {
    Hex8InputPackT<scalar_t>    u_pack;
    Hex8InputPackT<scalar_t>    du_pack;
    Hex8ResidualPackT<scalar_t> outp;
    Hex8CoordPackT<scalar_t>    xyz;
    Hex8RhieChowPackT<scalar_t> rcp;
    Hex8UGradPackT<scalar_t>    hop, hovp;
    hop.limiter  = limiter;
    hop.venkat_c = venkat_c;
        for (ptrdiff_t begin = x.e_start; begin < x.e_end; begin += cvfem_hex8_vec_size<scalar_t>) {
            const int nlanes = int(MIN((ptrdiff_t)cvfem_hex8_vec_size<scalar_t>, x.e_end - begin));
            alignas(ALIGN_BYTES) scalar_t cof0[cvfem_hex8_vec_size<scalar_t>], cof1[cvfem_hex8_vec_size<scalar_t>],
                    cof2[cvfem_hex8_vec_size<scalar_t>];
            alignas(ALIGN_BYTES) scalar_t cof3[cvfem_hex8_vec_size<scalar_t>], cof4[cvfem_hex8_vec_size<scalar_t>],
                    cof5[cvfem_hex8_vec_size<scalar_t>];
            alignas(ALIGN_BYTES) scalar_t cof6[cvfem_hex8_vec_size<scalar_t>], cof7[cvfem_hex8_vec_size<scalar_t>],
                    cof8[cvfem_hex8_vec_size<scalar_t>];
            alignas(ALIGN_BYTES) scalar_t det[cvfem_hex8_vec_size<scalar_t>];
            gather_hex8_action_simd_from_pack(pack_elems,
                                              pack_u,
                                              pack_dir,
                                              adj_ptr,
                                              det_ptr,
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
                cvfem_hex8_gather_rc_from_pack(pack_elems, pk.pgx, pk.pgy, pk.pgz,
                                               begin, nlanes, rcp);
                cvfem_hex8_gather_rc_coeff(rc_coeff, rc_w, rc_cfg, begin, nlanes, rcp);
            }
            if (with_qg)
                cvfem_hex8_gather_qg_from_pack(pack_elems, qg.x, qg.y, qg.z, begin, nlanes, rcp);
            if (with_ho) {
                for (int lane = 0; lane < cvfem_hex8_vec_size<scalar_t>; ++lane) {
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
                        const idx_t gn = mesh_elems[a][e];
                        hop.x[a][lane] = scalar_t(points[0][gn]);
                        hop.y[a][lane] = scalar_t(points[1][gn]);
                        hop.z[a][lane] = scalar_t(points[2][gn]);
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
                                                      rhie_chow_scale,
                                                      with_qg,
                                                      scalar_t(0),
                                                      with_ho ? &hop : nullptr,
                                                      with_ho ? &hovp : nullptr);
            scatter_hex8_simd_to_pack(pack_elems, pack_out, begin, nlanes, outp);
        }
}


// ONE PACK'S LANE LOOP FOR THE JACOBIAN ACTION, ISOPARAMETRIC. The Jacobian comes per
// sub-control surface from the coordinates the pack staged, so this takes no adjugate table and
// no Rhie-Chow -- the isoparametric SIMD kernel carries no such term, which is why the driver
// refuses --rhie-chow on this geometry for a pack-based layout.
template <typename scalar_t, typename idx_t, typename pack_idx_t>
static SFEM_INLINE void cvfem_hex8_action_lanes_isoparam(
        const Hex8PackExtentT<idx_t>       &x,
        const Hex8PackCoordsT<scalar_t>    &pk,
        pack_idx_t **const SFEM_RESTRICT    pack_elems,
        const scalar_t *const SFEM_RESTRICT pack_u,
        const scalar_t *const SFEM_RESTRICT pack_dir,
        scalar_t *const SFEM_RESTRICT       pack_out,
        const scalar_t                      rho,
        const scalar_t                      mu) {
    Hex8InputPackT<scalar_t>    u_pack;
    Hex8InputPackT<scalar_t>    du_pack;
    Hex8ResidualPackT<scalar_t> outp;
    Hex8CoordPackT<scalar_t>    xyz;
        for (ptrdiff_t begin = x.e_start; begin < x.e_end; begin += cvfem_hex8_vec_size<scalar_t>) {
            const int nlanes = int(MIN((ptrdiff_t)cvfem_hex8_vec_size<scalar_t>, x.e_end - begin));
            gather_hex8_isoparam_action_simd_from_pack(pack_elems,
                                                       pack_u,
                                                       pack_dir,
                                                       pk.x,
                                                       pk.y,
                                                       pk.z,
                                                       begin,
                                                       nlanes,
                                                       u_pack,
                                                       du_pack,
                                                       xyz);
            cvfem_hex8_ns_upwind_jacobian_action_isoparam_simd(rho, mu, xyz, u_pack, du_pack, outp);
            scatter_hex8_simd_to_pack(pack_elems, pack_out, begin, nlanes, outp);
        }
}

#endif  // CVFEM_HEX8_PACK_STAGING_HPP
