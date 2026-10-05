#ifndef CVFEM_HEX8_ELEMENT_GATHER_HPP
#define CVFEM_HEX8_ELEMENT_GATHER_HPP

// THE PER-ELEMENT GATHER, ONCE.
//
// Both operator families defined gather_element_fields and gather_element_coords, and the note
// at the top of cvfem_hex8_ns_core.hpp explains why they are kept apart in general: they differ
// in physics, not only in layout, and the two MeshData types never appear in one translation
// unit. That reasoning covered these two while they took a MeshData -- each gathered out of its
// own family's mesh.
//
// It stopped covering them when they stopped taking one. Converted to arrays they became byte
// identical, which makes them one operation defined twice, and that is a redundant path whatever
// the history. One copy, here, with the kernels that call it.
//
// Hex8ExtraScratch comes along because it is the per-element staging those gathers fill for the
// reference path, and it has no staging dependency left either.
#include "kernels/microkernels/hex8/affine/cvfem_hex8_affine_geometry.hpp"
#include "kernels/cvfem_hex8_flags.hpp"

#include <cstring>   // memcpy, for the SoA geometry gather

// The affine geometry, out of the arrays MeshData publishes. These took the mesh until the
// cascade reached them; with only arrays left they are kernel helpers like the two above.
template <typename scalar_t>
static SFEM_INLINE void load_hex8_adj(const scalar_t *const *const SFEM_RESTRICT adj_ptr,
                                      const scalar_t *const SFEM_RESTRICT        det_ptr,
                                      const ptrdiff_t e, scalar_t adj[9], scalar_t *det) {
    for (int c = 0; c < 9; ++c) adj[c] = adj_ptr[c][(size_t)e];
    *det = det_ptr[(size_t)e];
}

template <typename scalar_t>
static SFEM_INLINE void gather_hex8_adj_soa(const scalar_t *const *const SFEM_RESTRICT adj_ptr,
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
    const size_t n = (size_t)nlanes * sizeof(scalar_t);
    std::memcpy(cof0, adj_ptr[0] + begin, n);
    std::memcpy(cof1, adj_ptr[1] + begin, n);
    std::memcpy(cof2, adj_ptr[2] + begin, n);
    std::memcpy(cof3, adj_ptr[3] + begin, n);
    std::memcpy(cof4, adj_ptr[4] + begin, n);
    std::memcpy(cof5, adj_ptr[5] + begin, n);
    std::memcpy(cof6, adj_ptr[6] + begin, n);
    std::memcpy(cof7, adj_ptr[7] + begin, n);
    std::memcpy(cof8, adj_ptr[8] + begin, n);
    std::memcpy(det, det_ptr + begin, n);
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


template <typename scalar_t, typename idx_t>
static SFEM_INLINE void gather_element_fields(idx_t **const SFEM_RESTRICT elems,
                                              const scalar_t *const SFEM_RESTRICT ux_src,
                                              const scalar_t *const SFEM_RESTRICT uy_src,
                                              const scalar_t *const SFEM_RESTRICT uz_src,
                                              const scalar_t *const SFEM_RESTRICT p_src,
                                              const ptrdiff_t                  e,
                                              scalar_t *const SFEM_RESTRICT    ux,
                                              scalar_t *const SFEM_RESTRICT    uy,
                                              scalar_t *const SFEM_RESTRICT    uz,
                                              scalar_t *const SFEM_RESTRICT    p) {
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        const idx_t g = elems[a][e];
        ux[a]                = ux_src[g];
        uy[a]                = uy_src[g];
        uz[a]                = uz_src[g];
        p[a]                 = p_src[g];
    }
}

template <typename scalar_t, typename geom_t, typename idx_t>
static SFEM_INLINE void gather_element_coords(idx_t **const SFEM_RESTRICT elems,
                                              geom_t **const SFEM_RESTRICT points,
                                              const ptrdiff_t               e,
                                              scalar_t *const SFEM_RESTRICT x,
                                              scalar_t *const SFEM_RESTRICT y,
                                              scalar_t *const SFEM_RESTRICT z) {
    const auto *const px = points[0];
    const auto *const py = points[1];
    const auto *const pz = points[2];
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        const idx_t g = elems[a][e];
        x[a]                 = scalar_t(px[g]);
        y[a]                 = scalar_t(py[g]);
        z[a]                 = scalar_t(pz[g]);
    }
}

struct Hex8ExtraScratch {
    scalar_t     x[CVFEM_HEX8_N_NODES], y[CVFEM_HEX8_N_NODES], z[CVFEM_HEX8_N_NODES];
    scalar_t     pgx[CVFEM_HEX8_N_NODES], pgy[CVFEM_HEX8_N_NODES], pgz[CVFEM_HEX8_N_NODES];
    scalar_t     qgx[CVFEM_HEX8_N_NODES], qgy[CVFEM_HEX8_N_NODES], qgz[CVFEM_HEX8_N_NODES];
    // The advecting velocity, for the convective branch of the Rhie-Chow time scale.
    scalar_t     ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES], uz[CVFEM_HEX8_N_NODES];
    Hex8RhieChow rc{};
    int          fmask{0};

    // Takes the arrays, not the mesh. This method is reached from inside a pack sweep, so a
    // MeshData parameter here is the last thing keeping that sweep's signature tied to the
    // staging layer. The sources carry a _src suffix because this object's own members already
    // own the short names -- its whole job is to copy pgx[] out of pgx_src[].
    SFEM_INLINE void load(idx_t **const SFEM_RESTRICT  elems,
                          geom_t **const SFEM_RESTRICT points,
                          const uint8_t *const SFEM_RESTRICT  face_mask,
                          const scalar_t *const SFEM_RESTRICT pgx_src,
                          const scalar_t *const SFEM_RESTRICT pgy_src,
                          const scalar_t *const SFEM_RESTRICT pgz_src,
                          const scalar_t *const SFEM_RESTRICT qgx_src,
                          const scalar_t *const SFEM_RESTRICT qgy_src,
                          const scalar_t *const SFEM_RESTRICT qgz_src,
                          const scalar_t *const SFEM_RESTRICT ux_src,
                          const scalar_t *const SFEM_RESTRICT uy_src,
                          const scalar_t *const SFEM_RESTRICT uz_src,
                          const scalar_t *const *const SFEM_RESTRICT adj_ptr,
                          const scalar_t *const SFEM_RESTRICT        det_ptr,
                          const Hex8Extras &opt, const ptrdiff_t e) {
        if (!opt.with_rc && !opt.with_bnd) return;
        gather_element_coords(elems, points, e, x, y, z);
        if (opt.with_bnd) fmask = (int)face_mask[(size_t)e];
        if (opt.with_rc) {
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const idx_t g = elems[a][e];
                pgx[a]               = pgx_src[g];
                pgy[a]               = pgy_src[g];
                pgz[a]               = pgz_src[g];
                ux[a]                = ux_src[g];
                uy[a]                = uy_src[g];
                uz[a]                = uz_src[g];
            }
            rc = Hex8RhieChow{};
            rc.x = x; rc.y = y; rc.z = z;
            rc.pgx = pgx; rc.pgy = pgy; rc.pgz = pgz;
            rc.scale = opt.rcfg.scale;
            rc.ux = ux; rc.uy = uy; rc.uz = uz;
            rc.tau = opt.rcfg.tau;
            // The affine edge vectors, so this reference discretises the same operator the
            // vectorised kernels do. Without them the reference differences node coordinates
            // while the kernel under test takes the Jacobian column, and the two disagree
            // wherever the mesh is not exactly affine in floating point -- amplified by the
            // near-cancellation in the Rhie-Chow correction into a visible error.
            {
                scalar_t adj_[9], det_;
                load_hex8_adj(adj_ptr, det_ptr, e, adj_, &det_);
                scalar_t ex_[3], ey_[3], ez_[3];
                cvfem_hex8_affine_edge_cols(adj_[0], adj_[1], adj_[2], adj_[3], adj_[4], adj_[5],
                                            adj_[6], adj_[7], adj_[8], det_, ex_, ey_, ez_);
                for (int q_ = 0; q_ < 3; ++q_) {
                    rc.ecol[0 * 3 + q_] = ex_[q_];
                    rc.ecol[1 * 3 + q_] = ey_[q_];
                    rc.ecol[2 * 3 + q_] = ez_[q_];
                }
                rc.has_ecol = true;
            }
            if (opt.with_qg) {
                for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                    const idx_t g = elems[a][e];
                    qgx[a]               = qgx_src[g];
                    qgy[a]               = qgy_src[g];
                    qgz[a]               = qgz_src[g];
                }
                rc.qgx = qgx;
                rc.qgy = qgy;
                rc.qgz = qgz;
            }
        }
    }
};

#endif  // CVFEM_HEX8_ELEMENT_GATHER_HPP
