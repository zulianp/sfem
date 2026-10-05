#ifndef CVFEM_SSHEX8_NS_AFFINE_HPP
#define CVFEM_SSHEX8_NS_AFFINE_HPP

// The semi-structured format's AFFINE sweeps: one geometry hoisted over a macro element's
// micro cells, and no test anywhere of whether that is allowed.
//
// WHY THIS SPLIT IS TWO RANGES AND NOT A TEMPLATE PARAMETER. Whether a macro element is
// curved is mesh data, not a configuration -- one mesh carries curved and straight macro
// elements side by side, and which is which is known only per element at run time. So the
// distinction cannot be a template argument the way the flat layouts' geometry could. But it
// does not change between applies either, so SSMeshData partitions the macro elements by
// curvature once per level (see sscvfem_classify_macros) and each sweep is handed the range
// that holds its own kind. Straight elements first, curved ones after, with n_straight the
// boundary; the sweeps index through macro_order, which is null when nothing is curved and
// then means the identity.
//
// What a sweep here gains over the branching one it came from is the branch itself, out of a
// loop that runs L^3 times per macro element -- seven of them in the Jacobian action alone.
//
// What stays shared is in ../cvfem_sshex8_ns.hpp: the micro-cell kernels, the macro-element
// gather, the scatter and the shared reduction. That is most of each sweep, and is why a
// naive folder split would have duplicated ten sweeps instead of separating them. Same
// discipline as the packed layout, whose staging, extent and drain went into
// ../../packed/cvfem_pack_scratch.hpp before that format was split.

#include "kernels/semistructured/cvfem_sshex8_ns.hpp"

// The naive apply over STRAIGHT macro elements: the hoisted geometry, with nothing asking
// whether it applies. The curved half is sscvfem_apply_naive_isoparam.
inline SFEM_NOINLINE void sscvfem_apply_naive_affine(
        // The range this call is to cover, as positions in macro_order. DESIGN.md: the
        // threading is abstract outside the sweep and what arrives is a range, so the sweep
        // owns no parallel region.
        const cvfem_range r,
        // The curvature partition; null means the identity, which is a mesh with nothing
        // curved. See SSMeshData::macro_order.
        const ptrdiff_t *const SFEM_RESTRICT macro_order,
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const scalar_t box_lx,
        const scalar_t box_ly,
        const scalar_t box_lz,
        idx_t **const SFEM_RESTRICT elems,
        const int level,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx_src,
        const scalar_t *const SFEM_RESTRICT pgy_src,
        const scalar_t *const SFEM_RESTRICT pgz_src,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t upwind_eps,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
        const Hex8RcConfig rcfg, const scalar_t rho, const scalar_t mu,
                                              const scalar_t *const SFEM_RESTRICT dir,
                                              scalar_t *const SFEM_RESTRICT       jv) {
    const int L = level;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const ptrdiff_t e = macro_order ? macro_order[i] : i;
        // The geometry every micro cell of this macro element uses; see sscvfem_hoisted_cell.
        // Real positions stay per cell.
        scalar_t hx[8], hy[8], hz[8];
        {
            int ext[8];
            sscvfem_macro_corner_offsets(L, ext);
            for (int a = 0; a < 8; ++a) {
                const idx_t gm = elems[ext[a]][e];
                hx[a] = (scalar_t)points[0][gm];
                hy[a] = (scalar_t)points[1][gm];
                hz[a] = (scalar_t)points[2][gm];
            }
            sscvfem_hoisted_cell(hx, hy, hz, L, hx, hy, hz);
        }
        for (int zi = 0; zi < L; ++zi)
            for (int yi = 0; yi < L; ++yi)
                for (int xi = 0; xi < L; ++xi)
                    sscvfem_apply_naive_cell(box_lx, box_ly, box_lz, elems, pres, pgx_src, pgy_src,
                                             pgz_src, points, upwind_eps, ux_src, uy_src, uz_src, rcfg,
                                             rho, mu, e, L, xi, yi, zi, off, hx, hy, hz, dir, jv);
    }
}

#endif  // CVFEM_SSHEX8_NS_AFFINE_HPP
