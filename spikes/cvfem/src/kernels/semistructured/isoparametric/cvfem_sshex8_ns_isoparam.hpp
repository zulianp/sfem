#ifndef CVFEM_SSHEX8_NS_ISOPARAM_HPP
#define CVFEM_SSHEX8_NS_ISOPARAM_HPP

// The semi-structured format's ISOPARAMETRIC sweeps: every micro cell derives its own
// geometry from its own eight corners, because its macro element is curved and a hoisted
// geometry would be wrong in a way refinement does not fix.
//
// The other half of the split; see affine/cvfem_sshex8_ns_affine.hpp for why it is two ranges
// rather than a template parameter, and ../cvfem_sshex8_ns.hpp for the shared micro-cell
// kernels these sweeps drive.
//
// A sweep here HOISTS NOTHING, and that is the second thing the split buys beyond the removed
// branch: the branching sweep built the hoisted macro geometry for every macro element,
// including the curved ones that then threw it away.

#include "kernels/semistructured/cvfem_sshex8_ns.hpp"

// The naive apply over CURVED macro elements. The affine half is sscvfem_apply_naive_affine.
inline SFEM_NOINLINE void sscvfem_apply_naive_isoparam(
        // The range this call is to cover, as positions in macro_order. DESIGN.md: the
        // threading is abstract outside the sweep and what arrives is a range, so the sweep
        // owns no parallel region.
        const cvfem_range r,
        // The curvature partition. This sweep's range is empty unless something is curved, so
        // null is never indexed; the identity is kept for a caller that passes a whole range.
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
        sscvfem_apply_naive_curved_macro(box_lx, box_ly, box_lz, elems, level, pres, pgx_src, pgy_src,
                                         pgz_src, points, upwind_eps, ux_src, uy_src, uz_src, rcfg, rho,
                                         mu, e, off, dir, jv);
    }
}


// The macro-local apply over CURVED macro elements: every micro cell derives its own geometry,
// and nothing hoists.
inline SFEM_NOINLINE void sscvfem_apply_macro_local_isoparam(
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
        const int nxe_src,
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
    const int L   = level;
    const int nxe = nxe_src;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    const SSMacroScratch s = sscvfem_macro_scratch(nxe, CVFEM_HEX8_N_FIELDS, true, false, false);

    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const ptrdiff_t e = macro_order ? macro_order[i] : i;
        sscvfem_macro_gather(s, elems, points, pres, pgx_src, pgy_src, pgz_src, nullptr, nullptr,
                             nullptr, nullptr, ux_src, uy_src, uz_src, dir, e, nxe,
                             CVFEM_HEX8_N_FIELDS);

        sscvfem_apply_macro_local_curved_macro(box_lx, box_ly, box_lz, rcfg, rho, mu, upwind_eps, s,
                                               level, off);
        sscvfem_macro_drain(s, nullptr, nullptr, nxe, e, jv);
    }
}


// The naive residual over CURVED macro elements: every micro cell derives its own geometry, and
// nothing hoists.
inline SFEM_NOINLINE void sscvfem_residual_naive_isoparam(
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
        const Hex8PecletConfig<scalar_t> peclet,
        idx_t **const SFEM_RESTRICT elems,
        const int level,
        const ptrdiff_t nnodes,
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
                                                 scalar_t *const SFEM_RESTRICT res) {
    // The destination arrives ZEROED. It used to be zeroed here, which was correct while this
    // sweep owned its parallel region and ran once; driven by a range it runs once per thread,
    // and every thread would re-zero the whole array -- over the contributions the others had
    // already accumulated. The launcher zeroes it before the region opens.

    const int L = level;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const ptrdiff_t e = macro_order ? macro_order[i] : i;
        sscvfem_residual_naive_curved_macro(box_lx, box_ly, box_lz, peclet, elems, level, pres,
                                            pgx_src, pgy_src, pgz_src, points, upwind_eps, ux_src,
                                            uy_src, uz_src, rcfg, rho, mu, e, off, res);
    }
}


// The naive block diagonal over CURVED macro elements: every micro cell derives its own
// geometry, and nothing hoists.
inline SFEM_NOINLINE void sscvfem_block_diag_naive_isoparam(
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
        const ptrdiff_t nnodes,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx_src,
        const scalar_t *const SFEM_RESTRICT pgy_src,
        const scalar_t *const SFEM_RESTRICT pgz_src,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
        const Hex8RcConfig rcfg, const scalar_t rho, const scalar_t mu,
                                                   scalar_t *const SFEM_RESTRICT out) {

    const int L = level;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const ptrdiff_t e = macro_order ? macro_order[i] : i;
        sscvfem_block_diag_naive_curved_macro(box_lx, box_ly, box_lz, elems, level, pres, pgx_src,
                                              pgy_src, pgz_src, points, ux_src, uy_src, uz_src, rcfg,
                                              rho, mu, e, off, out);
    }
}

#endif  // CVFEM_SSHEX8_NS_ISOPARAM_HPP
