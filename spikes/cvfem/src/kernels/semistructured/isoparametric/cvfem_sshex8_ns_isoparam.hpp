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
template <typename scalar_t, typename geom_t, typename idx_t>
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
        const Hex8RcConfigT<scalar_t> rcfg, const scalar_t rho, const scalar_t mu,
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
template <typename scalar_t, typename geom_t, typename idx_t>
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
        const Hex8RcConfigT<scalar_t> rcfg, const scalar_t rho, const scalar_t mu,
                                                    const scalar_t *const SFEM_RESTRICT dir,
                                                    scalar_t *const SFEM_RESTRICT       jv) {
    const int L   = level;
    const int nxe = nxe_src;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    const SSMacroScratchT<scalar_t, idx_t> s = sscvfem_macro_scratch<scalar_t, idx_t>(nxe, CVFEM_HEX8_N_FIELDS, true, false, false);

    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const ptrdiff_t e = macro_order ? macro_order[i] : i;
        sscvfem_macro_gather(s, elems, points, pres, pgx_src, pgy_src, pgz_src, static_cast<const scalar_t *>(nullptr), static_cast<const scalar_t *>(nullptr),
                             static_cast<const scalar_t *>(nullptr), static_cast<const scalar_t *>(nullptr), ux_src, uy_src, uz_src, dir, e, nxe,
                             CVFEM_HEX8_N_FIELDS);

        sscvfem_apply_macro_local_curved_macro(box_lx, box_ly, box_lz, rcfg, rho, mu, upwind_eps, s,
                                               level, off);
        sscvfem_macro_drain(s, static_cast<const int *>(nullptr), static_cast<scalar_t *>(nullptr), nxe, e, jv);
    }
}


// The naive residual over CURVED macro elements: every micro cell derives its own geometry, and
// nothing hoists.
template <typename scalar_t, typename geom_t, typename idx_t>
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
        const Hex8RcConfigT<scalar_t> rcfg, const scalar_t rho, const scalar_t mu,
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
template <typename scalar_t, typename geom_t, typename idx_t>
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
        const Hex8RcConfigT<scalar_t> rcfg, const scalar_t rho, const scalar_t mu,
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


// The nodal-gradient scatter sweep over CURVED macro elements.
//
// IT BUILDS NO MACRO GEOMETRY AT ALL, which is more than the removed branch: the branching sweep
// gathered the macro element's eight corners and evaluated its Jacobian for every macro element,
// and on a curved one used neither -- the only thing it took from them was the degenerate-
// determinant guard, and a curved cell carries its own.
template <typename scalar_t, typename geom_t, typename idx_t>
inline void sscvfem_nodal_grad_scatter_isoparam(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        idx_t **const SFEM_RESTRICT elems,
        const int level,
        const int nxe,
        geom_t **const SFEM_RESTRICT points,
        
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const ptrdiff_t n_slots,
        const ptrdiff_t *const SFEM_RESTRICT red_idx,
        const ptrdiff_t *const SFEM_RESTRICT red_ptr,
        const idx_t *const SFEM_RESTRICT shared_node,
        const int *const SFEM_RESTRICT slot,
        scalar_t *const SFEM_RESTRICT stage,
        const ptrdiff_t n_shared, const scalar_t *const SFEM_RESTRICT src,
                                             const int stride, scalar_t *const SFEM_RESTRICT ogx,
                                             scalar_t *const SFEM_RESTRICT ogy, scalar_t *const SFEM_RESTRICT ogz,
                                             // The element range this call covers, as positions
                                             // in macro_order. DESIGN.md: the threading is
                                             // abstract outside the sweep.
                                             const cvfem_range r,
                                             // The curvature partition; null means the identity.
                                             // See SSMeshData::macro_order.
                                             const ptrdiff_t *const SFEM_RESTRICT macro_order) {
    if (r.begin >= r.end) return;

    const int L = level;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    // The staging buffer is pre-zeroed by the caller when this is a PARTIAL range; see
    // sscvfem_nodal_grad_strided. It is shared work that must happen once, not per thread.

    // Per-thread scratch from the kernels' own arena, not std::vector locals: slots 5/6 are the
    // semi-structured element sweeps', shared between them because only one is live inside a
    // parallel region and they all want the same macro-element size.
    scalar_t *const SFEM_RESTRICT _arena5 = thread_scratch<scalar_t>(5, ((size_t)nxe) + ((size_t)nxe * SSCVFEM_NGRAD));
    scalar_t *const SFEM_RESTRICT lp = _arena5;
    scalar_t *const SFEM_RESTRICT lacc = _arena5 + ((size_t)nxe);
    idx_t *const SFEM_RESTRICT _arena6 = thread_scratch<idx_t>(6, ((size_t)nxe));
    idx_t *const SFEM_RESTRICT lg = _arena6;

    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const ptrdiff_t e = macro_order ? macro_order[i] : i;
        if (slot) std::fill(lacc, lacc + ((size_t)nxe * SSCVFEM_NGRAD), scalar_t(0));
        // Only the field. The coordinates used to be gathered for every node of the
        // macro-element -- three arrays of (L+1)^3 -- to feed a geometry computation that
        // is the same for all of them.
        for (int a = 0; a < nxe; ++a) {
            const idx_t g = elems[a][e];
            lg[(size_t)a]        = g;
            lp[(size_t)a]        = src[(ptrdiff_t)g * stride];
        }

        for (int zi = 0; zi < L; ++zi)
            for (int yi = 0; yi < L; ++yi)
                for (int xi = 0; xi < L; ++xi)
                    sscvfem_nodal_grad_cell(elems, points, lp, lg, slot, lacc, e, L, xi, yi, zi, off,
                                            static_cast<const scalar_t *>(nullptr), scalar_t(0), ogx, ogy, ogz);

        if (slot) {
            scalar_t *dst[SSCVFEM_NGRAD] = {ogx, ogy, ogz};
            sscvfem_scatter_element_soa_w<SSCVFEM_NGRAD>(slot, const_cast<scalar_t *>(stage), nxe, e, lg, lacc, dst);
        }
    }

    // The shared reduction is the caller's: a second, independent loop over the reduction
    // rows, run after this pass's threads have joined.

    // No normalisation here: the caller divides once, after whichever passes it ran. See
    // sscvfem_nodal_grad_normalize.
}


// The hoisted Jacobian action over CURVED macro elements: every micro cell builds its own
// SSMacroGeom, and the macro element's is never built at all -- the branching sweep built one
// per macro element and the curved branch overwrote it in the first cell.
template <typename scalar_t, typename geom_t, typename idx_t>
inline SFEM_NOINLINE void sscvfem_apply_macro_hoisted_isoparam(
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
        const scalar_t bc_p,
        const scalar_t bc_tx,
        const scalar_t bc_ty,
        const scalar_t bc_tz,
        idx_t **const SFEM_RESTRICT elems,
        const int level,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const uint8_t *const SFEM_RESTRICT natural_mask,
        const uint8_t *const SFEM_RESTRICT pressure_mask,
        const uint8_t *const SFEM_RESTRICT traction_mask,
        const int nxe_src,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx_src,
        const scalar_t *const SFEM_RESTRICT pgy_src,
        const scalar_t *const SFEM_RESTRICT pgz_src,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT qgx_src,
        const scalar_t *const SFEM_RESTRICT qgy_src,
        const scalar_t *const SFEM_RESTRICT qgz_src,
        const scalar_t upwind_eps,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
        const Hex8RcConfigT<scalar_t> rcfg,
        
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const ptrdiff_t *const SFEM_RESTRICT red_idx,
        const ptrdiff_t *const SFEM_RESTRICT red_ptr,
        const idx_t *const SFEM_RESTRICT shared_node,
        const int *const SFEM_RESTRICT slot,
        scalar_t *const SFEM_RESTRICT stage,
        const ptrdiff_t n_shared, const scalar_t rho, const scalar_t mu,
                                                            const scalar_t *const SFEM_RESTRICT dir,
                                                            scalar_t *const SFEM_RESTRICT       jv) {
    const int L   = level;
    const int nxe = nxe_src;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    // Whether the direction's pressure gradient is there at all. Above the scratch because the
    // scratch is sized from it.
    const bool           has_qg = qgx_src;
    const SSMacroScratchT<scalar_t, idx_t> s = sscvfem_macro_scratch<scalar_t, idx_t>(nxe, CVFEM_HEX8_N_FIELDS, true, has_qg, false);

    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const ptrdiff_t e = macro_order ? macro_order[i] : i;
        sscvfem_macro_gather(s, elems, points, pres, pgx_src, pgy_src, pgz_src, qgx_src, qgy_src,
                             qgz_src, static_cast<const scalar_t *>(nullptr), ux_src, uy_src, uz_src, dir, e, nxe,
                             CVFEM_HEX8_N_FIELDS);

        // Per macro element, and the curved sweep reads the same one: a call per micro cell
        // there took this unit past the point where GCC inlines sscvfem_rc_config, which then
        // became a call in every cell of the block diagonal, 9% slower on boxes.
        const Hex8RcConfigT<scalar_t> rc_macro = rcfg;
        sscvfem_action_hoisted_curved_macro(box_lx, box_ly, box_lz, bc_p, bc_tx, bc_ty, bc_tz,
                                            face_mask, natural_mask, pressure_mask, traction_mask,
                                            rc_macro, rho, mu, upwind_eps, s, has_qg, e, level, off);

        sscvfem_macro_drain(s, slot, const_cast<scalar_t *>(stage), nxe, e, jv);
    }

    // The shared reduction is the caller's: a second, independent loop over the reduction
    // rows, run after this sweep's threads have joined. See sscvfem_drain_shared.
}


// The residual over CURVED macro elements: every micro cell builds its own geometry and takes
// its Rhie-Chow distances from its own corners, which is what keeps it consistent with the
// Jacobian action's.
template <typename scalar_t, typename geom_t, typename idx_t>
inline SFEM_NOINLINE void sscvfem_residual_isoparam(
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
        const scalar_t bc_p,
        const scalar_t bc_tx,
        const scalar_t bc_ty,
        const scalar_t bc_tz,
        const int conv_ho,
        const int conv_limiter,
        const Hex8PecletConfig<scalar_t> peclet,
        const scalar_t conv_venkat_c,
        idx_t **const SFEM_RESTRICT elems,
        const int level,
        Hex8LimiterStats *const limiter_stats,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const uint8_t *const SFEM_RESTRICT natural_mask,
        const uint8_t *const SFEM_RESTRICT pressure_mask,
        const uint8_t *const SFEM_RESTRICT traction_mask,
        const int nxe_src,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx_src,
        const scalar_t *const SFEM_RESTRICT pgy_src,
        const scalar_t *const SFEM_RESTRICT pgz_src,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT ugrad_f,
        const scalar_t upwind_eps,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
        const Hex8RcConfigT<scalar_t> rcfg,
        
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const ptrdiff_t *const SFEM_RESTRICT red_idx,
        const ptrdiff_t *const SFEM_RESTRICT red_ptr,
        const idx_t *const SFEM_RESTRICT shared_node,
        const int *const SFEM_RESTRICT slot,
        scalar_t *const SFEM_RESTRICT stage,
        const ptrdiff_t n_shared, const scalar_t rho,
                                                 const scalar_t                mu,
                                                 scalar_t *const SFEM_RESTRICT res) {
    const int L   = level;
    const int nxe = nxe_src;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    const SSMacroScratchT<scalar_t, idx_t> s = sscvfem_macro_scratch<scalar_t, idx_t>(nxe, CVFEM_HEX8_N_FIELDS, false, false, conv_ho != 0);

    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const ptrdiff_t e = macro_order ? macro_order[i] : i;
        sscvfem_macro_gather(s, elems, points, pres, pgx_src, pgy_src, pgz_src, static_cast<const scalar_t *>(nullptr), static_cast<const scalar_t *>(nullptr),
                             static_cast<const scalar_t *>(nullptr), ugrad_f, ux_src, uy_src, uz_src, static_cast<const scalar_t *>(nullptr), e, nxe,
                             CVFEM_HEX8_N_FIELDS);

        // Per macro element, and the curved sweep reads the same one: a call per micro cell
        // there took this unit past the point where GCC inlines sscvfem_rc_config, which then
        // became a call in every cell of the block diagonal, 9% slower on boxes.
        const Hex8RcConfigT<scalar_t> rc_macro = rcfg;
        sscvfem_residual_curved_macro(box_lx, box_ly, box_lz, bc_p, bc_tx, bc_ty, bc_tz, conv_ho,
                                      conv_limiter, peclet, conv_venkat_c, limiter_stats, face_mask,
                                      natural_mask, pressure_mask, traction_mask, upwind_eps, rcfg,
                                      rc_macro, rho, mu, s, e, level, off, res);

        sscvfem_macro_drain(s, slot, const_cast<scalar_t *>(stage), nxe, e, res);
    }

    // The shared reduction is the launcher's: a second, independent loop over the reduction
    // rows, which needs its own range and runs after this sweep's threads have joined.
}


// The 2x2 field-block apply over CURVED macro elements: every micro cell builds its own
// geometry, and the macro element's is never built.
template <int Blocks, typename scalar_t, typename geom_t, typename idx_t>
inline SFEM_NOINLINE void sscvfem_apply_blocks_isoparam(
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
        const scalar_t bc_p,
        const scalar_t bc_tx,
        const scalar_t bc_ty,
        const scalar_t bc_tz,
        idx_t **const SFEM_RESTRICT elems,
        const int level,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const uint8_t *const SFEM_RESTRICT natural_mask,
        const uint8_t *const SFEM_RESTRICT pressure_mask,
        const uint8_t *const SFEM_RESTRICT traction_mask,
        const int nxe_src,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx_src,
        const scalar_t *const SFEM_RESTRICT pgy_src,
        const scalar_t *const SFEM_RESTRICT pgz_src,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT qgx_src,
        const scalar_t *const SFEM_RESTRICT qgy_src,
        const scalar_t *const SFEM_RESTRICT qgz_src,
        const scalar_t upwind_eps,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
        const Hex8RcConfigT<scalar_t> rcfg,
        
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const ptrdiff_t *const SFEM_RESTRICT red_idx,
        const ptrdiff_t *const SFEM_RESTRICT red_ptr,
        const idx_t *const SFEM_RESTRICT shared_node,
        const int *const SFEM_RESTRICT slot,
        scalar_t *const SFEM_RESTRICT stage,
        const ptrdiff_t n_shared, const scalar_t rho, const scalar_t mu,
                                                    const scalar_t *const SFEM_RESTRICT dir,
                                                    scalar_t *const SFEM_RESTRICT       jv) {
    constexpr bool up = (Blocks & SSBLOCK_UP) != 0;
    constexpr bool pp = (Blocks & SSBLOCK_PP) != 0;

    const int L   = level;
    const int nxe = nxe_src;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    // The direction's pressure gradient is read only where a pressure column is wanted, so a
    // momentum-row block skips this gather along with the rest of the pressure work.
    const bool           has_qg = (up || pp) && qgx_src;
    const SSMacroScratchT<scalar_t, idx_t> s = sscvfem_macro_scratch<scalar_t, idx_t>(nxe, CVFEM_HEX8_N_FIELDS, true, has_qg, false);

    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const ptrdiff_t e = macro_order ? macro_order[i] : i;
        sscvfem_blocks_gather<Blocks>(elems, points, pres, pgx_src, pgy_src, pgz_src, qgx_src,
                                      qgy_src, qgz_src, ux_src, uy_src, uz_src, dir, s, has_qg, e,
                                      nxe);

        // Per macro element, and the curved sweep reads the same one: a call per micro cell
        // there took this unit past the point where GCC inlines sscvfem_rc_config, which then
        // became a call in every cell of the block diagonal, 9% slower on boxes.
        const Hex8RcConfigT<scalar_t> rc_macro = rcfg;
        sscvfem_blocks_curved_macro<Blocks>(box_lx, box_ly, box_lz, bc_p, bc_tx, bc_ty, bc_tz,
                                            face_mask, natural_mask, pressure_mask, traction_mask,
                                            upwind_eps, rc_macro, rho, mu, s, has_qg, e, level, off);

        sscvfem_blocks_drain<Blocks>(s, slot, const_cast<scalar_t *>(stage), nxe, e, jv);
    }

    // The shared reduction is the launcher's: a second, independent loop over the reduction
    // rows, run once after the element pass's threads have joined.
}


// The block diagonal over CURVED macro elements: every micro cell with its own geometry, and no
// hoisted geometry built at all.
template <typename scalar_t, typename geom_t, typename idx_t>
inline SFEM_NOINLINE void sscvfem_block_diag_isoparam(
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
        const scalar_t bc_p,
        const scalar_t bc_tx,
        const scalar_t bc_ty,
        const scalar_t bc_tz,
        idx_t **const SFEM_RESTRICT elems,
        const int level,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const uint8_t *const SFEM_RESTRICT natural_mask,
        const uint8_t *const SFEM_RESTRICT pressure_mask,
        const uint8_t *const SFEM_RESTRICT traction_mask,
        const int nxe_src,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const Hex8RcConfigT<scalar_t> rcfg,
        
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const ptrdiff_t *const SFEM_RESTRICT red_idx,
        const ptrdiff_t *const SFEM_RESTRICT red_ptr,
        const idx_t *const SFEM_RESTRICT shared_node,
        const int *const SFEM_RESTRICT slot,
        scalar_t *const SFEM_RESTRICT stage16,
        const ptrdiff_t n_shared, const scalar_t rho, const scalar_t mu,
                                                   scalar_t *const SFEM_RESTRICT out) {
    const int L   = level;
    const int nxe = nxe_src;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    // Sixteen wide: this destination is a 4x4 block per node, not a residual.
    const SSMacroScratchT<scalar_t, idx_t> s = sscvfem_macro_scratch<scalar_t, idx_t>(nxe, 16, false, false, false);

    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const ptrdiff_t e = macro_order ? macro_order[i] : i;
        sscvfem_macro_gather(s, elems, points, pres, pgx, pgy, pgz, static_cast<const scalar_t *>(nullptr), static_cast<const scalar_t *>(nullptr), static_cast<const scalar_t *>(nullptr),
                             static_cast<const scalar_t *>(nullptr), ux, uy, uz, static_cast<const scalar_t *>(nullptr), e, nxe, 16);
        sscvfem_block_diag_curved_macro(box_lx, box_ly, box_lz, bc_p, bc_tx, bc_ty, bc_tz, level,
                                        face_mask, natural_mask, pressure_mask, traction_mask, rcfg,
                                        rho, mu, s, e, off);

        sscvfem_macro_drain_w<16>(s, slot, const_cast<scalar_t *>(stage16), nxe, e, out);
    }

    // The shared reduction is NOT here. It is a second, independent loop -- over the reduction
    // rows rather than the macro elements -- and DESIGN.md's threading rule applies to it as much
    // as to the element pass, so it needs its own range and its own entry point. The launcher
    // runs it after this sweep's threads have joined, which is also the barrier it needs: a row
    // sums staging slots that other macro elements wrote.
}


// The same volume sum over CURVED macro elements: every micro cell's own determinant, and no
// macro geometry built at all. See sscvfem_cell_volume_affine for what the two callers are and
// why `scale` is a parameter.
template <typename scalar_t, typename geom_t, typename idx_t>
static SFEM_NOINLINE void sscvfem_cell_volume_isoparam(
        // The staging object is gone; what this kernel reads out of it is what it takes.
        idx_t **const SFEM_RESTRICT elems,
        geom_t **const SFEM_RESTRICT points, const ptrdiff_t e, const int L, const int off[8],
        const scalar_t scale,
        scalar_t *const SFEM_RESTRICT dst) {
    for (int zi = 0; zi < L; ++zi)
        for (int yi = 0; yi < L; ++yi)
            for (int xi = 0; xi < L; ++xi) {
                const int base = sscvfem_lidx(L, xi, yi, zi);
                scalar_t  cx[8], cy[8], cz[8], cadj[9], cdet;
                sscvfem_cell_corners(elems, points, e, base, off, cx, cy, cz);
                sscvfem_micro_geom(cx, cy, cz, cadj, &cdet);
                const scalar_t vol = std::fabs(cdet);
                if (vol < scalar_t(1e-30)) continue;
                const scalar_t v = vol * scale;
                for (int a = 0; a < 8; ++a) atomic_add(dst, elems[base + off[a]][e], v);
            }
}

#endif  // CVFEM_SSHEX8_NS_ISOPARAM_HPP
