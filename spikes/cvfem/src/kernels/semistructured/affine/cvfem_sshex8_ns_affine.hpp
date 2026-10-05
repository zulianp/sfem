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


// The macro-local apply over STRAIGHT macro elements: one geometry hoisted over the micro
// cells, with nothing asking whether that is allowed.
inline SFEM_NOINLINE void sscvfem_apply_macro_local_affine(
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

        scalar_t hx[8], hy[8], hz[8];
        sscvfem_macro_hoisted_corners(s, L, hx, hy, hz);
        for (int zi = 0; zi < L; ++zi)
            for (int yi = 0; yi < L; ++yi)
                for (int xi = 0; xi < L; ++xi)
                    sscvfem_apply_macro_local_cell(box_lx, box_ly, box_lz, rcfg, rho, mu, upwind_eps, s,
                                                   L, xi, yi, zi, off, nullptr, scalar_t(0), hx, hy, hz);

        // Scatter once per macro node instead of once per element-node incidence.
        sscvfem_macro_drain(s, nullptr, nullptr, nxe, e, jv);
    }
}


// macro_local with the macro element's geometry LIFTED OUT of the micro-cell loop, over the
// straight macro elements.
//
// The flat kernel loads a precomputed adjugate and determinant per element; the two variants
// above rebuild the Jacobian from eight corners for every micro cell, so they were doing
// strictly more work than the kernel they are meant to beat. Inside an affine macro element
// every micro cell is a translate of the same box, so adj and det are invariant over the whole
// L^3 sweep and belong outside it. Worth 1.28x over sscvfem_apply_macro_local_affine, which is
// what keeping both variants is for.
//
// THERE IS NO ISOPARAMETRIC TWIN OF THIS SWEEP, and that is the point of it rather than a gap:
// lifting the geometry out of the micro-cell loop is precisely what a curved macro element
// cannot do. A trilinear macro element's Jacobian varies across its lattice, and hoisting one
// anyway does not merely lose accuracy -- neighbouring macro elements hoist different
// geometries, the sub-control surfaces of a node's control volume stop closing, and a uniform
// velocity acquires a discrete divergence (measured at 1.39 of the flux scale on the FDA
// nozzle, against 0.085 for a flat mesh). So the curved range runs
// sscvfem_apply_macro_local_isoparam, which is also the curved range of the variant above: as
// branches inside the two sweeps those two paths were bit-identical, and they are one sweep now.
//
// What this sweep used to carry and does not: a `curved_e` test in seven places, and an assert
// comparing the last micro cell's geometry against the hoisted value to catch a curved macro
// element reaching it. The range it is given cannot contain one.
inline SFEM_NOINLINE void sscvfem_apply_macro_lifted_affine(
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

        // Once per macro element: micro cell 0's corners hoisted, and the geometry built from
        // them -- the adjugate AND the coordinates the Rhie-Chow term differences.
        //
        // The lattice inside a macro element is uniform, so every micro cell is congruent to
        // cell 0 and one adjugate serves all of them. This used to hoist the adjugate but then
        // hand the Rhie-Chow struct each cell's OWN coordinates, and the two agree only to the
        // precision the node positions are stored in. geom_t is float32, so the block diagonal
        // disagreed with the action it is supposed to be the diagonal of by 4.23e-08 -- eight
        // orders above round-off, and invisible until the q-independent consistency gate looked.
        //
        // Only DIFFERENCES of these are taken (d = x_j - x_i), so cell 0's coordinates are exact
        // for the purpose, not an approximation.
        scalar_t madj[9], mdet;
        scalar_t c0x[8], c0y[8], c0z[8];
        sscvfem_macro_hoisted_corners(s, L, c0x, c0y, c0z);
        sscvfem_micro_geom(c0x, c0y, c0z, madj, &mdet);

        for (int zi = 0; zi < L; ++zi)
            for (int yi = 0; yi < L; ++yi)
                for (int xi = 0; xi < L; ++xi)
                    sscvfem_apply_macro_local_cell(box_lx, box_ly, box_lz, rcfg, rho, mu, upwind_eps, s,
                                                   L, xi, yi, zi, off, madj, mdet, c0x, c0y, c0z);

        sscvfem_macro_drain(s, nullptr, nullptr, nxe, e, jv);
    }
}


// The naive residual over STRAIGHT macro elements: the hoisted geometry, with nothing asking
// whether it applies.
inline SFEM_NOINLINE void sscvfem_residual_naive_affine(
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
                    sscvfem_residual_naive_cell(box_lx, box_ly, box_lz, peclet, elems, pres, pgx_src,
                                                pgy_src, pgz_src, points, upwind_eps, ux_src, uy_src,
                                                uz_src, rcfg, rho, mu, e, L, xi, yi, zi, off, hx, hy,
                                                hz, res);
    }
}


// The naive block diagonal over STRAIGHT macro elements: the hoisted geometry, with nothing
// asking whether it applies.
inline SFEM_NOINLINE void sscvfem_block_diag_naive_affine(
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
                    sscvfem_block_diag_naive_cell(box_lx, box_ly, box_lz, elems, pres, pgx_src,
                                                  pgy_src, pgz_src, points, ux_src, uy_src, uz_src,
                                                  rcfg, rho, mu, e, L, xi, yi, zi, off, hx, hy, hz,
                                                  out);
    }
}


// The nodal-gradient scatter sweep over STRAIGHT macro elements.
//
// The geometry is computed ONCE per macro element rather than once per micro cell. A macro
// element is subdivided uniformly, so its micro cells are translates of one another and share a
// Jacobian exactly -- which is what the `hoisted` in the apply sweeps means. This sweep did not
// always do that, and paid L^3 geometry evaluations per macro element where one is needed: eight
// times too many at level 2 and sixty-four at level 4.
inline void sscvfem_nodal_grad_scatter_affine(
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

        scalar_t adj[9], det;
        {
            scalar_t ex[8], ey[8], ez[8];
            for (int a = 0; a < 8; ++a) {
                const idx_t g = elems[off[a]][e];
                ex[a]                = (scalar_t)points[0][g];
                ey[a]                = (scalar_t)points[1][g];
                ez[a]                = (scalar_t)points[2][g];
            }
            sscvfem_micro_geom(ex, ey, ez, adj, &det);
        }
        if (std::fabs(det) < scalar_t(1e-30)) continue;
        const scalar_t sgn = det > 0 ? scalar_t(1) : scalar_t(-1);

        for (int zi = 0; zi < L; ++zi)
            for (int yi = 0; yi < L; ++yi)
                for (int xi = 0; xi < L; ++xi)
                    sscvfem_nodal_grad_cell(elems, points, lp, lg, slot, lacc, e, L, xi, yi, zi, off,
                                            adj, sgn, ogx, ogy, ogz);

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


// Mirrors cvfem_hex8_ns_upwind_jacobian_action with the macro element's invariants hoisted out
// of the micro-cell loop: the adjugate, the determinant, the sub-control-surface areas, the
// difference vectors and the Rhie-Chow coefficient are all built once per macro element. Over
// STRAIGHT macro elements, where that is exact.
inline SFEM_NOINLINE void sscvfem_apply_macro_hoisted_affine(
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
        const Hex8RcConfig rcfg,
        
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
    const SSMacroScratch s = sscvfem_macro_scratch(nxe, CVFEM_HEX8_N_FIELDS, true, has_qg, false);

    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const ptrdiff_t e = macro_order ? macro_order[i] : i;
        sscvfem_macro_gather(s, elems, points, pres, pgx_src, pgy_src, pgz_src, qgx_src, qgy_src,
                             qgz_src, nullptr, ux_src, uy_src, uz_src, dir, e, nxe,
                             CVFEM_HEX8_N_FIELDS);

        // Per macro element, and the curved sweep reads the same one: a call per micro cell
        // there took this unit past the point where GCC inlines sscvfem_rc_config, which then
        // became a call in every cell of the block diagonal, 9% slower on boxes.
        const Hex8RcConfig rc_macro = rcfg;
        SSMacroGeom mg;
        {
            scalar_t ex[8], ey[8], ez[8];
            sscvfem_macro_hoisted_corners(s, L, ex, ey, ez);
            sscvfem_macro_geom(ex, ey, ez, rho, mu, rc_macro.scale, rc_macro.tau, mg);
        }

        for (int zi = 0; zi < L; ++zi)
            for (int yi = 0; yi < L; ++yi)
                for (int xi = 0; xi < L; ++xi)
                    sscvfem_action_hoisted_cell(box_lx, box_ly, box_lz, bc_p, bc_tx, bc_ty, bc_tz,
                                                face_mask, natural_mask, pressure_mask, traction_mask,
                                                rho, mu, upwind_eps, s, mg, has_qg, e, L, xi, yi, zi,
                                                off);

        sscvfem_macro_drain(s, slot, const_cast<scalar_t *>(stage), nxe, e, jv);
    }

    // The shared reduction is the caller's: a second, independent loop over the reduction
    // rows, run after this sweep's threads have joined. See sscvfem_drain_shared.
}

#endif  // CVFEM_SSHEX8_NS_AFFINE_HPP
