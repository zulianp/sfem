#pragma once

// HEX8 CVFEM Navier-Stokes on a semi-structured (sshex8) mesh: THE SWEEPS.
//
// Split out of kernels/semistructured/cvfem_sshex8_ns.hpp, which held the sweeps and the staging side in one
// file. What is here takes nothing but the quantities it reads -- no SSMeshData, no SSScatter,
// no PackedData, no std::vector, no library type -- which is what DESIGN.md asks of this
// directory. Everything that resolves an option, owns an allocation, builds a cached table or
// decides which variant runs is in frontend/ss/cvfem_sshex8_ns.hpp, which includes this file.
//
// scalar_t, idx_t, count_t and geom_t are the including translation unit's, exactly as in every
// other header here.
//
// HEX8 CVFEM Navier-Stokes on a semi-structured (sshex8) mesh.
//
// This exists to answer one question: how much of the flat kernel's cost is the indexed
// gather? T2 measured the flat matrix-free action at 2.36 ns/dof on Grace while its
// compulsory traffic ran at 4% of memory peak, so it is limited by neither the data it
// must move nor arithmetic -- and every element re-reads its eight nodes through
// d.elems[a][e], so each node is fetched about eight times per sweep.
//
// A semi-structured mesh removes that by construction. Nodes within a macro-element are
// numbered lexicographically, lidx(L,x,y,z) = z(L+1)^2 + y(L+1) + x, so the eight corners
// of every micro-element sit at the SAME eight constant offsets from their base:
//
//     {0, 1, Lp1+1, Lp1, Lp1^2, Lp1^2+1, Lp1^2+Lp1+1, Lp1^2+Lp1}
//
// So a macro-element's (L+1)^3 nodes can be gathered once and its L^3 micro-elements read
// from contiguous local buffers with no indirection at all. Indexed loads per element
// fall from 8 to (L+1)^3/L^3 -- 1.95 at L=4, 1.42 at L=8, 1.20 at L=16 -- and the atomic
// scatter falls by the same factor, since a macro-element writes its nodes once instead
// of once per element-node incidence.
//
// Two variants are provided and they must agree to round-off. `naive` keeps the flat
// gather, reading every node through the global id, and exists only as the control:
// it is the same physics on the same mesh, differing from `macro_local` in the gather
// alone, so the difference between them is the transformation and nothing else.
//
// The element kernels are reused verbatim from cvfem_hex8_ns_core.hpp. Nothing about the
// physics is reimplemented here, which is what makes the comparison meaningful.

#include "kernels/microkernels/hex8/affine/cvfem_hex8_ns_upwind_affine.hpp"
#include "kernels/cvfem_hex8_flags.hpp"
#include "kernels/cvfem_phases.hpp"
#include "kernels/cvfem_portability.hpp"
#include "kernels/cvfem_range.hpp"
#include "kernels/cvfem_scatter.hpp"
#include "kernels/microkernels/hex8/cvfem_hex8_boundary_scs.hpp"
#include "kernels/microkernels/hex8/cvfem_hex8_ns_upwind_kernels.hpp"
#include "kernels/microkernels/limiters/cvfem_venkata_limiter.hpp"
#include "kernels/packed/cvfem_hex8_ns_packed.hpp"
#include "kernels/packed/cvfem_pack_scratch.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>


// Per-element scatter: exclusive nodes straight out, shared ones staged. Templated on the
// number of values per node so the same tables serve the 4-wide kernels (Jacobian action,
// residual, block split) and the 16-wide block diagonal.
template <int W>
static SFEM_INLINE void sscvfem_scatter_element_w(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const int *const SFEM_RESTRICT slot, const int nxe, const ptrdiff_t e,
                                                  const idx_t *const SFEM_RESTRICT lg,
                                                  const scalar_t *const SFEM_RESTRICT     lout,
                                                  scalar_t *const SFEM_RESTRICT           dst,
                                                  scalar_t *const SFEM_RESTRICT           stage) {
    for (int a = 0; a < nxe; ++a) {
        const int sl = slot[(size_t)e * nxe + a];
        if (sl < 0) {
            const ptrdiff_t g = (ptrdiff_t)lg[a] * W;
            for (int c = 0; c < W; ++c) dst[g + c] += lout[(size_t)a * W + c];
        } else {
            for (int c = 0; c < W; ++c) stage[(size_t)sl * W + c] = lout[(size_t)a * W + c];
        }
    }
}

template <int W>
inline void sscvfem_reduce_shared_w(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const ptrdiff_t *const SFEM_RESTRICT red_idx,
        const ptrdiff_t *const SFEM_RESTRICT red_ptr,
        const idx_t *const SFEM_RESTRICT shared_node,
        const ptrdiff_t n_shared, scalar_t *const SFEM_RESTRICT dst,
                                    const scalar_t *const SFEM_RESTRICT stage) {
    const ptrdiff_t nrows = n_shared;
#pragma omp parallel for schedule(static)
    for (ptrdiff_t r = 0; r < nrows; ++r) {
        scalar_t acc[W] = {0};
        for (ptrdiff_t k = red_ptr[(size_t)r]; k < red_ptr[(size_t)r + 1]; ++k)
            for (int c = 0; c < W; ++c) acc[c] += stage[(size_t)red_idx[(size_t)k] * W + c];
        const ptrdiff_t g = (ptrdiff_t)shared_node[(size_t)r] * W;
        for (int c = 0; c < W; ++c) dst[g + c] += acc[c];
    }
}

static SFEM_INLINE void sscvfem_scatter_element(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const int *const SFEM_RESTRICT slot,
        scalar_t *const SFEM_RESTRICT stage, const int nxe, const ptrdiff_t e,
                                                const idx_t *const SFEM_RESTRICT lg,
                                                const scalar_t *const SFEM_RESTRICT     lout,
                                                scalar_t *const SFEM_RESTRICT           jv) {
    sscvfem_scatter_element_w<CVFEM_HEX8_N_FIELDS>(slot, nxe, e, lg, lout, jv, const_cast<scalar_t *>(stage));
}

// The same, for four separate destination arrays rather than one interleaved one. The
// nodal pressure gradient accumulates pgx, pgy, pgz and a volume weight, and it fed the
// apply, so leaving it atomic left the whole operator non-reproducible even after the
// Jacobian action's own scatter was fixed.
//
// Templated on the width because not every user wants four. The nodal gradient wants three --
// it stopped carrying a volume weight once that became cached geometry -- and a third of the
// staging, of the scatter and of the shared reduction is a third of each pass's memory
// traffic. The stage is allocated at the widest width any user needs, so a narrower pass
// simply addresses less of it; write and read must agree, which is why the width is a
// template parameter and not an argument that could differ between the two calls.
template <int W>
static SFEM_INLINE void sscvfem_scatter_element_soa_w(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const int *const SFEM_RESTRICT slot,
        scalar_t *const SFEM_RESTRICT stage, const int nxe, const ptrdiff_t e,
                                                      const idx_t *const SFEM_RESTRICT lg,
                                                      const scalar_t *const SFEM_RESTRICT     lacc,
                                                      scalar_t *const                        dst[W]) {
    for (int a = 0; a < nxe; ++a) {
        const int sl = slot[(size_t)e * nxe + a];
        if (sl < 0) {
            const idx_t g = lg[a];
            for (int c = 0; c < W; ++c) dst[c][g] += lacc[(size_t)a * W + c];
        } else {
            for (int c = 0; c < W; ++c) stage[(size_t)sl * W + c] = lacc[(size_t)a * W + c];
        }
    }
}

template <int W>
inline void sscvfem_reduce_shared_soa_w(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const ptrdiff_t *const SFEM_RESTRICT red_idx,
        const ptrdiff_t *const SFEM_RESTRICT red_ptr,
        const idx_t *const SFEM_RESTRICT shared_node,
        scalar_t *const SFEM_RESTRICT stage,
        const ptrdiff_t n_shared, scalar_t *const dst[W]) {
    const ptrdiff_t nrows = n_shared;
#pragma omp parallel for schedule(static)
    for (ptrdiff_t r = 0; r < nrows; ++r) {
        scalar_t acc[W] = {0};
        for (ptrdiff_t k = red_ptr[(size_t)r]; k < red_ptr[(size_t)r + 1]; ++k)
            for (int c = 0; c < W; ++c) acc[c] += stage[(size_t)red_idx[(size_t)k] * W + c];
        const idx_t g = shared_node[(size_t)r];
        for (int c = 0; c < W; ++c) dst[c][g] += acc[c];
    }
}



// Second pass: each shared node gathers its own contributions, in slot order.
inline void sscvfem_reduce_shared(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const ptrdiff_t *const SFEM_RESTRICT red_idx,
        const ptrdiff_t *const SFEM_RESTRICT red_ptr,
        const idx_t *const SFEM_RESTRICT shared_node,
        scalar_t *const SFEM_RESTRICT stage,
        const ptrdiff_t n_shared, scalar_t *const SFEM_RESTRICT jv) {
    sscvfem_reduce_shared_w<CVFEM_HEX8_N_FIELDS>(red_idx, red_ptr, shared_node, n_shared, jv, stage);
}

static SFEM_INLINE int sscvfem_lidx(const int L, const int x, const int y, const int z) {
    const int Lp1 = L + 1;
    return z * (Lp1 * Lp1) + y * Lp1 + x;
}

// The eight corner offsets, constant for every micro-element in the macro-element.
static SFEM_INLINE void sscvfem_corner_offsets(const int L, int off[8]) {
    const int Lp1 = L + 1;
    off[0]        = 0;
    off[1]        = 1;
    off[2]        = Lp1 + 1;
    off[3]        = Lp1;
    off[4]        = Lp1 * Lp1;
    off[5]        = Lp1 * Lp1 + 1;
    off[6]        = Lp1 * Lp1 + Lp1 + 1;
    off[7]        = Lp1 * Lp1 + Lp1;
}

// Geometry of one micro-element from its eight corners. The macro-elements here come from
// a box mesh, so each micro-element is affine and the adjugate is constant; evaluating at
// the centre is therefore exact rather than an approximation.
static SFEM_INLINE void sscvfem_micro_geom(const scalar_t x[8], const scalar_t y[8], const scalar_t z[8],
                                           scalar_t adj[9], scalar_t *det) {
    cvfem_hex8_geom_at(x, y, z, scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), adj, det);
}

// The macro element's own eight corners: CVFEM local corner a sits at the lattice extreme
// (xi_a L, eta_a L, zeta_a L). sscvfem_corner_offsets gives the corners of micro cell 0.
static SFEM_INLINE void sscvfem_macro_corner_offsets(const int L, int ext[8]) {
    for (int a = 0; a < 8; ++a)
        ext[a] = sscvfem_lidx(L, (int)CVFEM_HEX8_REF_XI[a][0] * L, (int)CVFEM_HEX8_REF_XI[a][1] * L,
                              (int)CVFEM_HEX8_REF_XI[a][2] * L);
}

// The micro cell every hoisted geometry is computed from, given the macro element's corners:
// the affine cell of reference size 1/L centred on the macro element, with the macro
// element's Jacobian at its centre. Its adjugate and determinant are the macro centre's
// scaled to one micro cell, and its corner DIFFERENCES are what the Rhie-Chow term takes.
//
// For an affine macro element this is a translate of micro cell 0, so nothing changes but
// round-off. For a curved one -- a warped mesh such as the FDA nozzle -- micro cell 0 is a
// corner cell and its Jacobian is off by the macro element's curvature, first order in the
// macro size; the centre Jacobian is the midpoint approximation of the whole macro element.
// Neither improves with L, only with smaller macro elements. Reads every corner before
// writing any, so it may be called in place.
static SFEM_INLINE void sscvfem_hoisted_cell(const scalar_t mx[8], const scalar_t my[8], const scalar_t mz[8],
                                             const int L, scalar_t cx[8], scalar_t cy[8], scalar_t cz[8]) {
    scalar_t c[3] = {0, 0, 0};
    scalar_t J[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};  // J[i][k] = dx_i / dxi_k at the centre
    for (int a = 0; a < 8; ++a) {
        const scalar_t m[3] = {mx[a], my[a], mz[a]};
        for (int i = 0; i < 3; ++i) {
            c[i] += scalar_t(0.125) * m[i];
            for (int k = 0; k < 3; ++k)
                J[i][k] += scalar_t(0.25) * (CVFEM_HEX8_REF_XI[a][k] > 0.5 ? m[i] : -m[i]);
        }
    }
    const scalar_t hL = scalar_t(1) / scalar_t(L);
    for (int a = 0; a < 8; ++a) {
        scalar_t o[3] = {c[0], c[1], c[2]};
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k) o[i] += J[i][k] * (scalar_t(CVFEM_HEX8_REF_XI[a][k]) - scalar_t(0.5)) * hL;
        cx[a] = o[0];
        cy[a] = o[1];
        cz[a] = o[2];
    }
}

// Whether macro element e is curved, so that hoisting one geometry over its micro cells would
// be wrong in a way refinement does not fix.
//
// Hoisting is exact for an affine macro element, whose micro cells are translates of one
// another. For a curved one it is not merely inaccurate: neighbouring macro elements hoist
// different geometries, so the sub-control surfaces a node's control volume is assembled from
// no longer close, and a uniform velocity has a discrete divergence. Measured on the FDA
// nozzle as the continuity row of u = (1,0,0), p = 0, relative to the flux scale: 1.39 at
// macro core 2 / L 2 and 1.49 at L 4 -- not falling with the level -- against 0.085 for the
// flat mesh. That spurious source drove a backward flow fifty times the physical velocity and
// stalled Newton with an exact Jacobian and a dense LU. A curved macro element therefore gives
// each micro cell its own geometry, as the flat operator gives each element its own.
static SFEM_INLINE bool sscvfem_macro_curved(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const uint8_t *const SFEM_RESTRICT macro_curved, const ptrdiff_t e) {
    return macro_curved && macro_curved[(size_t)e];
}

// The eight corners of the micro cell at lattice index `base` of macro element e.
static SFEM_INLINE void sscvfem_cell_corners(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        idx_t **const SFEM_RESTRICT elems,
        geom_t **const SFEM_RESTRICT points, const ptrdiff_t e, const int base,
                                             const int off[8], scalar_t x[8], scalar_t y[8], scalar_t z[8]) {
    for (int a = 0; a < 8; ++a) {
        const idx_t gn = elems[base + off[a]][e];
        x[a]                  = (scalar_t)points[0][gn];
        y[a]                  = (scalar_t)points[1][gn];
        z[a]                  = (scalar_t)points[2][gn];
    }
}

// ---------------------------------------------------------------------------
// Nodal pressure gradient, the pre-pass Rhie-Chow interpolation needs. Mirrors
// assemble_nodal_p_grad: a volume-weighted average of the element gradients.

// The reconstruction over an arbitrary strided nodal scalar. It is linear in that scalar
// with geometry-only weights, so applying it to a Jacobian direction q gives exactly the
// derivative of applying it to p -- the term the Rhie-Chow Jacobian was missing.
// Micro-element boundary-face mask, derived from the macro element's mask and the lattice
// position. A micro element carries a macro face only where it sits against that face of the
// lattice, so this is six tests and no storage.
//
// Level-independent by construction, which is why one macro-level mask serves every level of
// the multigrid hierarchy: nothing has to be rebuilt or transferred when a level is derefined.
// A negative macro mask means "no mask", and the coordinate test is used instead.
static SFEM_INLINE int sscvfem_micro_face_mask(const int macro, const int L, const int xi,
                                               const int yi, const int zi) {
    if (macro < 0) return -1;
    int m = 0;
    if (xi == 0)     m |= macro & 0x01;  // CVFEM face 0, x-min
    if (xi == L - 1) m |= macro & 0x02;  // face 1, x-max
    if (yi == 0)     m |= macro & 0x04;  // face 2, y-min
    if (yi == L - 1) m |= macro & 0x08;  // face 3, y-max
    if (zi == 0)     m |= macro & 0x10;  // face 4, z-min
    if (zi == L - 1) m |= macro & 0x20;  // face 5, z-max
    return m;
}

// The boundary data for one micro cell, with its face selectors projected down from the
// macro element exactly as the face and natural masks are.
//
// The values are per-sideset constants and need no projection; only the masks that say
// WHICH faces carry them do. Declared after sscvfem_micro_face_mask because it uses it, and
// after Hex8BoundaryDataT, which the boundary header defines.
static SFEM_INLINE Hex8BoundaryDataT<scalar_t> sscvfem_bd(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const scalar_t bc_p,
        const scalar_t bc_tx,
        const scalar_t bc_ty,
        const scalar_t bc_tz,
        const uint8_t *const SFEM_RESTRICT pressure_mask,
        const uint8_t *const SFEM_RESTRICT traction_mask, const ptrdiff_t e,
                                                          const int L, const int xi, const int yi,
                                                          const int zi) {
    Hex8BoundaryDataT<scalar_t> bd;
    bd.tx    = bc_tx;
    bd.ty    = bc_ty;
    bd.tz    = bc_tz;
    bd.p_bar = bc_p;
    bd.tmask = !traction_mask
                       ? 0
                       : sscvfem_micro_face_mask((int)traction_mask[(size_t)e], L, xi, yi, zi);
    bd.pmask = !pressure_mask
                       ? 0
                       : sscvfem_micro_face_mask((int)pressure_mask[(size_t)e], L, xi, yi, zi);
    return bd;
}

// The reconstruction over PACKED macro-elements.
//
// Same operator as the SSScatter path and the same summation structure as the flat packed
// gradient, which this mirrors deliberately: gather the field into a pack-local buffer,
// accumulate there with a plain `+=`, write the pack's owned rows straight out because no
// other pack owns them, and close the rest with the ghost reduction.
//
// What packing buys here is not the writes but WHAT COUNTS AS SHARED. The SSScatter path
// stages every macro-element face, edge and corner, because a node between two macro-elements
// cannot be written by either alone. Group those macro-elements into a pack and the node
// between two of them inside the pack is owned by the pack -- only the pack's outer boundary
// is left. At level 8 that is the difference between staging 53% of the nodes and 12%.
//
// The denominator is folded into the owned writes and into the ghost reduction rather than
// applied in a pass of its own, exactly as the flat twin does it, because the average is
// linear in its numerator: (owned + ghost) * w is owned * w + ghost * w. That removes a full
// read-modify-write over three nodal arrays from every call.
// apply_weight=false leaves the reconstruction UNNORMALISED, for the caller to divide once at
// the end. That is what lets a second sweep add to this one: the weight is per node, so
// folding it in here would normalise this pass's contribution and then the remainder's
// contribution would be normalised a second time when the caller finished the job.
//
// The default folds it into the drain as before -- one pass over the nodes saved, which is
// the whole reason the packed path beats the scatter one -- and multiplying by an exact 1
// otherwise, so the default path is unchanged bit for bit rather than merely equivalent.
inline void sscvfem_nodal_grad_packed_sweep(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const scalar_t *const SFEM_RESTRICT grad_w_inv,
        const int level,
        const uint8_t *const SFEM_RESTRICT macro_curved,
        const ptrdiff_t nmacro,
        geom_t **const SFEM_RESTRICT points, 
        // The staging object is gone; what this sweep reads out of it is what it takes.
        pack_idx_t **const SFEM_RESTRICT pack_elems,
        scalar_t *const SFEM_RESTRICT ghost_buf,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t max_actual_nodes_per_pack,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t n_packed_elements,
        const ptrdiff_t n_packs,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
                                      const scalar_t *const SFEM_RESTRICT src, const int stride,
                                      scalar_t *const SFEM_RESTRICT gx_out,
                                      scalar_t *const SFEM_RESTRICT gy_out,
                                      scalar_t *const SFEM_RESTRICT gz_out,
                                      const bool apply_weight) {
    const scalar_t *const SFEM_RESTRICT w      = grad_w_inv;
    const ptrdiff_t node_n = max_actual_nodes_per_pack > 0 ? max_actual_nodes_per_pack : 1;

    const int L = level;
    int       off[8];
    sscvfem_corner_offsets(L, off);
    const auto *const px = points[0];
    const auto *const py = points[1];
    const auto *const pz = points[2];

#pragma omp parallel
    {
        // Slots 7 and 8, which belong to this routine; see CVFEM_PACK_SCRATCH_SLOTS.
        scalar_t *const SFEM_RESTRICT pack_f   = thread_scratch<scalar_t>(7, (size_t)node_n);
        scalar_t *const SFEM_RESTRICT pack_out = thread_scratch<scalar_t>(8, 3 * (size_t)node_n);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start      = pack * n_elements_per_pack;
            // Bounded by the PACKED element count, not by the block's.
            //
            // pack_elems is sized to the elements the packs cover. Those used to be all of them,
            // so bounding by nmacro was harmless; on a distributed mesh the packs span only
            // the owned-not-shared prefix, and the last pack then walks this array past its
            // allocation. The fallback keeps the old bound for any caller that has not filled
            // the field in.
            const ptrdiff_t e_limit      = n_packed_elements > 0 ? n_packed_elements : nmacro;
            const ptrdiff_t e_end        = MIN(e_limit, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t owned        = owned_nodes_ptr[pack];
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned;
            const ptrdiff_t n_ghost      = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts    = &ghost_idx[ghost_ptr[pack]];
            const ptrdiff_t                         ghost_off = ghost_ptr[pack];

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) pack_f[k] = src[(owned + k) * stride];
            for (ptrdiff_t k = 0; k < n_ghost; ++k)
                pack_f[n_contiguous + k] = src[(ptrdiff_t)ghosts[k] * stride];
            std::memset(pack_out, 0, (size_t)n_pack_nodes * 3 * sizeof(scalar_t));

            for (ptrdiff_t e = e_start; e < e_end; ++e) {
                // One geometry per macro-element: its micro-elements are translates and share
                // a Jacobian exactly, which is what sscvfem_macro_geom has always relied on.
                scalar_t ex[8], ey[8], ez[8], adj[9], det;
                for (int a = 0; a < 8; ++a) {
                    const idx_t g = cvfem_pack_local_to_global(owned, ghosts, n_contiguous, pack_elems[off[a]][e]);
                    ex[a]                = (scalar_t)px[g];
                    ey[a]                = (scalar_t)py[g];
                    ez[a]                = (scalar_t)pz[g];
                }
                sscvfem_micro_geom(ex, ey, ez, adj, &det);
                const bool curved_e = sscvfem_macro_curved(macro_curved, e);
                if (!curved_e && std::fabs(det) < scalar_t(1e-30)) continue;
                // |det| times a gradient carrying 1/det: only the sign survives.
                const scalar_t sgn = det > 0 ? scalar_t(1) : scalar_t(-1);

                for (int zi = 0; zi < L; ++zi) {
                    for (int yi = 0; yi < L; ++yi) {
                        for (int xi = 0; xi < L; ++xi) {
                            const int base = sscvfem_lidx(L, xi, yi, zi);
                            scalar_t  ep[8], gx, gy, gz;
                            for (int a = 0; a < 8; ++a) ep[a] = pack_f[pack_elems[base + off[a]][e]];
                            if (curved_e) {
                                scalar_t cx[8], cy[8], cz[8], cadj[9], cdet;
                                for (int a = 0; a < 8; ++a) {
                                    const idx_t gn = cvfem_pack_local_to_global(
                                            owned, ghosts, n_contiguous, pack_elems[base + off[a]][e]);
                                    cx[a] = (scalar_t)px[gn];
                                    cy[a] = (scalar_t)py[gn];
                                    cz[a] = (scalar_t)pz[gn];
                                }
                                sscvfem_micro_geom(cx, cy, cz, cadj, &cdet);
                                if (std::fabs(cdet) < scalar_t(1e-30)) continue;
                                cvfem_hex8_grad_scalar(cadj, cdet > 0 ? scalar_t(1) : scalar_t(-1), ep, gx, gy, gz);
                            } else {
                                cvfem_hex8_grad_scalar(adj, sgn, ep, gx, gy, gz);
                            }
                            for (int a = 0; a < 8; ++a) {
                                scalar_t *const SFEM_RESTRICT o =
                                        pack_out + (ptrdiff_t)pack_elems[base + off[a]][e] * 3;
                                o[0] += gx;
                                o[1] += gy;
                                o[2] += gz;
                            }
                        }
                    }
                }
            }

            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                const scalar_t wi = apply_weight ? w[owned + k] : scalar_t(1);
                gx_out[owned + k] = pack_out[k * 3 + 0] * wi;
                gy_out[owned + k] = pack_out[k * 3 + 1] * wi;
                gz_out[owned + k] = pack_out[k * 3 + 2] * wi;
            }
            scalar_t *const SFEM_RESTRICT bx = ghost_buf + 0 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT by = ghost_buf + 1 * n_ghost_entries;
            scalar_t *const SFEM_RESTRICT bz = ghost_buf + 2 * n_ghost_entries;
            for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                const scalar_t *const SFEM_RESTRICT o = pack_out + (n_contiguous + k) * 3;
                bx[ghost_off + k]                     = o[0];
                by[ghost_off + k]                     = o[1];
                bz[ghost_off + k]                     = o[2];
            }
        }
    }

    // The packed layout's ghost reduction, which is cvfem_hex8_ghost_reduce_soa_range at width
    // three with the reconstruction's weight folded in. This was a copy of that loop; the one in
    // kernels/packed/ is now templated on the width and on whether it scales, so both widths
    // compile from one body.
    {
        scalar_t *const g3[3] = {gx_out, gy_out, gz_out};
#pragma omp parallel
        cvfem_hex8_ghost_reduce_soa_range<3, /*SCALED=*/true>(
                cvfem_range_split(0, n_ghost_reduce_rows, 1, cvfem_thread_index(),
                                  cvfem_n_threads()),
                ghost_reduce_dest, ghost_reduce_ptr, ghost_reduce_idx, n_ghost_entries,
                ghost_buf, apply_weight ? w : nullptr, g3);
    }
}

// Defined below, declared here because sscvfem_nodal_grad_strided calls it.
inline void sscvfem_nodal_grad_scatter_range(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        idx_t **const SFEM_RESTRICT elems,
        const int level,
        const uint8_t *const SFEM_RESTRICT macro_curved,
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
                                             const ptrdiff_t e_begin, const ptrdiff_t e_end);

// The scatter sweep over one element range, accumulating RAW into ogx/ogy/ogz.
//
// The caller zeroes the outputs and normalises afterwards, because this is now used twice:
// once for the whole mesh, and once for the elements the packs do not cover. Everything it
// writes is additive -- exclusive nodes go straight out with +=, shared ones through the
// staged reduce, which also accumulates -- so a partial range adds to whatever is already
// there instead of replacing it.
inline void sscvfem_nodal_grad_scatter_range(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        idx_t **const SFEM_RESTRICT elems,
        const int level,
        const uint8_t *const SFEM_RESTRICT macro_curved,
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
                                             const ptrdiff_t e_begin, const ptrdiff_t e_end) {
    if (e_begin >= e_end) return;

    const int L = level;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    // Three fields where there were four. The weight used to ride along through the
    // per-element accumulator, the scatter and the shared reduction, which is a third of the
    // traffic of each spent re-deriving a quantity that does not change.
    static constexpr int NG = 3;

    // A partial sweep writes only the staging slots of the elements it visits, and s.stage is
    // zeroed once when the scatter is BUILT rather than per call. So the slots belonging to
    // elements the packed pass handled would still hold values from the previous call, and the
    // reduce below would add them in -- silently, and looking like a convergence problem rather
    // than a stale read. Zero them here.
    //
    // Skipped for a full range, where every slot is written before it is read and this would be
    // pure cost on the path that runs in serial.
    if (slot && e_begin > 0 && n_slots > 0) {
        std::fill(stage, stage + (size_t)n_slots * NG, scalar_t(0));
    }

#pragma omp parallel
    {
        // Per-thread scratch from the kernels' own arena, not std::vector locals: slots 5/6 are the semi-structured element sweeps',
        // shared between them because only one is live inside a parallel region and
        // they all want the same macro-element size.
        scalar_t *const SFEM_RESTRICT _arena5 = thread_scratch<scalar_t>(5, ((size_t)nxe) + ((size_t)nxe * NG));
        scalar_t *const SFEM_RESTRICT lp = _arena5;
        scalar_t *const SFEM_RESTRICT lacc = _arena5 + ((size_t)nxe);
        idx_t *const SFEM_RESTRICT _arena6 = thread_scratch<idx_t>(6, ((size_t)nxe));
        idx_t *const SFEM_RESTRICT lg = _arena6;

#pragma omp for schedule(static)
        for (ptrdiff_t e = e_begin; e < e_end; ++e) {
            if (slot) std::fill(lacc, lacc + ((size_t)nxe * NG), scalar_t(0));
            // Only the field. The coordinates used to be gathered for every node of the
            // macro-element -- three arrays of (L+1)^3 -- to feed a geometry computation that
            // is the same for all of them.
            for (int a = 0; a < nxe; ++a) {
                const idx_t g = elems[a][e];
                lg[(size_t)a]        = g;
                lp[(size_t)a]        = src[(ptrdiff_t)g * stride];
            }

            // The geometry, once per macro-element rather than once per micro-element.
            //
            // A macro-element is subdivided uniformly, so its micro-elements are translates of
            // one another and share a Jacobian exactly. sscvfem_macro_geom has always relied
            // on this -- it is what the `hoisted` in apply_macro_local_hoisted means, and it
            // computes the geometry of the first micro-element and reuses it for all L^3.
            // This sweep did not, and paid L^3 geometry evaluations per macro-element where
            // one is needed: eight times too many at level 2 and sixty-four at level 4.
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
            const bool curved_e = sscvfem_macro_curved(macro_curved, e);
            if (!curved_e && std::fabs(det) < scalar_t(1e-30)) continue;
            // |det| * grad, where grad itself carries a 1/det. The determinant cancels and
            // only its SIGN survives, so the division the gradient used to do and the
            // multiplication that undid it both disappear.
            const scalar_t sgn = det > 0 ? scalar_t(1) : scalar_t(-1);

            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);
                        scalar_t  ep[8];
                        for (int a = 0; a < 8; ++a) ep[a] = lp[(size_t)(base + off[a])];
                        scalar_t gx, gy, gz;
                        if (curved_e) {
                            scalar_t cx[8], cy[8], cz[8], cadj[9], cdet;
                            sscvfem_cell_corners(elems, points, e, base, off, cx, cy, cz);
                            sscvfem_micro_geom(cx, cy, cz, cadj, &cdet);
                            if (std::fabs(cdet) < scalar_t(1e-30)) continue;
                            cvfem_hex8_grad_scalar(cadj, cdet > 0 ? scalar_t(1) : scalar_t(-1), ep, gx, gy, gz);
                        } else {
                            cvfem_hex8_grad_scalar(adj, sgn, ep, gx, gy, gz);
                        }
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            if (slot) {
                                scalar_t *const acc = lacc + (size_t)l * NG;
                                acc[0] += gx;
                                acc[1] += gy;
                                acc[2] += gz;
                            } else {
                                const idx_t id = lg[(size_t)l];
                                atomic_add(ogx, id, gx);
                                atomic_add(ogy, id, gy);
                                atomic_add(ogz, id, gz);
                            }
                        }
                    }
                }
            }

            if (slot) {
                scalar_t *dst[NG] = {ogx, ogy, ogz};
                sscvfem_scatter_element_soa_w<NG>(slot, const_cast<scalar_t *>(stage), nxe, e, lg, lacc, dst);
            }
        }
    }

    if (slot) {
        scalar_t *dst[NG] = {ogx, ogy, ogz};
        sscvfem_reduce_shared_soa_w<NG>(red_idx, red_ptr, shared_node, const_cast<scalar_t *>(stage), n_shared, dst);
    }

    // No normalisation here: the caller divides once, after whichever passes it ran. See
    // sscvfem_nodal_grad_normalize.
}


// ---------------------------------------------------------------------------
// Control: the flat gather, on the semi-structured mesh. Every micro-element reads its
// eight nodes through the global id, exactly as the flat kernel does.

inline SFEM_NOINLINE void sscvfem_apply_naive(
        // The range this call is to cover. DESIGN.md: the threading is abstract outside the
        // sweep and what arrives is a range, so the sweep owns no parallel region.
        const cvfem_range r,
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const scalar_t box_lx,
        const scalar_t box_ly,
        const scalar_t box_lz,
        idx_t **const SFEM_RESTRICT elems,
        const int level,
        const uint8_t *const SFEM_RESTRICT macro_curved,
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
    CVFEM_TRACE_SCOPE("sscvfem::apply_naive");
    const int L = level;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    for (ptrdiff_t e = r.begin; e < r.end; ++e) {
        // The geometry every micro cell of this macro element uses, as the hoisted
        // variants use it; see sscvfem_hoisted_cell. Real positions stay per cell.
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
        const bool curved_e = sscvfem_macro_curved(macro_curved, e);
        for (int zi = 0; zi < L; ++zi) {
            for (int yi = 0; yi < L; ++yi) {
                for (int xi = 0; xi < L; ++xi) {
                    const int base = sscvfem_lidx(L, xi, yi, zi);

                    idx_t g[8];
                    scalar_t     x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
                    scalar_t     vx[8], vy[8], vz[8], q[8], pgx[8], pgy[8], pgz[8];
                    scalar_t     r[CVFEM_HEX8_N_DOF];
                    for (int a = 0; a < 8; ++a) {
                        g[a]   = elems[base + off[a]][e];
                        x[a]   = (scalar_t)points[0][g[a]];
                        y[a]   = (scalar_t)points[1][g[a]];
                        z[a]   = (scalar_t)points[2][g[a]];
                        ux[a]  = ux_src[(size_t)g[a]];
                        uy[a]  = uy_src[(size_t)g[a]];
                        uz[a]  = uz_src[(size_t)g[a]];
                        p[a]   = pres[(size_t)g[a]];
                        vx[a]  = dir[(size_t)g[a] * 4 + 0];
                        vy[a]  = dir[(size_t)g[a] * 4 + 1];
                        vz[a]  = dir[(size_t)g[a] * 4 + 2];
                        q[a]   = dir[(size_t)g[a] * 4 + 3];
                        pgx[a] = pgx_src[(size_t)g[a]];
                        pgy[a] = pgy_src[(size_t)g[a]];
                        pgz[a] = pgz_src[(size_t)g[a]];
                    }
                    // A curved macro element: this cell's own geometry, not the hoisted one, selected
                    // through pointers so the hoisted corners stay loop-invariant.
                    const scalar_t *const gx = curved_e ? x : hx;
                    const scalar_t *const gy = curved_e ? y : hy;
                    const scalar_t *const gz = curved_e ? z : hz;


                    const Hex8RhieChow rc{gx,      gy, gz,   pgx, pgy, pgz, rcfg.scale, nullptr, nullptr,
                                          nullptr, ux, uy, uz,  rcfg.tau};
                    scalar_t           adj[9], det;
                    sscvfem_micro_geom(gx, gy, gz, adj, &det);
                    cvfem_hex8_ns_upwind_jacobian_action<0>(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r,
                                                        rc, p, upwind_eps);
                    boundary_scs_add_jacobian_action<false>(rho, mu, adj, det, box_lx, box_ly, box_lz, x, y, z, ux, uy, uz,
                                                     vx, vy, vz, q, r);

                    for (int a = 0; a < 8; ++a)
                        for (int c = 0; c < CVFEM_HEX8_N_FIELDS; ++c)
                            atomic_add(jv + (ptrdiff_t)g[a] * CVFEM_HEX8_N_FIELDS + c, 0, r[a * 4 + c]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The transformation: gather the macro-element's nodes once, run its L^3 micro-elements
// against constant offsets into contiguous local buffers, scatter once at the end.

inline SFEM_NOINLINE void sscvfem_apply_macro_local(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const scalar_t box_lx,
        const scalar_t box_ly,
        const scalar_t box_lz,
        idx_t **const SFEM_RESTRICT elems,
        const int level,
        const uint8_t *const SFEM_RESTRICT macro_curved,
        const ptrdiff_t nmacro,
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
    CVFEM_TRACE_SCOPE("sscvfem::apply_macro_local");
    const int L   = level;
    const int nxe = nxe_src;
    int       off[8];
    sscvfem_corner_offsets(L, off);

#pragma omp parallel
    {
        // One allocation per thread for the whole sweep, not per macro-element.
        // Per-thread scratch from the kernels' own arena, not std::vector locals: slots 5/6 are the semi-structured element sweeps',
        // shared between them because only one is live inside a parallel region and
        // they all want the same macro-element size.
        scalar_t *const SFEM_RESTRICT _arena5 = thread_scratch<scalar_t>(5, ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe * CVFEM_HEX8_N_FIELDS));
        scalar_t *const SFEM_RESTRICT lx = _arena5;
        scalar_t *const SFEM_RESTRICT ly = _arena5 + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lz = _arena5 + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lux = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT luy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT luz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lp = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lvx = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lvy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lvz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lq = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgx = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lout = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        idx_t *const SFEM_RESTRICT _arena6 = thread_scratch<idx_t>(6, ((size_t)nxe));
        idx_t *const SFEM_RESTRICT lg = _arena6;

#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < nmacro; ++e) {
            // Gather once. This is the only indirection in the sweep.
            for (int a = 0; a < nxe; ++a) {
                const idx_t g = elems[a][e];
                lg[(size_t)a]        = g;
                lx[(size_t)a]        = (scalar_t)points[0][g];
                ly[(size_t)a]        = (scalar_t)points[1][g];
                lz[(size_t)a]        = (scalar_t)points[2][g];
                lux[(size_t)a]       = ux_src[(size_t)g];
                luy[(size_t)a]       = uy_src[(size_t)g];
                luz[(size_t)a]       = uz_src[(size_t)g];
                lp[(size_t)a]        = pres[(size_t)g];
                lvx[(size_t)a]       = dir[(size_t)g * 4 + 0];
                lvy[(size_t)a]       = dir[(size_t)g * 4 + 1];
                lvz[(size_t)a]       = dir[(size_t)g * 4 + 2];
                lq[(size_t)a]        = dir[(size_t)g * 4 + 3];
                lpgx[(size_t)a]      = pgx_src[(size_t)g];
                lpgy[(size_t)a]      = pgy_src[(size_t)g];
                lpgz[(size_t)a]      = pgz_src[(size_t)g];
            }
            std::fill(lout, lout + ((size_t)nxe * CVFEM_HEX8_N_FIELDS), scalar_t(0));

            // The geometry every micro cell of this macro element uses, as the hoisted
            // variants use it; see sscvfem_hoisted_cell. Real positions stay per cell.
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
            const bool curved_e = sscvfem_macro_curved(macro_curved, e);
            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);

                        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
                        scalar_t vx[8], vy[8], vz[8], q[8], pgx[8], pgy[8], pgz[8];
                        scalar_t r[CVFEM_HEX8_N_DOF];
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];  // no indirection
                            x[a]        = lx[(size_t)l];
                            y[a]        = ly[(size_t)l];
                            z[a]        = lz[(size_t)l];
                            ux[a]       = lux[(size_t)l];
                            uy[a]       = luy[(size_t)l];
                            uz[a]       = luz[(size_t)l];
                            p[a]        = lp[(size_t)l];
                            vx[a]       = lvx[(size_t)l];
                            vy[a]       = lvy[(size_t)l];
                            vz[a]       = lvz[(size_t)l];
                            q[a]        = lq[(size_t)l];
                            pgx[a]      = lpgx[(size_t)l];
                            pgy[a]      = lpgy[(size_t)l];
                            pgz[a]      = lpgz[(size_t)l];
                        }
                        // A curved macro element: this cell's own geometry, not the hoisted one, selected
                        // through pointers so the hoisted corners stay loop-invariant.
                        const scalar_t *const gx = curved_e ? x : hx;
                        const scalar_t *const gy = curved_e ? y : hy;
                        const scalar_t *const gz = curved_e ? z : hz;

                        const Hex8RhieChow rc{gx,      gy, gz,   pgx, pgy, pgz, rcfg.scale, nullptr, nullptr,
                                              nullptr, ux, uy, uz,  rcfg.tau};
                        scalar_t           adj[9], det;
                        sscvfem_micro_geom(gx, gy, gz, adj, &det);
                        cvfem_hex8_ns_upwind_jacobian_action<0>(rho, mu, adj, det, ux, uy, uz, vx, vy, vz, q, r,
                                                        rc, p, upwind_eps);
                        boundary_scs_add_jacobian_action<false>(rho, mu, adj, det, box_lx, box_ly, box_lz, x, y, z, ux, uy, uz,
                                                         vx, vy, vz, q, r);

                        // Accumulate locally: no atomic, no contention, contiguous.
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            for (int c = 0; c < CVFEM_HEX8_N_FIELDS; ++c) lout[(size_t)l * CVFEM_HEX8_N_FIELDS + c] += r[a * 4 + c];
                        }
                    }
                }
            }

            // Scatter once per macro node instead of once per element-node incidence.
            for (int a = 0; a < nxe; ++a) {
                const idx_t g = lg[(size_t)a];
                for (int c = 0; c < CVFEM_HEX8_N_FIELDS; ++c)
                    atomic_add(jv + (ptrdiff_t)g * CVFEM_HEX8_N_FIELDS + c, 0, lout[(size_t)a * CVFEM_HEX8_N_FIELDS + c]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// macro_local, plus the geometry hoisted out of the micro-element loop.
//
// The flat kernel loads a precomputed adjugate and determinant per element; the two
// variants above recompute the Jacobian from eight corners for every micro-element, so
// they were doing strictly more work than the kernel they are meant to beat. Inside an
// affine macro-element every micro-element is a translate of the same box, so adj and det
// are invariant over the whole L^3 sweep and belong outside it.
//
// This is only valid when the macro-element is affine, which is true of the box meshes
// benchmarked here and false in general -- a trilinear macro-element has a Jacobian that
// varies across its lattice. The assert guards it: the geometry of the last micro-element
// is compared against the hoisted value, so a curved macro-element fails loudly rather
// than silently returning a wrong operator.
inline SFEM_NOINLINE void sscvfem_apply_macro_local_affine(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const scalar_t box_lx,
        const scalar_t box_ly,
        const scalar_t box_lz,
        idx_t **const SFEM_RESTRICT elems,
        const int level,
        const uint8_t *const SFEM_RESTRICT macro_curved,
        const ptrdiff_t nmacro,
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
    CVFEM_TRACE_SCOPE("sscvfem::apply_macro_local_affine");
    const int L   = level;
    const int nxe = nxe_src;
    int       off[8];
    sscvfem_corner_offsets(L, off);

#pragma omp parallel
    {
        // Per-thread scratch from the kernels' own arena, not std::vector locals: slots 5/6 are the semi-structured element sweeps',
        // shared between them because only one is live inside a parallel region and
        // they all want the same macro-element size.
        scalar_t *const SFEM_RESTRICT _arena5 = thread_scratch<scalar_t>(5, ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe * CVFEM_HEX8_N_FIELDS));
        scalar_t *const SFEM_RESTRICT lx = _arena5;
        scalar_t *const SFEM_RESTRICT ly = _arena5 + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lz = _arena5 + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lux = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT luy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT luz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lp = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lvx = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lvy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lvz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lq = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgx = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lout = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        idx_t *const SFEM_RESTRICT _arena6 = thread_scratch<idx_t>(6, ((size_t)nxe));
        idx_t *const SFEM_RESTRICT lg = _arena6;

#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < nmacro; ++e) {
            for (int a = 0; a < nxe; ++a) {
                const idx_t g = elems[a][e];
                lg[(size_t)a]        = g;
                lx[(size_t)a]        = (scalar_t)points[0][g];
                ly[(size_t)a]        = (scalar_t)points[1][g];
                lz[(size_t)a]        = (scalar_t)points[2][g];
                lux[(size_t)a]       = ux_src[(size_t)g];
                luy[(size_t)a]       = uy_src[(size_t)g];
                luz[(size_t)a]       = uz_src[(size_t)g];
                lp[(size_t)a]        = pres[(size_t)g];
                lvx[(size_t)a]       = dir[(size_t)g * 4 + 0];
                lvy[(size_t)a]       = dir[(size_t)g * 4 + 1];
                lvz[(size_t)a]       = dir[(size_t)g * 4 + 2];
                lq[(size_t)a]        = dir[(size_t)g * 4 + 3];
                lpgx[(size_t)a]      = pgx_src[(size_t)g];
                lpgy[(size_t)a]      = pgy_src[(size_t)g];
                lpgz[(size_t)a]      = pgz_src[(size_t)g];
            }
            std::fill(lout, lout + ((size_t)nxe * CVFEM_HEX8_N_FIELDS), scalar_t(0));

            // Once per macro-element, from its first micro-element.
            // Micro-cell 0's corners, hoisted: the geometry AND the coordinates the
            // Rhie-Chow term differences.
            //
            // The lattice inside a macro element is uniform, so every micro-cell is congruent
            // to cell 0 and one adjugate serves all of them -- that is what the action does.
            // This used to hoist the adjugate but then hand the Rhie-Chow struct each cell's
            // OWN coordinates, and the two agree only to the precision the node positions are
            // stored in. geom_t is float32, so the block diagonal disagreed with the
            // action it is supposed to be the diagonal of by 4.23e-08 -- eight orders above
            // round-off, and invisible until the q-independent consistency gate looked.
            //
            // Only DIFFERENCES of these are taken (d = x_j - x_i), so cell 0's coordinates are
            // exact for the purpose, not an approximation. The boundary closure below still
            // gets each cell's real position, because it tests where the cell actually is.
            scalar_t madj[9], mdet;
            scalar_t c0x[8], c0y[8], c0z[8];
            {
                int ext[8];
                sscvfem_macro_corner_offsets(L, ext);
                for (int a = 0; a < 8; ++a) {
                    const int l = ext[a];
                    c0x[a]      = lx[(size_t)l];
                    c0y[a]      = ly[(size_t)l];
                    c0z[a]      = lz[(size_t)l];
                }
                sscvfem_hoisted_cell(c0x, c0y, c0z, L, c0x, c0y, c0z);
                sscvfem_micro_geom(c0x, c0y, c0z, madj, &mdet);
            }

            const bool curved_e = sscvfem_macro_curved(macro_curved, e);
            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);

                        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
                        scalar_t vx[8], vy[8], vz[8], q[8], pgx[8], pgy[8], pgz[8];
                        scalar_t r[CVFEM_HEX8_N_DOF];
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            x[a]        = lx[(size_t)l];
                            y[a]        = ly[(size_t)l];
                            z[a]        = lz[(size_t)l];
                            ux[a]       = lux[(size_t)l];
                            uy[a]       = luy[(size_t)l];
                            uz[a]       = luz[(size_t)l];
                            p[a]        = lp[(size_t)l];
                            vx[a]       = lvx[(size_t)l];
                            vy[a]       = lvy[(size_t)l];
                            vz[a]       = lvz[(size_t)l];
                            q[a]        = lq[(size_t)l];
                            pgx[a]      = lpgx[(size_t)l];
                            pgy[a]      = lpgy[(size_t)l];
                            pgz[a]      = lpgz[(size_t)l];
                        }
                        // A curved macro element: this cell's own geometry, not the hoisted one. Selected
                        // through pointers so the hoisted madj, mdet and corners stay loop-invariant --
                        // overwriting them here cost the affine path 7% in the block diagonal.
                        scalar_t cadj[9], cdet = 0;
                        if (curved_e) sscvfem_micro_geom(x, y, z, cadj, &cdet);
                        const scalar_t *const gadj = curved_e ? cadj : madj;
                        const scalar_t        gdet = curved_e ? cdet : mdet;
                        const scalar_t *const gx = curved_e ? x : c0x;
                        const scalar_t *const gy = curved_e ? y : c0y;
                        const scalar_t *const gz = curved_e ? z : c0z;

                        // The hoisted cell's distances, matching madj: see sscvfem_residual.
                        const Hex8RhieChow rc{gx,      gy,  gz,  pgx, pgy, pgz, rcfg.scale, nullptr, nullptr,
                                              nullptr, ux, uy, uz,  rcfg.tau};
                        cvfem_hex8_ns_upwind_jacobian_action<0>(rho, mu, gadj, gdet, ux, uy, uz, vx, vy, vz, q, r,
                                                             rc, p, upwind_eps);
                        boundary_scs_add_jacobian_action<false>(rho, mu, gadj, gdet, box_lx, box_ly, box_lz, x, y, z, ux, uy, uz,
                                                         vx, vy, vz, q, r);

                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            for (int c = 0; c < CVFEM_HEX8_N_FIELDS; ++c) lout[(size_t)l * CVFEM_HEX8_N_FIELDS + c] += r[a * 4 + c];
                        }
                    }
                }
            }

            for (int a = 0; a < nxe; ++a) {
                const idx_t g = lg[(size_t)a];
                for (int c = 0; c < CVFEM_HEX8_N_FIELDS; ++c)
                    atomic_add(jv + (ptrdiff_t)g * CVFEM_HEX8_N_FIELDS + c, 0, lout[(size_t)a * CVFEM_HEX8_N_FIELDS + c]);
            }
        }
    }
}

// The micro-cells of a macro element are congruent, so everything here is computed once per
// macro element and read by all L^3 of them. The Rhie-Chow time scale broke that: its
// advective branch carries the velocity, which varies cell to cell.
//
// The split keeps the hoist. Only |u|^2 is per-cell, and it enters the time scale as
// (2|u|/h)^2 = 4|u|^2/h^2, so the macro element can hold everything else --
//
//   rc_num[s]  = rc_scale * A2/Adotd     the whole geometric factor, zero where degenerate
//   rc_base[s] = (2 a0/dt)^2 + (4 nu/h^2)^2   the transient and diffusive branches
//   inv_h2[s]  = 1/|d|^2
//
// -- and a cell pays one add, one multiply, one square root and one divide rather than the
// twelve full coefficient evaluations it would otherwise need.
struct SSMacroGeom {
    scalar_t adj[9];
    scalar_t det;
    scalar_t A[3][3];
    scalar_t rc_num[CVFEM_HEX8_N_SCS];
    scalar_t rc_base[CVFEM_HEX8_N_SCS];
    scalar_t inv_h2[CVFEM_HEX8_N_SCS];
    // The coefficient's velocity-sensitivity weight, pure geometry and so hoisted with the rest:
    // g = coeff^3 * rc_duw (cvfem_hex8_rhie_chow_du_weight), one multiply per surface in the sweep.
    scalar_t rc_duw[CVFEM_HEX8_N_SCS];
    scalar_t dvec[CVFEM_HEX8_N_SCS][3];
};

// The per-cell half of the coefficient. Identical to cvfem_hex8_rhie_chow_mdot_coeff by
// construction -- the flat-versus-semi-structured parity test is what holds the two together.
static SFEM_INLINE scalar_t sscvfem_rc_coeff(const SSMacroGeom &g, const int s, const scalar_t u2) {
    return g.rc_num[s] / std::sqrt(g.rc_base[s] + scalar_t(4) * u2 * g.inv_h2[s]);
}

inline void sscvfem_macro_geom(const scalar_t x[8], const scalar_t y[8], const scalar_t z[8],
                               const scalar_t rho, const scalar_t mu, const scalar_t rc_scale,
                               const Hex8RcTau &tau, SSMacroGeom &g) {
    sscvfem_micro_geom(x, y, z, g.adj, &g.det);
    cvfem_hex8_dir_areas(g.adj, g.A);
    const scalar_t nu = (mu > scalar_t(1e-30) ? mu : scalar_t(1e-30)) / (rho > scalar_t(0) ? rho : scalar_t(1));
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int i = CVFEM_HEX8_SCS[s].i;
        const int j = CVFEM_HEX8_SCS[s].j;
        const int d = s >> 2;
        g.dvec[s][0] = x[j] - x[i];
        g.dvec[s][1] = y[j] - y[i];
        g.dvec[s][2] = z[j] - z[i];

        const scalar_t dx = g.dvec[s][0], dy = g.dvec[s][1], dz = g.dvec[s][2];
        const scalar_t ax = g.A[d][0], ay = g.A[d][1], az = g.A[d][2];
        const scalar_t h2    = dx * dx + dy * dy + dz * dz;
        const scalar_t Adotd = ax * dx + ay * dy + az * dz;
        const scalar_t A2    = ax * ax + ay * ay + az * az;
        const scalar_t lim   = scalar_t(1e-30) * (std::sqrt(A2 * h2) + scalar_t(1e-30));
        // A degenerate surface contributes nothing, exactly as the flat guard makes it:
        // a zero numerator over a one denominator, so the divide below stays finite.
        if (rc_scale == scalar_t(0) || rho == scalar_t(0) || std::fabs(Adotd) < lim) {
            g.rc_num[s]  = scalar_t(0);
            g.rc_base[s] = scalar_t(1);
            g.inv_h2[s]  = scalar_t(0);
            g.rc_duw[s]  = scalar_t(0);
            continue;
        }
        const scalar_t ct = scalar_t(2) * tau.inv_dt_a0;
        const scalar_t cd = scalar_t(4) * nu / h2;
        g.rc_num[s]  = rc_scale * (A2 / Adotd);
        g.rc_base[s] = ct * ct + cd * cd;
        g.inv_h2[s]  = tau.u2_scale / h2;
        g.rc_duw[s]  = cvfem_hex8_rhie_chow_du_weight(rc_scale, A2, Adotd, h2, tau.u2_scale);
    }
}

// sscvfem_macro_geom for one micro cell of a curved macro element, out of line on purpose.
//
// Every curved branch of the hoisted sweeps rebuilds the geometry per cell. Inlined, that put a
// copy of the whole construction into each sweep and each of apply_blocks' eight instantiations,
// and the unit grew past the point where GCC still inlined the small per-cell helpers the affine
// sweeps depend on: sscvfem_rc_config became a call in every micro cell of the block diagonal,
// 9% slower on meshes with no curved element at all. A call per curved cell costs nothing next to
// the construction it wraps.
static SFEM_NOINLINE void sscvfem_macro_geom_cell(const scalar_t x[8], const scalar_t y[8], const scalar_t z[8],
                                                  const scalar_t rho, const scalar_t mu, const Hex8RcConfig &rc,
                                                  SSMacroGeom &g) {
    sscvfem_macro_geom(x, y, z, rho, mu, rc.scale, rc.tau, g);
}

// Mirrors cvfem_hex8_ns_upwind_jacobian_action with the invariants passed in.
static SFEM_INLINE void sscvfem_action_hoisted(const scalar_t rho, const scalar_t mu, const SSMacroGeom &g,
                                               const scalar_t *const SFEM_RESTRICT ux,
                                               const scalar_t *const SFEM_RESTRICT uy,
                                               const scalar_t *const SFEM_RESTRICT uz,
                                               const scalar_t *const SFEM_RESTRICT vx,
                                               const scalar_t *const SFEM_RESTRICT vy,
                                               const scalar_t *const SFEM_RESTRICT vz,
                                               const scalar_t *const SFEM_RESTRICT q,
                                               const scalar_t *const SFEM_RESTRICT p,
                                               const scalar_t *const SFEM_RESTRICT pgx,
                                               const scalar_t *const SFEM_RESTRICT pgy,
                                               const scalar_t *const SFEM_RESTRICT pgz,
                                               const scalar_t *const SFEM_RESTRICT qgx,
                                               const scalar_t *const SFEM_RESTRICT qgy,
                                               const scalar_t *const SFEM_RESTRICT qgz,
                                               scalar_t *const SFEM_RESTRICT       r,
                                              const scalar_t ueps = scalar_t(0)) {
    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    scalar_t dgrad[9];
    cvfem_hex8_grad_sumfact(g.adj, g.det, vx, vy, vz, dgrad);

    for (int d = 0; d < 3; ++d) {
        scalar_t tx, ty, tz;
        cvfem_hex8_traction(mu, dgrad[0], dgrad[1], dgrad[2], dgrad[3], dgrad[4], dgrad[5], dgrad[6], dgrad[7],
                            dgrad[8], g.A[d][0], g.A[d][1], g.A[d][2], tx, ty, tz);
        for (int e = 0; e < 4; ++e) {
            const int i = CVFEM_HEX8_DIR_EDGES[d][e][0];
            const int j = CVFEM_HEX8_DIR_EDGES[d][e][1];
            r[i * 4 + 0] -= tx;
            r[i * 4 + 1] -= ty;
            r[i * 4 + 2] -= tz;
            r[j * 4 + 0] += tx;
            r[j * 4 + 1] += ty;
            r[j * 4 + 2] += tz;
        }
    }

    const scalar_t half = scalar_t(0.5);
    const scalar_t one  = scalar_t(1);
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int      i  = CVFEM_HEX8_SCS[s].i;
        const int      j  = CVFEM_HEX8_SCS[s].j;
        const int      d  = s >> 2;
        const scalar_t ax = g.A[d][0], ay = g.A[d][1], az = g.A[d][2];

        const scalar_t adv_x = half * (ux[i] + ux[j]);
        const scalar_t adv_y = half * (uy[i] + uy[j]);
        const scalar_t adv_z = half * (uz[i] + uz[j]);
        const scalar_t c = sscvfem_rc_coeff(g, s, adv_x * adv_x + adv_y * adv_y + adv_z * adv_z);

        // -coeff * ((p_j - p_i) - avg(grad p) . d), with coeff and d both loop invariants.
        const scalar_t corr = (p[j] - p[i]) - (half * (pgx[i] + pgx[j]) * g.dvec[s][0] +
                                               half * (pgy[i] + pgy[j]) * g.dvec[s][1] +
                                               half * (pgz[i] + pgz[j]) * g.dvec[s][2]);
        const scalar_t mdot_rc = -c * corr;

        const scalar_t mdot  = rho * (adv_x * ax + adv_y * ay + adv_z * az) + mdot_rc;
        scalar_t amdot, sgn;
        cvfem_upwind_abs(mdot, ueps, amdot, sgn);
        const scalar_t mpos  = half * (mdot + amdot);
        const scalar_t mneg  = half * (mdot - amdot);
        const scalar_t d_pos = half * (one + sgn);
        const scalar_t d_neg = half * (one - sgn);

        // Mirror the residual's corr above: corr_q = (q_j - q_i) - avg(qg_i, qg_j) . d, which
        // contributes -c * corr_q. Keeping only c*(q_i - q_j) freezes the pressure-gradient
        // reconstruction, leaving the continuity rows ~4% wrong and capping Newton at a linear
        // rate. qgx == nullptr restores that old behaviour. See SFEM_FD_CHECK.
        const scalar_t dcorr = qgx ? (half * (qgx[i] + qgx[j]) * g.dvec[s][0] +
                                      half * (qgy[i] + qgy[j]) * g.dvec[s][1] +
                                      half * (qgz[i] + qgz[j]) * g.dvec[s][2])
                                   : scalar_t(0);
        // The coefficient's own velocity dependence; see cvfem_hex8_rhie_chow_coeff_du.
        const scalar_t gdu   = cvfem_hex8_rhie_chow_coeff_du(c, g.rc_duw[s]);
        const scalar_t dmdot = rho * half * ((vx[i] + vx[j]) * ax + (vy[i] + vy[j]) * ay + (vz[i] + vz[j]) * az) +
                               c * (q[i] - q[j]) + c * dcorr +
                               gdu * corr *
                                       (adv_x * half * (vx[i] + vx[j]) + adv_y * half * (vy[i] + vy[j]) +
                                        adv_z * half * (vz[i] + vz[j]));
        const scalar_t dpos = d_pos * dmdot;
        const scalar_t dneg = d_neg * dmdot;
        const scalar_t qmid = half * (q[i] + q[j]);
        const scalar_t fx   = dpos * ux[i] + mpos * vx[i] + dneg * ux[j] + mneg * vx[j] + qmid * ax;
        const scalar_t fy   = dpos * uy[i] + mpos * vy[i] + dneg * uy[j] + mneg * vy[j] + qmid * ay;
        const scalar_t fz   = dpos * uz[i] + mpos * vz[i] + dneg * uz[j] + mneg * vz[j] + qmid * az;
        r[i * 4 + 0] += fx;
        r[i * 4 + 1] += fy;
        r[i * 4 + 2] += fz;
        r[i * 4 + 3] += dmdot;
        r[j * 4 + 0] -= fx;
        r[j * 4 + 1] -= fy;
        r[j * 4 + 2] -= fz;
        r[j * 4 + 3] -= dmdot;
    }
}

inline SFEM_NOINLINE void sscvfem_apply_macro_local_hoisted(
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
        const uint8_t *const SFEM_RESTRICT macro_curved,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const uint8_t *const SFEM_RESTRICT natural_mask,
        const uint8_t *const SFEM_RESTRICT pressure_mask,
        const uint8_t *const SFEM_RESTRICT traction_mask,
        const ptrdiff_t nmacro,
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
    CVFEM_TRACE_SCOPE("sscvfem::apply_macro_local_hoisted");
    const int L   = level;
    const int nxe = nxe_src;
    int       off[8];
    sscvfem_corner_offsets(L, off);


#pragma omp parallel
    {
        // Whether the direction's pressure gradient is there at all. Above the scratch because
        // the scratch is sized from it.
        const bool                has_qg = qgx_src;
        // Per-thread scratch from the kernels' own arena, not std::vector locals: slots 5/6 are the semi-structured element sweeps',
        // shared between them because only one is live inside a parallel region and
        // they all want the same macro-element size.
        scalar_t *const SFEM_RESTRICT _arena5 = thread_scratch<scalar_t>(5, ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)(has_qg ? nxe : 0)) + ((size_t)(has_qg ? nxe : 0)) + ((size_t)(has_qg ? nxe : 0)) + ((size_t)nxe * CVFEM_HEX8_N_FIELDS));
        scalar_t *const SFEM_RESTRICT lx = _arena5;
        scalar_t *const SFEM_RESTRICT ly = _arena5 + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lz = _arena5 + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lux = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT luy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT luz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lp = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lvx = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lvy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lvz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lq = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgx = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lqgx = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lqgy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)(has_qg ? nxe : 0));
        scalar_t *const SFEM_RESTRICT lqgz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)(has_qg ? nxe : 0)) + ((size_t)(has_qg ? nxe : 0));
        scalar_t *const SFEM_RESTRICT lout = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)(has_qg ? nxe : 0)) + ((size_t)(has_qg ? nxe : 0)) + ((size_t)(has_qg ? nxe : 0));
        idx_t *const SFEM_RESTRICT _arena6 = thread_scratch<idx_t>(6, ((size_t)nxe));
        idx_t *const SFEM_RESTRICT lg = _arena6;

#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < nmacro; ++e) {
            for (int a = 0; a < nxe; ++a) {
                const idx_t g = elems[a][e];
                lg[(size_t)a]        = g;
                lx[(size_t)a]        = (scalar_t)points[0][g];
                ly[(size_t)a]        = (scalar_t)points[1][g];
                lz[(size_t)a]        = (scalar_t)points[2][g];
                lux[(size_t)a]       = ux_src[(size_t)g];
                luy[(size_t)a]       = uy_src[(size_t)g];
                luz[(size_t)a]       = uz_src[(size_t)g];
                lp[(size_t)a]        = pres[(size_t)g];
                lvx[(size_t)a]       = dir[(size_t)g * 4 + 0];
                lvy[(size_t)a]       = dir[(size_t)g * 4 + 1];
                lvz[(size_t)a]       = dir[(size_t)g * 4 + 2];
                lq[(size_t)a]        = dir[(size_t)g * 4 + 3];
                lpgx[(size_t)a]      = pgx_src[(size_t)g];
                lpgy[(size_t)a]      = pgy_src[(size_t)g];
                lpgz[(size_t)a]      = pgz_src[(size_t)g];
                if (has_qg) {
                    lqgx[(size_t)a] = qgx_src[(size_t)g];
                    lqgy[(size_t)a] = qgy_src[(size_t)g];
                    lqgz[(size_t)a] = qgz_src[(size_t)g];
                }
            }
            std::fill(lout, lout + ((size_t)nxe * CVFEM_HEX8_N_FIELDS), scalar_t(0));

            // Per macro element, and the curved branch below reads the same one: a call per
            // micro cell there took this unit past the point where GCC inlines sscvfem_rc_config,
            // which then became a call in every cell of the block diagonal, 9% slower on boxes.
            const Hex8RcConfig rc_macro = rcfg;
            SSMacroGeom mg;
            {
                scalar_t ex[8], ey[8], ez[8];
                int ext[8];
                sscvfem_macro_corner_offsets(L, ext);
                for (int a = 0; a < 8; ++a) {
                    const int l = ext[a];
                    ex[a]       = lx[(size_t)l];
                    ey[a]       = ly[(size_t)l];
                    ez[a]       = lz[(size_t)l];
                }
                sscvfem_hoisted_cell(ex, ey, ez, L, ex, ey, ez);
                sscvfem_macro_geom(ex, ey, ez, rho, mu, rc_macro.scale, rc_macro.tau, mg);
            }

            const bool curved_e = sscvfem_macro_curved(macro_curved, e);
            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);

                        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
                        scalar_t vx[8], vy[8], vz[8], q[8], pgx[8], pgy[8], pgz[8];
                        scalar_t qgx[8], qgy[8], qgz[8];
                        scalar_t r[CVFEM_HEX8_N_DOF];
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            x[a]        = lx[(size_t)l];
                            y[a]        = ly[(size_t)l];
                            z[a]        = lz[(size_t)l];
                            ux[a]       = lux[(size_t)l];
                            uy[a]       = luy[(size_t)l];
                            uz[a]       = luz[(size_t)l];
                            p[a]        = lp[(size_t)l];
                            vx[a]       = lvx[(size_t)l];
                            vy[a]       = lvy[(size_t)l];
                            vz[a]       = lvz[(size_t)l];
                            q[a]        = lq[(size_t)l];
                            pgx[a]      = lpgx[(size_t)l];
                            pgy[a]      = lpgy[(size_t)l];
                            pgz[a]      = lpgz[(size_t)l];
                            if (has_qg) {
                                qgx[a] = lqgx[(size_t)l];
                                qgy[a] = lqgy[(size_t)l];
                                qgz[a] = lqgz[(size_t)l];
                            }
                        }
                        // A curved macro element: this cell's own geometry, not the hoisted one.
                        if (curved_e) {
                            sscvfem_macro_geom_cell(x, y, z, rho, mu, rc_macro, mg);
                        }

                        sscvfem_action_hoisted(rho, mu, mg, ux, uy, uz, vx, vy, vz, q, p, pgx, pgy, pgz,
                                               has_qg ? qgx : nullptr, has_qg ? qgy : nullptr,
                                               has_qg ? qgz : nullptr, r, upwind_eps);
                        boundary_scs_add_jacobian_action<false>(rho, mu, mg.adj, mg.det, box_lx, box_ly, box_lz, x, y, z,
                                                         ux, uy, uz, vx, vy, vz, q, r,
                                                         !face_mask
                                                          ? -1
                                                          : sscvfem_micro_face_mask(
                                                                    (int)face_mask[(size_t)e],
                                                                    L, xi, yi, zi),
                                                  sscvfem_micro_face_mask(
                                                          !natural_mask ? 0
                                                              : (int)natural_mask[(size_t)e],
                                                          L, xi, yi, zi),
                                                  sscvfem_bd(bc_p, bc_tx, bc_ty, bc_tz, pressure_mask, traction_mask, e, L, xi, yi, zi));

                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            for (int c = 0; c < CVFEM_HEX8_N_FIELDS; ++c) lout[(size_t)l * CVFEM_HEX8_N_FIELDS + c] += r[a * 4 + c];
                        }
                    }
                }
            }

            if (slot)
                sscvfem_scatter_element(slot, const_cast<scalar_t *>(stage), nxe, e, lg, lout, jv);
            else
                for (int a = 0; a < nxe; ++a) {
                    const idx_t g = lg[(size_t)a];
                    for (int c = 0; c < CVFEM_HEX8_N_FIELDS; ++c)
                        atomic_add(jv + (ptrdiff_t)g * CVFEM_HEX8_N_FIELDS + c, 0, lout[(size_t)a * CVFEM_HEX8_N_FIELDS + c]);
                }
        }
    }

    if (slot) sscvfem_reduce_shared(red_idx, red_ptr, shared_node, const_cast<scalar_t *>(stage), n_shared, jv);
}

// ---------------------------------------------------------------------------
// The linear part as a small dense matrix, applied as a matvec.
//
// For a fixed state the Jacobian action is linear in the direction, and under the
// affine-macro assumption everything in it except the convective flux has coefficients
// that are pure geometry. So that part is one constant 32x32 matrix for the whole macro,
// and applying it is a dense matvec -- the same shape as the element-matrix path SFEM
// already uses for semi-structured linear elasticity (sfem_SemiStructuredEMLinearElasticity
// and operators/stencil/sshex8_stencil_element_matrix_apply*).
//
// The matrix is not written out by hand. With u = 0, p = 0 and grad p = 0 the existing
// action kernel reduces exactly to the geometry-linear operator: mdot vanishes, so the
// upwind weights mpos and mneg vanish with it, and what survives is the viscous term, the
// pressure gradient qmid*A in the momentum rows, and the whole continuity row. So the
// matrix is obtained by probing the unmodified kernel with the 32 unit vectors -- it is
// consistent with the kernel by construction, which the hand-written variant above is not.
// 32 probes are amortised over L^3 micro-elements: 6% at L=8.
//
// What is left outside is the convective momentum flux alone, whose upwind weights depend
// on the state. The continuity row needs no remainder at all.
//
// Whether this is faster is not obvious and is not argued here: the matvec is about twice
// the FLOPs of evaluating those terms directly, but it is branch-free, contiguous, and the
// matrix stays in L1 across the whole macro-element. The benchmark decides.

// ---------------------------------------------------------------------------
// The 2x2 field-block split: (velocity, pressure) x (velocity, pressure).
//
//        | A_uu  B^T |   momentum rows
//   J =  |           |
//        | B     C   |   continuity rows
//
// Solution schemes want these separately. A Schur approximation needs B and B^T to form
// B A^-1 B^T; a segregated or projection scheme solves the momentum rows alone; the
// pressure preconditioner investigated in the standalone driver needs C by itself; and a
// Vanka or block smoother wants to address them independently. Evaluating the whole
// operator and discarding three quarters of it is the thing to avoid.
//
// Where each term lands, which is not one-to-one with the code's own structure:
//
//   viscous                     -> A_uu
//   qmid * A                    -> B^T
//   convective mpos/mneg on v   -> A_uu
//   continuity dmdot            -> split, see below
//
// The convective flux contributes to BOTH A_uu and B^T, because the mass-flux derivative
// carries a velocity part and a pressure part:
//
//   dmdot = rho/2 (v_i + v_j).A   +   c (q_i - q_j)
//           \_____ velocity _____/     \___ pressure ___/
//
// so d_pos * dmdot * u_i splits along the same line, and the continuity row splits into
// B (the velocity half) and C (the Rhie-Chow half). Getting that wrong would put the
// Rhie-Chow coupling in A_uu, where it would quietly break any Schur approximation built
// on these blocks.

enum SSBlock : int {
    SSBLOCK_UU  = 1 << 0,  // momentum rows, velocity columns
    SSBLOCK_UP  = 1 << 1,  // momentum rows, pressure column   (B^T)
    SSBLOCK_PU  = 1 << 2,  // continuity row, velocity columns (B)
    SSBLOCK_PP  = 1 << 3,  // continuity row, pressure column  (C)
    SSBLOCK_MOM = SSBLOCK_UU | SSBLOCK_UP,
    SSBLOCK_CON = SSBLOCK_PU | SSBLOCK_PP,
    SSBLOCK_ALL = SSBLOCK_MOM | SSBLOCK_CON
};

// Fast path: the hoisted action with the unwanted terms compiled out.
template <int Blocks>
static SFEM_INLINE void sscvfem_action_blocks(const scalar_t rho, const scalar_t mu, const SSMacroGeom &g,
                                              const scalar_t *const SFEM_RESTRICT ux,
                                              const scalar_t *const SFEM_RESTRICT uy,
                                              const scalar_t *const SFEM_RESTRICT uz,
                                              const scalar_t *const SFEM_RESTRICT vx,
                                              const scalar_t *const SFEM_RESTRICT vy,
                                              const scalar_t *const SFEM_RESTRICT vz,
                                              const scalar_t *const SFEM_RESTRICT q,
                                              const scalar_t *const SFEM_RESTRICT p,
                                              const scalar_t *const SFEM_RESTRICT pgx,
                                              const scalar_t *const SFEM_RESTRICT pgy,
                                              const scalar_t *const SFEM_RESTRICT pgz,
                                              const scalar_t *const SFEM_RESTRICT qgx,
                                              const scalar_t *const SFEM_RESTRICT qgy,
                                              const scalar_t *const SFEM_RESTRICT qgz,
                                              scalar_t *const SFEM_RESTRICT       r,
                                              const scalar_t ueps = scalar_t(0)) {
    constexpr bool uu = (Blocks & SSBLOCK_UU) != 0;
    constexpr bool up = (Blocks & SSBLOCK_UP) != 0;
    constexpr bool pu = (Blocks & SSBLOCK_PU) != 0;
    constexpr bool pp = (Blocks & SSBLOCK_PP) != 0;
    constexpr bool mom = uu || up;

    for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) r[i] = scalar_t(0);

    // Viscous: A_uu only. Skipped entirely for a pressure-block evaluation, which is most
    // of what makes C cheap to get on its own.
    if constexpr (uu) {
        scalar_t dgrad[9];
        cvfem_hex8_grad_sumfact(g.adj, g.det, vx, vy, vz, dgrad);
        for (int d2 = 0; d2 < 3; ++d2) {
            scalar_t tx, ty, tz;
            cvfem_hex8_traction(mu, dgrad[0], dgrad[1], dgrad[2], dgrad[3], dgrad[4], dgrad[5], dgrad[6],
                                dgrad[7], dgrad[8], g.A[d2][0], g.A[d2][1], g.A[d2][2], tx, ty, tz);
            for (int e = 0; e < 4; ++e) {
                const int i = CVFEM_HEX8_DIR_EDGES[d2][e][0];
                const int j = CVFEM_HEX8_DIR_EDGES[d2][e][1];
                r[i * 4 + 0] -= tx;
                r[i * 4 + 1] -= ty;
                r[i * 4 + 2] -= tz;
                r[j * 4 + 0] += tx;
                r[j * 4 + 1] += ty;
                r[j * 4 + 2] += tz;
            }
        }
    }

    const scalar_t half = scalar_t(0.5);
    const scalar_t one  = scalar_t(1);
    for (int s = 0; s < CVFEM_HEX8_N_SCS; ++s) {
        const int      i  = CVFEM_HEX8_SCS[s].i;
        const int      j  = CVFEM_HEX8_SCS[s].j;
        const int      dd = s >> 2;
        const scalar_t ax = g.A[dd][0], ay = g.A[dd][1], az = g.A[dd][2];
        const scalar_t uax = half * (ux[i] + ux[j]);
        const scalar_t uay = half * (uy[i] + uy[j]);
        const scalar_t uaz = half * (uz[i] + uz[j]);
        const scalar_t c   = sscvfem_rc_coeff(g, s, uax * uax + uay * uay + uaz * uaz);

        // The upwind weights are needed only by the momentum rows. The continuity row is
        // dmdot_v + dmdot_q with no sgn in it, so for a pressure-row evaluation the whole
        // upwind computation -- the Rhie-Chow correction, the mass flux, the sign and the
        // four weights -- is dead. That is most of what makes C cheap to ask for.
        scalar_t mpos = 0, mneg = 0, d_pos = 0, d_neg = 0;
        // The state's correction: the momentum rows need it for the mass flux, and the velocity
        // columns for the coefficient's velocity dependence.
        const scalar_t corr = (mom || pu) ? (p[j] - p[i]) - (half * (pgx[i] + pgx[j]) * g.dvec[s][0] +
                                                             half * (pgy[i] + pgy[j]) * g.dvec[s][1] +
                                                             half * (pgz[i] + pgz[j]) * g.dvec[s][2])
                                          : scalar_t(0);
        if constexpr (mom) {
            const scalar_t mdot = rho * (half * (ux[i] + ux[j]) * ax + half * (uy[i] + uy[j]) * ay +
                                         half * (uz[i] + uz[j]) * az) - c * corr;
            scalar_t amdot, sgn;
            cvfem_upwind_abs(mdot, ueps, amdot, sgn);
            mpos  = half * (mdot + amdot);
            mneg  = half * (mdot - amdot);
            d_pos = half * (one + sgn);
            d_neg = half * (one - sgn);
        }

        // The two halves of the mass-flux derivative, kept apart so the blocks can be.
        // Including the Rhie-Chow coefficient's velocity dependence, a velocity-column term, as
        // sscvfem_action_hoisted carries it; see cvfem_hex8_rhie_chow_coeff_du.
        const scalar_t dmdot_v = (uu || pu) ? rho * half * ((vx[i] + vx[j]) * ax + (vy[i] + vy[j]) * ay +
                                                            (vz[i] + vz[j]) * az) +
                                                      cvfem_hex8_rhie_chow_coeff_du(c, g.rc_duw[s]) * corr *
                                                              (uax * half * (vx[i] + vx[j]) + uay * half * (vy[i] + vy[j]) +
                                                               uaz * half * (vz[i] + vz[j]))
                                            : scalar_t(0);
        // Rhie-Chow differentiates through the nodal pressure-gradient reconstruction, and
        // that derivative is a pressure-column term -- it is built from the gradient of the
        // *direction's* pressure -- so it belongs to B^T and C and to neither velocity-column
        // block. sscvfem_action_hoisted carries it as c * dcorr. Omitting it here did not
        // make any one block wrong in an obvious way; it made the four of them fail to sum
        // back to the operator, which is exactly the second check the bench performs and had
        // been reporting at 1.0e-01 since the exact term was introduced.
        // qgx == nullptr is the frozen-pg form, as in the hoisted kernel.
        const scalar_t dcorr = ((up || pp) && qgx) ? (half * (qgx[i] + qgx[j]) * g.dvec[s][0] +
                                                      half * (qgy[i] + qgy[j]) * g.dvec[s][1] +
                                                      half * (qgz[i] + qgz[j]) * g.dvec[s][2])
                                                   : scalar_t(0);
        const scalar_t dmdot_q = (up || pp) ? c * ((q[i] - q[j]) + dcorr) : scalar_t(0);

        if constexpr (mom) {
            scalar_t fx = 0, fy = 0, fz = 0;
            if constexpr (uu) {
                const scalar_t apos = d_pos * dmdot_v;
                const scalar_t aneg = d_neg * dmdot_v;
                fx += apos * ux[i] + mpos * vx[i] + aneg * ux[j] + mneg * vx[j];
                fy += apos * uy[i] + mpos * vy[i] + aneg * uy[j] + mneg * vy[j];
                fz += apos * uz[i] + mpos * vz[i] + aneg * uz[j] + mneg * vz[j];
            }
            if constexpr (up) {
                const scalar_t apos = d_pos * dmdot_q;
                const scalar_t aneg = d_neg * dmdot_q;
                const scalar_t qmid = half * (q[i] + q[j]);
                fx += apos * ux[i] + aneg * ux[j] + qmid * ax;
                fy += apos * uy[i] + aneg * uy[j] + qmid * ay;
                fz += apos * uz[i] + aneg * uz[j] + qmid * az;
            }
            r[i * 4 + 0] += fx;
            r[i * 4 + 1] += fy;
            r[i * 4 + 2] += fz;
            r[j * 4 + 0] -= fx;
            r[j * 4 + 1] -= fy;
            r[j * 4 + 2] -= fz;
        }

        if constexpr (pu || pp) {
            scalar_t dm = 0;
            if constexpr (pu) dm += dmdot_v;
            if constexpr (pp) dm += dmdot_q;
            r[i * 4 + 3] += dm;
            r[j * 4 + 3] -= dm;
        }
    }
}

// Macro-local sweep for a chosen set of blocks.
//
// The boundary sub-control-surface term is handled by input masking rather than by
// specialising it: it is a shared kernel that writes all four blocks, and restating it
// here to split it would be a second copy of arithmetic that already exists. Zeroing the
// direction components outside the wanted columns, and dropping the rows outside the
// wanted rows, gives its contribution to those blocks exactly. It is a boundary term, so
// it runs on a vanishing fraction of the elements and its cost does not drive this.
template <int Blocks>
inline SFEM_NOINLINE void sscvfem_apply_blocks_impl(
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
        const uint8_t *const SFEM_RESTRICT macro_curved,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const uint8_t *const SFEM_RESTRICT natural_mask,
        const uint8_t *const SFEM_RESTRICT pressure_mask,
        const uint8_t *const SFEM_RESTRICT traction_mask,
        const ptrdiff_t nmacro,
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
    constexpr bool uu = (Blocks & SSBLOCK_UU) != 0;
    constexpr bool up = (Blocks & SSBLOCK_UP) != 0;
    constexpr bool pu = (Blocks & SSBLOCK_PU) != 0;
    constexpr bool pp = (Blocks & SSBLOCK_PP) != 0;
    constexpr bool mom = uu || up;

    const int L   = level;
    const int nxe = nxe_src;
    int       off[8];
    sscvfem_corner_offsets(L, off);


#pragma omp parallel
    {
        // construction, so this gather is skipped along with the rest of the pressure work.
        const bool                has_qg = (up || pp) && qgx_src;
        // Per-thread scratch from the kernels' own arena, not std::vector locals: slots 5/6 are the semi-structured element sweeps',
        // shared between them because only one is live inside a parallel region and
        // they all want the same macro-element size.
        scalar_t *const SFEM_RESTRICT _arena5 = thread_scratch<scalar_t>(5, ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)(has_qg ? nxe : 0)) + ((size_t)(has_qg ? nxe : 0)) + ((size_t)(has_qg ? nxe : 0)) + ((size_t)nxe * CVFEM_HEX8_N_FIELDS));
        scalar_t *const SFEM_RESTRICT lx = _arena5;
        scalar_t *const SFEM_RESTRICT ly = _arena5 + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lz = _arena5 + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lux = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT luy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT luz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lp = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lvx = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lvy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lvz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lq = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgx = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lqgx = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lqgy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)(has_qg ? nxe : 0));
        scalar_t *const SFEM_RESTRICT lqgz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)(has_qg ? nxe : 0)) + ((size_t)(has_qg ? nxe : 0));
        scalar_t *const SFEM_RESTRICT lout = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)(has_qg ? nxe : 0)) + ((size_t)(has_qg ? nxe : 0)) + ((size_t)(has_qg ? nxe : 0));
        idx_t *const SFEM_RESTRICT _arena6 = thread_scratch<idx_t>(6, ((size_t)nxe));
        idx_t *const SFEM_RESTRICT lg = _arena6;

#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < nmacro; ++e) {
            // Gather only what this block reads. On Grace the gather and scatter alone are
            // 35% of the full operator, so a block that still loads all fourteen arrays
            // cannot get far below that however little arithmetic it does -- C was 46%
            // against a 35% floor.
            //
            // Coordinates and the state velocity are always needed: the first by the macro
            // geometry, the second by the boundary term, which takes ux, uy, uz whatever
            // is being masked. The state pressure and its gradient are read only by the
            // upwind switch, which lives in the momentum rows. The direction velocity is
            // read by A_uu and B; the direction pressure by B^T and C.
            // It is also read by the continuity row: dmdot_v differentiates the Rhie-Chow
            // coefficient through corr (see cvfem_hex8_rhie_chow_coeff_du above), so B reads the
            // state pressure and its gradient as well. The guard on corr itself was widened for
            // that term; this one was not, which silently dropped it from B wherever corr is not
            // zero to round-off -- i.e. on any mesh whose spacing is not a binary fraction.
            constexpr bool need_state_p = mom || pu;        // upwind correction, and B's RC coefficient
            constexpr bool need_dir_v   = uu || pu;
            constexpr bool need_dir_q   = up || pp;

            for (int a = 0; a < nxe; ++a) {
                const idx_t g = elems[a][e];
                lg[(size_t)a]        = g;
                lx[(size_t)a]        = (scalar_t)points[0][g];
                ly[(size_t)a]        = (scalar_t)points[1][g];
                lz[(size_t)a]        = (scalar_t)points[2][g];
                lux[(size_t)a]       = ux_src[(size_t)g];
                luy[(size_t)a]       = uy_src[(size_t)g];
                luz[(size_t)a]       = uz_src[(size_t)g];
                if constexpr (need_state_p) {
                    lp[(size_t)a]   = pres[(size_t)g];
                    lpgx[(size_t)a] = pgx_src[(size_t)g];
                    lpgy[(size_t)a] = pgy_src[(size_t)g];
                    lpgz[(size_t)a] = pgz_src[(size_t)g];
                }
                if constexpr (need_dir_v) {
                    lvx[(size_t)a] = dir[(size_t)g * 4 + 0];
                    lvy[(size_t)a] = dir[(size_t)g * 4 + 1];
                    lvz[(size_t)a] = dir[(size_t)g * 4 + 2];
                }
                if constexpr (need_dir_q) lq[(size_t)a] = dir[(size_t)g * 4 + 3];
                if (has_qg) {
                    lqgx[(size_t)a] = qgx_src[(size_t)g];
                    lqgy[(size_t)a] = qgy_src[(size_t)g];
                    lqgz[(size_t)a] = qgz_src[(size_t)g];
                }
            }
            // Anything not gathered must still read as zero, since the element kernels and
            // the boundary term take all of them regardless.
            if constexpr (!need_state_p) {
                std::fill(lp, lp + ((size_t)nxe), scalar_t(0));
                std::fill(lpgx, lpgx + ((size_t)nxe), scalar_t(0));
                std::fill(lpgy, lpgy + ((size_t)nxe), scalar_t(0));
                std::fill(lpgz, lpgz + ((size_t)nxe), scalar_t(0));
            }
            if constexpr (!need_dir_v) {
                std::fill(lvx, lvx + ((size_t)nxe), scalar_t(0));
                std::fill(lvy, lvy + ((size_t)nxe), scalar_t(0));
                std::fill(lvz, lvz + ((size_t)nxe), scalar_t(0));
            }
            if constexpr (!need_dir_q) std::fill(lq, lq + ((size_t)nxe), scalar_t(0));
            std::fill(lout, lout + ((size_t)nxe * CVFEM_HEX8_N_FIELDS), scalar_t(0));

            // Per macro element, and the curved branch below reads the same one: a call per
            // micro cell there took this unit past the point where GCC inlines sscvfem_rc_config,
            // which then became a call in every cell of the block diagonal, 9% slower on boxes.
            const Hex8RcConfig rc_macro = rcfg;
            SSMacroGeom mg;
            {
                scalar_t ex[8], ey[8], ez[8];
                int ext[8];
                sscvfem_macro_corner_offsets(L, ext);
                for (int a = 0; a < 8; ++a) {
                    const int l = ext[a];
                    ex[a]       = lx[(size_t)l];
                    ey[a]       = ly[(size_t)l];
                    ez[a]       = lz[(size_t)l];
                }
                sscvfem_hoisted_cell(ex, ey, ez, L, ex, ey, ez);
                sscvfem_macro_geom(ex, ey, ez, rho, mu, rc_macro.scale, rc_macro.tau, mg);
            }

            const bool curved_e = sscvfem_macro_curved(macro_curved, e);
            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);

                        scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
                        scalar_t vx[8], vy[8], vz[8], q[8], pgx[8], pgy[8], pgz[8];
                        scalar_t qgx[8], qgy[8], qgz[8];
                        scalar_t r[CVFEM_HEX8_N_DOF];
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            x[a]        = lx[(size_t)l];
                            y[a]        = ly[(size_t)l];
                            z[a]        = lz[(size_t)l];
                            ux[a]       = lux[(size_t)l];
                            uy[a]       = luy[(size_t)l];
                            uz[a]       = luz[(size_t)l];
                            p[a]        = lp[(size_t)l];
                            vx[a]       = lvx[(size_t)l];
                            vy[a]       = lvy[(size_t)l];
                            vz[a]       = lvz[(size_t)l];
                            q[a]        = lq[(size_t)l];
                            pgx[a]      = lpgx[(size_t)l];
                            pgy[a]      = lpgy[(size_t)l];
                            pgz[a]      = lpgz[(size_t)l];
                            if (has_qg) {
                                qgx[a] = lqgx[(size_t)l];
                                qgy[a] = lqgy[(size_t)l];
                                qgz[a] = lqgz[(size_t)l];
                            }
                        }
                        // A curved macro element: this cell's own geometry, not the hoisted one.
                        if (curved_e) {
                            sscvfem_macro_geom_cell(x, y, z, rho, mu, rc_macro, mg);
                        }

                        // upwind_eps, not the default zero: every other call site passes it,
                        // and a block apply that smooths the upwind switch differently from
                        // the operator is not a restriction of it either.
                        sscvfem_action_blocks<Blocks>(rho, mu, mg, ux, uy, uz, vx, vy, vz, q, p, pgx, pgy, pgz,
                                                      has_qg ? qgx : nullptr, has_qg ? qgy : nullptr,
                                                      has_qg ? qgz : nullptr, r, upwind_eps);

                        // Boundary term, by input masking. Two passes only when both
                        // column groups are wanted, which for the full operator is the
                        // single unmasked pass below.
                        // A row group that wants every column needs no input masking at
                        // all: run the boundary term once as it stands and keep the rows.
                        // Without this, asking for the momentum rows costs more than the
                        // whole operator, because the two masked passes outweigh the terms
                        // the specialisation removes.
                        constexpr bool all_cols_mom = uu && up;
                        constexpr bool all_cols_con = pu && pp;
                        constexpr bool no_masking =
                                (Blocks == SSBLOCK_ALL) ||
                                (all_cols_mom && !pu && !pp) || (all_cols_con && !uu && !up);

                        // The same boundary data the full action is given. Without it the
                        // closure fell back to a bounding-box plane test: on a box that marks the
                        // right faces but closes the do-nothing outlet, and on anything else it closes
                        // interior faces that happen to lie on a bounding plane. Measured on the FDA
                        // nozzle, apply_blocks(all) differed from apply by 65% in the continuity rows,
                        // and Vanka could not solve a system the dense LU solved in two Newton steps.
                        if constexpr (no_masking) {
                            scalar_t rb[CVFEM_HEX8_N_DOF];
                            for (int k = 0; k < CVFEM_HEX8_N_DOF; ++k) rb[k] = scalar_t(0);
                            boundary_scs_add_jacobian_action<false>(rho, mu, mg.adj, mg.det, box_lx, box_ly, box_lz, x, y, z,
                                                             ux, uy, uz, vx, vy, vz, q, rb,
                                                             !face_mask
                                                                     ? -1
                                                                     : sscvfem_micro_face_mask(
                                                                               (int)face_mask[(size_t)e],
                                                                               L, xi, yi, zi),
                                                             sscvfem_micro_face_mask(
                                                                     !natural_mask ? 0
                                                                         : (int)natural_mask[(size_t)e],
                                                                     L, xi, yi, zi),
                                                             sscvfem_bd(bc_p, bc_tx, bc_ty, bc_tz, pressure_mask, traction_mask, e, L, xi, yi, zi));
                            for (int a = 0; a < 8; ++a) {
                                if constexpr (uu || up)
                                    for (int cc = 0; cc < 3; ++cc) r[a * 4 + cc] += rb[a * 4 + cc];
                                if constexpr (pu || pp) r[a * 4 + 3] += rb[a * 4 + 3];
                            }
                        } else {
                            scalar_t zero8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                            scalar_t rb[CVFEM_HEX8_N_DOF];
                            if constexpr (uu || pu) {
                                for (int k = 0; k < CVFEM_HEX8_N_DOF; ++k) rb[k] = scalar_t(0);
                                boundary_scs_add_jacobian_action<false>(rho, mu, mg.adj, mg.det, box_lx, box_ly, box_lz, x, y, z,
                                                                 ux, uy, uz, vx, vy, vz, zero8, rb,
                                                             !face_mask
                                                                     ? -1
                                                                     : sscvfem_micro_face_mask(
                                                                               (int)face_mask[(size_t)e],
                                                                               L, xi, yi, zi),
                                                             sscvfem_micro_face_mask(
                                                                     !natural_mask ? 0
                                                                         : (int)natural_mask[(size_t)e],
                                                                     L, xi, yi, zi),
                                                             sscvfem_bd(bc_p, bc_tx, bc_ty, bc_tz, pressure_mask, traction_mask, e, L, xi, yi, zi));
                                for (int a = 0; a < 8; ++a) {
                                    if constexpr (uu)
                                        for (int cc = 0; cc < 3; ++cc) r[a * 4 + cc] += rb[a * 4 + cc];
                                    if constexpr (pu) r[a * 4 + 3] += rb[a * 4 + 3];
                                }
                            }
                            if constexpr (up || pp) {
                                for (int k = 0; k < CVFEM_HEX8_N_DOF; ++k) rb[k] = scalar_t(0);
                                boundary_scs_add_jacobian_action<false>(rho, mu, mg.adj, mg.det, box_lx, box_ly, box_lz, x, y, z,
                                                                 ux, uy, uz, zero8, zero8, zero8, q, rb,
                                                             !face_mask
                                                                     ? -1
                                                                     : sscvfem_micro_face_mask(
                                                                               (int)face_mask[(size_t)e],
                                                                               L, xi, yi, zi),
                                                             sscvfem_micro_face_mask(
                                                                     !natural_mask ? 0
                                                                         : (int)natural_mask[(size_t)e],
                                                                     L, xi, yi, zi),
                                                             sscvfem_bd(bc_p, bc_tx, bc_ty, bc_tz, pressure_mask, traction_mask, e, L, xi, yi, zi));
                                for (int a = 0; a < 8; ++a) {
                                    if constexpr (up)
                                        for (int cc = 0; cc < 3; ++cc) r[a * 4 + cc] += rb[a * 4 + cc];
                                    if constexpr (pp) r[a * 4 + 3] += rb[a * 4 + 3];
                                }
                            }
                        }

                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            for (int c = 0; c < CVFEM_HEX8_N_FIELDS; ++c) lout[(size_t)l * CVFEM_HEX8_N_FIELDS + c] += r[a * 4 + c];
                        }
                    }
                }
            }

            // Scatter only the rows written: a continuity-row block touches one component
            // of four, and the atomics are the expensive half of the scatter.
            // The two-pass scatter moves all four components. The rows this block
            // selection does not write are zero in lout, so they contribute nothing, and
            // the saving the component-wise atomics bought no longer applies once the
            // scatter is a plain write.
            if (slot)
                sscvfem_scatter_element(slot, const_cast<scalar_t *>(stage), nxe, e, lg, lout, jv);
            else
                for (int a = 0; a < nxe; ++a) {
                    const idx_t g = lg[(size_t)a];
                    if constexpr (uu || up)
                        for (int c = 0; c < 3; ++c)
                            atomic_add(jv + (ptrdiff_t)g * CVFEM_HEX8_N_FIELDS + c, 0, lout[(size_t)a * CVFEM_HEX8_N_FIELDS + c]);
                    if constexpr (pu || pp)
                        atomic_add(jv + (ptrdiff_t)g * CVFEM_HEX8_N_FIELDS + 3, 0, lout[(size_t)a * CVFEM_HEX8_N_FIELDS + 3]);
                }
        }
    }

    if (slot) sscvfem_reduce_shared(red_idx, red_ptr, shared_node, const_cast<scalar_t *>(stage), n_shared, jv);
}

// Subtract the body force from the momentum rows of an interleaved residual. Mirrors
// apply_body_force in cvfem_hex8_ns_core.hpp; see the sign argument there.
inline void sscvfem_apply_body_force_sweep(
        // The range this call is to cover. DESIGN.md: the threading is abstract outside the
        // sweep and what arrives is a range, so the sweep owns no parallel region.
        const cvfem_range r,
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const scalar_t *const SFEM_RESTRICT fx,
        const scalar_t *const SFEM_RESTRICT fy,
        const scalar_t *const SFEM_RESTRICT fz,
        const scalar_t *const SFEM_RESTRICT node_vol, scalar_t *const SFEM_RESTRICT res) {
    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const scalar_t v = node_vol[(size_t)i];
        res[i * CVFEM_HEX8_N_FIELDS + 0] -= fx[(size_t)i] * v;
        res[i * CVFEM_HEX8_N_FIELDS + 1] -= fy[(size_t)i] * v;
        res[i * CVFEM_HEX8_N_FIELDS + 2] -= fz[(size_t)i] * v;
    }
}

// The transient term on an interleaved residual. Mirrors apply_transient in
// cvfem_hex8_ns_core.hpp -- same coefficients, same lumped control volume, same reason for
// being a post-pass rather than a term inside the macro-element sweeps.
inline void sscvfem_apply_transient_sweep(
        // The range this call is to cover. DESIGN.md: the threading is abstract outside the
        // sweep and what arrives is a range, so the sweep owns no parallel region.
        const cvfem_range r,
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const scalar_t dt,
        const scalar_t *const SFEM_RESTRICT node_vol,
        const scalar_t *const SFEM_RESTRICT u_prev,
        const scalar_t *const SFEM_RESTRICT u_prev2,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz, const scalar_t rho, const BdfCoeffs c,
                                          scalar_t *const SFEM_RESTRICT res) {
    CVFEM_TRACE_SCOPE("sscvfem::apply_transient_sweep");
    const bool      two = c.order >= 2;
    const scalar_t  a0 = c.a0, a1 = c.a1, a2 = c.a2;
    const scalar_t inv = scalar_t(1) / dt;
    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const scalar_t w   = rho * node_vol[(size_t)i] * inv;
        const size_t   k   = (size_t)i * 3;
        const scalar_t u[3] = {ux[(size_t)i], uy[(size_t)i], uz[(size_t)i]};
        for (int c = 0; c < 3; ++c) {
            const scalar_t prev2 = two ? u_prev2[k + (size_t)c] : scalar_t(0);
            res[i * CVFEM_HEX8_N_FIELDS + c] += w * (a0 * u[c] + a1 * u_prev[k + (size_t)c] + a2 * prev2);
        }
    }
}

inline void sscvfem_apply_transient_action_sweep(
        // The range this call is to cover. DESIGN.md: the threading is abstract outside the
        // sweep and what arrives is a range, so the sweep owns no parallel region.
        const cvfem_range r,
        const scalar_t *const SFEM_RESTRICT node_vol,
        const scalar_t transient_w, const scalar_t rho,
                                                 const scalar_t *const SFEM_RESTRICT dir,
                                                 scalar_t *const SFEM_RESTRICT       jv) {
    CVFEM_TRACE_SCOPE("sscvfem::apply_transient_action_sweep");
    const scalar_t a = transient_w;
    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const scalar_t w = a * node_vol[(size_t)i];
        for (int c = 0; c < 3; ++c) jv[i * CVFEM_HEX8_N_FIELDS + c] += w * dir[i * CVFEM_HEX8_N_FIELDS + c];
    }
}

inline SFEM_NOINLINE void sscvfem_residual_naive_sweep(
        // The range this call is to cover. DESIGN.md: the threading is abstract outside the
        // sweep and what arrives is a range, so the sweep owns no parallel region.
        const cvfem_range r,
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const scalar_t box_lx,
        const scalar_t box_ly,
        const scalar_t box_lz,
        const Hex8PecletConfig<scalar_t> peclet,
        idx_t **const SFEM_RESTRICT elems,
        const int level,
        const uint8_t *const SFEM_RESTRICT macro_curved,
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
    CVFEM_TRACE_SCOPE("sscvfem::residual_naive_sweep");
    // The destination arrives ZEROED. It used to be zeroed here, which was correct while this
    // sweep owned its parallel region and ran once; driven by a range it runs once per thread,
    // and every thread would re-zero the whole array -- over the contributions the others had
    // already accumulated. The launcher zeroes it before the region opens.

    const int L = level;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    for (ptrdiff_t e = r.begin; e < r.end; ++e) {
        // The geometry every micro cell of this macro element uses, as the hoisted
        // variants use it; see sscvfem_hoisted_cell. Real positions stay per cell.
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
        const bool curved_e = sscvfem_macro_curved(macro_curved, e);
        for (int zi = 0; zi < L; ++zi) {
            for (int yi = 0; yi < L; ++yi) {
                for (int xi = 0; xi < L; ++xi) {
                    const int    base = sscvfem_lidx(L, xi, yi, zi);
                    idx_t g[8];
                    scalar_t     x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], pgx[8], pgy[8], pgz[8];
                    scalar_t     r[CVFEM_HEX8_N_DOF];
                    for (int a = 0; a < 8; ++a) {
                        g[a]   = elems[base + off[a]][e];
                        x[a]   = (scalar_t)points[0][g[a]];
                        y[a]   = (scalar_t)points[1][g[a]];
                        z[a]   = (scalar_t)points[2][g[a]];
                        ux[a]  = ux_src[(size_t)g[a]];
                        uy[a]  = uy_src[(size_t)g[a]];
                        uz[a]  = uz_src[(size_t)g[a]];
                        p[a]   = pres[(size_t)g[a]];
                        pgx[a] = pgx_src[(size_t)g[a]];
                        pgy[a] = pgy_src[(size_t)g[a]];
                        pgz[a] = pgz_src[(size_t)g[a]];
                    }
                    // A curved macro element: this cell's own geometry, not the hoisted one, selected
                    // through pointers so the hoisted corners stay loop-invariant.
                    const scalar_t *const gx = curved_e ? x : hx;
                    const scalar_t *const gy = curved_e ? y : hy;
                    const scalar_t *const gz = curved_e ? z : hz;
                    const Hex8RhieChow rc{gx,      gy, gz,   pgx, pgy, pgz, rcfg.scale, nullptr, nullptr,
                                          nullptr, ux, uy, uz,  rcfg.tau};
                    scalar_t           adj[9], det;
                    sscvfem_micro_geom(gx, gy, gz, adj, &det);
                    cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, adj, det, ux, uy, uz, p, r, rc,
                                                         upwind_eps,
                                                         (const scalar_t *)nullptr,
                                                         (const scalar_t *)nullptr,
                                                         (const scalar_t *)nullptr,
                                                         (const scalar_t *)nullptr, 0, scalar_t(0),
                                                         nullptr, peclet);
                    boundary_scs_add_residual<false>(rho, mu, adj, det, box_lx, box_ly, box_lz, x, y, z, ux, uy, uz, p, r);
                    for (int a = 0; a < 8; ++a)
                        for (int c = 0; c < CVFEM_HEX8_N_FIELDS; ++c)
                            atomic_add(res + (ptrdiff_t)g[a] * CVFEM_HEX8_N_FIELDS + c, 0, r[a * 4 + c]);
                }
            }
        }
    }
}

// ho_override: -1 takes SFEM_CONV_HO from the environment as before, 0 or 1 forces it. The
// freezing path below needs the same residual evaluated both ways on one state, and it is the
// only caller that passes it; every existing call keeps the default and the existing behaviour.
// The residual's element pass. Everything the options decide -- which convection scheme, which
// limiter, whether the correction is frozen, whether a nodal velocity gradient exists at all --
// has been decided by the launcher below and reaches here as data.
inline SFEM_NOINLINE void sscvfem_residual_sweep(
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
        const uint8_t *const SFEM_RESTRICT macro_curved,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const uint8_t *const SFEM_RESTRICT natural_mask,
        const uint8_t *const SFEM_RESTRICT pressure_mask,
        const uint8_t *const SFEM_RESTRICT traction_mask,
        const ptrdiff_t nmacro,
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
        const Hex8RcConfig rcfg,
        
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const ptrdiff_t *const SFEM_RESTRICT red_idx,
        const ptrdiff_t *const SFEM_RESTRICT red_ptr,
        const idx_t *const SFEM_RESTRICT shared_node,
        const int *const SFEM_RESTRICT slot,
        scalar_t *const SFEM_RESTRICT stage,
        const ptrdiff_t n_shared, const scalar_t rho,
                                                 const scalar_t                mu,
                                                 scalar_t *const SFEM_RESTRICT res) {
    CVFEM_TRACE_SCOPE("sscvfem::residual_sweep");
    const int L   = level;
    const int nxe = nxe_src;
    int       off[8];
    sscvfem_corner_offsets(L, off);


#pragma omp parallel
    {
        // Per-thread scratch from the kernels' own arena, not std::vector locals: slots 5/6 are the semi-structured element sweeps',
        // shared between them because only one is live inside a parallel region and
        // they all want the same macro-element size.
        scalar_t *const SFEM_RESTRICT _arena5 = thread_scratch<scalar_t>(5, ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + (conv_ho ? (size_t)nxe * 9 : 0) + ((size_t)nxe * CVFEM_HEX8_N_FIELDS));
        scalar_t *const SFEM_RESTRICT lx = _arena5;
        scalar_t *const SFEM_RESTRICT ly = _arena5 + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lz = _arena5 + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lux = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT luy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT luz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lp = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgx = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lug = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lout = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + (conv_ho ? (size_t)nxe * 9 : 0);
        idx_t *const SFEM_RESTRICT _arena6 = thread_scratch<idx_t>(6, ((size_t)nxe));
        idx_t *const SFEM_RESTRICT lg = _arena6;

#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < nmacro; ++e) {
            for (int a = 0; a < nxe; ++a) {
                const idx_t g = elems[a][e];
                lg[(size_t)a]        = g;
                lx[(size_t)a]        = (scalar_t)points[0][g];
                ly[(size_t)a]        = (scalar_t)points[1][g];
                lz[(size_t)a]        = (scalar_t)points[2][g];
                lux[(size_t)a]       = ux_src[(size_t)g];
                luy[(size_t)a]       = uy_src[(size_t)g];
                luz[(size_t)a]       = uz_src[(size_t)g];
                lp[(size_t)a]        = pres[(size_t)g];
                lpgx[(size_t)a]      = pgx_src[(size_t)g];
                lpgy[(size_t)a]      = pgy_src[(size_t)g];
                lpgz[(size_t)a]      = pgz_src[(size_t)g];
                if (conv_ho)
                    for (int k = 0; k < 9; ++k) lug[(size_t)a * 9 + (size_t)k] = ugrad_f[(size_t)g * 9 + (size_t)k];
            }
            std::fill(lout, lout + ((size_t)nxe * CVFEM_HEX8_N_FIELDS), scalar_t(0));

            // Per macro element, and the curved branch below reads the same one: a call per
            // micro cell there took this unit past the point where GCC inlines sscvfem_rc_config,
            // which then became a call in every cell of the block diagonal, 9% slower on boxes.
            const Hex8RcConfig rc_macro = rcfg;
            SSMacroGeom mg;
            // The hoisted cell's corners outlive the block below, because the Rhie-Chow term
            // takes its node distances from them -- as the Jacobian action takes them from
            // mg.dvec, which is built from the same corners. Each micro cell's own coordinates
            // agree with those only on an affine macro element. On a curved one they did not,
            // and the residual and its Jacobian action disagreed in every continuity row:
            // measured on the FDA nozzle by SFEM_FD_CHECK, 6.0e-02 at macro core 2 / L 2,
            // 3.2e-02 at L 4 and 1.2e-02 at macro core 4 / L 2, and exact with Rhie-Chow off.
            scalar_t ex[8], ey[8], ez[8];
            {
                int ext[8];
                sscvfem_macro_corner_offsets(L, ext);
                for (int a = 0; a < 8; ++a) {
                    const int l = ext[a];
                    ex[a]       = lx[(size_t)l];
                    ey[a]       = ly[(size_t)l];
                    ez[a]       = lz[(size_t)l];
                }
                sscvfem_hoisted_cell(ex, ey, ez, L, ex, ey, ez);
                sscvfem_macro_geom(ex, ey, ez, rho, mu, rc_macro.scale, rc_macro.tau, mg);
            }

            const bool curved_e = sscvfem_macro_curved(macro_curved, e);
            for (int zi = 0; zi < L; ++zi) {
                for (int yi = 0; yi < L; ++yi) {
                    for (int xi = 0; xi < L; ++xi) {
                        const int base = sscvfem_lidx(L, xi, yi, zi);
                        scalar_t  x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], pgx[8], pgy[8], pgz[8];
                        scalar_t g8[CVFEM_HEX8_N_NODES * 9];
                        scalar_t  r[CVFEM_HEX8_N_DOF];
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            x[a]        = lx[(size_t)l];
                            y[a]        = ly[(size_t)l];
                            z[a]        = lz[(size_t)l];
                            ux[a]       = lux[(size_t)l];
                            uy[a]       = luy[(size_t)l];
                            uz[a]       = luz[(size_t)l];
                            p[a]        = lp[(size_t)l];
                            pgx[a]      = lpgx[(size_t)l];
                            pgy[a]      = lpgy[(size_t)l];
                            pgz[a]      = lpgz[(size_t)l];
                            if (conv_ho)
                                for (int k = 0; k < 9; ++k) g8[a * 9 + k] = lug[(size_t)l * 9 + (size_t)k];
                        }
                        // A curved macro element: this cell's own geometry, not the hoisted one.
                        if (curved_e) {
                            sscvfem_macro_geom_cell(x, y, z, rho, mu, rc_macro, mg);
                            std::copy(x, x + 8, ex);
                            std::copy(y, y + 8, ey);
                            std::copy(z, z + 8, ez);
                        }
                        const Hex8RhieChow rc{ex,      ey, ez, pgx, pgy, pgz, rcfg.scale, nullptr, nullptr,
                                              nullptr, ux, uy, uz,  rcfg.tau};
                        // Deferred-correction convection, on the path the production solver
                        // actually runs: FGMRES preconditioned by multigrid needs this lattice,
                        // so a correction that existed only on the flat mesh could not be used
                        // for anything at scale. Null when off, which is the arithmetic this
                        // call did before.
                        const bool ho = conv_ho != 0;
                        cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, mg.adj, mg.det, ux, uy, uz, p, r,
                                                             rc, upwind_eps,
                                                             ho ? g8 : nullptr,
                                                             ho ? x : nullptr, ho ? y : nullptr,
                                                             ho ? z : nullptr, conv_limiter,
                                                             conv_venkat_c, limiter_stats,
                                                             peclet);
                        boundary_scs_add_residual<false>(rho, mu, mg.adj, mg.det, box_lx, box_ly, box_lz, x, y, z,
                                                  ux, uy, uz, p, r,
                                                  !face_mask
                                                          ? -1
                                                          : sscvfem_micro_face_mask(
                                                                    (int)face_mask[(size_t)e],
                                                                    L, xi, yi, zi),
                                                  sscvfem_micro_face_mask(
                                                          !natural_mask ? 0
                                                              : (int)natural_mask[(size_t)e],
                                                          L, xi, yi, zi),
                                                  sscvfem_bd(bc_p, bc_tx, bc_ty, bc_tz, pressure_mask, traction_mask, e, L, xi, yi, zi));
                        for (int a = 0; a < 8; ++a) {
                            const int l = base + off[a];
                            for (int c = 0; c < CVFEM_HEX8_N_FIELDS; ++c) lout[(size_t)l * CVFEM_HEX8_N_FIELDS + c] += r[a * 4 + c];
                        }
                    }
                }
            }

            if (slot)
                sscvfem_scatter_element(slot, const_cast<scalar_t *>(stage), nxe, e, lg, lout, res);
            else
                for (int a = 0; a < nxe; ++a) {
                    const idx_t g = lg[(size_t)a];
                    for (int c = 0; c < CVFEM_HEX8_N_FIELDS; ++c)
                        atomic_add(res + (ptrdiff_t)g * CVFEM_HEX8_N_FIELDS + c, 0, lout[(size_t)a * CVFEM_HEX8_N_FIELDS + c]);
                }
        }
    }

    if (slot) sscvfem_reduce_shared(red_idx, red_ptr, shared_node, const_cast<scalar_t *>(stage), n_shared, res);
}

// ---------------------------------------------------------------------------
// Block diagonal: the 4x4 block per node, which is what a block-Jacobi smoother inverts.
//
// The multigrid path needs one of these at every level, so it is not the once-per-Newton
// cost it looked like when only a single-level solve existed.
//
// Unlike the flat path, this can use the slot mask. cvfem_hex8_ns_upwind_jacobian_add_slots
// writes exclusively through cvfem_hex8_bsr_acc, which drops a negative slot, so passing
// -1 everywhere off the diagonal makes the full element kernel produce the block diagonal
// with none of the off-diagonal write traffic. The flat assemble_block_diag cannot do that:
// its affine path runs the SymPy kernel, whose 768 writes go straight to values[...] with
// no guard, so a negative slot there is an out-of-bounds write and it has to assemble the
// whole element into a 64-block scratch and throw away seven eighths of it.
//
// Both variants below use the same kernel as the apply, so the diagonal and the operator
// cannot drift apart.

// Control: the flat gather, one masked element assembly per micro-element, atomics to a
// node-indexed destination.
inline SFEM_NOINLINE void sscvfem_block_diag_naive_sweep(
        // The range this call is to cover. DESIGN.md: the threading is abstract outside the
        // sweep and what arrives is a range, so the sweep owns no parallel region.
        const cvfem_range r,
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const scalar_t box_lx,
        const scalar_t box_ly,
        const scalar_t box_lz,
        idx_t **const SFEM_RESTRICT elems,
        const int level,
        const uint8_t *const SFEM_RESTRICT macro_curved,
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
    CVFEM_TRACE_SCOPE("sscvfem::block_diag_naive_sweep");

    const int L = level;
    int       off[8];
    sscvfem_corner_offsets(L, off);

    for (ptrdiff_t e = r.begin; e < r.end; ++e) {
        // The geometry every micro cell of this macro element uses, as the hoisted
        // variants use it; see sscvfem_hoisted_cell. Real positions stay per cell.
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
        const bool curved_e = sscvfem_macro_curved(macro_curved, e);
        for (int zi = 0; zi < L; ++zi) {
            for (int yi = 0; yi < L; ++yi) {
                for (int xi = 0; xi < L; ++xi) {
                    const int base = sscvfem_lidx(L, xi, yi, zi);

                    idx_t g[8];
                    scalar_t     x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], pgx[8], pgy[8], pgz[8];
                    for (int a = 0; a < 8; ++a) {
                        g[a]   = elems[base + off[a]][e];
                        x[a]   = (scalar_t)points[0][g[a]];
                        y[a]   = (scalar_t)points[1][g[a]];
                        z[a]   = (scalar_t)points[2][g[a]];
                        ux[a]  = ux_src[(size_t)g[a]];
                        uy[a]  = uy_src[(size_t)g[a]];
                        uz[a]  = uz_src[(size_t)g[a]];
                        p[a]   = pres[(size_t)g[a]];
                        pgx[a] = pgx_src[(size_t)g[a]];
                        pgy[a] = pgy_src[(size_t)g[a]];
                        pgz[a] = pgz_src[(size_t)g[a]];
                    }
                    // A curved macro element: this cell's own geometry, not the hoisted one, selected
                    // through pointers so the hoisted corners stay loop-invariant.
                    const scalar_t *const gx = curved_e ? x : hx;
                    const scalar_t *const gy = curved_e ? y : hy;
                    const scalar_t *const gz = curved_e ? z : hz;

                    // Diagonal slots address the destination by node; everything else is
                    // dropped by the guard in cvfem_hex8_bsr_acc.
                    // count_t, not ptrdiff_t: boundary_scs_add_jacobian takes count_t
                    // slots. It is signed, so -1 still means "drop this block".
                    count_t sl[64];
                    for (int a = 0; a < 8; ++a) {
                        for (int b = 0; b < 8; ++b) sl[a * 8 + b] = -1;
                        sl[a * 8 + a] = (count_t)g[a];
                    }

                    const Hex8RhieChow rc{gx,      gy, gz,   pgx, pgy, pgz, rcfg.scale, nullptr, nullptr,
                                          nullptr, ux, uy, uz,  rcfg.tau};
                    scalar_t           adj[9], det;
                    sscvfem_micro_geom(gx, gy, gz, adj, &det);
                    cvfem_hex8_ns_upwind_jacobian_add_slots<true>(rho, mu, adj, det, ux, uy, uz, sl, out, rc, p);
                    boundary_scs_add_jacobian<true, false>(rho, mu, adj, det, box_lx, box_ly, box_lz, x, y, z, ux, uy, uz, sl, out);
                }
            }
        }
    }
}

// The default: gather the macro-element once, accumulate into a macro-local destination
// addressed by local node so the element assembly needs no atomics at all, and scatter
// once per macro node at the end. Geometry is lifted out of the loop as in the apply.
// One micro cell of the block diagonal: gather, Rhie-Chow, element Jacobian, boundary closure.
//
// `hadj`, `hdet` and (hx, hy, hz) are the hoisted geometry and the corners it was built from.
// nullptr asks for this cell's own geometry instead, which is what a curved macro element needs
// (see sscvfem_macro_curved). The affine sweep passes local arrays, so once this is inlined the
// choice folds away and that loop is the loop it always was: a runtime selection inside the
// sweep, by overwriting or by pointer, measured 8% slower on affine meshes, and compiling the
// sweep twice from one generic body cost 25% and slowed unrelated kernels in the same unit.
static SFEM_INLINE void sscvfem_block_diag_cell(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const scalar_t box_lx,
        const scalar_t box_ly,
        const scalar_t box_lz,
        const scalar_t bc_p,
        const scalar_t bc_tx,
        const scalar_t bc_ty,
        const scalar_t bc_tz,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const uint8_t *const SFEM_RESTRICT natural_mask,
        const uint8_t *const SFEM_RESTRICT pressure_mask,
        const uint8_t *const SFEM_RESTRICT traction_mask,
        const Hex8RcConfig rcfg, const scalar_t rho, const scalar_t mu,
                                                const ptrdiff_t e, const int L, const int xi, const int yi,
                                                const int zi, const int off[8], const scalar_t *const lx,
                                                const scalar_t *const ly, const scalar_t *const lz,
                                                const scalar_t *const lux, const scalar_t *const luy,
                                                const scalar_t *const luz, const scalar_t *const lp,
                                                const scalar_t *const lpgx, const scalar_t *const lpgy,
                                                const scalar_t *const lpgz, const scalar_t *const hadj,
                                                const scalar_t hdet, const scalar_t *const hx,
                                                const scalar_t *const hy, const scalar_t *const hz,
                                                scalar_t *const lout) {
    const int base = sscvfem_lidx(L, xi, yi, zi);

    scalar_t x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8], pgx[8], pgy[8], pgz[8];
    count_t sl[64];
    for (int a = 0; a < 8; ++a) {
        const int l = base + off[a];
        x[a]        = lx[(size_t)l];
        y[a]        = ly[(size_t)l];
        z[a]        = lz[(size_t)l];
        ux[a]       = lux[(size_t)l];
        uy[a]       = luy[(size_t)l];
        uz[a]       = luz[(size_t)l];
        p[a]        = lp[(size_t)l];
        pgx[a]      = lpgx[(size_t)l];
        pgy[a]      = lpgy[(size_t)l];
        pgz[a]      = lpgz[(size_t)l];
        for (int b = 0; b < 8; ++b) sl[a * 8 + b] = -1;
    }
    // Local node index: the destination is this macro-element's own
    // buffer, so no thread can be writing the same entry.
    for (int a = 0; a < 8; ++a) sl[a * 8 + a] = (count_t)(base + off[a]);

    scalar_t              cadj[9], cdet = 0;
    const scalar_t       *adj = hadj;
    scalar_t              det = hdet;
    const scalar_t       *rx = hx, *ry = hy, *rz = hz;
    if (!hadj) {
        sscvfem_micro_geom(x, y, z, cadj, &cdet);
        adj = cadj;
        det = cdet;
        rx  = x;
        ry  = y;
        rz  = z;
    }

    const Hex8RhieChow rc{rx,      ry,  rz,  pgx, pgy, pgz, rcfg.scale, nullptr, nullptr,
                          nullptr, ux,  uy,  uz,  rcfg.tau};
    cvfem_hex8_ns_upwind_jacobian_add_slots<false>(rho, mu, adj, det, ux, uy, uz, sl, lout, rc, p);
    boundary_scs_add_jacobian<false, false>(rho, mu, adj, det, box_lx, box_ly, box_lz, x, y, z, ux, uy, uz, sl, lout,
                                     !face_mask
                                             ? -1
                                             : sscvfem_micro_face_mask((int)face_mask[(size_t)e], L,
                                                                       xi, yi, zi),
                                     sscvfem_micro_face_mask(!natural_mask
                                                                     ? 0
                                                                     : (int)natural_mask[(size_t)e],
                                                             L, xi, yi, zi),
                                     sscvfem_bd(bc_p, bc_tx, bc_ty, bc_tz, pressure_mask, traction_mask, e, L, xi, yi, zi));
}

// The block diagonal's micro-cell sweep for a curved macro element, every cell with its own
// geometry. Out of line so the affine sweep in sscvfem_block_diag carries none of it.
// sscvfem_block_diag is flattened because this second call site of the cell kernel costs the
// affine sweep its inlining otherwise: gcc outlines the Jacobian slot and boundary kernels once
// they have two callers, and the per-micro-cell call measured 3% slower on the box.
static SFEM_NOINLINE void sscvfem_block_diag_curved_macro(
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const scalar_t box_lx,
        const scalar_t box_ly,
        const scalar_t box_lz,
        const scalar_t bc_p,
        const scalar_t bc_tx,
        const scalar_t bc_ty,
        const scalar_t bc_tz,
        const int level,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const uint8_t *const SFEM_RESTRICT natural_mask,
        const uint8_t *const SFEM_RESTRICT pressure_mask,
        const uint8_t *const SFEM_RESTRICT traction_mask,
        const Hex8RcConfig rcfg, const scalar_t rho,
                                                          const scalar_t mu, const ptrdiff_t e,
                                                          const int off[8], const scalar_t *const lx,
                                                          const scalar_t *const ly, const scalar_t *const lz,
                                                          const scalar_t *const lux, const scalar_t *const luy,
                                                          const scalar_t *const luz, const scalar_t *const lp,
                                                          const scalar_t *const lpgx, const scalar_t *const lpgy,
                                                          const scalar_t *const lpgz, scalar_t *const lout) {
    const int L = level;
    for (int zi = 0; zi < L; ++zi)
        for (int yi = 0; yi < L; ++yi)
            for (int xi = 0; xi < L; ++xi)
                sscvfem_block_diag_cell(box_lx, box_ly, box_lz, bc_p, bc_tx, bc_ty, bc_tz, face_mask, natural_mask, pressure_mask, traction_mask, rcfg, rho, mu, e, L, xi, yi, zi, off, lx, ly, lz, lux, luy, luz, lp, lpgx,
                                        lpgy, lpgz, nullptr, scalar_t(0), nullptr, nullptr, nullptr, lout);
}

// The transient term's diagonal: rho V a0 / dt on each velocity component, nothing on
// pressure. A post-pass over nodes rather than part of the macro-element sweep, for the same
// reason sscvfem_apply_transient is one, so the two stay consistent by construction.
inline void sscvfem_block_diag_transient(
        // The range this call is to cover. DESIGN.md: the threading is abstract outside the
        // sweep and what arrives is a range, so the sweep owns no parallel region.
        const cvfem_range r,
        const scalar_t *const SFEM_RESTRICT node_vol,
        const scalar_t transient_w, const scalar_t rho,
                                         scalar_t *const SFEM_RESTRICT out) {
    const scalar_t a = transient_w;
    if (a == scalar_t(0)) return;
    for (ptrdiff_t i = r.begin; i < r.end; ++i) {
        const scalar_t w = a * node_vol[(size_t)i];
        for (int c = 0; c < 3; ++c) out[(size_t)i * 16 + (size_t)c * 4 + (size_t)c] += w;
    }
}

inline SFEM_NOINLINE __attribute__((flatten)) void sscvfem_block_diag_sweep(
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
        const uint8_t *const SFEM_RESTRICT macro_curved,
        const uint8_t *const SFEM_RESTRICT face_mask,
        const uint8_t *const SFEM_RESTRICT natural_mask,
        const uint8_t *const SFEM_RESTRICT pressure_mask,
        const uint8_t *const SFEM_RESTRICT traction_mask,
        const ptrdiff_t nmacro,
        const int nxe_src,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const Hex8RcConfig rcfg,
        
        // The staging object is gone; what this sweep reads out of it is what it takes.
        const ptrdiff_t *const SFEM_RESTRICT red_idx,
        const ptrdiff_t *const SFEM_RESTRICT red_ptr,
        const idx_t *const SFEM_RESTRICT shared_node,
        const int *const SFEM_RESTRICT slot,
        scalar_t *const SFEM_RESTRICT stage16,
        const ptrdiff_t n_shared, const scalar_t rho, const scalar_t mu,
                                             scalar_t *const SFEM_RESTRICT out) {
    CVFEM_TRACE_SCOPE("sscvfem::block_diag_sweep");

    const int L   = level;
    const int nxe = nxe_src;
    int       off[8];
    sscvfem_corner_offsets(L, off);


#pragma omp parallel
    {
        // Per-thread scratch from the kernels' own arena, not std::vector locals: slots 5/6 are the semi-structured element sweeps',
        // shared between them because only one is live inside a parallel region and
        // they all want the same macro-element size.
        scalar_t *const SFEM_RESTRICT _arena5 = thread_scratch<scalar_t>(5, ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe * 16));
        scalar_t *const SFEM_RESTRICT lx = _arena5;
        scalar_t *const SFEM_RESTRICT ly = _arena5 + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lz = _arena5 + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lux = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT luy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT luz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lp = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgx = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgy = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lpgz = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        scalar_t *const SFEM_RESTRICT lout = _arena5 + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe) + ((size_t)nxe);
        idx_t *const SFEM_RESTRICT _arena6 = thread_scratch<idx_t>(6, ((size_t)nxe));
        idx_t *const SFEM_RESTRICT lg = _arena6;

#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < nmacro; ++e) {
            for (int a = 0; a < nxe; ++a) {
                const idx_t g = elems[a][e];
                lg[(size_t)a]        = g;
                lx[(size_t)a]        = (scalar_t)points[0][g];
                ly[(size_t)a]        = (scalar_t)points[1][g];
                lz[(size_t)a]        = (scalar_t)points[2][g];
                lux[(size_t)a]       = ux[(size_t)g];
                luy[(size_t)a]       = uy[(size_t)g];
                luz[(size_t)a]       = uz[(size_t)g];
                lp[(size_t)a]        = pres[(size_t)g];
                lpgx[(size_t)a]      = pgx[(size_t)g];
                lpgy[(size_t)a]      = pgy[(size_t)g];
                lpgz[(size_t)a]      = pgz[(size_t)g];
            }
            std::fill(lout, lout + ((size_t)nxe * 16), scalar_t(0));

            // Micro-cell 0's corners, hoisted: the geometry AND the coordinates the
            // Rhie-Chow term differences.
            //
            // The lattice inside a macro element is uniform, so every micro-cell is congruent
            // to cell 0 and one adjugate serves all of them -- that is what the action does.
            // This used to hoist the adjugate but then hand the Rhie-Chow struct each cell's
            // OWN coordinates, and the two agree only to the precision the node positions are
            // stored in. geom_t is float32, so the block diagonal disagreed with the
            // action it is supposed to be the diagonal of by 4.23e-08 -- eight orders above
            // round-off, and invisible until the q-independent consistency gate looked.
            //
            // Only DIFFERENCES of these are taken (d = x_j - x_i), so cell 0's coordinates are
            // exact for the purpose, not an approximation. The boundary closure below still
            // gets each cell's real position, because it tests where the cell actually is.
            scalar_t madj[9], mdet;
            scalar_t c0x[8], c0y[8], c0z[8];
            {
                int ext[8];
                sscvfem_macro_corner_offsets(L, ext);
                for (int a = 0; a < 8; ++a) {
                    const int l = ext[a];
                    c0x[a]      = lx[(size_t)l];
                    c0y[a]      = ly[(size_t)l];
                    c0z[a]      = lz[(size_t)l];
                }
                sscvfem_hoisted_cell(c0x, c0y, c0z, L, c0x, c0y, c0z);
                sscvfem_micro_geom(c0x, c0y, c0z, madj, &mdet);
            }

            if (sscvfem_macro_curved(macro_curved, e)) {
                sscvfem_block_diag_curved_macro(box_lx, box_ly, box_lz, bc_p, bc_tx, bc_ty, bc_tz, level, face_mask, natural_mask, pressure_mask, traction_mask, rcfg, rho, mu, e, off, lx, ly, lz, lux,
                                                luy, luz, lp, lpgx, lpgy,
                                                lpgz, lout);
            } else {
                for (int zi = 0; zi < L; ++zi)
                    for (int yi = 0; yi < L; ++yi)
                        for (int xi = 0; xi < L; ++xi)
                            sscvfem_block_diag_cell(box_lx, box_ly, box_lz, bc_p, bc_tx, bc_ty, bc_tz, face_mask, natural_mask, pressure_mask, traction_mask, rcfg, rho, mu, e, L, xi, yi, zi, off, lx, ly,
                                                    lz, lux, luy, luz, lp,
                                                    lpgx, lpgy, lpgz, madj, mdet, c0x, c0y,
                                                    c0z, lout);
            }
            if (slot)
                sscvfem_scatter_element_w<16>(slot, nxe, e, lg, lout, out,
                                              const_cast<scalar_t *>(stage16));
            else
                for (int a = 0; a < nxe; ++a) {
                    const idx_t g = lg[(size_t)a];
                    for (int k = 0; k < 16; ++k)
                        atomic_add(out + (ptrdiff_t)g * 16 + k, 0, lout[(size_t)a * 16 + k]);
                }
        }
    }

    if (slot) sscvfem_reduce_shared_w<16>(red_idx, red_ptr, shared_node, n_shared, out, stage16);

}
