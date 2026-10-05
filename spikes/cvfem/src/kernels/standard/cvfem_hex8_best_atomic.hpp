#ifndef CVFEM_HEX8_BEST_ATOMIC_HPP
#define CVFEM_HEX8_BEST_ATOMIC_HPP

// Atomic layout: a flat parallel sweep over elements that writes into the global
// residual / matrix with #pragma omp atomic on every entry. No mesh partitioning
// and no scratch, which makes it the simplest and the reference for correctness,
// but assembly pays ~1024 atomic read-modify-writes per element.

#include "kernels/microkernels/hex8/affine/cvfem_hex8_ns_upwind_affine.hpp"
#include "kernels/microkernels/hex8/isoparametric/cvfem_hex8_ns_upwind_isoparam.hpp"
#include "kernels/cvfem_scatter.hpp"
#include "kernels/cvfem_phases.hpp"
// Hex8ExtraScratch and the element gathers the shared assembly bodies below use.
#include "kernels/cvfem_hex8_element_gather.hpp"
#include "kernels/microkernels/hex8/cvfem_hex8_boundary_scs.hpp"
// NOTHING FROM OUTSIDE THIS DIRECTORY. The include that used to sit here --
// frontend/staging/cvfem_hex8_best_common.hpp, the bench's staging header -- is gone, because after the 27
// sweeps in this file stopped taking MeshData and BSR4 the only names left were the ones the
// includer supplies by contract (scalar_t, idx_t, count_t, geom_t and the SFEM_ spellings) and
// MIN, which now lives beside the scatter helpers that use it.

// ---------------------------------------------------------------------------
// Block diagonal only, for the block-Jacobi preconditioner.
//
// When the preconditioner is all that is being rebuilt, there is no reason to touch
// the off-diagonal blocks. Passing a slot array that is -1 everywhere except the
// diagonal makes the existing element kernel drop those writes (cvfem_hex8_bsr_acc
// returns on a negative slot), so this reuses the full element kernel rather than
// duplicating it -- the same trick the CUDA path uses.
//
// The destination is indexed by node, 16 doubles per node, not by BSR block.

// The masked slot array for one element: -1 everywhere but the diagonal, where it is the
// global node index into the node-indexed destination.
template <typename idx_t>
static SFEM_INLINE void diag_node_slots(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        idx_t **const SFEM_RESTRICT mesh_elems, const ptrdiff_t e, ptrdiff_t sl[64]) {
    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
        for (int b = 0; b < CVFEM_HEX8_N_NODES; ++b) sl[a * 8 + b] = -1;
        sl[a * 8 + a] = (ptrdiff_t)mesh_elems[a][e];
    }
}


// The boundary closure's diagonal pass, SHARED BY BOTH GEOMETRIES: it is templated on the
// geometry of the boundary kernel it calls, so there is one sweep and two instantiations.
// It sat in affine/ when the sweeps were sorted by whether they touch an adjugate, which is
// the wrong question -- what separates the two sets is where the Jacobian comes from, and
// this one is told.

// The boundary closure's contribution to the block diagonal, as a sweep over the compacted
// boundary shell -- the same arrangement assemble_boundary_scs_jacobian_pass uses for the
// full matrix, and for the same reason: the closure is a per-face term that neither
// geometry nor kernel choice changes, so it does not belong inside the element loops.
template <typename scalar_t, typename geom_t, typename idx_t>
static SFEM_NOINLINE void assemble_diag_boundary_scs_pass(
        // The staging objects are gone; what this sweep reads out of them is what it takes.
        const scalar_t box_lx,
        const scalar_t box_ly,
        const scalar_t box_lz,
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const ptrdiff_t *const SFEM_RESTRICT bnd_elems,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        idx_t **const SFEM_RESTRICT mesh_elems,
        const uint8_t *const SFEM_RESTRICT face_mask_eff,
        const scalar_t *const SFEM_RESTRICT pres,
        geom_t **const SFEM_RESTRICT points,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
                                              // Two facts that belong to the containers rather
                                              // than to the arrays they hold: whether the mask
                                              // exists at all, and how many boundary elements
                                              // were compacted. A pointer carries neither --
                                              // vector::data() on an empty vector is not
                                              // required to be null -- so the caller states them.
                                              const bool      with_bnd,
                                              const ptrdiff_t n_bnd,
                                                          const scalar_t        rho,
                                                          const scalar_t        mu,
                                                          const int             isoparam,
                                                          scalar_t *const SFEM_RESTRICT diag) {
    if (!with_bnd) return;
    // The effective face mask arrives built. cvfem_hex8_build_face_mask_eff reads the mesh
    // and caches on it, so it is a once-per-solve setup rather than part of this pass; the
    // caller runs it, and both diag sweeps that reach here need it, so it hoists above them.
    scalar_t *const SFEM_RESTRICT values = diag;
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n_bnd; ++i) {
        const ptrdiff_t e     = bnd_elems[(size_t)i];
        const int       fmask = (int)face_mask_eff[(size_t)e];
        ptrdiff_t       sl[64];
        scalar_t        x[8], y[8], z[8], ux[8], uy[8], uz[8], p[8];
        diag_node_slots(mesh_elems, e, sl);
        gather_element_coords(mesh_elems, points, e, x, y, z);
        gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
        scalar_t adj[9], det = scalar_t(0);
        if (!isoparam) load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
        if (isoparam)
            boundary_scs_add_jacobian<true, true>(rho, mu, (const scalar_t *)nullptr, det, box_lx, box_ly, box_lz,
                                        x, y, z, ux, uy, uz, sl, values, fmask, 0);
        else
            boundary_scs_add_jacobian<true, false>(rho, mu, adj, det, box_lx, box_ly, box_lz,
                                        x, y, z, ux, uy, uz, sl, values, fmask, 0);
        (void)p;
    }
}

// ONE ELEMENT'S ASSEMBLY CONTRIBUTION, affine. Shared by the atomic assembly and the
// pack-coloured one, which differ in exactly one thing: whether the accumulation into the
// global matrix needs an atomic. That is already a template parameter of the element kernel, so
// it is one here -- and it is a property of the LAYOUT, not an option a caller chooses.
//
// Colouring is what makes `false` safe: no two elements of a colour share a node, so a plain
// `+=` cannot race. The atomic layout has no such guarantee and pays for it per entry.
template <bool ATOMIC, typename scalar_t, typename idx_t, typename geom_t>
static SFEM_INLINE void cvfem_hex8_assemble_element_affine(
        idx_t **const SFEM_RESTRICT         mesh_elems,
        geom_t **const SFEM_RESTRICT        points,
        const uint8_t *const SFEM_RESTRICT  face_mask,
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
        const Hex8Extras                   &opt,
        const int *const SFEM_RESTRICT      slots,
        const ptrdiff_t                     e,
        const scalar_t                      rho,
        const scalar_t                      mu,
        scalar_t *const SFEM_RESTRICT       values) {
    scalar_t ux[8], uy[8], uz[8], p[8];
    gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
    Hex8ExtraScratch ex;
    ex.load(mesh_elems, points, face_mask, pgx, pgy, pgz, qgx, qgy, qgz, ux_src, uy_src, uz_src,
            adj_ptr, det_ptr, opt, e);
    scalar_t adj[9], det;
    load_hex8_adj(adj_ptr, det_ptr, e, adj, &det);
    // rc and p go through the same upwind switch the residual uses, so this matches the
    // matrix-free action. Without --rhie-chow the pressure-pressure block of this matrix is
    // structurally zero, which is the saddle-point structure the solver's block-Jacobi cannot
    // invert -- see cvfem_hex8_ns_core.hpp on why the benchmark's assembly is a different
    // operator from the solver's.
    cvfem_hex8_ns_upwind_jacobian_add_slots<ATOMIC>(
            rho, mu, adj, det, ux, uy, uz, slots + (size_t)e * 64, values, ex.rc, p);
}


// ONE ELEMENT'S ASSEMBLY CONTRIBUTION, isoparametric. The affine twin's reasoning applies; the
// geometry comes per sub-control surface from the element's node coordinates instead of from the
// adjugate table.
//
// It follows the ATOMIC sweep's buffer reuse rather than the coloured one's: Hex8ExtraScratch
// gathers the coordinates when the Rhie-Chow term or the boundary closure gives it a reason to,
// and this gathers into the same buffers only when it did not. The coloured sweep gathered into
// its own locals unconditionally, which is one redundant gather per element whenever either term
// is on.
template <bool ATOMIC, typename scalar_t, typename idx_t, typename geom_t>
static SFEM_INLINE void cvfem_hex8_assemble_element_isoparam(
        idx_t **const SFEM_RESTRICT         mesh_elems,
        geom_t **const SFEM_RESTRICT        points,
        const uint8_t *const SFEM_RESTRICT  face_mask,
        const scalar_t *const *const SFEM_RESTRICT adj_ptr,
        const scalar_t *const SFEM_RESTRICT det_ptr,
        const scalar_t *const SFEM_RESTRICT pres,
        const scalar_t *const SFEM_RESTRICT pgx,
        const scalar_t *const SFEM_RESTRICT pgy,
        const scalar_t *const SFEM_RESTRICT pgz,
        const scalar_t *const SFEM_RESTRICT qgx,
        const scalar_t *const SFEM_RESTRICT qgy,
        const scalar_t *const SFEM_RESTRICT qgz,
        const scalar_t *const SFEM_RESTRICT ux_src,
        const scalar_t *const SFEM_RESTRICT uy_src,
        const scalar_t *const SFEM_RESTRICT uz_src,
        const Hex8Extras                   &opt,
        const int *const SFEM_RESTRICT      slots,
        const ptrdiff_t                     e,
        const scalar_t                      rho,
        const scalar_t                      mu,
        scalar_t *const SFEM_RESTRICT       values) {
    scalar_t         ux[8], uy[8], uz[8], p[8];
    Hex8ExtraScratch ex;
    ex.load(mesh_elems, points, face_mask, pgx, pgy, pgz, qgx, qgy, qgz, ux_src, uy_src, uz_src,
            adj_ptr, det_ptr, opt, e);
    if (!opt.with_rc && !opt.with_bnd) gather_element_coords(mesh_elems, points, e, ex.x, ex.y, ex.z);
    gather_element_fields(mesh_elems, ux_src, uy_src, uz_src, pres, e, ux, uy, uz, p);
    cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<ATOMIC>(
            rho, mu, ex.x, ex.y, ex.z, ux, uy, uz, slots + (size_t)e * 64, values, ex.rc, p);
}


#endif  // CVFEM_HEX8_BEST_ATOMIC_HPP
