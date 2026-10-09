#pragma once

// THE AFFINE GEOMETRY DERIVATION: one adjugate and one determinant for the whole element.
//
// Separate from cvfem_hex8_ns_upwind_affine.hpp, which holds the kernels built on it, because
// the layering runs the other way for these two: the element gather and the pack staging derive
// the geometry before any kernel is reached, and they sit below the kernel layer. Pulling seven
// hundred lines of residual and Jacobian in to get an adjugate would invert that.
//
// cvfem_hex8_affine_edge_cols expands the adjugate into the three edge columns the
// sum-factorised forms contract against; it is affine for the same reason the adjugate is --
// one Jacobian for the element.

// cvfem_hex8_geom_at, the trilinear map's Jacobian at a reference point, which is in the leaf
// layer because the isoparametric kernels evaluate it per cell. No cycle: the leaf header
// includes neither this one nor the element gather.
#include "kernels/microkernels/hex8/cvfem_hex8_ns_upwind_kernels.hpp"
#include "kernels/cvfem_portability.hpp"

#include <cstddef>



template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_affine_adj(const scalar_t *const SFEM_RESTRICT x,
                                              const scalar_t *const SFEM_RESTRICT y,
                                              const scalar_t *const SFEM_RESTRICT z,
                                              scalar_t *const SFEM_RESTRICT       adj,
                                              scalar_t *const SFEM_RESTRICT       det) {
    cvfem_hex8_geom_at(x, y, z, scalar_t(0.5), scalar_t(0.5), scalar_t(0.5), adj, det);
}

// THE THREE EDGE VECTORS OF AN AFFINE HEX8, FROM THE ADJUGATE THE SWEEP ALREADY HOLDS.
//
// The twelve sub-control surfaces are grouped four per reference direction -- (0,1),(3,2),(4,5),
// (7,6) along xi, and so on -- and for a constant Jacobian all four surfaces of a group share
// one physical edge vector, which is that direction's column of J. Evaluating the trilinear
// Jacobian at the reference centre makes this exact and unconditional: column k comes out as the
// arithmetic mean of the four true edge vectors of group k, for ANY hex8, not only a
// parallelepiped. Checked symbolically against cvfem_hex8_adjugate_and_det's own expressions.
//
// So the affine variant has already committed to that mean, and a kernel on this path that forms
// its edge vector as x[J] - x[I] from node coordinates is not being more accurate -- it is using
// a different geometry from the one the flux uses, in the same kernel. Taking the column instead
// removes that inconsistency. On a mesh where the affine variant is valid -- a cube, a
// parallelepiped -- the two agree to the last bit; where they disagree, the disagreement measures
// how wrong the affine variant already is, and the isoparametric variant is the answer there, not
// a truer edge vector bolted onto this one.
//
// No new storage: adj(adj(A)) = det(A) * A for 3x3, so J = adj(adj)/det, and the adjugate and
// determinant are already in registers for the area vectors. Roughly forty flops and one
// reciprocal per element, amortised over twelve surfaces.
//
// LAYOUT, which is easy to transpose by accident: ex/ey/ez are indexed by DIRECTION, not by
// component. ex[k], ey[k], ez[k] are the x, y and z components of the edge vector of direction
// group k, so surface S of that group reads (ex[S/4], ey[S/4], ez[S/4]).
template <typename scalar_t>
static SFEM_INLINE SFEM_HOST_DEVICE void cvfem_hex8_affine_edge_cols(const scalar_t c0, const scalar_t c1, const scalar_t c2,
                                                                    const scalar_t c3, const scalar_t c4, const scalar_t c5,
                                                                    const scalar_t c6, const scalar_t c7, const scalar_t c8,
                                                                    const scalar_t det,
                                                                    scalar_t *const SFEM_RESTRICT ex,
                                                                    scalar_t *const SFEM_RESTRICT ey,
                                                                    scalar_t *const SFEM_RESTRICT ez) {
    // adj applied a second time, by the same expressions cvfem_hex8_adjugate_and_det uses.
    const scalar_t inv = scalar_t(1) / det;
    ex[0] = (c4 * c8 - c5 * c7) * inv;
    ex[1] = (c2 * c7 - c1 * c8) * inv;
    ex[2] = (c1 * c5 - c2 * c4) * inv;
    ey[0] = (-c3 * c8 + c5 * c6) * inv;
    ey[1] = (c0 * c8 - c2 * c6) * inv;
    ey[2] = (-c0 * c5 + c2 * c3) * inv;
    ez[0] = (c3 * c7 - c4 * c6) * inv;
    ez[1] = (-c0 * c7 + c1 * c6) * inv;
    ez[2] = (c0 * c4 - c1 * c3) * inv;
}
