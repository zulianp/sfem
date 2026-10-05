# `semistructured/isoparametric/` — every micro cell derives its own geometry

Ten sweeps, each the curved half of a pair. None tests whether its macro element is curved: the
range it is handed holds nothing else.

| sweep | what it derives per micro cell |
|---|---|
| `sscvfem_apply_naive_isoparam` | the adjugate and determinant, from the cell's own corners |
| `sscvfem_apply_macro_local_isoparam` | the same — and this is also the curved half of the *lifted* variant |
| `sscvfem_apply_macro_hoisted_isoparam` | a whole `SSMacroGeom` per cell |
| `sscvfem_apply_blocks_isoparam<Blocks>` | the same, with the unwanted field blocks compiled out |
| `sscvfem_residual_isoparam` | the `SSMacroGeom`, and the Rhie–Chow distances from the same corners |
| `sscvfem_residual_naive_isoparam` | the adjugate and determinant |
| `sscvfem_block_diag_isoparam` | the adjugate, via `sscvfem_block_diag_curved_macro` |
| `sscvfem_block_diag_naive_isoparam` | the adjugate and determinant |
| `sscvfem_nodal_grad_scatter_isoparam` | the adjugate and the sign of its determinant |
| `sscvfem_nodal_grad_pack_element_isoparam` | the same, inside a pack |

Ten and not eleven, because `sscvfem_apply_macro_lifted_affine` has no twin here: lifting the
Jacobian out of the micro-cell loop is precisely what a curved macro element cannot do. Its
curved range runs `sscvfem_apply_macro_local_isoparam`, which is also the curved range of the
macro-local variant — as branches inside those two sweeps the paths were bit-identical, and two
paths for one operation is what the one-path rule forbids.

## Why a curved macro element cannot be hoisted

Hoisting one geometry over a macro element's micro cells is exact when the macro element is
affine: its cells are translates of one another. For a curved one it is not merely inaccurate.
Neighbouring macro elements hoist *different* geometries, so the sub-control surfaces a node's
control volume is assembled from no longer close, and a uniform velocity acquires a discrete
divergence. Measured on the FDA nozzle as the continuity row of u = (1,0,0), p = 0, relative to
the flux scale: **1.39** at macro core 2 / L 2 and **1.49** at L 4 — not falling with the level —
against 0.085 for the flat mesh. That spurious source drove a backward flow fifty times the
physical velocity and stalled Newton with an exact Jacobian and a dense LU.

`sscvfem_residual_isoparam` carries the second half of the same requirement. Its Rhie–Chow
distances come from the cell's own corners, because they have to agree with the geometry the
Jacobian action differences — and a cell's own coordinates agree with the hoisted ones only on an
affine macro element. On a curved one they did not, and the residual and its Jacobian action
disagreed in every continuity row: 6.0e-02 by `SFEM_FD_CHECK` at macro core 2 / L 2, 3.2e-02 at
L 4, and exact with Rhie–Chow off.

## What these sweeps do *not* do, which is the second thing the split bought

They build no macro geometry at all. The branching sweeps these came from gathered the macro
element's eight corners and evaluated its Jacobian for **every** macro element, and a curved one
then threw the result away — in the hoisted apply it was overwritten in the first micro cell. In
the nodal gradient the only thing the curved path took from it was a degenerate-determinant
guard, and a curved cell carries its own.

## Where the rest of it is documented

`../affine/README.md` has the parts the two halves share: how the curvature partition is built
and handed out, what moved into `../cvfem_sshex8_ns.hpp` before the split so that nothing is
duplicated between the folders, the two measured conventions the micro-cell kernels depend on
(the geometry choice as a pointer, the curved loop out of line), and what gates the split —
`cvfem_flat_vs_ss_test`'s warped arms, together with the reason the semi-structured bench's
four-variant agreement is not a gate for it.
