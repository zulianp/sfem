# `semistructured/affine/` and `../isoparametric/` — empty, and this one is a finding

There is no geometry split to make in this format, and it is not for want of trying: **every one
of its ten sweeps branches on `curved_e` per macro-element.** The count, from the sweeps
themselves:

| sweep | branches on curvature |
|---|---|
| `sscvfem_apply_macro_local_affine` | 7 |
| `sscvfem_apply_naive`, `_macro_local`, `_residual_naive_sweep`, `_block_diag_naive_sweep` | 4 each |
| `sscvfem_nodal_grad_packed_sweep`, `_scatter_range` | 3 each |
| `sscvfem_apply_macro_local_hoisted`, `_apply_blocks_impl`, `_residual_sweep` | 2 each |

`sscvfem_apply_macro_local_affine` is named for **affine hoisting**, not for an affine-only
assumption: it lifts the macro-element's invariants out of the micro-cell loop and still gives a
curved macro-element its own per-cell geometry. That is what the seven branches are.

**Why it cannot be separated.** `macro_curved` is mesh data, not a configuration: one mesh has
curved and straight macro-elements side by side, and which is which is known only per element at
run time. So the branch cannot become a template parameter the way `bool ISO` did for the packed
layout, and a folder split would mean two copies of every sweep — the gather, the hoisting, the
scatter and the shared reduction — with the one-path rule broken for a distinction the data does
not make.

The right reading of DESIGN.md's clause here is its own parenthetical, "now they are mixed in
with enum and booleans": there is no enum and no user-level boolean choosing the geometry in this
format. There is a per-element fact about the mesh, and both sweeps answer it.
