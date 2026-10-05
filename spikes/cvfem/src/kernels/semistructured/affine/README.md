# `semistructured/affine/` — empty, and the split it is waiting for

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

**And that is not an exemption — it is a partition, and the partition is owed.** This file used
to end by reading DESIGN.md's clause as satisfied by its own parenthetical ("now they are mixed
in with enum and booleans"): no enum, no user-level boolean, just a per-element fact about the
mesh that both sweeps answer. The correction rules that out — **"I indicated separate folders for
geometry affine vs isoparametric. This implies that the kernels should be separated. Quite
obvious isn't it?"** — and the standing rule for a precondition that holds for only part of the
data is to split the work, not to abandon the fast path for all of it.

So the design this folder is waiting for is:

1. **Order the macro-elements by curvature, once per level.** Which are straight is mesh data
   and does not change between applies, so the partition is setup work, like the pack ordering
   and the element colouring already are.
2. **Two sweeps over two ranges.** The straight range runs a sweep with no curvature branch at
   all, hoisting the macro-element's geometry once for every micro-cell; the curved range runs
   the sweep that derives it per cell. Each lands in its folder, and neither tests the other's
   case.
3. **The shared parts stay shared**, the way the packed layout's staging, extent and drain went
   into `../../packed/cvfem_pack_scratch.hpp` before that format was split. Here that is the
   gather, the scatter and the shared reduction — which is most of each sweep, and the reason a
   naive folder split would have duplicated ten sweeps.

What this buys beyond the structure is the same thing it bought the packed layout, and more of
it: the straight sweep loses seven runtime branches from inside its micro-cell loop in
`sscvfem_apply_macro_local_affine` alone, and a branch inside a lane loop is the shape of guard
this spike has measured at 1.83x.

Until then the two folders hold this note and the sweeps branch per macro-element, which is the
honest state rather than a reading of the clause.
