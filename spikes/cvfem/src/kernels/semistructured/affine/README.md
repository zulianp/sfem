# `semistructured/affine/` — one geometry hoisted over a macro element's micro cells

Eleven sweeps, each the affine half of a pair. None of them tests whether hoisting is allowed:
the range it is handed cannot contain a curved macro element.

| sweep | what it hoists |
|---|---|
| `sscvfem_apply_naive_affine` | the macro element's corners |
| `sscvfem_apply_macro_local_affine` | the corners, with the Jacobian rebuilt per cell |
| `sscvfem_apply_macro_lifted_affine` | the corners **and** the Jacobian, once per macro element |
| `sscvfem_apply_macro_hoisted_affine` | the whole `SSMacroGeom`: adjugate, areas, difference vectors, Rhie–Chow coefficient |
| `sscvfem_apply_blocks_affine<Blocks>` | the same, with the unwanted field blocks compiled out |
| `sscvfem_residual_affine` | the `SSMacroGeom` and the Rhie–Chow corners |
| `sscvfem_residual_naive_affine` | the corners |
| `sscvfem_block_diag_affine` | the adjugate and the Rhie–Chow corners |
| `sscvfem_block_diag_naive_affine` | the corners |
| `sscvfem_nodal_grad_scatter_affine` | the adjugate and the sign of its determinant |
| `sscvfem_nodal_grad_pack_element_affine` | the same, per macro element inside a pack |

`sscvfem_apply_macro_lifted_affine` is the one with no isoparametric twin, and that is the
content of the split rather than a gap in it: lifting the Jacobian out of the micro-cell loop is
exactly what a curved macro element cannot do. Its curved range runs
`sscvfem_apply_macro_local_isoparam`, which is also the curved range of the variant below it —
as branches inside the two sweeps those paths were bit-identical.

## How the two halves are selected, since the distinction is not a configuration

Whether a macro element is curved is **mesh data**. One mesh carries curved and straight macro
elements side by side and which is which is known only per element, so this cannot be a template
parameter the way the flat layouts' geometry is. But it does not change between applies either,
so it is setup work:

1. `sscvfem_classify_macros` partitions the macro elements once per level — straight first,
   curved after, with `n_straight` the boundary — into `SSMeshData::macro_order`. Each half stays
   in ascending element order, and a mesh with nothing curved gets no array at all, which makes
   the straight sweep index the element directly.
2. The launcher hands each sweep its own range of **positions** in that order, through
   `sscvfem_affine_range` / `sscvfem_isoparam_range`.
3. A sweep covering only part of the mesh — the nodal gradient's scatter tail on a distributed
   mesh, or one pack's elements — restricts each half with `sscvfem_order_run`, a binary search
   that is correct precisely because each half is ascending.

`cvfem_ss_curvature_partition_test` asserts all of that directly: a permutation, each range
holding only its own kind, `n_straight` accounting for every element, both halves ascending, and
a partial element range restricting exactly over thirty-six sub-ranges. It is there because most
of those properties are invisible downstream — a mere permutation, or a straight element put in
the curved range, changes no answer at all.

## What stayed shared, and why it had to

The two halves of a sweep differ in the geometry their micro-cell loop uses and in nothing else,
so a folder split alone would have copied everything around that loop. What went into
`../cvfem_sshex8_ns.hpp` first:

* `SSMacroScratch`, `sscvfem_macro_scratch`, `sscvfem_macro_gather`,
  `sscvfem_macro_hoisted_corners` and `sscvfem_macro_drain_w<W>` — the macro-element gather and
  write-out, which eleven sweeps had each spelled out as fourteen arena offsets written as chains
  of `+ ((size_t)nxe)` one term longer than the last;
* one micro-cell kernel per sweep, plus the curved macro element's loop over it;
* `sscvfem_order_run`, the partition search, shared with the front end rather than written twice.

Same order of operations as the packed layout, whose staging, extent and drain moved into
`../../packed/cvfem_pack_scratch.hpp` before that format was split.

## Two measured conventions these sweeps depend on

**The geometry choice is a pointer, not a branch.** A micro-cell kernel takes the hoisted
quantities and treats `nullptr` as "use this cell's own", so the choice folds away at each
inlined call site. A runtime selection inside the sweep measured **8% slower** on affine meshes,
and compiling the sweep twice from one generic body cost **25%** and slowed unrelated kernels in
the same unit.

**The curved micro-cell loop is out of line.** Inlined, the per-cell geometry construction grew
the unit past the point where gcc still inlined the small per-cell helpers the affine sweeps
depend on: `sscvfem_rc_config` became a call in every micro cell of the block diagonal, **9%
slower on meshes with no curved element at all**. `sscvfem_block_diag_affine` carries
`__attribute__((flatten))` for the other side of the same effect — the cell kernel has two call
sites, gcc outlines the Jacobian and boundary kernels once they do, and the per-cell call
measured 3% slower on the box.

## What gates this

A box has no curved macro element, so the hoisted geometry **is** each cell's own and no oracle
on one can see how the geometry was derived. `cvfem_flat_vs_ss_test`'s four **warped** arms are
the gate: on a curved macro element the isoparametric sweep gives every micro cell its own
geometry, which is exactly what the flat operator does, so the two must still agree. Forcing
`sscvfem_macro_curved` to false fails those four checks and nothing else, at both pack sizes.

The four-variant agreement inside `cvfem_sshex8_bench` is **not** a gate for this, and that was
measured too: the same break leaves both of its arms passing, because its four sweeps read one
geometry source and agree with each other whether it is right or wrong.
