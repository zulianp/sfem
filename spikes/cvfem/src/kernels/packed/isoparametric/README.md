# `packed/affine/` and `../isoparametric/` — empty, because the separation is the template parameter

The packed layout's three sweeps — `apply_residual_packed_range`,
`assemble_jacobian_packed_range`, `apply_jacobian_action_packed_range` — and the store layout's
`assemble_jacobian_store_range` are each **one sweep templated on `bool ISO`**, and that is the
separation DESIGN.md asks for rather than a substitute for it.

The clause reads "logically separated (now they are mixed in with enum and booleans)". The
parenthetical names what was wrong: `GeomKind` arrived as an argument and was tested per pack,
inside the sweep. That is gone — no `GeomKind` argument and no geometry boolean survives in any
signature under `src/kernels/` — and the caller now picks the instantiation, so the geometry is
fixed at the point where it matters, the lane loop. The sweeps' own notes record the 1.83x that
guard cost when it was not.

**Why not folders as well.** The two geometries share the pack staging, the owned-node drain, the
ghost staging and the ghost reduction; they differ in the element loop between them. A folder
split would either duplicate all of that per geometry, which the one-path rule forbids, or
extract the element loop out of the `#pragma omp parallel` region in the spike's headline kernel
— a change that can move inlining and so would need a Grace measurement to justify, for a
structural gain the template parameter already delivers.

Where a format's two geometries ARE separate sweeps, they are in folders: see
`../../standard/affine/` and `../../standard/isoparametric/`, which were split by moving whole
functions with nothing duplicated, and `../../microkernels/hex8/affine/` and
`../../microkernels/hex8/isoparametric/`.
