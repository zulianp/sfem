# `semistructured/axis_aligned/` — empty, and the likeliest first user

DESIGN.md asks each mesh format for `affine` / `isoparametric` / `axis_aligned`. This one has no
kernels yet, and the note is here because git does not track empty directories.

The semi-structured format is where an axis-aligned variant would pay first, and the reason is
structural rather than incidental: a macro-element's micro-cells are congruent by construction,
so `sscvfem_macro_geom` already computes one geometry per macro-element and all L^3 cells read
it. On an axis-aligned macro-element that geometry is three scalars rather than an adjugate and a
determinant, and `sscvfem_rc_coeff`'s per-face work collapses to one component. The hoisting
machinery the gain needs is therefore already in place, which is not true of the flat formats.

The invariant is stated in `../../microkernels/hex8/axis_aligned/README.md`.
