# `colored/axis_aligned/` — empty, and why

DESIGN.md asks each mesh format for `affine` / `isoparametric` / `axis_aligned`. This one has
no kernels, and it carries this note rather than being an empty directory, because git does not
track empty directories: without a file here the structure DESIGN.md describes would exist in one
working tree and not in a fresh clone.

The invariant it would assume, and why no kernel assumes it yet, is in
`../../microkernels/hex8/axis_aligned/README.md`. The short form: a diagonal, element-constant
Jacobian collapses the adjugate to three scalars and every sub-control-surface normal to a
coordinate axis, and nothing in the spike's case set is axis-aligned except the box, which is
what the affine path is already measured on.

A colored sweep here would differ from the affine one only in which geometry kernel it calls, so
when the microkernels gain an axis-aligned variant this directory gets a sweep that is the
affine sweep with that one substitution -- not a copy of it.
