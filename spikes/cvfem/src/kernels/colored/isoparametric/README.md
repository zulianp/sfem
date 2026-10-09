# `colored/isoparametric/` — empty, and why

The element-coloured layout has no isoparametric sweep. Both of its sweeps are in `../affine/`.

This is not an oversight. The element-coloured file is, in its own words, "the atomic sweep
verbatim apart from the loop bounds and the write-back" — it exists to isolate the *scatter
strategy*, with the kernel, the lane blocking and the geometry gather held identical to the
atomic layout so that the measurement separates colouring from everything else. An isoparametric
arm would be the same substitution the atomic layout already carries
(`../../standard/isoparametric/`), measured against the same atomic baseline, so it would answer
a question that layout has already answered.

What it would take, if the question changes: the isoparametric sweeps gather node coordinates per
element instead of reading an adjugate from a table, so the colour-ordered range would need the
coordinate gather the atomic isoparametric sweeps use, and nothing else. The colouring itself is
geometry-blind.
