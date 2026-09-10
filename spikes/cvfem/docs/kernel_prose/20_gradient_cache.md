## Caching the nodal pressure gradient is worth 2.4x on the apply

1759 MDOF/s with the gradient hoisted out of the timed loop against 741 with it rebuilt
inside every apply, on the packed residual. That is the apply alone;
`docs/README_alps.md` records the same option (`SFEM_PGRAD_CACHE`) as 1.26x off the whole
linear solve, which is consistent — a solve is more than its applies. Anything that forces
the gradient to be rebuilt per apply gives back more than half the operator.

The effect is a layout property as much as a physics one: on the atomic layout the same
pair reads 706 against 442, a factor of 1.60. The cache is worth most exactly where the
element sweep is fastest, because it is a fixed extra sweep and the kernel it is added to
is what varies.

This is the one term in the cascade that is a scheduling decision rather than a
discretisation choice. Rhie–Chow, the boundary closure and the transient term are all
things the operator either has or does not; where the gradient is computed is free to
choose, and choosing wrongly is the largest single number in this report.
