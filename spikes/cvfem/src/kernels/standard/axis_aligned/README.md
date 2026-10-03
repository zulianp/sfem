# `standard/axis_aligned` — the placeholder DESIGN.md asks for

Empty on purpose. No kernel has been written against this geometry yet, and this file records the
invariant one would be allowed to assume so that the folder is not filled with something weaker.

## The invariant

An axis-aligned hexahedron has a **diagonal, element-constant Jacobian**. Everything the flux
needs then collapses:

- the Jacobian is `diag(hx, hy, hz)`, so its determinant is `hx*hy*hz` and its adjugate is
  `diag(hy*hz, hx*hz, hx*hy)` — three scalars where the affine kernel carries nine;
- a sub-control-surface area vector has one non-zero component, the one normal to the face, so the
  nine multiplies and six adds of `A . u` become one multiply;
- the edge vector of a direction group is one scalar, not three, which is most of what the
  Rhie–Chow coefficient reads;
- the reference gradient is a tensor product of 1D factors with no cross terms.

That is a strictly stronger assumption than `affine/`, which already takes the Jacobian as
element-constant but general. It is weaker than requiring a uniform grid: the spacings may vary
from element to element, they may differ between the three directions, and nothing here assumes
a lattice.

## Why it is worth a folder rather than a flag

The saving is in the **working set**, not the arithmetic. The affine kernel stages nine adjugate
cofactors and a determinant per lane; this one stages three spacings. On a kernel whose measured
limit is how much per-element geometry fits in registers, that is the kind of reduction that pays
— and the reason it has to be a separate kernel rather than a branch is the same reason `ISO`
became a template parameter: a geometry test inside the lane loop is the guard shape this tree's
own notes record costing 1.83x, and the vectorisation gate now refuses it outright.

## Before writing one

The oracle has to be a mesh that is axis-aligned but **not** uniform, and the result has to be
checked against the `affine` kernel on the same mesh. A cube oracle cannot distinguish a kernel
that assumed a diagonal Jacobian from one that assumed a constant one.
