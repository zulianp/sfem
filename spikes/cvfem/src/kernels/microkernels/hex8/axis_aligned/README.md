# `axis_aligned/` — empty, and what it is for

DESIGN.md asks each mesh format for `affine` / `isoparametric` / `axis_aligned`. The first two
have kernels; this one does not yet, and the directory exists with this note rather than being
left unmentioned, so that the gap is a stated gap.

**The invariant it would assume.** An axis-aligned HEX8 element has a diagonal Jacobian,
constant over the element. Three consequences follow, and they are the reason the variant is
worth having rather than a special case of `affine/`:

- the adjugate is diagonal, so `cvfem_hex8_pushforward` and `cvfem_hex8_grad_at` collapse from
  nine multiplies to three, and the determinant is `hx * hy * hz`;
- every sub-control surface normal is a coordinate axis, so `cvfem_hex8_area_dir` returns one
  nonzero component and the flux dot products lose two thirds of their work;
- the element carries three scalars rather than ten, which is the part that matters for a
  register-starved kernel: the affine path reads an adjugate and a determinant per element from
  a table, and this one would read `hx, hy, hz` — or, for a uniform grid, nothing at all.

**Why there are none yet.** Nothing in the spike's case set is axis-aligned except the box, and
the box is what the affine path is already measured on, so a kernel here would have no
configuration that is both realistic and faster than what exists. The semi-structured family is
the natural first user: its macro-elements are congruent by construction, and
`sscvfem_macro_geom` already hoists the invariants a diagonal Jacobian would make trivial.
