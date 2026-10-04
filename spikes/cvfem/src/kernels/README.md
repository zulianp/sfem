# `src/kernels/` — the contract, and how each of DESIGN.md's clauses is met

This directory is what DESIGN.md constrains most tightly. Most of the clauses are held by a
test rather than by this note; `src/tests/compile/cvfem_kernels_self_contained.py` fails the
build on a violation and `cvfem_null_tested_args.py` fails it on the convention below. What is
written here is the part a test cannot state: which reading of a clause the code takes, and why.

## Held by a test

- **Header only, and self-contained.** No translation unit lives here, and no header `#include`s
  a quoted path that does not start with `kernels/`.
- **No library dependency.** No `std::vector` and no `smesh::` or `sfem::` name in code.
  Twenty qualified spellings survived here for a while and no host build could see them, because
  a host translation unit always has `smesh` in scope; `nvcc` said "namespace smesh has no
  member idx_t".
- **An absent input is a null pointer.** A sweep signals a missing mask, forcing or history with
  null, and the caller must spell `d.X.empty() ? nullptr : d.X.data()` — `vector::data()` is only
  required to return null for a vector that never allocated, and the Op clears four masks on
  every setup. The check propagates the null test through calls, because the sweep that takes the
  pointer is rarely the one that tests it.
- **No trace scope in a range-driven sweep.** `CVFEM_TRACE_SCOPE` is a `ScopedEvent` recording on
  an unlocked singleton, and a range-driven sweep is entered once per thread. That cost one
  segfault in eight runs before a bisect found it.
- **Vectorisation.** The lane-blocked kernels are compiled into their own objects and the
  emitted instructions are counted; a kernel that goes scalar fails the build, not a test. Two
  Jacobian arms are exempt by name under clang and the exemption is checked in both directions.

## The readings this code takes

**"Templated types."** The leaf element kernels are templated on the scalar type — 64 of them,
and the CUDA smoke test instantiates them at both `float` and `double`, which is what makes the
templating real rather than decorative. The **sweeps** are not: they take the build's `scalar_t`
through the alias the including translation unit supplies (`support/cvfem_default_types.hpp` for
a unit without a family header).

That is not laziness, it is the lane blocking. `CVFEM_HEX8_VEC_SIZE` is
`VEC_BYTES / sizeof(scalar_t)` at namespace scope, so a sweep templated on the scalar type could
be instantiated at `float` while the lane width stayed the one computed for `double`. Templating
the sweeps therefore means making the lane width a template-dependent quantity first — a
redesign of the lane blocking rather than a signature change.

**"Only the SIMD version is kept, the rest is moved to subpar."** Ten scalar matrix-free sweeps
remain and none of them can move. Eight are verification oracles that run in the default build,
one is reached from the solver family's own core, and one was explicitly declined by a recorded
earlier decision. `subpar/README.md` has the table, the oracle each one serves, and the Grace
measurements taken while settling it.

**"The threading model for atomics free kernels is abstract outside the function."** Every
atomics-free kernel takes a `cvfem_range` and owns no parallel region: the packed, element-
coloured, store and semi-structured sweeps, the shared and ghost reductions, and the boundary
shell's gather. The **atomic** layout keeps its own `#pragma omp parallel for`, which the clause
excludes by its own words, as do three zeroing and mask-building utilities that the atomic
sweeps call from inside their regions.

**"affine / isoparametric / axis_aligned logically separated."** Settled per format, by moving
functions where the two geometries were already separate and by a stated finding where they were
not. Each format's `affine/`, `isoparametric/` and `axis_aligned/` carries a README with its own
answer; `packed/affine/README.md` and `semistructured/affine/README.md` are the two that explain
why a folder split would duplicate a sweep.
