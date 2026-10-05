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

## Where DESIGN.md's clauses stand

Three of the readings this file used to defend were overruled by the corrections at the end of
DESIGN.md. They are recorded here as superseded rather than deleted, because each was wrong in a
way worth not repeating.

**"Templated types."** NOT DONE. The leaf element kernels are templated on the scalar type — 64
of them, instantiated at both `float` and `double` by the CUDA smoke test, which is what makes
the templating real rather than decorative. The **sweeps** are not: they take the build's types
through the aliases the including translation unit supplies
(`support/cvfem_default_types.hpp` for a unit without a family header).

This file argued that the lane blocking exempted them: `CVFEM_HEX8_VEC_SIZE` is
`VEC_BYTES / sizeof(scalar_t)` at namespace scope, so a sweep instantiated at `float` would keep
the lane width computed for `double`. The correction: **"the kernels should be templated as well.
They should support different types for the computation, template scalar_t, geom_t, idx_t, etc...
(in a short time we would like to try single precision kernels as well)."** The obstacle is real
but it is the work, not an exemption — the lane width has to become template-dependent, and the
lane-blocked pack structs with it, so that an `f32` instantiation gets twice the lanes. `geom_t`
and `idx_t` are named explicitly because the coordinate precision and the index width are
separate choices from the accumulation precision.

**"Only the SIMD version is kept, the rest is moved to subpar."** DONE, and this file's survey
was overruled. It had found that eight of the ten scalar matrix-free sweeps were verification
oracles running in the default build and concluded that none could move. Serving as an oracle is
not a reason to keep a second variant in the tree; `subpar/README.md` records what happened to
each oracle, which was milder than the survey predicted — one had been aborting, one had become
a duplicate of the row above it, and two were strengthened by the retirement. The one place a
scalar sweep stays is the CUDA verify driver's host reference, because the device kernels call
the scalar `SFEM_HOST_DEVICE` leaf templates and so run the same arithmetic.

**"affine / isoparametric / axis_aligned logically separated."** DONE for `packed`, `store`,
`standard` and `colored` (the element colouring). This file argued that `template <bool ISO>`
satisfied the clause; the correction is that the folders meant folders. See
`packed/affine/README.md` for what the split bought beyond the structure — a templated sweep has
to take the union of both geometries' inputs, so neither half could have a lean signature.

Outstanding: the **pack-coloured** sweeps, which still test `GeomKind` inside their pack loop and
still live in `frontend/staging/` rather than here, and the **semi-structured** sweeps, which
branch on `curved_e` per macro-element. Both need their shared loop factored out first, the way
the packed layout's staging and drain were, so that splitting duplicates nothing.

**"The threading model for atomics free kernels is abstract outside the function."** DONE. Every
atomics-free kernel takes a `cvfem_range` and owns no parallel region: the packed, element-
coloured, store and semi-structured sweeps, the shared and ghost reductions, and the boundary
shell's gather. The **atomic** layout keeps its own `#pragma omp parallel for`, which the clause
excludes by its own words, as do three zeroing and mask-building utilities that the atomic
sweeps call from inside their regions. The pack-coloured sweeps are the exception and are listed
as outstanding above.

**"No user level option flags are propagated down here."** DONE for the micro-kernel selector —
`--kernel` and `KernelKind` are gone, and what chooses a kernel is the layout, the geometry and
which terms the operator carries. Two globals remain: `g_kernel_only` and `g_dense_flush`, both
read inside sweeps, both set by a driver flag.
