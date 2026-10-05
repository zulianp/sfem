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

**"Templated types."** PARTLY DONE, and the blocker is gone. The correction: **"the kernels
should be templated as well. They should support different types for the computation, template
scalar_t, geom_t, idx_t, etc... (in a short time we would like to try single precision kernels as
well)."**

This file used to argue that the lane blocking exempted the sweeps: `CVFEM_HEX8_VEC_SIZE` was
`VEC_BYTES / sizeof(scalar_t)` at namespace scope, so a sweep instantiated at `float` would keep
the width computed for `double`. That obstacle was real and it was the work, not an exemption.

What is done, for HEX8:

* `cvfem_hex8_vec_size<S>` — the lane width travels with the type, and an `f32` instantiation
  gets 32 lanes against `f64`'s 16, in a lane group that is 128 bytes at both. That is what lane
  blocking means here: a fixed byte width per group, so the same traffic carries twice the
  elements.
* the five lane-blocked packs, `Hex8RcTau` and `Hex8RcConfig` are templates with aliases at the
  build's types; every existing caller spells the plain name and is unaffected.
* the lane-blocked residual and Jacobian-action kernels, affine and isoparametric, and the six
  leaf kernels beneath them.
* eighty uses of `CVFEM_HEX8_VEC_SIZE` **inside** those kernels became
  `cvfem_hex8_vec_size<scalar_t>`. This was the real bug and it is not a compile error: a kernel
  indexing with the build's width walks a 16-element stride through a 32-lane pack.
* `cvfem_mixed_precision_packs` holds it, by comparing the f32 answer against the f64 one rather
  than by checking that lanes were written — a lane has several writers, so a short stride in one
  of them is covered up by the next. Measured agreement: **8.5e-08 relative**.

What is still owed: the **sweeps**, which take their types through the includer's aliases; and
`geom_t` and `idx_t` beyond `Hex8PackExtentT`, which already takes the index type because it
holds only counts and a node pointer. TET4 and the semi-structured kernels are untouched.

**"Only the SIMD version is kept, the rest is moved to subpar."** DONE, and this file's survey
was overruled. It had found that eight of the ten scalar matrix-free sweeps were verification
oracles running in the default build and concluded that none could move. Serving as an oracle is
not a reason to keep a second variant in the tree; `subpar/README.md` records what happened to
each oracle, which was milder than the survey predicted — one had been aborting, one had become
a duplicate of the row above it, and two were strengthened by the retirement. The one place a
scalar sweep stays is the CUDA verify driver's host reference, because the device kernels call
the scalar `SFEM_HOST_DEVICE` leaf templates and so run the same arithmetic.

**"affine / isoparametric / axis_aligned logically separated."** DONE for every HEX8 format:
`packed`, `store`, `standard`, `colored` (the element colouring) and the **pack**-coloured
sweeps. This file argued that `template <bool ISO>` satisfied the clause; the correction is that
the folders meant folders. See `packed/affine/README.md` for what the split bought beyond the
structure — a templated sweep has to take the union of both geometries' inputs, so neither half
could have a lean signature.

No sweep under `src/kernels/` takes a `GeomKind` or a geometry boolean. The front end chooses,
from `--geom` at run time, in the four launchers: `cvfem_hex8_packed_launch.hpp`,
`cvfem_hex8_store_launch.hpp`, `cvfem_hex8_ecolored_launch.hpp` and
`cvfem_hex8_best_colored.hpp`.

What the splits shared rather than copied, each extracted before the format was cut:

* the pack staging, extent, element preamble and four drains (`cvfem_pack_scratch.hpp`,
  `cvfem_hex8_pack_staging.hpp`) — three for the packed layouts, one for the coloured ones,
  which accumulate straight into the globals because colouring removes the reduction pass;
* the **Jacobian action's** lane loop, one per geometry, because the contiguous and
  pack-coloured sweeps differ in the drain and in nothing else. The residual's is deliberately
  NOT shared: sharing it measured −8% to −20% on Grace across two A/B runs, and
  `packed/affine/README.md` carries the table. The numerical gates were silent throughout,
  which is the argument for running the throughput A/B after any kernel restructuring;
* `cvfem_hex8_assemble_element_{affine,isoparam}<ATOMIC>`, because the coloured assembly is the
  ATOMIC assembly's element body with the colouring standing in for the atomics — that sweep is
  element-indexed with global gathers, not a packed sweep, and the split is what made it plain.

Outstanding: the **semi-structured** sweeps, which branch on `curved_e` per macro-element.
`semistructured/affine/README.md` states the partition this needs and why it is owed rather than
impossible.

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
