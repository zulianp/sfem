# Naming Conventions

What the generator is allowed to call things. Every name in `frontend/ops/generated/` is emitted by
this framework, so a name is a decision the generator makes thousands of times — 65.7% of the
generated tree's 15.9 MB is identifier bytes.

The architecture this serves is in `PRESCRIBED_ARCHITECTURE.md`; where it stands is in
`ARCHITECTURE.html`. This document is the vocabulary those two assume.

## The rule

> **A name is short because its scope is small.** The generator spells a concept at full length
> exactly once — where it crosses a boundary a human or another program reads — and at its
> abbreviation everywhere inside a scope where the abbreviation is unambiguous.

Length follows scope, not taste. That is what makes the tables below defensible, and it is what
`tests/test_names_come_from_the_plan.py` can check.

## One owner per name

A name is composed by the plan layer and printed by emission. It is **never** rebuilt from a format
string in an emitter, a backend, or the packaging layer.

| Slot | Owner |
|---|---|
| `<mat>_<elem>` head, mesh source name | `plans/generation.py` · `MeshKernelPlan` |
| `<mat>_d<n>_<family>` head, local header name | `plans/generation.py` · `LocalKernelPlan` |
| the verb | `plans/form_emission.py` · `FORM_ORDER_BY_KERNEL` |
| variant tail | `plans/apply_variants.py` · `ApplyVariant.suffix` |
| field stream suffixes, strides, scratch slots | `plans/streams.py` · `MeshFieldRole` |
| geometry stream names | `plans/geometry_quantities.py` |
| memory-space prefixes, lane index, vector width symbol | `targets/targets.py` · `LoopLoweringPolicy` |

Two rules keep this from re-entering emission as a decision, which
`tests/test_emission_is_a_printer.py` counts and forbids:

1. **The naming API returns values, never predicates.** `plan.entry_point(verb, variant)` is a
   string. There is no `plan.publishes_packed` to branch on.
2. **Emitters iterate.** `for variant in plan.variants: emit(plan.entry_point(verb, variant))` — an
   empty sequence emits nothing and takes no decision.

## Types and compile-time constants

| Concept | Name | Note |
|---|---|---|
| kernel scalar | `s_t` | the template parameter is named `s_t`; no alias is emitted |
| geometry scalar, internal | `g_t` | template parameter |
| geometry scalar, at the C ABI | `geom_t` | SFEM-wide typedef, not ours to rename |
| index, count, pointer difference | `idx_t` · `count_t` · `ptrdiff_t` | SFEM / POSIX, unchanged |
| vector width | `VS` | |
| quadrature points | `NQ` | per-dimension: `NQ1` |
| shape functions | `NS` | per-dimension: `NS1`; per-field: `U_NS` |
| field components | `NC` | |
| spatial dimension | `ND` | |

**Why the type names keep SFEM's `_t` suffix.** A bare capital would be shorter, and it is
the wrong economy here: `S` is the second Piola-Kirchhoff stress and `G` the shear modulus in
standard continuum-mechanics notation, `two_phase_flow` already declares parameters `T`, `R` and
`Z`, and this document says in the same breath that bare capitals belong to the material author.
Claiming `S` would mean refusing a material that spells its stress the conventional way. `s_t` and
`g_t` are unmistakably types by the suffix SFEM already uses, cost two characters more, and take
nothing from the material's namespace.

**Why the scalar type is a renamed template parameter and not an alias.** Callers pass template
arguments positionally — `operators/hex8/hex8_linear_elasticity.cpp:290` writes
`..._apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(...)` using *its own* names — so the
generated parameter name is invisible outside. Renaming it shortens the signature as well as the
body and emits no alias line, which keeps the standing rule that kernels carry no gratuitous
aliases. An alias appears in exactly one place: the non-template `extern "C"` bodies, where the
concrete type is `double` or `float`.

**Constants are never a bare single capital.** Single capitals belong to the material author:
`two_phase_flow` publishes parameters named `T`, `R` and `Z`. `NQ1` and `NS1` exist rather than the
`constexpr int Q` and `int S` the tensor-product and Neumann kernels declare today.

## Function names

```
mesh kernel, public C ABI, per element   <mat>_<elem>_<verb>[_<qual>...]_<geom>_msoa[_float]
mesh kernel, dimension-generic dispatch  <mat>_<verb>[_<qual>...]_<n>d_<geom>_msoa[_float]
local / micro kernel (sfem::codegen)     <mat>_d<n>_<family>[_<elem>]_<verb>_blk[_<var>]
internal template behind a C wrapper     <mesh kernel name>_impl
```

| Slot | Alphabet |
|---|---|
| `<verb>` | `objective` · `objective_steps` · `gradient` · `apply` · `hessian_<fmt>` |
| `<geom>` | `a` affine · `i` isoparametric |
| `<qual>` | `pk` packed · `2p` two-pass · `aos` · `unit` |
| level | `_msoa` mesh · `_esoa` element |
| `<family>` | `spx` simplex · `tp` tensor-product |
| `<mat>` | the material's declared `symbol` |
| `<elem>` | SFEM element label, lowercased; `proteus_` → `p` |

**The verb is the kernel, not the mathematics.** `objective` is the 0-form (an energy, or a
residual-based merit), `gradient` the 1-form (a gradient, or the negated residual), `apply` the
2-form (a Hessian action, or a Jacobian action). `plans/form_emission.py` owns this correspondence.
There is no `residual` verb and no `jacobian_action` verb — those were the residual path's words for
`gradient` and `apply`, and a formulation does not get its own vocabulary.

**`<elem>` appears once.** A prefix that already ends in its element does not get a second one;
`plans/generation.py` implements this idempotence rule, and nothing else may restate it.

**`<mat>` is declared, not derived.** The material author sets `symbol` on the `CodeGenerator`
beside `op_name`, so the abbreviation is a specification fact rather than a table the generator
guesses from.

```
mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_packed_two_pass_isoparametric_mesh_soa_float   109
mrkvn_elastic_phex8_gradient_pk_2p_i_msoa_float                                                                  46
```

## Temporaries

Emit **semantic** temporaries, not CSE serial numbers. The tree contains 356 bytes of `x0`/`c3`-style
names and several megabytes of names that say what they hold; that is the right way round, and a
reader of a generated kernel depends on it.

Shorten only the provenance the enclosing scope already fixes. Inside the lane loop there is no
other scope, so `_lane` is noise; inside a block-staged kernel `b` already says block.

| Concept | Name |
|---|---|
| block-staged adjugate, determinant, coordinates | `badj{n}` · `bdet{n}` · `bcoord{n}` |
| lane-local adjugate, determinant | `adj{n}` · `det{n}` |
| inverse determinant | `idet` |
| geometry offset | `goff` |
| field gradient, reference and physical | `gu_r{n}` · `gu{n}` |
| reference basis gradient | `grad_ref`, `grad_ref_x` / `_y` / `_z` |
| element node ids | `ev{n}` |
| pack-local shape index | `eshape` |

## Buffers, scratch, and memory spaces

Per-thread scratch is named and slotted by the field role, from `plans/streams.py`; the slot is
`role.index + 1` and is never a literal at the point of emission.

| Slot | Buffer |
|---|---|
| 0 | `pk_coord` |
| 1 | `pk_cur` |
| 2 | `pk_dir` |
| 3 | `pk_out` |

Memory spaces, for the targets that have them:

| Prefix | Space |
|---|---|
| `gl_` | global / device |
| `sh_` | shared / workgroup |
| `lo_` | thread-local array |
| `rg_` | register-resident scalar — normally elided, no prefix |

**`g_` is not available for "global".** It already means *geometry* at the C ABI
(`g_jacobian_adjugate0`, `g_geom_metric0`) and those names are frozen.

## Loop indices

`lane` · `q` quadrature · `s`, `sx`, `sy`, `sz` shape · `e` element · `k` pack-local node.

**`lane` stays spelled out.** It costs 145,748 bytes and shortening it would be the largest
single-token saving left after the tables above — and it is the innermost index of every
`#pragma omp simd` loop and the most-read token in the arithmetic. This is a deliberate exception,
recorded with its number so it reads as a decision rather than an oversight.

## The reserved namespace

The generator reserves, and materials may not declare:

```
types        s_t  g_t
constants    NQ NS NC ND VS NQ1 NS1 U_NS
indices      lane q s sx sy sz e k
prefixes     g_ gl_ sh_ lo_ rg_ pk_ b ev
```

A material declaring a colliding field or parameter is **refused at generation**, in
`CodeGenerator.__post_init__`, beside the existing name validation. A reservation that is not
enforced is a coincidence waiting to end.

## What is frozen, and why

Renaming these breaks something outside the generator. They are not style choices.

| Frozen | Because |
|---|---|
| public `extern "C"` entry points | called from hand-written `frontend/ops/sfem_LinearElasticity.cpp` and `frontend/tests/sfem_MatrixFromatsTest.cpp`; `package/op_wrappers.py` reconstructs them by name and **silently emits a different code path** when one is missing |
| **C ABI parameter names** | `tools/reproducibility.py` seeds its test input from an FNV hash *of the parameter name string*. Renaming `kappa` changes the input data and therefore every recorded digest — indistinguishable from an arithmetic regression |
| `sfem::codegen::` helpers in the shared headers | `operators/hex8/hex8_linear_elasticity.cpp` and `drivers/bench/neohookean_assemble.exe.cpp` call them |
| `<material>_hessian_{2,3}d_element_soa` | `drivers/bench/neohookean_assemble.exe.cpp` |
| `two_phase_flow_<elem>_*_element_soa_diagnostics` | reconstructed by token pasting in `drivers/simulations/generated_two_phase_flow.exe.cpp` |
| Op class, factory keys, registration functions | resolved by string from drivers and env vars |
| generated file paths | `tools/apply_bench.py` pins `<material>_<element>_operator.cpp` |

Geometry parameters are the one nuance: they are bound *structurally* by regex rather than seeded,
so they could be renamed without moving a digest. That is a separate, narrower option and not part
of this convention.

## Formatting

Emitted bodies are indented **two spaces per level**, through the printer. A line that carries its
own leading whitespace as part of a string literal is a bug in the emitter, not a formatting choice:
leading indentation is 15.45% of the generated tree, and it can only be governed if it is all
governed in one place.

## Known, and not naming

Measured while writing this, recorded so nobody re-derives it:

- The `op/` dispatch layer is 3.37 MB and 87.6% signatures plus forwarding argument lists. Its 1,400
  `extern "C"` declarations are **all distinct**, so there is no cross-file redundancy to hoist and
  naming reaches only ~265 KB of it. The real levers there are structural — the `double`/`float`
  twin duplicates every signature, and each declaration is written out where an include would do.
- `SFEM_RESTRICT` is 624 KB (3.9%). The generated headers already emit their own `#ifndef` fallback,
  so a generator-local short spelling in the shared prelude is available.
