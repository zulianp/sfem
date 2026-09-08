# Prescribed Architecture

What each layer must own, and the rules that make the boundaries real. Prescriptive only —
for what is actually built, see `ARCHITECTURE.html`.

## Lowering order

```
symbolic → fem → plans → ir → targets → emitters → backends
   L0/L1    L2     L3     L4     L5        L6        (orchestrator)

pipeline/   the driver: calls each layer in sequence, imported by none
package/    L7: packaging, runs only after emission
materials/  generators/   the frontend: sits above the whole stack
```

## Layer contracts

| | Layer | Owns | Ends here | Must not |
|---|---|---|---|---|
| **L0** | Specification | fields · operators · equations · qualifiers | — | name any other layer |
| **L1** | Form lowering | differentiation · 0/1/2-forms · pruning · block structure | provenance | keep a back-pointer to the system |
| **L2** | Discretization | element families · basis · quadrature · geometry mode · mixed order | — | emit anything |
| **L3** | Kernel planning | expression graph · CSE · scope hoisting · streams · phases · layouts · cost | SymPy | know a target or a language |
| **L4** | Kernel IR | loop nests · buffers · gather/scatter · AST passes | loop structure | know a target or a language |
| **L5** | Target lowering | OpenMP · CUDA/HIP · AVX512 · SVE/SME · atomics · widths · spaces | the target | know a form or an element |
| **L6** | Emission | AST + dialect → text · file assembly | — | make a decision (see below) |
| **L7** | Packaging | shared headers · C ABI · `sfem::Op` · factory · FLOP/AI reports | — | change a kernel body |

## The three rules

1. **One direction.** A layer may name the artifact it receives and never a layer that comes
   after it. Imports point up only.
2. **Closed handoffs.** Above L3, severability: cut the back-pointer, re-run the layers below,
   get an identical result. From L3 down, where SymPy has ended, a JSON round trip.
3. **One dispatch axis.** A layer may branch on the concept it owns and on nothing a layer
   above owns. No `if kind == "energy_soa"` in a backend, no `if "crs" in formats` in a printer.

A boundary is real when a concept is *nameable above it and unnameable below it*.

## Emission

L6 prints what the layers above it decided. The test is not size, it is decisions: **a branch
whose test reads planning-layer input** — a dependency set, an element specialization, a
system, a rule, a plan — is emission choosing *what* to emit rather than *how to spell it*.

- **Iterate a plan; do not interrogate it.** `for layout in plan.packed_mesh_layouts` is
  correct — an empty sequence emits nothing and no decision is taken at the point of emission.
  `if plan.emits_packed` is the same decision in the same wrong layer. This is the single most
  useful conversion available, and reading a plan is not by itself compliance.
- **Not headed for zero.** A branch on how C spells a declaration is emission's own business.

## Kernel rules

- **Lean kernels.** A generated kernel is a hot loop over elements. No validation, no
  precondition checks, no error flags, no diagnostic accumulators, no unused temporaries, no
  gratuitous aliases. A branch inside an element loop is also a vectorisation hazard.
- **Validate at setup, then assume.** A precondition that cannot change after `Op::initialize`
  is established there, once, and assumed by every kernel below. Hoisting it only as far as the
  per-call wrapper still pays for it on every call.
- **Never trade measured CPU performance for cleaner code.** Refactors are performance-neutral
  and proven so by benchmark.

## Design invariants

- **Forms unify below L1.** Energy-based and residual-based formulations both lower to
  0-form (energy, or a residual-based merit), 1-form (gradient, or the negated residual) and
  2-form (Hessian, or the Jacobian action). Nothing below the form layer distinguishes them.
  A material mixing an energy with a residual computes one residual merit; the energy part
  enters through its gradient, never as a potential added to it.
- **Evaluation strategy follows the element**, never the material and never how the material
  was written: tensor-product → sum factorization, always; lowest-order simplex → closed form,
  no quadrature loop and no per-point data; higher-order simplex → quadrature. *Expanded* refers
  to the loop and quadrature structure, not to the algebra — expanding the algebra defeats the rule.
- **Applicability is pattern-matched, not enumerated.** Ask the structural question of the
  lowered form (is the isolated map free of the gradient, and symmetric?), not a question whose
  only answer is the operator you already had in mind. Element-varying parameters ride *inside*
  the isolated matrix.
- **Scalar type is a runtime parameter, not a symbol multiplier.** One `extern "C"` symbol,
  scalar buffers as `void *`, one `const enum smesh::PrimitiveType` per group of buffers that
  share a type. Geometry is off the axis: its precision is a build-time property.
- **Vector-valued problems assemble into BSR.** Other sparse formats are not their target.

## Verification

Two postures. Pick by whether the output is *meant* to move.

| Output must not move | Output moves on purpose |
|---|---|
| `codegen_snapshot verify` — byte-identity of every generated file. Settles correctness and performance at once. | `tools/reproducibility.py` — l1/l2 digest per kernel against a committed baseline, plus geometry-mode parity, asserted rather than assumed. |
| | `tools/apply_bench.py` — cross-variant parity (four tolerance classes), answer digest, throughput. |

`codegen_snapshot check-tree` is separate from both and non-negotiable: it compares the
generator against **the tree the build actually compiles**, not against a record of the
generator. Every other gate can pass while the shipped artifact is stale.

Report every timing with the dof count, the thread count and the machine, and size the problem
until throughput saturates. A serial number alone can invert the conclusion.

## Ratchets

Shrink-only. Lower a budget when something moves; never raise one to make a change fit. Several
fail in both directions, so an unrecorded improvement is also a failure.

| Test | Forbids |
|---|---|
| `test_layering` | an import pointing down the stack |
| `test_plans_are_consumed` | a plan type losing its only consumer |
| `test_emission_is_a_printer` | a decision entering emission — *and* an improvement going unrecorded |
| `test_emitters_do_not_analyse` | an emitter deciding what to emit by analysing symbols |
| `test_target_binding` | a literal `#pragma` bypassing the target |
| `test_codegen_snapshot` | the shipped-tree exemption list growing |
| `test_module_imports` | a test module that stops loading |

A change that works, compiles and passes every functional test is still rejected if it moves a
ratchet the wrong way. That has happened twice, and the ratchet was right both times.
