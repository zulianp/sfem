# CVFEM Navier–Stokes: performance assessment

What the operator costs, on one Grace socket, measured rather than modelled. Every number
here is from a run on 72 Neoverse V2 cores with `OMP_PROC_BIND=true`, `OMP_PLACES=cores` and
an exclusive node, in fp64. Where a figure comes from a single measurement it says so, because
this machine's run-to-run spread is 5–11% and several conclusions in this document turned on
distinguishing a real effect from that.

`docs/CVFEM_Throughput.md` is the generated companion: per-scope tables for every
configuration at five sizes. This page is the argument.

## The operators

Four, all solving the same discretisation on the same fine mesh, differing in how the mesh is
stored and how a sweep writes its results.

| operator | mesh | scatter |
|---|---|---|
| flat | explicit HEX8 element table | packed, owned rows contiguous, ghosts reduced |
| ss L2 / L4 | macro-elements holding an `L³` lattice | `SSScatter`, every macro-element skin staged |
| **ss L8 packed** | the same, macro-elements grouped into packs | packed, only the pack boundary staged |

## Headline

At 4,343,300 dof, the element sweep and the reconstruction, per call:

| operator | sweep | nodal gradient | gradient's share of the solve |
|---|---:|---:|---:|
| flat | 4248 µs | 863 µs | 14% |
| ss L2 | 5737 µs | 1950 µs | 25% |
| ss L4 | 5127 µs | 1400 µs | 22% |
| **ss L8 packed** | **4380 µs** | **499 µs** | **10%** |

The semi-structured operator began the day 1.41× slower than flat overall, with its
reconstruction 2.97× slower. It now runs the reconstruction **1.6× faster than flat**, and has
the lowest gradient share of any configuration.

## Why: the staged fraction, not the arithmetic

A semi-structured sweep cannot write a node shared between two macro-elements, so it stages it
and reduces later. That skin is `(L+1)³ − (L−1)³` of `(L+1)³` nodes: 96% at level 2, 53% at
level 8. The reconstruction's cost tracked that fraction almost exactly:

| level | staged | gradient vs flat |
|---|---:|---:|
| 2 | 96.3% | 2.26× |
| 4 | 78.4% | 1.62× |
| 8 | 52.9% | 1.32× |
| 16 | 31.3% | 1.33× |

Two things follow. The arithmetic was never the cost — hoisting the geometry out of the micro
loop removed sixty-four times the work at level 4 and bought 9%. And the level is not a
"higher is better" knob: at level 16 less is staged but the **sweep** gets worse, 1.15× → 1.28×,
because `nxe` is 4,913 nodes there and the per-thread staging stops fitting in cache.

Packing the macro-element mesh collapses the staged fraction, because a node between two
macro-elements *in the same pack* is owned by the pack:

| level | staged, scatter | staged, packed (8 per pack) |
|---|---:|---:|
| 2 | 96.3% | 48.1% |
| 4 | 78.4% | 24.6% |
| 8 | 52.9% | **12.4%** |

and the gradient follows it down: 1886 → 635 µs at level 2, 1158 → 499 µs at level 8.

`smesh::PackedMesh` already accepted a semi-structured element type, so this needed no new
mechanism. The pack size is capped by arithmetic rather than tuning: `pack_idx_t` is `uint16_t`,
so a pack holds at most 65,535 nodes and a macro-element carries `(L+1)³` — 89 macro-elements
at level 8. Within that range the size barely matters; 4 to 32 span 2%.

## In context versus in isolation

A kernel benchmark applies the element kernel with nothing around it. A solver applies an
operator with everything around it. The gap is usually attributed to the Krylov method and
mostly should not be. Decomposed on one run at 4,343,300 dof:

| layer | flat | ss L8 packed |
|---|---:|---:|
| `Function::apply` — operator and constraints | 7101 µs | **5499 µs** |
| `CVFEMNavierStokes::apply` — the operator | 7033 | 5439 |
| element sweep — the kernel alone | 4482 | 4380 |
| nodal gradient | 2465 | 1510 |
| boundary closure | 368 | fused |
| constraint copy | 63 | 57 |

**Context costs the flat path 58% on top of its kernel and the packed path 26%.** What a bench
omits is the direction's gradient reconstruction and the boundary closure, not the Krylov
method. The element sweeps are within 2% of each other; the semi-structured one fuses the
boundary closure that costs flat a further 368 µs, so on like-for-like work it is 1.11× ahead
and the whole operator in context is 1.29× ahead.

Per-call figures across scopes with different call counts are not additive: the gradient is
evaluated ten times where the operator is applied six, because it serves the residual as well
as the Jacobian.

## The flat pack size is under-tuned

Best of three passes over the whole sweep at 4,121,204 dof:

| pack_size | jac MDOF/s | res MDOF/s |
|---|---:|---:|
| 256 | 938.4 | 1292.2 |
| **512** | **947.3** | 1331.1 |
| 1024 | 927.3 | **1336.3** |
| 2048 *(default)* | 895.0 | 1306.1 |
| 4096 | 758.2 | 1115.8 |
| 8192 | 720.3 | 1034.1 |

512 beats the default by **5.8%** on the Jacobian action; above 2048 it falls off 15–25%. A
single pass had read the opposite ordering — 909.9 at 256 against 829.1 at 512 — which is
noise, and is why the table is best-of-three.

`SFEM_PACK_SIZE` has no effect on a semi-structured run at all: `initialize()` returns before
the packing block, and such a hierarchy has no flat level for it to act on. Measured: 783
iterations and ~115 s at pack 0, 512 and 2048 alike.

## Scaling

Packed semi-structured, level 8, a few GMRES steps per size so this measures the operator
rather than the nonlinear solver:

| dof | sweep µs | MDOF/s | peak GB | setup s |
|---:|---:|---:|---:|---:|
| 8,586,756 | 8,901 | 965 | 2.4 | 1.1 |
| 67,898,372 | 70,037 | 969 | 19.3 | 10.1 |
| 132,304,644 | 136,864 | 967 | 37.7 | 20.9 |
| 228,266,500 | 238,480 | 957 | 65.2 | 39.8 |

Throughput is flat within 1.3% across a 27× range in problem size, memory is a steady 285
bytes/dof, and setup is linear.

This is the regime the semi-structured path exists for. At 592M dof its connectivity is
**0.84 GB** against the flat equivalent's 4.71 GB element table plus 11.78 GB of precomputed
affine geometry — a mesh that would not be worth storing becomes one that costs under a
gigabyte. What scales instead is the Krylov basis, at 4.74 GB per vector, which is why the
restart has to be chosen deliberately at these sizes rather than left at 30.

## What is not done

The **element sweep still runs through `SSScatter`**. It is 1.06–1.35× flat depending on level
and is the remaining application of the same lever that fixed the reconstruction.

`nodal_velocity_gradient` — and so every flow diagnostic — **fails on the semi-structured
path** by design rather than reconstructing on the macro mesh, which would silently report a
dissipation far too small. The semi-structured twin exists and is not yet wired up.
