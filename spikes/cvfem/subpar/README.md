# subpar — variants that were measured and lost

Nothing here is broken code kept out of sentiment. Each item was built to answer a real
question about the CVFEM HEX8 spike, the question got an answer, and the answer was that
this variant is not the one to use. They are kept, and kept compiling, because the
removals were made on measured grounds: if the hardware changes, the measurement should
be repeatable rather than re-derived from scratch.

Build with `cmake -DCVFEM_ENABLE_SUBPAR=ON`. Off by default.

## `cuda/cvfem_hex8_ns_cuda_colored.cuh` — pack-coloured assembly on the device

**Why it existed.** To answer "is assembly limited by atomic throughput?" Colouring the
packs so that no two packs of a colour share a node removes the race, and therefore the
atomics, and the difference would be the cost of the atomics.

**Why it lost.** The reasoning is a CPU intuition that does not survive the move to a GPU.
On the CPU a pack is one thread, so removing the inter-pack race removes all of it. On the
device a pack is a whole block, so the race *within* the pack remains untouched. The
kernel is correct only with `blockDim.x == 1`.

| configuration | result |
|---|---|
| `blockDim.x > 1` | wrong, relative error 0.54 |
| `blockDim.x == 1` | correct to 7.7e-16, **1.2 MDOF/s** |
| atomic path, for comparison | 238.3 MDOF/s |

About 200× slower in the only configuration where it is correct.

**What replaced it.** Element colouring (`src/core/cvfem_element_coloring.hpp`,
`cvfem_cuda_assemble_ecolored`), which colours elements rather than packs and so removes
the race that actually exists on the device. It is the fastest GPU assembly measured:
**277.3 MDOF/s** with the `sympy_block` kernel at n=128, against 238.3 for the best atomic
variant.

The answer to the original question, incidentally, turned out to be no. Element colouring
removes the atomics and is only 16% faster, and it is *slower* than atomics for the
hand-written kernel (166.6 against 218.2) — because `atomicAdd` compiles to a
fire-and-forget reduction instruction while the plain `+=` that colouring permits has to
wait on the load.

## Element colouring with the hand-written kernel

Not a separate file — it is `CVFEM_CUDA_JAC_HANDWRITTEN` in `launch_ecolored`, compiled
only under `CVFEM_ENABLE_SUBPAR`.

**Why it lost.** 166.6 MDOF/s against 218.2 for the same kernel on the atomic path.
Colouring buys the right to accumulate with a plain `+=` instead of `atomicAdd`, and on
Hopper that is the wrong trade: `atomicAdd` compiles to a fire-and-forget reduction
instruction, while `+=` is a read-modify-write that has to wait on the load.

**Why the other colouring stayed.** The trade pays once there is enough arithmetic per
write to hide the dependency. With `sympy_block` element colouring reaches **277.3
MDOF/s**, 18% above the same kernel on the atomic path and the fastest GPU assembly
measured anywhere in this spike. So element colouring is kept — for the fused kernels
only, which is the answer to "does removing the atomics help?": only when the kernel is
busy enough not to notice the atomics in the first place.

## `cvfem_sshex8_em.hpp` — the linear part as an element matrix, applied with a gemm

**Why it existed.** For a fixed state the Jacobian action is linear in the direction, and
under the affine-macro assumption everything in it except the convective flux has
coefficients that are pure geometry. So that part is one constant matrix for the whole
macro-element, and it can be applied to all `L^3` micro-elements with a single gemm --
the shape SFEM already uses for semi-structured linear elasticity
(`sfem_SemiStructuredEMLinearElasticity`, `operators/stencil/sshex8_stencil_element_matrix_apply*`).
Two versions: a 24x24 carrying the viscous momentum block, which is exactly what
`cvfem_hex8_ns_upwind_jacobian_add_slots_linear` assembles, and a 32x32 that also carries
the pressure gradient, the continuity divergence and the Rhie-Chow coupling.

**Why it lost.** It is slower than evaluating the same terms directly, on both machines
tried, and the gap did not close. At 4343300 dofs and L=8 on one Grace socket the 24x24
gemm is 1.205 ns/dof against 1.097 for the direct path -- 10% behind. On an M1 with
Accelerate, at 561924 dofs, 11.27 against 10.62.

Three revisions each helped and none was enough:

| variant | M1, L=8, ns/dof |
|---|---|
| per-element 32x32 matvec | 16.15 |
| batched into one gemm per macro | 11.37 |
| 24x24 momentum block, assembled on the fly | 11.27 |
| 32x32 including the pressure terms | 10.91 |
| direct evaluation, for comparison | **10.62** |

The arithmetic is against it. The 24x24 gemm is 576 multiply-adds per micro-element where
evaluating the same terms costs roughly 250-300 FLOPs, and going to 32x32 buys 1024
entries to remove about 180 FLOPs from the sub-control-surface loop. The blocks it gains
are sparse -- each pressure row touches only its handful of sub-control-surface
neighbours -- and a dense gemm cannot exploit that. BLAS barely helps either: Accelerate
is worth about 2% over the fallback loop for a 24x24 by 512 gemm, which is a small matrix
called once per macro-element.

**What replaced it.** `sscvfem_apply_macro_local_hoisted`, which keeps the direct
evaluation and lifts the affine-macro invariants out of the loop instead -- the direction
areas, the node-separation vectors, and the twelve Rhie-Chow coefficients, each of which
costs a square root and a division and was being recomputed `12 * L^3` times per macro to
produce the same twelve numbers.

**What would be worth trying before writing this off.** The sparsity is the obvious gap: a
dense gemm does substantial work on structural zeros, and `operators/stencil/` already has
`element_matrix_to_category_stencils` and `to_tensor_coeffs`, which turn an element matrix
into a constant-coefficient stencil over the lattice and need fewer operations than any
gemm. And none of this was measured on a GPU, where the arithmetic-intensity argument is
different enough that the ordering could invert -- as packing already does between Grace
and Hopper.

## `sympy_action*` — generated CSE for the HEX8 Jacobian action

**Why they existed.** HEX8 had no generated Jacobian action at all: the four `sympy*`
names covered the residual and the assembly, no `apply_jacobian_action_*` took a kernel
selector, and the CSE arrangement question had therefore never been asked of the operation
a Krylov solve spends its time in. The assembly evidence could not answer it — the
residual is arrangement-independent by construction, and the action had been measured
under exactly one kernel name.

Four arrangements over one expression tree, differing only in the scope handed to one
`sp.cse` call.

**Why they lost — and by how much less than they might have.** Grace, one socket, 72
cores, `--exclusive`, `OMP_PROC_BIND=true`, `--layout atomic --jac-action`, best of three.
The `sumfact` row is the hand-written scalar action:

| arrangement | scope | temporaries | of which geometry | 1,098,500 dof | 4,121,204 dof | vs flat |
|---|---|---:|---:|---:|---:|---:|
| `sumfact` | hand-written | — | — | **772.0** | **760.8** | — |
| `sympy_action` | all 32 outputs at once | 1295 | 0 | 285.8 | 284.9 | 1.00x |
| `sympy_action_comp` | one component, x4 | 1411 | 0 | 298.7 | 305.8 | 1.07x |
| `sympy_action_node` | 4 dofs of a node, x8 | 1597 | 0 | 308.9 | 309.5 | 1.09x |
| `sympy_action_geom` | flat, geometry hoisted | 1299 | 225 | 328.6 | 332.3 | 1.17x |
| `sympy_action_face` | one sub-control surface, x12 | 1497 | 0 | 388.2 | 394.8 | 1.39x |
| `sympy_action_geomface` | face-wise, geometry hoisted | 1035 | 189 | **420.8** | **429.3** | **1.51x** |

All six lose to the hand-written kernel, so generated CSE does not pay for this operator.
But **the arrangement alone is worth 1.51x**, which is far too large to leave unrecorded,
and the two effects behind it are separable and additive.

**Cutting the CSE scope is worth 1.39x, and the cut must follow the physics.** Flat sees
all 32 outputs, has the most to factor and the *fewest* temporaries — and is the slowest
of the six. Live ranges cost more here than reuse buys. But it is not simply "finer is
better": node-wise cuts to four outputs per scope and still loses to face-wise at eight.
A face is one flux and its algebra is self-contained; a node's four dofs are assembled
from twelve different fluxes and share almost nothing. **Cut along the flux, not along the
output index.**

**Hoisting the geometry is worth a further 1.09x on top of that, and 1.17x on its own.**
The two-level arrangement factors the geometry-only subtrees in their own pass and then
CSEs the field algebra with them reduced to atoms. It exists because cutting the scope
*costs* cross-scope reuse: twelve face scopes each re-derive the shared adjugate
independently, and hoisting hands that back — face-wise drops from 1497 temporaries to
1035, a 31% cut, without lengthening a single field live range.

**Temporary count predicts nothing.** It is worth stating plainly, because it is the
number one reaches for first. Flat has the fewest temporaries of the unhoisted four and
is the slowest. The hoist adds four net temporaries to flat (1295 to 1299) and gains 17%;
it removes 462 from face-wise (1497 to 1035) and gains 9%. Whatever is being optimised
here, it is not the count.

**Why the action still loses overall.** The assembly has 1024 entries all sharing one
adjugate; the action has 32 outputs and far less shared work to recover, while still
paying the full price of the arrangement — roughly a thousand fully-unrolled temporaries
against a compact twelve-face loop.

Note the assembly verdict against face-wise does **not** transfer: face-wise lost there
at 24.0 against 54.3 MDOF/s because it issued 2016 `CVFEM_ATOMIC_ADD`s against flat's
768, and the action accumulates into a local `r[]` where those are register traffic. It
is the best of the six here for the same reason it was the worst there.

### The `sp.cse` options, and why only one of them matters

Three settings were swept alongside the arrangements. Two close by counting, before
anything is compiled:

* **Dead-temporary pruning is unnecessary.** Zero dead temporaries out of 14,001 across
  the whole generated header. `sp.cse` only emits a subexpression used twice or more, so
  there is nothing for a liveness pass in `cvfem_codegen.py` to remove.
* **`optimizations=None` is strictly worse**: 17,692 temporaries against 14,001 (+26%),
  25,468 lines against 21,777 (+17%), and the blockwise assembly kernel +50%. `"basic"`
  stays.

The third could not be settled by counting. `order='none'` produces exactly the same
14,001 temporaries -- the identical factorisation -- but emits 12,762 lines differently.
Measured the same way as everything else:

| run | canonical | `order='none'` | | output |
|---|---:|---:|---:|---|
| residual, sympy | 866.5 | 872.6 | 1.007x | bit-identical |
| assemble, sympy | 56.1 | 56.2 | 1.001x | bit-identical |
| assemble, sympy_block | 53.9 | 53.9 | 1.000x | bit-identical |
| **action, flat** | 310.6 | **337.9** | **1.088x** | differ |
| action, face-wise | 434.3 | 433.4 | 0.998x | differ |

**It helps exactly one kernel, and that kernel is the one with the longest live ranges.**
Flat CSE gains 8.8% from nothing but a reordering of its assignments; face-wise, whose
scope is already cut, is unaffected. Residual and assembly do not move at all and their
output stays bit-identical.

So it is not adopted: it buys nothing where it counts, since face-wise is both faster
(434 against 338) and indifferent to it. What it does is corroborate the mechanism --
emission arrangement matters while live ranges are long and stops mattering once they are
short, which is the same thing the scope sweep says from the other direction.

Reproduce with `jobs/cse_action.sbatch`. Correctness is pinned by
`tests/cvfem_sympy_action_test.cpp`, which holds all four against the hand-written action
and against each other at 1e-16, so these remain measurable rather than merely present.
