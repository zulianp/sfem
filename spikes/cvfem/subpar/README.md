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

**What replaced it.** Element colouring (`cvfem_element_coloring.hpp`,
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
