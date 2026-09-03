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
