# MI300A performance of the MGSDF contact solve

The MaMAL benchmark (`workflows/bench_MaMAL`) runs 3.5-3.8x slower on one MI300A than on one GH200
at 136-623 M dofs, and 10-25x slower below 5 M dofs, with the same iterations and cycles on both
(hip-port, 2026-11-13: Alps job 5344149, Nibi job 27200994). The MI300A has the higher memory
bandwidth (5.3 against 4.0 TB/s) and FP64 matrix peak, so the gap is in the code, not the device.

## Rules

- Every change for AMD is a separate path, selected at compile time (`#if defined(SFEM_ENABLE_HIP)`
  or a HIP-only CMake option); the CUDA build is unchanged by construction, so GH200 needs no A/B
  per step.
- Each step is measured on MI300A only: the MaMAL case at 136 M dofs (sphere, base resolution 6,
  level 8, line search; 69.4 s) and at 0.7 M dofs (base resolution 1; 20.7 s), plus the operator
  benchmark (`drivers/bench/bench_op_apply`) for kernel steps. The same iterations and cycles, and
  ctest 119/119 on MI300A, gate each step.
- At the end, each AMD path is built for CUDA once and run on GH200; where it is faster there too,
  adopting it for both is proposed.

## Where the time goes (136 M dofs, synchronised build: a sync after each launch)

test_contact: GH200 23.0 s, MI300A 104.6 s. Per operation (MI300A share of its 104.6 s):

| operation | GH200 | MI300A | ratio | share | GH200 kernel (nsys) |
|---|---|---|---|---|---|
| level-8 apply, `cu_linear_elasticity_apply_729` | 8.1 s, 10.0 ms/call | 50.6 s, 62.9 ms/call | 6.3 | 48% | `..._apply_tensor_product_matrix_kernel<double, 8>` (FP64 `mma.sync`) |
| coarse HEX8 apply, `cu_linear_elasticity_apply_8` (coarse CG, 25 315 calls) | 1.9 s, 76 us/call | 22.0 s, 868 us/call | 11.4 | 21% | `cu_affine_hex8_linear_elasticity_apply_kernel` (25.8 us) |
| smoother, `StationaryIteration::apply` | 3.1 s | 8.3 s | 2.7 | 8% | `sbv3_apply` (41 us), `taxpby` (619 us) |
| level-4 apply, `cu_linear_elasticity_apply_125` | 1.6 s | 6.0 s | 3.9 | 6% | `..._apply_warp_kernel<double, 4>` |
| small kernels (copy_constrained_dofs, sbv_mult3, ...) | 22.6 us/call | 102.7 us/call | 3-5 | 5% | 1-10 us each |

Under HIP the level-8 apply is not the tensor-product kernel: its contraction is NVIDIA's FP64
tensor-core PTX, so HIP defaults to kernel 0 (`local_mem_seg`, `SFEM_HEX8_LINEAR_ELASTICITY_LEVEL8_KERNEL`).
The synchronised build inflates per-call times on both devices (MI300A's syncs cost more), so small
kernels are measured without it (step 0).

## Steps

0. **Kernel timings on MI300A.** There is no rocprof on Nibi; make roctracer's tool library
   (`libroctracer_tool.so`/`libhip_stats.so`, ROCm 6.4 module) report per-kernel times, or failing
   that time the launches with events in a profiling-only build. The per-kernel table of the 136 M
   case on MI300A, next to nsys's on GH200, replaces the per-operation one above.
1. **Hardware FP64 atomics.** The HIP compiler emits a compare-and-swap loop for `atomicAdd(double*)`
   unless `-munsafe-fp-atomics`; MI300A has the instruction for device memory, which is where SFEM's
   atomics write. Add the flag for HIP sources, check no atomic targets host or managed memory, and
   measure the coarse and level-4 applies (atomic scatter) and the restriction.
2. **Level-8 apply (48%).** First the existing kernels 0, 2, 3 and 5 on MI300A
   (`bench_op_apply`, `SFEM_HEX8_LINEAR_ELASTICITY_LEVEL8_KERNEL`). Then an AMD tensor-product kernel:
   the same element-matrix contraction on the FP64 matrix cores (`__builtin_amdgcn_mfma_f64_16x16x4f64`)
   with a 64-lane wavefront layout, next to the CUDA one in `sshex8_linear_elasticity_cuda.cu` under
   `SFEM_ENABLE_HIP`. Target: GH200's 10 ms per apply at 136 M dofs.
3. **Coarse HEX8 apply (21%).** 25 315 applies of 26 us on GH200, 10x that on MI300A (step 0 gives
   the kernel's own time). After step 1: a wavefront-64 launch shape and, if the kernel is
   latency-bound, an element-per-thread layout without atomics (coloured or gathered).
4. **Coarse CG latency.** Each of its ~25 000 iterations launches the apply, two dots read back to the
   host and three vector updates; on MI300A each launch and sync costs more. An AMD path keeping the
   scalars on the device and fusing the vector updates, or one captured iteration relaunched as MPRGP
   does under HIP, measured as time per iteration.
5. **Smoother, level-4 apply and vector kernels.** `sbv3_apply`, `taxpby` and the warp kernel
   `<double, 4>` keep SFEM's 32-lane logical warp, half a wavefront on AMD; wavefront-64 variants
   where the profile shows them bandwidth- or occupancy-bound.
6. **Final check on GH200.** Each AMD path from steps 1-5 built for CUDA and timed on GH200
   (MaMAL 136 M and 0.7 M, `bench_op_apply`); report and propose adoption where it wins.

## Progress

| step | change | MaMAL 136 M (GH200 19.0 s) | MaMAL 0.7 M (GH200 1.4 s) |
|---|---|---|---|
| start | hip-port | 69.4 s | 20.7 s |
| 2a | HIP default level-8 kernel: gather (5) | 35.0 s | 2.2 s |
| 4a, 4b | dot grid cap; the coarse CG's loop on the device, captured | 31.4 s (34.0 without: three runs each) | 1.98 s (2.1-2.4) |
| 2b | level-8 apply on the FP64 matrix cores (kernel 7) | 25.7 s (31.7 with kernel 5, same node) | 1.6-1.8 s |
| 5a | level-4 apply on the FP64 matrix cores | 25.5-25.6 s (28.6-29.1 with the warp kernel, same node) | 1.8-1.9 s (1.7-2.1) |
| 3a | HIP launch bounds on 17 hand-written kernels (coarse HEX8 apply, level-2 apply, ...) | 26.3 s (another node) | |

- Step 1, hardware FP atomics: no effect. With `-munsafe-fp-atomics` the applies of every level of the
  hierarchy (44^3 macro elements, levels 8, 4, 2 and the HEX8 coarse one) time within noise of the
  build without it (HEX8: 97.0 against 98.0 us); the compiler already takes the hardware atomics
  for these kernels. Not kept.
- Step 2a: of the level-8 kernels HIP can build, the gather kernel (5) applies 44^3 macro elements
  (44 M nodes) in 15.3 ms on MI300A against 54.1 (0, the former default), 49.6 (2) and 49.5 ms (3);
  GH200 takes 9.6 ms with its tensor-product kernel (4) and 11.0 ms with the gather one. The
  0.7 M case drops more than the kernel time explains (20.7 to 2.2 s); kernel 0's per-thread stack
  is the likely cost of each launch on AMD (scratch set-up), to be confirmed by step 0's timings.
  Same iterations and cycles; ctest 119/119 on MI300A.
- Step 0: roctracer's tool library records kernel activity (`HSA_TOOLS_LIB=libroctracer_tool.so`,
  `ROCTRACER_DOMAIN=hip`, lines `start:end device:queue name:id:pid`); copies' times there include
  their wait in the queue (they sum past the wall time), so only kernels are compared. 136 M case,
  GPU kernel time, after step 2a:

  | kernel | MI300A | GH200 (nsys) | per call |
  |---|---|---|---|
  | level-8 apply (MI300A gather, GH200 FP64 tensor-core TP) | 13.5 s | 7.8 s | 16.8 / 9.7 ms |
  | level-4 apply, warp kernel `<double, 4>` | 3.4 s | 0.74 s | 6.7 / 1.5 ms |
  | `sbv3_apply` (smoother's block Jacobi) | 2.3 s | 1.1 s | 86 / 41 us |
  | HEX8 coarse apply | 2.2 s | 0.65 s | 88 / 26 us |
  | `tdot` (coarse CG) | 2.2 s | 0.48 s | 42 / 9.4 us |
  | `taxpby` | 2.1 s | 2.1 s | 623 / 619 us |
  | all kernels | 27.9 s | 14.1 s | |

- Step 4a, the dot's grid on HIP: `tdot` launched a block per 128 entries, each adding into the one
  word; capped at four blocks per compute unit it takes 12.2 us (was 42.5), all kernels 24.7 s.
  The wall time does not move (35.0 s): the coarse CG is bound by the host. Its ~25 000 iterations
  take 646 us each on MI300A and 337 us on GH200 (cycle(3), 24 calls: 16.4 against 8.4 s), against
  ~130 and ~55 us of kernels; each iteration waits twice for a dot on the host and launches about
  eight kernels. The 0.7 M case is unchanged (2.11-2.37 s over three runs).
- Next, by the time they leave on the critical path: the coarse CG's host round trips (step 4,
  ~11 s on MI300A), then the level-8 apply (step 2b), the level-4 apply and the coarse apply.
- Step 4b, the preconditioned CG's loop on the device: `ConjugateGradientKernels::
  run_precond`, set by HIP builds only. The scalars stay in a device array, each iteration's x and r
  update is skipped once stopped, and the host waits once per 16 iterations; the iterations and the
  relative residuals are the host loop's (16/32 and 12/24, 3.92097e-08 and 3.55533e-08).
  - Captured as one graph per 16 iterations: 136 M 32.5 s (was 35.0), 0.7 M 1.94 s (was 2.1-2.4);
    the coarse CG's loop takes 175 ms a solve (~165 us an iteration, was ~646). But HIP 6.4 crashes
    capturing a product that allocates on the stream, as the generated operators do (their permuted
    HEX8 connectivity, `cudaMallocAsync` in each apply): sfem_PlaygroundContactTest's ssmgc runs.
  - Launched without the graph: 34.3 s and 2.5 s, no gain; the launches, not the waits, are the cost.
  - So the gain needs allocation-free products. Under HIP the generated launchers now keep their
    permuted connectivity in a table per host thread and stream (`permuted_connectivity_table`),
    rewritten in stream order before each kernel; the CUDA branch of the generated code is the
    former one. Captured again: ctest 119/119 on MI300A; three runs of the 136 M case 31.37, 31.53
    and 31.37 s against 33.17, 33.32 and 35.44 s without steps 4a and 4b; 0.7 M 1.98 s.
- Step 2b, the level-8 apply on AMD's FP64 matrix cores (`..._apply_level8_mfma_kernel`, HIP only,
  the HIP default): the element matrix shared by the macro element's 512 micro elements turns the
  apply into the GEMM Y = U K^T (512 x 24 x 24), done with `v_mfma_f64_16x16x4f64` (lane layout
  measured on gfx942: D(i, j) in slot i / 4 of lane j + 16 (i % 4)), K^T in registers, U from LDS,
  the results scattered through the same LDS. 44^3 macro elements: 7.66 ms (8 wavefronts, 244
  VGPRs, two waves per SIMD; 8.08 with 4, 16.3 with 16 spilling) against 15.3 ms for the gather
  kernel and GH200's 9.6 ms (its FP64 tensor-core kernel); `|y|` the gather kernel's to 1e-16.
  MaMAL 136 M, alternated on one node: 25.82, 25.58, 25.71 s against 31.85, 31.70, 31.68 s;
  the same iterations and residuals; ctest 119/119.
  - Where its time goes (one phase skipped at a time, 8.3 ms then): the global atomics of the
    node values 2.6 ms, the global gather 2.0, the matrix instructions 1.0, the LDS scatter 1.0,
    the element matrix 0.8; the kernel's frame (zeroing, barriers, 85 k blocks) 1.2. So the
    connectivity is read once (LDS) and the 7^3 interior nodes, the macro element's alone, are
    added without atomics (8.12 to 7.66 ms). The rest is the gather and scatter every level-8
    kernel pays, on both vendors.
  - Literature (survey 2026-11-14): FP64 matrix units pay where LDS traffic, not HBM, bounds the
    kernel (MFEM sum factorisation on GH200's DMMA 35-59% a kernel, Tu et al. 2026; CEED's MI250X
    report); memory-bound kernels gain at most ~1.33x. Here one 24x24 matrix per macro element is
    ~17 FLOP/B, compute-bound for the vector units.
  - The code generator has no counterpart: it reads a semi-structured mesh as one order-L Lagrange
    element per macro element, and its stored element matrices differ per element (memory-bound).
    A matrix-core path there would be a different kernel (the high-order sum factorisation's 1D
    contractions, 9x9 padded to 16 at order 8).
- Step 2c, the generated high-order applies (one Qp element per macro element) on the FP64 matrix
  cores: hand-written prototype first, then the generator. `bench_op_apply` with `SFEM_OP` (a yaml
  op) and `SFEM_HIGH_ORDER=1`, x = sin(0.37 i), best of 10-20 applies; `|y|` and x^T y checked.
  - Baseline, generated kernels (a thread per element, 256 per block):

    | op (dofs) | MI300A | GH200 |
    |---|---|---|
    | elasticity Q2, isoparametric (24.4 M) | 25.4 ms | 18.7 ms |
    | elasticity Q2, affine (`apply_assume_affine: 1`) | 7.54 ms | 2.85 ms |
    | Neo-Hookean Ogden Q2, isoparametric (x is not admissible: `|y|` NaN) | 40.9 ms | 24.7 ms |
    | Saint Venant-Kirchhoff Q3, isoparametric (17.8 M) | 35.6 ms | 18.3 ms |
    | hand-written micro HEX8 level 2, same mesh | 11.2 ms | 2.03 ms |

    On MI300A the generated kernels spill: 1098 VGPRs (5.3 KB scratch a lane) for the affine Q2
    apply, 2936 (21 KB) for the isoparametric one, at 128 VGPRs (256 threads a block).
  - The prototype (`..._apply_{a,i}_mfma_impl` in the emitted
    `linear_elasticity_proteus_hex27_operator.cu`, HIP only, `SFEM_PROTEUS_HEX27_MFMA`; not
    committed): a wavefront takes 16 elements, one per column. With G[r][s] the reference gradient
    matrix (r = NQ k + q, s the node), the reference gradients of each component are the GEMM
    G U_c (16x16x4 FP64 MFMA, G in LDS); lane l then holds the three directions of element l % 16
    at four quadrature points, so the Jacobian (isoparametric: the same GEMM on the coordinates),
    the stress and the flux are formed in registers, and the flux tiles are already the B operand
    of the test contraction G^T F_c. The output rows are the nodes the lane gathered. Quadrature
    points are taken a tile row at a time to bound the live accumulators (two waves per SIMD).
  - MI300A, 1 M elements (24.4 M dofs), alternated on one node:

    | apply | generated | MFMA prototype | |
    |---|---|---|---|
    | affine | 7.48-7.57 ms | 2.32-2.34 ms (8 waves, 2 per SIMD, no spills) | 3.2x |
    | isoparametric | 25.2 ms | 4.87-4.90 ms (8 waves, 54 spilled VGPRs) | 5.2x |

    `|y|` agrees to 15-16 digits, x^T y to rounding (413092.2668479641 affine, 413092.288488787
    isoparametric). The affine prototype's time is half global atomics (plain stores: 1.22 ms;
    no gather: 2.20 ms); the matrix instructions alone are ~0.3 ms at peak.
  - In the library and the generator (HIP builds only; the CUDA build's code is unchanged):
    - `src/frontend/ops/cuda/sfem_cuda_TensorProductMatrixCoreApply.cuh`
      (`sfem::tensor_product::hip::matrix_core_apply`): the prototype's shape for any tensor-product
      element, component count, geometry and material, the material at a point through a functor.
      G is tabulated in LDS when it fits (Q2; formed from the 1D tables it cost registers: 138
      spilled VGPRs against 45), else formed from the 1D tables. Blocks of 8 waves (256 registers
      a lane) for a linear material whose accumulators fit, else 4 (512). `SFEM_MATRIX_CORE_APPLY=0`
      turns it off: the generated kernel then runs.
    - The generator publishes `<prefix>_apply_point` in each device tensor-product local header
      (the apply block's point section as a functor; `plans.direct_assembly.matrix_core_apply_admits`)
      and, in the apply's double entry points, a guarded launch of the library's kernel
      (`matrix_core_apply_launch_admits`: Q2 and Q3). Linear elasticity, Neo-Hookean Ogden,
      Saint Venant-Kirchhoff and modified Mooney-Rivlin on PROTEUS_HEX27 (HEX27 forwards to it),
      the last two also on PROTEUS_HEX64; the Mooney-Rivlin Prony variants carry point history and
      keep the generated kernel.
    - Q3: with the 16 k-steps of the gradient GEMM unrolled and the nodes held, the entries of G
      and the indices overflowed the registers (SVK 319/303 spilled VGPRs at 512 a lane, MMR
      500-600). Beyond Q2 the k-steps are rolled and the nodes read again (SVK 64/75, MMR
      334/299 spilled: its point is the heaviest). (The generated SVK Q3 kernel fails in the
      compiler's allocator when the regenerated `line_p3_q5`/`quad_line_q5` tables differ from the
      committed ones in their last digits; the committed ones are kept.)
  - MI300A, 1 M elements (24.4 M dofs), best of 10, generated (`SFEM_MATRIX_CORE_APPLY=0`) against
    the matrix cores, alternated on one node, state 1e-3 x; ctest 119/119:

    | apply (Q2) | generated | matrix cores | |
    |---|---|---|---|
    | linear elasticity, affine | 7.92 ms | 2.37 ms | 3.3x |
    | linear elasticity, isoparametric | 27.2 ms | 4.69 ms | 5.8x |
    | Neo-Hookean Ogden, affine | 14.7 ms | 4.04 ms | 3.6x |
    | Neo-Hookean Ogden, isoparametric | 45.7 ms | 13.2 ms | 3.5x |
    | Saint Venant-Kirchhoff, isoparametric | 42.6 ms | 12.0 ms | 3.5x |
    | modified Mooney-Rivlin, isoparametric | 170 ms | 39.0 ms | 4.4x |

    | apply (Q3, 216 k elements, 17.8 M dofs, state 1e-5 x) | generated | matrix cores | |
    |---|---|---|---|
    | Saint Venant-Kirchhoff, affine | 11.5 ms | 3.35 ms | 3.4x |
    | Saint Venant-Kirchhoff, isoparametric | 37.0 ms | 9.89 ms | 3.7x |
    | modified Mooney-Rivlin, affine | 39.2 ms | 7.24 ms | 5.4x |
    | modified Mooney-Rivlin, isoparametric | 93.5 ms | 15.9 ms | 5.9x |

    `|y|` and x^T y agree to rounding where finite. At 1e-3 x the isoparametric Neo-Hookean and
    Mooney-Rivlin applies give NaN on both paths (an inadmissible state at those points); at
    u = 0 and 1e-6 x (7^3 elements) the generated kernel, the matrix cores and the host agree to
    15-16 digits for every material and geometry.
- Step 5a, the level-4 apply on the FP64 matrix cores (`..._apply_level4_mfma_kernel`, HIP only,
  the HIP default; `SFEM_HEX8_LINEAR_ELASTICITY_LEVEL4_KERNEL=2` the warp kernel): a wavefront per
  macro element, whose 64 micro elements are four 16-row tiles of Y = U K^T and whose 64 lanes are
  K's vertex pairs, so each wave builds its own K^T in LDS. 44^3 macro elements: 1.39-1.42 ms
  (two waves a block; 1.47-1.49 with four, 1.40-1.47 with eight) against 6.75-6.86 ms for the warp
  kernel and GH200's 1.5 ms. The two differ by ~2e-11 relative: the warp kernel's 2-point rule is
  written to 10 digits, this one uses the exact reference gradients. MaMAL 136 M alternated three
  times on one node: 25.51, 25.52, 25.60 s against 29.11, 28.55, 28.73 s; ctest 119/119.
  - Kernel table after it (136 M, GPU time): level-8 apply 7.07 s (8.8 ms a call), HEX8 coarse
    apply 2.22 s (87 us; GH200 26 us), `taxpby` 2.06 s (as GH200), `sbv3_apply` 1.19 s (44 us;
    GH200 41 us), level-4 apply 0.76 s (1.5 ms, as GH200), `tdot` 0.62 s, level-2 apply 0.51 s.
- Step 3a, HIP's register cap: a HIP kernel without launch bounds is compiled for blocks of 1024
  threads, which caps a thread at 128 VGPRs (CUDA allows 255), so kernels that fit on NVIDIA spill
  to scratch on AMD. A survey of the HIP build (each source compiled device-only with the register
  report; 174 of 251 sources before the time limit) found 908 of 1812 kernels spilling, almost all
  at exactly 128 VGPRs (generated: 812 of 873, median 276 spilled VGPRs). `SFEM_HIP_LAUNCH_BOUNDS(n)`
  (`sfem_cuda_base.hpp`, empty under CUDA) now gives 17 spilling hand-written kernels their block
  size (128, or the warp kernel's LEVEL^2 / LEVEL^3 threads); their spills are gone but for four cold
  ones. Kernel table (136 M): the HEX8 coarse apply 86.9 -> 39.2 us a call (2.22 -> 1.00 s; GH200
  26 us), the level-2 apply 1009 -> 413 us; ctest 119/119. On CUDA the same kernels do not spill (the
  build's ptxas report), so the bounds are HIP-only.
- Step 2d, the same lessons on GH200 (step 6's question, asked early): the generated high-order
  applies spill there too (ptxas: linear elasticity Q2 isoparametric 22.6 KB of stack, Neo-Hookean
  28.0 KB, Saint Venant-Kirchhoff Q3 60.9 KB), while the hand-written ones do not, so HIP's launch
  bounds are an AMD lesson and the cooperative shape a general one.
  - `matrix_core_apply` gains a CUDA kernel on the FP64 tensor cores (mma.sync m8n8k4 f64, sm_80
    and later; it declines below, and the generated kernel runs): a warp takes 8 elements and
    computes the transposes R^T = U^T G^T and Y^T = F^T G, so each lane's flux slot j is the A
    operand of the second GEMM's k-step j (the sum's order is free) and nothing moves between them.
    The library API is vendor-neutral (`sfem::tensor_product::matrix_core_apply`), and the generator
    emits the point and the launch for both vendors.
  - Generated mesh kernels carry `SFEM_GENERATED_LAUNCH_BOUNDS(256)` (`__launch_bounds__` under HIP,
    empty elsewhere; `targets.MESH_BLOCK_SIZE`, the launch's block size; all 1045 launches of a
    bounded kernel use it). MI300A, the generated kernels alone: linear elasticity HEX8 2.82 -> 1.63
    ms, Neo-Hookean HEX8 12.6 -> 6.3 ms, the Q2 ones 1.4-7x (Mooney-Rivlin Q2 isoparametric 170 ->
    24 ms); TET4 unchanged or 8% faster.
  - Generated kernel against the matrix units, both with the launch bounds (state 1e-5 x; x^T y
    within 1e-15 relative; Q2 1 M elements, Q3 216 k):

    | apply | MI300A off / on (ms) | | GH200 off / on (ms) | |
    |---|---|---|---|---|
    | linear elasticity Q2 affine | 4.27 / 2.35 | 1.82x | 2.83 / 2.20 | 1.29x |
    | linear elasticity Q2 isoparametric | 15.0 / 4.57 | 3.28x | 18.7 / 4.88 | 3.82x |
    | Neo-Hookean Q2 affine | 6.86 / 4.00 | 1.71x | 5.91 / 4.24 | 1.40x |
    | Neo-Hookean Q2 isoparametric | 20.5 / 13.0 | 1.58x | 24.9 / 8.81 | 2.83x |
    | Saint Venant-Kirchhoff Q2 affine | 6.67 / 3.50 | 1.90x | 5.90 / 3.58 | 1.65x |
    | Saint Venant-Kirchhoff Q2 isoparametric | 20.2 / 11.7 | 1.73x | 24.9 / 7.65 | 3.25x |
    | modified Mooney-Rivlin Q2 affine | 9.72 / 15.0 | 0.65x | 11.4 / 18.6 | 0.61x |
    | modified Mooney-Rivlin Q2 isoparametric | 24.2 / 36.0 | 0.67x | 38.0 / 38.1 | 1.00x |
    | Saint Venant-Kirchhoff Q3 affine | 5.52 / 3.26 | 1.69x | 7.02 / 3.36 | 2.09x |
    | Saint Venant-Kirchhoff Q3 isoparametric | 16.9 / 9.59 | 1.76x | 18.1 / 7.82 | 2.31x |
    | modified Mooney-Rivlin Q3 affine | 7.25 / 6.88 | 1.05x | 10.2 / 13.9 | 0.73x |
    | modified Mooney-Rivlin Q3 isoparametric | 19.6 / 15.2 | 1.29x | 24.2 / 27.8 | 0.87x |

  - So the plan launches the matrix units only for a point of at most 200 common subexpressions
    (`apply_point_temporaries`: linear elasticity 7, Saint Venant-Kirchhoff 83, Neo-Hookean 113,
    modified Mooney-Rivlin 367): Mooney-Rivlin's point, evaluated with a tile's accumulators live,
    spills. Its generated kernel, with the launch bounds, is the faster on both vendors at Q2.
- Seen on the way, not part of this plan: on GH200 the level-8 warp kernel (2) never launches
  (231 registers per thread, 242 on main, 512 threads per block: 118 k registers against 64 k
  per SM) and returns zeros without an error; it is not the default on CUDA.

Out of scope: the host-side set-up (`to_semistructured`, `SPMG::init_solver`, 1.3-2.5x), which runs
on the APU's 24 cores against Grace's 72.
