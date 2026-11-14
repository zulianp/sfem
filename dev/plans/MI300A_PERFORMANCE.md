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
- Seen on the way, not part of this plan: on GH200 the level-8 warp kernel (2) never launches
  (231 registers per thread, 242 on main, 512 threads per block: 118 k registers against 64 k
  per SM) and returns zeros without an error; it is not the default on CUDA.

Out of scope: the host-side set-up (`to_semistructured`, `SPMG::init_solver`, 1.3-2.5x), which runs
on the APU's 24 cores against Grace's 72.
