# Compressed quadrature-point partial assembly: results

The spike behind `docs/compressed_partial_assembly`. Every comparison is against BSR, with
double- and single-precision values; the exact matrix-free and the projected (P0) operators are
secondary rows. Material: neo-Hookean Ogden, lambda = 2.2, mu = 2.3333, unless a Poisson ratio is
given (then E = 1).

## What is here

| file | what |
|---|---|
| `make_qp_kernels.py` | Emits, per element, the fixed stages of the application (`qp_element.hpp`), the 81-term contraction (`qp_contract.hpp`), and the set-up kernel (`qp_setup.hpp`). The set-up kernel is the generated P0 tangent kernel with its point sum replaced by per-point writes (or P2-moment accumulation) and a call to the encoder. |
| `qp_store.hpp` | The store: codes (f64, f32, f16, int8), schemes (plain, per component, mean + deviation), layouts (SoA, blocked), and the fused encoder. |
| `qp_apply.hpp` | The application: one element-block body, an atomic scatter (`apply`) and the packed two-pass scatter (`apply_packed`). Two loop forms, `Lanes` (lane loop innermost) and `Registers`. |
| `bsr_quant.hpp` | The assembled matrix (BSR) with quantised values: int8 with an f32 scale per block, or f16 with one power-of-two scale; full, difference or upper-only form; SELL-C-sigma layout (16 rows side by side); upper-only with int32 columns or uint16 offsets plus a remainder, encodable in place over the f32 values. |
| `bench_qp.cpp` | Throughput, set-up time, memory and errors of every variant against BSR f64/f32, exact MF and P0. |
| `newton_qp.exe.cpp` | A playground run with the material's inexact pair replaced by the qp kernels. |
| `cases/*.yaml` | The Newton study: a box stretched by 50% and sheared, 4 quasi-static steps. |
| `cost_model.py` | The model the paper's tables come from. |

Build: `cmake -S spikes/inexact_apply_compare -B build-spike-E -DSPIKE_ELEMENT=E` (E = TET4, TET10,
HEX8); targets `bench_qp` and `newton_qp`. `-DSPIKE_QP_ELEMENT=TET10Q4` or `TET10P11` selects the
other TET10 stores.

## Correctness

- **The uncompressed store is the exact operator.** f64 against the exact matrix-free apply on a
  warped state (w = 0.4): TET4 1.9e-16, HEX8 2.2e-16 (relative l2), on every mesh size tried.
- **TET10 integrates with the 11-point rule.** In the playground, BSR, the matrix-free apply and the
  residual all use it; only the affine exact apply the harness calls uses a 4-point rule, and the
  two differ by 1.4e-4. The default TET10 store is therefore the P2 moments of the 11-point rule
  (450 numbers), checked against BSR f64. An 11-point store of point values (`TET10P11`, 495
  numbers) gives the same difference to BSR to every printed digit (4.06e-07 l2, 3.21e-09 energy):
  the moments reproduce the point values. The remaining 4e-7 is single-precision geometry (BSR
  recomputes the Jacobian from float coordinates; the qp set-up reads a float adjugate), the same
  order as BSR against the exact apply on HEX8 (1.9e-6).
- **Packed and atomic scatters agree** to the digits printed for every variant.
- **Newton reproduces the exact operator.** With the f64 store, Newton and CG iteration counts equal
  BSR's and the matrix-free operator's on all three elements (below).

## Encoding error against the Poisson ratio

Relative energy error |h^T (H~ - H) h| / h^T H h, white-noise h, warp 0.4, laptop, small meshes.

| element | nu | f16 | int8 / element | int8 / component | int8 mean+dev | P0 |
|---|---|---|---|---|---|---|
| HEX8 | 0.30 | 8.4e-06 | 4.8e-04 | 7.0e-04 | 6.5e-09 | 6.1e-07 |
| HEX8 | 0.45 | 1.8e-07 | 1.2e-03 | 6.7e-04 | 6.6e-08 | 1.3e-06 |
| HEX8 | 0.49 | 5.8e-07 | 5.6e-03 | 6.7e-04 | 1.6e-07 | 1.7e-06 |
| TET10 (4-pt) | 0.30 | 4.6e-06 | 1.1e-04 | 1.0e-04 | 3.7e-07 | 5.2e-05 |
| TET10 (4-pt) | 0.45 | 4.2e-06 | 2.6e-04 | 1.7e-04 | 6.3e-07 | 1.0e-04 |
| TET10 (4-pt) | 0.49 | 4.0e-06 | 4.6e-04 | 2.3e-04 | 8.5e-07 | 1.4e-04 |

- int8 with one scale per element grows with lambda/mu, as the error model predicts.
- f16 and mean + deviation do not.
- Per-component scales cost 45 floats per element and lose to mean + deviation at the same size.

## Throughput on Grace

One GH200 Grace, 72 threads, `OMP_PROC_BIND=true`, `OMP_PLACES=cores`, gcc 13.3,
`-mcpu=neoverse-v2`, 10,328,853 dof, morton3 ordering, best of 3 x 10. Atomic scatter.
"blk" is the blocked layout, "a32" single-precision arithmetic.

| | TET4 MDOF/s | TET10 MDOF/s | HEX8 MDOF/s |
|---|---|---|---|
| BSR f64 | 993 | 545 | 571 |
| BSR f32 | 1857 | 1061 | 1029 |
| exact matrix-free | 201 | 406 (4-point rule) | 225 |
| P0 f32 (approximate) | 318 | 847 | 703 |
| qp f64 SoA a64 | 262 | 241 | 253 |
| qp f16 blk a32, lanes | 393 | 736 | 735 |
| qp f16 blk a32, registers | 314 | 409 | 425 |
| qp int8/el blk a32, lanes | 417 | 804 | 765 |
| qp int8 mean+dev blk a32, lanes | n/a | 720 | 659 |
| qp int8 mean+dev blk a64, lanes | n/a | 476 | 472 |

| set-up, ms | TET4 | TET10 | HEX8 |
|---|---|---|---|
| BSR f64 / f32 | 348 / 316 | 471 / 409 | 394 / 336 |
| qp (any encoding) | 69-77 | 187-204 | 102-122 |
| P0 | 52 | 68 | 62 |

| operator memory, B/dof | TET4 | TET10 | HEX8 |
|---|---|---|---|
| BSR f64 / f32 | 377 / 199 | 721 / 380 | 676 / 357 |
| qp f16 | 184 | 222 | 237 |
| qp int8 mean+dev | n/a | 155 | 178 |
| qp int8 / element | 96 | 111 | 119 |

- **gcc vectorises the `Lanes` form, not the `Registers` form** (1.7x on TET10 and HEX8); clang on
  the M1 preferred `Registers` for the plain codecs. `Lanes` is the form.
- **Single-precision arithmetic pays** 1.4-1.5x over double for the same store.
- **The blocked layout pays** over SoA (f16: 736 against 569 on TET10).
- **TET10 and HEX8:** the compressed store is 1.2-1.5x faster than BSR f64 and 0.64-0.76x of BSR
  f32 per application, at 2.0-4.6x less memory than BSR f32/f64 and 2-3.5x cheaper set-up.
- **TET4:** BSR wins clearly (2.4x f64, 4.4x f32). Two elements per dof and 15 blocks per row make
  the sparse matrix cheap and the element loop expensive; the P0 kernel is no better (318).

## Newton (laptop, small meshes)

`cases/*_stretch_shear.yaml`: 4 quasi-static steps, Newton rtol 1e-8, CG rtol 1e-3, no
preconditioner for any operator (the generated material has no Hessian diagonal). Totals over
the 4 steps, summed from the step lines: Newton iterations / CG iterations. M1, 8 threads.
HEX8 is the 8^3 box (2187 dof); TET4 and TET10 are smesh's 6^3 TET4 cube (12 tetrahedra per cell),
TET10 promoted from it.

| | HEX8 | TET10 | TET4 |
|---|---|---|---|
| BSR f64 | 16 / 493 | 14 / 1235 | 15 / 589 |
| matrix-free | 16 / 493 | 14 / 1235 | 15 / 589 |
| P0 (projected) | 21 / 577 | 23 / 1611 | 15 / 589 |
| qp f64 | 16 / 493 | 14 / 1235 | 15 / 589 |
| qp f16 | 16 / 494 | 15 / 1336 | 15 / 590 |
| qp int8 / element | 19 / 611 | 33 / 2736 | 20 / 790 |
| qp int8 mean+dev | 16 / 493 | 16 / 1408 | n/a |

- **The projected operator costs Newton iterations** on HEX8 (+31%) and TET10 (+64%) under large
  deformation, and nothing on TET4, where it is exact.
- **The f64 qp store reproduces the exact operator's iteration counts** on all three elements.
- **int8 with one scale per element is not usable** as a Newton operator.
- **f16 and int8 mean + deviation** keep the exact counts on HEX8 and TET4 and cost 1-2 Newton
  iterations (8-14% more CG iterations) on TET10.

## Packed scatter on Grace

The library's two-pass packed layout (deterministic, no atomics), same runs as above; logs in
`measured/bench_*_packed_*.out` and `measured/bench_*_meandev_*.out`. The spike's packs are 512
elements; the library's (sized to 16-bit node indices) need not align with the store's blocks,
and the packed apply splits its blocks at store-block boundaries.

| MDOF/s, 10,328,853 dof | TET4 | TET10 | HEX8 |
|---|---|---|---|
| BSR f64 | 993 | 545 | 571 |
| BSR f32 | 1857 | 1061 | 1029 |
| P0 f32 packed (approximate) | 541 | 1261 | 1078 |
| qp f32 packed | 773 | 750 | 739 |
| qp f16 packed | 979 | 1044 | 1080 |
| qp f16 mean+dev packed | n/a | 885 | 881 |
| qp int8 mean+dev packed | n/a | 970 | 952 |
| qp int8 / element packed | 1191 | 1133 | 1172 |

- **Packing is worth 1.3-1.5x on TET10 and HEX8 and 2.3x on TET4** over the atomic scatter.
- **On TET10 and HEX8 the compressed store applies at the speed of BSR f32** (0.83-1.05x) and
  1.6-2.0x faster than BSR f64, from 2.4-4.6x less memory than BSR f64.
- **On TET4 BSR f32 stays 1.9x ahead.**

## Newton on Grace, 1.07-1.08M dof

`measured/newton_*_1M_*.out`; cases as above with the load halved (25% stretch) so that the fine
meshes converge without step cuts; TET10 in 16 steps of 0.0625 (coarser increments diverge from
the boundary jump on fine P2 meshes, for every operator). HEX8 70^3 box; TET4 smesh cube 56^3;
TET10 promoted from the 28^3 cube. Newton iterations / CG iterations, 72 threads.

| | HEX8 | TET10 | TET4 |
|---|---|---|---|
| BSR f64 | 13 / 2892 | 48 / 17822 | 12 / 3929 |
| BSR f32 | 13 / 2892 | 48 / 17821 | 12 / 3929 |
| matrix-free | 13 / 2892 | 48 / 17822 | 12 / 3929 |
| P0 | 16 / 3419 | 74 / 22841 | 12 / 3929 |
| qp f64 | 13 / 2892 | 48 / 17822 | 12 / 3929 |
| qp f16 | 13 / 2884 | 48 / 17470 | 13 / 4305 |
| qp f16 mean+dev | 13 / 2893 | 48 / 17824 | n/a |
| qp int8 mean+dev | 13 / 2893 | 48 / 17763 | n/a |
| qp int8 / element | 20 / 4437 | 125 / 43658 | 20 / 6473 |

- **The compressed stores keep the exact Newton iteration count** on HEX8 and TET10, where the
  projected operator costs 23% and 54% more.
- **The run times of these logs did not compare the operators.** The library sizes packs to its
  16-bit node indices (8167 HEX8 elements), which at 1M dof leaves 42 packs for 72 threads; the qp
  apply (and the library's packed P0 apply) then ran at half its saturated rate. The campaign
  below sets the pack size for every operator.

## The campaign with the optimised kernels (current)

`measured/` holds the logs of one campaign (Alps jobs 5180953-5180991, 5181639-5181642) with the
optimised kernels (lane-form HEX27 stages in float, vectorised element-matrix set-up, persistent
node-major pack scratch) and packs of about 20 per thread for every packed operator, in the
benchmarks and in Newton (`SMESH_ELEMENTS_PER_PACK` = max(16, min(8192, elements / 1440)), rounded
down to 16). The earlier logs, including those the tables above quote, are in `measured/pre_opt/`.
The paper's tables, macros and figures are generated from `measured/`:

    python results_tables.py measured/ ../../../docs/compressed_partial_assembly/tables measured_qsym measured_qbsr
    python paper_figures.py measured/ ../../../docs/compressed_partial_assembly/figures

Newton at 10M dof is split over several logs per element (one per group of operators); the
scripts merge them. On TET4 at 10M every operator meets a diverging load step, and the iterations
spent before the divergence is detected differ by operator and decide the time.

## Quantised BSR (int8 per block, f16), Grace

`measured_qbsr/` (Alps jobs 5184487-5184506). The assembled matrix, assembled in f32 and then
encoded (`bsr_quant.hpp`). Difference form: only the off-diagonal blocks, applied as
y_i = sum_{j != i} K_ij (x_j - x_i), so translations stay in the kernel. Full form: every block.
BSR f32 at 10M dof and the element matrices come from `measured/`.

Application, 10.3M dof, 72 threads (MDOF/s, B/dof):

| element | BSR f32 | BSR i8 diff | BSR i8 full | BSR f16 diff | BSR f16 full | ea i8 pair |
|---|---|---|---|---|---|---|
| HEX8  | 1021, 357 | 1874, 147 | 1808, 153 | 1625, 190 | 1715, 197 | 1719, 93 |
| TET4  | 1867, 199 | 3433, 80  | 3182, 86  | 3005, 103 | 3015, 110 | 1159, 126 |
| TET10 | 1058, 380 | 1736, 161 | 1676, 166 | 1568, 208 | 1559, 215 | 1526, 111 |
| HEX27 | 461, 842  | 809, 356  | 828, 362  | 791, 461  | 753, 468  | 1629, 144 |

Relative l2 error of the application: int8 3.3e-3 to 4.2e-3, f16 1.1e-4 to 2.5e-4. Set-up: f32
assembly plus 10-40% for the encoding. The Newton driver keeps the f32 values between updates,
so its peak memory is BSR f32 plus the store; the store alone is what the B/dof column counts.

Newton, total iterations / time to solution (s):

| case | BSR f32 | BSR i8 diff | BSR f16 diff | BSR f16 full | ea f16 diff | ea i8 pair |
|---|---|---|---|---|---|---|
| HEX8 1M   | 13 / 3.16 | 16 / 3.04 | 13 / 2.67 | 16 / 3.29 | 13 / 2.54 | 16 / 2.81 |
| TET4 1M   | 12 / 2.55 | 16 / 3.09 | 13 / 2.46 | 17 / 3.36 | 12 / 4.84 | 17 / 6.34 |
| TET10 1M  | 48 / 20.5 | 64 / 20.5 | 48 / 16.9 | 69 / 23.9 | 48 / 18.7 | 64 / 21.8 |
| HEX27 1M  | 24 / 53.1 | 32 / 40.6 | 24 / 40.0 | 33 / 44.3 | 24 / 23.1 | 32 / 23.3 |
| HEX8 10M  | 16 / 96.8 | 16 / 65.2 | 17 / 84.6 | 22 / 103.8 | 16 / 74.7 | 16 / 65.3 |
| TET4 10M  | 19 / 231.9 | 20 / 134.9 | 18 / 189.6 | - | 18 / 282.6 | 20 / 217.5 |
| TET10 10M | 99 / 3066 | 172 / 3006 | 118 / 2959 | - | 109 / 2719 | 138 / 3196 |
| HEX27 10M | 48 / 3407 | 64 / 2413 | 48 / 3087 | - | 48 / 1838 | 64 / 1466 |
| twist HEX8  | 64 / 18.9 | 74 / 17.1 | 64 / 15.2 | 81 / 19.8 | 64 / 14.7 | 64 / 12.5 |
| twist HEX27 | 98 / 211.0 | 103 / 112.6 | 98 / 171.6 | 117 / 136.9 | 97 / 90.9 | 103 / 62.2 |
| twist TET4  | 195 / 216.6 | 423 / 240.2 | 195 / 204.7 | 288 / 242.5 | 195 / 432.9 | 471 / 706.5 |
| twist TET10 | 273 / 791.4 | 508 / 906.2 | 273 / 729.3 | failed | 273 / 688.2 | 454 / 628.2 |

Iterations count accepted load steps only; the time includes the failed attempts of the adaptive
continuation (the twists take different paths per operator, so their time is the comparison).

- **The full int8 form fails every Newton run** (line search fails in the first load step, 5
  halvings). The difference form is required for int8. The full f16 form converges but needs
  15-45% more iterations and fails the TET10 twist.
- **f16 difference form** takes the same iterations as BSR f32 in 7 of 12 cases, one more or
  fewer in 3, and 19% more on TET10 at 10M (118 against 99); it is 1.04-1.33x faster at
  0.52-0.55x the memory.
- **int8 difference form** applies 1.6-1.8x faster than BSR f32 at 0.41x the memory, but takes
  about 33% more Newton iterations (74% on TET10 at 10M), as the element matrices' pair scale
  does. It wins in time on HEX8, TET4 and HEX27 stretch-shear at 10M (1.4-1.7x) and the HEX27
  twist (1.9x), breaks even on TET10 and loses on TET4 at 1M and on the TET4 and TET10 twists.
- **Against the element matrices:** quantised BSR wins on TET4 (the element matrices apply at a
  third of its rate), ties on HEX8 and TET10, and loses on HEX27 by 1.6x, where the element
  matrices fetch each of the 27 nodes once per element.

## Upper-only quantised BSR, offsets, in-place encoding (Grace)

`measured_qsym/` (Alps jobs 5185726-5185750; memory runs 5186030-5186031).  Upper only: each pair
i < j stored once, d = x_j - x_i, y_i += K_ij d, y_j -= K_ij^T d; symmetric by construction, no
symmetrisation.  Rows cut into one part per thread; blocks whose column lies beyond the part, or
beyond the reach of uint16 offsets, go to a per-part remainder (5-9% of the blocks at 10M with 72
parts); updates beyond the part go to halo slots added after a barrier.  Deterministic, no atomics.

Application, 10.3M dof (MDOF/s, B/dof):

| element | BSR f32 | i8 diff | i8 upper | i8 upper u16 | f16 upper u16 | ea i8 pair |
|---|---|---|---|---|---|---|
| HEX8  | 1027, 357 | 1881, 147 | 2154, 78  | 2122, 70  | 1961, 92  | 1761, 93  |
| TET4  | 1822, 199 | 3371, 80  | 3803, 43  | 3874, 38  | 3643, 50  | 1122, 126 |
| TET10 | 1057, 380 | 1714, 161 | 1920, 88  | 1906, 79  | 1685, 103 | 1514, 111 |
| HEX27 | 457, 842  | 748, 356  | 831, 194  | 847, 175  | 797, 229  | 1473, 144 |

Fastest time to solution per element and case (Newton iterations, speed-up over BSR f32, B/dof):

| element | stretch 1M | stretch 10M | twist 1M |
|---|---|---|---|
| HEX8  | BSR f16 upper 2.33 s, 13 (1.36x), 100 | BSR i8 upper 58.1 s, 16 (1.67x), 78 | ea i8 pair 12.5 s, 64 (1.46x), 93 |
| TET4  | BSR f16 upper 2.43 s, 13 (1.05x), 55 | BSR f16 upper u16 105.5 s, 18 (2.20x), 50 | BSR i8 upper 178 s, 320 (1.22x), 43 |
| TET10 | BSR f16 upper u16 14.8 s, 48 (1.39x), 103 | BSR f16 upper u16 2462 s, 117 (1.25x), 103 | BSR f16 upper u16 626 s, 273 (1.26x), 103 |
| HEX27 | ea f16 23.1 s, 24 (2.30x), 258 | ea i8 pair 1466 s, 64 (2.32x), 144 | ea i8 pair 62.2 s, 103 (2.99x), 144 |

- f16 upper-only takes the iterations of BSR f32 except on TET10 at 10M (117 against 99, as the
  f16 difference form); int8 takes about a third more, as the element matrices' pair scale.
- uint16 offsets save about 10% of the store at the same speed (within 2%).
- In place: the records are written over the f32 values; identical results and set-up time.
  Peak resident memory of the whole Newton run at 10M (newton_qp, only what each operator needs):
  HEX8 BSR f32 6711 MB, BSR i8 upper u16 7405, f16 7450, ea f16 2682, ea i8 pair 1931, P0 933;
  TET4 BSR f32 5768, BSR i8 upper u16 6339, f16 6381, ea i8 pair 4758, P0 6532.  The quantised
  BSR peak follows BSR f32 (+10%): the full-pattern f32 assembly and the construction of the node
  graph set it, not the store.  On TET4 every operator adds about 3 GB during the run (not yet
  attributed).

## GPU: one H100 of a GH200 node (Alps)

`measured_gpu/`.  Device code in `cuda/`, built in `build-spike-cuda-<element>` against the CUDA
libsfem (`build-cuda`), on the kernels the CPU trees generated.

### Device Newton with the library's operators (gate G0)

Time-dependent Dirichlet conditions and rotations now run on the device (the host conditions'
`set_time`, then `refresh_device`), and the device Dirichlet constraint has `value`, `value_steps`
and typed BSR identity rows (it had none, so a device Newton with an energy line search aborted).
`g0_small_5190028.out`: HEX8 and PROTEUS_HEX27 at 512 elements, stretch and 90 degree twist, BSR
f64 and f32: Newton counts identical to the host in every step, CG within 3 (the host's own
run-to-run spread), final displacements within 3e-11 relative.  `g0_bsr_1M_5190179.out`: device
BSR f64 and f32 on the CPU campaign's 1M-dof stretch cases reproduce the CPU logs' per-step
Newton and CG counts exactly (HEX27's step halvings included), except TET10 f32's last three steps
(CG within 17).  The library's device BSR is its own kernel, one thread per scalar row (cuSPARSE is
not built: `SFEM_ENABLE_CUSPARSE` has no CMake option); cuSPARSE `bsrmv` f64 is slower than it.

### Quantised BSR layout (kernel-shape test, `qshape_*`)

`cuda/qbsr_shape`: the node graph of each element at 10.3M dof, random symmetric blocks quantised
once, every layout SELL-32 (one warp per group of rows), sigma = 1024 rows; the upper columns are
int16-offset packed records.  MDOF/s (B/dof):

| element | library BSR f32 | f32 full i32 (scalar) | i8 full off16 packed | i8 full i32 packed | i8 upper atomic | i8 upper gather | f16 full off16 packed | f16 full i32 packed |
|---|---|---|---|---|---|---|---|---|
| HEX8  | 5457 (357) | 9232 (344) | 18623 (140) | 17362 (173) | 6379 (72)  | 9750 (152)  | 14534 (175) | 14719 (207) |
| TET4  | 10021 (199) | 16668 (186) | 32648 (76) | 30031 (94)  | 11797 (39) | 28588 (60)  | 27306 (95)  | 26093 (112) |
| TET10 | 4222 (380) | 7237 (377) | 12477 (153) | 12642 (189) | 5740 (79)  | 7269 (139)  | 10374 (191) | 10653 (227) |
| HEX27 | 2004 (842) | 3405 (838) | 5714 (342)  | 6069 (420)  | 2593 (177) | 2633 (336)  | 4804 (428)  | 5071 (503)  |

- The GPU reverses the CPU's choice: the full difference form (each pair stored twice, no scatter)
  is 2-3x faster than the upper form with `atomicAdd` and 1.3-2.2x faster than the upper form that
  gathers the lower blocks from their partner's record.  Every layout is deterministic except the
  atomic one (bitwise repeats held here, but the order of the atomics is not fixed).
- Packed records (int8: codes, the int16 offset and the scale in 16 bytes, one 128-bit load; f16:
  20 bytes) beat one load per entry by 1.2-1.3x.
- int16 offsets plus a remainder (3-6% of the blocks) against int32 columns: 7-9% faster on HEX8 and
  TET4, within 3% on TET10 (smaller, so chosen), 6% slower on HEX27 (rows of 125 blocks, 6%
  remainder).  Decision: full difference form, packed records, sigma = 1024 rows, int16 offsets
  on TET4, TET10 and HEX8, int32 columns on HEX27.
- The library's BSR f32 kernel moves 2.0 TB/s; the same values in the SELL layout move 3.4 TB/s
  (1.7x), the int8 records another 2x on top.  The BSR f32 Newton baseline on the GPU is the
  library's kernel (what a device Newton runs); the SELL f32 row says how much of a quantised
  format's gain is layout rather than bytes.

### Element stores on the GPU (`bench_qp_cuda`, `eshape_*`)

`cuda/qp_device.cu`: the stores' host layouts with blocks of 32 elements (one warp).  TET4, TET10 and
HEX8 run one thread per element on the generated stages with one lane; PROTEUS_HEX27 runs one thread
per point (point stores, codes [element][component][point]) or one block of 128 threads per element
(element matrices).  The set-up runs on the device, in double, and gives the host's codes bit for
bit (TET10 element matrices: 1 code in 10^9 differs by one step).  The application's shape was
measured at 10.3M dof: atomicAdd per element node against a block-local scatter (the block of 128
elements gathers its distinct nodes into shared memory, accumulates there, and adds each node once),
and 1, 2 or 4 resident blocks per SM (the register cap).  MDOF/s:

| element | qp f16 | qp i8 mean+dev | ea f16 | ea i8 pair |
|---|---|---|---|---|
| HEX8, atomic / best | 8348 / 8897 (block, 4) | 8166 (atomic, 1) | 9738 / 11281 (block, 4) | 9709 (atomic, 1) |
| TET4, atomic / best | 3370 / 5101 (block, 1) | - | 3305 / 4942 (block, 1) | 3269 / 4074 (block, 1) |
| TET10, atomic / best | 8344 / 8807 (block, 4) | 8213 (atomic, 1) | 8701 (atomic, 1) | 8374 (atomic, 1) |

The block-local scatter pays where many elements share a node (TET4: six elements per node, 12
atomics per element); elsewhere it trades the global atomics for shared-memory atomics and a
gather through shared memory, and wins only with the register cap.  The stores take these shapes
(`create_for` in `qp_device.cu`).  The element stores reach 1.2-1.9 TB/s against the quantised
BSR's 2.9-3.0: latency and atomics, not bytes, limit them on the GPU.

### The library's device BSR product (`bsrgroup_*`)

The BSR f32 baseline on the GPU is the library's product, tuned for it: `sfem::d_bsr_spmv_grouped`
(`src/algebra/bsr/cuda/sfem_BSR_cuda.cu`), a group of G threads per block row, each reading whole
3x3 blocks (consecutive threads consecutive blocks, so a warp's loads of the values are
contiguous) and the partial sums added by shuffles in a fixed order; `d_bsr_spmv_native` and
`d_bsr_spmv_ranges` run it for blocks up to 4x4, so the device BSR tangent uses it.  The previous
kernel ran one thread per scalar row.  MDOF/s at 10.3M dof, f32 values (f64 in parentheses):

| element | previous kernel | G = 4 | G = 8 | G = 16 | G = 32 | SELL f32 (qshape) |
|---|---|---|---|---|---|---|
| HEX8  | 5457 (3278) | 7750 (3836) | 7688 (4383) | 8344 (5170) | 7701 (5175) | 9232 |
| TET4  | 10021 (6051) | 12461 (6342) | 15003 (8776) | 14619 (9090) | 9625 (8054) | 16668 |
| TET10 | 4222 (2862) | 7133 (4196) | 7567 (4484) | 7738 (4837) | 6997 (4899) | 7237 |
| HEX27 | 2004 (1316) | 3096 (1675) | 3386 (1816) | 4022 (2139) | 3725 (2256) | 3405 |

G = 16 is the default (within 3% of the best G for f32 everywhere, within about 5% for f64): 1.5-2.0x
the previous kernel, about 3.1-3.6 TB/s, and within 10% of the same values in the SELL layout
(ahead of it on TET10 and HEX27).  cuSPARSE's bsrmv is measured beside it (bench_qp_cuda).

The comparison with cuSPARSE in detail (`bsrmv`, the generic SpMV on BSR and on scalar CSR, and
Nsight Compute counters for why): `CUSPARSE_BSR.md`.

### The campaign on the tuned baseline (current)

`measured_gpu/` (Alps jobs 5195091-5195130, 5195780-5195783; the first campaign, on the previous
BSR kernel, is in `measured_gpu/v1/`): the bench, the application sweep, Newton stretch at 1M and
10M dof, the twist and the Newton sweep, every format, against the G = 16 library BSR, with
cuSPARSE `bsrmv` f64 and f32 rows beside it in the bench.  The paper's tables, macros and figures:

    python results_tables.py --machine gpu measured_gpu/ ../../../docs/compressed_partial_assembly/tables
    python paper_figures.py --machine gpu measured_gpu/ ../../../docs/compressed_partial_assembly/figures

- The library BSR runs at 3.0-3.5 TB/s and 1.9-3.8x the speed of cuSPARSE at the same precision.
  cuSPARSE's f32 product takes f32 vectors only.
- Newton counts against Grace: identical per step in every format on the 1M stretch (TET4's int8
  element matrix one Newton step fewer in its last load step), and on HEX27 at 10M.  Runs with
  rejected load steps (twist, TET4 and TET10 at 10M) take other paths, BSR f64 included: the
  device's summation order differs.  Every run ends with status 0.
- Time to solution at 10M against BSR f32 (`summary_gpu.tex`): quantized BSR fastest on TET4 (f16)
  and HEX8 (i8), the point stores on TET10 (f16) and HEX27 (i8 mean+dev).  The quantized matrix's
  set-up (f32 assembly and encoding) is 18-22% of its run on HEX8, TET10 and HEX27, against 7-9% on
  Grace; the point stores' about 1%.
- Inside Newton the products run at 0.88-0.92 of the bench rate on HEX8 (same mesh).  On TET4 the
  Newton mesh (`meshes/tet4_120`, twelve tetrahedra per cell around a centre node) is not the
  bench's six-tetrahedra box.  `meshbench_tet4_120_5196246.out` is the bench on that mesh
  (`QP_MESH=<folder>`, morton3 as the Newton case): every operator is slower there (int8 quantized
  BSR 0.78 of its box rate, BSR f32 0.84, the element formats 0.80-0.81), the quantized matrix still
  1.60x BSR f32 (1.72 on the box).  The mesh, not the Newton run, makes the gap.
- The HEX27 application sweep first stopped at n = 75: n = 100 segfaulted in the host BSR assembly.
  The generated BSR scatter formed the value offset `entries[...] * NC * NC` in `count_t` (int32),
  which overflows past 2^31 / 9 = 2.39e8 blocks (n = 75: 2.28e8; n = 100: 5.1e8).  Fixed in the
  generator (`energy_codegen.py`, `residual_codegen.py`: `(ptrdiff_t)entries[...]`); all operators
  regenerated, the diff is exactly that line in 114 files; the spike's own kernel copies were
  regenerated beside the old ones (`build-spike-<E>/gen2` on Alps, one line differing per element)
  and the CUDA trees point at them.  HEX8 n = 200 (2.2e8 blocks) and TET10 n = 100 (2.3e8) were just
  under the limit.  The CPU trees' `gen/` still has the 32-bit offset; no CPU size reaches it.
  cuSPARSE's legacy `bsrmv` indexes values with 32 bits as well: the bench skips it beyond 2^31
  values and says so.  Sweep log: `bsweepauto_hex27_5196335.out` (the failed one is in
  `measured_gpu/superseded/`).
- The bench checks the status of every timed call (set-up, assembly, product, cuSPARSE) and stops on
  a failure instead of timing it.
- nid005669 gives illegal memory accesses in any job (three runs lost in this campaign, one in
  the first); `gpu_campaign.sh` excludes it.  Its runs are listed in `gpu_ids_failed.txt` on Alps
  and not fetched.


## The rerun of 2026-10-29: one output contract, int32 columns on the GPU, no retried load steps

Three faults in the campaigns above, found while profiling on application meshes; the rerun (Alps
jobs 5199790-5199848, `rerun_campaign.sh`) replaces every log they affect, and the replaced logs
move to `superseded/` in their folders.

- **The benches timed the matrices writing y, the stores adding into it.**  Inside Newton every
  operator applies y += A x (`BSRTangent::op`'s default scale 1; `newton_qp` calls the quantized
  matrix with ADD; the stores scatter into y).  `bench_qp` and `bench_qp_cuda` timed BSR (and the
  Grace quantized matrix) as y = A x, which skips the read of y.  On the H100 that read costs the
  library BSR product 12-17% (HEX27 4%), the quantized matrix 20-30%: the GPU throughput table
  understated the compressed formats (TET4 int8 quantized 1.71x BSR f32 as measured, 2.07x with both
  adding).  Both benches now time every format as y += A x (output zeroed before the first call), and
  cuSPARSE with beta = 1.
- **The GPU quantized matrix's layout was chosen writing y.**  `qbsr_shape` measured int16 offsets
  plus a remainder by row ahead of int32 columns; with y += A x (`QS_ACCUMULATE=1`) int32 columns
  are ahead on every mesh measured: +8% on the TET4 box, +15-30% on the application meshes, whose
  morton-ordered unstructured graphs send 4-11% of the blocks to the slower remainder kernel
  (`measured_real/realqs_5199067.out`, job 5199166).  The device format (`cuda/qp_device.cu`) now has
  int32 columns only, as HEX27 had.
- **Several Newton cases retried most load steps.**  The step control retries the full dt after a
  halved step succeeds, so a case whose dt is too large fails and halves on every step: TET10 at 1e7
  31 failed attempts per run (about 300 discarded Newton iterations against about 100 counted),
  HEX27 at 1e6 and 1e7, and the TET4, TET10 and HEX27 twists (the TET10 twist 229 failed attempts per
  run).  The counts reported converged steps only, and the time to solution was mostly retries.  The
  cases now take a step every format converges in, from the smallest step the old runs needed:
  HEX27 stretch dt 1/8 (case file; 1e7: `DT=1/32`), TET4 and TET10 stretch at 1e7 `DT=1/8` and
  `1/32`, twists TET4 1/128, TET10 1/256, HEX27 1/32 (case files).  HEX8 and the 1e6 stretch cases of
  TET4 and TET10 never retried and are unchanged.
- The TET10 fp16 formats' extra Newton iterations at 1e7 (116-123 against 99) were a threshold
  effect: with CG at 1e-3 each Newton iteration gains about three digits, so BSR's third iterate lands
  at 0.44-0.98 of the 1e-8 tolerance and an encoding error near 1e-4 puts it just above, costing a
  fourth iteration in most steps.

**What the rerun gave** (logs in `measured/`, `measured_gpu/`, `measured_real/realcpu_5199799/`,
`realcpu_5199881/`, `realgpu_5199831/`; the replaced ones in each folder's `superseded/pre_rerun/`):

- No run of the paper's eight formats rejects a load step.  The only retries left are formats
  outside them: the element matrix with one int8 scale per element (HEX27 stretch at 1e6, five
  failed attempts; the HEX27 twist fails, status 1) and the full fp16 element matrix (TET10 twist,
  two failed attempts).  The full int8 quantized matrix fails as before, by design.
- The H100 now takes Grace's Newton counts: exactly for BSR and int8 mean+dev, within a few
  iterations for the other fp16 formats, within a tenth for the int8 matrices.  The HEX27 twist
  (32 steps) spreads by about a tenth between formats on either machine, BSR f64 against f32
  included (145 against 136 on Grace).
- The threshold effect holds at 32 steps: TET10 at 1e7 on Grace, BSR ends 29 of 32 steps after three
  Newton iterations at 0.44-0.98 of the tolerance; the fp16 point store takes four in 24 steps
  (125 against 99), the int8 mean+dev store stays at 99.
- Time to solution at 1e7 against BSR f32 (the summary table): the old TET4 and TET10 figures were
  flattered by BSR's retries.  Grace: TET4 quantized fp16 0.73 (was 0.46), TET10 0.92 (0.80); HEX27
  now led by the fp16 element matrix (0.42).  H100: TET4 quantized fp16 0.73, HEX8 quantized int8
  0.59, TET10 int8 mean+dev point store 0.73, HEX27 fp16 point store 0.62.
- With y += A x the H100 int8 quantized matrix applies at 2.24, 1.80, 2.28 and 1.45x BSR f32 (TET4,
  TET10, HEX8, HEX27), more than on Grace except on HEX27; our BSR against cuSPARSE: CUSPARSE_BSR.md.
- The remainder share of the Grace quantized matrix comes from the `upper store:` diagnostic, which
  only the ubench logs print; it is a property of the layout, not of the output contract, so
  `measured_qsym/ubench_*.out` stay in place for it (their rates are shadowed by the new benches,
  checked: no other macro changes).
- Two faults behind the HEX27 bench sweep's crash at 2.4e7 dof (n=100), both fixed:
  - **The spike's kernel copies were stale.**  CMake generated `build-spike-<E>/gen/` only when it
    was missing, so the generator's 64-bit BSR offsets (de7dac7ac) never reached it: the reference
    BSR assembly computed `values[entries * 9]` in 32-bit int and, past 2^31 values, wrote out of
    bounds (SIGSEGV in `hessian_i_msoa_assemble_impl`; job 5199829 survived it by luck).  The spike
    now stamps `gen/` with a hash of the generator's sources and regenerates when it changes.  The
    regenerated kernels of all four elements differ from the old ones in that cast only, so no
    measured log changes (every other run stays below 2^31 values).
  - **The GPU trees could link a stale libsfem.**  `build_spike_cuda.sh` defaulted to `build-cuda`
    (built 2026-10-28 16:24), which predates the grouped device BSR product; the campaign's
    binaries linked `build-cuda-2`, but a rebuild with the default linked the old product (HEX27
    BSR f32 1,925 MDOF/s against 3,786) and `qbsr_shape` failed to link.  The default is now
    `build-cuda-2`; the logs of jobs 5201474 and 5201532 (stale library) are in
    `superseded/pre_rerun/`.
  - cuSPARSE 12.5's CSR product fails past 2^31 values: with 64-bit indices and 22.6 GB free, HEX27 at
    1.8e7 dof (3.4e9 values) returns an internal error from both `cusparseSpMV_preprocess` and
    `cusparseSpMV`, while 64-bit indices below 2^31 work (`QP_CSR_INDEX64=1` at n=35 and n=75).
    `bench_qp_cuda` skips the CSR rows past 2^31 values, and where the scalar copy does not fit the
    free device memory, saying which; a cuSPARSE failure now names the call and the free memory.
  - The Alps library trees `build` (Grace) and `build-cuda` predated de7dac7ac and are rebuilt from
    the current sources; a smoke run of the rebuilt HEX27 benches at 1.1e6 dof reproduces
    `bsweepauto_hex27_5199798` and `_5202324` within 4%, errors identical.
  - The sweep reruns complete (job 5202324, `measured_gpu/bsweepauto_hex27_5202324.out`).
