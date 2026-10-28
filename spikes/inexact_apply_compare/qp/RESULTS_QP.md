# Compressed quadrature-point partial assembly: results

The spike behind `docs/compressed_partial_assembly`. Every comparison is against BSR, with
double- and single-precision values; the exact matrix-free and the projected (P0) operators are
secondary rows. Material: neo-Hookean Ogden, lambda = 2.2, mu = 2.3333, unless a Poisson ratio is
given (then E = 1).

## What is here

| file | what |
|---|---|
| `make_qp_kernels.py` | Emits, per element, the fixed stages of the application (`qp_element.hpp`), the 81-term contraction (`qp_contract.hpp`), and the set-up kernel (`qp_setup.hpp`). The set-up kernel is the generated P0 tangent kernel with its point sum replaced by per-point writes (or P2-moment accumulation) and a call to the encoder. |
| `qp_store.hpp` | The store: codes (f64, f32, f16, int8), schemes (plain, per component, mean + deviation, whitened), layouts (SoA, blocked), and the fused encoder. |
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

| element | nu | f16 | int8 / element | int8 / component | int8 mean+dev | int8 whitened | P0 |
|---|---|---|---|---|---|---|---|
| HEX8 | 0.30 | 8.4e-06 | 4.8e-04 | 7.0e-04 | 6.5e-09 | 5.5e-06 | 6.1e-07 |
| HEX8 | 0.45 | 1.8e-07 | 1.2e-03 | 6.7e-04 | 6.6e-08 | 1.6e-06 | 1.3e-06 |
| HEX8 | 0.49 | 5.8e-07 | 5.6e-03 | 6.7e-04 | 1.6e-07 | nan | 1.7e-06 |
| TET10 (4-pt) | 0.30 | 4.6e-06 | 1.1e-04 | 1.0e-04 | 3.7e-07 | 1.3e-05 | 5.2e-05 |
| TET10 (4-pt) | 0.45 | 4.2e-06 | 2.6e-04 | 1.7e-04 | 6.3e-07 | 1.4e-05 | 1.0e-04 |
| TET10 (4-pt) | 0.49 | 4.0e-06 | 4.6e-04 | 2.3e-04 | 8.5e-07 | nan | 1.4e-04 |

- int8 with one scale per element grows with lambda/mu, as the error model predicts.
- f16 and mean + deviation do not.
- **Whitening is withdrawn.** The element mean is near-singular on rotations; with a small shift the
  factor amplifies the rounding (4.9e7 on HEX8 at w = 0.4, nan at nu = 0.49), with a large one it is
  no better than mean + deviation. It stays in the code behind `QP_WHITENED` and out of the runs.
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

    python results_tables.py measured/ ../../../docs/compressed_partial_assembly/tables
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

| element | BSR f32 | BSR i8 diff | BSR i8 full | BSR f16 diff | BSR f16 full | ea f16 diff | ea i8 pair |
|---|---|---|---|---|---|---|---|
| HEX8  | 1021, 357 | 1874, 147 | 1808, 153 | 1625, 190 | 1715, 197 | 1537, 166 | 1719, 93 |
| TET4  | 1867, 199 | 3433, 80  | 3182, 86  | 3005, 103 | 3015, 110 | 1129, 220 | 1159, 126 |
| TET10 | 1058, 380 | 1736, 161 | 1676, 166 | 1568, 208 | 1559, 215 | 1398, 200 | 1526, 111 |
| HEX27 | 461, 842  | 809, 356  | 828, 362  | 791, 461  | 753, 468  | 1403, 258 | 1629, 144 |

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
