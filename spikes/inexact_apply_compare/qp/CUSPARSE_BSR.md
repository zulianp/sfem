# Our BSR product against cuSPARSE on the H100

The comparison: `sfem::d_bsr_spmv_grouped` (G = 16, `src/algebra/bsr/cuda/sfem_BSR_cuda.cu`), the
BSR product the paper uses as its GPU baseline, against every cuSPARSE path that takes the same
matrix.

**In short** (every product applied as y += A x, the contract inside Newton; cuSPARSE with
beta = 1):
- cuSPARSE's BSR product, `bsrmv`, runs at 0.8–1.7 TB/s. It executes 8–12× more warp
  instructions per block than ours, so it is limited by instruction issue, not by memory. Ours
  reads the same bytes from DRAM and runs at 3.0–3.4 TB/s in f64 on the bench's box meshes:
  **1.9–2.4× faster in f64, 2.5–3.7× in f32.**
- cuSPARSE's generic `cusparseSpMV` does not take BSR (cuSPARSE 12.5.2). Its tuned path is scalar
  CSR, and that kernel is as efficient as ours: 3.0–3.4 TB/s. It loses because CSR stores a
  column index per scalar. With 1.4–1.8× more bytes per product, it is **1.4–1.5× slower in f64 and
  1.5–1.7× slower in f32.** Against cuSPARSE's deterministic CSR algorithm the gap is 1.8–2.6×.
- On application meshes (unstructured TET4 valve meshes, 0.15–7.4 M dof) the order holds: ours is
  2.0–2.2× faster than `bsrmv` in f64 and 2.4–3.1× in f32, and 1.25–1.39× faster than CSR in
  f64 and 1.26–1.74× in f32. There our kernel streams less than on the box (2.3–2.9 TB/s in f64,
  2.0–2.3 in f32, against CSR f32's 2.1–3.0), so the speed-up over CSR falls short of the byte ratio
  (see "On application meshes").
- Adding into y costs our product more than it costs CSR: on the box our f32 rate is 14–17% below
  the y = A x rate of the first version of this document (HEX27 4%), and our f32 kernel streams
  2.5–2.6 TB/s of matrix on TET4, HEX8 and TET10 (3.3 on HEX27), against CSR f32's 3.0–3.4.

So we do not reach more bandwidth than the vendor can. The vendor's BSR kernel spends too many
instructions on 3×3 blocks to reach it, and its kernel that does reach it needs a format with more
bytes.

## Set-up

- **Machine:** one H100 of an Alps GH200 node, CUDA 12.6 (cuSPARSE 12.5.2).
- **Matrices:** the neo-Hookean tangent at the bench state on the bench's structured box meshes,
  10,328,853 dof (3,442,951 block rows of 3×3), assembled on the device by our code. The same
  values go into every product. Application meshes: their own section below.

  | element | blocks | blocks per row |
  |---|---|---|
  | TET4 n=150 | 51.1 M | 14.8 |
  | HEX8 n=150 | 91.7 M | 26.6 |
  | TET10 n=75 | 97.8 M | 28.4 |
  | HEX27 n=75 | 217.1 M | 63.1 |

- **The products:**
  - **ours:** `d_bsr_spmv_native`, which runs `d_bsr_spmv_grouped` for 3×3 blocks. Values are f64
    or f32; vectors and accumulation are f64 in both cases.
  - **`bsrmv`:** `cusparse{D,S}bsrmv`, row-major blocks, on the same `rowptr`, `colidx` and values.
    Vectors are in the value type, since `bsrmv` has no f32-matrix/f64-vector mode. Its f32 row
    therefore reads half our vector bytes.
  - **SpMV BSR:** `cusparseCreateBsr` + `cusparseSpMV`. Rejected: "sparse matrix format BSR is not
    supported" (`CUSPARSE_STATUS_NOT_SUPPORTED`).
  - **SpMV CSR:** the same matrix expanded to scalar CSR (9 values and 9 column indices per block,
    32-bit indices), with `CUSPARSE_SPMV_CSR_ALG1` (the default) and `CSR_ALG2`, which cuSPARSE
    documents as bit-wise deterministic. The buffer and `cusparseSpMV_preprocess` are done outside
    the timing. Each algorithm needs its own descriptor: preprocessing ALG2 on a descriptor that
    ALG1 had already preprocessed gave wrong results (relative error 1.0).
- **Timing:** `bench_qp_cuda` (`QP_ONLY="BSR f64|BSR f32|cusparse"`, n, 10 repeats). Host clock
  around 10 calls plus a device synchronisation, best of 3. Every call's status is checked. The products agree with each
  other to round-off: in f64, each has the same error against the host reference as ours.
- **Bytes:** matrix plus graph, vectors excluded, which is what B/dof below counts. GB/s =
  MDOF/s × B/dof. Nsight Compute's DRAM counters confirm the model: TET4 f64 reads 4.0 GB, against
  3.90 GB of matrix plus 0.08 GB for x, so x is read from DRAM once.
- **Profile:** Nsight Compute (`ncu --metrics …`, default clock control, i.e. base clocks), 4
  launches per kernel, median launch reported. Its durations are longer than the bench's because of
  the clock lock; the rates below come from the bench, the counters from `ncu`.

Logs:
- `measured_gpu/bench_{tet4,hex8,tet10,hex27}_51998xx.out`: rates, y += A x (Alps jobs
  5199822–5199828, the paper's campaign).
- `measured_gpu/ncu_{TET4,PROTEUS_HEX27}_5198662.csv`: counters, untracked. Taken with y = A x
  (`measured_gpu/superseded/pre_rerun/cusparse_5198662.out`); the instruction counts and the
  limits they show do not depend on the read of y.

## Rates

MDOF/s, with GB/s in parentheses, and our speed-up over each product:

**f64**

| element | B/dof BSR / CSR | ours | bsrmv | × | SpMV CSR | × | SpMV CSR alg2 | × |
|---|---|---|---|---|---|---|---|---|
| TET4 | 377 / 538 | 7941 (2996) | 3456 (1304) | 2.30 | 5744 (3092) | 1.38 | 4335 (2334) | 1.83 |
| HEX8 | 676 / 963 | 4737 (3204) | 2497 (1689) | 1.90 | 3260 (3140) | 1.45 | 2420 (2331) | 1.96 |
| TET10 | 721 / 1027 | 4470 (3223) | 1893 (1365) | 2.36 | 3117 (3200) | 1.43 | 2305 (2366) | 1.94 |
| HEX27 | 1599 / 2274 | 2134 (3412) | 976 (1561) | 2.19 | 1489 (3386) | 1.43 | 1091 (2480) | 1.96 |

**f32 values** (ours with f64 vectors, cuSPARSE with f32 vectors)

| element | B/dof BSR / CSR | ours | bsrmv | × | SpMV CSR | × | SpMV CSR alg2 | × |
|---|---|---|---|---|---|---|---|---|
| TET4 | 199 / 360 | 12399 (2470) | 3843 (766) | 3.23 | 8227 (2963) | 1.51 | 5802 (2090) | 2.14 |
| HEX8 | 357 / 644 | 7292 (2600) | 2941 (1049) | 2.48 | 4844 (3117) | 1.51 | 3354 (2158) | 2.17 |
| TET10 | 380 / 686 | 6812 (2589) | 2061 (783) | 3.31 | 4559 (3127) | 1.49 | 3180 (2181) | 2.14 |
| HEX27 | 842 / 1517 | 3907 (3290) | 1065 (897) | 3.67 | 2261 (3431) | 1.73 | 1507 (2287) | 2.59 |

Against CSR ALG1:
- f64: CSR/BSR bytes 1.42–1.43, speed-up 1.38–1.45. Both kernels run at the same bandwidth; the
  format makes the difference.
- f32: CSR/BSR bytes 1.80–1.81, speed-up 1.49–1.73. Ours reads and writes y in f64 and gathers x in
  f64 (8 + 16 bytes per dof beside 199–842 of matrix), CSR f32 in f32; with the read of y our f32
  kernel streams less than CSR's except on HEX27, whose long rows hide it.

## Profile

TET4 n=150 and HEX27 n=75, y = A x (job 5198662; see Logs). Columns: ours, `bsrmv` (kernel
`bsrmv_tiny_core<T,2,3,2,5,0>`), and CSR ALG1 (`csrmv_v3_kernel`):

| | ours f64 | bsrmv f64 | CSR f64 | ours f32 | bsrmv f32 | CSR f32 |
|---|---|---|---|---|---|---|
| **TET4** | | | | | | |
| DRAM read (GB) | 4.0 | 4.0 | 5.7 | 2.2 | 2.1 | 3.8 |
| DRAM throughput, % of peak | **88** | 30 | 69 | 61 | 18 | 64 |
| warp instructions per block | **4.3** | 38.8 | 21.9 | **4.9** | 37.4 | 16.3 |
| issue slots busy, % | 28 | **75** | **77** | 40 | **79** | **79** |
| warps stalled on memory (long scoreboard), % | **71** | 35 | 32 | **76** | 33 | 14 |
| achieved / theoretical occupancy, % | 90 / 100 | 73 / 75 | 97 / 100 | 93 / 100 | 73 / 75 | 97 / 100 |
| registers per thread | 32 | 40 | 32 | 32 | 40 | 32 |
| threads per block, grid | 128, 430k | 32×4, 861k | 128, 719k | 128, 430k | 32×4, 861k | 256, 225k |
| **HEX27** | | | | | | |
| DRAM read (GB) | 16.8 | 16.6 | 23.7 | 8.8 | 8.8 | 15.8 |
| DRAM throughput, % of peak | **87** | 35 | 78 | **81** | 20 | 74 |
| warp instructions per block | **2.1** | 24.3 | 18.0 | **2.8** | 23.6 | 13.8 |
| issue slots busy, % | 13 | 56 | 75 | 33 | 58 | 80 |
| warps stalled on memory, % | **85** | 35 | 37 | **72** | 34 | 18 |
| achieved / theoretical occupancy, % | 90 / 100 | 74 / 75 | 95 / 100 | 90 / 100 | 74 / 75 | 93 / 100 |

What the counters show:

1. **`bsrmv` moves the same bytes as ours.** Its DRAM reads match ours to within 2% in f64; in f32
   they are up to 5% lower, because its vectors are f32. Its L2 read sectors match to within 2% on
   TET4. Its matrix is ours, and it reads it once. Traffic is not the reason it is slower.
2. **`bsrmv` is limited by instruction issue.**
   - It executes 8–12× more warp instructions per 3×3 block than ours (38.8 against 4.3 on TET4,
     24.3 against 2.1 on HEX27).
   - Its issue slots are 56–79% busy while DRAM runs at 18–35% of peak.
   - Its warps wait on memory only a third of the time.
   - Its launch is one warp per block row (32×4 threads, a grid of rows/4). Its occupancy (73–74%,
     against a theoretical 75% at 40 registers) is not the limit.
   - What the extra instructions do cannot be seen from the counters. The kernel is templated
     (`bsrmv_tiny_core<T,2,3,2,5,0>`), so its block size may be fixed at compile time too.
3. **Ours is limited by memory, as a sparse product should be.**
   - DRAM runs at 81–88% of peak under `ncu` (61% for TET4 f32, the shortest kernel), and
     3.0–3.5 TB/s in the bench.
   - 71–85% of warps wait on loads, while only 13–40% of issue slots are used.
   - Nothing but bytes is left to remove, which is what the compressed formats then do.
4. **The vendor's CSR kernel is efficient but works on a bigger format.**
   - `csrmv_v3_kernel` reaches 69–78% of DRAM peak in f64 and 3.1–3.5 TB/s in the bench, on a par
     with ours. It reads 1.4× (f64) and 1.8× (f32) more bytes: a 4-byte column index for each scalar
     instead of one per nine.
   - It also issues 3–9× our instructions per block, with issue slots 75–80% busy, so it is close
     to being limited by both.

## Why a specialised kernel can beat the vendor here

The kernel is 42 lines (`bsr_spmv_group_kernel`). Its design keeps the instruction count per
block low enough that only memory limits it, with the one shape that fits these matrices chosen by
measurement:

- **The block size is a template parameter.** For a 3×3 block, a thread does one column-index load,
  3 loads of x (contiguous, 24 bytes in f64), 9 value loads and 9 fused multiply-adds, fully
  unrolled, with the 3 row sums in registers. That is the 2–5 warp instructions per block above.
- **A thread owns whole blocks; a group of threads owns a block row.** G = 16 threads per block row
  fits finite-element graphs of 15–63 blocks per row: 1–4 blocks per thread, two rows per warp, and
  a 3-value shuffle reduction per row. Consecutive threads take consecutive blocks, so the warp's 9
  value loads cover one contiguous stretch of 32 blocks. The L1 absorbs the strides: 68–85% hits,
  and every byte is fetched from DRAM once.
- **G was measured, not guessed** (`bsrgroup_*_51935xx.out`, y = A x, MDOF/s f32 and the bandwidth
  it reaches):

  | element | G = 1 | G = 4 | G = 8 | G = 16 | G = 32 |
  |---|---|---|---|---|---|
  | TET4  | 6499 (1.40 TB/s) | 12461 | 15003 | 14619 (3.15 TB/s) | 9625 |
  | HEX8  | 2632 (0.98 TB/s) | 7750 | 7688 | 8344 (3.11 TB/s) | 7701 |
  | TET10 | 3412 (1.35 TB/s) | 7133 | 7567 | 7738 (3.07 TB/s) | 6997 |
  | HEX27 | 1364 (1.17 TB/s) | 3096 | 3386 | 4022 (3.45 TB/s) | 3725 |

  One thread per block row (G = 1) lands where `bsrmv` is, at 1.0–1.4 TB/s. A warp per row
  (G = 32, `bsrmv`'s launch shape) leaves lanes idle on TET4's 15-block rows. On these box meshes
  G = 16 is within 3% of the best G in f32 and within about 5% in f64. On the application meshes
  (TET4, 14–15 blocks per row) G = 8 is 1–4% faster than the better of two G = 16 measurements
  in the same run (`measured_real/realqs_5199067.out`), and those two differ by up to 5.5%: G = 8
  may suit TET4 better, which a dedicated repeat would have to settle.
- **Nothing outside the product.** There is no analysis or preprocessing step and no workspace, and
  each product is one kernel launch. Each block row is written once, by one thread.
- **Deterministic at full speed.** The G partial sums are added in a fixed shuffle order with no
  atomics, so the product is bit-wise reproducible. cuSPARSE's deterministic CSR algorithm (ALG2)
  costs 27–38% against its ALG1, which is why the gap to it is 1.9–2.8×.
- **Mixed precision for free.** f32 values with f64 vectors and accumulation is a template
  argument. It halves the matrix bytes and keeps the product's error at 3–5e-8 against `bsrmv` f32's
  9–10e-8 (TET10 and HEX27). cuSPARSE has no f32-matrix/f64-vector mode for either `bsrmv` or SpMV, so
  its f32 rows also round the vectors.

None of this relies on the box meshes: the application meshes below keep the order against
`bsrmv` and CSR. Our code uses the kernel for every block size from 1 to 4
(`d_bsr_spmv_ranges`, `d_bsr_spmv_native`).

## On application meshes

The same bench on unstructured TET4 meshes from two pump simulations, the parts their cases
compute: one valve (`fp70c`, `fp70`, two resolutions) and a pressure and a suction valve
(`ff200`, `ff200r`, two resolutions). Each mesh as given (0) and refined once (1, each tetrahedron
split into eight), coordinates scaled to a unit largest extent (`qp/real_meshes.py`, `QP_MESH`), at
13–15 blocks per row. Logs: `measured_real/realgpu_5199831/gpu_<mesh>_<level>.out` (Alps job
5199831, y += A x). MDOF/s of ours (TB/s of matrix), and our speed-up over each cuSPARSE product:

| mesh | dof | ours f64 | × bsrmv | × CSR | × CSR alg2 | ours f32 | × bsrmv | × CSR | × CSR alg2 | CSR f32 TB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| ff200 0 | 150,525 | 6622 (2.26) | 2.03 | 1.33 | 2.10 | 11118 (2.00) | 2.43 | 1.74 | 2.73 | 2.09 |
| fp70c 0 | 223,332 | 6647 (2.34) | 2.03 | 1.26 | 1.91 | 10512 (1.96) | 2.50 | 1.49 | 2.30 | 2.37 |
| fp70 0 | 526,809 | 7405 (2.67) | 2.15 | 1.25 | 1.78 | 10356 (1.97) | 2.71 | 1.28 | 1.84 | 2.79 |
| ff200r 0 | 973,812 | 7636 (2.77) | 2.19 | 1.27 | 1.88 | 10849 (2.08) | 2.81 | 1.26 | 1.87 | 2.97 |
| ff200 1 | 1,084,416 | 7751 (2.79) | 2.21 | 1.26 | 1.72 | 11455 (2.18) | 2.90 | 1.31 | 1.85 | 3.00 |
| fp70c 1 | 1,659,165 | 7822 (2.86) | 2.24 | 1.34 | 1.84 | 11573 (2.24) | 2.98 | 1.34 | 1.87 | 3.01 |
| fp70 1 | 4,000,896 | 7879 (2.92) | 2.24 | 1.39 | 1.86 | 11812 (2.31) | 3.07 | 1.44 | 2.04 | 2.90 |
| ff200r 1 | 7,423,662 | 7827 (2.91) | 2.23 | 1.38 | 1.82 | 11778 (2.31) | 3.02 | 1.42 | 2.04 | 2.94 |

- **f64:** ours streams 2.8–2.9 TB/s from 10⁶ dof, below its 3.0–3.4 on the box, and the
  speed-up over CSR (1.26–1.39) falls short of the byte ratio (1.43).
- **f32:** ours streams 2.2–2.3 TB/s from 10⁶ dof while CSR f32 keeps 2.9–3.0. The speed-up over
  CSR (1.31–1.44) falls short of the byte ratio (1.81). Under Nsight Compute on the largest mesh
  (`measured_real/ncu/ff200r_1.csv`, y = A x) our f32 kernel reads 11% more L2 sectors per dof than
  on the box and reaches 55% of DRAM peak against 62%. Our f32 product gathers f64 vectors and adds
  into an f64 y, CSR f32 uses f32 ones, so the gathers of x on an unstructured graph cost ours more;
  that is the likely cause, not a measured one.
- **Below 10⁶ dof** every product runs short of the bandwidth (launch and tail effects at 13–110
  µs per product); the order is unchanged.

## Limits of the comparison

- **One cuSPARSE version (12.5.2, CUDA 12.6) and one GPU.** `bsrmv` is the legacy interface. A
  later cuSPARSE that takes BSR in `cusparseSpMV` should be measured when available.
- **`bsrmv` indexes values with 32 bits.** It faults beyond 2³¹ values (HEX27 at 2.4×10⁷ dof); the
  bench skips it there. The generic API takes 64-bit indices, but its CSR product fails past 2³¹
  values too: with 64-bit indices and memory to spare, HEX27 at 1.8×10⁷ dof (3.4×10⁹ values)
  returns an internal error from both preprocess and product, while 64-bit indices below 2³¹ work.
  The bench skips the CSR rows there.
- **The f32 rows differ in vector precision.** cuSPARSE's read f32 vectors (fewer bytes) and are
  still slower.
- **The CSR expansion is the bench's, not our code's.** A CSR tangent would cost 1.4–1.8× the
  memory of BSR. Our code stores vector problems in BSR only.

## Reproduce

On one GH200 GPU (Alps jobs 5199822–5199828 for the rates, 5198662 for the counters), with the
sizes TET4 150, HEX8 150, TET10 75 and HEX27 75:

```
QP_ONLY="BSR f64|BSR f32|cusparse" bench_qp_cuda <n> 10
ncu --csv --page raw -k regex:"bsr_spmv_group_kernel|bsrmv_tiny_core|csrmv_v3_kernel" --metrics \
gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_srcunit_tex_op_read.sum,\
lts__t_sector_hit_rate.pct,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sector_hit_rate.pct,launch__grid_size,\
launch__block_size,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__maximum_warps_per_active_cycle_pct,smsp__inst_executed.sum,\
smsp__thread_inst_executed_per_inst_executed.ratio,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__issue_active.avg.pct_of_peak_sustained_active \
  bench_qp_cuda <n> 1
```

"Warp instructions per block" is `smsp__inst_executed.sum` divided by the number of blocks.
