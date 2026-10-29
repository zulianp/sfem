# The library's BSR product against cuSPARSE on the H100

The comparison: `sfem::d_bsr_spmv_grouped` (G = 16, `src/algebra/bsr/cuda/sfem_BSR_cuda.cu`), the
BSR product the paper uses as its GPU baseline, against every cuSPARSE path that takes the same
matrix.

**In short:**
- cuSPARSE's BSR product, `bsrmv`, runs at 0.8–1.8 TB/s. It executes 8–12× more warp
  instructions per block than ours, so it is limited by instruction issue, not by memory. Ours
  reads the same bytes from DRAM and runs at 3.0–3.5 TB/s: **1.9–2.5× faster in f64, 2.8–3.8×
  in f32.**
- cuSPARSE's generic `cusparseSpMV` does not take BSR (cuSPARSE 12.5.2). Its tuned path is scalar
  CSR, and that kernel is as efficient as ours: 3.1–3.5 TB/s. It loses only because CSR stores a
  column index per scalar. With 1.4–1.8× more bytes per product, it is **1.4–1.5× slower in f64 and
  1.6–1.8× slower in f32.** Against cuSPARSE's deterministic CSR algorithm the gap is 1.9–2.8×.

So we do not reach more bandwidth than the vendor can. The vendor's BSR kernel spends too many
instructions on 3×3 blocks to reach it, and its kernel that does reach it needs a format with more
bytes.

## Set-up

- **Machine:** one H100 of an Alps GH200 node, CUDA 12.6 (cuSPARSE 12.5.2).
- **Matrices:** the neo-Hookean tangent at the bench state, 10,328,853 dof (3,442,951 block rows
  of 3×3), assembled on the device by the library. The same values go into every product.

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
- `measured_gpu/cusparse_5198662.out`: rates.
- `measured_gpu/ncu_{TET4,PROTEUS_HEX27}_5198662.csv`: counters, untracked.
- The paper's campaign (`bench_*_51950xx.out`) agrees within 1%.

## Rates

MDOF/s, with GB/s in parentheses, and our speed-up over each product:

**f64**

| element | B/dof BSR / CSR | ours | bsrmv | × | SpMV CSR | × | SpMV CSR alg2 | × |
|---|---|---|---|---|---|---|---|---|
| TET4  | 377 / 538   | 9061 (3419) | 3606 (1361) | 2.51 | 6068 (3266) | 1.49 | 4408 (2373) | 2.06 |
| HEX8  | 676 / 963   | 5177 (3501) | 2678 (1811) | 1.93 | 3460 (3333) | 1.50 | 2514 (2422) | 2.06 |
| TET10 | 721 / 1027  | 4842 (3491) | 1959 (1412) | 2.47 | 3309 (3397) | 1.46 | 2362 (2425) | 2.05 |
| HEX27 | 1599 / 2274 | 2136 (3414) |  989 (1581) | 2.16 | 1524 (3466) | 1.40 | 1116 (2539) | 1.91 |

**f32 values** (ours with f64 vectors, cuSPARSE with f32 vectors)

| element | B/dof BSR / CSR | ours | bsrmv | × | SpMV CSR | × | SpMV CSR alg2 | × |
|---|---|---|---|---|---|---|---|---|
| TET4  | 199 / 360   | 14958 (2980) | 4081 (813)  | 3.67 | 8497 (3061) | 1.76 | 5870 (2114) | 2.55 |
| HEX8  | 357 / 644   |  8685 (3097) | 3077 (1097) | 2.82 | 5052 (3251) | 1.72 | 3296 (2121) | 2.64 |
| TET10 | 380 / 686   |  7898 (3002) | 2126 (808)  | 3.72 | 4816 (3303) | 1.64 | 3060 (2099) | 2.58 |
| HEX27 | 842 / 1517  |  4073 (3429) | 1081 (910)  | 3.77 | 2326 (3528) | 1.75 | 1437 (2181) | 2.83 |

Against CSR ALG1, the speed-up follows the byte ratio:
- f64: CSR/BSR bytes 1.42–1.43, speed-up 1.40–1.50.
- f32: CSR/BSR bytes 1.80–1.81, speed-up 1.64–1.76.

Both kernels run at the same bandwidth; the format makes the difference.

## Profile

TET4 n=150 and HEX27 n=75. Columns: ours, `bsrmv` (kernel `bsrmv_tiny_core<T,2,3,2,5,0>`), and
CSR ALG1 (`csrmv_v3_kernel`):

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

The kernel is about 25 lines (`bsr_spmv_group_kernel`). Its design keeps the instruction count per
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
- **G was measured, not guessed** (`bsrgroup_*_51935xx.out`, MDOF/s f32 and the bandwidth it
  reaches):

  | element | G = 1 | G = 4 | G = 8 | G = 16 | G = 32 |
  |---|---|---|---|---|---|
  | TET4  | 6499 (1.40 TB/s) | 12461 | 15003 | 14619 (3.15 TB/s) | 9625 |
  | HEX8  | 2632 (0.98 TB/s) | 7750 | 7688 | 8344 (3.11 TB/s) | 7701 |
  | TET10 | 3412 (1.35 TB/s) | 7133 | 7567 | 7738 (3.07 TB/s) | 6997 |
  | HEX27 | 1364 (1.17 TB/s) | 3096 | 3386 | 4022 (3.45 TB/s) | 3725 |

  One thread per block row (G = 1) lands where `bsrmv` is, at 1.0–1.4 TB/s. A warp per row
  (G = 32, `bsrmv`'s shape) leaves lanes idle on TET4's 15-block rows. G = 16 is within 3% of the
  best G everywhere in f32 and within about 5% in f64.
- **Nothing outside the product.** There is no analysis or preprocessing step and no workspace, and
  each product is one kernel launch. Each block row is written once, by one thread.
- **Deterministic at full speed.** The G partial sums are added in a fixed shuffle order with no
  atomics, so the product is bit-wise reproducible. cuSPARSE's deterministic CSR algorithm (ALG2)
  costs 27–38% against its ALG1, which is why the gap to it is 1.9–2.8×.
- **Mixed precision for free.** f32 values with f64 vectors and accumulation is a template
  argument. It halves the matrix bytes and keeps the product's error at 3–5e-8 against `bsrmv` f32's
  9–10e-8 (TET10 and HEX27). cuSPARSE has no f32-matrix/f64-vector mode for either `bsrmv` or SpMV, so
  its f32 rows also round the vectors.

None of this is specific to SFEM's matrices beyond small square blocks and short rows, which every
vector-valued finite-element tangent has. The library uses the kernel for every block size from 1
to 4 (`d_bsr_spmv_ranges`, `d_bsr_spmv_native`).

## Limits of the comparison

- **One cuSPARSE version (12.5.2, CUDA 12.6) and one GPU.** `bsrmv` is the legacy interface. A
  later cuSPARSE that takes BSR in `cusparseSpMV` should be measured when available.
- **`bsrmv` indexes values with 32 bits.** It faults beyond 2³¹ values (HEX27 at 2.4×10⁷ dof); the
  bench skips it there. The generic API takes 64-bit indices.
- **The f32 rows differ in vector precision.** cuSPARSE's read f32 vectors (fewer bytes) and are
  still slower.
- **The CSR expansion is the bench's, not the library's.** A CSR tangent would cost 1.4–1.8× the
  memory of BSR. The library stores vector problems in BSR only.

## Reproduce

On one GH200 GPU (Alps job 5198662), with the sizes TET4 150, HEX8 150, TET10 75 and HEX27 75:

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
