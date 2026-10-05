# `subpar/jobs/` — the selection campaigns, and one job they superseded

DESIGN.md's second correction leaves **one micro-kernel per kernel**, the one Grace measured
fastest, and removes the selector that chose between them. The `--kernel` flag went with it, on
both the HEX8 and the TET4 benchmark.

These four jobs are what made that choice, or what the choice stranded. They are kept for the
reason `../README.md` gives for the headers next to them — the removals were made on measured
grounds, so the measurement should be repeatable rather than re-derived — but unlike those
headers they are **not runnable against the current driver**: every one of them selects a variant
through a flag that no longer exists. Restoring one means restoring the flag with it.

| job | what it selected | where the answer went |
|---|---|---|
| `tet4_arms.sbatch` | thirteen TET4 arms behind that benchmark's own `--kernel` enum, none of them on record for Grace | the one survivor is the only TET4 micro-kernel; the rest are `../cvfem_tet4_retired.hpp` |
| `scalar_arms.sbatch` | the three isoparametric scalar pairs the affine campaign had said nothing about, plus whether `split` earned its path | `../cvfem_hex8_atomic_retired.hpp` |
| `cse_action.sbatch` | three CSE arrangements of the generated HEX8 Jacobian action against the hand-written scalar one | `../cvfem_hex8_ns_upwind_sympy_subpar.hpp` |
| `fuse_res.sbatch` | fused vs per-face convective faces in the residual, over both `sumfact` and the generated per-limiter kernels | `jobs/fuse_res_confirm.sbatch` and `jobs/fuse_res_final.sbatch`, which confirmed the one arm that won on five passes |

`fuse_res.sbatch` is the odd one out: it is here because it is **superseded**, not because it lost.
Its first-order Rhie–Chow arm is the standing performance gate and 5.4% sat only just above that
benchmark's 4.3% band, so the claim was re-measured properly by the two `fuse_res_*` jobs that
remain in `jobs/`. Its higher-order arms went through `--kernel sympy` to the generated
per-limiter kernels, which is the half that cannot run.

## What stayed in `jobs/`

Every job that passed `--kernel sumfact` — all fifteen, including the twelve that feed the
paper's generated macros — simply lost the flag: `sumfact` was the surviving kernel, so asking
for it by name was already a no-op. Three jobs that carried the kernel as a *parameter* lost the
parameter instead: `bench_hex8_alps.sbatch` dropped its `kernel` sweep and the positional its
`measure` helper took, `perf_hex8_alps.sbatch` dropped the `kernel` column from its perf CSV
(it could only ever hold one value, and the driver prints the name it ran in the raw log), and
`profile_perf_alps.sbatch` dropped a `KERNEL` default that named a variant now in `..`.

`jobs/kernels.sbatch` kept its layout and geometry rows — `docs/CVFEM_Kernels.md` is still
generated from them — and lost the three sweeps that crossed layout with kernel.
