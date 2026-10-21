# Valve containment report

```
make           build main.pdf
make figures   regenerate every figure, table and macro from data/
make selftest  the generator's own checks
make check     clean build, no undefined references, no LaTeX errors, no overfull boxes
```

No measured number is typed into `main.tex`: everything arrives through `\input{}` of a file
`python/ldp_figures.py` generates from `data/` (stdlib only, so it runs on Alps too).

## Data and the jobs behind it

All runs on Alps GH200 (account c40, debug partition), build `build-cuda` of the
`ldp-containment` worktree at commit 3b3668835 (the QP operator in three passes), cases from `~/Desktop/cloud/owncloud_HSLU/Patrick/2025/sfem/*/case.yaml`
(on Alps `$SCRATCH/ldp_cases`).

| file | content | produced by |
|---|---|---|
| `summary.csv`, `runs.csv` | every case in every configuration, 3 repeats, the case files with `obstacle.qp.rtol: 1e-3` (inexact Newton) | `workflows/ldp_containment/submit.sh` (`bench.sbatch.sh`), collected by `python/collect.py`: jobs 5116369, 5116370, 5116378, 5116386, 5116396 in `$SCRATCH/ldp_bench_fused`. The runs before the three-pass operator are kept in `$SCRATCH/ldp_bench_eta`, the exact-QP benchmark in `$SCRATCH/ldp_bench` and `ldp_bench_r20` |
| `agreement.csv` | last displacement of every configuration against `host_1x72`, by original node id | the first repeat of each run, compared on Alps |
| `ab.csv` | the `host_1x72` result against utopia's `contained_out.e` on the valve nodes | `workflows/ldp_containment/python/ab_utopia.py` |
| `kkt.csv` | KKT measure (Newton 0 of a solve taking no step) at utopia's state on its own and on SFEM's grid, and at SFEM's `host_1x72` state | `kkt.yaml` of each case, one rank: utopia's states job 5107749, SFEM's job 5116544 |
| `newton.csv` | Newton residuals and QP iterations of the `host_1x72` runs | their `run.log` |
| `utopia.csv` | utopia's timers (`summary.8.0.csv` next to each utopia case): 8 ranks, machine not recorded | copied |
| `coordinates.csv` | coordinate ranges of the valve meshes (their float32 spacing) | the runs' `mesh/` |
| `precision.csv`, `precision_runs.csv` | FF200 and FP70 cases in `host_16x18`, `gpu_1` and `gpu_4` with `solver.linear.storage_bytes` 8 and 4, alternating per repeat, 3 repeats each | `bench.sbatch.sh` once per repeat and precision (`LDP_REPEATS=1`, `LDP_SET`), collected by `python/collect.py` per precision and per repeat folder: jobs 5116419, 5116427, 5116447 in `$SCRATCH/ldp_f32ab_fused`; `storage_bytes` column added, repeats numbered by their `rep<k>` folder |
| `nccl.csv`, `nccl_runs.csv` | the four cases on 4 GPUs (1 node) and on 4 and 8 GPUs (2 nodes), communication within a node by MPI (the default) and by NCCL (`nccl: true`), alternating per repeat, 3 repeats each | `nccl_ab.sbatch`: `bench.sbatch.sh` once per repeat and variant, collected by `python/collect.py` per variant and per repeat folder: jobs 5116473, 5116485 in `$SCRATCH/ldp_ncclab_fused` (`new`, `nccl`); `communication` column added, repeats numbered by their `rep<k>` folder |
| `precision_agreement.csv` | last displacement of every such run against the same configuration's first double run | `outcmp.py` on `$SCRATCH/ldp_f32ab_fused/cmp` |
| `omp_regions.csv` | an empty OpenMP region and a pass with a sum over one rank's share of FF200's unknowns, per run shape on Grace; the regions a QP iteration of FF200 on $1\times72$ opens | `workflows/ldp_containment/omp/omp_regions.sbatch` (`region_cost.c`, `region_count.c`): job 5116563 |
| `machine.txt` | node, CPU, GPU, driver, compilers, MPI | job `ldp-machine` on a debug node |
