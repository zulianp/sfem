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
`ldp-containment` worktree at commit d647b1e69 (MPRGP's CG steps in one OpenMP region), cases from `~/Desktop/cloud/owncloud_HSLU/Patrick/2025/sfem/*/case.yaml`
(on Alps `$SCRATCH/ldp_cases`).

| file | content | produced by |
|---|---|---|
| `summary.csv`, `runs.csv` | every case in every configuration, 3 repeats, the case files with `obstacle.qp.rtol: 1e-3` (inexact Newton) | `workflows/ldp_containment/submit.sh` (`bench.sbatch.sh`), collected by `python/collect.py`: jobs 5121338, 5121339, 5121365, 5121376, 5121474 in `$SCRATCH/ldp_bench_team`; the two-node jobs also ran $4\times72$ and 4 GPUs, half-filled nodes, whose rows are removed here as from `agreement.csv`, `nccl.csv` and `nccl_runs.csv` (the harness now refuses them). The runs of the three-pass operator (3b3668835) are kept in `$SCRATCH/ldp_bench_fused`, those before it in `$SCRATCH/ldp_bench_eta`, the exact-QP benchmark in `$SCRATCH/ldp_bench` and `ldp_bench_r20` |
| `agreement.csv` | last displacement of every configuration against `host_1x72`, by original node id | the first repeat of each run, compared on Alps |
| `ab.csv` | the `host_1x72` result against utopia's `contained_out.e` on the valve nodes, the distance field sampled by the solver's cubic B-spline (`sdf_compare.Grid`) | `workflows/ldp_containment/python/ab_utopia.py` on `$SCRATCH/ldp_bench_team/5121338/<case>/host_1x72/r1`, job 5275367 (before: trilinear sampling) |
| `kkt.csv` | KKT measure (Newton 0 of a solve taking no step) at utopia's state on its own and on SFEM's grid, and at SFEM's `host_1x72` state | `kkt.yaml` of each case, one rank: utopia's states job 5107749, SFEM's job 5121722 |
| `newton.csv` | Newton residuals and QP iterations of the `host_1x72` runs | their `run.log` |
| `utopia.csv` | utopia's timers (`summary.8.0.csv` next to each utopia case): 8 ranks, machine not recorded | copied |
| `coordinates.csv` | coordinate ranges of the valve meshes (their float32 spacing) | the runs' `mesh/` |
| `precision.csv`, `precision_runs.csv` | FF200 and FP70 cases in `host_16x18`, `gpu_1` and `gpu_4` with `solver.linear.storage_bytes` 8 and 4, alternating per repeat, 3 repeats each | `bench.sbatch.sh` once per repeat and precision (`LDP_REPEATS=1`, `LDP_SET`), collected by `python/collect.py` per precision and per repeat folder: jobs 5121523, 5121587, 5121600 in `$SCRATCH/ldp_f32ab_team`; `storage_bytes` column added, repeats numbered by their `rep<k>` folder |
| `nccl.csv`, `nccl_runs.csv` | the four cases on 4 GPUs (1 node) and on 8 GPUs (2 nodes), communication within a node by MPI (the default) and by NCCL (`nccl: true`), alternating per repeat, 3 repeats each; NCCL's set-up traced (`nccl_init`, `nccl_wait`) | `nccl_ab2.sbatch`: `bench.sbatch.sh` once per repeat and variant, collected by `python/collect.py` per variant and per repeat folder: jobs 5124953, 5124954 in `$SCRATCH/ldp_ncclab4` (`mpi`, `nccl`); `communication` column added, repeats numbered by their `rep<k>` folder. The text's claims about NCCL between nodes rest on jobs 5123902, 5123903 (messages: wrong values unless every message goes by rendezvous; latency against `osu_latency`) and 5124847 in `$SCRATCH/ldp_ncclab3` (the QP's reductions by NCCL over two nodes against MPI's, a build that did so) |
| `precision_agreement.csv` | last displacement of every such run against the same configuration's first double run | `outcmp.py` on `$SCRATCH/ldp_f32ab_team/cmp` |
| `omp_regions.csv` | an empty OpenMP region and a pass with a sum over one rank's share of FF200's unknowns, per run shape on Grace; the regions a QP iteration of FF200 on $1\times72$ opens | `workflows/ldp_containment/omp/omp_regions.sbatch` (`region_cost.c`, `region_count.c`): job 5121320 |
| `team_cost.csv` | a stand-in for MPRGP's CG step on FF200's unknowns per run shape: a region per pass with reduction clauses against one region with barriers and fixed-chunk sums, us per step | `workflows/ldp_containment/omp/team_cost.sbatch` (`team_cost.c`): job 5119257 |
| `release_runs.csv` | the four cases on `host_4x72` with the overlap released over 1, 2 and 4 quasi-static pseudo-steps (`regime=quasi_static`, `time.dt` 1, 0.5, 0.25, `time.max_cutbacks=4`, `obstacle.release.type=linear_ramp`), each converged (`solver.newton.fixed_iterations=false`, `max_it=40`), with the case's damped line search and with `line_search.rule=armijo`, `damping=1`; alternating per repeat, 3 repeats each | `release_ab.sbatch`: `bench.sbatch.sh` once per repeat and variant (`LDP_REPEATS=1`, `LDP_SET`), the build with the commit that adds this file (its `--set` creates nested keys correctly), collected by `python/collect.py` per variant and per repeat folder: jobs 5123692 (damped), 5123693 (Armijo) in `$SCRATCH/ldp_release`; `rule` and `pseudo_steps` columns added, repeats numbered by their `rep<k>` folder |
| `machine.txt` | node, CPU, GPU, driver, compilers, MPI | job `ldp-machine` on a debug node |
| `transient_runs.csv`, `transient_history.csv`, `transient_failures.csv` | FF200 through four pump strokes (`workflows/ldp_containment/transient`, simulation 2) on one GPU with 72 cores: the contact-stabilised Newmark damped by gamma 0.6, the first BDF2 run, and BDF2 with every change (Newton from u_n + dt v_n, float32 tangent, MPRGP's CG steps in device runs, the displacement written, the non-monotone line search); totals and traced times, every converged step, every failed attempt | `python/collect_transient.py` on Alps over `$SCRATCH/ldp_cases/FF200_ExodusMesh/transient_strokes4_{damped,bdf2,final}`: jobs 5131942, 5132710, each a snapshot of the build of its time (`sfem-ldp-snapshots`), simulation 1 from `transient_strokes4`; the last run (`final`) and the same with the monotone line search (`final_mono`, `solver.newton.line_search.nonmonotone=1`) with the build of branch valve-overheads (8622b96f9: the observers off the step's critical path, the files written by a thread of their own) in job 5290458 (`runs/strokes7.sbatch`, `$SCRATCH/ldp_strokes7/{final,final_mono}`, simulation 1 from `prony3_sim1` on the SFC-renumbered mesh), four runs at once on the node's four GPUs |
| `transient_ablation.csv` | the same case over its first 16 or 30 ms, one change at a time against the build before it | `python/collect_transient.py` over `$SCRATCH/ldp_fuse/{before,fused}` (job 5132727), `{fused2,fused_f32}` (5132754), `{steps_f32,runs_f32}` (5133192), `$SCRATCH/ldp_kind1/{base,start}` (5133342), and the line search over 30 ms with the build of branch valve-overheads (8622b96f9), `$SCRATCH/ldp_strokes7/{w30_mono,w30_nm}` (5290458) |
| `fp70_membrane_pressure.csv` | the chamber pressure of the strokes | copied from `workflows/ldp_containment/transient` |
| `prony_runs.csv`, `prony_series.csv` | the Prony history's storage formats, each reduced one with the slow terms as remainders (the default, horizon the simulated span) and without (`<format>_plain`, `remainder_horizon: 0`), and float32 stress slots (`<run>_s32`): FF200 simulation 2 (four strokes, BDF2) on one GPU with 72 cores, on TET4 and on its TET10 promotion (`db/solid_p2`); and the EPDM torsion bar (`workflows/prony_torsion/cases/epdm_torsion_release.yaml`) on Grace 1x72 on HEX8 40 and 80, HEX27 20 and 40, TET10 40 and TET4 40; per run the Prony bytes and the displacement against the first double run, and e_u(t) | `python/collect_prony.py runs` on Alps (`runs/collect8.sbatch`, job 5290591, into `$SCRATCH/sfem-prony-md/runs/paper_data8`); no run cuts back a step: the valves with the build of branch valve-overheads (build-cuda, 8622b96f9: the observers off the step's critical path; `runs/prony6_valves.sbatch`), TET4 job 5290460 in `$SCRATCH/ldp_cases/FF200_ExodusMesh/prony8_s0_<run>` (simulation 1: `prony3_sim1`), TET10 jobs 5290461, 5290463-64 (`solver.newton.tangent_projection: on_failure`) in `prony8_p2_<run>` (simulation 1: `p2_sim1`, job 5286158); torsion job 5286553 (build-psd2, b498e93ab) in `$SCRATCH/sfem-prony-md/runs/torsion6_<element>_<n>`, meshes from `torsion_<element>_<n>` (HEX27 from `verification_and_validation/common/geometry.py`'s tensor_product_mesh) |
| `prony_diag.csv` | the torsion bar on HEX8 80, HEX27 40, TET10 40 and TET4 40 with the build that counts the history updates a format rounds away (`SFEM_ENABLE_PRONY_DIAGNOSTICS`) | `python/collect_prony.py runs`: job 5286553 in `$SCRATCH/sfem-prony-md/runs/torsion6_diag_<element>_<n>` (build-diag, b498e93ab), HEX27 40 `int16_scaled_plain` again in job 5286718 (d1d3aa950: the torque probe had abandoned it on round-off) |
| `prony_tangent_FF200.csv`, `prony_tangent_FF200_refined.csv` | the float32 and float64 BSR tangent's and the mesh's bytes, from the valve meshes' node graph | `python/collect_prony.py tangent` over each run's `sim2/mesh` |
| `prony_bench.csv` | a step's two passes over the history per format, without the remainder form and with it (horizon 0.18 s), host (Grace 72 threads) and device (H100), at 242,336 and 1,768,274 one-point elements, 250,000 elements of 8 and of 11 points and 40,000 of 64 | `drivers/bench/bench_prony_history`: job 5286741 (`runs/prony6_bench.sbatch`, build-cuda, b498e93ab) |
| `prony_tolerance_series.csv` | the Newton tolerance's own effect: e_u(t) of the first double run at 1e-6 N against the same at 1e-9 N, torsion HEX8 80 and the TET4 valves (double tangent, 60 iterations; the valves' tight run cuts a step back at 24 ms) | `python/collect_prony.py runs` over `runs/tight.sbatch` (job 5286847, `<run>_tight`) in `runs/ctight.sbatch` (job 5286850, `runs/tight_data`) |

The refined run (row `refined`) is FF200_mesh_refined through the same four strokes on a full node:
simulation 1 on Grace, 4 ranks x 72 threads (480 s), with the build of branch prony-md (c2d08374d)
in job 5284753 (`runs/strokes6_refined.sbatch`, `transient_strokes6`); simulation 2 from it on the
four GPUs, 4 ranks, with the build of branch valve-overheads (8622b96f9) in job 5290459
(`runs/strokes7_refined.sbatch`), in `$SCRATCH/ldp_cases/FF200_mesh_refined/transient_strokes7`. `collect_transient.py` reads its valve
centres from the written displacement at the node nearest each point, which on the one-GPU run
agrees exactly with the probe.

`figures/stills/` holds frames 0, 38, 260 and 275 of the last four-stroke run (no step is cut back
before them, so frame n is at n x 0.1 ms), each view on its own:
`render_strokes.py sim2 stills --frames=0,38,260,275 --panels --size=1200,900` under the
`paraview/6.1:v2` uenv on a debug node (job 5284906, `runs/stills6.sbatch`), cropped with `sips -c`
(overview 230 x 960 at offset 320,120; top 710 x 1090 at 94,56; valve sections 430 x 1200 at 240,0).
`figures/stills_refined/` holds the refined run's same frames, from the same job, cropped the same way.
