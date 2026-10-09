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
| `fuse_res.sbatch` | fused vs per-face convective faces in the residual, over both `sumfact` and the generated per-limiter kernels | `fuse_res_confirm.sbatch` and `fuse_res_final.sbatch`, which confirmed the one arm that won on five passes (both moved here on 2026-10-09) |

`fuse_res.sbatch` is the odd one out: it is here because it is **superseded**, not because it lost.
Its first-order Rhie–Chow arm is the standing performance gate and 5.4% sat only just above that
benchmark's 4.3% band, so the claim was re-measured properly by the two `fuse_res_*` jobs that
remained in `jobs/` until 2026-10-09. Its higher-order arms went through `--kernel sympy` to the generated
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

Of the jobs named in this section, `kernels.sbatch` and `profile_perf_alps.sbatch` moved here on
2026-10-09 with the retirement below; `bench_hex8_alps.sbatch` and `perf_hex8_alps.sbatch` stayed.

## Retired 2026-10-09: everything not feeding the paper or verification

`jobs/` now holds only the jobs that produce the paper's data (`wip/paper/data/`, one job per
file prefix), the verification and validation gates, and the two Alps tools `docs/README_alps.md`
documents; `jobs/README.md` lists them. The 170 jobs below moved here with `git mv`. None
is called by code: every reference from `src/`, `scripts/` or `python/` is a provenance comment.
Citations of `jobs/X.sbatch` in documents and comments now resolve to this folder; the comments
were left as written. Paths here are relative to the spike root.

Unlike the selection campaigns above, most of these still run against the current tree, but
many pin a build directory or a commit in their own header, so read it before resubmitting.

### Solver studies

The solver half of the work, which the paper no longer reports (`wip/paper`, commit 51ff15747 "drop the solver results, keep the kernels"). Each answered its question in its own output; the standing solver choices they led to are in `DESIGN.md` and `docs/MG_REDESIGN.md`.

| job | what it measured | where the answer is |
|---|---|---|
| `adjoint.sbatch` | Rung 2: is multigrid + block-Jacobi decomposition invariant now? | its own output |
| `cgc_split.sbatch` | Rung 2: is multigrid + block-Jacobi decomposition invariant now? | its own output |
| `cgc_sweep.sbatch` | Rung 2: is multigrid + block-Jacobi decomposition invariant now? | its own output |
| `coarse_op_rc_cases.sbatch` | Does the Rhie-Chow rescaling that fixes coarse-operator CONSISTENCY ever buy ITERATIONS? | its own output |
| `coarse_op_rc_decay.sbatch` | Is the coarse operator's velocity/pressure inconsistency the Rhie-Chow coefficient? | its own output |
| `coarse_size.sbatch` | How big should the coarsest level be, at constant fine resolution. | its own output |
| `coarseconv.sbatch` | Does the coarse BiCGStab converge, or is it stagnating? | its own output |
| `coarsefix.sbatch` | Does the coarse-gather fix hold on Cray-MPICH, and on the full solve? | its own output |
| `coarsemat.sbatch` | Is the matrix handed to the coarse factorisation complete? | its own output |
| `cycle_bisect.sbatch` | WHICH HALF of the V-cycle carries the decomposition dependence? | its own output |
| `cycle_vs_costmatched.sbatch` | Is the hierarchy worth anything multi-rank, or is it only an expensive smoother? | its own output |
| `deepfix.sbatch` | Three defects, on Cray-MPICH | its own output |
| `diagsum.sbatch` | Rung 2: is multigrid + block-Jacobi decomposition invariant now? | its own output |
| `diagsum8.sbatch` | Is the coarse block diagonal complete on owned nodes at eight ranks? | its own output |
| `egal_alloc.sbatch` | Why don't the grid transfers scale? | its own output |
| `egal_clamp_scope.sbatch` | Should the clamp apply only to the coarse space? | its own output |
| `egal_converge.sbatch` | Full Newton solve to convergence, probe against element-wise Galerkin. | its own output |
| `egal_converge_em.sbatch` | Full Newton solve to convergence, probe against element-wise Galerkin. | its own output |
| `egal_coord.sbatch` | Rung 2: is multigrid + block-Jacobi decomposition invariant now? | its own output |
| `egal_perf.sbatch` | Where does a V-cycle actually spend its time? | its own output |
| `egal_reynolds.sbatch` | Solver robustness against Reynolds number, element-wise Galerkin throughout. | its own output |
| `egal_reynolds_one.sbatch` | Solver robustness against Reynolds number, element-wise Galerkin throughout. | its own output |
| `egal_reynolds_scope.sbatch` | How hard does the linear solve get as Reynolds rises? | its own output |
| `egal_threads.sbatch` | Is the thread clamp throttling the fine level? | its own output |
| `egal_threads_rep.sbatch` | Is the thread clamp throttling the fine level? | its own output |
| `egal_vanka_re.sbatch` | Vanka against Reynolds number, element-wise Galerkin coarse operators. | its own output |
| `egal_xfer_scale.sbatch` | Do the grid transfers scale with threads? | its own output |
| `exact_cost_P30.sbatch` | What does the operator/smoother inconsistency cost, in iterations? | its own output |
| `fixcheck.sbatch` | Does the coarse-packing fix hold under MPI, and is the one slower arm noise? | its own output |
| `galsplit.sbatch` | Is the multi-pack defect in the Galerkin fold, or in the transfers / coarse operator? | its own output |
| `gatesplit.sbatch` | One defect, two symptoms: the rank-local fold | its own output |
| `getrf_bench_P7.sbatch` | dgetrf alone on a 2,320 x 2,320 matrix -- the nozzle's coarsest level -- with the uenv's OpenBLAS, to separate … | its own output |
| `gmres_context.sbatch` | What does a GMRES iteration cost, and how much of it is the operator? | its own output |
| `gmres_invariance.sbatch` | Should GMRES not take the same number of iterations at every rank count? | its own output |
| `gradsum_probe.sbatch` | Should GMRES not take the same number of iterations at every rank count? | its own output |
| `ho_newton_ab.sbatch` | What the exact higher-order Jacobian buys the Newton loop. | its own output |
| `krylov_ladder.sbatch` | THE LADDER: debug the Krylov method before the smoother, and the smoother before the coarse space. | its own output |
| `nocoarse.sbatch` | Without a coarse space, is the iteration count the same at every rank count -- for Vanka as well as for Jacobi? | its own output |
| `nozzle_affine.sbatch` | The FDA nozzle with its curved elements approximated as affine: how much does it cost in accuracy, and does it … | its own output |
| `nozzle_affine_perf.sbatch` | Two things jobs/nozzle_affine.sbatch leaves out. | its own output |
| `nozzle_blasbind_P6.sbatch` | Why getrf on the 2,320-dof coarsest level takes ~580 ms, the cost of one core: the driver's OpenMP binds its … | its own output |
| `nozzle_coarseasm_P8.sbatch` | The coarsest dense matrix assembled with hessian_bsr instead of probed (build_p5) against the probe (build_p4 … | its own output |
| `nozzle_curved_perf.sbatch` | The semi-structured kernels after the curved-macro-element branch, against the committed kernels, on affine … | its own output |
| `nozzle_defaults_P11.sbatch` | The warped FDA nozzle at throat Re 500 on the driver's DEFAULTS alone (build_p7): linear tolerance, coarse … | its own output |
| `nozzle_fgmres_P2.sbatch` | A/B of FGMRES on SFEM's OpenMP BLAS (build_p1) against the serial FGMRES (build_w), both at … | its own output |
| `nozzle_galerkin_P8.sbatch` | Coarse operators: rediscretised (SFEM_GMG_GALERKIN=0, the default -- the coarsest level's dense matrix is … | its own output |
| `nozzle_lapack_P4.sbatch` | A/B of the coarse dense LU through LAPACK getrf (build_p4) against the hand-written serial loop (build_p1). | its own output |
| `nozzle_lrtol_R1.sbatch` | The warped FDA nozzle at throat Re 500 (macro core 2), FGMRES (SFEM_FGMRES=1, restart 480) + geometric … | its own output |
| `nozzle_lrtol_R2.sbatch` | Companion to R1: the same linear-tolerance A/B without multigrid (FGMRES + Vanka alone) at L4, and with … | its own output |
| `nozzle_outflow_ab.sbatch` | The do-nothing outflow against the convective one, on the FDA nozzle. | its own output |
| `nozzle_patterncache_P16.sbatch` | A/B of the Galerkin pattern cache: each level's sparsity pattern, scatter and block->slot map kept across … | its own output |
| `nozzle_perf_P1.sbatch` | Profile of the current production set-up: warped FDA nozzle, macro core 2, throat Re 500, first-order upwind … | its own output |
| `nozzle_perf_P13.sbatch` | Profile of the production set-up at HEAD f201fd76d (driver defaults: tolerance 1e-3, Galerkin coarse operators … | its own output |
| `nozzle_rcdecay_P10.sbatch` | Coarse operators, the fair comparison: rediscretised with the Rhie-Chow decay docs/MG_REDESIGN.md calls a … | its own output |
| `nozzle_setuptrace_P14.sbatch` | Where the Vanka setup's time goes (build_p9: HEAD f201fd76d plus trace scopes on the fine matrix assembly's steps). | its own output |
| `nozzle_smooth_P5.sbatch` | Vanka sweeps per level (pre and post), 1 / 2 / 3, on the current production set-up: the fine-level smoother is … | its own output |
| `nozzle_smooth_gal_P9.sbatch` | Vanka sweeps per level (pre and post), 1 / 2 / 3, with element-wise Galerkin coarse operators … | its own output |
| `nozzle_vanka_P3.sbatch` | A/B of the multiplicative Vanka sweep on a per-macro lattice stencil (build_p2) against the global-BSR row walk … | its own output |
| `nozzle_vankaprec_P21.sbatch` | The Vanka storage precision as a setting (build_p14: SFEM_VANKA_PRECISION, single by default, double on … | its own output |
| `nozzle_vankasetup_P15.sbatch` | A/B of the fine-level Vanka setup: the chunked assembly's inverse and accumulation made chunk-local and … | its own output |
| `nozzle_vankasp_P12.sbatch` | A/B of the Vanka smoother with single-precision storage (build_p8: per-cell factors and the row walk's matrix … | its own output |
| `nozzle_vankastencil_P18.sbatch` | A/B of the multiplicative Vanka sweep reading a per-macro 27-point stencil in single precision and lattice … | its own output |
| `nozzle_zerostart_P17.sbatch` | A/B of zero-start pre-smoothing: Multigrid tells its smoothers when their solution is zero (coarse levels … | its own output |
| `outflow_ab.sbatch` | The do-nothing outflow against the convective one, on a transient. | its own output |
| `peclet_blend.sbatch` | Does switching the upwind term off for momentum work here. | its own output |
| `phasea.sbatch` | Phase a on a consistent toolchain. | its own output |
| `precond_freeze.sbatch` | What reusing the preconditioner's setup buys, and what it costs in iterations. | its own output |
| `pressure_attribution.sbatch` | Which variable actually caused the port failures: the linear solver, or the size? | its own output |
| `pressure_continuation.sbatch` | The prescribed-pressure continuation, on the path the laptop cannot check. | its own output |
| `pressure_needs_continuation.sbatch` | Does the production solver need the pressure continuation at all? | its own output |
| `pressure_solver_matrix.sbatch` | The prescribed-pressure cases across the solver stacks that actually exist. | its own output |
| `pump_cycle_large.sbatch` | The diaphragm pump, one full cycle, at the size and Reynolds number the debug partition cannot reach. | its own output |
| `pump_cycle_n32.sbatch` | One full diaphragm cycle at N=32 and Re=200, written frame by frame for ParaView. | its own output |
| `pump_dt_probe.sbatch` | Does a smaller timestep pay for itself? | its own output |
| `pump_highre_small.sbatch` | High Reynolds number on a small mesh with a fine timestep, to find out whether the case runs there at all … | its own output |
| `pump_sweep.sbatch` | The diaphragm pump at higher resolution and higher Reynolds number. | its own output |
| `pump_visualize.sbatch` | One full diaphragm cycle at a resolution worth looking at, written frame by frame for ParaView. | its own output |
| `rc_tau_remedy.sbatch` | Two candidate remedies for the stages that abandon under the corrected time scale. | comment in `src/drivers/cvfem_hex8_ns_ssgmg.cpp` |
| `rc_tau_sweep.sbatch` | The Rhie-Chow time scale, swept over the quantity it was wrong about. | its own output |
| `rung2_postfix.sbatch` | Rung 2: is multigrid + block-Jacobi decomposition invariant now? | its own output |
| `slotcheck.sbatch` | Does multi-pack renumbering break the 27-slot lattice key? | its own output |
| `smoother_step0.sbatch` | Step 0 of the coarse-space work: the smoother, alone, with no coarse space at all. | its own output |
| `solver_ab.sbatch` | Two solver binaries against each other on one Grace socket, at sizes that saturate it. | its own output |
| `step_budget.sbatch` | The numerical dissipation where it is supposed to be large. | its own output |
| `step_turb_probe.sbatch` | Does the turbulent step case run, and what makes its time steps hard? | its own output |
| `tolerance_ab.sbatch` | Is the "decomposition dependence" a tolerance artifact? | its own output |
| `vanka_default_P28.sbatch` | The stencils are now the only fine-level Vanka path, so this checks the DEFAULT rather than an option: the … | its own output |
| `vanka_nomg_P29.sbatch` | The stencil path without multigrid. | its own output |
| `vanka_pcouple_P31.sbatch` | How sensitive is the solver to the patch pressure couplings? | its own output |
| `vankacomplete.sbatch` | The cross-rank completion of the Vanka patch entries, on Cray-MPICH | its own output |
| `vankafix.sbatch` | The Vanka smoother under decomposition, after the coarse path was fixed | its own output |

### MPI, rank-count and scale probes

Decomposition-invariance bisects, rank-by-thread shape sweeps and large-size runs of the solver. Pinned to the builds and defects of their day.

| job | what it measured | where the answer is |
|---|---|---|
| `decomposition_dependence.sbatch` | WHERE does the multi-rank gap enter, and is the algorithm decomposition-dependent? | its own output |
| `gmg_consistency_P22.sbatch` | Consistency of the operators with the coarse space, on the current build (build_p14: LAPACK coarse LU, Galerkin … | its own output |
| `inv288.sbatch` | Does invariance still hold at a full node, with the coarse-packing fix? | its own output |
| `mp288.sbatch` | Invariance at a full node on a multi-pack case -- sized so it actually fits. | its own output |
| `multipackmpi.sbatch` | The one configuration where this session's two halves meet. | its own output |
| `poi_20M.sbatch` | Poiseuille at ~20M dofs, with the linear solver's convergence history. | its own output |
| `poi_sweep.sbatch` | Where does the Poiseuille case stop scaling, and what does the linear solver do there? | its own output |
| `poiseuille_200M.sbatch` | Poiseuille at ~200M dofs, to capture the linear solver's convergence history. | its own output |
| `rank_bisect.sbatch` | At what rank count does the distributed setup corrupt the heap? | its own output |
| `rank_bisect_trim.sbatch` | At what rank count does the distributed setup corrupt the heap? | its own output |
| `ranksweep.sbatch` | Where between one rank and eight does the coarse correction break? | its own output |
| `scale288.sbatch` | Does decomposition invariance hold at full node width? | its own output |
| `scale_600M.sbatch` | How far does the packed semi-structured operator go, and what does a matvec cost there? | its own output |
| `shape288.sbatch` | Rank x thread shape at a fixed 288 cores -- where is the sweet spot? | its own output |
| `shapefat.sbatch` | Below the baseline: do fewer, fatter ranks win further? | its own output |
| `shapeiter.sbatch` | Does the shape optimum move once the coarse factorisation is gone? | its own output |
| `shapelegacy.sbatch` | The communication-overlap cost at each shape -- the half i have never measured. | its own output |
| `shapesat.sbatch` | The shape sweep at a size that actually fills the node. | its own output |
| `state_by_rank.sbatch` | Does the linearisation state itself differ between decompositions? | its own output |
| `state_falsify.sbatch` | Is the Vanka-only decomposition dependence caused by an UNGATHERED LINEARISATION STATE? | its own output |
| `verif_20M_grace.sbatch` | The three Farrell/Mitchell/Wechsung verification cases at ~20M dofs, one Grace socket. | superseded by `jobs/verif_one.sbatch` (one case per allocation) |
| `verifymatrix.sbatch` | Is serial bit-identical after the four MPI fixes? | its own output |
| `verifysubset.sbatch` | An early signal on the fix, hours before the full matrix can run. | its own output |

### Build and migration one-offs

Each built or tested one commit or one install; none is a gate to revive (no CI/CD gate for the spike). The spike's ctest on Alps is `jobs/ctest.sbatch`.

| job | what it measured | where the answer is |
|---|---|---|
| `gate_build_P23.sbatch` | Builds for the commit gate, from clean exports of committed trees (no uncommitted edits from any session): HEAD … | its own output |
| `gate_ctest_P24.sbatch` | The spike's unit tests on the clean HEAD 935297062 build (gate_build_P23). | its own output |
| `port_probe.sbatch` | Why does the pressure-port case fail on Grace and converge on the laptop? | its own output |
| `sfem_ctest_P19.sbatch` | SFEM's own test suite on a clean tree at HEAD plus the two algebra edits of the zero-start pre-smoothing change … | its own output |
| `sfem_ctest_parallel_P20.sbatch` | SFEM's MPI-parallel tests on the clean tree with the zero-start algebra edits ($SCRATCH/sfem-zsg-src/build-test). | its own output |
| `sfem_rebuild_P27.sbatch` | Rebuild the SFEM OpenMP install after the Alps scratch migration. | its own output |

### Superseded measurements

A job in `jobs/` now measures the same thing better; the successor is named per row.

| job | what it measured | where the answer is |
|---|---|---|
| `bench_determinism.sbatch` | Is the bench kernel bit-reproducible with itself? | superseded by `jobs/det_layout.sbatch` (every layout) |
| `camp_sat.sbatch` | The layout campaign at the saturating sizes only. | superseded by `jobs/camp_full.sbatch` |
| `camp_smoke.sbatch` | A short smoke run of the layout campaign. | superseded by `jobs/camp_full.sbatch` |
| `pack_size_sweep.sbatch` | Is the flat baseline actually tuned? | superseded by `jobs/packsize_paper.sbatch` |
| `packbisect.sbatch` | Where does the multi-pack defect live? | multi-pack defect, fixed |
| `packprobe.sbatch` | Which assembled object is wrong when there is more than one pack? | multi-pack defect, fixed |
| `packsev.sbatch` | What does this defect do at the default pack size, on a real problem size? | multi-pack defect, fixed |
| `perf_rc.sbatch` | Where the Rhie-Chow residual's time goes. | superseded by `jobs/dram_traffic.sbatch` and `jobs/kernel_mix.sbatch` |
| `perf_stat.sbatch` | Exact totals, to turn the sampled per-symbol percentages into absolutes. | `docs/CVFEM_Throughput.md` |
| `perf_sweep.sbatch` | Why the same packed sweep runs at 1072 MDOF/s in the solver and 1482 in the benchmark. | `docs/CVFEM_Throughput.md` |
| `ss_level_sweep.sbatch` | How much of the semi-structured operator's gap to the flat one is the macro-element level? | superseded by `jobs/ab_semistructured.sbatch` |
| `ss_packed_sweep.sbatch` | What packing the macro-element mesh is worth, against the scatter it replaces. | superseded by `jobs/ab_semistructured.sbatch` |
| `ss_packsize_level.sbatch` | Does pack size matter to a semi-structured run, and does the answer depend on the level? | superseded by `jobs/ab_semistructured.sbatch` |
| `throughput.sbatch` | Where the time goes, on one Grace socket, across problem sizes and on both operators. | superseded by `jobs/camp_full.sbatch` |
| `throughput16.sbatch` | First honest throughput numbers, and why the earlier ones were not | superseded by `jobs/camp_full.sbatch` |
| `throughput_ab.sbatch` | The throughput sweep with both binaries in one allocation, on one node. | superseded by `jobs/ab_refactor.sbatch` |
| `vanka_invariance_ab.sbatch` | Is the arena conversion invariant on Grace? | `perf/vanka_arena_invariance_grace.txt`; standing check: `jobs/determinism.sbatch` |
| `vanka_selfcmp.sbatch` | Is the solver bitwise reproducible against itself? | `perf/vanka_arena_invariance_grace.txt`; standing check: `jobs/determinism.sbatch` |

### One-change A/Bs and kernel decisions

Each decided one change, and the decision is in the code and, where named, a perf note. Standing A/Bs are `jobs/ab_refactor.sbatch` and `jobs/ab_semistructured.sbatch`.

| job | what it measured | where the answer is |
|---|---|---|
| `ab_first_order.sbatch` | The first-order packed residual, before against after, in one allocation. | its own output |
| `ab_step5.sbatch` | Clearing the step-5 move: the generated Jacobian-action arrangements and the affine generated residual … | its own output |
| `ab_tet4.sbatch` | The TET4 A/B, which the HEX8 and semi-structured gates do not cover. | TET4 kernels, one codegen change |
| `allfour.sbatch` | Four decomposition defects, on Cray-MPICH | its own output |
| `applysum.sbatch` | Is the coarse operator's action decomposition invariant? | its own output |
| `bench_ab_many.sbatch` | cvfem_sshex8_bench for several builds in one allocation, interleaved rep by rep so drift between them cannot … | its own output |
| `bjcontrol.sbatch` | Correcting an invalid control of my own making. | its own output |
| `bnd_skip.sbatch` | The boundary closure, which swept the whole mesh to touch its skin. | `docs/CVFEM_Throughput.md` |
| `conv_cost.sbatch` | What the convective scheme COSTS, per arm. | its own output |
| `conv_frozen.sbatch` | The order of accuracy of each convective arm, which is the half of the limiter question … | its own output |
| `conv_ladder.sbatch` | The order of accuracy of each convective arm, which is the half of the limiter question … | its own output |
| `conv_perf.sbatch` | WHERE the deferred correction's 3.16x goes, as opposed to how big it is. | its own output |
| `fuse_res_confirm.sbatch` | Confirming the one arm that won: the first-order residual with Rhie-Chow. | the fused face loop that won, in the kernel (see `fuse_res` above) |
| `fuse_res_final.sbatch` | Confirming the one arm that won: the first-order residual with Rhie-Chow. | the fused face loop that won, in the kernel (see `fuse_res` above) |
| `fused_rc.sbatch` | What the packed Jacobian action costs once it carries the term the solver actually runs. | `docs/CVFEM_Throughput.md` |
| `grad_precision_ab.sbatch` | THE GRADIENT FIELDS' STORAGE PRECISION, double against single, interleaved in one allocation. | `perf/grad_precision_grace.txt` |
| `ho_limiter_vec.sbatch` | What the unvectorised Venkatakrishnan kernel costs. | its own output |
| `live_vectors.sbatch` | Does the Krylov working set account for the gap between the bench and the solver? | `docs/CVFEM_Throughput.md` |
| `profile_perf_alps.sbatch` | Linux perf profiling for the CVFEM TET4 Navier-Stokes upwind benchmark on CSCS ALPS. | TET4 profiling; HEX8 profiling stays as `jobs/perf_hex8_alps.sbatch` |
| `rc_hoist.sbatch` | The hoisted Rhie-Chow coefficient, against the version that computed it in the face loop. | `docs/CVFEM_Throughput.md` |
| `rcjac_packed_ab.sbatch` | Packed Rhie-Chow Jacobian action, reference against the Rhie-Chow velocity-derivative change, interleaved in … | its own output |
| `sfc_driver.sbatch` | What space-filling the driver's mesh is worth. | `docs/CVFEM_Throughput.md` |
| `stencil.sbatch` | Are the assembled Vanka stencils decomposition invariant? | its own output |

### Jobs behind spike documents that are not the paper

The documents keep their numbers; rerunning them means running these from here.

| job | what it measured | where the answer is |
|---|---|---|
| `bench_solve_kernels.sbatch` | The determinism study, repeated on the benchmark, and the dof rate of every kernel the solve calls. | its own output |
| `dns_cost.sbatch` | Stage 1 of the transient-DNS plan: measure what a time step costs, so the production mesh is sized from data … | `docs/CVFEM_DNS_Cost.md` |
| `dt_campaign.sbatch` | Stage 6's one outstanding measurement. | `docs/CVFEM_DNS_Cost.md` |
| `kernels.sbatch` | The measurement behind docs/CVFEM_Kernels.md. | `docs/CVFEM_Kernels.md` |
| `paper_confirm.sbatch` | The two rows that came out negative, at eleven repetitions, to separate a change from noise. | `perf/paper_kernel_table_grace.csv` |
| `paper_ho_table.sbatch` | THE HIGHER-ORDER CONVECTIVE FLUX, at the size and layout the paper's cost table uses. | `perf/paper_ho_table_grace.csv`, `docs/CVFEM_NSE.tex` |
| `paper_kernel_table.sbatch` | The paper's kernel-cost table, remeasured. | `perf/paper_kernel_table_grace.csv`, `docs/CVFEM_NSE.tex` |
| `vanka_freeze.sbatch` | Does freezing the Vanka smoother across Newton iterations pay? | `docs/CVFEM_DNS_Cost.md` |
| `vanka_freeze_tail.sbatch` | The one row P34 (job 4685821) did not reach before its two-hour limit: L=8 with the Peclet blend on and … | `docs/CVFEM_DNS_Cost.md` |
