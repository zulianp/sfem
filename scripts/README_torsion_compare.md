# Newmark torsion history-storage comparison

## CSCS: SBATCH → experiment script → YAML → local plots

The workflow is:

```text
scripts/run_torsion.sbatch             resources, environment, one MPI rank
  → scripts/run_torsion_study.sh       resolution(s), mode(s), cases, optional short T
    → run_torsion_history_compare.py   mesh, independent case runs, validation, CSV reduction
      → newmark_torsion_release.yaml  material, loading, dynamics, solver, T=30, dt=0.005
```

No C++ changes or rebuild are needed for these script changes if the current mixed-precision
torsion executable is already built on CSCS. Do not transfer the macOS executable to CSCS.

### 1. Environment and debug pilot

Run on CSCS from the repository root:

```bash
cd /capstor/store/cscs/pasc/c40/hyang/sfem
# Only start a uenv if you are not already inside the matching build environment.
uenv start prgenv-gnu/25.6:v2 --view=default
source .venv/bin/activate
python3 -c 'import numpy, pandas, matplotlib, yaml; print("Python dependencies OK")'
ls -lh spikes/prony-series/build/prony_visco_torsion
ldd spikes/prony-series/build/prony_visco_torsion
```

All runtime libraries must resolve (no `not found`). The SBATCH repeats Python and library
checks on the compute node. It activates `$ROOT_DIR/.venv`; set `TORSION_VENV` if yours is
elsewhere. `PRONY_EXE` overrides the executable path. No username-specific executable path
is hard-coded in the scripts. The SBATCH does not install packages, compile or start a nested uenv.

Short coarse-grid pilot, both modes and all five cases:

```bash
TORSION_RESOLUTIONS=coarse TORSION_MODES="per_qp per_elem" TORSION_END_TIME=0.025 \
  sbatch --uenv-passthrough=use --partition=debug --time=00:30:00 \
  --cpus-per-task=8 scripts/run_torsion.sbatch
```

The `debug` partition changes the wall-clock limit, NOT the simulated end time.
Only `TORSION_END_TIME` shortens the saved YAML copy. Use `TORSION_END_TIME=5.1` for a
release-crossing pilot; 0.025 only tests startup. A time limit can still interrupt either pilot.

The default SBATCH requests one exclusive node, one MPI rank, 72 OpenMP CPUs and 6 hours in
`normal`, account `c40`. Adjust the resource directives or override them on the `sbatch`
command line. 72 threads is a starting allocation, not a measured optimal thread count.
The driver is single-rank; never launch 72 MPI tasks. Local tests do not verify CSCS scheduling.

### 2. Full experiment: six jobs, 30 solves

`run_torsion_study.sh` is the place to edit experiment defaults. Its `TORSION_*` settings can
also be overridden when submitting. This submits six independent jobs; do it after the pilot:

```bash
for resolution in coarse medium fine; do
  for mode in per_qp per_elem; do
    TORSION_RESOLUTIONS="$resolution" TORSION_MODES="$mode" TORSION_END_TIME=30 \
      TORSION_CASES="fp64 fp32 fp16 fp16_tensor fp16_element_prony" \
      TORSION_POSTPROCESS=1 \
      sbatch --uenv-passthrough=use scripts/run_torsion.sbatch
  done
done
```

Each job sequentially runs five cases, then reduces their raw displacement fields to error
CSVs against that mode's FP64 reference. It does not draw plots. To run all groups in one
allocation instead, use `TORSION_RESOLUTIONS="coarse medium fine" TORSION_MODES="per_qp per_elem"`;
they will run sequentially and share the job's wall-clock limit, so separate jobs are preferable.

Output: `build_torsion_runs/study_JOBID/coarse_per_qp/` (and corresponding other groups).
Each group contains a YAML snapshot, mesh, manifest, per-case logs/fields and `compare/`.
The top-level base can be changed with `TORSION_OUT_BASE`; existing directories are refused.
Nothing automatically resumes or deletes previous results. A timeout preserves completed
cases and partial files but does not count as a successful experiment.

```bash
squeue -j JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
tail -n 80 torsion-JOBID.out
```

The Slurm log is created in the submission directory; simulation output is under the repo root.

### 3. Split cases and compute comparisons later

For debug, select cases, e.g. `TORSION_CASES="fp64 fp32"` in one job and
`TORSION_CASES="fp16 fp16_tensor fp16_element_prony"` in another. Each receives a distinct job
directory. Without FP64 plus at least one candidate, automatic reduction is skipped.
Set `TORSION_POSTPROCESS=0` to skip reduction explicitly. Partial case selection does not
resume a truncated time trajectory: a resubmitted case starts from time zero.

Once both jobs finish, combine their complete, non-overlapping results on CSCS, using a
compute allocation for large-field reduction:

```bash
python3 scripts/run_torsion_history_compare.py --compare-only \
  --runs build_torsion_runs/study_JOB1/coarse_per_qp \
         build_torsion_runs/study_JOB2/coarse_per_qp \
  --out build_torsion_runs/coarse_per_qp_merged
```

Inputs must have identical YAML settings, mesh parameters and executable hash. Duplicate
mode/case pairs, incomplete jobs and failed cases are rejected rather than silently selected.
Completed comparison tables are also preserved: use `--plot-only` to redraw, or a new `--out`
with `--runs` to recompute them. An interrupted reduction without `comparison.json` may be retried.
Do not include a failed job alongside its replacement. For a comparison of both history
modes, pass a complete `coarse_per_qp` directory and a complete `coarse_per_elem` directory
to the same `--runs` command, with a new `--out`. The reference then becomes `per_qp_fp64`:
this measures combined spatial-representation and storage-precision differences. Per-mode
comparisons remain referenced to each mode's own FP64. Never combine different resolutions.

### 4. Transfer compact results and plot locally

The `compare/` directory contains all scalar histories and whole-field error time series needed
for the summary plots. Raw mesh, displacement, velocity and acceleration fields stay on CSCS.
Run this on your LOCAL machine; replace JOBID and the SSH host alias as appropriate:

```bash
mkdir -p build_torsion_runs/from_cscs/coarse_per_qp/compare
rsync -av \
  daint:/capstor/store/cscs/pasc/c40/hyang/sfem/build_torsion_runs/study_JOBID/coarse_per_qp/compare/ \
  build_torsion_runs/from_cscs/coarse_per_qp/compare/
venv/bin/python scripts/run_torsion_history_compare.py --plot-only \
  --out build_torsion_runs/from_cscs/coarse_per_qp
```

Use `.venv/bin/python` instead if that is your local environment. No solver or CSCS paths are
needed for cached-CSV plotting. For new spatial visualizations not represented in the cached
tables, retain/download the relevant raw fields. Do not delete raw results until analysis is complete.

## Direct Python entry point

Run from the SFEM repository root, after building `spikes/prony-series/build/prony_visco_torsion`:

```bash
OMP_NUM_THREADS=1 venv/bin/python scripts/run_torsion_history_compare.py \
  --out build_torsion_runs/newmark_prony4_full_01
```

Use an unused output directory. The runner refuses to overwrite simulations, including partial
runs. The old `build_torsion_runs/fp64` is not modified or reused. `--exe` (or `PRONY_EXE`) can
select a different torsion executable; use the Python environment with NumPy, pandas,
matplotlib and PyYAML installed (`.venv/bin/python` on CSCS if that is your environment).

The runner executes these five policies sequentially. `--history-mode per_qp` is the default;
use `--history-mode per_elem` for one history tensor per element per Prony branch. `--cases`
selects a subset by directory names below; `--run-only` skips comparison (also permits no FP64).
Without `--run-only`, a direct Python run computes tables and plots after simulation.

| Directory | History storage | Scaling |
| --- | --- | --- |
| fp64 | float64 | none |
| fp32 | float32 | none |
| fp16 | float16 | none |
| fp16_tensor | float16 | tensor |
| fp16_element_prony | float16 | element_prony |

All runs use the same executable, a snapshot of `spikes/prony-series/cases/newmark_torsion_release.yaml`,
and one shared HEX8 mesh. Geometry is fixed at 1 x 0.2 x 0.2, matching that YAML. Node counts
default to 16 x 5 x 5; override `PRONY_NX`, `PRONY_NY`, `PRONY_NZ` for a mesh study.
Alternatively, `--resolution` selects the study meshes below and takes precedence over those
environment variables. Without it, the existing small/custom mesh behavior is unchanged.

| Resolution | HEX8 elements in x, y, z | Nodes in x, y, z | Total elements |
| --- | --- | --- | --- |
| coarse | 40 x 8 x 8 | 41 x 9 x 9 | 2,560 |
| medium | 80 x 16 x 16 | 81 x 17 x 17 | 20,480 |
| fine | 160 x 32 x 32 | 161 x 33 x 33 | 163,840 |

For example, run the five per-QP policies on the coarse mesh for the default 30 s:

```bash
venv/bin/python scripts/run_torsion_history_compare.py \
  --resolution coarse --out build_torsion_runs/coarse_per_qp_T30_01
```

Use `medium` or `fine` with a separate output directory for the other meshes, and
`--history-mode per_elem` to run the other history mode. Under `per_elem`, the tensor and
element-Prony scale groups coincide; both policies are retained in the requested five-case matrix.

`SFEM_T` and the old cantilever material environment variables do not configure this driver:
the physics comes from the YAML. The script does not build or modify the solver.

The case includes inertia: density = 1, Newmark beta = 0.64 and gamma = 0.6. These parameters
are kept as supplied, including their numerical damping. Material C10 = 0.3, C01 = 0.05,
K = 50; Prony weights = [0.4, 0.4, 0.1, 0.05], times = [1, 2, 5, 10], g_inf = 0.05.
The twist ramps to 6.4 rad over 4 s, is released at 5 s, and the run ends at 30 s with dt = 0.005
(6,000 steps per policy). Only the history storage policy differs between runs; mass and
the time-integration parameters are identical.

For a startup smoke test, shorten only the saved YAML's end time, without editing the source:

```bash
OMP_NUM_THREADS=1 venv/bin/python scripts/run_torsion_history_compare.py \
  --out build_torsion_runs/newmark_smoke_02 --end-time 0.025
```

Use `--end-time 5.1` to also exercise release at 5 s. A startup-only smoke test cannot establish
release stability or long-time accuracy. The override must be an integer multiple of dt and
cannot extend beyond the source YAML's end time. Omit it for the full experiment.

Detailed history diagnostics default to `SFEM_HISTORY_CHECK=1`; set `SFEM_HISTORY_CHECK=0`
before invoking the runner to disable them. The actual setting is saved in `manifest.json`.
The scale floor and zero-history handling remain active regardless of this switch.
Use diagnostics for validation; disable them for timing runs (the comparison workflow itself
is not a dedicated timing benchmark). A nonzero process exit, non-finite history, missing time steps, incorrect release
schedule or recorded Newton residual above tolerance stops the batch. Logs and partial data
remain for diagnosis. A successful process exit alone is not treated as proof of convergence.

## Outputs and definitions

- `case.yaml`, `mesh/`: shared inputs.
- `manifest.json`: Git commit, executable and YAML hashes, requested storage policies,
  thread count, exit codes and elapsed times (not a performance claim).
- `<policy>/run.log`: complete solver log.
- `<policy>/results_newmark/history.csv`: scalar response at every step.
- `<policy>/results_newmark/out/`: raw displacement, velocity and acceleration fields at export
  times. The current field comparison measures displacement only.
- `compare/torsion_comparison.pdf` / `.png`: torque, signed control-point y displacement,
  and their precision errors.
- `compare/torsion_errors.csv`: absolute and peak-normalized scalar errors over time.
- `compare/torsion_summary.csv`: peak scalar errors, iteration counts and maximum residual.
- `compare/torsion_responses.csv`: scalar responses for every selected case, including FP64.
- `compare/comparison.json`: saved case settings, reference label and source-run paths for provenance.
- `compare/coupled_comparison.pdf` / `.png` / `.csv`: whole-field displacement errors,
  reusing the existing comparison functions. Per-policy details are in `compare/<policy>/`.

Torque error is `abs(T_candidate(t) - T_FP64(t))`. Its normalized value is this error divided
by `max_t abs(T_FP64(t))`, multiplied by 100. Do not divide by instantaneous reference torque:
it may cross zero during oscillations and is almost zero on the traction-free face after release.
The driver's reaction torque includes the material and inertia contributions. Do not apply a
quasi-static monotone-relaxation or Prony-weight-fit test to this dynamic response.

Control-point vector error is `norm(u_candidate(t) - u_FP64(t))`, using all three components,
normalized by `max_t norm(u_FP64(t))`. The response plot shows signed `uy` at the
YAML control point's nearest mesh node, so reversals remain visible. It is not the cross-section mean. The CSV `angle`
is not used as a recovery measurement: the driver writes zero after release.

Whole-field relative error is `100 * norm(u_candidate(t) - u_FP64(t)) / norm(u_FP64(t))`,
the unweighted Euclidean norm over all nodal displacement components. Initial zero fields
are checked explicitly. Error plots omit exact zeros on logarithmic axes; CSV files retain them.
These are differences from a same-mesh FP64 numerical reference, not errors against an exact solution.
An instantaneous field-relative error can grow when the reference displacement norm becomes
small during recovery; read it together with absolute and peak-normalized errors.

The runner does not invoke the unavailable XDMF export helper. CSV and raw-field comparisons
work without it. The new SBATCH is only a wrapper; the old cantilever submission scripts are unchanged.

To recreate plots from compact tables without rerunning the solver:

```bash
venv/bin/python scripts/run_torsion_history_compare.py \
  --out build_torsion_runs/newmark_prony4_full_01 --plot-only
```

Small automated check:

```bash
venv/bin/python scripts/test_torsion_history_compare.py
```
