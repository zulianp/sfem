# Newmark torsion history-storage comparison

Run from the SFEM repository root, after building `spikes/prony-series/build/prony_visco_torsion`:

```bash
OMP_NUM_THREADS=1 venv/bin/python scripts/run_torsion_history_compare.py \
  --out build_torsion_runs/newmark_prony4_full_01
```

Use an unused output directory. The runner refuses to overwrite simulations, including partial
runs. The old `build_torsion_runs/fp64` is not modified or reused. `--exe` (or `PRONY_EXE`) can
select a different torsion executable; use the Python environment with NumPy, pandas,
matplotlib and PyYAML installed (`.venv/bin/python` on CSCS if that is your environment).

The runner executes these five policies sequentially, with `SFEM_HISTORY_MODE=per_qp`:

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
`SFEM_T` and the old cantilever material environment variables do not configure this driver:
the physics comes from the YAML. The script does not build or modify the solver.

The case includes inertia: density = 1, Newmark beta = 0.64 and gamma = 0.6. These parameters
are kept as supplied, including their numerical damping. Material C10 = 0.3, C01 = 0.05,
K = 50; Prony weights = [0.4, 0.4, 0.1, 0.05], times = [1, 2, 5, 10], g_inf = 0.05.
The twist ramps to 6.4 rad over 4 s, is released at 5 s, and the run ends at 80 s with dt = 0.005
(16,000 steps per policy). Only the history storage policy differs between runs; mass and
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
work without it. It does not submit Slurm jobs or change the old cantilever scripts.

To recreate plots without rerunning the solver:

```bash
venv/bin/python scripts/run_torsion_history_compare.py \
  --out build_torsion_runs/newmark_prony4_full_01 --plot-only
```

Small automated check:

```bash
venv/bin/python scripts/test_torsion_history_compare.py
```
