# Torsion history-storage comparison

Run from the SFEM repository root, after building `spikes/prony-series/build/prony_visco_torsion`:

```bash
OMP_NUM_THREADS=1 venv/bin/python scripts/run_torsion_history_compare.py \
  --out build_torsion_runs/precision_compare_02
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

All runs use the same executable, a snapshot of `spikes/prony-series/cases/torsion_release.yaml`,
and one shared HEX8 mesh. Geometry is fixed at 1 x 0.2 x 0.2, matching that YAML. Node counts
default to 16 x 5 x 5; override `PRONY_NX`, `PRONY_NY`, `PRONY_NZ` for a mesh study.
`SFEM_T` and the old cantilever material environment variables do not configure this driver:
the physics comes from the YAML. The script does not build or modify the solver.

`SFEM_HISTORY_CHECK=1` is enabled for every run. These are accuracy/diagnostic runs, not timing
benchmarks. A nonzero process exit, non-finite history, missing time steps, incorrect release
schedule or recorded Newton residual above tolerance stops the batch. Logs and partial data
remain for diagnosis. A successful process exit alone is not treated as proof of convergence.

## Outputs and definitions

- `case.yaml`, `mesh/`: shared inputs.
- `manifest.json`: Git commit, executable and YAML hashes, requested storage policies,
  thread count, exit codes and elapsed times (not a performance claim).
- `<policy>/run.log`: complete solver log.
- `<policy>/results_release/history.csv`: scalar response at every step.
- `<policy>/results_release/out/`: raw displacement fields at export times.
- `compare/torsion_comparison.pdf` / `.png`: torque, transverse control-point displacement,
  and their precision errors.
- `compare/torsion_errors.csv`: absolute and peak-normalized scalar errors over time.
- `compare/torsion_summary.csv`: peak scalar errors, iteration counts and maximum residual.
- `compare/coupled_comparison.pdf` / `.png` / `.csv`: whole-field displacement errors,
  reusing the existing comparison functions. Per-policy details are in `compare/<policy>/`.

Torque error is `abs(T_candidate(t) - T_FP64(t))`. Its normalized value is this error divided
by `max_t abs(T_FP64(t))`, multiplied by 100. Do not divide by instantaneous reference torque:
after release it is almost zero.

Control-point vector error is `norm(u_candidate(t) - u_FP64(t))`, using all three components,
normalized by `max_t norm(u_FP64(t))`. The response plot shows `sqrt(uy^2 + uz^2)` at the
YAML control point's nearest mesh node. It is not the cross-section mean. The CSV `angle`
is not used as a recovery measurement: the driver writes zero after release.

Whole-field relative error is `100 * norm(u_candidate(t) - u_FP64(t)) / norm(u_FP64(t))`,
the unweighted Euclidean norm over all nodal displacement components. Initial zero fields
are checked explicitly. Error plots omit exact zeros on logarithmic axes; CSV files retain them.
These are differences from a same-mesh FP64 numerical reference, not errors against an exact solution.

The runner does not invoke the unavailable XDMF export helper. CSV and raw-field comparisons
work without it. It does not submit Slurm jobs or change the old cantilever scripts.

To recreate plots without rerunning the solver:

```bash
venv/bin/python scripts/run_torsion_history_compare.py \
  --out build_torsion_runs/precision_compare_02 --plot-only
```

Small automated check:

```bash
venv/bin/python scripts/test_torsion_history_compare.py
```
