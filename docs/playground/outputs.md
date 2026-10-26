# Playground outputs

## Files under `output.path`

With `output.path: P` the playground writes:

| Path | Content |
|---|---|
| `P/mesh/` | The mesh, in SFEM's raw format: `meta.yaml`, the connectivity `i<k>.<int type>`, the coordinates `x/y/z.<float type>`, and `node_mapping` (each node's id in the input mesh). A refined mesh is written as its standard elements. Under MPI, rank 0 writes the whole mesh read from `mesh.path`. |
| `P/out/<field>.<c>.<NNNNNNNNN>.<type>` | One raw file per field, component `c` and export `NNNNNNNNN` (9 digits, from 0 for the initial state), e.g. `out/disp.0.000000004.float64`. |
| `P/out/time.txt` | The time of each export, one per line. |
| `P/sdf/` | With an obstacle: its signed distance grid (`meta.yaml`, the samples and `sdf.xdmf`); contour it at 0 to see the obstacle. |
| `P/quantities.yaml` | With `output.quantities: true`: per step, the material's objective and the reaction resultants of every Dirichlet condition. |

The fields are `disp` always, and those listed in `output.fields`: `velocity`, `acceleration`,
`material_reaction`, `gap` and `contact_force`. Exports happen for the initial state, every
`export_freq`-th accepted step, and the last step.

A rerun with fewer steps leaves the previous run's later files in `P/out`: clear the folder
first, as the scripts in `workflows/` do.

## The history CSV

`output.history_csv` gets one row per accepted step, written by rank 0, plus a first row for the
initial state:

| Column | Meaning |
|---|---|
| `step`, `t`, `dt` | Step number, time at its end, length. |
| `status` | `initial`, `converged`, `converged_at_resolution`, `iterations_done` (with `newton.fixed_iterations`), ... |
| `attempts` | 1, plus the cutbacks the step needed. |
| `newton`, `linear` | Newton and linear (or QP) iterations. |
| `capped` | Linear or QP solves stopped by their iteration limit. |
| `reference`, `final` | The convergence measure's reference (its first value) and its final value; `final / reference` is the `|r|/|r0|` printed per step. |
| `min_alpha` | The smallest line-search step taken. |

then the probes' columns ([case reference](case-reference.md#output)): `ux, uy, uz` for a
control point, `angle, released, torque` for a rotation, `reaction_x, reaction_y, reaction_z,
touched` for contact, Prony history measures; each prefixed with the probe's `name` when it has
one. Failed attempts are reported on the console only.

## Console

```
Playground: <regime>, <material>, <n> elements, <n> dofs, <k> steps of <dt>[, <r> ranks]
  Newton 1: |r| = ...                     (with solver.newton.verbose)
step 1 t=0.25 dt=0.25: converged, 4 Newton iterations, 78 linear, |r|/|r0| = 7.308e-12
```

## Timings

Every run writes a trace of its main phases (`Playground::create`, `Stepper::run`,
`NodeObstacleStep::solve`, ...) as `name,calls,total,avg` to `smesh.trace.csv` in the working
directory, or to the file named by `SMESH_TRACE_FILE`.

## Looking at the results

The fields are raw arrays on the written mesh. To open them in ParaView, write an XDMF file
for the whole time series:

```sh
workflows/hyperelasticity_bdf2/write_transient_xdmf.sh $PWD/P/mesh $PWD/P
```

It writes `P/output.xdmf` and `P/output.h5` (a third argument names the XDMF file otherwise),
with every `disp`, `velocity`, `acceleration`, `gap` and `contact_force` export at the times of
`P/out/time.txt`. Give it absolute paths. It uses smesh's `raw_to_db` (a `build64/raw_to_db`,
one on `PATH`, or `external/smesh/python/smesh/raw_to_db.py`) and the Python of `$PYTHON`
(default `venv/bin/python`), which needs `numpy` and `h5py`.

`external/smesh/python/smesh/raw_to_exodusII.py` writes Exodus II instead (needs `netCDF4`).
