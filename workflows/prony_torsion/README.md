# Prony-series viscoelastic torsion

A beam of `GeneratedMooneyRivlinProny` twisted about its axis, held while the torque relaxes,
and released, run by the elasticity playground (`drivers/mech/playground.exe.cpp`). The cases
and tools came from `spikes/prony-series`; [SPIKE.md](SPIKE.md) is that spike's report, kept
as it was written. Its findings stand, but its build and run instructions describe the retired
`prony_visco_torsion` driver.

## Running

```bash
export PRONY_EXE=<build>/playground       # default: <repo>/build/playground
./run_case.sh                             # list the cases
./run_case.sh torsion_release             # mesh, solve, XDMF, validation
./run_case.sh creep_torsion <workdir>
./run_torsion_release.sh                  # release and, with PRONY_RUN_CONTROL=1, the held control
tools/sweep_severity.py --sweep dt        # dt, angle, bulk, frequency or control
```

`run_case.sh` builds a 16 x 5 x 5 HEX8 beam (`PRONY_NX/NY/NZ`) once per workdir, runs
`playground <case>.yaml`, and validates the history: `tools/analyse_hysteresis.py` for a case
whose rotation has a `sine` profile, `tools/validate_torsion_release.py` otherwise, with the
arguments a case names on a `# validate_args:` line.

## From the spike's schema

| spike | playground |
|---|---|
| `mesh: mesh` | `mesh: {path: mesh, packed: false, sidesets: [{name: right, file: ...}]}` |
| `material: {C10, C01, K, prony, wlf, history_storage}` | `material: {type: GeneratedMooneyRivlinProny, C10, C01, K, relaxation: {iso: {prony, wlf: {C1, C2, T_ref}, temperature, history_storage}}}` |
| `dynamics.type: quasi_static \| bdf2 \| newmark` | `regime: quasi_static`, or `regime: dynamic` with `dynamics.scheme` |
| `torsion: {sideset, axis, rotation_center, angle}` | `constraints: [{type: rotation, sideset, axis, center, angle}]` |
| `torsion.ramp_time` | the rotation's `profile: {type: linear_ramp, end_time: ...}` |
| `torsion.profile: cyclic, period` | `profile: {type: sine, period}` |
| `torsion.release: {mode: free, time}` / `{mode: hold}` | `release: time` / no `release` |
| `torsion.control: torque, torque, torque_tol, torque_max_it` | `control: {torque, tol, max_it}`, the profile scaling the torque |
| `solver.newton.tol` (absolute) | `solver.newton: {atol, rtol: 0}` |
| `solver.newton.line_search: true` | the default: Armijo on the material's energy merit |
| `solver.linear: {rtol: 1e-10, preconditioner: true}` | `solver.linear: {preconditioner: jacobi}` at the default rtol 1e-3 |
| `output.control_point` | `output.probes: [{type: control_point, at}, {type: torque, constraint: 0}]` |
| `validate.prony_recovery: false` | `# validate_args: --no-prony-recovery` |

The history (`output.history_csv`) has the columns `step, t, dt, status, attempts, newton,
linear, capped, reference, final, min_alpha`, then `ux, uy, uz` (with `vx..az` under inertia
unless `rates: false`) and `angle, released, torque`. Its first row is the initial state.
Under torque control the `newton` and `linear` columns count every solve of the secant
iteration; the spike counted the last one only.
