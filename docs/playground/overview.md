# The Playground

`playground` runs a solid mechanics simulation described by one YAML file: the mesh, the
material, loads and boundary conditions, the time regime, the solver and what to write. It is
SFEM's main entry point for simulations; the source is `drivers/mech/playground.exe.cpp` on top
of `src/frontend/mech/sfem_Playground.cpp` and `sfem_ElasticityCase.cpp`.

It covers:

- **Regimes**: a single static solve (`steady`), a sequence of loaded static states
  (`quasi_static`), and elastodynamics (`dynamic`, with Newmark, BDF2 or backward Euler).
- **Materials**: the generated operators (linear elasticity, NeoHookean-Ogden, Mooney-Rivlin with
  Prony-series viscoelasticity, linear and Mooney-Rivlin Kelvin-Voigt) and the hand-written
  NeoHookean-Ogden variants.
- **Loads and constraints**: body force, tractions and follower pressure, displacement
  conditions, prescribed or torque-controlled rotations, all scaled by time profiles.
- **Contact** with a rigid obstacle given as a signed distance field or an analytic shape.
- **Execution** on CPU threads, on a GPU, and on several MPI ranks.

## Building it

`playground` is built with SFEM when `SFEM_ENABLE_RYAML=ON` (the default). Obstacles need
`SFEM_ENABLE_SRESAMPLE=ON` (the default), `execution_space: device` needs a CUDA build
(`SFEM_ENABLE_CUDA=ON`), and `nccl: true` an MPI and CUDA build that found NCCL. See
[building](../building.md).

## A first case

A unit cube of NeoHookean material, clamped on `x = 0` and stretched by 40% along `x` in four
steps:

```yaml playground
# stretch.yaml
mesh:
  box: {n: [8, 8, 8], min: [0, 0, 0], max: [1, 1, 1]}
  sidesets:
    - {name: left, plane: {axis: x, at: min}}
    - {name: right, plane: {axis: x, at: max}}
material: {type: GeneratedNeoHookeanOgden, mu: 1, lmbda: 2}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
  - type: sideset
    format: name
    name: right
    component: 0
    value: 0.4
    profile: {type: linear_ramp, start_time: 0, end_time: 1, start_value: 0, end_value: 1}
regime: quasi_static
time: {dt: 0.25, t_end: 1}
output:
  path: out
  history_csv: out/history.csv
  probes: [{type: control_point, at: [1, 0.5, 0.5]}]
```

```sh
build/playground stretch.yaml
```

It prints a summary and one line per step:

```
Playground: quasi_static, GeneratedNeoHookeanOgden, 512 elements, 2187 dofs, 4 steps of 0.25
step 1 t=0.25 dt=0.25: converged, 4 Newton iterations, 78 linear, |r|/|r0| = 7.308e-12
...
step 4 t=1 dt=0.25: converged, 3 Newton iterations, 57 linear, |r|/|r0| = 8.452e-09
```

and writes `out/mesh/` (the mesh), `out/out/disp.<component>.<step>.float64` (the displacement
at every step) and `out/history.csv` (one row per step, with the control point's displacement).
[Outputs](outputs.md) describes them and how to look at them in ParaView.

## Command line

```
playground case.yaml [--set key.path=value]... [--dry-run]
```

- `case.yaml`: the case. Relative paths in it (meshes, sidesets, condition files, tabulated
  profiles, initial conditions, `output.path`, `output.history_csv`) are read against **the
  case file's directory**, not the working directory; so are paths given with `--set`.
- `--set key.path=value` (repeatable) overrides one key before the case is checked. The value
  is YAML, so scalars, lists and maps all work, and a numeric path segment indexes a list:
  ```sh
  playground stretch.yaml --set time.dt=0.1 --set mesh.box.n=[16,16,16]
  playground stretch.yaml --set dirichlet_conditions.1.value=0.2
  playground stretch.yaml --set output.path=/scratch/run1 --set execution_space=device
  ```
  A missing intermediate key is created; an index past the end of a list is an error.
- `--dry-run` checks the case against the schema, applies the overrides, prints the resolved
  case and exits without assembling anything.

The exit code is 0 when every step was solved and 1 for a case that does not check, a failed
assembly, or a step that could not be solved.

## How a case is checked

Every key is checked against the schema: a misspelt key is an error (`ElasticityCase: unknown
key "<path>"`), not a default silently taken. A few subtrees belong to the component that reads
them and are passed through unchecked: `material`, `loads.neumann`, `dirichlet_conditions`,
every `profile`, `obstacle.release` and `obstacle.ssmgc`. Their readers mostly ignore keys they
do not know, so check spellings there against the [case reference](case-reference.md).

Values are checked when the simulation is built: a wrong value, a missing required key or a
combination the playground does not support stops it with a message starting with `Playground:`.

## Documents

- [Case reference](case-reference.md): every key, with its type, default and restrictions.
- [Examples](examples.md): the cases in `workflows/` and what each shows.
- [Outputs](outputs.md): fields, history, probes, traces, and making XDMF for ParaView.
- [Parallel and GPU runs](parallel-and-gpu.md): MPI, `execution_space: device`, NCCL, and runs
  on Alps.
