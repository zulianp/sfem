# Redesigning SFEM workflows through the API

These are proposals to shorten SFEM's end-to-end workflows by moving work the scripts do around a run into the library. Counts are `rg` results in this worktree at `717c5cfbf`, excluding `build*/`, `venv/` and `.git`.

## How SFEM is used today

- **Playground.** `drivers/mech/playground.exe.cpp` (50 lines) is the main entry point. It calls `ElasticityCase::read(case.yaml, --set overrides)` and then `Playground::create(comm, case)->run()` (`frontend/mech/sfem_Playground.cpp`, 1,302 lines). The case format is documented in `frontend/mech/sfem_ElasticityCase.hpp` and checked against the key list in `sfem_ElasticityCase.cpp:17-133`. The repository holds 29 playground cases: 9 in prony_torsion, 5 in hyperelasticity, 4 in the LDP transient folders, and others in workflows/ and tests/.
- **workflows/.** 26 folders with 61 `.sh` and 15 `.py` files. A typical script builds a mesh with Python or C tools, exports `SFEM_*` variables, runs a driver, then converts the output (`raw_to_db`, `raw_to_exodusII`, `raw_point_data_to_parent`).
- **drivers/.** 60 `*.exe.cpp` and 9 plain `.cpp` mains. They are configured mostly through the environment: drivers read 189 distinct `SFEM_*` names.
- **python/.** `python/sfem` holds mesh generators, converters, SDF tools, `obs.py` and regression scripts. `python/codegen` is the kernel generator. `python/bindings/pysfem.cpp` (nanobind, 991 lines) exposes Mesh, FunctionSpace, Function, Op, conditions, buffers and solvers. `external/smesh/python/smesh` holds a second set of converters.
- **Alps.** 213 `sbatch` files, most under `spikes/`. 95 files name the uenv `prgenv-gnu/24.11:v2`.

**One run end to end (LDP transient, `workflows/ldp_containment/transient`).**

1. Outside the repository, `prepare.sh` builds `db/`. The obstacle grid `db/sdf_stops` comes from `mesh_to_sdf` with `SSDF_HMAX=3e-5 SSDF_MARGIN=1.6e-3`; the parameters are recorded only in a comment at the top of `sim1.yaml`.
2. `run_transient.sh` sources `<case>/../env.sh` to get `PLAYGROUND`, `PYTHON` and `SMESH_PY`. It copies `sim1.yaml`, `sim2.yaml` and the pressure table into `<case>/transient/` so that `../db/solid` resolves, and probes `$PYTHON` for netCDF4.
3. It runs sim1, then calls `raw_point_data_to_parent.py` twice (once for sim2's start, once for viewing) and `raw_to_exodusII.py` once.
4. It runs sim2, which reads `sim1/whole/point_data/disp_{x,y,z}.float64`, then calls `raw_point_data_to_parent.py` and `raw_to_exodusII.py` again.
5. For benchmarks, `bench.sbatch.sh` writes `env.txt` from a heredoc, sets `LD_PRELOAD` and `MPICH_GPU_SUPPORT_ENABLED` per configuration, and `workflows/ldp_containment/python/collect.py` recovers sizes and iteration counts from `run.log` with regexes.

That is two solver runs, five Python invocations and three copied files, configured through variables from `env.sh` (`PLAYGROUND`, `PYTHON`, `SMESH_PY`, `SFEM_BUILD`, `LAUNCH`) and the batch script.

---

## P1. Format conversion around every run

**Today.** `run_transient.sh:31-46` makes five Python calls per case: `raw_point_data_to_parent.py` three times and `raw_to_exodusII.py` twice. The `.e` files need netCDF4 in `$PYTHON`; without it the script silently produces no `.e` files. `workflows/hyperelasticity/hyperelasticity.sh:52` converts with `raw_to_db ... --transient --n_time_steps=25`, the step count typed by hand. `workflows/prony_torsion/run_case.sh` documents a failure mode: a shorter rerun leaves higher-index field files behind, and `raw_to_db` later fails with "Invalid sequence length". The parent-mesh mapping exists because a run on `mesh.blocks` with packing writes fields in its own numbering. The playground already reads initial conditions in the file numbering through `node_mapping` (`sfem_Playground.cpp:569-572`); only its output lacks that numbering.

**Proposal.** The playground writes a time-series XDMF beside the raw fields. It also gains an output numbering option that writes fields on the parent mesh in its file numbering, with zero outside the selected blocks. That is what `raw_point_data_to_parent.py` computes. smesh already writes a static mesh XDMF (`Mesh::write_with_xdmf`, `external/smesh/src/io/smesh_write_xdmf.cpp`). The temporal collection points at the raw binaries, so the run needs neither netCDF nor HDF5. Exodus becomes an explicit conversion on demand.

```yaml
output:
  path: sim1
  numbering: parent        # mesh (default) | parent: fields on mesh.path, file numbering
  xdmf: true               # sim1/fields.xdmf: this run's exported steps, with their times
```

**Removes:** five Python calls per LDP case, the netCDF4 probe, the hand-typed `--n_time_steps`, and the stale-file failure (the XDMF lists exactly this run's steps). With `numbering: parent`, sim2 reads sim1's output directly.

**Effort and risk:** M. A temporal XDMF writer in smesh I/O plus the parent scatter in the playground output, about 300 lines. The risk is low because the raw files are unchanged.

## P2. Runs chained through files

**Today.** `FF200_mesh_refined/sim2.yaml` repeats 40 of the 47 non-comment lines of `sim1.yaml`: the mesh, the material with its 15 Prony terms, and the obstacle. `FF200_ExodusMesh/` holds the same pair again. sim2 starts from sim1's displacement only, after the Python mapping and a float64 round trip. Time and Prony history are not carried; "starts relaxed" is the right choice for this case, but the user cannot choose otherwise. `run_transient.sh` needs `--sim1-only` and `--sim2-only` to rerun one half.

**Proposal.**
- (a) **Stages in one case.** Each stage is deep-merged over the base keys with the machinery `--set` already uses (lists are replaced). Stages write to `output.path/<stage>/`, and the state passes between stages in memory.

```yaml
# base keys: mesh, material, obstacle, solver, output ...
stages:
  - name: release
    regime: quasi_static
    time: {dt: 2.5e4, t_end: 1e5}
    loads: {neumann: [...]}
  - name: strokes
    carry: [displacement]               # displacement | velocity | acceleration | history | time
    regime: dynamic
    dynamics: {scheme: bdf2, density: 1150, mass: lumped}
    time: {dt: 1e-4, t_end: 0.18, max_cutbacks: 8}
```

- (b) **Checkpoint and restart.** `output.checkpoint: true` writes the end state: u, v, a, history variables, time and step, with `node_mapping`. `initial_conditions: {checkpoint: ../release/checkpoint, carry: [...]}` starts from it, and `playground case.yaml --stage strokes` uses that to rerun one stage.
- C++ shape: `ElasticityCase::stage(i)` returns the merged case, `Playground::state()` returns the end state, and `Playground::create(comm, case, const State *start)` takes a start state. `HistoryStorage` (`frontend/ops/prony/sfem_HistoryStorage.hpp`) serialises the Prony state.
- Smaller first step: `extends: ../common.yaml` (a base document plus overrides) removes the YAML duplication alone.

**Removes:** the duplicated YAML, the stage flags in `run_transient.sh`, and the mapping step for sim2's start.

**Effort and risk:** M-L. The merge exists; the State object and history serialisation are new. The risk is medium: carrying state across schemes must respect each scheme's start, for example BDF2's `u_n + dt v_n` predictor (commit `98a021bea`).

## P3. SDF generation outside the run

**Today.**
- The three `tests/sdf_obstacle/cases/*/setup.sh` scripts chain `box_mesh.py`, `skin`, `mesh_to_sdf.py` and `raw_to_xdmf.py`, and write `obstacle/meta.yaml` with `echo`.
- LDP uses the C++ `mesh_to_sdf` (`external/ssdf/demo/mesh_to_sdf.exe.cpp`), configured only by environment (`SSDF_HMAX`, `SSDF_MARGIN[_X|_Y|_Z]`, `SSDF_BOX_MESH`, `SSDF_BOX_BLOCKS`, `SSDF_SCALE`).
- `python/sfem/sdf/mesh_to_sdf.py` is a second implementation, used by workflows/sdf, workflows/resample and workflows/rough.
- SFEM already links ssdf (`frontend/contact/sfem_SelfContact.cpp` calls `ssdf::closest_within_radius_local_bvh` and `potential_contact_quads_bvh`), and it builds analytic SDFs in-process for `obstacle.shape` (`create_obstacle_sdf`, `frontend/mech/sfem_ObstacleShape.hpp:49`).

**Proposal.** Compute the SDF of a surface mesh in-process with `ssdf::sample_sdf_bvh`, the call `mesh_to_sdf.exe.cpp:176` makes, and cache it in the output folder.

```yaml
obstacle:
  sdf:
    surface: ../db/fluid/skin          # surface mesh (or {mesh: ..., sideset: ...})
    hmax: 3e-5
    margin: 1.6e-3                     # or [mx, my, mz]
    box: {mesh: ../db/solid, blocks: [PRESSURE_VALVE, SUCTION_VALVE]}
    scale: 1                           # -1 flips the sign
    cache: sdf_stops                   # reused while the inputs' hash in its meta.yaml matches
```

In C++: `SharedGrid<geom_t> create_sdf_from_surface(comm, surface, const SDFGridSpec &spec)` in the obstacle module. The existing `obstacle.sdf.file` stays.

**Removes:** the SDF stage of `setup.sh` and `prepare.sh`, the environment-configured binary, the YAML comment standing in for the parameters, and the Python duplicate (to `RETIRE.md`).

**Effort and risk:** S-M, since the code exists in the demo. The risk is cost at fine `hmax`. The cache pays it once; under MPI, rank 0 computes and broadcasts, or the distributed `ssdf::pedf` (`external/ssdf/distributed/psdf.hpp`) does the work.

## P4. Every script finds the binaries differently

**Today.**
- `PLAYGROUND` and `SMESH_PY` (set by `env.sh` outside the repository).
- `SFEM_BUILD` (10 files: bench, submit, omp).
- `SFEM_PATH` (11 workflow scripts with "must be defined", then `PATH=$SFEM_PATH/bin`).
- `PRONY_EXE`/`PRONY_PYTHON` (prony_torsion, defaulting to `$ROOT/build/playground` and `$ROOT/venv/bin/python`).
- `SFEM_DIR` (23 files) carries three meanings: the install prefix (`tests/sdf_obstacle/env.sh`), the source root (`workflows/sfc/example.sh`, `workflows/viz`), and CMake's package directory `lib/cmake` (`spikes/cvfem`).
- The generated `sfem_config.sh` (referenced in 55 files) exports only type names. `data/exodus/LeakageTest_Fluid.sh` sources `$SFEM_DIR/workflows/sfem_config.sh`, which does not exist.

**Proposal.** CMake generates one environment file per build and per install: `<build>/sfem_env.sh` and `<prefix>/share/sfem/sfem_env.sh`, from `configuration/sfem_config.sh.in`. It exports:
- `SFEM_ROOT` (source) and `SFEM_PREFIX` (build or install);
- `PATH` (drivers, `python/sfem` tools, smesh Python tools);
- `PYTHONPATH` (pysfem, sfem, smesh);
- `SFEM_PYTHON` (the interpreter the build was configured with);
- today's type exports.

Scripts start with `source "${SFEM_ENV:?source <build>/sfem_env.sh first}"` and call tools by name. `SFEM_DIR` stays only as CMake's package variable.

**Removes:** `SFEM_PATH`, `SFEM_BUILD`, `PRONY_EXE`, `PLAYGROUND` and `SMESH_PY` from scripts, and the existence checks in each script.

**Effort and risk:** S. The risk is low, and scripts can migrate one at a time.

## P5. Duplicated converters that have drifted

**Today.** Three converters exist in both trees:
- `python/sfem/mesh/db_to_raw.py` (166 lines) against smesh's (197 lines): 225 differing lines.
- `exodusII_to_raw.py` (317 against 275 lines): 482 differing lines.
- `python/sfem/grid/raw_to_xdmf.py` is identical to smesh's.

`raw_to_db`, `raw_to_exodusII` and `raw_point_data_to_parent` live only in smesh. CMake installs `python/sfem` into `scripts/` (`CMakeLists.txt:613`), and `tests/sdf_obstacle/env.sh` puts `scripts/sfem/mesh` first on `PATH`. So `db_to_raw` resolves to sfem's copy there, while `run_transient.sh` calls smesh's by path. 42 files call `db_to_raw`.

**Proposal.** smesh owns mesh conversion. Reconcile the two copies by running both on the repository's test meshes and comparing outputs byte for byte, port sfem-only fixes to smesh `main`, and delete sfem's copies. sfem installs smesh's Python tools next to its own, and P4's `PATH` covers both.

**Removes:** three duplicate files and behaviour that depends on `PATH` order.

**Effort and risk:** S, plus the reconciliation. The risk is that a fix exists in only one copy; the byte-for-byte comparison finds it.

## P6. Mesh preprocessing chains

**Today.** `workflows/hyperelasticity/hyperelasticity.sh:17-39` runs this chain:
- `cylinder`, then `db_to_raw --select_elem_type=tetra10`, then `sfc`;
- `create_sideset` twice (point plus radius), then `skin`;
- `surface_from_sideset` twice and `raw_to_db` four times, for viewing only.

The playground `mesh:` section already covers `path` or `box`, `blocks`, `reorder` (`hilbert3`/`morton3`, which is what `sfc` does), `promote_to_p2`, `refine`, `packed`, and sidesets from `plane` or `file`.

**Proposal.** Extend `mesh:` with the remaining steps:

```yaml
mesh:
  cylinder: {radius: 0.1, height: 1, order: 2, refinements: 1}   # generators beside box
  reorder: hilbert3
  sidesets:
    - {name: inlet, point: [-0.51, 0, 0], radius: 0.5}           # create_sideset's selector
    - {name: outlet, point: [0.51, 0, 0], radius: 0.5}
output:
  mesh_views: [skin, sidesets]       # written with XDMF (P1), instead of surface_from_sideset + raw_to_db
```

The generators cover what `python/sfem/mesh` offers (box, rectangle, cylinder, cyclic sector, star), in smesh C++. A foreign format (`path: mesh.vtk`/`.exo`) is converted once into a cached raw folder, with a hash of the source; Exodus reading needs netCDF, which stays in Python.

**Removes:** the `geometry/` stage of most workflow scripts. The mesh becomes a function of the case.

**Effort and risk:** M, since each generator is small. The risk is low-medium.

## P7. Run behaviour split across YAML, `--set` and the environment

**Today.**
- **Reads.** The repository's C/C++ reads 551 distinct `SFEM_*` names (`getenv`, `Env::read`, `SFEM_READ_ENV`). 119 of them are in library code outside drivers, examples, spikes, benchmarks and tests: 68 in frontend, 32 in ssmg, 16 in operators, 4 in algebra.
- **Knobs on the playground path:**
  - `SFEM_ENABLE_MIXED_PRECISION`, read seven times in `frontend/sfem_LinearOperatorFactory.cpp` (`:123
307
323
343
363
533
587`, `:`, ...) next to the YAML key `solver.linear.storage_bytes` for the same choice;
  - `SFEM_USE_PARTIAL_ASSEMBLY`, `SFEM_NEOHOOKEAN_OGDEN_USE_AOS` and `SFEM_HEX8_ASSUME_AFFINE`, alongside material YAML (which also accepts `ASSUME_AFFINE_*` keys);
  - 25 `SFEM_MAMAL_*` knobs;
  - `SFEM_SHIFTED_PENALTY_MULTIGRID_STATS_PATH`.
- **Sets.** Scripts in workflows/, tests/ and scripts/ set 118 distinct `SFEM_*` names, and 25 of them are read by no C/C++ code. For example, `hyperelasticity.sh` exports `SFEM_ELEMENTS_PER_PACK` for the playground, but only two bench drivers read it; smesh reads `SMESH_ELEMENTS_PER_PACK`. `workflows/rough/sim.sh` sets `SFEM_TRACE_FILE`, which nothing reads: the tracer reads `SMESH_TRACE_FILE`, and `cmake/SFEMOptions.cmake:47` documents the wrong one.
- A misspelt or renamed variable is ignored without a word (see `RENAME.md` T2).

**Proposal.**
1. **One registry.** Each knob is declared once: `SFEM_OPTION(name, type, default, doc, yaml_key)`. `sfem::initialize` prints the non-default effective values and warns about every `SFEM_*`/`SMESH_*` variable in the environment that no option declares. That would have caught all 25 above.
2. **One precedence.** The YAML key wins over `--set`, which wins over the environment. Every library knob gets a YAML key; tuning knobs go in an `options:` section of the case, for example `options: {partial_assembly: true, elements_per_pack: 2048}`. The environment stays for drivers and experiments.
3. **One name per choice.** Knobs that duplicate a YAML key are retired, for example `SFEM_ENABLE_MIXED_PRECISION` against `storage_bytes`.
4. **Trace into the output.** `output.trace: true` writes `<output.path>/trace.csv`. The tracer accepts `SFEM_TRACE_FILE` as an alias of `SMESH_TRACE_FILE`.

**Removes:** variables that are set and silently ignored, and the need to read the scripts to know what a run did. The effective options feed P10's `run.yaml`.

**Effort and risk:** M, a mechanical conversion of about 120 call sites in library code. The risk is low if the first step only warns.

## P8. Relative paths resolve inconsistently

**Today.**
- **Consistent inside a case.** Paths in a playground case resolve against the case file's directory (`ElasticityCase::resolve`; `LoadProfile::from_yaml` and the Dirichlet/Neumann readers receive `base_dir`).
- **`--set` values** resolve against the case directory, not the shell's working directory. `run_transient.sh` works because it `cd`s into the copy; `bench.sbatch.sh` passes absolute paths.
- **Cases cannot run in place.** `run_transient.sh` copies the YAML files so that `../db/solid` resolves.
- **Stale assumptions.** The comment at `prony_torsion/run_case.sh:8` says inline Dirichlet paths resolve against the working directory, and the script is written around that.
- **Opt-in flag.** The obstacle `meta.yaml` read by `sfem::Input` has an `rpath: true` flag to opt into relative resolution (`frontend/sfem_ContactConditions.cpp:558`).
- **Mixed key names.** Path keys are spelt `path` (`mesh.path`, `profile.path`, Neumann `path`) or `file` (`sdf.file`, `sidesets[].file`, `fields[].file`, `initial_conditions.*.file`).

**Proposal.** One rule: a path written in a file resolves against that file's directory, and a path given on the command line resolves against the working directory. The schema (P9) marks the path-typed keys, so `--set` knows which values are paths. `file:` names a single file and `path:` a folder; the old keys are aliases with a warning. `rpath` goes, because a path in `meta.yaml` is always relative to it.

**Removes:** the YAML copying and `cd` in `run_transient.sh`, and the `rpath` flag.

**Effort and risk:** S. It changes the meaning of relative `--set` paths, so the first release warns when the two readings differ.

## P9. Unknown YAML keys are not rejected below the top level

**Today.** `ElasticityCase` validates its own keys. It hands these subtrees to other readers whole: `material.*`, `loads.neumann.*`, `loads.profile.*`, `dirichlet_conditions.*`, `constraints[].profile.*`, `fields[].profile.*`, `obstacle.release.*` and `obstacle.ssmgc.*` (`sfem_ElasticityCase.cpp:39-91`). Of those readers, only `RelaxationGroups` rejects unknown keys (`frontend/ops/prony/sfem_RelaxationGroups.cpp:255`). A misspelt material parameter takes the generated default (`material_from_yaml` with `material_defaults`, e.g. `sfem_GeneratedMooneyRivlinProny.cpp:1258-1305`). Generated ops accept two spellings for each affine option. The key list and the documentation comment in `sfem_ElasticityCase.hpp` are kept separately and have drifted: `obstacle.gap_tol` and `output.probes[].name` are in the list but not in the documentation. The other 37 `create_from_yaml` readers, and `sfem::Input`, do not validate.

**Proposal.** Each YAML reader declares its keys as `static const YAMLSchema &X::yaml_schema()`, giving key, type, default, doc and whether it is a path. A shared `yaml_validate(node, schema, dotted_path)` reports unknown keys and type errors. `ElasticityCase` composes the subtree schemas; `material:` gets `OpFactory::yaml_schema(type)`, which the generator emits from the parameter names it already knows. `playground --schema` prints the documented schema, replacing the hand-written comment.

**Removes:** silent defaults, and documentation that drifts from the code.

**Effort and risk:** M. The risk is low.

## P10. Hand-written run bookkeeping

**Today.** `bench.sbatch.sh` writes `env.txt` with a heredoc: ranks, threads, preload, `mpich_gpu_support`, the playground binary's mtime, a sha of `case.yaml`, and the wall time from `date`. `workflows/ldp_containment/python/collect.py` (174 lines) recovers sizes and iteration counts from `run.log` with four regexes (`STEP`, `FAILED`, `PLAYGROUND`, `CONTACT`, at `:37-40`) and reads `trace.csv` for phases. Rewording a log line breaks the benchmark tables.

**Proposal.** The playground writes `<output.path>/run.yaml` at exit, on failure too. It holds:
- the resolved case (`ElasticityCase::emit()`) and its overrides;
- sizes: global and owned elements and dofs, contact nodes;
- per-step and total iterations: Newton, linear, QP, cutbacks;
- phase times from the tracer: setup, solve, qp, linearize, assemble, exchange;
- status and wall time;
- the environment: ranks, threads, `OMP_*`, devices, `CUDA_VISIBLE_DEVICES`, `MPICH_GPU_SUPPORT_ENABLED`, `LD_PRELOAD`, and the non-default options (P7);
- the build: git sha and dirty flag, build type, scalar types, enabled options from `sfem_config.h`;
- hosts and start time.

`collect.py` then reduces to globbing `run.yaml` files into a table (P11).

**Removes:** the `env.txt` heredoc, the log scraping, and the coupling between log wording and benchmarks.

**Effort and risk:** S-M. The risk is low.

## P11. Each workflow post-processes on its own

**Today.** Of the 15 Python scripts in workflows/, 5 read `history.csv`, 5 read raw fields with `np.fromfile`, and 3 plot. Examples are `render_strokes.py`, `mooney_rivlin_gravity/plot_response.py`, `prony_torsion/tools/{validate_torsion_release,analyse_hysteresis,sweep_severity}.py`, `collect.py`, `ab_utopia.py` and `sdf_compare.py`. Each reimplements reading the raw layout and its times, finding the node nearest a point, and picking `history.csv` columns. The playground's probes are `control_point`, `torque`, `contact` and `prony`.

**Proposal.**
- (a) `python/sfem/results.py` provides `Run(path)` with:
  - `.case` and `.meta` (from `run.yaml`);
  - `.history` (a DataFrame) and `.times`;
  - `.field(name, step, numbering="parent")`;
  - `.at(point)` (the time series of the nearest node);
  - `.stages`.
- (b) More probes, so common quantities come out of the run into named `history.csv` columns: `sideset_force` (a reaction), `field_extrema {field, sideset}`, `energy` (strain and kinetic), `nodeset_mean`.

**Removes:** the copy-pasted readers; the workflow scripts shrink to their plots.

**Effort and risk:** M. The risk is low.

## P12. The Python bindings lack the playground, Stepper and Newton

**Today.** `python/bindings/pysfem.cpp` exposes the discretisation and the linear solvers, and nothing from `frontend/mech` or `frontend/nonlinear`: `ElasticityCase`, `Playground`, `Stepper`, `NewtonOptions`, `TimeScheme` and probes are missing. Sweeps such as `prony_torsion/tools/sweep_severity.py` and LDP's `submit.sh` shell out to the binary and parse files.

**Proposal.**

```python
import sfem
case = sfem.ElasticityCase.read("sim2.yaml", overrides=["time.dt=5e-5"])   # or .parse(dict, base_dir)
pg = sfem.Playground(sfem.communicator(), case)
pg.on_step(lambda rec: print(rec.time, rec.newton_iterations))              # Observer
status = pg.run()                                                           # GIL released
u = pg.function().state("displacement")                                     # numpy view
```

`Stepper.step()` allows driving one step at a time, and `case.emit()` returns the resolved case.

**Removes:** subprocess-and-parse loops in sweeps; enables notebooks.

**Effort and risk:** M. The risk is low-medium (MPI and device runs from Python).

## P13. Alps launcher knowledge lives in each script

**Today.**
- **Per-configuration settings in `bench.sbatch.sh`.** It hard-codes CSCS's `cuPointerGetAttribute` override (`/capstor/.../cuptrgetattr_override.so`) for device runs, and `MPICH_GPU_SUPPORT_ENABLED=0` for host runs, because MPICH checks every buffer and that check has been slow since driver 590.
- **Layout and binding.** The same script sets `OMP_PROC_BIND=true OMP_PLACES=cores` and `srun --cpu-bind=cores --distribution=block:block`, and enforces the fill-the-node rules in `parse()`.
- **Copied settings.** `prgenv-gnu/24.11:v2` appears in 95 files, with `24.7:v3` and `25.6:v2` stragglers. `OMP_PROC_BIND` appears in 232 scripts and `~/init.sh` in 73.
- **Settings the script has to know.** `nccl: true` requires `execution_space: device`; the playground rejects it otherwise (`sfem_Playground.cpp:1245`).

**Proposal.**
- (a) One launcher, `scripts/alps/sfem-srun --kind host|gpu --ranks R --threads T -- playground case.yaml ...`. It:
  - sets the binding;
  - sets `MPICH_GPU_SUPPORT_ENABLED` by kind;
  - preloads the override for GPU runs when it is present;
  - sets `CUDA_VISIBLE_DEVICES` per local rank before `MPI_Init`;
  - refuses half-filled nodes.

  The uenv version lives in one file that the submit wrappers pass to `sbatch --uenv`.
- (b) The library checks at `sfem::initialize`. It warns about `MPICH_GPU_SUPPORT_ENABLED=1` on a host run and about its absence on a device run under MPI, and records both in `run.yaml`.

**Removes:** launcher logic duplicated across the bench, omp and spike batch files.

**Effort and risk:** S. The risk is low.

## P14. Stray artifacts

**Today.**
- **Tracked outputs** (listed in `DEAD.md`): `smesh.trace.csv` and `spmg_stats.csv` at the root, `spikes/inexact_apply_compare/smesh.trace.csv`, `docs/partial_assembly.aux`, `wip/paper/prevs.aux`.
- **Library defaults write to the working directory:** the tracer writes `./smesh.trace.csv` (`external/smesh/src/profile/smesh_tracer.cpp:28`), and `ShiftedPenaltyMultigrid` writes `./spmg_stats.csv` (`algebra/sfem_ShiftedPenaltyMultigrid.hpp:1157`).
- **Stale fields.** A rerun into the same `output.path` leaves stale field files (P1).
- **Main checkout.** It also has untracked `*.bak_*` files and the `-h/` and `arning:p` leftovers of mistyped commands.

**Proposal.**
- (a) Every file a run writes goes under `output.path` (`trace.csv`, `spmg_stats.csv`, MaMAL dumps). The environment names stay as overrides, and without `output.path` nothing is written unless asked.
- (b) The playground owns its output folder. `output.clean: true` is the default and removes only files matching its own patterns before the run. The XDMF and `run.yaml` list exactly this run's files.
- (c) `.gitignore` gains `*.trace.csv`, `spmg_stats.csv`, `*.aux` and `*.bak_*`, and the tracked files are removed per `DEAD.md`.

**Effort and risk:** S. The risk is low, given that `clean` is restricted to the playground's own file patterns.

---

## Priorities

| # | Proposal | Impact | Effort | Depends on API cleanup |
|---|----------|--------|--------|------------------------|
| 1 | P8 path rule (file-relative or CWD-relative), `file`/`path` keys | high, every run | S | no |
| 2 | P4 generated `sfem_env.sh` | high, every script | S | no |
| 3 | P14 run outputs under `output.path`, `output.clean` | medium | S | no |
| 4 | P10 `run.yaml` | high for benchmarks | S-M | no (better after P7) |
| 5 | P1 XDMF time series and `numbering: parent` | high, removes 5 conversions per LDP case | M | no |
| 6 | P5 one set of converters (smesh) | medium | S | no |
| 7 | P7 option registry, environment warnings | high, removes silent no-ops | M | partly (options that move into ExecutionContext) |
| 8 | P9 per-reader YAML schemas, `--schema` | high, correctness | M | yes: schemas sit on `X::create_from_yaml`/`OpFactory` (RENAME F13-F15) |
| 9 | P3 in-process SDF from a surface | medium | S-M | yes, lightly: lands in the obstacle module, not `sfem_API.hpp` |
| 10 | P2 stages and checkpoint/restart | high for LDP and Prony | M-L | yes: the carried State holds device buffers through ExecutionContext |
| 11 | P13 Alps launcher and initialise checks | medium, Alps only | S | the checks in (b) belong to ExecutionContext creation |
| 12 | P11 `results.py` and more probes | medium | M | no (reads P1/P10 outputs) |
| 13 | P6 mesh generators and selectors in YAML | medium | M | no |
| 14 | P12 Python playground, Stepper, Newton | medium | M | yes: bind after the renames and ExecutionContext, or bind twice |

**Independent of the API cleanup** (they can start now): P1, P4, P5, P6, P8, P10, P11 and P14.

**After the `sfem_API.hpp` reorganisation and the RENAME.md factories:** P9, P3 and P12. Their YAML schemas and bindings attach to the final `X::create`/`create_from_yaml` names.

**After the ExecutionContext** (BLAS, execution space and stream in one object): P2's carried state, P13(b), P12's device runs, and the precision and storage knobs of P7.
