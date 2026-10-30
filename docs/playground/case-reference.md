# Playground case reference

Every key a playground case accepts, by section. Keys are optional unless marked
**required**. Relative paths are read against the case file's directory.

The schema (`src/frontend/mech/sfem_ElasticityCase.cpp`) rejects unknown keys everywhere except
in the subtrees marked *passed through* below, which their own readers interpret.

- [Execution](#execution)
- [mesh](#mesh)
- [material](#material)
- [loads](#loads)
- [dirichlet_conditions](#dirichlet_conditions)
- [constraints](#constraints)
- [fields](#fields)
- [initial_conditions](#initial_conditions)
- [time](#time)
- [dynamics](#dynamics)
- [obstacle](#obstacle)
- [solver](#solver)
- [output](#output)
- [Profiles](#profiles)
- [Restrictions](#restrictions)

## Execution

| Key | Values | Default | Meaning |
|---|---|---|---|
| `execution_space` | `host`, `device` | `host` | `device` runs on the GPU; needs a CUDA build. |
| `nccl` | bool | `false` | With `device` and more than one rank: the exchange and the contact QP's reductions within a node go through NCCL. Its set-up costs about 3 s once and pays off from about 90,000 QP iterations (measured on Alps). Needs a build with `SFEM_ENABLE_NCCL` that found NCCL. |
| `regime` | `steady`, `quasi_static`, `dynamic` | `steady` | `steady` is one solve (`dt = 1`, one step). `quasi_static` steps through time without inertia; `dynamic` with it (see [dynamics](#dynamics)). |

## mesh

**Required**, with either `path` or `box`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `path` | path | | An SFEM (smesh) raw mesh folder. |
| `blocks` | list of names | all | With `path`: read only these blocks and their nodes. The written mesh's `node_mapping` gives each node's id in the file. |
| `box.n` | `[nx, ny, nz]` | `[1, 1, 1]` | A HEX8 box mesh with `n` elements per axis, instead of `path`. |
| `box.min`, `box.max` | `[x, y, z]` | `[0, 0, 0]`, `[1, 1, 1]` | The box's corners. |
| `reorder` | `none`, `hilbert3`, `morton3` | `none` | Reorder elements and nodes along a space-filling curve, before the sidesets are read. |
| `promote_to_p2` | bool | `false` | TET4 to TET10. Not with `refine`; one rank only. |
| `refine` | int | `0` | Semi-structured refinement level; one rank only. |
| `basis` | `micro`, `high_order` | `micro` | On a refined mesh: trilinear micro elements, or one Lagrange element of the level per macro element. |
| `packed` | bool | `true` | The packed element layout (faster operators). |
| `renumber` | bool | `true` | Packing renumbers the nodes; `false` keeps the numbering of the file, for results compared node by node with it (refused if packing would have to move nodes). |
| `sidesets` | list | | Named boundaries for conditions, constraints and the obstacle; see below. |

Each entry of `sidesets` has a **required** `name` and either a `file` (a sideset folder) or a
`plane`. An empty sideset is an error.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `plane.axis` | `x`, `y`, `z` (or `0`, `1`, `2`) | **required** | The plane's normal axis. |
| `plane.at` | `min`, `max` or a coordinate | **required** | Where, against the mesh's bounding box. |
| `plane.tol` | real | `1e-6` | Tolerance, relative to the box's extent along `axis`. |
| `plane.block` | name | | Only faces of this block's elements. |
| `plane.center`, `plane.radius` | `[x, y, z]`, real > 0 | | Only faces whose centre lies within `radius` of `center` (a disc). Both or neither. |

## material

**Required**. Passed through to the operator factory; `type` names the operator.

| `type` | Parameters (default) |
|---|---|
| `GeneratedLinearElasticity` | `mu` (1), `lmbda` (1) |
| `GeneratedNeoHookeanOgden` | `mu` (1), `lmbda` (1) |
| `GeneratedMooneyRivlinProny` | `C10` (1), `C01` (0), `K` (1), `relaxation` |
| `GeneratedLinearKelvinVoigt` | `mu`, `lmbda`, `eta_s` (0.1), `eta_b` (0), `u_dt_shift` (1) |
| `GeneratedMooneyRivlinKelvinVoigt` | as `GeneratedLinearKelvinVoigt` |
| `NeoHookeanOgden`, `NeoHookeanOgdenPacked`, `NeoHookeanOgdenActiveStrainPacked` | `mu`, `lambda` (hand-written; note `lambda`, not `lmbda`) |

For the generated materials:

- parameters may also sit under `parameters:`;
- `young_modulus` and `poisson_ratio` (0 <= nu < 0.5) can replace `mu` and `lmbda` for linear
  elasticity, NeoHookean-Ogden and the Kelvin-Voigt materials (mixing the two pairs is an error);
- `blocks: [{name: B, <parameters>}]` overrides parameters per block, and restricts the operator to
  the listed blocks;
- `assume_affine: true` (and `assume_affine_objective`, `_gradient`, `_hessian_action`, `_apply`)
  selects the affine-element kernels.

Unknown parameters are ignored: check their spelling.

**Prony relaxation** (`GeneratedMooneyRivlinProny`, group `iso`):

```yaml
material:
  type: GeneratedMooneyRivlinProny
  C10: 0.5
  C01: 0.1
  K: 50
  relaxation:
    iso:
      prony: [{g: 0.3, tau: 0.1}, {g: 0.2, tau: 1.0}]
      integrator: linear            # or midpoint
      history_storage: full         # full | float32 | float16 | float16_scaled | int16_scaled | int8_scaled
      wlf: {C1: 17.4, C2: 51.6, T_ref: 20}   # optional time-temperature shift
      temperature: 20
```

Unknown keys under `relaxation.<group>` and undeclared groups are errors. A history-dependent
material needs `quasi_static` or `dynamic`; `dynamic` needs a material that supports time
stepping.

## loads

| Key | Type | Default | Meaning |
|---|---|---|---|
| `body_force.g` | list, one entry per component | **required** with `body_force` | Acceleration, e.g. `[0, 0, -9.81]`. |
| `body_force.density` | real | `1` | |
| `neumann` | list, or `{file: f.yaml}` | | Tractions and pressures; passed through. The file form reads the file's `neumann_conditions` key. |
| `profile` | [profile](#profiles) | constant 1 | Scales every Neumann condition, on top of each condition's own `profile`. |

A `neumann` entry:

- `type: sideset`, or `type: pressure` for a follower pressure with a scalar `value`;
- `format: name` with `name` (a sideset of `mesh.sidesets`), or `format: file` with `path`;
- `component` (int or list), `value` (scalar, list, or `{path: values.raw}` for one component);
- an optional `profile`.

```yaml
loads:
  neumann:
    - {type: sideset, format: name, name: right, component: 2, value: -0.01}
  profile: {type: linear_ramp, start_time: 0, end_time: 1}
```

## dirichlet_conditions

A list, or `{file: f.yaml}` whose file holds a `dirichlet_conditions` key; passed through.
Each entry:

- **on a sideset**: `type: sideset`, `format: name` with `name`, or `format: file` with `path`;
- **on a node set**: no `type`, and `nodes: [...]` or `format: file` with `path`. Node ids are
  the file's numbering; they are mapped through packing;
- `component` (int or list), `value` (scalar, list, or `{path: values.raw}` with one value per
  node of the set), and an optional `profile` (host only).

```yaml
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
  - {type: sideset, format: name, name: right, component: 0, value: 0.1,
     profile: {type: linear_ramp, start_time: 0, end_time: 1}}
```

## constraints

Rigid rotations of a sideset (host only).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `type` | `rotation` | **required** | |
| `sideset` | name | **required** | |
| `axis` | `x`, `y`, `z` | **required** | |
| `center` | `[x, y, z]` | `[0, 0, 0]` | A point on the axis. |
| `angle` | real | | The angle, times `profile(t)`. Exactly one of `angle` or `control`. |
| `control.torque` | real, not 0 | **required** with `control` | The angle is found at which the reaction torque is `torque * profile(t)`. At most one rotation may be torque-controlled. |
| `control.tol`, `control.max_it` | real, int | `1e-6` (relative), `20` | |
| `profile` | [profile](#profiles) | constant 1 | |
| `release` | time | | The constraint is removed from the first step that ends at or after this time. |

```yaml
constraints:
  - {type: rotation, sideset: right, axis: x, center: [0, 0.5, 0.5], angle: 0.5,
     profile: {type: linear_ramp, start_time: 0, end_time: 1}, release: 2}
```

## fields

Fields handed to the material (`Op::set_field`), each with a **required** `name` and either:

- `file`: a raw `real_t` file with `values_per_element` (default 9) values per element;
- `inclusion`: an active strain `diag(a, 1, 1)` in a ball, `a(t) = 1 - (1 - a11) profile(t)`,
  with `a11` (default 0.5), `radius` (default a quarter of the box's smallest side) and `center`
  (default the box's centre);

and an optional `profile`.

## initial_conditions

`displacement`, `velocity` and `acceleration`, each `{file: f.raw}` (interleaved, one value per
component per node) or `{components: [u.0.raw, u.1.raw, u.2.raw]}`. Values are in the mesh file's
node numbering.

- `velocity` needs `dynamic` with Newmark or BDF2; `acceleration` Newmark (otherwise the run starts
  from zero acceleration).
- With a displacement, history-dependent materials start relaxed in that state.
- Under MPI every rank reads the whole file.

## time

| Key | Type | Default | Meaning |
|---|---|---|---|
| `dt` | real > 0 | **required** (not for `steady`) | Step length. |
| `steps` | int | | Number of steps; takes precedence over `t_end`. |
| `t_end` | real | | Final time; `steps = round(t_end / dt)`. One of `steps` or `t_end` is **required**. |
| `max_cutbacks` | int | `4` | A failed step is retried with half its length, this many times. |

## dynamics

Used when `regime: dynamic`.

| Key | Values | Default | Meaning |
|---|---|---|---|
| `scheme` | `newmark`, `bdf2`, `backward_euler` | `newmark` | |
| `density` | real | `1` | |
| `mass` | `lumped`, `consistent` | `lumped` | `consistent` is refused with `backward_euler` and with an obstacle. |
| `beta`, `gamma` | real | `0.25`, `0.5` | Newmark parameters; `gamma >= 0.5` and `beta >= gamma / 2` are required. `beta = (gamma + 1/2)^2 / 4` damps high frequencies. |

With an obstacle, Newmark becomes the contact-stabilised Newmark scheme.

## obstacle

A rigid obstacle the body must not penetrate. Needs a 3D mesh and an SFEM built with
`SFEM_ENABLE_SRESAMPLE`. The body approaches along `-direction`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `backend` | `node_sdf`, `ssmgc` | `node_sdf` | `node_sdf`: every contact node against the SDF's cubic B-spline, a projected-gradient QP (MPRGP) per Newton step. `ssmgc`: the shifted-penalty multigrid on a semi-structured mesh. |
| `nodes` | `skin` or a sideset name | `skin` | The contact nodes. `node_sdf` with a sideset needs `mesh.packed: false` or `mesh.renumber: false`. |
| `direction` | `x`, `y`, `z` | `z` | |
| `sdf.file` | path | | An SDF grid folder (`meta.yaml` + samples), instead of `shape`. |
| `shape.type` | `plane`, `slope`, `staircase`, `hemisphere` | `hemisphere` | An analytic obstacle, sampled to a grid. |
| `shape.plane` | real | `0` | The plane's height, or the top of the hemisphere. |
| `shape.slope`, `shape.slope_axis` | real, axis | `0.2`, the first tangential axis | `slope`. |
| `shape.step_width`, `shape.step_drop` | real | `0.25`, `-0.05` | `staircase` (`step_drop < 0`). |
| `shape.radius`, `shape.center` | real, `[x, y, z]` | a quarter of the footprint's smaller side; under the body | `hemisphere`. |
| `shape.fillet` | real | the step depth (staircase), `radius / 2` (otherwise) | Rounds edges. |
| `shape.grid_cells` | int | `128` | Grid resolution along the longest side (finer if the fillet needs it). |
| `gap` | `nodal`, `dual_weighted` | `nodal` | `ssmgc`: the gap at each node, or weighted over its surface patch. |
| `curvature` | bool | `true` (`node_sdf`); `true` with the nodal gap (`ssmgc`) | The obstacle's curvature in the Newton tangent. |
| `gap_tol` | length | `1e-10` × extent (`node_sdf`), `1e-8` × extent (`ssmgc`) | Admitted penetration. |
| `shift` | real | `0` | `node_sdf`: a clearance kept from the obstacle. |
| `cutoff` | real | `10000` | `node_sdf`: no contact for a node farther than this. |
| `release` | [profile](#profiles) | | `node_sdf`: a start inside the obstacle is let out of it as the profile goes from 0 to 1. |
| `qp.max_it`, `qp.rtol` | int, real | `20000`, `1e-10` | `node_sdf`'s QP. |
| `qp.accept_capped` | bool | `false` | Accept a QP stopped by `max_it` as the step. |
| `ssmgc` | map | | Passed through to SSMGC (e.g. `penalty_param`, `coarse_op_type`, `nlsmooth_steps`), over the step's own settings. |

Where each backend applies:

- `node_sdf`: every regime. In `dynamic`, with Newmark or BDF2.
- `ssmgc`: `dynamic` with Newmark, one rank, `mesh.refine >= 2` with the `micro` basis, and a
  linear material. `shift`, `cutoff` and `release` are not available.

```yaml
obstacle:
  backend: node_sdf
  direction: z
  shape: {type: hemisphere, plane: 0, radius: 0.25}
```

## solver

| Key | Default | Meaning |
|---|---|---|
| `newton.max_it` | `30` | |
| `newton.atol`, `newton.rtol` | `1e-9`, `1e-8` | Converged when the residual measure is below `max(atol, rtol × first)`. |
| `newton.divergence_factor` | `1e4` | |
| `newton.merit` | `auto` | `auto`, `energy`, `residual`. |
| `newton.fixed_iterations` | `false` | `true` accepts the state after `max_it` iterations. |
| `newton.verbose` | `true` | One line per Newton iteration. |
| `newton.line_search.enabled` | on for nonlinear materials | |
| `newton.line_search.steps` | `[1, 1/2, ..., 1/128]` | |
| `newton.line_search.c1` | `1e-4` | |
| `newton.line_search.batched` | `true` | |
| `newton.line_search.rule` | `armijo` | Or `damped`: `damping` (in (0, 1]) times the step, shortened only past an inverted element. |
| `linear.type` | `cg` | Or `bcgs`. |
| `linear.format` | `BSR` | The tangent's format: `MF` (matrix-free), `BSR`, `BSR_SYM`, `CRS`, `CRS_SYM`, `SPLITCRS`, `ALIGNEDCRS`, `DIA`, `COO_SYM`, `SPLITDACRS`, `SELL`, `INEXACT`. |
| `linear.preconditioner` | `jacobi` | Or `none`. |
| `linear.rtol` | `1e-3` (nonlinear), `newton.rtol` (linear) | |
| `linear.atol` | `1e-16` | |
| `linear.max_it` | `2000` | |
| `linear.storage_bytes` | `8` | `4` stores the BSR tangent in single precision (`node_sdf`'s too); `2` in half precision (host, without constraints or obstacle). |
| `linear.inexact.store` | `points` | With `format: INEXACT`, what the material's operator stores of its tangent: `points` (the tangent at every quadrature point, moments on TET10), `element_matrix` (element matrices in difference form), or `bsr` (its assembled tangent, quantized). |
| `linear.inexact.encoding` | `f32` | `points`: `f64`, `f32`, `f16`, `i8md` (int8 deviations from an f32 element mean). `element_matrix`: `f64`, `f32`, `f16`, `i8pair` (int8 with a factor per node pair). `bsr`: `f16`, `i8` (a scale per block). |
| `linear.inexact.fold` | `auto` | The folded point store of the hexahedra (exact, fewer numbers): `auto`, `on`, `off`. On the device only HEX8 folds. |
| `linear.inexact.traversal` | `auto` | `packed` (the packed mesh, deterministic) when the space has one, `atomic` otherwise. On the device the store chooses (`auto` only). |

With an obstacle, the obstacle's step replaces the linear solver.

`format: INEXACT` applies the stored tangent of every operator that has one (GeneratedNeoHookeanOgden,
GeneratedLinearElasticity, GeneratedLinearKelvinVoigt, GeneratedMooneyRivlinKelvinVoigt,
GeneratedMooneyRivlinProny, on the 3D elements each is generated for) and the exact apply of the
others. The
stored tangent is the exact one up to its encoding's rounding: `f64` gives BSR's iterations. A
material whose tangent is not symmetric (Mooney-Rivlin Kelvin-Voigt) needs `linear.type: bcgs`.

```yaml playground
mesh:
  box: {n: [4, 4, 4]}
  sidesets:
    - {name: left, plane: {axis: x, at: min}}
    - {name: right, plane: {axis: x, at: max}}
material: {type: GeneratedNeoHookeanOgden, young_modulus: 1, poisson_ratio: 0.3}
dirichlet_conditions:
  - {type: sideset, format: name, name: left, component: [0, 1, 2], value: [0, 0, 0]}
  - {type: sideset, format: name, name: right, component: [0], value: [0.1]}
regime: steady
solver:
  linear: {format: INEXACT, preconditioner: none, inexact: {store: points, encoding: f16}}
```

## output

| Key | Default | Meaning |
|---|---|---|
| `path` | none: no field output | Where the mesh, fields and obstacle grid go. Needed by `fields` and `quantities`. |
| `export_freq` | `1` | Write every k-th step (the initial and the last state always). |
| `history_csv` | none | A CSV row per step: status, iterations and probe values. Independent of `path`. |
| `fields` | `[]` | Written beside the displacement: `velocity`, `acceleration` (dynamic), `material_reaction`, `gap`, `contact_force` (with an obstacle). |
| `quantities` | `false` | `path/quantities.yaml`: objective and reaction resultants per step; one rank only. |
| `probes` | `[]` | Columns of the history; see below. |

Probes, each with a `type` and an optional `name` that prefixes its columns:

| `type` | Keys | Columns |
|---|---|---|
| `control_point` | `at: [x, y, z]` (the nearest node), `rates` (default `true` in `dynamic`) | `ux, uy, uz`, and `vx..vz, ax..az` with rates |
| `torque` | `constraint` (index into `constraints`, default 0) | `angle, released, torque` |
| `contact` | | `reaction_x, reaction_y, reaction_z, touched` (with an obstacle) |
| `prony` | | `<group>_gamma`, `<group>_H<k>` (with a Prony material) |

See [outputs](outputs.md) for the files.

## Profiles

Every `profile` (and `loads.profile`, `obstacle.release`) is a map with a **required** `type`:

| `type` | Keys (default) | Value at time t |
|---|---|---|
| `constant` | `value` (1) | `value` |
| `linear_ramp` (or `ramp`) | `start_time` (0), `end_time` (1), `start_value` (0), `end_value` (1) | linear between the two, constant outside |
| `smooth_ramp` | as `linear_ramp` | cosine blend |
| `hold` | `start_time` (0), `before_value` (0), `value` (1) | `before_value`, then `value` from `start_time` |
| `pulse` | `start_time` (0), `end_time` (1), `before_value` (0), `value` (1), `after_value` (0) | |
| `sine` | `value` (1), `start_time` (0), `period` (1), `end_time` (infinite) | `value sin(2 pi (t - start_time) / period)`, 0 after `end_time` |
| `tabulated` | `path` (**required**) | Linear interpolation of a file of `time, value` lines (commas or spaces, `#` comments, increasing times), constant beyond its ends. |

## Restrictions

**Under MPI** (more than one rank):

- `mesh.refine` and `mesh.promote_to_p2` are not available;
- `obstacle.backend: ssmgc` is not available;
- `output.path` needs `mesh.path`, because rank 0 gathers the fields onto the file's mesh;
- `output.quantities` is not available;
- `initial_conditions` need the mesh's node ids.

**On the device** (`execution_space: device`):

- a torque-controlled or released rotation is not available.
