# Verification and Validation Implementation Plan

## Outcome

Implement the cases in [CASES.md](CASES.md) as self-contained folders under
`verification_and_validation`. The existing top-level `run_all.py` remains the
single entry point and returns nonzero whenever setup, convergence, output
integrity, or an oracle comparison fails. No case depends on files under
`workflows` or on network access.

## Design Constraints

- Drivers execute the simulations; verifier scripts do not substitute their
  own finite-element solve.
- `case.yaml` is the source of truth for geometry, materials, loads, time
  integration, variants, oracle provenance, and tolerances.
- Meshes, sidesets, nodesets, nonuniform boundary data, and initial conditions
  are deterministically generated inside each case output directory.
- Oracles are independent of SFEM kernels. Cross-operator agreement is recorded
  only as a diagnostic.
- The suite remains runnable with the repository `venv/bin/python` and
  `PYTHONPATH=python`.
- Output stays outside source case folders, is isolated per case/variant, and
  can be deleted without losing inputs or oracle data.

## Phase 0: Freeze the Suite Contract (Complete)

1. Document a versioned `case.yaml` schema and retain schema-version-1 support
   for `cylindrical_pressure_vessel`.
2. Add schema-version-2 support for a `variants` list. Each variant declares an
   operator, element, mesh resolution, driver, environment, expected output,
   and tolerance overrides.
3. Add runner selectors for family, dimension, tier, operator, and variant, for
   example `--family hyperelastic --dimension 3 --tier fast`.
4. Give every result one of `PASS`, `FAIL`, `ERROR`, or `SKIP`. `SKIP` requires
   a machine-readable reason and never counts as coverage.
5. Extend `report.yaml` with family, dimension, operator, element, resolution,
   oracle type, per-variant timing, and aggregate coverage counts.
6. Validate all YAML before running a driver. Reject empty variants, duplicate
   IDs, missing tolerances, missing oracle provenance, and reports without
   physical checks.

**Acceptance:** the existing pressure-vessel case produces the same pass/fail
decision through the new runner, malformed test manifests fail before driver
execution, and `--list` displays planned coverage clearly.

## Phase 1: Shared Case Infrastructure (Complete)

Create `verification_and_validation/common/` with small, reusable modules for:

- SFEM raw-array and mesh metadata I/O;
- deterministic rectangle, box, annulus, cylindrical-sector, and spherical-
  shell mesh generation;
- sideset/nodeset construction and orientation checks;
- file-backed nodal boundary values and initial fields;
- displacement gradients, deformation gradients, small strain, `J`, first
  Piola stress, Cauchy stress, energy, and boundary resultants;
- absolute, relative L2, weighted L2, maximum, and curve errors;
- spatial and temporal convergence-rate fits;
- creation and validation of the common `verification.json` format.

Keep constitutive oracle formulas in each case, or in explicitly named oracle
modules, so the independent equations remain reviewable. Do not create one
shared implementation that mirrors the generated SFEM material framework.

**Acceptance:** unit tests exercise raw I/O, sideset orientation, norm floors,
resultant integration, convergence fitting, and JSON validation on synthetic
data.

## Phase 2: Close Driver and YAML Gaps (Complete)

Implement only the input capabilities required by the catalog:

1. Add optional Neumann YAML input to the static `linear_elasticity` driver and
   preserve its existing command-line form for compatibility. Let the same
   driver receive per-block material parameters from the case configuration.
2. Export unconstrained material reactions and objective values from static
   drivers after convergence, before constraints zero or replace residual
   entries. Homogeneous cases use these driver-produced quantities for their
   energy and resultant checks.
3. Add file-backed per-node values to Dirichlet YAML. The manifest references
   generated files; absolute paths are resolved by the runner.
4. Add file-backed initial displacement and velocity fields to the transient
   drivers.
5. Add simple YAML load profiles: constant, linear ramp, hold, pulse, and a
   tabulated time/value file. Apply the same profile machinery to Dirichlet and
   Neumann data.
6. Implement conservative 3D follower pressure for oriented `TRI3` and `QUAD4`
   surfaces, including value, residual, and consistent tangent. Add unit tests
   for orientation, rigid motion, finite differences, and pressure resultants.
7. Add or adapt a history-aware driver for `MooneyRivlinVisco`: initialize
   history, solve each prescribed state, commit history only after convergence,
   and export the reaction/time history.
8. Register `GeneratedSaintVenantKirchhoff` in the generated-operator factory
   and build lists. Add a driver-creation smoke test before enabling its V&V
   variants.
9. Normalize material naming at the case layer: generated operators use
   `lmbda`, while legacy operators may use `lambda`; the manifest maps one
   physical parameter to the correct operator key.

**Acceptance:** each new input feature has a focused frontend/driver test, old
driver invocations still work, and no V&V case relies on procedural state hidden
inside a driver.

## Phase 3: Fast Exact Verification Cases

Implement these folders first because they give broad coverage at low runtime:

1. `linear_patch_2d`
2. `linear_patch_3d`
3. `hyperelastic_modes_2d`
4. `hyperelastic_modes_3d`

Each folder contains `case.yaml`, `generate_mesh.py`, YAML templates, an
independent `oracle.py`, `verify.py`, and a short README defining equations and
units. Generate affine boundary values rather than embedding raw mesh data in
the repository.

For every mode, compare the free/interior residual, total energy, and integrated
face reactions. Run the legacy/generated and simplex/tensor-product variants
against the same expected values. Add second-order and packed variants only
after the primary variants pass.

**Acceptance:** all algebraic checks meet the tolerances in CASES.md, deliberate
perturbations of `mu`, `lambda/lmbda`, a displacement component, and element
orientation each cause a clear failure.

## Phase 4: Spatial Structural Cases

1. Complete `cylindrical_pressure_vessel` provenance by recording the pinned
   upstream data revision, units, interpolation convention, and extraction
   locations. Add `10 x 10` and `40 x 40` diagnostic refinements while keeping
   the published `20 x 20` gate.
2. Implement `kirsch_plate_hole_2d` with three generated polar meshes,
   analytical outer-boundary displacement, stress sampling rays, and a fitted
   displacement convergence rate.
3. Implement `lame_spherical_shell_3d` after 3D pressure support. Verify mesh
   orientation and surface area before solving, then compare displacement,
   radial/hoop stresses, and pressure resultant over three refinements.
4. Implement `inflated_spherical_shell_3d` last in this phase. Validate the
   independent radial solver against limiting small-strain and incompressible
   formulas before using it as an oracle.

**Acceptance:** every case meets both its finest-grid tolerance and required
convergence behavior. The report includes sampled CSV profiles and compact
machine-readable diagnostics, not only plots.

## Phase 5: Time-Dependent Viscoelastic Cases

1. Implement `finite_strain_kv_creep_2d` and
   `finite_strain_kv_creep_3d` using the generated Mooney-Rivlin Kelvin-Voigt
   driver. Start with shear viscosity only, then enable bulk viscosity.
2. Implement `linear_kv_damped_mode_3d` using file-backed initial fields. Project
   every output state onto the analytical mode and report off-mode energy as a
   diagnostic.
3. Implement `prony_relaxation_3d` with a one-term series before the multi-term
   series. Add the WLF variants only after reference-temperature relaxation is
   correct.
4. Run three time-step sizes for every transient case. Compare full histories,
   fitted physical parameters, and temporal convergence, not just the final
   state.
5. Check history commit semantics explicitly by forcing one rejected/non-
   converged trial in a focused driver test; rejected states must not advance
   Prony history.

**Acceptance:** all transient checks and temporal-rate requirements in CASES.md
pass, time arrays are complete and strictly increasing, and missing intermediate
outputs fail report validation.

## Phase 6: Extended Operator Coverage

After the core physics cases are stable:

- add `TRI6`, `TET10`, `HEX27`, generated Proteus, packed, and semi-structured
  variants where supported;
- run assembled BSR and matrix-free paths against the same oracle;
- add device variants when the build exposes them, with an explicit `SKIP`
  reason on hosts without a device;
- implement `active_strain_homogeneous_3d` and file-backed `Fa` input;
- add multi-block cases with different material parameters only after each
  single-material operator passes independently.

These are conformance extensions. They must not duplicate oracle files or
create backend-specific tolerances without a documented numerical reason.

## Reporting and Regression Policy

The terminal summary should remain compact: one row per case and a short list
of failed checks. `report.yaml` contains all variant details. Each case may also
write CSV profiles suitable for plotting, but image generation is not part of
the pass path.

A regression is any of the following:

- a driver or generator exits nonzero;
- required output is absent, incomplete, non-finite, or non-converged;
- a physical oracle check exceeds its tolerance;
- a required spatial or temporal convergence rate is lost;
- a finite-strain quadrature point has `J <= 0`;
- a required operator/element variant is silently skipped;
- an oracle or tolerance changes without a corresponding reviewed source
  change.

The top-level process returns nonzero for `FAIL` or `ERROR`. Optional extended
variants may be skipped only for declared unavailable build capabilities. The
suite remains outside `workflows`; any later automation should invoke
`verification_and_validation/run_all.py` rather than duplicate case logic.

## Proposed Delivery Order

| Milestone | Cases/capability | Completion signal |
| --- | --- | --- |
| M1 | Schema v2, common report tools, exact 2D/3D linear patches | Both elastic dimensions pass on simplex and tensor-product elements. |
| M2 | Hyperelastic mode cases and Saint Venant-Kirchhoff activation | Every active hyperelastic law reproduces energy and reactions in its supported dimensions. |
| M3 | Cylindrical vessel hardening and Kirsch case | Published 2D hyperelastic and analytical 2D elastic structural gates pass. |
| M4 | 3D pressure support and both spherical-shell cases | Linear and nonlinear 3D structural profiles converge to independent radial oracles. |
| M5 | Finite-strain Kelvin-Voigt creep in 2D/3D | Full histories and temporal rates pass. |
| M6 | Legacy Kelvin-Voigt and Prony/WLF histories | All active viscoelastic implementations have an independent transient oracle. |
| M7 | Higher-order, packed, semi-structured, assembled, and device variants | Extended coverage is reported without weakening core tolerances. |

## Definition of Done for a Case

A case is complete only when:

1. its folder is self-contained except for SFEM drivers and shared V&V helpers;
2. a clean output directory can be generated from `case.yaml`;
3. the oracle equations, source, units, extraction method, and tolerances are
   documented;
4. all required operator/element/resolution variants execute through drivers;
5. `verification.json` contains nonempty physical checks and diagnostics;
6. the top-level runner reports pass and returns zero;
7. a controlled perturbation demonstrates that the intended regression is
   detected;
8. no checked-in mesh, solver output, cache, or network fetch is required.
