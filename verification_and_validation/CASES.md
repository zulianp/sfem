# SFEM Solid-Mechanics Verification and Validation Cases

## Purpose

This catalog defines the solid-mechanics cases that belong in
`verification_and_validation`. A successful driver exit is not evidence of
correctness. Every case must reproduce an independent analytical,
semi-analytical, manufactured, or measured oracle within the tolerances stated
here and in its `case.yaml`.

The planned suite covers 2D and 3D elasticity, hyperelasticity, and
viscoelasticity. It emphasizes constitutive modes first, then spatial and
temporal convergence, and finally structural benchmarks. A case may contain
several operator or element variants when the physical problem and oracle are
identical.

## Supported Material Scope

The scope below reflects the operators present in the repository, not every
solid-related source file.

| Family | Operator | Current dimensional/element scope | V&V treatment |
| --- | --- | --- | --- |
| Linear elastic | `LinearElasticity` | Standard 2D/3D and semi-structured 3D paths | Reference implementation in both patch cases; semi-structured variants are execution variants, not new physics cases. |
| Linear elastic | `GeneratedLinearElasticity` | 2D `TRI3`, `TRI6`, `QUAD4`; 3D `TET4`, `TET10`, `HEX8`, `HEX27`; generated Proteus variants | Required in both patch and structural cases. |
| Hyperelastic | `GeneratedNeoHookeanOgden` | Same generated 2D/3D element families | Required in homogeneous 2D and 3D cases. |
| Hyperelastic | `GeneratedModifiedMooneyRivlin` | Same generated 2D/3D element families | Required in homogeneous cases and pressure-vessel cases. Its 2D formulation is a plane-strain 3D embedding. |
| Hyperelastic | `NeoHookeanOgden` | Legacy 3D `TET4`/`TET10`/`HEX8` paths | Run as 3D conformance variants against the same oracle as the generated operator. |
| Hyperelastic | `NeoHookeanOgdenPacked` | Legacy packed 3D path with operation-dependent element support | Enable an element variant only after a driver capability check confirms all operations required by the solver. |
| Hyperelastic | `MooneyRivlin`, `MooneyRivlinActiveStrainPacked` | Legacy packed 3D `HEX8` path | Pure-material mode belongs in the homogeneous case; active strain gets a separate case. |
| Hyperelastic | `GeneratedSaintVenantKirchhoff` | Generated 2D/3D source and kernels are present, but the operator is not currently registered by `sfem_generated_ops_registration.cpp` | Activate before adding it as a required homogeneous-case variant. It must not be reported as covered until driver creation succeeds. |
| Viscoelastic | `GeneratedMooneyRivlinKelvinVoigtNewmark` | Generated 2D and 3D element families | Canonical finite-strain Kelvin-Voigt target in both dimensions. |
| Viscoelastic | `KelvinVoigtNewmark` | Legacy 3D `HEX8` and semi-structured hex only | Cover with the analytical damped-mode case. |
| Viscoelastic | `MooneyRivlinVisco` | Legacy 3D `HEX8` only; Prony series and optional WLF shift | Cover with homogeneous relaxation and reduced-time WLF variants. |

`Hyperelasticity` is a generic/plugin-facing operator rather than a distinct
constitutive law, so it does not receive a separate physics case.
`GeneratedMooneyRivlin` currently has a generator definition but no active
frontend operator and is not a required target. Poro-hyperelasticity, contact,
plasticity, shells, beams, and fluid-structure coupling are future expansions
outside this material-focused catalog.

## Coverage Summary

| Dimension/family | Exact or manufactured verification | Structural or transient validation |
| --- | --- | --- |
| 2D elastic | `linear_patch_2d` | `kirsch_plate_hole_2d` |
| 3D elastic | `linear_patch_3d` | `lame_spherical_shell_3d` |
| 2D hyperelastic | `hyperelastic_modes_2d` | `cylindrical_pressure_vessel` |
| 3D hyperelastic | `hyperelastic_modes_3d` | `inflated_spherical_shell_3d` |
| 2D viscoelastic | `finite_strain_kv_creep_2d` | Same case includes the complete transient history. |
| 3D viscoelastic | `finite_strain_kv_creep_3d` | `linear_kv_damped_mode_3d`, `prony_relaxation_3d` |

## Case Definitions

### EL-2D-01: `linear_patch_2d`

- **Type:** verification; fast gate.
- **Target:** `LinearElasticity` and `GeneratedLinearElasticity` on `TRI3` and
  `QUAD4`; add `TRI6` and generated Proteus variants to the extended lane.
- **Setup:** unit square with all boundary displacements prescribed from three
  affine fields: deviatoric extension, simple shear, and mixed volumetric
  strain. Include at least one non-axis-aligned mesh.
- **Oracle:** closed-form displacement, constant small strain,
  `sigma = 2 mu epsilon + lambda tr(epsilon) I`, total energy, and boundary
  reactions. Interpret the 2D law as plane strain when converting from
  `E, nu`.
- **Pass conditions:** displacement relative L2 error at or below `1e-10`,
  normalized free/interior residual at or below `1e-10`, and relative energy
  and resultant-reaction errors at or below `1e-9`.
- **Failure modes exposed:** component ordering, Lame parameter mapping,
  shear factors, element orientation, integration, and generated/legacy drift.

### EL-3D-01: `linear_patch_3d`

- **Type:** verification; fast gate.
- **Target:** `LinearElasticity` and `GeneratedLinearElasticity` on `TET4` and
  `HEX8`; `TET10`, `HEX27`, semi-structured, and packed modes are extended
  variants.
- **Setup:** unit cube with affine deviatoric extension, simple shear, and
  triaxial volumetric strain. Use both an aligned tensor-product mesh and a
  skewed tetrahedral mesh.
- **Oracle:** closed-form displacement, strain, stress, strain energy, and face
  resultants.
- **Pass conditions:** the same algebraic tolerances as EL-2D-01.
- **Failure modes exposed:** 3D coupling terms, z-component layout, tetra/hex
  dispatch, packed layout, and semi-structured specialization errors.

### EL-2D-02: `kirsch_plate_hole_2d`

- **Type:** verification with spatial convergence; medium gate.
- **Target:** the two linear-elastic operators on `TRI3` and `QUAD4`.
- **Setup:** quarter annulus around a traction-free circular hole. Apply
  symmetry on the coordinate axes and the analytical plane-strain Kirsch
  displacement on the finite outer arc. Generate at least three radial and
  circumferential refinement levels.
- **Oracle:** the Kirsch displacement and polar stress fields restricted to the
  finite computational domain. Sample radial and angular rays and exclude only
  the element layer touching polygonal corners when evaluating convergence.
- **Pass conditions:** finest-mesh relative displacement L2 error at or below
  `1%`, stress relative L2 error at or below `3%`, hole-boundary peak hoop
  stress error at or below `5%`, monotone error reduction, and an observed
  displacement L2 rate of at least `1.7` for first-order elements.
- **Failure modes exposed:** curved geometry, symmetry constraints, stress
  recovery, nonuniform displacement data, and convergence regressions hidden
  by a single mesh.

### EL-3D-02: `lame_spherical_shell_3d`

- **Type:** verification with spatial convergence; medium gate.
- **Target:** the two linear-elastic operators on `TET4` and `HEX8`.
- **Setup:** one octant of a thick spherical shell under internal pressure,
  with symmetry planes and a traction-free exterior. Generate three radial and
  angular refinements entirely from source.
- **Oracle:** Lame's spherical solution,
  `u_r = A r + B/r^2`, with analytical radial and hoop stresses. Use the
  pressure and radii declared in `case.yaml` to derive `A` and `B` at runtime.
- **Pass conditions:** finest-mesh displacement relative L2 error at or below
  `2%`, radial and hoop stress relative L2 errors at or below `4%`, inner-face
  pressure resultant error at or below `2%`, monotone refinement, and
  displacement L2 rate at least `1.6`.
- **Prerequisite:** 3D pressure traction support in YAML. Until that exists,
  analytical radial displacement may be prescribed on the inner and outer
  surfaces, but that reduced setup does not satisfy the pressure-resultant
  check and must be reported as partial coverage.

### HE-2D-01: `hyperelastic_modes_2d`

- **Type:** constitutive verification; fast gate.
- **Target:** `GeneratedNeoHookeanOgden`,
  `GeneratedModifiedMooneyRivlin`, and, after registration,
  `GeneratedSaintVenantKirchhoff`; `TRI3` and `QUAD4` are required.
- **Setup:** a unit square subjected independently to finite uniaxial plane
  strain, simple shear, and uniform in-plane dilation. Prescribe the exact
  affine boundary field and solve for interior nodes.
- **Oracle:** hand-written closed-form strain energy and first Piola stress
  obtained from each documented energy. The modified Mooney-Rivlin oracle must
  use `F33 = 1` and 3D invariants, matching its stated plane-strain model.
- **Pass conditions:** normalized free/interior residual at or below `1e-9`;
  relative energy and face-resultant errors at or below `1e-8`; positive
  deformation Jacobian at every sampled quadrature point.
- **Failure modes exposed:** finite-strain kinematics, determinant terms,
  invariant definitions, 2D embedding, objective/gradient inconsistency, and
  material parameter naming.

### HE-3D-01: `hyperelastic_modes_3d`

- **Type:** constitutive verification; fast gate.
- **Target:** the generated hyperelastic operators on `TET4` and `HEX8`, plus
  compatible legacy and packed Neo-Hookean/Mooney-Rivlin variants. Add
  `TET10` and `HEX27` to the extended lane.
- **Setup:** unit cube under finite uniaxial deformation, simple shear, and
  isotropic dilation. Add one combined nonsymmetric deformation gradient to
  exercise all off-diagonal terms.
- **Oracle:** independent closed-form energy and first Piola stress for the
  exact deformation gradient.
- **Pass conditions:** the same constitutive tolerances as HE-2D-01, evaluated
  separately for every operator/element variant.
- **Failure modes exposed:** 3D invariants, volumetric response, objectivity,
  tensor component layout, and legacy/generated divergence.

### HE-2D-02: `cylindrical_pressure_vessel`

- **Type:** published-data validation; medium gate; already implemented.
- **Target:** `GeneratedModifiedMooneyRivlin` on a quarter-cylinder `QUAD4`
  mesh using `hyperelasticity_bdf2` with zero density and ramped follower
  pressure.
- **Setup:** inner radius `7 m`, outer radius `18.625 m`, `20 x 20` cells,
  `c10 = 80 MPa`, `c01 = 20 MPa`, and final pressure `100 MPa`, matching the
  [solids4foam benchmark](https://www.solids4foam.com/tutorials/more-tutorials/solid-mechanics/hyperelasticity/cylindricalPressureVessel.html).
- **Oracle:** checked-in radial and hoop Cauchy-stress profiles attributed to
  the benchmark dataset and its cited large-strain reference solution.
- **Pass conditions:** radial and hoop relative L2 errors at or below `4%`,
  maximum absolute error for either stress at or below `5 MPa`, all load steps
  converged, and positive deformation Jacobian.
- **Extension:** add `10 x 10` and `40 x 40` meshes and require monotone error
  reduction; retain the published `20 x 20` comparison as the canonical gate.

### HE-3D-02: `inflated_spherical_shell_3d`

- **Type:** nonlinear structural verification; extended gate.
- **Target:** `GeneratedModifiedMooneyRivlin` and
  `GeneratedNeoHookeanOgden` on `TET4` and `HEX8` octant meshes.
- **Setup:** thick spherical shell inflated by ramped internal follower
  pressure. Run moderate-compressibility and nearly incompressible parameter
  sets over three mesh refinements.
- **Oracle:** an independent one-dimensional radial boundary-value solution.
  For the incompressible variant, use the spherical mapping
  `r(R)^3 = R^3 + c` and pressure equilibrium quadrature; for the compressible
  variant, solve radial equilibrium at high precision without calling SFEM
  kernels.
- **Pass conditions:** inner-radius displacement error at or below `2%`,
  radial/hoop stress relative L2 errors at or below `4%`, pressure-volume curve
  relative L2 error at or below `2%`, positive Jacobian, and monotone spatial
  error reduction.
- **Prerequisite:** conservative 3D follower-pressure value, gradient, and
  tangent contributions.

### VE-2D-01: `finite_strain_kv_creep_2d`

- **Type:** constitutive and time-integration verification; fast gate.
- **Target:** `GeneratedMooneyRivlinKelvinVoigtNewmark` on `TRI3` and `QUAD4`.
- **Setup:** laterally constrained strip under a smooth axial traction ramp
  followed by a hold. The constraints reduce the homogeneous response to one
  stretch variable while retaining finite strain. Test shear viscosity alone
  and combined shear/bulk viscosity.
- **Oracle:** independently evaluate the elastic first Piola stress and viscous
  stress, then integrate the resulting scalar first-order creep equation with
  a high-accuracy adaptive solver. Also check the small-strain exponential
  limit analytically.
- **Pass conditions:** stretch-history relative L2 error at or below `0.5%`,
  final-stretch error at or below `0.2%`, reaction-history error at or below
  `1%`, and temporal convergence rate at least `1.8` over three time steps.
- **Failure modes exposed:** 2D deviatoric projection, previous-state wiring,
  Newmark velocity reconstruction, bulk viscosity, and load scaling.

### VE-3D-01: `finite_strain_kv_creep_3d`

- **Type:** constitutive and time-integration verification; fast gate.
- **Target:** `GeneratedMooneyRivlinKelvinVoigtNewmark` on `TET4` and `HEX8`.
- **Setup and oracle:** 3D counterpart of VE-2D-01 with both lateral
  components constrained. Add a simple-shear state to distinguish 3D
  deviatoric and bulk viscosity.
- **Pass conditions:** the same history and temporal-rate limits as VE-2D-01.
- **Failure modes exposed:** 3D rate tensor, viscous Piola transformation,
  tetra/hex dispatch, and inertia-free transient behavior.

### VE-3D-02: `linear_kv_damped_mode_3d`

- **Type:** analytical transient verification; medium gate.
- **Target:** legacy `KelvinVoigtNewmark` on `HEX8` and one semi-structured hex
  variant.
- **Setup:** rectangular bar with lateral displacement constrained and an
  initial axial mode `sin(pi x/L)`. Release it with no external load in the
  underdamped regime. Generate the initial displacement and velocity fields as
  case artifacts.
- **Oracle:** the separated Kelvin-Voigt mode
  `q(t) = exp(-delta t) [q0 cos(omega_d t) +
  (v0 + delta q0)/omega_d sin(omega_d t)]`, with modal mass, stiffness, and
  damping derived from the continuum parameters and geometry.
- **Pass conditions:** modal-amplitude relative L2 error at or below `0.5%`,
  maximum normalized amplitude error at or below `0.2%`, fitted frequency and
  decay-rate errors at or below `1%`, and temporal convergence rate at least
  `1.8`.
- **Failure modes exposed:** mass, elastic and viscous scaling, Newmark
  parameters, initial conditions, and semi-structured specialization.

### VE-3D-03: `prony_relaxation_3d`

- **Type:** history-variable verification; medium gate.
- **Target:** `MooneyRivlinVisco` on `HEX8`.
- **Setup:** homogeneous constrained uniaxial step-and-hold deformation. Run a
  one-term Prony series first, then a separated multi-term series. A WLF variant
  repeats the history at two temperatures.
- **Oracle:** for the small-strain hold, the normalized relaxation modulus
  `G(t)/G0 = g_inf + sum(g_i exp(-t/tau_i))`. For WLF, replace time by reduced
  time using the independently evaluated shift factor and the operator's stated
  `tau_eff = tau_ref/a_T` convention. The finite-strain
  reaction is the elastic reaction multiplied by the corresponding hereditary
  factor for the prescribed hold.
- **Pass conditions:** reaction-history relative L2 error at or below `1%`,
  long-time modulus error at or below `0.5%`, each fitted relaxation-time error
  at or below `2%`, and WLF reduced-time curve-collapse error at or below `1%`.
- **Prerequisite:** a driver path that initializes, advances, and commits
  `MooneyRivlinVisco` history while reading the prescribed deformation history
  from YAML-referenced files.

### HE-3D-03: `active_strain_homogeneous_3d`

- **Type:** constitutive verification; optional extended gate.
- **Target:** `NeoHookeanOgdenActiveStrainPacked` and
  `MooneyRivlinActiveStrainPacked` on `HEX8`.
- **Setup:** unit cube with a constant diagonal active deformation gradient.
  Evaluate the undeformed configuration and the compatible stress-free total
  deformation.
- **Oracle:** the multiplicative decomposition used by the operators, evaluated
  independently for homogeneous `F` and `Fa`, including energy and face
  resultants.
- **Pass conditions:** relative energy and reaction errors at or below `1e-8`
  and normalized interior residual at or below `1e-9`.
- **Prerequisite:** file-backed active-strain fields in the driver. This case is
  not required for the initial elastic/hyperelastic/viscoelastic coverage gate.

## Element and Backend Policy

- Every physics case has one primary simplex and one primary tensor-product
  element where the operator supports both. Unsupported combinations are
  recorded as `SKIP` with a reason, never silently omitted.
- First-order CPU matrix-free variants form the required baseline. Second-order,
  packed, semi-structured, assembled BSR, and device variants reuse the same
  physical oracle as extended conformance runs.
- Agreement between two SFEM operators or backends is useful diagnostic data,
  but it is not an oracle and cannot be the sole pass condition.
- Spatial cases use at least three resolutions. Temporal cases use at least
  three time steps. A finest-grid tolerance alone is insufficient when a
  convergence rate is part of the case definition.

## Oracle and Tolerance Rules

1. The oracle implementation must not call the SFEM material operator or its
   generated kernels. Compact formulas may be derived with SymPy, but the
   checked-in verifier must expose the final equations and provenance.
2. Every reported check stores observed value, expected value, error norm,
   tolerance, units, oracle type, and pass/fail status in `verification.json`.
3. Dimensionless relative norms use an absolute floor for zero or near-zero
   quantities. Point values never replace field norms unless the benchmark
   defines only a measured point.
4. Tolerances above are initial acceptance limits. They may be tightened after
   calibration. Loosening requires a documented numerical or oracle reason and
   must not be an automatic baseline update.
5. Solver convergence, finite values, positive `J` for finite-strain cases,
   complete time histories, and expected output counts are mandatory guards in
   addition to the physical comparisons.
6. Reference datasets are checked in with license/provenance notes and a pinned
   upstream revision. Network access is never required to run a case.
