# Codegen Framework Refactoring Plan

The current code-generation framework is too fragmented. Energy-based,
residual-based, hyperelastic, two-phase, mixed-order, affine, and
isoparametric paths still carry duplicated logic and incompatible assumptions.
The target design is a layered framework where user-facing symbolic
specification, form manipulation, and platform code generation have clear
responsibilities and one shared downstream pipeline.

## Design Goals

- Use one general equation-system abstraction for scalar, vector, and coupled
  multiphysics formulations.
- Treat merit/energy and residual/gradient as different user-facing wordings
  that produce the same standardized 0-, 1-, and 2-form collection below the
  symbolic layer.
- Keep FEM concerns, symbolic form manipulation, and target/platform emission
  in separate modules.
- Generate both monolithic coupled kernels and individual block kernels for
  block systems.
- Support affine and isoparametric geometry through the same form plan, with
  sum factorization used for tensor-product isoparametric elements.
- Preserve SFEM performance expectations: SoA streams, vectorizable loops,
  minimal copies, and no unnecessary branches in hot loops.

## Target Architecture

### Symbolic Layer

The symbolic layer is the user-facing mathematical API.

Responsibilities:

- Provide UFL-conforming classes and operators that remain fully compatible with
  SymPy.
- Represent fields, test functions, trial functions, gradients, deformation
  gradients, geometric quantities, coefficients, and qualifiers as symbolic
  objects.
- Allow users to define systems from energy/merit functions or directly from
  residual/gradient forms.
- Automatically derive standardized 0-, 1-, and 2-forms:
  - 0-form: energy or merit function
  - 1-form: gradient or residual
  - 2-form: Hessian action or Jacobian action
- Preserve extra codegen qualifiers where useful, for example hyperelasticity
  markers that identify deformation-gradient-dependent quantities.

Input:

- User UFL-conforming specification.

Output:

- `EquationSystem` with fields, equations, qualifiers, and complete
  standardized form collection.

### Form Manipulation Layer

The form manipulation layer transforms standardized forms into a concrete
kernel plan. This layer decides what is computed at mesh level, what is
computed locally, and which transformations are needed.

Responsibilities:

- Build generation plans from an `EquationSystem` and element compatibility
  metadata.
- Apply automated policies for affine and isoparametric geometry.
- Use sum factorization for tensor-product elements whenever values,
  gradients, geometry, or test contractions are evaluated on isoparametric
  tensor-product cells.
- Separate the plan into mesh-level and local-kernel stages.

Mesh-level kernel policy:

1. Route affine geometry by passing Jacobian adjugate and determinant as SoA
   inputs, one value per element.
2. Compute isoparametric Jacobian, adjugate, and determinant per quadrature
   point from coordinates.
3. Use tensor-product/sum-factorized geometry evaluation for tensor-product
   isoparametric elements.
4. Gather coefficient vectors into SoA lane buffers before local kernel calls.
5. Scatter residuals/actions from SoA lane buffers back to global vectors.

Local-kernel policy:

1. Evaluate field values, reference gradients, and higher derivatives as
   required by the form.
2. Transform reference quantities with the geometric adjugate and determinant.
3. Evaluate material terms from energy/merit, gradient/residual, or
   Hessian/Jacobian action.
4. Use the left-operand style already used in the high-performance kernels so
   trial-related terms are arranged before test contractions.
5. Apply test-function contractions only when required by the target form.
6. Use sum factorization for tensor-product field evaluation and test
   contractions.

Input:

- `EquationSystem` with standardized form collection and qualifiers.

Output:

- `GenerationPlan` describing mesh kernels, local kernels, block kernels,
  geometry policy, element policy, data streams, diagnostics, and target
  requirements.

### Code Generation Layer

The code generation layer consumes expression graphs and a plan. It must not
know whether the user started from hyperelasticity, residual equations, or a
mixed multiphysics model.

Responsibilities:

- Emit general kernels for any combination of equations described by the plan.
- Generate both monolithic coupled kernels and separate block kernels for
  multiphysics systems.
- Emit OpenMP, CUDA, and future target-specific code through target classes.
- Emit common diagnostics and performance-report helpers from the same plan.
- Generate C ABI wrappers and `sfem::Op` subclasses from the same metadata.
- Keep target-specific implementation details out of symbolic and form
  manipulation modules.

Input:

- `GenerationPlan`
- target platform objects such as OpenMP and CUDA
- expression graphs

Output:

- generated local kernels
- generated mesh-level kernels
- generated diagnostics
- generated C ABI wrappers
- generated OOP wrappers

## Milestones

### M1. Symbolic Layer Unification

Goal: replace ad hoc symbolic entry points with a UFL-conforming symbolic API that
produces a complete `EquationSystem`.

Tasks:

1. Define symbolic field classes for scalar, vector, and tensor fields with
   SymPy-compatible behavior.
2. Define symbolic test and trial objects for each field.
3. Define UFL-conforming operators for value, gradient, divergence, deformation
   gradient, inner product, determinant, inverse, adjugate, and common tensor
   operations.
4. Add qualifier classes for codegen-relevant semantics, such as
   hyperelastic deformation-gradient dependence and material parameters.
5. Replace separate energy/residual input adapters with a single
   `EquationSystemBuilder`.
6. Implement automatic derivation of 0-, 1-, and 2-forms for energy/merit
   inputs.
7. Implement direct registration of residual/gradient forms and automatic
   derivation of Jacobian actions.
8. Normalize naming so merit/energy and residual/gradient share one internal
   form representation.
9. Add tests for scalar, vector, mixed scalar-vector, energy-derived, and
   residual-derived systems.
10. Update example materials to use only the new symbolic API.

Acceptance criteria:

- NeoHookean, two-phase flow, and poro-hyperelasticity can all be expressed
  through the same symbolic API.
- The output of the symbolic layer is always an `EquationSystem` with complete
  standardized forms.
- No downstream code branches on user-facing material type.

### M2. Standard Form Collection

Goal: make 0-, 1-, and 2-form data explicit, uniform, and independent of the
original user wording.

Tasks:

1. Introduce a `FormCollection` object owned by `EquationSystem`.
2. Store form order, equation name, fields, coefficients, qualifiers, and
   dependency metadata in one schema.
3. Remove separate hyperelastic and residual payload schemas where they encode
   the same information.
4. Add dependency analysis that removes unused current, previous, direction,
   parameter, geometry, and coefficient inputs.
5. Represent block structure explicitly for coupled systems with the goal to allow subproblems code gen (as well as mixed terms).
6. Add tests that verify unused quantities are not passed to generated kernels.
7. Update `PIPELINE.md` to describe the standardized form collection.

Acceptance criteria:

- Every equation exposes the same form-order interface.
- Block structure is available before code generation.
- Kernel signatures contain only quantities used by the selected form.

### M3. FEM and Geometry Policy Separation

Goal: move FEM-specific data, geometry evaluation, and element compatibility
out of material/codegen logic.

Tasks:

1. Create a dedicated FEM policy module for elements, bases, quadrature,
   field-family compatibility, and mixed-order pairs.
2. Represent affine and isoparametric geometry as explicit plan nodes.
3. Represent simplex and tensor-product basis evaluation as explicit plan
   nodes.
4. Implement one shared tensor-product sum-factorization abstraction for:
   - field values
   - field gradients
   - geometry Jacobians
   - test contractions
5. Replace duplicated tensor-product geometry code in residual and
   hyperelastic generators.
6. Add Taylor-Hood policies by detecting mixed-FE forms
7. Add tests for affine and isoparametric geometry plans on simplex and
   tensor-product elements.

Acceptance criteria:

- Geometry handling is specified by plan nodes, not by scattered generator
  branches.
- Tensor-product isoparametric geometry uses sum factorization through one
  shared abstraction.
- Mixed-order field shape and scatter sizes are selected from field-family
  compatibility metadata.

### M4. Generation Plan Unification

Goal: replace specialized payloads with one plan language consumed by all code
generators.

Tasks:

1. Define `GenerationPlan`, `KernelPlan`, `BlockPlan`, `GeometryPlan`, and
   `DataStreamPlan` classes.
2. Express mesh-level gather, geometry, local-call, and scatter phases as plan
   nodes.
3. Express local-kernel evaluation order as plan nodes:
   - evaluate trial quantities
   - transform reference quantities
   - evaluate material terms (left-operand style)
   - contract with tests
4. Add plan validation for unsupported element/form/target combinations.
5. Generate both monolithic and block plans for coupled systems.
6. Add plan dumps for inspection and debugging.
7. Update existing material scripts to emit and optionally save plan dumps.

Acceptance criteria:

- The same `GenerationPlan` representation covers hyperelasticity,
  residual-only models, and coupled poro/thermo-hyperelastic systems.
- Monolithic and block kernels are generated from the same plan.
- The code generation layer no longer reconstructs FEM or material semantics.

### M5. Target Platform Layer

Goal: isolate OpenMP, CUDA, and future target-specific code generation.

Tasks:

1. Define target classes for OpenMP and CUDA.
2. Move target-specific pragmas, memory qualifiers, launch structures, and
   vectorization assumptions into target classes.
3. Add target hooks for SoA layout, vector size, alignment, restrict
   qualifiers, and diagnostic instrumentation.
4. Make OpenMP target generate AVX512-friendly loops by default.
5. Make CUDA target use CUDA-safe math helpers, including specialized
   `pow_y(x)` functions.
6. Add compile tests for target-specific generated code.
7. Add vectorization diagnostics for OpenMP generated kernels.

Acceptance criteria:

- Symbolic and form-manipulation layers are target-independent.
- OpenMP and CUDA codegen share plan traversal but differ only through target
  classes.
- Compile/vectorization tests catch regressions in generated hot loops.

### M6. Unified Kernel Emission

Goal: make generated code uniformly structured across all materials and
elements.

Tasks:

1. Emit local kernels by dimension and family, not by material-specific
   special cases.
2. Emit mesh-level kernels per element type or compatible element type.
3. Emit monolithic coupled kernels for complete systems.
4. Emit separate block kernels for each block in a coupled system. Make sure to detect 0 blocks!
5. Ensure affine kernels receive adjugate and determinant as SoA inputs.
6. Ensure isoparametric kernels compute geometry from coordinates according to
   the geometry plan.
7. Remove redundant temporary buffers and avoid copies where pointer swapping
   or direct SoA lane buffers are sufficient.
8. Generate diagnostics for each emitted kernel from the shared diagnostics
   plan.

Acceptance criteria:

- Generated filenames and signatures follow one convention across materials.
- Local kernels are reusable by family and dimension when possible.
- Mesh-level kernels are specialized by element and geometry policy.

### M7. OOP Wrapper and C ABI Generation

Goal: generate runtime integration code from the same plan metadata.

Tasks:

1. Generate C ABI declarations for all monolithic and block kernels.
2. Generate `sfem::Op` subclasses for complete equation systems.
3. Support independent affine/isoparametric choices for objective, gradient,
   residual, Hessian action, and Jacobian action.
4. Generate `create_from_yaml` using model parameters from the symbolic layer.
5. Support block-system runtime dispatch.
6. Integrate with SFEM Dirichlet-condition design instead of hard-coded
   boundary handling.
7. Add wrapper compile tests for generated operators.

Acceptance criteria:

- OOP wrappers are generated for energy-only, residual-only, and coupled
  systems.
- YAML model parameters are handled consistently.
- Runtime wrappers call generated kernels without material-specific manual
  glue code.

### M8. Migration and Cleanup

Goal: remove legacy duplicated paths once the unified path is complete.

Tasks:

1. Move user-intended material examples into `python/codegen/framework/materials`.
2. Move infrastructure APIs into `sfem.gen`.
3. Delete legacy single-physics material adapter internals (make sure the unified framework can reproduce the generations).
4. Remove duplicate local-kernel and mesh-kernel generation code.
5. Remove duplicated tensor-product geometry generation.
6. Remove material-specific special cases from the code generation layer.
7. Update all docs, examples, and scripts.
8. Add regression tests comparing generated kernels against reference Python
   implementations for representative systems.

Acceptance criteria:

- The framework has one symbolic-to-plan-to-code pipeline.
- Legacy APIs are thin compatibility wrappers or removed.
- Existing examples still generate, compile, and pass action/reference tests.

## Required Regression Coverage

- Symbolic API tests for scalar, vector, tensor, mixed, energy-derived, and
  residual-derived equations.
- Form-collection tests for 0-, 1-, and 2-form consistency.
- Dependency-pruning tests for current, previous, direction, parameters, and
  geometry inputs.
- Plan-validation tests for affine/isoparametric and simplex/tensor-product
  combinations.
- Codegen compile tests for OpenMP generated kernels.
- Vectorization diagnostics tests for OpenMP hot loops.
- Kernel action tests against hardcoded Python references for `TRI3`, `TET4`,
  `HEX8`, Taylor-Hood pairs, and `HEX27`.
- Generated wrapper compile tests.

## Non-Goals

- Do not add material-specific code paths to fix individual examples.
- Do not let the code generation layer inspect user callbacks.
- Do not silently fall back from mixed-order formulations to equal-order
  kernels.
- Do not duplicate tensor-product geometry, field evaluation, or test
  contraction logic across material families.


DO NOT MAINTAIN RETROCOMPATIBILITY!!!

## TODO

### Acceptance Gaps To Close

The following items are not yet fully satisfied by the current implementation.
Keep this list focused on acceptance criteria, not general cleanup.

#### M2. Standard Form Collection

- Kernel-signature dependency pruning is only proven for selected residual
  paths. Add generated-signature tests for energy-derived kernels and boundary
  residual kernels, and make unused parameters/geometry/current/previous/
  direction inputs impossible across all emitters, not only residual_codegen.

#### M3. FEM and Geometry Policy Separation

- Geometry plan nodes exist, but geometry handling is still partly encoded in
  scattered generator branches in `symbolic.py`, `residual_codegen.py`, and
  `boundary_codegen.py`. Move affine/isoparametric routing and geometry preamble
  emission behind shared plan/backend helpers so code generators consume
  geometry plans instead of reconstructing geometry policy locally.
- Tensor-product isoparametric geometry uses shared helpers in several paths,
  but residual, hyperelastic, and boundary emitters still enter those helpers
  through separate code paths. Make tensor-product field evaluation, geometry
  Jacobian evaluation, and test contraction dispatch come from one shared
  abstraction in the backend.

#### M4. Generation Plan Unification

- `GenerationPlan` and `KernelPlan` are present, but backend emission still
  dispatches through specialized generator entry points:
  `generate_sfem_soa_cpp_files_for_element`,
  `generate_coupled_residual_sfem_files`,
  `generate_mixed_residual_sfem_files`, and
  `generate_boundary_residual_sfem_files`. Replace these with one backend
  traversal over `KernelPlan`, `BlockPlan`, geometry plans, basis plans, and
  data-stream plans.
- `EnergyCodeGenerationPayload` still carries energy-specific emission state.
  Remove it once the plan contains all kernel expressions, diagnostics, and
  data-stream metadata needed by the backend.
- The code-generation layer still reconstructs some FEM/material semantics
  during emission, e.g. diagonal mixed block models and field-specific element
  labels. Move these decisions into form manipulation so code generation only
  consumes explicit plan data.

#### M5. Target Platform Layer

- CUDA is only represented by target metadata. There is no CUDA backend
  traversing the unified plan, no CUDA generated kernel path, and no CUDA
  compile tests.
- OpenMP and CUDA do not yet share plan traversal with target-specific hooks.
  Implement target classes that own pragmas, qualifiers, alignment/vector-size
  assumptions, math helpers, and diagnostics while consuming the same plan.
- OpenMP vectorization diagnostics exist for selected generated kernels, but
  they do not yet cover every representative hot loop family required by the
  acceptance criteria. Extend vectorization tests to all emitted OpenMP kernel
  families that are expected to vectorize.

#### M6. Unified Kernel Emission

- Local and mesh kernel emission is still split between energy-style
  `symbolic.py`, residual-style `residual_codegen.py`, and boundary-specific
  `boundary_codegen.py`. Merge these into one local-kernel and one mesh-kernel
  emission path driven by the unified plan.
- Local kernels are reusable by dimension/family in several cases, but not
  uniformly across energy, residual, mixed, and boundary kernels. Ensure local
  kernel naming, signatures, template parameters, reference data, and include
  structure are generated by common code for every material family.
- Diagnostics are emitted for generated kernels, but the diagnostic generation
  path is still coupled to specialized emitters. Generate diagnostics from the
  shared plan for every emitted monolithic and block kernel.

#### M7. OOP Wrapper and C ABI Generation

- Generated wrappers exist for energy-only, residual-only, boundary, and
  coupled systems, but runtime registration still requires manually maintained
  includes/registrations in SFEM frontend code. Generate or automate the
  factory-registration metadata from the same plan/wrapper output.
- Boundary-condition handling supports generated Neumann sidesets, but the
  generated wrapper layer is not yet a general SFEM condition integration
  mechanism. Keep Dirichlet/Neumann/runtime condition handling aligned with the
  existing SFEM condition abstractions.
- Wrapper compile coverage exists, but action/reference tests through the
  generated `sfem::Op` wrappers are still limited. Add runtime wrapper tests
  that execute generated energy-only, residual-only, and coupled operators, not
  only compile and factory-create them.

#### M8. Migration and Cleanup

- Legacy low-level generator APIs are still exported and used directly by tests
  and backend code. Decide which low-level APIs remain internal implementation
  details, stop exporting them as public framework API, and route examples
  through `sfem.gen.CodeGenerator`.
- `python/codegen/framework/twophaseflow.py` and other historical helper paths
  should be audited for obsolete standalone pipeline logic. Remove or convert
  anything not using the unified `EquationSystem` -> `FormCollection` ->
  `GenerationPlan` path.
- Existing reference/action tests cover NeoHookean on `TRI3`, `TET4`, `HEX8`,
  and `HEX27`, and residual diffusion on `TRI3`, `TET4`, and `HEX8`. Add
  hardcoded Python reference action tests for Taylor-Hood generated kernels
  (`TRI6_TRI3`, `TET10_TET4`, `HEX27_HEX8`) and at least one coupled
  poro-hyperelastic block/monolithic path.
- Add regression coverage that generated files for all maintained material
  examples still compile after regeneration, including generated Op wrappers
  and C ABI declarations.
