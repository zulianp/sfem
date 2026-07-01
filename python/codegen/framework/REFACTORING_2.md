# Codegen Framework Refactoring Plan 2

This plan turns the remaining TODO items from `REFACTORING.md` into concrete
implementation tasks. The goal is to close the remaining acceptance gaps without
adding new material-specific paths.

## Principles

- After `EquationSystem.form_collection(...)`, every downstream stage consumes
  `FormCollection` and explicit plan data only.
- Code generation must be driven by `GenerationPlan`, `KernelPlan`,
  `BlockPlan`, geometry plans, basis plans, and data-stream plans.
- Target-specific choices belong to target/backend classes.
- Generated hot loops must remain SoA, branch-light, and vectorization-friendly.
- No retro-compatibility shims for removed APIs.

## Architecture
The architecture is layered: Symbolic Layer (no string generation), Form unification layer, 
laning layer, Code generation / emission layer (string generation). 
Emitters take the full plan and generate the kernels

## M1. Complete Dependency-Pruned Form Collections

Goal: make dependency pruning uniform for energy, residual, and boundary forms.

Status: implemented.

Tasks:

1. Extend `FormMetadata` dependency coverage for energy-derived 0-, 1-, and
   2-forms so it records only used parameters, geometry, current values,
   previous values, and directions.
2. Extend boundary residual lowering so `FormCollection.form_metadata(...)`
   carries complete dependency metadata for generated boundary kernels.
3. Add generated-signature tests for energy kernels that prove unused material
   parameters are not emitted in objective, gradient, or apply signatures.
4. Add generated-signature tests for boundary residual kernels that prove unused
   parameters, current fields, previous fields, directions, and geometry inputs
   are not emitted.
5. Make backend emission reject a kernel plan whose signature requests inputs
   not declared by the selected `FormMetadata`.

Acceptance criteria:

- Every emitted kernel signature is explainable from `FormMetadata`.
- Dependency-pruning tests cover energy, residual, boundary residual, and
  coupled block kernels.
- No emitter performs ad hoc dependency analysis outside form lowering or form
  manipulation.

Verification:

- `PYTHONPATH=python python -m unittest python.codegen.framework.test_gen_api`

## M2. Move Geometry and Basis Policy Fully Into Plans

Goal: remove geometry and basis-policy reconstruction from emitters.

Status: implemented. Context-specialized emission plans now carry affine and
isoparametric geometry phase data, geometry streams, field-specific basis plans,
reference-data streams, and tensor-product sum-factorization plans. OpenMP
emission validates and consumes geometry modes through a shared
`ElementEmissionPlan` before calling the current low-level generators. Energy,
coupled residual, mixed residual, and boundary residual entry points now require
this shared plan instead of reconstructing element family from element names or
quadrature flags. Shared `emission_plan_for_element(...)` construction is used
for field-element plans such as synthetic diagonal mixed-order residual blocks,
so the OpenMP backend no longer reconstructs FEM geometry/basis details itself.
Energy and mixed residual local/mesh/diagnostic reference routing uses the
basis-plan-derived family, including reference parameter, pointer,
call-argument, wrapper-argument, and diagnostic-data helpers. Isoparametric
Jacobian generation is routed separately through the geometry-plan-derived
family, so tensor-product geometry sum-factorization is no longer tied to basis
reference staging.

Tasks:

1. Add explicit mesh geometry phase data to `MeshPhasePlan`, including affine
   adjugate/determinant inputs, isoparametric coordinate inputs, Jacobian scope,
   and tensor-product sum-factorization requirements.
2. Add explicit basis phase data to `LocalPhasePlan`, including field-specific
   element type, shape count, quadrature count, reference-data source, and
   tensor-product operation plan.
3. Move affine/isoparametric routing currently spread across `symbolic.py`,
   `residual_codegen.py`, and `boundary_codegen.py` into shared plan-building
   helpers.
4. Replace direct calls to local tensor-product geometry helpers from material
   emitters with backend calls that consume geometry phase plans.
5. Add tests that inspect generated plans for simplex, tensor-product, mixed
   Taylor-Hood, affine, and isoparametric cases.

Acceptance criteria:

- Emitters no longer infer geometry mode from element names or local flags.
- Tensor-product field evaluation, geometry Jacobian evaluation, and test
  contractions are selected by plan nodes.
- Boundary, energy, residual, and mixed kernels use the same geometry/basis plan
  schema.

Verification:

- `PYTHONPATH=python python -m unittest python.codegen.framework.test_gen_api`
- `PYTHONPATH=python python -m unittest python.codegen.framework.test_neohookean_ogden`
- `PYTHONPATH=python python -m unittest python.codegen.framework.test_residual`

## M3. Replace Specialized Generator Entry Points With One Backend Traversal

Goal: make OpenMP emission consume one unified kernel-plan representation.

Status: implemented. `KernelExpressionPlan` is now part of the backend plan
schema and energy, residual, boundary, and block units populate it without
material-family payloads. Energy kernel forms and diagnostic graphs are carried
by expression plans instead of `EnergyCodeGenerationPayload`.
Residual monolithic and block coefficient routing also comes from expression
plans. Boundary residual coefficient/dependency routing is also driven by the
form-1 expression plan. OpenMP emission now builds one traversal object for
local and mesh codegen setup across energy, residual, mixed residual, and
boundary kernels. Legacy low-level generator functions remain available only
from their concrete implementation modules and are no longer exported through
the high-level framework API.

Tasks:

- [x] Define a backend-level `KernelExpressionPlan` that stores local expression
   graphs, output roles, diagnostics, and required data streams for 0-, 1-, and
   2-form kernels.
- [x] Move energy-specific kernel-form data out of `EnergyCodeGenerationPayload`
   and into `KernelExpressionPlan`.
- [x] Convert residual monolithic and block coefficient data into the same
   expression-plan schema.
- [x] Convert boundary residual coefficient data into the same expression-plan
   schema.
- [x] Replace backend calls to:
   - `generate_sfem_soa_cpp_files_for_element`
   - `generate_coupled_residual_sfem_files`
   - `generate_mixed_residual_sfem_files`
   - `generate_boundary_residual_sfem_files`
   with one OpenMP backend traversal over the unified plans.
- [x] Keep old low-level generator functions internal during the migration, then
   remove or de-export them once the unified backend covers all maintained
   examples.

Acceptance criteria:

- `OpenMPSoABackend.emit(...)` has one local-kernel path and one mesh-kernel
  path for all form kinds.
- `CodeGenerationUnit` does not carry material-family payloads.
- Generated files for NeoHookean, Mooney-Rivlin, two-phase flow, Stokes,
  poro-hyperelasticity, Neumann, and Neumann-general are emitted through the
  same backend traversal.

## M4. Unify Local and Mesh Kernel Structure

Goal: make emitted C++ uniform across energy, residual, mixed, and boundary
kernels.

Status: in progress. Common local and mesh kernel signature planners now derive
template parameters and stream arguments from `KernelPlan`,
`KernelExpressionPlan`, and `ElementEmissionPlan`; the OpenMP traversal carries
these signatures for energy, residual, mixed residual, and boundary kernels.
Local signatures now carry reuse keys, and the mixed/local suffix policy is
centralized so `_mixed` is added only when the local block really depends on
mixed-order streams. Mesh operator labels are now selected through one planner,
including compatible mixed systems, single-field equations inside mixed
contexts, and explicit diagonal-block element specializations. Reference data is
now described by a shared `ReferenceDataPlan` with affine/isoparametric dataset
entries, simplex/tensor-product accessors, and mixed-order field-element
mappings derived from the same emission plan used by the backend. Diagnostics
are now described by a shared `KernelDiagnosticsPlan` built from the same
kernel expression plans, mesh/local signatures, and reference-data plan used by
backend emission, with emitter-side validation of generated diagnostic ABI
names.

Tasks:

- [x] Introduce common local-kernel signature generation from plan data:
   `scalar_t`, `N_QP`, `N_SHAPE`, and `VECTOR_SIZE` template parameters, plus
   plan-derived stream arguments.
- [x] Introduce common mesh-kernel signature generation from plan data:
   element count, node count, connectivity, geometry inputs, material
   parameters, field streams, direction streams, and output streams.
- [x] Generate local kernels by dimension and family only when their plan and
   signature are reusable; otherwise encode only the necessary block/form suffix.
- [x] Generate mesh kernels by element or compatible-element label with one naming
   convention across all form kinds.
- [x] Generate reference-data includes and accessors through one reference-data
   planner for affine/isoparametric, simplex/tensor-product, and mixed-order
   cases.
- [x] Generate diagnostics from the same kernel expression and data-stream plan
   used by the kernel body.

Acceptance criteria:

- Filename, function-name, template, and ABI conventions are identical across
  maintained material families.
- Repeated local headers are emitted once per reusable dimension/family plan.
- Diagnostics are emitted for every monolithic and block kernel from shared
  diagnostic logic.

## M5. Complete Target Platform Layer

Goal: make OpenMP and CUDA target-specific behavior pluggable while sharing plan
traversal.

Status: in progress. `TargetPlatform` now exposes target hooks for generated
function qualifiers, restrict qualifiers, parallel/vector/atomic pragmas,
alignment assumptions, math helper names, diagnostics/profiling helper names,
kernel launch style, wrapper style, and device-kernel capability. `OpenMPTarget`
and `CUDATarget` specialize those hooks while preserving the existing target API.
Target loop lowering policies now distinguish OpenMP vector-lane execution from
CUDA SIMT execution so CUDA backends do not inherit host `lane` loop assumptions.
The first CUDA backend skeleton lowers a generic `ExpressionGraph` through the
same evaluation plan used by the C++/OpenMP path into a grid-stride SIMT kernel
with a host launcher.
Current energy-SoA progress: `EnergySoAKernelEmissionPlan` construction now
lives in the planning layer, OpenMP/CUDA backends build that full plan before
emission, and the energy emitters only consume the plan to generate code.
Energy-SoA OpenMP pragmas, lane-loop lowering, parallel-loop lowering, scatter
atomics, work-item naming, and target includes are routed through
`OpenMPTarget`/`CUDATarget`. Residual, boundary, tensor-product kernel, and
tensor-product geometry generators now route OpenMP parallel/vector/atomic
pragmas, inline qualifiers, and residual vector-lane loop headers through
`OpenMPTarget` instead of embedding raw OpenMP strings. Generated OpenMP compile
coverage exists for representative emitted operators, and generated CUDA energy
operators now have an `nvcc`-gated compile test for the emitted `.cu` artifact.
OpenMP compiler-report diagnostics now cover simplex energy, tensor-product
energy, simplex residual, tensor-product residual, and mixed Taylor-Hood local
SIMD loops. Boundary residual local accumulation loops now get target-routed
OpenMP SIMD pragmas, with source-level coverage; they are not part of the
compiler-report table because the boundary emitter is still scalar element
oriented rather than vector-lane SoA oriented. Energy and residual mesh
operators route every OpenMP vector-lane loop through target SIMD lowering so
generated `lane` loops cannot silently become scalar loops.

Tasks:

- [x] Extend `TargetPlatform` with hooks for:
   - function qualifiers
   - restrict qualifiers
   - vectorization pragmas
   - alignment assumptions
   - math helper names
   - diagnostic/profiling emission
   - kernel launch/wrapper style
- [x] Move OpenMP-specific pragmas and vectorization assumptions out of emitters
   and into `OpenMPTarget`.
- [x] Implement a CUDA backend skeleton that consumes the same kernel plans and
   emits CUDA-safe local/device code for at least one simple residual kernel.
- [x] Ensure CUDA emission uses `kernel_math.hpp` helpers including specialized
   `pow_y(x)` instead of generic `pow` where possible.
- [x] Add compile tests for generated OpenMP and generated CUDA code when the CUDA
   compiler is available.
- [x] Extend vectorization diagnostics tests to all OpenMP hot-loop families:
   simplex energy, tensor-product energy, simplex residual, tensor-product
   residual, mixed Taylor-Hood, and boundary residual.

Acceptance criteria:

- Symbolic and form-manipulation layers contain no OpenMP/CUDA-specific code.
- OpenMP and CUDA backends traverse the same plan objects.
- Vectorization and compile tests fail when expected hot loops stop vectorizing.

## M6. Automate C ABI, OOP Wrapper, and Factory Integration

Goal: generate runtime integration from plan metadata without manual frontend
maintenance.

Status: implemented. Generated OpenMP `sfem::Op` wrappers now emit a structured
`op/sfem_<Op>_manifest.json` next to the wrapper header/source and C ABI header.
The manifest records wrapper paths, C ABI header path, generated include roots,
factory entry-point names, and extracted C ABI declarations for energy,
residual/block, mixed Taylor-Hood, poro-hyperelastic, and boundary wrappers.
Each wrapper also emits a generated registration source with a single
`Factory::register_op(...)` entry point named in the manifest. Manifests can now
be fed to `sfem.gen.generate_op_registration_files(...)` to emit an aggregate
factory-registration translation unit, and
`generate_op_registration_files.py` provides the same manifest-driven path for
scripts. The frontend factory now consumes the generated aggregate registration
unit, so maintained generated material wrappers no longer require
hand-maintained includes or registration calls in `sfem_OpFactory.cpp`.
Energy-only and coupled energy/residual wrappers now assemble generated kernel
calls from form dependency metadata, so unused current, previous, direction, and
parameter inputs are not forwarded through the wrapper layer. Runtime affine
and isoparametric selection is emitted from wrapper metadata and remains valid
before and after initialization. Generated boundary wrappers now consume
`NeumannConditions::Condition`, so sideset Neumann handling is integrated with
the existing SFEM condition abstraction instead of defining a separate generated
runtime condition type. `sfem_GeneratedOpWrapperCompileTest` now executes
generated energy-only, residual-only, coupled energy/residual, and boundary
residual wrappers.

Tasks:

1. [x] Generate C ABI declarations for every emitted monolithic and block kernel
   from the same plan data used by the backend.
2. [x] Generate `sfem::Op` wrappers from `CodeGenerationPlan` metadata, including
   block-system units and boundary units.
3. [x] Generate a registration manifest listing wrapper headers, wrapper sources,
   factory names, and required generated include paths.
   - The manifest now also derives runtime operation variants from generated C
     ABI declarations (`affine`, `isoparametric`, `sideset`) so wrapper dispatch
     metadata is produced at the code-generation boundary.
4. [x] Use the manifest to update or generate SFEM factory registration instead of
   manually editing frontend includes/registration calls.
5. [x] Make runtime affine/isoparametric options plan-derived for objective,
   gradient, residual, Hessian action, and Jacobian action.
   - Generated wrappers now use one emitted affine-option table per wrapper for
     `set_option` and YAML parsing instead of duplicating operation alias logic
     in each wrapper family.
   - Generated wrappers now make affine options valid both before and after
     wrapper initialization: YAML affine flags are read before `initialize()`,
     and `set_option(..., true)` refreshes cached affine geometry for already
     initialized factory-created operators.
6. [x] Align generated boundary-condition support with SFEM condition abstractions;
   keep generated Neumann sideset handling as one condition implementation, not
   a separate runtime design.
7. [x] Add runtime tests that execute generated `sfem::Op` wrappers for:
   - energy-only operator
   - residual-only operator
   - coupled energy/residual operator
   - boundary residual operator

Acceptance criteria:

- Regenerating maintained materials also regenerates wrapper/factory metadata.
- Frontend registration does not need hand-maintained generated-op includes.
- Runtime wrapper tests execute generated kernels, not only compile or factory
  create wrappers.

## M7. Remove Legacy Public APIs and Historical Paths

Goal: leave one public user-facing framework path.

Tasks:

1. Audit `codegen.framework.__all__` and `sfem.gen.__all__` for low-level
   generator functions that should no longer be public.
2. Stop exporting legacy low-level generator APIs once backend traversal covers
   maintained materials.
3. Move tests that need low-level helpers to internal test modules or update
   them to generate through `sfem.gen.CodeGenerator`.
4. Audit `python/codegen/framework/twophaseflow.py` and other historical files
   for standalone pipeline logic.
5. Remove or convert historical helpers that do not use:
   `EquationSystem` -> `FormCollection` -> `GenerationPlan` -> backend.
6. Ensure all scripts under `python/codegen/framework` and
   `python/codegen/framework/docs` call `sfem.gen.run(...)` or
   `sfem.gen.generate(...)`.

Acceptance criteria:

- User examples do not import low-level codegen internals.
- Public API documentation only shows `sfem.gen` and UFL-style symbolic
  construction.
- Historical material-specific generation paths are removed or converted.

## M8. Complete Regression Coverage

Goal: verify the unified backend with generated code and hardcoded references.

Tasks:

1. Add hardcoded Python reference action tests for Taylor-Hood generated kernels
   on:
   - `TRI6_TRI3`
   - `TET10_TET4`
   - `HEX27_HEX8`
2. Add at least one hardcoded Python reference test for a coupled
   poro-hyperelastic monolithic path.
3. Add at least one hardcoded Python reference test for a generated block kernel
   from a coupled formulation.
4. Add generated compile tests for all maintained material examples:
   - NeoHookean Ogden
   - Mooney-Rivlin
   - two-phase flow
   - Stokes
   - poro-hyperelasticity
   - Neumann
   - Neumann-general
5. Add generated wrapper compile and runtime execution tests for all maintained
   `op_name` materials.
6. Add plan-dump schema tests that verify every maintained material has
   explicit geometry, basis, data-stream, local-phase, mesh-phase, diagnostics,
   and ABI metadata.
7. Add a single bash entry point that runs Python tests, generated compile
   tests, wrapper compile tests, and optional vectorization/CUDA tests.

Acceptance criteria:

- Maintained examples regenerate and compile from a clean output directory.
- Representative generated kernels match hardcoded Python references.
- The regression script reports clearly which optional target checks were
  skipped because the compiler/toolchain was unavailable.

## Suggested Order

1. M1, because dependency metadata must be reliable before plan emission is
   fully unified.
2. M2 and M3 together, because geometry/basis plans and backend traversal are
   tightly coupled.
3. M4, to normalize generated C++ structure once backend traversal is shared.
4. M6, because wrappers should consume the stable plan/ABI metadata.
5. M7, after the replacement public path is complete.
6. M8 continuously, adding regression tests as each milestone becomes
   functional.
7. M5 can start with OpenMP target cleanup early, but CUDA completion can run in
   parallel once backend traversal is stable.
