# Goals


1. The user writes either the system energy or residual
2. The code generator detects patterns (see current content of codegen: e.g., gpu_linear_elasticity_op.py) and uses intermediate symbols to store optimal intermediate evluations
3. Evalutates the expression graph for quantities that should be evaluated or be outside loop scopes (e.g., trial loop vs test loop, quadrature loop or mesh wide loop)
4. Keeps track of the arithmetic intensity and has euristics for register pressure
5. Reference shape functions, gradients, are passed as arrays, the generate code is generic for the dimension (specialized for 1D, 2D, 3D) so that any element can be used with the same math/materials, element related sizes are passed to the kernel as template parameters 
6. For tensor product elements, specializes the codes to use sum-factorization and matrix-units (e.g., tensor-cores, matrix-cores, SME)
7. The code generator targets and specializes for
	- OpenMP
	- CUDA/Hip
	- AVX512
	- ARM SVE and SME
8. It generates
	- Matrix free kernels for
		- Hessian/Jacobian application
		- Gradient application
		- Standard mesh format (see SFEM)
		- Packed mesh format (see SFEM, in particular PackedLaplacian for nearly-optimal implementation style. Data-layouts can still be improved at set-up to fit directly the computational layout). Implement the two pass scheme (as well as the one pass) described in /Users/patrickzulian/Desktop/cloud/owncloud_USI/zulian/scientific_collaborator/papers/packedop_paper/main.tex
		- Generate variants for per thread per-warp optimized set-ups and executation
		- Patch based (overall and with index for specific nodes)
		- Element base (as it is already)
	- Matrix assembly (see how it is done now in SFEM)
		- CRS
		- BSR
		- DIA
		- COO
		- Patch-based assembly with index for specific nodes
	- Objective/Energy evaluation (see value_steps in SFEM), when available, merit function otherwise
	- All the generated kernels will have FLOP counting and arithmetic intensity functions that are used to generate performance analyses autmatically
9. Specializations for hyperelasticity (see sr_hyperleasticity.py and neohookean_partial_assembly.py)
10. Clean and usable software design
11. Generated kernels are in procedural style, with OOP wrapper (as it is now SFEM)
12. Use SoA (priority), AoS, and AoSoSoA 

It is a code-generator, the runs are done by compiling the kernels and running them within the SFEM library

# References

- SFEM for reproducing (and improving) current kernel generation outputs
- HOG: https://i10git.cs.fau.de/hyteg/hog for hybrid grids loops
- ExaStencils: https://github.com/lssfau/ExaStencils for loop manipulation

# Dependencies

- SymPy (for symbolic manipulation)
- NetworkX (for graph manipulation)

# Milestones and Tasks

This PRD incorporates the full `REFACTORING_2.md` plan and the implementation
updates completed for M8.

## Guidelines

- Every step expands the NeoHookean Ogden test as a testing ground.
- Prioritize the notation objective, gradient, and Hessian for NeoHookean Ogden.

## Principles

- After `EquationSystem.form_collection(...)`, every downstream stage consumes
  `FormCollection` and explicit plan data only.
- Code generation is driven by `GenerationPlan`, `KernelPlan`, `BlockPlan`,
  geometry plans, basis plans, and data-stream plans.
- Target-specific choices belong to target/backend classes.
- Generated hot loops remain SoA, branch-light, and vectorization-friendly.
- Removed APIs do not receive retro-compatibility shims.

## Architecture

The framework is layered:

- Symbolic layer: no string generation.
- Form unification layer: form metadata and dependency-pruned form collections.
- Laning layer: vector-lane, thread, warp, and mesh execution structure.
- Code generation and emission layer: emitters consume complete plans and
  produce procedural kernels plus SFEM-style wrappers.

## Backend Product Requirements

The refactoring plan does not replace the original CPU and GPU backend
requirements; it provides the plan-driven implementation structure for them.

CPU backends:

- Generate procedural OpenMP kernels with OOP wrappers matching SFEM style.
- Generate AVX512-oriented kernels with unit-stride memory access, branch-free
  hot loops, and compiler-friendly temporaries.
- Generate ARM SVE and SME variants with vector-length-aware loops and
  matrix-unit paths where applicable.
- Provide matrix-free kernels for Hessian/Jacobian application and gradient
  application on standard and packed mesh formats.
- Generate element, patch, per-thread, and per-warp variants where the execution
  model benefits from them.

GPU backends:

- Generate CUDA and HIP kernels for standard mesh and packed mesh formats.
- Generate per-thread and per-warp matrix-free variants for Hessian/Jacobian
  application, gradient application, and patch kernels.
- Specialize tensor-product elements with sum-factorization.
- Use tensor cores, matrix cores, or equivalent matrix units for tensor-product
  microkernels when precision and shape permit.
- Keep data movement explicit and expose arithmetic intensity for roofline
  analysis.

Shared generated-kernel products:

- Matrix-free Hessian/Jacobian application and gradient application.
- Matrix assembly for CRS, BSR, DIA, COO, and patch-based formats.
- Objective/energy evaluation when energy is available; merit functions
  otherwise.
- FLOP counting and arithmetic-intensity functions for every generated kernel.
- Hyperelasticity support for residual, Jacobian-action, energy, and objective
  paths, including NeoHookean Ogden and related existing generators.

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

- `PYTHONPATH=python venv/bin/python -m unittest python.codegen.framework.tests.test_gen_api`

## M2. Move Geometry and Basis Policy Fully Into Plans

Goal: remove geometry and basis-policy reconstruction from emitters.

Status: implemented. Context-specialized emission plans carry affine and
isoparametric geometry phase data, geometry streams, field-specific basis plans,
reference-data streams, and tensor-product sum-factorization plans. OpenMP
emission validates and consumes geometry modes through a shared
`ElementEmissionPlan`. Energy, coupled residual, mixed residual, and boundary
residual entry points require this shared plan instead of reconstructing element
family from element names or quadrature flags. Shared
`emission_plan_for_element(...)` construction is used for field-element plans,
including synthetic diagonal mixed-order residual blocks. Energy and mixed
residual local/mesh/diagnostic reference routing uses the basis-plan-derived
family. Isoparametric Jacobian generation is routed through geometry-plan data,
so tensor-product geometry sum-factorization is no longer tied to basis
reference staging.

Tasks:

1. Add explicit mesh geometry phase data to `MeshPhasePlan`, including affine
   adjugate/determinant inputs, isoparametric coordinate inputs, Jacobian scope,
   and tensor-product sum-factorization requirements.
2. Add explicit basis phase data to `LocalPhasePlan`, including field-specific
   element type, shape count, quadrature count, reference-data source, and
   tensor-product operation plan.
3. Move affine/isoparametric routing from `symbolic.py`, `residual_codegen.py`,
   and `boundary_codegen.py` into shared plan-building helpers.
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

- `PYTHONPATH=python venv/bin/python -m unittest python.codegen.framework.tests.test_gen_api`
- `PYTHONPATH=python venv/bin/python -m unittest python.codegen.framework.tests.test_neohookean_ogden`
- `PYTHONPATH=python venv/bin/python -m unittest python.codegen.framework.tests.test_residual`

## M3. Replace Specialized Generator Entry Points With One Backend Traversal

Goal: make OpenMP emission consume one unified kernel-plan representation.

Status: implemented. `KernelExpressionPlan` is part of the backend plan schema,
and energy, residual, boundary, and block units populate it without
material-family payloads. Energy kernel forms and diagnostic graphs are carried
by expression plans instead of `EnergyCodeGenerationPayload`. Residual
monolithic and block coefficient routing comes from expression plans. Boundary
residual coefficient/dependency routing is driven by the form-1 expression plan.
OpenMP emission builds one traversal object for local and mesh codegen setup
across energy, residual, mixed residual, and boundary kernels. Legacy low-level
generator functions remain available only from concrete implementation modules
and are no longer exported through the high-level framework API.

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
- [x] Replace backend calls to `generate_sfem_soa_cpp_files_for_element`,
  `generate_coupled_residual_sfem_files`,
  `generate_mixed_residual_sfem_files`, and
  `generate_boundary_residual_sfem_files` with one OpenMP backend traversal.
- [x] Keep old low-level generator functions internal during migration, then
  remove or de-export them once the unified backend covers all maintained
  examples.

Acceptance criteria:

- `OpenMPSoABackend.emit(...)` has one local-kernel path and one mesh-kernel path
  for all form kinds.
- `CodeGenerationUnit` does not carry material-family payloads.
- Generated files for NeoHookean, Mooney-Rivlin, two-phase flow, Stokes,
  poro-hyperelasticity, Neumann, and Neumann-general are emitted through the
  same backend traversal.

## M4. Unify Local and Mesh Kernel Structure

Goal: make emitted C++ uniform across energy, residual, mixed, and boundary
kernels.

Status: in progress. Common local and mesh kernel signature planners derive
template parameters and stream arguments from `KernelPlan`,
`KernelExpressionPlan`, and `ElementEmissionPlan`; the OpenMP traversal carries
these signatures for energy, residual, mixed residual, and boundary kernels.
Local signatures carry reuse keys, and the mixed/local suffix policy is
centralized so `_mixed` is added only when the local block depends on
mixed-order streams. Mesh operator labels are selected through one planner,
including compatible mixed systems, single-field equations inside mixed
contexts, and explicit diagonal-block element specializations. Reference data is
described by a shared `ReferenceDataPlan` with affine/isoparametric dataset
entries, simplex/tensor-product accessors, and mixed-order field-element
mappings derived from the same emission plan used by the backend. Diagnostics
are described by a shared `KernelDiagnosticsPlan` built from the same kernel
expression plans, mesh/local signatures, and reference-data plan used by backend
emission, with emitter-side validation of generated diagnostic ABI names.

Tasks:

- [x] Introduce common local-kernel signature generation from plan data:
  `scalar_t`, `N_QP`, `N_SHAPE`, and `VECTOR_SIZE` template parameters, plus
  plan-derived stream arguments.
- [x] Introduce common mesh-kernel signature generation from plan data: element
  count, node count, connectivity, geometry inputs, material parameters, field
  streams, direction streams, and output streams.
- [x] Generate local kernels by dimension and family only when their plan and
  signature are reusable; otherwise encode only the necessary block/form suffix.
- [x] Generate mesh kernels by element or compatible-element label with one
  naming convention across all form kinds.
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

Goal: integrate the CPU and GPU backend requirements above behind a shared
target-platform layer, so OpenMP, AVX512, ARM SVE/SME, CUDA, HIP, and
matrix-unit-specific behavior is pluggable while sharing plan traversal.

Status: implemented. `TargetPlatform` exposes target hooks for generated
function qualifiers, restrict qualifiers, parallel/vector/atomic pragmas,
alignment assumptions, math helper names, diagnostics/profiling helper names,
kernel launch style, wrapper style, and device-kernel capability. `OpenMPTarget`
and `CUDATarget` specialize those hooks while preserving the existing target API.
Target loop lowering policies distinguish OpenMP vector-lane execution from CUDA
SIMT execution so CUDA backends do not inherit host `lane` loop assumptions. The
first CUDA backend skeleton lowers a generic `ExpressionGraph` through the same
evaluation plan used by the C++/OpenMP path into a grid-stride SIMT kernel with
a host launcher. Energy-SoA planning now lives in the planning layer, and
OpenMP/CUDA backends build that full plan before emission. Residual, boundary,
tensor-product kernel, and tensor-product geometry generators route OpenMP
parallel/vector/atomic pragmas, inline qualifiers, and residual vector-lane loop
headers through `OpenMPTarget`. Generated OpenMP compile coverage exists for
representative emitted operators. Generated CUDA energy operators have an
`nvcc`-gated compile test. OpenMP compiler-report diagnostics cover simplex
energy, tensor-product energy, simplex residual, tensor-product residual, and
mixed Taylor-Hood local SIMD loops. Boundary residual local accumulation loops
use target-routed OpenMP SIMD pragmas. Energy and residual mesh operators route
every OpenMP vector-lane loop through target SIMD lowering.
AVX512, ARM SVE, ARM SME, and HIP are first-class target policies behind the
same backend registry. AVX512 records 512-bit unit-stride vector lowering and
compiler diagnostics, SVE/SME record vector-length-aware lowering with SME
matrix-unit capability, CUDA/HIP record SIMT matrix-unit and per-thread/per-warp
variant policies, and packed-mesh one-pass/two-pass support is represented by
target policy instead of emitter-side branching. HIP energy emission reuses the
GPU plan traversal and emits HIP-flavored SIMT operator sources.

Tasks:

- [x] Extend `TargetPlatform` with hooks for function qualifiers, restrict
  qualifiers, vectorization pragmas, alignment assumptions, math helper names,
  diagnostic/profiling emission, and kernel launch/wrapper style.
- [x] Move OpenMP-specific pragmas and vectorization assumptions out of emitters
  and into `OpenMPTarget`.
- [x] Implement a CUDA backend skeleton that consumes the same kernel plans and
  emits CUDA-safe local/device code for at least one simple residual kernel.
- [x] Ensure CUDA emission uses `kernel_math.hpp` helpers including specialized
  `pow_y(x)` instead of generic `pow` where possible.
- [x] Add compile tests for generated OpenMP and generated CUDA code when the
  CUDA compiler is available.
- [x] Extend vectorization diagnostics tests to all OpenMP hot-loop families:
  simplex energy, tensor-product energy, simplex residual, tensor-product
  residual, mixed Taylor-Hood, and boundary residual.
- [x] Complete AVX512-specific lowering and diagnostics for unit-stride,
  branch-light hot loops.
- [x] Complete ARM SVE/SME lowering, including vector-length-aware and
  matrix-unit tensor-product paths.
- [x] Extend CUDA/HIP coverage from the skeleton to maintained matrix-free,
  assembly, patch, and tensor-product kernels.
- [x] Preserve backend-specific packed-mesh and per-thread/per-warp variants
  through target policy, not emitter-side branches.

Acceptance criteria:

- Symbolic and form-manipulation layers contain no OpenMP/CUDA-specific code.
- CPU and GPU backends traverse the same plan objects.
- Vectorization and compile tests fail when expected hot loops stop vectorizing.
- Backend-specific kernels satisfy the product backend requirements without
  duplicating symbolic or form-lowering logic.

## M6. Automate C ABI, OOP Wrapper, and Factory Integration

Goal: generate runtime integration from plan metadata without manual frontend
maintenance.

Status: implemented. Generated OpenMP `sfem::Op` wrappers emit a structured
`op/sfem_<Op>_manifest.json` next to the wrapper header/source and C ABI header.
The manifest records wrapper paths, C ABI header path, generated include roots,
factory entry-point names, and extracted C ABI declarations for energy,
residual/block, mixed Taylor-Hood, poro-hyperelastic, and boundary wrappers.
Each wrapper emits a generated registration source with a single
`Factory::register_op(...)` entry point named in the manifest. Manifests can be
fed to `sfem.gen.generate_op_registration_files(...)` to emit an aggregate
factory-registration translation unit, and
`codegen.framework.generators.op_registration` provides the same
manifest-driven path for scripts. The frontend factory consumes the generated
aggregate registration unit, so maintained generated material wrappers no longer
require hand-maintained includes or registration calls in `sfem_OpFactory.cpp`.
Aggregate registration validates the generated-op manifest schema, wrapper
paths, registration/factory metadata, C ABI declarations, runtime-operation
links, include paths, and duplicate operator names before emitting factory
registration files.
Energy-only and coupled energy/residual wrappers assemble generated kernel calls
from form dependency metadata, so unused current, previous, direction, and
parameter inputs are not forwarded through the wrapper layer. Runtime affine and
isoparametric selection is emitted from wrapper metadata and remains valid before
and after initialization. Generated boundary wrappers consume
`NeumannConditions::Condition`, so sideset Neumann handling is integrated with
the existing SFEM condition abstraction. `sfem_GeneratedOpWrapperCompileTest`
executes generated energy-only, residual-only, coupled energy/residual, and
boundary residual wrappers.

Tasks:

1. [x] Generate C ABI declarations for every emitted monolithic and block kernel
   from the same plan data used by the backend.
2. [x] Generate `sfem::Op` wrappers from `CodeGenerationPlan` metadata,
   including block-system units and boundary units.
3. [x] Generate a registration manifest listing wrapper headers, wrapper
   sources, factory names, and required generated include paths.
4. [x] Use the manifest to update or generate SFEM factory registration instead
   of manually editing frontend includes/registration calls.
5. [x] Make runtime affine/isoparametric options plan-derived for objective,
   gradient, residual, Hessian action, and Jacobian action.
6. [x] Align generated boundary-condition support with SFEM condition
   abstractions; keep generated Neumann sideset handling as one condition
   implementation, not a separate runtime design.
7. [x] Add runtime tests that execute generated `sfem::Op` wrappers for
   energy-only, residual-only, coupled energy/residual, and boundary residual
   operators.

Acceptance criteria:

- Regenerating maintained materials also regenerates wrapper/factory metadata.
- Frontend registration does not need hand-maintained generated-op includes.
- Runtime wrapper tests execute generated kernels, not only compile or factory
  create wrappers.

## M7. Remove Legacy Public APIs and Historical Paths

Goal: leave one public user-facing framework path.

Status: implemented. The public `__all__` surfaces and package-root exports for
`codegen.framework` and `sfem.gen` no longer expose backend emitters, backend
classes, old direct kernel string generators, generation-plan internals,
residual helper generator APIs, or historical two-phase implicit-Euler helper
classes. Backend-focused tests import signature, diagnostics, reference-data,
and mesh-plan helpers from internal modules instead of treating them as
`sfem.gen` API. Public examples and docs show `sfem.gen.CodeGenerator` plus
`gen.generate(...)` or `gen.run(...)`. The last material-agnostic direct
residual string-generation path, `CoupledResidualSystem.generate_cpp_kernels(...)`,
has been removed; its coverage now generates and compiles through
`EquationSystemBuilder` and the unified backend.

Tasks:

1. [x] Audit `codegen.framework.__all__` and `sfem.gen.__all__` for low-level
   generator functions that should no longer be public.
2. [x] Stop exporting legacy low-level generator APIs once backend traversal
   covers maintained materials.
3. [x] Move tests that need low-level helpers to internal test modules or update
   them to generate through `sfem.gen.CodeGenerator`.
4. [x] Audit `python/codegen/framework/materials/two_phase_flow_model.py` and
   other historical files for standalone pipeline logic.
5. [x] Remove or convert historical helpers that do not use
   `EquationSystem` -> `FormCollection` -> `GenerationPlan` -> backend.
6. [x] Ensure all scripts under `python/codegen/framework` and
   `python/codegen/framework/docs` call `sfem.gen.run(...)` or
   `sfem.gen.generate(...)`.

Acceptance criteria:

- User examples do not import low-level codegen internals.
- Public API documentation only shows `sfem.gen` and UFL-style symbolic
  construction.
- Historical material-specific generation paths are removed or converted.

## M8. Form Transformations, Specializations, and Stokes Verification

Status: implemented for the current milestone scope.

### Objective

Make the framework capable of generating lean specialized kernels for simple
affine cases while preserving the layered symbolic-plan-emitter architecture,
then validate generated Taylor-Hood Stokes operators with reproducible
manufactured-solution studies.

### Requirements

- Specialize simplex affine kernels so known-zero reference-gradient entries do
  not survive into trial-gradient or contraction hot loops.
- Add a generated Laplace material path that can reproduce SFEM-style affine
  metric grouping, including FFF/AoS metric precomputation for `TRI3` and
  `TET4`.
- Keep affine simplification model-driven so the same transformation path can be
  reused by Laplace, linear elasticity, and later materials instead of adding
  material-specific string templates.
- Preserve strict layering:
  - symbolic/material definitions describe forms;
  - transformation plans describe affine, metric, and tensor-product structure;
  - emitters consume plans without rediscovering mathematical policy.
- Provide tensor-product and sum-factorization model abstractions suitable for
  lowering to CPU, GPU, Metal, and MLIR inspection artifacts.
- Add `drivers/verification` as a reusable verification workflow for generated
  steady Stokes operators from HAL `cea-02434556`, ignoring unsteady and
  Navier-Stokes cases for this milestone.
- Provide extraction, convergence, and plotting scripts that compare numerical
  output against analytical manufactured solutions.
- Use SFEM dedicated output functionality for generated verification fields
  instead of local raw-file writers.

### Delivered Scope

- Added generated Laplace coverage for affine simplex specializations, including
  metric-specialized `TRI3`/`TET4` kernels and wrapper routing through
  `smesh::FFF`.
- Added tensor-product Laplace IR and inspection artifacts for linalg, vector,
  matrix-unit, GPU, Metal, and batched element-by-element lowering paths.
- Added Stokes manufactured cases for:
  - Bercovier-Engelman 2D steady Stokes;
  - Taylor-Green 3D steady Stokes.
- Added `generated_stokes_fvca8.cpp`, a serial generated-`GeneratedStokes`
  verification driver that builds Taylor-Hood systems, projects forcing,
  applies exact Dirichlet velocity constraints, constrains pressure-only DOFs,
  computes errors, and writes `summary.csv`.
- The generated Stokes driver writes nodal fields through `smesh::Output`, using
  typed SFEM files such as `x.float32`, `y.float32`, `u0.float64`,
  `u1.float64`, `u2.float64`, and `p.float64`.
- Updated verification readers to prefer typed SFEM output while retaining
  fallback support for legacy `.raw` data.
- Added regression coverage that typed SFEM output wins over stale `.raw`
  fallback files in Stokes verification extraction and error collection.
- Updated tensor-product artifact evidence so Metal toolchain probing compiles
  IREE-emitted Metal executable sources recorded in the manifest when those
  sources are available.
- Added reusable Python tooling:
  - `run_generated_stokes_fvca8.py` for multi-level generated-driver runs;
  - `run_stokes_convergence.py` for error/rate tables;
  - `extract_stokes_fields.py` for numerical/exact/error field export;
  - `plot_convergence.py` and `plot_stokes_fields.py` for figures;
  - `stokes_mms.py` for analytical cases.
- Added unit coverage for paper-case availability, exact-field zero-error
  collectors, typed SFEM output readers, and Taylor-Green forcing semantics.

### Acceptance Criteria

- Specialized affine kernels remove avoidable zero-gradient work in hot loops.
- Generated Laplace affine kernels use precomputed metric data where the plan
  marks it profitable and compile through the normal generated wrapper path.
- Tensor-product lowering emits inspectable artifacts without leaking target
  policy into symbolic/material definitions.
- Generated Stokes verification runs produce convergence data for 2D and 3D
  manufactured cases.
- Verification scripts consume SFEM typed output by default and remain
  compatible with archived `.raw` runs.
- The generated Stokes driver builds as `generated_stokes_fvca8` and writes all
  solution and coordinate fields through SFEM `Output` APIs.

### Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:. venv/bin/python -m unittest \
  python.codegen.framework.tests.test_gen_api \
  python.codegen.framework.mlir.test_ebe \
  drivers.verification.test_stokes_verification

cmake --build build64 --target generated_stokes_fvca8 -j 4

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python venv/bin/python \
  drivers/verification/run_generated_stokes_fvca8.py \
  --exe build64/generated_stokes_fvca8 \
  --case bercovier_engelman_2d \
  --resolution 2 \
  --resolution 4 \
  --out-root /private/tmp/sfem_fvca8_output_api \
  --csv /private/tmp/sfem_fvca8_output_api.csv
```

## M9. Complete Regression Coverage

Goal: verify the unified backend with generated code and hardcoded references.

Status: implemented for the current maintained-material scope.

Tasks:

1. [x] Add hardcoded Python reference action tests for Taylor-Hood generated kernels
   on `TRI6_TRI3`, `TET10_TET4`, and `HEX27_HEX8`.
2. [x] Add at least one hardcoded Python reference test for a coupled
   poro-hyperelastic monolithic path.
3. [x] Add at least one hardcoded Python reference test for a generated block kernel
   from a coupled formulation.
4. [x] Add generated compile tests for all maintained material examples:
   NeoHookean Ogden, Mooney-Rivlin, two-phase flow, Stokes,
   poro-hyperelasticity, Neumann, and Neumann-general.
5. [x] Add generated wrapper compile and runtime dispatch tests for all maintained
   `op_name` materials.
6. [x] Add plan-dump schema tests that verify every maintained material has explicit
   geometry, basis, data-stream, local-phase, mesh-phase, diagnostics, and ABI
   metadata.
7. [x] Add a single bash entry point that runs Python tests, generated compile
   tests, wrapper compile tests, and optional vectorization/CUDA tests.

Delivered scope:

- Added `python/codegen/framework/tests/test_m9_regression.py`, covering:
  - symbolic reference checks for Taylor-Hood Stokes residual/action
    coefficients on `TRI6_TRI3`, `TET10_TET4`, and `HEX27_HEX8`;
  - coupled poro-hyperelastic monolithic residual/action coefficient checks;
  - generated coupled block-kernel coefficient checks;
  - clean-output regeneration and generated operator compile coverage for all
    maintained material examples;
  - generated Op manifest, C ABI, factory, registration, include-path, runtime
    operation, and wrapper dispatch metadata;
  - generated wrapper syntax compilation when frontend optional dependencies
    such as `ryml.hpp` and `c4core` headers are available in the local build
    tree;
  - plan-dump schema checks for geometry, basis, stream, local-phase,
    mesh-phase, dependency/diagnostic, and ABI-facing metadata.
- Added `python/codegen/framework/run_m9_regression.sh` as the single required
  entry point for Python, generated compile, wrapper compile, and optional
  CUDA/vectorization checks.

Acceptance criteria:

- Maintained examples regenerate and compile from a clean output directory.
- Representative generated kernels match hardcoded Python references.
- The regression script reports clearly which optional target checks were
  skipped because the compiler/toolchain was unavailable.

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:. venv/bin/python -m unittest \
  python.codegen.framework.tests.test_m9_regression

bash python/codegen/framework/run_m9_regression.sh
```

## M10. NASA Wall-Mounted Hump Application

Goal: use the generated-operator framework to simulate the NASA Wall-Mounted
Hump as an end-to-end incompressible-flow application.

Status: implemented for the baseline generated-operator hump workflow.

Scope:

- Implemented `codegen.framework.materials.navier_stokes` as a transient
  incompressible Taylor-Hood material with residual and Jacobian-action forms,
  previous-velocity state, viscosity/density/timestep parameters, convection
  scaling, and body-force parameters.
- Checked in generated `GeneratedNavierStokes` OpenMP/SoA kernels and wrapper
  for `TRI6_TRI3`, `TET10_TET4`, and `HEX27_HEX8`, and registered the generated
  op in the normal SFEM generated-op registry.
- Added `Mesh::create_wall_mounted_hump` in SMESH by warping cube and
  semistructured cube meshes, preserving support for high-order
  `PROTEUS_HEX*` geometry generation/export.
- Added `drivers/simulations/wall_mounted_hump.cpp` to build a baseline hump
  mesh, mark inlet/outlet/wall/span nodes, initialize velocity/pressure fields,
  create `GeneratedNavierStokes` through the SFEM op factory, attach it to an
  SFEM `Function` with `DirichletConditions`, solve each time step with SFEM's
  constrained residual, matrix-free linear-operator, and BiCGStab APIs, write
  AoS time-step states through `sfem::Output`, write restartable split fields
  through `smesh::Output`, and emit residual/correction diagnostics in the
  solve-stage schedule. The solver driver supports `SFEM_ELEM_TYPE=HEX27` and
  accepts `PROTEUS_HEX27` by reordering the level-2 semistructured connectivity
  to the standard `HEX27` convention used by the checked-in `HEX27_HEX8`
  Taylor-Hood kernels.
- Added `drivers/simulations/run_wall_mounted_hump.sh`,
  `drivers/simulations/postprocess_wall_mounted_hump.py`, and
  `drivers/simulations/wall_mounted_hump.md` with generation, build, run,
  restart-field, validation-data, and post-processing notes.
- Fixed the generated residual-only wrapper parameter mapping so Jacobian action
  calls use only action dependencies, while parameter storage remains large
  enough for all material defaults.

Acceptance criteria:

- The mesh generator produces the NASA hump domain and boundary markers needed
  by the solver.
- The generated Navier-Stokes operator compiles through the normal SFEM build.
- The executable can run documented `HEX27` and `PROTEUS_HEX27` baseline hump
  cases, reject unsupported solver element types with a clear diagnostic, and write
  restartable fields through SFEM output APIs.
- Validation data, run scripts, and post-processing are documented next to the
  driver.

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:. venv/bin/python \
  -m codegen.framework.materials.navier_stokes \
  --out-dir /private/tmp/sfem_m10_navier_gen \
  --element TRI6_TRI3 --compile --dump-plan

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:. venv/bin/python \
  -m unittest python.codegen.framework.tests.test_m10_navier_stokes

cmake --build build64 --target wall_mounted_hump -j 4

./build64/wall_mounted_hump /private/tmp/sfem_wall_hump_m10

PYTHONPATH=python venv/bin/python \
  drivers/simulations/postprocess_wall_mounted_hump.py \
  /private/tmp/sfem_wall_hump_m10 \
  --csv /private/tmp/sfem_wall_hump_m10/summary.csv
```

Smoke post-processing summary:

```csv
n_nodes,inlet_nodes,outlet_nodes,wall_nodes,span_nodes,u_min,u_max,p_min,p_max,has_solve_stages
2145,65,65,310,682,0.0,1.0103999206328473,-2.895625728792179,4.688691952033825,True
```

## M11. Matrix Formats and Assembly Backends

Goal: keep matrix-format generation as a first-class milestone, covering both
assembled operators and format-aware matrix-free paths.

Status: planned.

Scope:

- Generate matrix assembly for CRS, BSR, DIA, and COO formats.
- Generate patch-based assembly, including optional node-index filtering.
- Generate format-specific operator application paths when they are more
  efficient than a generic assembled apply.
- Support standard SFEM mesh layout and packed mesh layout, including one-pass
  and two-pass packed schemes.
- Preserve SoA-first data movement and avoid STL containers in generated hot
  paths.
- Expose format-specific FLOP, byte, and arithmetic-intensity diagnostics.
- Integrate matrix-format selection with generated `sfem::Op` wrappers and
  factory metadata.

Acceptance criteria:

- Maintained materials can emit CRS, BSR, DIA, COO, and patch assembly variants
  from the same form/plan data.
- Generated matrix-format kernels compile from a clean output directory.
- Reference tests compare assembled action against matrix-free action for
  representative simplex, tensor-product, mixed Taylor-Hood, and coupled block
  cases.
- Packed-format kernels document and report their expected memory traffic and
  arithmetic intensity.

## Suggested Order

1. M1, because dependency metadata must be reliable before plan emission is
   fully unified.
2. M2 and M3 together, because geometry/basis plans and backend traversal are
   tightly coupled.
3. M4, to normalize generated C++ structure once backend traversal is shared.
4. M6, because wrappers should consume the stable plan/ABI metadata.
5. M7, after the replacement public path is complete.
6. M8, to replicate or improve special-case performance and add generated
   Stokes verification.
7. M9 continuously, adding regression tests as each milestone becomes
   functional.
8. M5 can start with OpenMP target cleanup early, while CUDA completion runs in
   parallel once backend traversal is stable.
9. M10 after the generated Stokes path is stable enough to support the
   wall-mounted-hump Navier-Stokes application.
10. M11 continuously with M4, M5, and M9, because matrix formats affect kernel
    structure, target lowering, and regression coverage.
