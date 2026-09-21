# Kernel Plans

This document summarizes the code-generation plans used by the framework and
the matrix-assembly plans required by M11.

## Pipeline

1. `UserInputStage`: captures material, elements, vector size, target, and
   optional matrix-format requests.
2. `FormCollection`: stores dependency-pruned 0-, 1-, and 2-forms produced from
   the symbolic equation system.
3. `GenerationPlan`: owns the kernel units selected for emission.
4. `KernelPlan`: describes one monolithic or block kernel for one form family.
5. `ElementEmissionPlan`: describes element geometry, basis, reference data,
   and tensor-product policy for one specialized element context.
6. Backend traversal: lowers plans to procedural local kernels, mesh kernels,
   wrappers, diagnostics, and optional matrix-format artifacts.

## Existing Kernel Plans

`GenerationPlan`

- Top-level collection of `KernelPlan` objects.
- Selects kernels matching a specialized element context.

`KernelPlan`

- Describes one generated kernel unit.
- Carries name, form kind, dimension, scope, coupling, target, streams, blocks,
  mesh phases, expression plans, and optional matrix-format plan.
- Supports monolithic kernels and block kernels.

`KernelExpressionPlan`

- Describes the local symbolic computation.
- Carries form order, output role, expression graph, coefficients,
  dependencies, required streams, fields, blocks, and diagnostics.
- Must remain independent of emitted C/C++ syntax.

`LocalKernelPlan`

- Names the reusable element-local header.
- Encodes prefix, dimension, element family, and optional suffix.
- Produces names such as `<prefix>_d3_simplex_local.hpp`.

`MeshKernelPlan`

- Names the mesh-level operator source.
- Encodes prefix and element label.
- Produces names such as `<prefix>_<element>_operator.cpp`.

`BlockPlan`

- Describes a selected row/column field pair for mixed systems.
- Carries local phase plans, streams, basis plans, and block form order.
- Preserves asymmetric blocks such as pressure-velocity and velocity-pressure.

`LocalPhasePlan`

- Describes the phases inside one element-local computation:
  `evaluate_trial`, `transform_reference`, `evaluate_material`,
  `contract_test`.
- Carries phase-local streams, basis plans, and transformations.

`MeshPhasePlan`

- Describes mesh traversal phases: `gather`, `geometry`, `local_call`,
  `scatter`.
- Carries streams, geometry plans, and block plans used by that phase.

`GeometryPlan`

- Describes affine or isoparametric geometry handling.
- Carries geometry inputs, Jacobian/adjugate/determinant policy, and
  tensor-product geometry requirements.

`DataStreamPlan`

- Describes a typed input/output stream.
- Carries role, layout, scalar type, component count, item count, and source.

`MatrixFormatPlan`

- Describes requested matrix-format variants attached to a `KernelPlan`.
- Contains duplicate-free `MatrixAssemblyVariantPlan` entries.

`MatrixAssemblyVariantPlan`

- Describes one requested matrix layout variant.
- Carries format, mesh layout, packed pass, patch index flag, format-aware
  apply flag, element DOF counts, entry counts, bytes, FLOPs, and arithmetic
  intensity metadata.

## M11 Matrix Assembly Plans

`ElementAssemblyKernelPlan`

- New local plan for producing one element matrix.
- Consumes a 2-form `KernelExpressionPlan`, `GeometryPlan`, basis plans,
  coefficient streams, and element DOF layout.
- Produces a dense or structured element-local matrix buffer.

`ElementDofPlan`

- Defines row fields, column fields, components, row DOFs, column DOFs, and
  local entry count.
- Must preserve mixed block asymmetry.

`ElementLocalMatrixPlan`

- Defines the local matrix output layout.
- Examples: dense row-major, component-blocked, diagonal-only, patch-local.
- This plan is format-neutral; global CRS/BSR/DIA/COO scatter is separate.

`ElementLoopPlan`

- Defines fixed local loops over quadrature, test shape, trial shape,
  components, and vector lanes.
- Must expose bounds as constants or template parameters for unrolling and
  vectorization.

`ElementTemporaryPlan`

- Defines hoisted intermediate values and scratch storage.
- Tracks reuse, lifetime, register-pressure estimates, and alignment needs.

`CRSAssemblyPlan`

- Defines row pointer, column index, value stream, element connectivity,
  accumulation policy, and optional packed traversal.
- Consumes the element-local matrix and scatters into CRS storage.

`BSRAssemblyPlan`

- Defines block size, block row/column graph, component ordering, and block
  value layout.
- Should use unit-stride writes inside block values.

`DIAAssemblyPlan`

- Defines diagonal offsets, stride, value layout, and compatibility checks.
- Valid only when the element/block structure has compact diagonal support.

`COOAssemblyPlan`

- Defines triplet row, column, and value streams.
- Emits deterministic duplicate entries; sorting/reduction belongs outside the
  generated hot element loop.

`PatchAssemblyPlan`

- Defines patch-local graph, patch value layout, and optional node-index
  filtering.
- Filtering should be hoisted outside innermost arithmetic loops when possible.

`BlockDiagSymAssemblyPlan`

- Defines compressed symmetric block-diagonal values.
- Uses node-major AoS storage with `DIM * (DIM + 1) / 2` entries per node.
- Consumes only diagonal node blocks from the element-local matrix.

`PackedAssemblyPlan`

- Defines standard versus packed mesh traversal.
- For packed layout, selects one-pass or two-pass behavior.
- One-pass assumes graph/value layout is ready before the kernel.
- Two-pass separates graph discovery from value fill.

`AssemblyDiagnosticsPlan`

- Defines per-element FLOPs, bytes, index reads, value writes, and arithmetic
  intensity.
- Must be emitted next to each generated assembly variant.

## Separation of Concerns

- Element-level plans compute local mathematics.
- Matrix-format plans define global storage and scatter.
- Mesh-layout plans define standard or packed traversal.
- Target plans define OpenMP, AVX512, SVE/SME, CUDA, or HIP lowering.
- Wrappers expose generated variants to runtime.
- Symbolic materials must not contain matrix-format, target, or scatter policy.

## Matrix Assembly Flow

```text
FormCollection 2-form
  -> KernelExpressionPlan
  -> ElementAssemblyKernelPlan
  -> ElementLocalMatrixPlan
  -> CRS/BSR/DIA/COO/Patch/BlockDiagSymAssemblyPlan
  -> target lowering
  -> emitted procedural kernel
  -> wrapper/factory runtime entry point
```

## Hot-Loop Requirements

- Prefer SoA streams and unit-stride accesses.
- Keep loop bounds explicit and specialization-friendly.
- Avoid branches inside quadrature, test, trial, and component loops.
- Avoid STL containers in emitted hot paths.
- Hoist lookup, filtering, and format decisions outside arithmetic loops.
- Keep generated code suitable for compiler vectorization on AVX512-class CPUs.



## Cleaner Generated Code

- Make sure to not duplicate reusable primitves (and files) that are shared between multiple kernels. on the top staging of "generated" kernels place headers with common functionalities (e.g., for computing jacobians, ...). The actual generated kernel files (and local algorithms) should include such headers and call the functions directly instead of regenerating every thime

- HEX8 kernels should just wrap PROTEUS_HEX8 versions after shufling correctly the array pointers
