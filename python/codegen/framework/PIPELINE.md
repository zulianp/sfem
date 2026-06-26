# Code Generation Pipeline

This file describes the current code-generator path for the reference
materials:

- NeoHookean Ogden hyperelasticity
- Two-phase flow residual formulation
- Poro-hyperelasticity mixed formulation

The core design requirement is that energy and residual models are not two
separate code-generation pipelines. They differ in user input and in the
specialized symbolic manipulation needed to turn equations into kernel-ready
data, but all material inputs are first normalized into an `EquationSystem`.
After that, the code-generation stage is one shared process.

This matters for mixed models such as thermo-hyperelasticity, where the same
material may contain energy-style and residual-style pieces.

## Stage Overview

### 1. User Input

This is the material-facing API. It records the mathematical model and
metadata, but it does not generate code.

Entry points:

- `sfem.gen.CodeGenerator`
- `sfem.gen.EquationSystem`
- `sfem.gen.generate(...)`
- `sfem.gen.run(...)`

`sfem.gen.generate(...)` normalizes this into `UserInputStage`:

- selected element types
- vector size
- quadrature order
- one `ElementGenerationContext` per element
- generated prefixes and local prefixes
- FEM specialization from `codegen.framework.fem`

### 2. Form Evaluation

This stage evaluates 0-, 1-, and 2-forms from the equations in the user input.

Before evaluation, every `CodeGenerator` already owns explicit
`EquationSystem` objects:

- `EquationField` records scalar/vector field metadata and field family.
- `Equation` records whether the equation is energy-form or residual-form.
- `EquationSystem` owns any number of fields and equations.
`CodeGenerator` receives an `EquationSystems` collection. User code creates
one concrete `EquationSystem` per required spatial dimension and passes the
collection to the material. Energy equations must pass explicit differentiation
variables with `variables=...`; `fields=...` only describes the assembled
unknowns.

Element compatibility is handled before form evaluation. Single-field
formulations use ordinary SFEM element names. Mixed formulations may use
compatible element descriptors such as Taylor-Hood pairs, where different field
families use different interpolation orders on the same cell.

Shared form abstractions live in `codegen.framework.forms`:

- `PipelineStage`
- `FormKind`
- `FormOrder`
- `FormPipeline`
- `FormEvaluation`
- `FormCollection`
- `FormMetadata`
- `FormBlock`
- `UnifiedForm`

The form order mapping is:

| Order | Energy Input | Residual Input |
| --- | --- | --- |
| `0` | energy/objective | merit function |
| `1` | gradient | residual |
| `2` | Hessian action | Jacobian action |

This stage still knows whether each equation is energy-based or
residual-based, because the symbolic derivatives are different. It does not
branch by material type and it must not emit kernel files.

### 3. Specialized Form Manipulation

This is the extra stage between symbolic form evaluation and code generation.
It is where each form family is converted into kernel-ready data.

Current implementation:

- `SpecializedFormManipulationStage`
- `FormCollection`
- `EnergyCodeGenerationPayload` for energy-only kernel form state
- `CodeGenerationUnit`
- `CodeGenerationPlan`

For energy forms, this stage currently builds the SoA weak-form data:

- `sfem_soa_weak_form(...)`
- `sfem_soa_kernel_form(...)`
- diagnostic graphs

For variational residual forms, this stage builds:

- residual weak coefficients
- Jacobian-action weak coefficients

Those residual coefficients are stored in the per-order `FormMetadata` of the
same `FormCollection`; the code generation unit does not create a separate
residual payload schema for fields, coefficients, dependencies, or blocks.
Residual systems also store explicit `FormBlock` objects:

- `FormOrder.ONE` blocks are row blocks, one per residual field.
- `FormOrder.TWO` blocks are row/column Jacobian-action blocks.
- Each block carries its expression, weak coefficients, dependencies, and
  diagonal/coupling classification.
- `FormCollection.blocks_for(...)`, `FormCollection.block(...)`, and
  `FormCollection.block_matrix(...)` expose the block structure before code
  generation, so monolithic and subproblem/block kernels can be planned from
  the same form schema.

The output is always a `CodeGenerationPlan`: a flat list of
`CodeGenerationUnit` objects. This is the common interface to the next stage.

This is also the natural extension point for thermo-hyperelasticity and other
coupled models: one material can produce more than one `CodeGenerationUnit`,
including both energy-derived and residual-derived units.

### 4. Unified Code Generation

`CodeGenerationStage` is the only stage that emits generated source files.
It consumes:

- `UserInputStage`
- `CodeGenerationPlan`

Its loop is shared:

```python
for context in user_input.element_contexts:
    for unit in codegen_plan.units_for_context(context):
        files = _emit_codegen_unit(unit, context)
```

The code-generation stage does not call user material construction code and
does not reconstruct material-specific symbolic expressions. It only sees
element contexts and code-generation units.

Generated files are merged, optional `sfem::Op` wrappers are added, and the
outputs are written/compiled by the common `generate(...)` path.

## NeoHookean Ogden Path

### User Input

Defined in:

- `python/codegen/framework/materials/neohookean_ogden.py`

The material is:

```python
material = gen.CodeGenerator(
    "neohookean_ogden",
    systems,
    op_name="GeneratedNeoHookeanOgden",
    parameter_defaults=(("mu", 1.0), ("lmbda", 1.0)),
)
```

The explicit system construction declares a displacement field, builds the
deformation-gradient expression, and registers the strain energy with
`add_energy(..., variables=(F,))`.

Generation scripts:

- `python/codegen/framework/materials/neohookean_ogden.py`
- `python/codegen/framework/generate_neohookean_ogden_files.py`

### Form Evaluation

`sfem.gen._evaluate_forms(...)` dispatches to
the generic equation evaluator.

For each spatial dimension required by the selected elements:

1. Build symbolic `F`.
2. Call `system.add_energy(..., variables=(F,))`.
3. Create `energy_form_pipeline(energy, tuple(F), directions)`.
4. Evaluate requested orders:
   - `objective` -> `FormOrder.ZERO`
   - `gradient` -> `FormOrder.ONE`
   - `apply` -> `FormOrder.TWO`

The result is a dimension-keyed `UnifiedFormEvaluation` containing one unnamed
`EnergyDimensionEvaluation`. This object contains evaluated forms only;
it does not generate files.

### Specialized Form Manipulation

`SpecializedFormManipulationStage` turns the evaluated energy form into a
`CodeGenerationUnit`.

For each dimension, it creates a `CodeGenerationUnit` with kind
`ENERGY_SOA`, the standardized `FormCollection`, and an
`EnergyCodeGenerationPayload` containing only energy-specific emission state:

- kernel forms
- diagnostic graph
- diagnostics flag

### Unified Code Generation

`CodeGenerationStage` emits the unit with `_emit_codegen_unit(...)`.

For `ENERGY_SOA`, the emitter calls:

```python
generate_sfem_soa_cpp_files_for_element(...)
```

This produces:

- `kernel_math.hpp`
- `kernel_diagnostics.hpp`
- dimension/family local kernels, for example
  `generated_neohookean_ogden_d3_tensor_product_local.hpp`
- element-specialized mesh kernels, for example
  `generated_neohookean_ogden_hex8_operator.cpp`
- diagnostic summary files when enabled

If `op_name` is set, `sfem._gen_op.generate_op_files(...)` adds:

- `sfem_GeneratedNeoHookeanOgden.hpp`
- `sfem_GeneratedNeoHookeanOgden.cpp`

## Two-Phase Flow Path

### User Input

Defined in:

- `python/codegen/framework/materials/two_phase_flow.py`

The material is:

```python
material = gen.CodeGenerator(
    "two_phase_flow",
    systems,
    elements=("TRI3", "TET4", "QUAD4", "HEX8"),
    op_name="GeneratedTwoPhaseFlow",
    parameter_defaults=(...),
)
```

The explicit system construction declares pressure spaces, builds the weak
residual expression, and registers it with:

```python
system.add_residual("", form, fields=(water, co2))
```

Generation scripts:

- `python/codegen/framework/materials/two_phase_flow.py`
- `python/codegen/framework/twophaseflow/generate_two_phase_flow_files.py`

### Form Evaluation

`sfem.gen._evaluate_forms(...)` dispatches to
the generic equation evaluator through `EquationSystem.form_collection(...)`.

For each spatial dimension required by the selected elements:

1. Select the explicit `EquationSystem` for that dimension.
2. Ask the system for a `FormCollection` for each equation.
3. For energy equations, derive forms from explicit differentiation variables.
4. For residual equations, lower the weak form into a `CoupledResidualSystem`
   and store that lowered system as `FormCollection.source`.
5. Evaluate:
   - `FormOrder.ZERO`: merit function
   - `FormOrder.ONE`: residual
   - `FormOrder.TWO`: Jacobian action

The result is a dimension-keyed `UnifiedFormEvaluation` whose units reference
the same `FormCollection` interface regardless of whether the user started from
energy or residual notation. This object contains evaluated forms only; it does
not generate files.

### Specialized Form Manipulation

`SpecializedFormManipulationStage` turns the residual evaluated form into a
`CodeGenerationUnit`.

For each dimension, it creates a `CodeGenerationUnit` with kind
`RESIDUAL_SOA` and the standardized `FormCollection`. The residual lowering
state is read from:

- `FormCollection.source` for the lowered `CoupledResidualSystem`
- `FormMetadata(FormOrder.ONE).coefficients` for residual weak coefficients
- `FormMetadata(FormOrder.ONE).blocks` for residual row blocks
- `FormMetadata(FormOrder.TWO).coefficients` for Jacobian-action weak coefficients
- `FormMetadata(FormOrder.TWO).blocks` for coupled row/column block structure

`FormCollection.block_matrix(FormOrder.TWO)` returns the coupled Jacobian-action
matrix in field order. Off-diagonal `FormBlock.is_coupling` entries are the
mixed terms that a later generation-plan stage can emit as separate block
kernels or assemble into a monolithic operator.

### Unified Code Generation

`CodeGenerationStage` emits the unit with `_emit_codegen_unit(...)`.

For `RESIDUAL_SOA`, the emitter calls:

```python
generate_coupled_residual_sfem_files(...)
```

This produces:

- `kernel_math.hpp`
- `kernel_diagnostics.hpp`
- dimension/family local kernels, for example
  `generated_two_phase_flow_d3_tensor_product_local.hpp`
- element-specialized mesh kernels, for example
  `generated_two_phase_flow_hex8_operator.cpp`

If `op_name` is set, `sfem._gen_op.generate_op_files(...)` adds:

- `sfem_GeneratedTwoPhaseFlow.hpp`
- `sfem_GeneratedTwoPhaseFlow.cpp`

## Poro-Hyperelasticity Path

### User Input

Defined in:

- `python/codegen/framework/materials/poro_hyperelasticity.py`

The material is:

```python
material = gen.CodeGenerator(
    "poro_hyperelasticity",
    systems,
    elements=gen.sfem_taylor_hood_element_types(),
    parameter_defaults=(...),
)
```

The user constructs explicit dimension-specialized systems and registers
multiple equations:

```python
def build_system(dim):
    system = gen.EquationSystemBuilder(dim)
    u = gen.Function(V, "u", qualifier=gen.DISPLACEMENT)
    p = gen.Function(Q, "p", qualifier=gen.PRESSURE)
    F = gen.variable(gen.Identity(system.dim) + gen.grad(u), name="F")
    system.add_energy("solid", strain_energy, fields=(u,), variables=(F,))
    system.add_residual("poro", pressure_residual, fields=(u, p))
    return system.build()

systems = gen.EquationSystems(build_system(2), build_system(3))
```

The `solid` unit is an energy-based NeoHookean Ogden model. The `poro` unit is
a residual-based pressure/displacement coupling with fields `u0`, `u1`, ...
and `p`. The pressure residual contains storage, pressure diffusion, and a
Biot-style volumetric coupling through `div(u) - div(u_old)`. The displacement
residual contains the pore-pressure contribution to the mechanics equations.

The compatible elements are Taylor-Hood pairs:

| Pair | Displacement | Pressure |
| --- | --- | --- |
| `TRI6_TRI3` | `TRI6` | `TRI3` |
| `TET10_TET4` | `TET10` | `TET4` |
| `HEX27_HEX8` | `HEX27` | `HEX8` |

Equal-order choices such as `TRI3` are intentionally not enabled for this
formulation.

### Form Evaluation

`sfem.gen._evaluate_forms(...)` sees only equations:

- energy equation `solid`
- residual equation `poro`

For each dimension, it evaluates both inputs and stores both in one
`UnifiedFormEvaluation`.

### Specialized Form Manipulation

`SpecializedFormManipulationStage` produces one `CodeGenerationPlan` with two
units:

- `ENERGY_SOA`, unit name `solid`
- `RESIDUAL_SOA`, unit name `poro`

The unit name is part of the generated prefix, so filenames do not collide.

### Unified Code Generation

`CodeGenerationStage` consumes both units through the same loop. For compatible
Taylor-Hood elements the solid unit can use the displacement/cell element, but
the poro residual unit also needs pressure field shape functions from the lower
order element.

Mixed `sfem::Op` wrapper generation is intentionally not enabled yet because
the runtime dispatch contract for combined value/gradient/apply across multiple
physics units has to be designed explicitly.

For the Taylor-Hood formulation, residual emission uses the compatible element
contract from `EquationSystem` field families. The cell geometry is evaluated on
the high-order displacement element, while each residual row uses its own
field-specific shape values, gradients, and scatter size. This keeps the poro
residual mixed-order instead of silently falling back to an equal-order kernel.

## Shared Boundary

After specialized form manipulation, the downstream interface is always:

- `CodeGenerationPlan`
- one or more `CodeGenerationUnit` objects
- one `ElementGenerationContext` per element

The unified code-generation stage is responsible for all generated source
emission. It is the same stage for NeoHookean, two-phase flow, and mixed
poro-hyperelasticity.

Material-specific differences are restricted to:

- user input construction
- form evaluation
- specialized form manipulation into code-generation units

Everything after that belongs to the single code-generation process.
