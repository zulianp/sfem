import argparse
import glob
import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum

import sympy as sp

from ._gen_op import generate_op_files
from codegen.framework import (
    CodegenQualifier,
    CoupledResidualSystem,
    DEFORMATION_GRADIENT,
    DISPLACEMENT,
    BasisDataLayout,
    BasisEvaluation,
    BasisFamily,
    BasisPlanNode,
    BlockPlan,
    DataStreamLayout,
    DataStreamPlan,
    DataStreamRole,
    EquationForm,
    EquationSystem,
    EquationSystemBuilder,
    EquationSystems,
    FiniteElement,
    FieldQualifier,
    Function,
    FunctionSpace,
    FormCollection,
    FormBlock,
    GeneratedKernelFile,
    Identity,
    HyperelasticQualifier,
    KernelExpressions,
    MATERIAL_PARAMETER,
    MaterialParameter,
    MaterialParameterQualifier,
    MixedFunctionSpace,
    PRESSURE,
    PREVIOUS_ARGUMENT,
    PreviousFunction,
    QualifiedExpression,
    ScalarField,
    SymbolicArgument,
    SymbolicField,
    TEST_ARGUMENT,
    TensorField,
    TensorProductDataLayout,
    TensorProductOperation,
    TensorProductSumFactorizationPlan,
    TestFunction,
    TRIAL_ARGUMENT,
    TrialFunction,
    TwoPhaseFlowConstitutiveModel,
    FormEvaluation,
    FormMetadata,
    FormOrder,
    FormQualifier,
    GenerationPlan,
    GeometryEvaluation,
    GeometryInputLayout,
    GeometryMode,
    GeometryPlan,
    GeometryPlanNode,
    KernelCoupling,
    KernelPlan,
    KernelEmission,
    KernelScope,
    KernelTarget,
    LocalKernelPlan,
    LocalPhase,
    LocalPhasePlan,
    MeshKernelPlan,
    MeshPhase,
    MeshPhasePlan,
    PipelineStage,
    StandardFormName,
    VectorField,
    VectorElement,
    VectorFunctionSpace,
    VectorFunction,
    TensorFunction,
    VELOCITY,
    current_geometric_dimension,
    energy_form_pipeline,
    geometry_plans_for_fem_policy,
    geometric_dimension_context,
    matrix_inner,
    residual_form_pipeline,
    sfem_soa_kernel_form,
    sfem_soa_weak_form,
    streams_in_shape_order,
    scalar_field,
    tensor_field,
    tensor_product_cartesian_shape_order,
    tensor_product_field_evaluation_plan,
    tensor_product_geometry_jacobian_plan,
    tensor_product_geometry_jacobian_plan_from_sizes,
    tensor_product_sum_factorization_plan,
    tensor_product_test_contraction_plan,
    test_function,
    trial_function,
    vector_field,
    adjugate,
    affine_geometry_plan,
    basis_plan_for_element_at_cell_rule,
    basis_plan_for_quadrature_rule,
    basis_plans_for_fem_policy,
    det,
    deformation_gradient,
    derivative,
    div,
    field_basis_plan_for_fem_policy,
    field_basis_plans_for_fem_policy,
    grad,
    inner,
    inv,
    isoparametric_geometry_plan,
    material_parameter,
    old,
    previous_function,
    qualifiers,
    qualify,
    variable,
    value,
)
from codegen.framework.fem import (
    SfemCompatibleElement,
    SfemElementBasisPolicy,
    SfemFEMPolicy,
    SfemFieldFamilyCompatibilityPolicy,
    SfemReferenceData,
    sfem_cell_rule_points,
    sfem_default_quadrature_order,
    sfem_detect_compatible_element_types,
    sfem_detect_taylor_hood_element_types,
    sfem_element_quadrature_rule,
    sfem_field_n_shape,
    sfem_fem_policy,
    sfem_is_proteus_hex_element,
    sfem_is_tensor_product_hex_element,
    sfem_mesh_reference_data,
    sfem_normalize_integration_case,
    sfem_proteus_hex_element_types,
    sfem_reference_data,
    sfem_shape_data_for_element_at_cell_rule,
    sfem_simplex_grad_ref_name,
    sfem_simplex_field_reference_data,
    sfem_soa_element_specialization,
    sfem_soa_element_specializations,
    sfem_supported_element_types,
    sfem_taylor_hood_element_types,
    sfem_tensor_product_field_reference_data,
    sfem_tensor_product_hex_order,
    sfem_tensor_hex_shape_index,
)
from codegen.framework.openmp_backend import OpenMPSoABackend


DEFAULT_VECTOR_SIZE = 16
OPENMP_SOA_BACKEND = OpenMPSoABackend()


@dataclass(frozen=True)
class QuadratureSetting:
    element_type: str
    order: int
    integration_case: str = ""

    def __post_init__(self):
        element_type = str(self.element_type).upper()
        integration_case = sfem_normalize_integration_case(self.integration_case)
        order = int(self.order)
        if not element_type:
            raise ValueError("quadrature setting requires an element type")
        if order <= 0:
            raise ValueError("quadrature setting order must be positive")
        object.__setattr__(self, "element_type", element_type)
        object.__setattr__(self, "integration_case", integration_case)
        object.__setattr__(self, "order", order)


@dataclass(frozen=True)
class CodeGenerator:
    name: str
    systems: object
    elements: tuple = None
    op_name: str = None
    parameter_defaults: tuple = ()
    quadrature_settings: tuple = ()

    def __post_init__(self):
        _validate_name(self.name)
        if callable(self.systems):
            raise TypeError("CodeGenerator requires equation systems, not a callback")
        object.__setattr__(self, "systems", _as_equation_systems(self.systems))
        if not self.systems:
            raise ValueError("code generators require at least one equation system")
        elements = tuple(self.elements) if self.elements else _default_elements_for_systems(self.systems)
        if not elements:
            raise ValueError("code generators require supported elements")
        object.__setattr__(self, "elements", elements)
        object.__setattr__(
            self,
            "quadrature_settings",
            tuple(_normalize_quadrature_setting(setting) for setting in self.quadrature_settings),
        )
        _validate_op(self.op_name, self.parameter_defaults)


@dataclass(frozen=True)
class GenerationResult:
    sources: tuple
    objects: tuple = ()
    plan: object = None
    plan_dump: object = None


@dataclass(frozen=True)
class _EnergyOpMaterialAdapter:
    name: str
    op_name: str
    parameter_defaults: tuple
    energy: bool = True


@dataclass(frozen=True)
class _ResidualOpMaterialAdapter:
    name: str
    define: object
    op_name: str
    parameter_defaults: tuple
    form_collections: object = None


@dataclass(frozen=True)
class ElementGenerationContext:
    material_name: str
    element_type: str
    label: str
    specialization: object
    affine_specialization: object
    fem_policy: object
    geometry_plans: tuple
    basis_plans: tuple
    compatible_element: object = None

    def __post_init__(self):
        object.__setattr__(self, "geometry_plans", tuple(self.geometry_plans))
        object.__setattr__(self, "basis_plans", tuple(self.basis_plans))
        seen = set()
        for plan in self.geometry_plans:
            if not isinstance(plan, GeometryPlanNode):
                raise TypeError("geometry_plans must contain GeometryPlanNode objects")
            if plan.element_type != self.element_type:
                raise ValueError(
                    "geometry plan element '%s' does not match context element '%s'"
                    % (plan.element_type, self.element_type)
                )
            if plan.dim != self.specialization.dim:
                raise ValueError(
                    "geometry plan dimension %d does not match context dimension %d"
                    % (plan.dim, self.specialization.dim)
                )
            if plan.mode in seen:
                raise ValueError("duplicate geometry plan mode '%s'" % plan.mode.value)
            seen.add(plan.mode)
        seen = set()
        for plan in self.basis_plans:
            if not isinstance(plan, BasisPlanNode):
                raise TypeError("basis_plans must contain BasisPlanNode objects")
            if plan.cell_element_type != self.element_type:
                raise ValueError(
                    "basis plan cell element '%s' does not match context element '%s'"
                    % (plan.cell_element_type, self.element_type)
                )
            if plan.dim != self.specialization.dim:
                raise ValueError(
                    "basis plan dimension %d does not match context dimension %d"
                    % (plan.dim, self.specialization.dim)
                )
            if plan.role in seen:
                raise ValueError("duplicate basis plan role '%s'" % plan.role)
            seen.add(plan.role)
        if self.affine_specialization.dim != self.specialization.dim:
            raise ValueError("affine and isoparametric specializations must have the same dimension")
        if self.affine_specialization.n_shape != self.specialization.n_shape:
            raise ValueError("affine and isoparametric specializations must have the same shape count")

    @classmethod
    def create(
        cls,
        material_name,
        element,
        vector_size,
        isoparametric_quadrature_order,
        isoparametric_integration_case="standard",
        affine_quadrature_order=None,
        affine_integration_case="standard",
    ):
        policy = sfem_fem_policy(
            element,
            vector_size,
            isoparametric_quadrature_order,
            integration_case=isoparametric_integration_case,
        )
        affine_policy = sfem_fem_policy(
            element,
            vector_size,
            affine_quadrature_order,
            integration_case=affine_integration_case,
        )
        return cls(
            material_name,
            policy.cell_element_type,
            policy.label,
            policy.specialization,
            affine_policy.specialization,
            policy,
            geometry_plans_for_fem_policy(policy),
            basis_plans_for_fem_policy(policy),
            policy.compatible_element,
        )

    @property
    def generated_prefix(self):
        return self.material_name

    @property
    def element_prefix(self):
        return "%s_%s" % (self.generated_prefix, self.label)

    @property
    def local_prefix(self):
        return "%s_d%d_%s" % (
            self.generated_prefix,
            self.specialization.dim,
            self.family,
        )

    @property
    def family(self):
        return self.fem_policy.family

    @property
    def is_mixed_order(self):
        return self.fem_policy.is_mixed_order

    def geometry_plan(self, mode):
        mode = GeometryMode(mode)
        for plan in self.geometry_plans:
            if plan.mode is mode:
                return plan
        raise ValueError("geometry mode '%s' is not available" % mode.value)

    def basis_plan(self, role="cell"):
        role = str(role)
        for plan in self.basis_plans:
            if plan.role == role:
                return plan
        raise ValueError("basis plan role '%s' is not available" % role)

    def field_basis_plan(self, field):
        return field_basis_plan_for_fem_policy(self.fem_policy, field)

    def field_basis_plans(self, fields):
        return field_basis_plans_for_fem_policy(self.fem_policy, fields)

    @property
    def isoparametric_specialization(self):
        return self.specialization


@dataclass(frozen=True)
class UserInputStage:
    material: object
    elements: tuple
    vector_size: int
    quadrature_order: object
    element_contexts: tuple

    @property
    def stage(self):
        return PipelineStage.USER_INPUT

    @classmethod
    def create(cls, material, elements, vector_size, quadrature_order):
        contexts = []
        for element in elements:
            isoparametric_case = _integration_case_for_material_element(
                material,
                element,
            )
            affine_case = _affine_integration_case_for_material_element(
                material,
                element,
            )
            contexts.append(
                ElementGenerationContext.create(
                    material.name,
                    element,
                    vector_size,
                    _quadrature_order_for_material_element(
                        material,
                        element,
                        quadrature_order,
                        isoparametric_case,
                    ),
                    isoparametric_case,
                    _quadrature_order_for_material_element(
                        material,
                        element,
                        quadrature_order,
                        affine_case,
                    ),
                    affine_case,
                )
            )
        return cls(
            material,
            tuple(elements),
            vector_size,
            quadrature_order,
            tuple(contexts),
        )


def _normalize_quadrature_setting(setting):
    if isinstance(setting, QuadratureSetting):
        return setting
    if isinstance(setting, dict):
        return QuadratureSetting(
            setting.get("element_type", setting.get("element", "")),
            setting["order"],
            setting.get("integration_case", setting.get("case", "")),
        )
    values = tuple(setting)
    if len(values) == 2:
        element_type, order = values
        return QuadratureSetting(element_type, order)
    if len(values) == 3:
        element_type, integration_case, order = values
        return QuadratureSetting(element_type, order, integration_case)
    raise ValueError(
        "quadrature settings must be QuadratureSetting, dict, (element, order), "
        "or (element, integration_case, order)"
    )


def _quadrature_order_for_material_element(material, element, explicit_order, integration_case=None):
    if explicit_order is not None:
        return explicit_order
    integration_case = (
        _integration_case_for_material_element(material, element)
        if integration_case is None
        else str(integration_case)
    )
    cell_element_type = _cell_element_type(element)
    element_label = _element_label(element)
    fallback = None
    for setting in getattr(material, "quadrature_settings", ()):
        if setting.element_type in (cell_element_type, element_label):
            if setting.integration_case == integration_case:
                return setting.order
            if not setting.integration_case:
                fallback = setting.order
    return fallback


def _affine_integration_case_for_material_element(material, element):
    integration_case = _integration_case_for_material_element(material, element)
    if integration_case == "isoparametric_mixed":
        return "affine_mixed"
    if integration_case == "energy":
        return "affine_energy"
    return integration_case


def _integration_case_for_material_element(material, element):
    compatible = element if isinstance(element, SfemCompatibleElement) else None
    if compatible is not None and compatible.is_mixed_order:
        return "isoparametric_mixed"
    cell_element_type = _cell_element_type(element)
    system = material.systems.for_dim(_element_dim(element))
    cases = []
    if any(equation.is_energy for equation in system.equations):
        cases.append("energy")
    residual_case = _value_residual_integration_case(system)
    if residual_case is not None:
        cases.append(residual_case)
    if not cases:
        return "standard"
    return max(
        cases,
        key=lambda case: sfem_default_quadrature_order(
            cell_element_type,
            integration_case=case,
        ),
    )


def _value_residual_integration_case(system):
    has_linear_value = False
    for equation in system.equations:
        if not equation.is_residual:
            continue
        residual_system = CoupledResidualSystem(system.dim)
        equation.define(residual_system)
        value_symbols = set()
        for field in residual_system.fields:
            value_symbols.add(field.value)
            value_symbols.add(field.direction_value)
            if field.previous_value is not None:
                value_symbols.add(field.previous_value)
        for field in residual_system.fields:
            expression = residual_system.residual_expression(field)
            expression = sp.sympify(expression)
            expression_values = expression.free_symbols.intersection(value_symbols)
            if not expression_values:
                continue
            if _is_linear_polynomial_in(expression, expression_values):
                has_linear_value = True
            else:
                return "value_residual"
    return "value_linear_residual" if has_linear_value else None


def _is_linear_polynomial_in(expression, symbols):
    try:
        polynomial = sp.Poly(expression, *tuple(sorted(symbols, key=str)))
    except (sp.PolynomialError, TypeError, ValueError):
        return False
    return polynomial.total_degree() <= 1


def _cell_element_type(element):
    if isinstance(element, SfemCompatibleElement):
        return element.cell_element_type
    return str(element).upper()


def _element_label(element):
    if isinstance(element, SfemCompatibleElement):
        return element.name.upper()
    return str(element).upper()


def _element_dim(element):
    return sfem_element_quadrature_rule(_cell_element_type(element)).dim


@dataclass(frozen=True)
class EnergyDimensionEvaluation:
    name: str
    form_evaluation: FormCollection
    deformation_gradient: object
    variables: tuple
    kernels: tuple
    diagnostics: bool


@dataclass(frozen=True)
class ResidualDimensionEvaluation:
    name: str
    form_evaluation: FormCollection


@dataclass(frozen=True)
class DimensionFormEvaluation:
    dim: int
    units: tuple


@dataclass(frozen=True)
class UnifiedFormEvaluation:
    material: object
    by_dim: dict

    @property
    def stage(self):
        return PipelineStage.FORM_EVALUATION


class CodeGenerationKind(Enum):
    ENERGY_SOA = "energy_soa"
    RESIDUAL_SOA = "residual_soa"


@dataclass(frozen=True)
class CodeGenerationUnit(KernelPlan):
    material_name: str = ""
    unit_name: str = ""


@dataclass(frozen=True)
class EnergyCodeGenerationPayload:
    kernel_forms: tuple
    diagnostic_graph: object
    diagnostics: bool


CodeGenerationPlan = GenerationPlan


@dataclass(frozen=True)
class SpecializedFormManipulationStage:
    user_input: UserInputStage
    form_evaluation: object

    @property
    def stage(self):
        return PipelineStage.SPECIALIZED_FORM_MANIPULATION

    def run(self):
        return _codegen_plan_from_form_evaluation(self.form_evaluation)


@dataclass(frozen=True)
class CodeGenerationStage:
    user_input: UserInputStage
    codegen_plan: CodeGenerationPlan

    @property
    def stage(self):
        return PipelineStage.CODE_GENERATION

    def run(self):
        outputs = {}
        for context in self.user_input.element_contexts:
            for unit in self.codegen_plan.emission_kernels_for_context(context):
                _merge_files(
                    outputs,
                    _layout_codegen_files(
                        unit,
                        context,
                        _emit_codegen_unit(unit, context),
                    ),
                )
        return outputs


def generate(
    material,
    out_dir,
    *,
    elements=None,
    vector_size=DEFAULT_VECTOR_SIZE,
    quadrature_order=None,
    compile=False,
    clean=True,
    dump_plan=False,
    plan_out=None,
):
    vector_size = int(vector_size)
    if vector_size <= 0:
        raise ValueError("vector_size must be positive")

    selected = _parse_elements(
        elements,
        material.elements or sfem_supported_element_types(),
    )
    out_dir = os.path.abspath(os.fspath(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    if clean:
        _clean_outputs(out_dir, material.name)

    user_input = UserInputStage.create(
        material,
        selected,
        vector_size,
        quadrature_order,
    )
    form_evaluation = _evaluate_forms(user_input)
    codegen_plan = SpecializedFormManipulationStage(
        user_input,
        form_evaluation,
    ).run()
    plan_dump = _write_plan_dump(codegen_plan, out_dir, material.name, plan_out) if dump_plan or plan_out else None
    files = CodeGenerationStage(user_input, codegen_plan).run()

    if material.op_name:
        files.update(_generate_op_wrapper_files(material, selected, user_input))

    source_paths = _write_files(out_dir, files)
    object_paths = _compile_operators(source_paths) if compile else ()
    return GenerationResult(source_paths, object_paths, codegen_plan, plan_dump)


def run(material, default_out_dir, argv=None):
    parser = argparse.ArgumentParser(
        description="Generate SFEM kernels for %s." % material.name.replace("_", " ")
    )
    parser.add_argument("--out-dir", default=os.fspath(default_out_dir))
    parser.add_argument(
        "--element",
        "--element-type",
        action="append",
        dest="elements",
        help="Element type; may be repeated or comma-separated.",
    )
    parser.add_argument("--quadrature-order", type=int)
    parser.add_argument("--vector-size", type=int, default=DEFAULT_VECTOR_SIZE)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--dump-plan",
        action="store_true",
        help="Write a JSON generation-plan dump next to generated sources.",
    )
    parser.add_argument(
        "--plan-out",
        help="Path for the JSON generation-plan dump. Implies --dump-plan.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep stale outputs from previous generator runs.",
    )
    args = parser.parse_args(argv)
    try:
        result = generate(
            material,
            args.out_dir,
            elements=args.elements,
            vector_size=args.vector_size,
            quadrature_order=args.quadrature_order,
            compile=args.compile,
            clean=not args.keep_existing,
            dump_plan=args.dump_plan,
            plan_out=args.plan_out,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    print("Generated:")
    for path in result.sources:
        print("  %s" % path)
    if result.objects:
        print("Compiled:")
        for path in result.objects:
            print("  %s" % path)
    if result.plan_dump:
        print("Plan:")
        print("  %s" % result.plan_dump)
    return result


def _write_plan_dump(plan, out_dir, material_name, plan_out=None):
    if plan_out is None:
        path = os.path.join(out_dir, "%s_plan.json" % material_name)
    else:
        path = os.fspath(plan_out)
        if os.path.isdir(path):
            path = os.path.join(path, "%s_plan.json" % material_name)
        elif not os.path.isabs(path):
            path = os.path.abspath(path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    plan.write_json(path)
    return path


def _evaluate_forms(user_input):
    by_dim = {}
    for context in user_input.element_contexts:
        dim = context.specialization.dim
        if dim in by_dim:
            continue
        material_system = user_input.material.systems.for_dim(dim)
        units = tuple(
            _evaluate_equation(
                dim,
                equation,
                material_system.form_collection(
                    equation,
                    orders=_equation_form_orders(equation),
                ),
            )
            for equation in material_system.equations
        )
        if not units:
            raise ValueError("material '%s' did not define any equations" % user_input.material.name)
        by_dim[dim] = DimensionFormEvaluation(dim, units)
    return UnifiedFormEvaluation(user_input.material, by_dim)


def _codegen_plan_from_form_evaluation(form_evaluation):
    units = []
    for dim_eval in form_evaluation.by_dim.values():
        for evaluated in dim_eval.units:
            if isinstance(evaluated, EnergyDimensionEvaluation):
                units.append(
                    _energy_codegen_unit(
                        form_evaluation.material.name,
                        dim_eval.dim,
                        evaluated,
                    )
                )
            elif isinstance(evaluated, ResidualDimensionEvaluation):
                units.append(
                    _residual_codegen_unit(
                        form_evaluation.material.name,
                        dim_eval.dim,
                        evaluated,
                    )
                )
            else:
                raise TypeError(
                    "unsupported evaluated form unit %s" % type(evaluated).__name__
                )
    return GenerationPlan(tuple(units))


def _energy_codegen_unit(material_name, dim, evaluated):
    if evaluated.deformation_gradient.shape != (dim, dim):
        raise ValueError(
            "energy code generation currently requires %d x %d explicit variables; "
            "got %d variables for energy unit '%s'"
            % (dim, dim, len(evaluated.variables), evaluated.name or material_name)
        )
    weak_form = sfem_soa_weak_form(
        evaluated.form_evaluation.form(FormOrder.ZERO).expression,
        evaluated.deformation_gradient,
    )
    kernel_forms = tuple(
        sfem_soa_kernel_form(
            kernel,
            weak_form=weak_form,
            has_direction=kernel == "apply",
            output_mode="accumulate",
        )
        for kernel in evaluated.kernels
    )
    diagnostic_graph = None
    if evaluated.diagnostics:
        diagnostic_graph = (
            KernelExpressions()
            .add(
                "operator_evaluation",
                weak_form.diagnostic_expressions(has_direction=True),
            )
            .build_graph(
                data_symbols=weak_form.deformation_gradient,
                temporary_prefix="%s_inspect_tmp" % material_name,
            )
        )
    return CodeGenerationUnit(
        name=_unit_output_name_from_parts(material_name, evaluated.name),
        kind=CodeGenerationKind.ENERGY_SOA,
        form_collection=evaluated.form_evaluation,
        dim=dim,
        mesh_phases=(
            MeshPhase.GEOMETRY,
            MeshPhase.LOCAL_CALL,
            MeshPhase.SCATTER,
        ),
        mesh_phase_plans=(
            MeshPhasePlan(MeshPhase.GEOMETRY),
            MeshPhasePlan(MeshPhase.LOCAL_CALL),
            MeshPhasePlan(MeshPhase.SCATTER),
        ),
        target=KernelTarget.OPENMP,
        payload=EnergyCodeGenerationPayload(
            kernel_forms,
            diagnostic_graph,
            evaluated.diagnostics,
        ),
        coupling=KernelCoupling.SINGLE_FIELD,
        material_name=material_name,
        unit_name=evaluated.name,
    )


def _residual_codegen_unit(material_name, dim, evaluated):
    blocks = _block_plans_from_form_collection(evaluated.form_evaluation)
    block_kernels = _block_codegen_units(
        material_name,
        dim,
        evaluated,
        blocks,
    )
    return CodeGenerationUnit(
        name=_unit_output_name_from_parts(material_name, evaluated.name),
        kind=CodeGenerationKind.RESIDUAL_SOA,
        form_collection=evaluated.form_evaluation,
        dim=dim,
        mesh_phases=(
            MeshPhase.GATHER,
            MeshPhase.GEOMETRY,
            MeshPhase.LOCAL_CALL,
            MeshPhase.SCATTER,
        ),
        mesh_phase_plans=(
            MeshPhasePlan(MeshPhase.GATHER),
            MeshPhasePlan(MeshPhase.GEOMETRY),
            MeshPhasePlan(MeshPhase.LOCAL_CALL, blocks=blocks),
            MeshPhasePlan(MeshPhase.SCATTER),
        ),
        blocks=blocks,
        block_kernels=block_kernels,
        scope=KernelScope.MONOLITHIC,
        coupling=_kernel_coupling_for_collection(evaluated.form_evaluation),
        target=KernelTarget.OPENMP,
        material_name=material_name,
        unit_name=evaluated.name,
    )


def _block_plans_from_form_collection(collection):
    return tuple(_block_plan_from_form_block(block) for block in collection.blocks)


def _block_plan_from_form_block(block):
    return BlockPlan(
        block.name,
        block.row_field,
        block.column_field or "",
        block.order,
        local_phase_plans=_residual_local_phase_plans(),
    )


def _residual_local_phase_plans():
    return (
        LocalPhasePlan(LocalPhase.EVALUATE_TRIAL),
        LocalPhasePlan(LocalPhase.TRANSFORM_REFERENCE),
        LocalPhasePlan(LocalPhase.EVALUATE_MATERIAL),
        LocalPhasePlan(LocalPhase.CONTRACT_TEST),
    )


def _residual_local_phases():
    return tuple(plan.phase for plan in _residual_local_phase_plans())


def _block_codegen_units(material_name, dim, evaluated, blocks):
    return tuple(
        CodeGenerationUnit(
            name=_unit_output_name_from_parts(
                material_name,
                _block_unit_name(evaluated.name, block),
            ),
            kind=CodeGenerationKind.RESIDUAL_SOA,
            form_collection=evaluated.form_evaluation,
            dim=dim,
            mesh_phases=(
                MeshPhase.GATHER,
                MeshPhase.GEOMETRY,
                MeshPhase.LOCAL_CALL,
                MeshPhase.SCATTER,
            ),
            mesh_phase_plans=(
                MeshPhasePlan(MeshPhase.GATHER),
                MeshPhasePlan(MeshPhase.GEOMETRY),
                MeshPhasePlan(MeshPhase.LOCAL_CALL, blocks=(block,)),
                MeshPhasePlan(MeshPhase.SCATTER),
            ),
            blocks=(block,),
            scope=KernelScope.BLOCK,
            coupling=KernelCoupling.BLOCK,
            block=block,
            emission=KernelEmission.FILES,
            target=KernelTarget.OPENMP,
            material_name=material_name,
            unit_name=_block_unit_name(evaluated.name, block),
        )
        for block in blocks
    )


def _kernel_coupling_for_collection(collection):
    if len(tuple(collection.fields)) > 1:
        return KernelCoupling.COMPLETE_SYSTEM
    return KernelCoupling.SINGLE_FIELD


def _block_unit_name(unit_name, block):
    if unit_name:
        return "%s_%s" % (unit_name, block.name)
    return block.name


def _emit_codegen_unit(unit, context):
    if unit.kind is CodeGenerationKind.ENERGY_SOA:
        return _emit_energy_soa(unit, context)
    if unit.kind is CodeGenerationKind.RESIDUAL_SOA:
        return _emit_residual_soa(unit, context)
    raise ValueError("unsupported code generation unit kind %s" % unit.kind)


def _emit_energy_soa(unit, context):
    payload = unit.payload
    files = list(OPENMP_SOA_BACKEND.emit(unit, context))
    if payload.diagnostics:
        report_prefix = "%s_%s" % (
            _unit_report_name(unit),
            context.element_type.lower(),
        )
        files.append(
            GeneratedKernelFile(
                "%s_summary.md" % report_prefix,
                _summary(
                    unit.material_name,
                    payload.diagnostic_graph,
                    context.specialization,
                ),
            )
        )
        files.append(
            GeneratedKernelFile(
                "%s_reduced_outputs.txt" % report_prefix,
                "\n\n".join(
                    str(output)
                    for output in payload.diagnostic_graph.reduced_outputs
                )
                + "\n",
            )
        )
    return tuple(files)


def _emit_residual_soa(unit, context):
    return tuple(OPENMP_SOA_BACKEND.emit(unit, context))


def _layout_codegen_files(unit, context, files):
    local_headers = tuple(
        generated.path
        for generated in files
        if generated.path.endswith("_local.hpp")
    )
    return tuple(
        GeneratedKernelFile(
            _layout_codegen_path(unit, context, generated.path),
            _layout_codegen_source(
                unit,
                context,
                generated.path,
                generated.source,
                local_headers,
            ),
        )
        for generated in files
    )


def _layout_codegen_path(unit, context, filename):
    directory = _codegen_file_directory(unit, context, filename)
    if not directory:
        return filename
    return os.path.join(directory, filename)


def _layout_codegen_source(unit, context, filename, source, local_headers):
    directory = _codegen_file_directory(unit, context, filename)
    replacements = {}
    for header in _CODEGEN_COMMON_HEADERS:
        replacements[header] = _relative_codegen_include(directory, header)
    for local in local_headers:
        replacements[local] = _relative_codegen_include(
            directory,
            os.path.join(_codegen_dimension_directory(context), local),
        )

    relocated = source
    for header, relative in replacements.items():
        if header != relative:
            relocated = relocated.replace(
                '#include "%s"' % header,
                '#include "%s"' % relative,
            )
    return relocated


def _codegen_file_directory(unit, context, filename):
    if filename in _CODEGEN_COMMON_HEADERS:
        return ""
    if filename.endswith("_local.hpp"):
        return _codegen_dimension_directory(context)
    return _codegen_output_directory(unit, context)


def _codegen_dimension_directory(context):
    return "d%d" % int(context.specialization.dim)


def _codegen_output_directory(unit, context):
    return os.path.join(
        _codegen_dimension_directory(context),
        _codegen_output_element_label(unit, context),
    )


_CODEGEN_COMMON_HEADERS = frozenset(
    (
        "kernel_math.hpp",
        "kernel_diagnostics.hpp",
        "tensor_product_kernels.hpp",
    )
)


def _relative_codegen_include(directory, target):
    if not directory:
        return target.replace(os.sep, "/")
    return os.path.relpath(target, start=directory).replace(os.sep, "/")


def _codegen_output_element_label(unit, context):
    if (
        unit.is_block
        and unit.block is not None
        and unit.block.form_order is FormOrder.TWO
        and unit.block.column_field
        and unit.block.row_field == unit.block.column_field
    ):
        return _field_element_label(unit.form_collection.fields, unit.block.row_field, context)
    field_labels = _unit_field_element_labels(unit, context)
    if len(field_labels) == 1:
        return field_labels[0]
    return context.label.lower()


def _unit_field_element_labels(unit, context):
    labels = []
    for field, element_type in context.fem_policy.field_element_types_for(unit.form_collection.fields):
        label = str(element_type).lower()
        if label not in labels:
            labels.append(label)
    return tuple(labels)


def _field_element_label(fields, field_name, context):
    for field, element_type in context.fem_policy.field_element_types_for(fields):
        if field.name == field_name:
            return str(element_type).lower()
    raise ValueError("field '%s' is not available in element context" % field_name)


def _material_equations(material, dim):
    if not isinstance(material, CodeGenerator):
        raise TypeError("generation requires CodeGenerator")
    return material.systems.for_dim(dim).equations


def _generate_op_wrapper_files(material, selected, user_input):
    if not user_input.element_contexts:
        raise ValueError("generated Op wrapper requires at least one element context")
    representative_dim = user_input.element_contexts[0].specialization.dim
    equations = _material_equations(material, representative_dim)
    if len(equations) != 1 or equations[0].name:
        raise ValueError(
            "generated Op wrappers for CodeGenerator require exactly one unnamed equation"
        )
    equation = equations[0]
    if equation.is_energy:
        return generate_op_files(
            _EnergyOpMaterialAdapter(
                material.name,
                material.op_name,
                material.parameter_defaults,
            ),
            selected,
        )
    if equation.is_residual:
        systems_by_dim = {
            context.specialization.dim: material.systems.for_dim(context.specialization.dim)
            for context in user_input.element_contexts
        }
        form_collections = {
            dim: system.form_collection(
                system.equations[0],
                orders=_equation_form_orders(equation),
            )
            for dim, system in systems_by_dim.items()
        }
        return generate_op_files(
            _ResidualOpMaterialAdapter(
                material.name,
                equation.define,
                material.op_name,
                material.parameter_defaults,
                form_collections,
            ),
            selected,
        )
    raise TypeError("unsupported unified equation form %s" % equation.form)


def _evaluate_equation(dim, equation, form_collection=None):
    if equation.is_energy:
        return _evaluate_energy_equation(dim, equation, form_collection)
    if equation.is_residual:
        return _evaluate_residual_equation(dim, equation, form_collection)
    raise TypeError("unsupported equation form %s" % equation.form)


def _equation_form_orders(equation):
    if equation.is_energy:
        return _energy_form_orders(equation.kernels)
    if equation.is_residual:
        return (FormOrder.ZERO, FormOrder.ONE, FormOrder.TWO)
    raise TypeError("unsupported equation form %s" % equation.form)


def _evaluate_energy_equation(dim, equation, form_collection=None):
    orders = _energy_form_orders(equation.kernels)
    variables = tuple(equation.variables)
    if not variables:
        raise ValueError("energy equation '%s' requires explicit variables" % equation.name)
    data_symbols = _energy_data_symbols(dim, variables)
    if form_collection is None:
        directions = None
        if FormOrder.TWO in orders:
            directions = tuple(equation.directions) or _default_energy_directions(variables)
        form_evaluation = energy_form_pipeline(
            equation.define,
            variables,
            directions,
        ).evaluate(orders)
    else:
        form_evaluation = form_collection
    return EnergyDimensionEvaluation(
        equation.name,
        form_evaluation,
        data_symbols,
        variables,
        equation.kernels,
        equation.diagnostics,
    )


def _energy_data_symbols(dim, variables):
    if len(variables) == dim * dim:
        return sp.Matrix(dim, dim, variables)
    return sp.Matrix(len(variables), 1, variables)


def _default_energy_directions(variables):
    directions = []
    for variable in variables:
        name = str(variable)
        if name.startswith("F["):
            directions.append(sp.Symbol("d%s" % name))
        elif "[" in name:
            directions.append(sp.Symbol("d_%s" % name))
        else:
            directions.append(sp.Symbol("%s_trial" % name))
    return tuple(directions)


def _evaluate_residual_equation(dim, equation, form_collection=None):
    if form_collection is None:
        system = CoupledResidualSystem(dim)
        equation.define(system)
        residual_vector = sp.Matrix(
            [system.residual_expression(field) for field in system.fields]
        )
        variables = tuple(
            symbol
            for field in system.fields
            for symbol in field.variables
        )
        directions = tuple(
            symbol
            for field in system.fields
            for symbol in field.directions
        )
        form_evaluation = residual_form_pipeline(
            residual_vector,
            variables,
            directions,
        ).evaluate((FormOrder.ZERO, FormOrder.ONE, FormOrder.TWO))
    else:
        system = form_collection.source
        form_evaluation = form_collection
    return ResidualDimensionEvaluation(
        equation.name,
        form_evaluation,
    )


def _unit_generated_prefix(unit):
    name = _unit_output_name(unit)
    return name


def _unit_output_name(unit):
    return _unit_output_name_from_parts(unit.material_name, unit.unit_name)


def _unit_output_name_from_parts(material_name, unit_name):
    if unit_name:
        return "%s_%s" % (material_name, unit_name)
    return material_name


def _unit_report_name(unit):
    return _unit_output_name(unit)


def _energy_form_orders(kernels):
    order_by_kernel = {
        "objective": FormOrder.ZERO,
        "gradient": FormOrder.ONE,
        "apply": FormOrder.TWO,
    }
    orders = [FormOrder.ZERO]
    for kernel in kernels:
        order = order_by_kernel.get(kernel)
        if order is not None and order not in orders:
            orders.append(order)
    return tuple(orders)


def _summary(name, graph, specialization):
    quadrature = specialization.quadrature_rule
    lines = [
        "# %s Generated Kernel Summary" % name.replace("_", " ").title(),
        "",
        "## Configuration",
        "",
        "- element_type: `%s`" % quadrature.element_type,
        "- quadrature_order: `%d`" % quadrature.order,
        "- dim: `%d`" % quadrature.dim,
        "- n_nodes: `%d`" % quadrature.n_shape,
        "- n_qp: `%d`" % quadrature.n_qp,
        "- vector_size: `%d`" % specialization.vector_size,
        "- outputs: `%d`" % len(graph.outputs),
        "- statements: `%d`" % len(graph.evaluation_plan.statements),
        "- temporaries: `%d`" % len(graph.evaluation_plan.intermediates),
        "- flops: `%d`" % graph.cost.flops,
        "- estimated_registers: `%d`" % graph.cost.estimated_registers,
        "",
        "## Template Parameters",
        "",
    ]
    lines.extend(
        "- `%s = %d` from `%s`" % (parameter.name, parameter.value, parameter.source)
        for parameter in graph.template_parameters
    )
    lines.append("")
    return "\n".join(lines)


def _parse_elements(values, defaults):
    default_entries = tuple(defaults)
    default_by_name = _element_selection_map(default_entries)
    supported = set(sfem_supported_element_types())
    supported.update(default_by_name)
    defaults = tuple(_element_selection_name(element) for element in default_entries)
    enabled = set(defaults)
    enabled.update(default_by_name)
    if not values:
        selected_names = defaults
    else:
        selected_names = []
        for value in values:
            for item in str(value).split(","):
                name = item.strip().upper()
                if not name:
                    continue
                if name == "ALL":
                    selected_names = list(defaults)
                    break
                selected_names.append(name)
        selected_names = tuple(dict.fromkeys(selected_names))

    invalid = tuple(element for element in selected_names if element not in supported)
    if invalid:
        raise ValueError(
            "unsupported element %s; expected one of %s"
            % (", ".join(invalid), ", ".join(sorted(supported)))
        )
    disabled = tuple(element for element in selected_names if element not in enabled)
    if disabled:
        raise ValueError(
            "element %s is not enabled for this material; expected one of %s"
            % (", ".join(disabled), ", ".join(defaults))
        )
    return tuple(default_by_name.get(name, name) for name in selected_names)


def _default_elements_for_systems(systems):
    detected = []
    for system in systems:
        for element in sfem_detect_compatible_element_types(system.fields):
            name = _element_selection_name(element)
            if name not in {_element_selection_name(existing) for existing in detected}:
                detected.append(element)
    if detected:
        return tuple(detected)
    return sfem_supported_element_types()


def _element_selection_map(elements):
    by_name = {}
    for element in elements:
        by_name[_element_selection_name(element)] = element
        if isinstance(element, SfemCompatibleElement):
            by_name[element.cell_element_type] = element
    return by_name


def _element_selection_name(element):
    if isinstance(element, SfemCompatibleElement):
        return element.name
    return str(element).upper()


def _as_equation_systems(systems):
    if isinstance(systems, EquationSystems):
        return systems
    if isinstance(systems, EquationSystem):
        return EquationSystems(systems)
    if isinstance(systems, (tuple, list)):
        return EquationSystems(*systems)
    raise TypeError("CodeGenerator requires EquationSystem or EquationSystems")


def _merge_files(outputs, files):
    for generated in files:
        existing = outputs.get(generated.path)
        if existing is not None and existing != generated.source:
            raise RuntimeError("conflicting generated source for %s" % generated.path)
        outputs[generated.path] = generated.source


def _write_files(out_dir, files):
    paths = []
    for filename, source in sorted(files.items()):
        path = os.path.join(out_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as output:
            output.write(source)
        paths.append(path)
    return tuple(paths)


def _clean_outputs(out_dir, name):
    patterns = (
        "generated_%s*.hpp" % name,
        "generated_%s*.cpp" % name,
        "generated_%s*.o" % name,
        "%s*.hpp" % name,
        "%s*.cpp" % name,
        "%s*.o" % name,
        "%s_*_summary.md" % name,
        "%s_*_reduced_outputs.txt" % name,
        "kernel_math.hpp",
        "kernel_diagnostics.hpp",
        "tensor_product_kernels.hpp",
    )
    for pattern in patterns:
        for path in glob.glob(os.path.join(out_dir, pattern)):
            os.remove(path)
    nested_patterns = (
        "generated_%s*.hpp" % name,
        "generated_%s*.cpp" % name,
        "generated_%s*.o" % name,
        "%s*.hpp" % name,
        "%s*.cpp" % name,
        "%s*.o" % name,
        "%s_*_summary.md" % name,
        "%s_*_reduced_outputs.txt" % name,
        "kernel_math.hpp",
        "kernel_diagnostics.hpp",
        "tensor_product_kernels.hpp",
    )
    for pattern in nested_patterns:
        for path in glob.glob(os.path.join(out_dir, "d*", pattern)):
            os.remove(path)
    for pattern in nested_patterns:
        for path in glob.glob(os.path.join(out_dir, "d*", "*", pattern)):
            os.remove(path)


def _compile_operators(paths):
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("c++ compiler is not available")
    objects = []
    for source in paths:
        if not source.endswith("_operator.cpp"):
            continue
        output = os.path.splitext(source)[0] + ".o"
        subprocess.run(
            [
                compiler,
                "-std=c++14",
                "-O3",
                "-fopenmp-simd",
                "-Werror",
                "-c",
                source,
                "-I",
                os.path.dirname(source),
                "-o",
                output,
            ],
            check=True,
        )
        objects.append(output)
    return tuple(objects)


def _validate_name(name):
    if not isinstance(name, str) or not name or not name.isidentifier():
        raise ValueError("material name must be a valid identifier")


def _validate_op(op_name, parameter_defaults):
    if op_name is not None and (
        not isinstance(op_name, str) or not op_name or not op_name.isidentifier()
    ):
        raise ValueError("op_name must be a valid C++ identifier")
    names = set()
    for name, _ in parameter_defaults:
        name = str(name)
        if not name or not name.isidentifier():
            raise ValueError("parameter names must be valid identifiers")
        if name in names:
            raise ValueError("duplicate parameter '%s'" % name)
        names.add(name)


__all__ = [
    "CoupledResidualSystem",
    "CodegenQualifier",
    "DEFAULT_VECTOR_SIZE",
    "DEFORMATION_GRADIENT",
    "DISPLACEMENT",
    "EquationForm",
    "EquationSystem",
    "EquationSystemBuilder",
    "EquationSystems",
    "FiniteElement",
    "FieldQualifier",
    "Function",
    "FunctionSpace",
    "FormCollection",
    "FormEvaluation",
    "FormMetadata",
    "FormOrder",
    "FormBlock",
    "FormQualifier",
    "GenerationResult",
    "BasisDataLayout",
    "BasisEvaluation",
    "BasisFamily",
    "BasisPlanNode",
    "BlockPlan",
    "DataStreamLayout",
    "DataStreamPlan",
    "DataStreamRole",
    "GenerationPlan",
    "GeometryEvaluation",
    "GeometryInputLayout",
    "GeometryMode",
    "GeometryPlan",
    "GeometryPlanNode",
    "KernelCoupling",
    "KernelPlan",
    "KernelEmission",
    "KernelScope",
    "KernelTarget",
    "LocalKernelPlan",
    "LocalPhase",
    "LocalPhasePlan",
    "MeshKernelPlan",
    "MeshPhase",
    "MeshPhasePlan",
    "OpenMPSoABackend",
    "OPENMP_SOA_BACKEND",
    "current_geometric_dimension",
    "geometric_dimension_context",
    "HyperelasticQualifier",
    "Identity",
    "MATERIAL_PARAMETER",
    "MaterialParameter",
    "MaterialParameterQualifier",
    "MixedFunctionSpace",
    "PRESSURE",
    "PREVIOUS_ARGUMENT",
    "PreviousFunction",
    "QuadratureSetting",
    "QualifiedExpression",
    "ScalarField",
    "SfemElementBasisPolicy",
    "SfemFEMPolicy",
    "SfemFieldFamilyCompatibilityPolicy",
    "SfemReferenceData",
    "StandardFormName",
    "SymbolicArgument",
    "SymbolicField",
    "TEST_ARGUMENT",
    "TensorField",
    "TensorProductDataLayout",
    "TensorProductOperation",
    "TensorProductSumFactorizationPlan",
    "TensorFunction",
    "TestFunction",
    "TRIAL_ARGUMENT",
    "TrialFunction",
    "TwoPhaseFlowConstitutiveModel",
    "CodeGenerator",
    "VELOCITY",
    "VectorField",
    "VectorFunction",
    "VectorElement",
    "VectorFunctionSpace",
    "adjugate",
    "affine_geometry_plan",
    "basis_plan_for_element_at_cell_rule",
    "basis_plan_for_quadrature_rule",
    "basis_plans_for_fem_policy",
    "det",
    "deformation_gradient",
    "derivative",
    "div",
    "field_basis_plan_for_fem_policy",
    "field_basis_plans_for_fem_policy",
    "generate",
    "geometry_plans_for_fem_policy",
    "grad",
    "inner",
    "inv",
    "isoparametric_geometry_plan",
    "material_parameter",
    "matrix_inner",
    "old",
    "previous_function",
    "qualifiers",
    "qualify",
    "run",
    "scalar_field",
    "sfem_cell_rule_points",
    "sfem_default_quadrature_order",
    "sfem_detect_compatible_element_types",
    "sfem_detect_taylor_hood_element_types",
    "sfem_element_quadrature_rule",
    "sfem_field_n_shape",
    "sfem_fem_policy",
    "sfem_is_proteus_hex_element",
    "sfem_is_tensor_product_hex_element",
    "sfem_mesh_reference_data",
    "sfem_normalize_integration_case",
    "sfem_proteus_hex_element_types",
    "sfem_reference_data",
    "sfem_shape_data_for_element_at_cell_rule",
    "sfem_simplex_grad_ref_name",
    "sfem_simplex_field_reference_data",
    "sfem_supported_element_types",
    "sfem_taylor_hood_element_types",
    "sfem_tensor_product_field_reference_data",
    "sfem_tensor_product_hex_order",
    "sfem_tensor_hex_shape_index",
    "streams_in_shape_order",
    "tensor_field",
    "tensor_product_cartesian_shape_order",
    "tensor_product_field_evaluation_plan",
    "tensor_product_geometry_jacobian_plan",
    "tensor_product_geometry_jacobian_plan_from_sizes",
    "tensor_product_sum_factorization_plan",
    "tensor_product_test_contraction_plan",
    "test_function",
    "trial_function",
    "vector_field",
    "variable",
    "value",
]
