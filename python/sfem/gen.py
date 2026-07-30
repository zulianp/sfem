import argparse
import glob
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from enum import Enum

import sympy as sp

from ._gen_op import generate_op_files, generate_op_registration_files
from codegen.framework import (
    CodegenQualifier,
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
    SpatialCoordinate,
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
    FormDependencies,
    FormEvaluation,
    FormKind,
    FormMetadata,
    FormOrder,
    FormQualifier,
    BoundaryIntegral,
    Measure,
    dx,
    ds,
    emission_plan_for_element,
    emission_plan_from_unit_context,
    GeometryEvaluation,
    GeometryInputLayout,
    GeometryMode,
    GeometryPlan,
    GeometryPlanNode,
    KernelCoupling,
    KernelExpressionPlan,
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
from codegen.framework.plans.emission import ElementEmissionPlan as _ElementEmissionPlan
from codegen.framework.plans.generation import GenerationPlan as _GenerationPlan
from codegen.framework.plans.matrix_formats import (
    MatrixAssemblyVariantPlan,
    MatrixFormat,
    MatrixFormatPlan,
    MatrixMeshLayout,
    PackedAssemblyPass,
    matrix_format_plan_from_request,
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
from codegen.framework.backends.cuda import CUDASoABackend as _CUDASoABackend
from codegen.framework.backends.openmp import OpenMPSoABackend as _OpenMPSoABackend
from codegen.framework.backends.targets import (
    AVX512Target,
    ARMSMETarget,
    ARMSVETarget,
    HIPTarget,
)


DEFAULT_VECTOR_SIZE = 16
OPENMP_SOA_BACKEND = _OpenMPSoABackend()
CUDA_SOA_BACKEND = _CUDASoABackend()
AVX512_SOA_BACKEND = _OpenMPSoABackend(target=AVX512Target())
ARM_SVE_SOA_BACKEND = _OpenMPSoABackend(target=ARMSVETarget())
ARM_SME_SOA_BACKEND = _OpenMPSoABackend(target=ARMSMETarget())
HIP_SOA_BACKEND = _CUDASoABackend(target=HIPTarget())
BACKENDS_BY_TARGET = {
    KernelTarget.OPENMP: OPENMP_SOA_BACKEND,
    KernelTarget.AVX512: AVX512_SOA_BACKEND,
    KernelTarget.ARM_SVE: ARM_SVE_SOA_BACKEND,
    KernelTarget.ARM_SME: ARM_SME_SOA_BACKEND,
    KernelTarget.CUDA: CUDA_SOA_BACKEND,
    KernelTarget.HIP: HIP_SOA_BACKEND,
}

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
    matrix_formats: tuple = ()
    matrix_mesh_layouts: tuple = ("standard",)
    matrix_packed_passes: tuple = ("one_pass", "two_pass")
    matrix_patch_node_index_filter: bool = False

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
        object.__setattr__(
            self,
            "matrix_format_plan",
            matrix_format_plan_from_request(
                self.matrix_formats,
                self.matrix_mesh_layouts,
                self.matrix_packed_passes,
                self.matrix_patch_node_index_filter,
            ),
        )
        _validate_op(self.op_name, self.parameter_defaults)


@dataclass(frozen=True)
class GenerationResult:
    sources: tuple
    objects: tuple = ()
    plan: object = None
    plan_dump: object = None


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
            (
                affine_geometry_plan(affine_policy),
                isoparametric_geometry_plan(policy),
            ),
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
    matrix_format_plan: object = None

    @property
    def stage(self):
        return PipelineStage.USER_INPUT

    @classmethod
    def create(
        cls,
        material,
        elements,
        vector_size,
        quadrature_order,
        matrix_format_plan=None,
    ):
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
        if matrix_format_plan is None:
            matrix_format_plan = getattr(material, "matrix_format_plan", None)
        return cls(
            material,
            tuple(elements),
            vector_size,
            quadrature_order,
            tuple(contexts),
            matrix_format_plan,
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
        residual_system = system.form_collection(equation).source
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
class LoweredEquationEvaluation:
    name: str
    form_evaluation: FormCollection
    data_symbols: object = None
    kernels: tuple = ()
    diagnostics: bool = True
    matrix_format_plan: object = None


@dataclass(frozen=True)
class DimensionFormEvaluation:
    dim: int
    units: tuple


@dataclass(frozen=True)
class UnifiedFormEvaluation:
    material: object
    by_dim: dict
    matrix_format_plan: object = None

    @property
    def stage(self):
        return PipelineStage.FORM_EVALUATION


class CodeGenerationKind(Enum):
    ENERGY_SOA = "energy_soa"
    RESIDUAL_SOA = "residual_soa"
    BOUNDARY_RESIDUAL_SOA = "boundary_residual_soa"


@dataclass(frozen=True)
class CodeGenerationUnit(KernelPlan):
    material_name: str = ""
    unit_name: str = ""


CodeGenerationPlan = _GenerationPlan


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
    target: object = "openmp"

    @property
    def stage(self):
        return PipelineStage.CODE_GENERATION

    def run(self):
        outputs = {}
        target = _normalize_generation_target(self.target)
        for context in self.user_input.element_contexts:
            for unit in self.codegen_plan.emission_kernels_for_context(context):
                _merge_files(
                    outputs,
                    _layout_codegen_files(
                        unit,
                        context,
                        _emit_codegen_unit(unit, context, target),
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
    target="openmp",
    matrix_formats=None,
    matrix_mesh_layouts=None,
    matrix_packed_passes=None,
    matrix_patch_node_index_filter=None,
):
    vector_size = int(vector_size)
    if vector_size <= 0:
        raise ValueError("vector_size must be positive")

    available_elements = _generation_available_elements(material.elements)
    selected = _parse_elements(elements, available_elements)
    selected = _with_tensor_product_proteus_alias_dependencies(selected, available_elements)
    out_dir = os.path.abspath(os.fspath(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    if clean:
        _clean_outputs(out_dir, material.name)

    matrix_format_plan = _selected_matrix_format_plan(
        material,
        matrix_formats,
        matrix_mesh_layouts,
        matrix_packed_passes,
        matrix_patch_node_index_filter,
    )
    target = _normalize_generation_target(target)
    backend = _backend_for_target(target)
    while True:
        user_input = UserInputStage.create(
            material,
            selected,
            vector_size,
            quadrature_order,
            matrix_format_plan,
        )
        form_evaluation = _evaluate_forms(user_input)
        codegen_plan = SpecializedFormManipulationStage(
            user_input,
            form_evaluation,
        ).run()
        files = CodeGenerationStage(user_input, codegen_plan, target).run()
        dependency = _missing_hex8_proteus_implementation_dependency(files, selected)
        if dependency is None:
            break
        selected = selected + (dependency,)
    plan_dump = _write_plan_dump(codegen_plan, out_dir, material.name, plan_out, user_input) if dump_plan or plan_out else None

    if material.op_name and backend.supports_op_wrapper:
        files.update(_generate_op_wrapper_files(material, selected, user_input, files))
        _replace_legacy_tensor_product_sources_with_proteus_aliases(files)

    files = _relocate_generated_primitive_headers(files, out_dir, material.name)
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
        "--target",
        choices=("openmp", "avx512", "arm_sve", "arm_sme", "cuda", "hip"),
        default="openmp",
        help="Backend target to emit.",
    )
    parser.add_argument(
        "--dump-plan",
        action="store_true",
        help="Write a JSON generation-plan dump next to generated sources.",
    )
    parser.add_argument(
        "--matrix-format",
        action="append",
        dest="matrix_formats",
        help="Matrix assembly format to emit: crs, bsr, dia, coo, patch, or all. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--matrix-layout",
        action="append",
        dest="matrix_mesh_layouts",
        help="Matrix assembly mesh layout: standard, packed, or all. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--packed-pass",
        action="append",
        dest="matrix_packed_passes",
        help="Packed mesh assembly pass: one_pass, two_pass, or all. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--patch-node-index-filter",
        action="store_true",
        help="Emit patch assembly metadata with node-index filtering enabled.",
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
            target=args.target,
            matrix_formats=args.matrix_formats,
            matrix_mesh_layouts=args.matrix_mesh_layouts,
            matrix_packed_passes=args.matrix_packed_passes,
            matrix_patch_node_index_filter=args.patch_node_index_filter,
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


def _write_plan_dump(plan, out_dir, material_name, plan_out=None, user_input=None):
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
    if user_input is None:
        plan.write_json(path)
    else:
        with open(path, "w", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    _specialized_plan_dump(plan, user_input),
                    indent=2,
                    sort_keys=True,
                )
            )
            output.write("\n")
    return path


def _specialized_plan_dump(plan, user_input):
    kernels = []
    for context in user_input.element_contexts:
        kernels.extend(
            unit.to_dict(include_block_kernels=False)
            for unit in plan.emission_kernels_for_context(context)
        )
    monolithic = [
        kernel
        for kernel in kernels
        if kernel["scope"] == KernelScope.MONOLITHIC.value
    ]
    blocks = [
        kernel
        for kernel in kernels
        if kernel["scope"] == KernelScope.BLOCK.value
    ]
    complete_system = [
        kernel
        for kernel in kernels
        if kernel["coupling"] == KernelCoupling.COMPLETE_SYSTEM.value
    ]
    return {
        "stage": plan.stage.value,
        "n_kernels": len(kernels),
        "n_monolithic_kernels": len(monolithic),
        "n_block_kernels": len(blocks),
        "n_complete_system_kernels": len(complete_system),
        "kernels": kernels,
    }


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
                user_input.matrix_format_plan,
            )
            for equation in material_system.equations
        )
        if not units:
            raise ValueError("material '%s' did not define any equations" % user_input.material.name)
        by_dim[dim] = DimensionFormEvaluation(dim, units)
    return UnifiedFormEvaluation(
        user_input.material,
        by_dim,
        user_input.matrix_format_plan,
    )


def _codegen_plan_from_form_evaluation(form_evaluation):
    units = []
    for dim_eval in form_evaluation.by_dim.values():
        for evaluated in dim_eval.units:
            if evaluated.form_evaluation.kind is FormKind.ENERGY:
                units.append(
                    _energy_codegen_unit(
                        form_evaluation.material.name,
                        dim_eval.dim,
                        evaluated,
                    )
                )
            elif evaluated.form_evaluation.kind is FormKind.RESIDUAL:
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
    return _GenerationPlan(tuple(units))


def _energy_codegen_unit(material_name, dim, evaluated):
    if evaluated.data_symbols.shape != (dim, dim):
        raise ValueError(
            "energy code generation currently requires %d x %d explicit variables; "
            "got %d variables for energy unit '%s'"
            % (dim, dim, len(evaluated.form_evaluation.variables), evaluated.name or material_name)
        )
    weak_form = sfem_soa_weak_form(
        evaluated.form_evaluation.form(FormOrder.ZERO).expression,
        evaluated.data_symbols,
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
        expression_plans=_energy_expression_plans(
            kernel_forms,
            diagnostic_graph,
            evaluated.form_evaluation,
        ),
        matrix_format_plan=_matrix_format_plan_for_evaluation(evaluated),
        coupling=KernelCoupling.SINGLE_FIELD,
        material_name=material_name,
        unit_name=evaluated.name,
    )


def _residual_codegen_unit(material_name, dim, evaluated):
    collection = evaluated.form_evaluation
    blocks = _block_plans_from_form_collection(collection)
    coupling = _kernel_coupling_for_collection(collection)
    if evaluated.form_evaluation.measure == "ds":
        return CodeGenerationUnit(
            name=_unit_output_name_from_parts(material_name, evaluated.name),
            kind=CodeGenerationKind.BOUNDARY_RESIDUAL_SOA,
            form_collection=collection,
            dim=dim,
            mesh_phases=_residual_mesh_phases(),
            mesh_phase_plans=_residual_mesh_phase_plans(blocks),
            blocks=blocks,
            expression_plans=_form_collection_expression_plans(collection),
            matrix_format_plan=_matrix_format_plan_for_evaluation(evaluated),
            target=KernelTarget.OPENMP,
            coupling=coupling,
            material_name=material_name,
            unit_name=evaluated.name,
        )
    block_kernels = (
        _block_codegen_units(material_name, dim, evaluated, blocks)
        if coupling is not KernelCoupling.SINGLE_FIELD
        else ()
    )
    return CodeGenerationUnit(
        name=_unit_output_name_from_parts(material_name, evaluated.name),
        kind=CodeGenerationKind.RESIDUAL_SOA,
        form_collection=collection,
        dim=dim,
        mesh_phases=_residual_mesh_phases(),
        mesh_phase_plans=_residual_mesh_phase_plans(blocks),
        blocks=blocks,
        block_kernels=block_kernels,
        scope=KernelScope.MONOLITHIC,
        coupling=coupling,
        expression_plans=_form_collection_expression_plans(collection),
        matrix_format_plan=_matrix_format_plan_for_evaluation(evaluated),
        target=KernelTarget.OPENMP,
        material_name=material_name,
        unit_name=evaluated.name,
    )


def _energy_expression_plans(kernel_forms, diagnostic_graph, collection):
    plans = []
    for kernel_form in kernel_forms:
        form_order = _form_order_for_kernel_name(kernel_form.name)
        dependencies = _metadata_dependencies(collection, form_order)
        source = replace(kernel_form, dependencies=dependencies)
        plans.append(
            KernelExpressionPlan(
                name=kernel_form.name,
                form_order=form_order,
                role=_role_for_form(collection, form_order),
                expression_graph=kernel_form.expression_graph,
                weak_form=kernel_form.weak_form,
                coefficients=collection.coefficients,
                dependencies=dependencies,
                diagnostics=diagnostic_graph if kernel_form.name == "apply" else None,
                fields=collection.fields,
                blocks=_metadata_blocks(collection, form_order),
                source=source,
                output_mode=kernel_form.output_mode,
                has_direction=kernel_form.has_direction,
            )
        )
    return tuple(plans)


def _form_collection_expression_plans(collection, block=None):
    plans = []
    for form in collection.forms:
        blocks = _metadata_blocks(collection, form.order)
        coefficients = _metadata_coefficients(collection, form.order)
        if block is not None:
            blocks = tuple(
                candidate
                for candidate in blocks
                if candidate.name == block.name
            )
            coefficients = tuple(
                coefficient
                for selected_block in blocks
                for coefficient in selected_block.coefficients
            )
        plans.append(
            KernelExpressionPlan(
                name=form.name,
                form_order=form.order,
                role=form.role,
                expression_graph=form.expression,
                coefficients=coefficients,
                dependencies=_metadata_dependencies(collection, form.order),
                fields=collection.fields,
                blocks=blocks,
                source=form,
            )
        )
    return tuple(plans)


def _form_order_for_kernel_name(name):
    if name == "objective":
        return FormOrder.ZERO
    if name == "gradient":
        return FormOrder.ONE
    if name == "apply":
        return FormOrder.TWO
    raise ValueError("unsupported energy kernel form '%s'" % name)


def _role_for_form(collection, order):
    return collection.form(order).role


def _metadata_for_order(collection, order):
    try:
        return collection.form_metadata(order)
    except ValueError:
        return None


def _metadata_coefficients(collection, order):
    metadata = _metadata_for_order(collection, order)
    if metadata is None:
        return collection.coefficients
    return metadata.coefficients


def _metadata_dependencies(collection, order):
    metadata = _metadata_for_order(collection, order)
    if metadata is None:
        return collection.dependencies
    return metadata.dependencies


def _metadata_blocks(collection, order):
    metadata = _metadata_for_order(collection, order)
    if metadata is None:
        return ()
    return metadata.blocks


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


def _residual_mesh_phase_plans(blocks):
    return (
        MeshPhasePlan(MeshPhase.GATHER),
        MeshPhasePlan(MeshPhase.GEOMETRY),
        MeshPhasePlan(MeshPhase.LOCAL_CALL, blocks=tuple(blocks)),
        MeshPhasePlan(MeshPhase.SCATTER),
    )


def _residual_mesh_phases():
    return tuple(plan.phase for plan in _residual_mesh_phase_plans(()))


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
            mesh_phases=_residual_mesh_phases(),
            mesh_phase_plans=_residual_mesh_phase_plans((block,)),
            blocks=(block,),
            scope=KernelScope.BLOCK,
            coupling=KernelCoupling.BLOCK,
            block=block,
            emission=KernelEmission.FILES,
            target=KernelTarget.OPENMP,
            expression_plans=_form_collection_expression_plans(
                evaluated.form_evaluation,
                block,
            ),
            matrix_format_plan=_matrix_format_plan_for_evaluation(evaluated),
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


def _emit_codegen_unit(unit, context, target=KernelTarget.OPENMP):
    backend = _backend_for_target(target)
    return tuple(backend.emit(unit, context))


def _backend_for_target(target):
    target = _normalize_generation_target(target)
    try:
        return BACKENDS_BY_TARGET[target]
    except KeyError as exc:
        raise ValueError("unsupported code generation target %s" % target) from exc


def _normalize_generation_target(target):
    if isinstance(target, KernelTarget):
        return target
    return KernelTarget(str(target).lower())


def _selected_matrix_format_plan(
    material,
    matrix_formats,
    matrix_mesh_layouts,
    matrix_packed_passes,
    matrix_patch_node_index_filter,
):
    material_plan = getattr(material, "matrix_format_plan", None)
    if matrix_formats is None:
        return material_plan
    mesh_layouts = (
        getattr(material, "matrix_mesh_layouts", ("standard",))
        if matrix_mesh_layouts is None
        else matrix_mesh_layouts
    )
    packed_passes = (
        getattr(material, "matrix_packed_passes", ("one_pass", "two_pass"))
        if matrix_packed_passes is None
        else matrix_packed_passes
    )
    patch_filter = (
        getattr(material, "matrix_patch_node_index_filter", False)
        if matrix_patch_node_index_filter is None
        else matrix_patch_node_index_filter
    )
    return matrix_format_plan_from_request(
        matrix_formats,
        mesh_layouts,
        packed_passes,
        patch_filter,
    )


def _matrix_format_plan_for_evaluation(evaluated):
    plan = getattr(evaluated, "matrix_format_plan", None)
    if plan is None or plan.is_empty:
        return None
    if not any(form.order is FormOrder.TWO for form in evaluated.form_evaluation.forms):
        return None
    return plan


def _replace_legacy_tensor_product_sources_with_proteus_aliases(files):
    c_abi_entries = [
        (path, source)
        for path, source in files.items()
        if path.endswith("_c_abi.hpp") and path.startswith("op/")
    ]
    if not c_abi_entries:
        return
    dispatch_declarations = []
    for path, source in files.items():
        if path.endswith("_dispatch.cpp") and path.startswith("op/"):
            dispatch_declarations.extend(_extern_c_declarations(source))

    aliases = (
        _tensor_product_proteus_alias(
            element_name="quad4",
            proteus_name="proteus_quad4",
            dim=2,
            n_shape=4,
        ),
        _tensor_product_proteus_alias(
            element_name="hex8",
            proteus_name="proteus_hex8",
            dim=3,
            n_shape=8,
        ),
        _tensor_product_proteus_alias(
            element_name="hex27",
            proteus_name="proteus_hex27",
            dim=3,
            n_shape=27,
        ),
    )
    for c_abi_path, c_abi_source in c_abi_entries:
        declarations = _unique_extern_c_declarations(
            tuple(_extern_c_declarations(c_abi_source)) + tuple(dispatch_declarations)
        )
        names = {declaration["name"] for declaration in declarations}
        for alias in aliases:
            alias_declarations = [
                declaration
                for declaration in declarations
                if _tensor_product_proteus_target_name(declaration["name"], alias) in names
            ]
            if not alias_declarations:
                continue

            for source_path in tuple(files):
                if not (
                    source_path.startswith(alias["source_prefix"])
                    and any(source_path.endswith(suffix) for suffix in alias["source_suffixes"])
                ):
                    continue
                proteus_path = source_path.replace(
                    alias["source_prefix"],
                    alias["target_prefix"],
                )
                for source_suffix, target_suffix in alias["suffix_pairs"]:
                    if proteus_path.endswith(source_suffix):
                        proteus_path = proteus_path[: -len(source_suffix)] + target_suffix
                        break
                if proteus_path not in files:
                    continue
                files[source_path] = _tensor_product_proteus_alias_source(
                    source_path,
                    c_abi_path,
                    alias_declarations,
                    alias,
                )


def _tensor_product_proteus_alias(element_name, proteus_name, dim, n_shape):
    return {
        "element_name": element_name,
        "proteus_name": proteus_name,
        "n_shape": int(n_shape),
        "shape_order": tensor_product_cartesian_shape_order(int(dim), int(n_shape)),
        "source_prefix": "d%d/%s/" % (int(dim), element_name),
        "target_prefix": "d%d/%s/" % (int(dim), proteus_name),
        "source_suffixes": (
            "_%s_operator.cpp" % element_name,
            "_%s_boundary_operator.cpp" % element_name,
        ),
        "suffix_pairs": (
            ("_%s_operator.cpp" % element_name, "_%s_operator.cpp" % proteus_name),
            ("_%s_boundary_operator.cpp" % element_name, "_%s_boundary_operator.cpp" % proteus_name),
        ),
    }


def _extern_c_declarations(source):
    pattern = re.compile(
        r'extern "C"\s+(?P<head>.*?)\((?P<params>.*?)\);',
        re.DOTALL,
    )
    declarations = []
    for match in pattern.finditer(source):
        head = match.group("head").strip()
        name_match = re.search(r"([A-Za-z_]\w*)\s*$", head)
        if name_match is None:
            continue
        declarations.append(
            {
                "return_type": head[: name_match.start()].rstrip(),
                "name": name_match.group(1),
                "params": match.group("params").strip(),
            }
        )
    return tuple(declarations)


def _tensor_product_proteus_alias_source(source_path, c_abi_path, declarations, alias):
    include_path = _relative_codegen_include(os.path.dirname(source_path), c_abi_path)
    declarations = tuple(
        declaration
        for declaration in declarations
        if "_matrix_assembly_" not in declaration["name"]
    )
    lines = [
        '#include "%s"' % include_path,
        "",
    ]
    target_declarations = []
    emitted_targets = set()
    for declaration in declarations:
        target_name = _tensor_product_proteus_target_name(declaration["name"], alias)
        if not target_name or target_name in emitted_targets:
            continue
        emitted_targets.add(target_name)
        target = dict(declaration)
        target["name"] = target_name
        target_declarations.append(target)
    for declaration in target_declarations:
        lines.extend(_extern_c_prototype_lines(declaration))
    if target_declarations:
        lines.append("")
    for declaration in declarations:
        lines.extend(
            _tensor_product_proteus_alias_function(
                declaration,
                alias,
            )
        )
    return "\n".join(lines)


def _tensor_product_proteus_alias_function(declaration, alias):
    return_type = declaration["return_type"]
    name = declaration["name"]
    target = _tensor_product_proteus_target_name(name, alias)
    params = _c_params(declaration["params"])
    lines = ['extern "C" %s %s(' % (return_type, name)]
    if params:
        for idx, param in enumerate(params):
            comma = "," if idx + 1 < len(params) else ""
            lines.append("        %s%s" % (param, comma))
    else:
        lines.append("        void")
    lines.append(") {")

    args = [_c_param_name(param) for param in params]
    element_index = _c_element_pointer_param_index(params)
    if element_index is not None:
        pointer_type = _c_element_pointer_type(params[element_index])
        shape_order = alias["shape_order"]
        lines.append("    %s *proteus_elements[%d] = {" % (pointer_type, alias["n_shape"]))
        for idx, source_index in enumerate(shape_order):
            comma = "," if idx + 1 < len(shape_order) else ""
            lines.append("        elements[%d]%s" % (source_index, comma))
        lines.append("    };")
        args[element_index] = "proteus_elements"

    call = "%s(%s)" % (target, ", ".join(args))
    if return_type == "void":
        lines.append("    %s;" % call)
    else:
        lines.append("    return %s;" % call)
    lines.extend(["}", ""])
    return lines


def _extern_c_prototype_lines(declaration):
    return_type = declaration["return_type"]
    name = declaration["name"]
    params = _c_params(declaration["params"])
    lines = ['extern "C" %s %s(' % (return_type, name)]
    if params:
        for idx, param in enumerate(params):
            comma = "," if idx + 1 < len(params) else ""
            lines.append("        %s%s" % (param, comma))
    else:
        lines.append("        void")
    lines.append(");")
    return lines


def _unique_extern_c_declarations(declarations):
    unique = []
    seen = set()
    for declaration in declarations:
        name = declaration["name"]
        if name in seen:
            continue
        seen.add(name)
        unique.append(declaration)
    return tuple(unique)


def _tensor_product_proteus_target_name(name, alias):
    element = alias["element_name"]
    proteus = alias["proteus_name"]
    if "_%s_%s_" % (element, element) in name:
        return name.replace(
            "_%s_%s_" % (element, element),
            "_%s_%s_" % (proteus, proteus),
        )
    if "_%s_" % element in name:
        return name.replace("_%s_" % element, "_%s_" % proteus)
    return ""


def _c_params(params):
    params = params.strip()
    if not params or params == "void":
        return ()
    return tuple(param.strip() for param in params.split(",") if param.strip())


def _c_param_name(param):
    tokens = param.replace("*", " * ").replace("&", " & ").split()
    if not tokens:
        raise ValueError("empty C parameter")
    return tokens[-1].split("[", 1)[0]


def _c_element_pointer_param_index(params):
    for idx, param in enumerate(params):
        if _c_param_name(param) == "elements" and "**" in param:
            return idx
    return None


def _c_element_pointer_type(param):
    if "uint16_t" in param:
        return "uint16_t"
    if "idx_t" in param:
        return "idx_t"
    raise ValueError("unsupported HEX27 element pointer parameter '%s'" % param)


def _layout_codegen_files(unit, context, files):
    local_headers = tuple(
        generated.path
        for generated in files
        if _is_codegen_local_header(generated.path)
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
    if _is_codegen_local_header(filename):
        return _codegen_dimension_directory(context)
    return _codegen_output_directory(unit, context)


def _is_codegen_local_header(filename):
    return (
        filename.endswith("_local.hpp")
        or filename.endswith("_local.cuh")
        or filename.endswith("_hessian.hpp")
        or filename.endswith("_hessian.cuh")
    )


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
        "kernel_math.cuh",
        "kernel_diagnostics.hpp",
        "kernel_diagnostics.cuh",
        "matrix_formats.hpp",
        "packed_thread_scratch.hpp",
        "tensor_product_kernels.hpp",
        "tensor_product_kernels.cuh",
        "geometry_kernels.hpp",
        "geometry_kernels.cuh",
    )
)

_CODEGEN_SHARED_PRIMITIVE_HEADERS = frozenset(
    (
        "kernel_math.hpp",
        "kernel_math.cuh",
        "kernel_diagnostics.hpp",
        "kernel_diagnostics.cuh",
        "packed_thread_scratch.hpp",
        "tensor_product_kernels.hpp",
        "tensor_product_kernels.cuh",
        "geometry_kernels.hpp",
        "geometry_kernels.cuh",
    )
)


def _relative_codegen_include(directory, target):
    if not directory:
        return target.replace(os.sep, "/")
    return os.path.relpath(target, start=directory).replace(os.sep, "/")


def _relocate_generated_primitive_headers(files, out_dir, material_name):
    if not _uses_generated_shared_primitive_headers(out_dir, material_name):
        return files

    relocated = {}
    header_targets = {}
    for header in _CODEGEN_SHARED_PRIMITIVE_HEADERS:
        if header in files:
            header_targets[header] = os.path.join("..", header)

    if not header_targets:
        return files

    for filename, source in files.items():
        target = header_targets.get(filename, filename)
        rewritten = _rewrite_generated_primitive_includes(filename, source, header_targets)
        existing = relocated.get(target)
        if existing is not None and existing != rewritten:
            raise RuntimeError("conflicting generated source for %s" % target)
        relocated[target] = rewritten
    return relocated


def _uses_generated_shared_primitive_headers(out_dir, material_name):
    normalized = os.path.normpath(os.path.abspath(out_dir))
    output_name = os.path.basename(normalized)
    generated_root = os.path.basename(os.path.dirname(normalized))
    return generated_root == "generated" and output_name in (
        material_name,
        "%s_cuda" % material_name,
        "%s_hip" % material_name,
    )


def _rewrite_generated_primitive_includes(filename, source, header_targets):
    directory = os.path.dirname(filename)
    rewritten = source
    for header, target in header_targets.items():
        local_include = _relative_codegen_include(directory, header)
        shared_include = _relative_codegen_include(directory, target)
        if local_include == shared_include:
            continue
        rewritten = rewritten.replace(
            '#include "%s"' % local_include,
            '#include "%s"' % shared_include,
        )
    return rewritten


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


def _generate_op_wrapper_files(material, selected, user_input, kernel_sources):
    if not user_input.element_contexts:
        raise ValueError("generated Op wrapper requires at least one element context")
    representative_dim = user_input.element_contexts[0].specialization.dim
    equations = _material_equations(material, representative_dim)
    if len(equations) == 1 and equations[0].name:
        raise ValueError(
            "single-equation generated Op wrappers require an unnamed equation"
        )
    return generate_op_files(material, selected, kernel_sources)


def _evaluate_equation(dim, equation, form_collection, matrix_format_plan=None):
    if not isinstance(form_collection, FormCollection):
        raise TypeError("equation evaluation requires a lowered FormCollection")
    if equation.is_energy:
        if form_collection.kind is not FormKind.ENERGY:
            raise TypeError("energy equation requires an energy FormCollection")
        variables = tuple(form_collection.variables)
        if not variables:
            raise ValueError("energy equation '%s' requires explicit variables" % equation.name)
        data_symbols = _energy_data_symbols(dim, variables)
        return LoweredEquationEvaluation(
            equation.name,
            form_collection,
            data_symbols=data_symbols,
            kernels=equation.kernels,
            diagnostics=equation.diagnostics,
            matrix_format_plan=matrix_format_plan if "apply" in equation.kernels else None,
        )
    if equation.is_residual:
        if form_collection.kind is not FormKind.RESIDUAL:
            raise TypeError("residual equation requires a residual FormCollection")
        return LoweredEquationEvaluation(
            equation.name,
            form_collection,
            matrix_format_plan=matrix_format_plan,
        )
    raise TypeError("unsupported equation form %s" % equation.form)


def _equation_form_orders(equation):
    if equation.is_energy:
        return _energy_form_orders(equation.kernels)
    if equation.is_residual:
        return (FormOrder.ZERO, FormOrder.ONE, FormOrder.TWO)
    raise TypeError("unsupported equation form %s" % equation.form)


def _energy_data_symbols(dim, variables):
    if len(variables) == dim * dim:
        return sp.Matrix(dim, dim, variables)
    return sp.Matrix(len(variables), 1, variables)


def _unit_generated_prefix(unit):
    name = _unit_output_name(unit)
    return name


def _unit_output_name(unit):
    return _unit_output_name_from_parts(unit.material_name, unit.unit_name)


def _unit_output_name_from_parts(material_name, unit_name):
    if unit_name:
        return "%s_%s" % (material_name, unit_name)
    return material_name


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


def _generation_available_elements(elements):
    available = tuple(elements or sfem_supported_element_types())
    additions = []
    for element, proteus in (("QUAD4", "PROTEUS_QUAD4"), ("HEX8", "PROTEUS_HEX8"), ("HEX27", "PROTEUS_HEX27")):
        if element in available and proteus not in available:
            additions.append(proteus)
    return available + tuple(additions)


def _with_tensor_product_proteus_alias_dependencies(selected, available):
    selected = tuple(selected)
    additions = []
    selected_names = {_element_selection_name(element) for element in selected}
    available_by_name = {
        _element_selection_name(element): element
        for element in available
    }
    for element, proteus in (("QUAD4", "PROTEUS_QUAD4"), ("HEX8", "PROTEUS_HEX8"), ("HEX27", "PROTEUS_HEX27")):
        if element in selected_names and proteus in available_by_name and proteus not in selected_names:
            additions.append(available_by_name[proteus])
    return selected + tuple(additions)


def _missing_hex8_proteus_implementation_dependency(files, selected):
    selected_names = {_element_selection_name(element) for element in selected}
    if "HEX27_HEX8" not in selected_names or "PROTEUS_HEX27_PROTEUS_HEX8" in selected_names:
        return None
    alias = _tensor_product_proteus_alias(
        element_name="hex8",
        proteus_name="proteus_hex8",
        dim=3,
        n_shape=8,
    )
    for source_path in files:
        if not (
            source_path.startswith(alias["source_prefix"])
            and any(source_path.endswith(suffix) for suffix in alias["source_suffixes"])
        ):
            continue
        proteus_path = source_path.replace(
            alias["source_prefix"],
            alias["target_prefix"],
        )
        for source_suffix, target_suffix in alias["suffix_pairs"]:
            if proteus_path.endswith(source_suffix):
                proteus_path = proteus_path[: -len(source_suffix)] + target_suffix
                break
        if proteus_path not in files:
            return _proteus_hex27_hex8_element()
    return None


def _proteus_hex27_hex8_element():
    return SfemCompatibleElement(
        "PROTEUS_HEX27_PROTEUS_HEX8",
        "PROTEUS_HEX27",
        (
            ("displacement", "PROTEUS_HEX27"),
            ("velocity", "PROTEUS_HEX27"),
            ("pressure", "PROTEUS_HEX8"),
        ),
    )


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
            output.write(source.rstrip() + "\n")
        paths.append(path)
    return tuple(paths)


def _clean_outputs(out_dir, name):
    patterns = (
        "generated_%s*.hpp" % name,
        "generated_%s*.cuh" % name,
        "generated_%s*.cpp" % name,
        "generated_%s*.cu" % name,
        "generated_%s*.hip" % name,
        "generated_%s*.o" % name,
        "%s*.hpp" % name,
        "%s*.cuh" % name,
        "%s*.cpp" % name,
        "%s*.cu" % name,
        "%s*.hip" % name,
        "%s*.o" % name,
        "%s_*_summary.md" % name,
        "%s_*_reduced_outputs.txt" % name,
        "kernel_math.hpp",
        "kernel_math.cuh",
        "kernel_diagnostics.hpp",
        "kernel_diagnostics.cuh",
        "matrix_formats.hpp",
        "packed_thread_scratch.hpp",
        "tensor_product_kernels.hpp",
        "tensor_product_kernels.cuh",
        "geometry_kernels.hpp",
        "geometry_kernels.cuh",
    )
    for pattern in patterns:
        for path in glob.glob(os.path.join(out_dir, pattern)):
            os.remove(path)
    nested_patterns = (
        "generated_%s*.hpp" % name,
        "generated_%s*.cuh" % name,
        "generated_%s*.cpp" % name,
        "generated_%s*.cu" % name,
        "generated_%s*.hip" % name,
        "generated_%s*.o" % name,
        "%s*.hpp" % name,
        "%s*.cuh" % name,
        "%s*.cpp" % name,
        "%s*.cu" % name,
        "%s*.hip" % name,
        "%s*.o" % name,
        "%s_*_summary.md" % name,
        "%s_*_reduced_outputs.txt" % name,
        "kernel_math.hpp",
        "kernel_math.cuh",
        "kernel_diagnostics.hpp",
        "kernel_diagnostics.cuh",
        "matrix_formats.hpp",
        "packed_thread_scratch.hpp",
        "tensor_product_kernels.hpp",
        "tensor_product_kernels.cuh",
        "geometry_kernels.hpp",
        "geometry_kernels.cuh",
    )
    for pattern in nested_patterns:
        for path in glob.glob(os.path.join(out_dir, "d*", pattern)):
            os.remove(path)
    for pattern in nested_patterns:
        for path in glob.glob(os.path.join(out_dir, "d*", "*", pattern)):
            os.remove(path)
    for pattern in ("sfem_*.hpp", "sfem_*.cuh", "sfem_*.cpp", "sfem_*.cu", "sfem_*.o", "sfem_*_manifest.json"):
        for path in glob.glob(os.path.join(out_dir, "op", pattern)):
            os.remove(path)


def _compile_operators(paths):
    compiler = _operator_compiler()
    if compiler is None:
        raise RuntimeError("C++ compiler is not available")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    include_dirs = [
        os.path.join(repo_root, "base"),
    ]
    include_dirs.extend(_smesh_source_include_dirs(repo_root))
    include_dirs.extend(_compile_config_include_dirs(repo_root))
    objects = []
    for source in paths:
        if not source.endswith("_operator.cpp"):
            continue
        include_flags = []
        for include_dir in include_dirs:
            include_flags.extend(("-I", include_dir))
        output = os.path.splitext(source)[0] + ".o"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O3",
                "-fopenmp-simd",
                "-Werror",
                "-c",
                source,
                "-I",
                os.path.dirname(source),
                *include_flags,
                "-o",
                output,
            ],
            check=True,
        )
        objects.append(output)
    return tuple(objects)


def _operator_compiler():
    requested = os.environ.get("CXX")
    if requested:
        return shutil.which(requested) or requested
    return shutil.which("mpic++") or shutil.which("mpicxx") or shutil.which("c++")


def _compile_config_include_dirs(repo_root):
    requested = os.environ.get("SFEM_BUILD_DIR")
    build_dirs = []
    if requested:
        build_dirs.append(os.path.abspath(requested))
    build_dirs.extend(
        os.path.join(repo_root, name)
        for name in ("build64", "build", "build_test", "build_serial")
    )
    build_dirs.extend(sorted(glob.glob(os.path.join(repo_root, "build*"))))

    include_dirs = []
    seen = set()
    for build_dir in build_dirs:
        config = os.path.join(build_dir, "sfem_config.h")
        smesh_config_dir = os.path.join(build_dir, "external", "smesh")
        if not os.path.exists(config):
            continue
        for include_dir in (build_dir, smesh_config_dir):
            if os.path.isdir(include_dir) and include_dir not in seen:
                include_dirs.append(include_dir)
                seen.add(include_dir)
    return include_dirs


def _smesh_source_include_dirs(repo_root):
    smesh_src = os.path.join(repo_root, "external", "smesh", "src")
    if not os.path.isdir(smesh_src):
        return []
    include_dirs = [smesh_src]
    include_dirs.extend(
        path for path in sorted(glob.glob(os.path.join(smesh_src, "*"))) if os.path.isdir(path)
    )
    return include_dirs


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
    "FormDependencies",
    "FormEvaluation",
    "FormKind",
    "FormMetadata",
    "FormOrder",
    "FormBlock",
    "FormQualifier",
    "BoundaryIntegral",
    "Measure",
    "dx",
    "ds",
    "GenerationResult",
    "KernelTarget",
    "current_geometric_dimension",
    "geometric_dimension_context",
    "HyperelasticQualifier",
    "Identity",
    "MATERIAL_PARAMETER",
    "MaterialParameter",
    "MaterialParameterQualifier",
    "MatrixAssemblyVariantPlan",
    "MatrixFormat",
    "MatrixFormatPlan",
    "MatrixMeshLayout",
    "MixedFunctionSpace",
    "PRESSURE",
    "PREVIOUS_ARGUMENT",
    "PreviousFunction",
    "QuadratureSetting",
    "QualifiedExpression",
    "ScalarField",
    "SpatialCoordinate",
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
    "det",
    "deformation_gradient",
    "derivative",
    "div",
    "generate",
    "generate_op_registration_files",
    "grad",
    "inner",
    "inv",
    "material_parameter",
    "matrix_format_plan_from_request",
    "matrix_inner",
    "old",
    "PackedAssemblyPass",
    "previous_function",
    "qualifiers",
    "qualify",
    "run",
    "scalar_field",
    "tensor_field",
    "test_function",
    "trial_function",
    "vector_field",
    "variable",
    "value",
]
