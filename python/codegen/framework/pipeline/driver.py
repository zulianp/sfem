import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from enum import Enum

import sympy as sp

from codegen.framework.package.op_wrappers import (
    generate_op_files,
    generate_op_registration_files,
)
from codegen.framework.emitters.inexact_apply_codegen import inexact_apply_files
from codegen.framework.emitters.artifacts import (
    GeneratedKernelFile,
)
from codegen.framework.fem.basis import (
    BasisDataLayout,
    BasisEvaluation,
    BasisFamily,
    BasisPlanNode,
    basis_plan_for_element_at_cell_rule,
    basis_plan_for_quadrature_rule,
    basis_plans_for_fem_policy,
    field_basis_plan_for_fem_policy,
    field_basis_plans_for_fem_policy,
)
from codegen.framework.fem.geometry import (
    GeometryEvaluation,
    GeometryInputLayout,
    GeometryMode,
    GeometryPlanNode,
    affine_geometry_plan,
    geometry_plans_for_fem_policy,
    isoparametric_geometry_plan,
)
from codegen.framework.fem.tensor_product import (
    TensorProductDataLayout,
    TensorProductOperation,
    TensorProductSumFactorizationPlan,
    streams_in_shape_order,
    tensor_product_cartesian_shape_order,
    tensor_product_field_evaluation_plan,
    tensor_product_geometry_jacobian_plan,
    tensor_product_geometry_jacobian_plan_from_sizes,
    tensor_product_sum_factorization_plan,
    tensor_product_test_contraction_plan,
)
from codegen.framework.plans import conventions
from codegen.framework.plans.emission import (
    emission_plan_for_element,
    emission_plan_from_unit_context,
)
from codegen.framework.plans.expression import (
    KernelExpressionPlan,
)
from codegen.framework.plans.generation import (
    BlockPlan,
    DataStreamLayout,
    DataStreamPlan,
    DataStreamRole,
    GeometryPlan,
    KernelCoupling,
    KernelEmission,
    KernelPlan,
    KernelScope,
    KernelTarget,
    LocalKernelPlan,
    LocalPhase,
    LocalPhasePlan,
    MeshKernelPlan,
    MeshPhase,
    MeshPhasePlan,
)
from codegen.framework.symbolic.boundary_forms import (
    BoundaryIntegral,
    Measure,
    ds,
    dx,
)
from codegen.framework.symbolic.constitutive import (
    TwoPhaseFlowConstitutiveModel,
)
from codegen.framework.symbolic.core import (
    KernelExpressions,
    matrix_inner,
)
from codegen.framework.forms.equations import (
    EquationForm,
    EquationSystem,
    EquationSystemBuilder,
    EquationSystems,
    TOTAL_RESIDUAL_UNIT_NAME,
    total_residual_collection,
)
from codegen.framework.symbolic.fields import (
    FiniteElement,
    Function,
    FunctionSpace,
    MixedFunctionSpace,
    PREVIOUS_ARGUMENT,
    PreviousFunction,
    ScalarField,
    SpatialCoordinate,
    SymbolicArgument,
    SymbolicField,
    TEST_ARGUMENT,
    TRIAL_ARGUMENT,
    TensorField,
    TensorFunction,
    TestFunction,
    TimeRate,
    TrialFunction,
    VectorElement,
    VectorField,
    VectorFunction,
    VectorFunctionSpace,
    current_geometric_dimension,
    geometric_dimension_context,
    previous_function,
    scalar_field,
    tensor_field,
    test_function,
    trial_function,
    vector_field,
)
from codegen.framework.forms.forms import (
    FormBlock,
    FormCollection,
    FormDependencies,
    FormEvaluation,
    FormKind,
    FormMetadata,
    FormOrder,
    FormQualifier,
    PipelineStage,
    StandardFormName,
    energy_form_pipeline,
    residual_form_pipeline,
)
from codegen.framework.symbolic.operators import (
    Identity,
    adjugate,
    deformation_gradient,
    derivative,
    det,
    div,
    dt,
    grad,
    inner,
    inv,
    old,
    value,
)
from codegen.framework.symbolic.qualifiers import (
    CodegenQualifier,
    DEFORMATION_GRADIENT,
    DISPLACEMENT,
    FieldQualifier,
    HyperelasticQualifier,
    MATERIAL_PARAMETER,
    MaterialParameter,
    MaterialParameterQualifier,
    PRESSURE,
    QualifiedExpression,
    VELOCITY,
    material_parameter,
    qualifiers,
    qualify,
    variable,
)
from codegen.framework.forms.weak_forms import (
    sfem_soa_kernel_form,
    sfem_soa_weak_form,
)
from codegen.framework.plans.emission import ElementEmissionPlan as _ElementEmissionPlan
from codegen.framework.plans.generation import GenerationPlan as _GenerationPlan
from codegen.framework.emitters.quadrature_codegen import REFERENCE_DIRECTORY
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
    sfem_default_element_types,
    sfem_supported_element_types,
    sfem_taylor_hood_element_types,
    sfem_tensor_product_field_reference_data,
    sfem_tensor_product_hex_order,
    sfem_tensor_hex_shape_index,
)
from codegen.framework.backends.cuda import CUDASoABackend as _CUDASoABackend
from codegen.framework.backends.openmp import OpenMPSoABackend as _OpenMPSoABackend
from codegen.framework.targets import (
    current_target,
    use_target,
    AVX512Target,
    ARMSMETarget,
    ARMSVETarget,
    HIPTarget,
)
from codegen.framework.plans.conventions import unit_output_name
from codegen.framework.plans.scheduling import build_expression_graph


from codegen.framework.plans.residual_structure import (
    block_plan_from_form_block as _block_plan_from_form_block,
    block_plans_from_form_collection as _block_plans_from_form_collection,
    residual_local_phase_plans as _residual_local_phase_plans,
    residual_local_phases as _residual_local_phases,
    residual_mesh_phase_plans as _residual_mesh_phase_plans,
    residual_mesh_phases as _residual_mesh_phases,
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
    #: Opt in to the projected apply of `plans.inexact_apply`.  Off by default
    #: and additive when on: it publishes an extra entry point beside the exact
    #: one rather than replacing it, because on anything but an affine simplex
    #: it computes a deliberately different operator.
    inexact_apply: bool = False

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
        _validate_reserved_names(self.name, self.systems, self.parameter_defaults)


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
        collection = system.form_collection(equation, orders=(FormOrder.ONE,))
        value_symbols = set()
        for field in collection.residual_fields:
            value_symbols.add(field.value)
            value_symbols.add(field.direction_value)
            if field.previous_value is not None:
                value_symbols.add(field.previous_value)
        for expression in collection.residual_expressions:
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
        material = self.user_input.material
        # The inexact-apply family goes through a backend like everything else.
        # It used to call its emitter straight from here, under whatever target
        # happened to be ambient, and the driver decided which targets got it by
        # naming one.  A driver that names a target is a driver that has to be
        # edited every time a target is added, and the one it named was too
        # narrow: AVX-512, SVE and SME are the same CPU backend and were being
        # refused a family they can emit.  The backend answers now, because the
        # backend is what knows whether its target has a lowering.
        wants_inexact = bool(getattr(material, "inexact_apply", False))
        backend = _backend_for_target(target)
        for context in self.user_input.element_contexts:
            for unit in self.codegen_plan.emission_kernels_for_context(context):
                _merge_files(
                    outputs,
                    _layout_codegen_files(
                        unit,
                        context,
                        _emit_codegen_unit(unit, context, target),
                        backend.target.source_subdirectory(),
                    ),
                )
                if wants_inexact:
                    _merge_files(
                        outputs,
                        _layout_codegen_files(
                            unit,
                            context,
                            tuple(backend.emit_inexact(material, unit, context)),
                            backend.target.source_subdirectory(),
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
    inexact_apply=None,
):
    vector_size = int(vector_size)
    if vector_size <= 0:
        raise ValueError("vector_size must be positive")

    # Tri-state, like the matrix-format arguments above it: None keeps whatever
    # the material declares, and either explicit value overrides it.  Without a
    # way to ask for the split from outside the material, the only way to get it
    # was to rewrite the material at the call site, which is how the shipped
    # tree came to contain kernels no generator run could reproduce.
    if inexact_apply is not None:
        material = replace(material, inexact_apply=bool(inexact_apply))

    available_elements = _generation_available_elements(material.elements)
    selected = _parse_elements(elements, available_elements)
    selected = _with_tensor_product_proteus_alias_dependencies(selected, available_elements)
    out_dir = os.path.abspath(os.fspath(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    target = _normalize_generation_target(target)
    backend = _backend_for_target(target)
    if clean:
        # Only this target's own files: the host and the device tree share a
        # directory now, and a clean that crossed between them would delete
        # what the other run wrote.  Resolving the backend first is what makes
        # the question answerable here.
        _clean_outputs(out_dir, material.name, backend.target.source_subdirectory())

    matrix_format_plan = _selected_matrix_format_plan(
        material,
        matrix_formats,
        matrix_mesh_layouts,
        matrix_packed_passes,
        matrix_patch_node_index_filter,
    )
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
        # The backend's target has to be bound here too.  `backend.emit` binds
        # it around the kernels and returns, so without this the wrapper is
        # generated under whatever is ambient -- which is OpenMP -- and a CUDA
        # generation would quietly emit a host-shaped Op.
        with use_target(backend.target):
            files.update(_generate_op_wrapper_files(material, selected, user_input, files))
        _replace_legacy_tensor_product_sources_with_proteus_aliases(files)

    files = _relocate_generated_primitive_headers(files, out_dir, material.name)
    files = _collapse_duplicate_operators(files)
    _validate_generated_call_graph(files)
    _validate_generated_op_overrides(files)
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
        help="Matrix assembly format to emit: crs, bsr, block_diag_sym, or all. May be repeated or comma-separated.",
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
    parser.add_argument(
        "--inexact-apply",
        dest="inexact_apply",
        action="store_true",
        default=None,
        help="Emit the projected apply beside the exact one. Defaults to the material's own setting.",
    )
    parser.add_argument(
        "--no-inexact-apply",
        dest="inexact_apply",
        action="store_false",
        help="Suppress the projected apply even if the material asks for it.",
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
            inexact_apply=args.inexact_apply,
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
        combined = _total_residual_unit(
            material_system,
            mixed_order=any(
                isinstance(element, SfemCompatibleElement)
                and element.is_mixed_order
                and _element_dim(element) == dim
                for element in user_input.elements
            ),
        )
        if combined is not None:
            units = units + (combined,)
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
    # Rows are field components, columns are spatial directions.  This demanded
    # `dim x dim` -- the deformation gradient of a displacement -- which
    # excluded every scalar field, a Laplacian potential among them.
    if evaluated.data_symbols.shape[1] != dim:
        raise ValueError(
            "energy code generation requires variables with %d columns, one per "
            "spatial direction; got shape %s for energy unit '%s'"
            % (dim, evaluated.data_symbols.shape, evaluated.name or material_name)
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
            build_expression_graph(
                KernelExpressions()
                .add(
                "operator_evaluation",
                weak_form.diagnostic_expressions(has_direction=True),
            ),

                data_symbols=weak_form.deformation_gradient,
                temporary_prefix="%s_inspect_tmp" % material_name,
            )
        )
    return CodeGenerationUnit(
        name=unit_output_name(material_name, evaluated.name),
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
            name=unit_output_name(material_name, evaluated.name),
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
            payload={"diagnostics": evaluated.diagnostics},
            material_name=material_name,
            unit_name=evaluated.name,
        )
    block_kernels = (
        _block_codegen_units(material_name, dim, evaluated, blocks)
        if coupling is not KernelCoupling.SINGLE_FIELD
        else ()
    )
    return CodeGenerationUnit(
        name=unit_output_name(material_name, evaluated.name),
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
        payload={"diagnostics": evaluated.diagnostics},
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


def _block_codegen_units(material_name, dim, evaluated, blocks):
    return tuple(
        CodeGenerationUnit(
            name=unit_output_name(
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
        # The mixed pair permutes the cell's connectivity, which is all its
        # callee needs: which of those 27 nodes carry the coarser field is a
        # property of the element and the Cartesian kernel derives it itself.
        _tensor_product_proteus_alias(
            element_name="hex27_hex8",
            proteus_name="proteus_hex27_proteus_hex8",
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
                # Only the aliases whose target this file's twin actually
                # defines.  The whole list was written into every rewritten
                # file, which is invisible while an element has one source and
                # produces duplicate symbols the moment it has two -- the
                # inexact apply is a second source for the same element.
                defined = _extern_c_defined_names(files[proteus_path])
                selected = [
                    declaration
                    for declaration in alias_declarations
                    if _tensor_product_proteus_target_name(declaration["name"], alias)
                    in defined
                ]
                if not selected:
                    continue
                files[source_path] = _tensor_product_proteus_alias_source(
                    source_path,
                    c_abi_path,
                    selected,
                    alias,
                )


# A generated `extern "C"` header: a prototype through its `;` or a definition
# through the `{` that opens its body.  A prototype ends in `);` and a
# definition in `) {`, so the two are distinguishable without parsing C++, and
# neither carries a default argument or a nested call -- so the parameter list
# holds no `;` and no brace, and the non-greedy run cannot walk past the
# function it belongs to into the next one.
_EXTERN_C_HEADER = re.compile(r'extern "C"\s+[^;{]*?\)\s*[;{]', re.DOTALL)


def _extern_c_headers(source):
    """Every `extern "C"` function this source spells, declared or defined.

    One parser for the question "what does this file say about a C symbol",
    which three callers ask three ways: which symbols it defines, which it
    declares, and with what arity each is spelled.
    """
    headers = []
    for match in _EXTERN_C_HEADER.finditer(source):
        text = match.group(0)
        open_index = text.index("(")
        name_match = re.search(r"([A-Za-z_]\w*)\s*$", text[:open_index])
        if name_match is None:
            continue
        headers.append(
            {
                "name": name_match.group(1),
                "return_type": text[len('extern "C"') : name_match.start()].strip(),
                "params": text[open_index + 1 : text.rindex(")")].strip(),
                "defines": text.rstrip().endswith("{"),
            }
        )
    return tuple(headers)


def _extern_c_defined_names(source):
    """The `extern "C"` functions this source *defines*, not merely declares.

    The distinction is the whole point here: a file that only declares a symbol
    is not the file that should be aliased onto.
    """
    return frozenset(
        header["name"] for header in _extern_c_headers(source) if header["defines"]
    )


_CALL_GRAPH_SOURCE_EXTENSIONS = (".cpp", ".hpp", ".cu", ".cuh", ".hip")


def _c_parameter_count(params):
    text = params.strip()
    if not text or text == "void":
        return 0
    count, depth = 1, 0
    for character in text:
        if character in "([{<":
            depth += 1
        elif character in ")]}>":
            depth -= 1
        elif character == "," and depth == 0:
            count += 1
    return count


def _c_argument_count(text, open_index):
    """Count the arguments of the call whose `(` sits at `open_index`.

    Returns `None` when the parentheses do not balance before the text runs
    out, which is what a match inside a macro or a truncated construct looks
    like; the caller skips those rather than guessing.
    """
    depth, count, index, seen = 0, 0, open_index, False
    while index < len(text):
        character = text[index]
        if character in "\"'":
            index += 1
            while index < len(text) and text[index] != character:
                index += 2 if text[index] == "\\" else 1
            index += 1
            seen = True
            continue
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
            if depth == 0:
                return count + 1 if seen else 0
        elif character == "," and depth == 1:
            count += 1
        elif not character.isspace():
            seen = True
        index += 1
    return None


def _masked_call_sites(source):
    """The source with every `extern "C"` header blanked out, same length.

    Blanking rather than deleting keeps every offset, so a diagnostic can still
    name the line a call sits on, and it stops a signature from being read as a
    call to itself.
    """
    masked = list(source)
    for match in _EXTERN_C_HEADER.finditer(source):
        for index in range(*match.span()):
            if masked[index] != "\n":
                masked[index] = " "
    return "".join(masked)


_OVERRIDE_CLASS = re.compile(r"\bclass\s+(\w+)\s*(?:final\s*)?:\s*public\b")


def _class_body(text, open_brace):
    """The text between `open_brace`'s `{` and its match, or `None` if unbalanced."""
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index]
    return None


def _declared_overrides(body):
    """The names a class body declares with `override` and does not define inline.

    A declaration ends at the first `;` outside any nested braces; one that
    opens a brace first carries its own body and needs nothing elsewhere.
    """
    names = []
    statement = []
    index = 0
    while index < len(body):
        character = body[index]
        if character == "{":
            skipped = _class_body(body, index)
            if skipped is None:
                break
            index += len(skipped) + 2
            statement = []
            continue
        if character == ";":
            text = "".join(statement)
            if re.search(r"\boverride\b", text) and not re.search(r"=\s*(0|default|delete)\s*$", text.strip()):
                call = re.search(r"(~?\w+)\s*\(", text)
                if call:
                    names.append(call.group(1))
            statement = []
            index += 1
            continue
        statement.append(character)
        index += 1
    return names


def _validate_generated_op_overrides(files):
    """Every `override` a generated class declares is defined by a generated source.

    A declaration and its definition are decided in different places: the
    declaration asks what the *material* wants, the definition needs kernels the
    *target* can lower.  Conflating the two produced a device `Op` whose header
    declared `inexact_supported`, `inexact_update` and `inexact_apply` while its
    source defined none of them -- three undefined vtable slots that compiled
    without complaint, survived the whole library build, and surfaced only when
    an executable was linked against it.

    Nothing here inspects the linker.  The header and the sources of one
    generation are in hand at once, so the mismatch is a text fact about what
    was just emitted, and it raises where the spelling is decided.
    """
    sources = {
        path: text
        for path, text in files.items()
        if path.endswith(_CALL_GRAPH_SOURCE_EXTENSIONS)
    }
    problems = []
    for path in sorted(sources):
        text = sources[path]
        for match in _OVERRIDE_CLASS.finditer(text):
            brace = text.find("{", match.end())
            if brace < 0:
                continue
            body = _class_body(text, brace)
            if body is None:
                continue
            klass = match.group(1)
            for name in _declared_overrides(body):
                qualified = re.compile(r"(?<![\w:])%s::%s\b" % (re.escape(klass), re.escape(name)))
                if any(qualified.search(other) for other in sources.values()):
                    continue
                problems.append(
                    "%s declares `%s::%s` with `override`; no generated source defines it"
                    % (path, klass, name)
                )
    if problems:
        raise ValueError(
            "generated classes declare %d override(s) nothing defines:\n  %s"
            % (len(problems), "\n  ".join(problems))
        )


def _validate_generated_call_graph(files):
    """Every generated symbol a generated source calls must link, and must be
    called with the arity it was emitted with.

    This gates one recurring defect: a call site that knows a symbol's
    *logical* name but not its *emitted* one.  The emitted spelling belongs to
    the target -- `entry_point_name` prefixes `cu_`, and
    `entry_point_suffix_parameters` appends a stream -- so an emitter that
    splices the logical name into a call builds something that no longer names,
    or no longer fits, the function it meant.  Six such sites were found one
    nvcc error at a time, each invisible until a material happened to publish
    both halves for the same element, and one of them (a metric dispatch handed
    nine adjugate components where six metric components belong) compiled
    cleanly and gave a wrong answer.

    It is static text analysis over what was emitted and it changes nothing: a
    violation raises, so the fix lands where the spelling is decided.
    """
    sources = {
        path: text
        for path, text in files.items()
        if path.endswith(_CALL_GRAPH_SOURCE_EXTENSIONS)
    }
    arities = {}
    for path in sorted(sources):
        for header in _extern_c_headers(sources[path]):
            arity = _c_parameter_count(header["params"])
            previous = arities.get(header["name"])
            if previous is not None and previous != arity:
                raise ValueError(
                    'generated `extern "C"` %s is spelled with %d parameters '
                    "and with %d; %s disagrees with an earlier file"
                    % (header["name"], previous, arity, path)
                )
            arities[header["name"]] = arity

    problems = []
    for path in sorted(sources):
        masked = _masked_call_sites(sources[path])
        for match in re.finditer(r"(?<![\w:.>])([A-Za-z_]\w*)\s*\(", masked):
            name = match.group(1)
            line = masked.count("\n", 0, match.start()) + 1
            if name not in arities:
                # A name the tree defines only under a target's prefix is a
                # logical name that escaped into a call site.  Anything else is
                # a call out of the tree -- the standard library, SFEM, a
                # control-flow keyword -- and is none of this check's business,
                # so only a name shaped like a published entry point is asked
                # about at all.
                if len(name) < 8 or "_" not in name:
                    continue
                emitted = sorted(
                    other
                    for other in arities
                    if other.endswith(name)
                    and other != name
                    and re.fullmatch(r"[A-Za-z_]\w*", other[: -len(name)])
                )
                if emitted:
                    problems.append(
                        "%s:%d calls %s, which nothing defines; the emitted "
                        "spelling is %s" % (path, line, name, emitted[0])
                    )
                continue
            count = _c_argument_count(masked, match.end() - 1)
            if count is not None and count != arities[name]:
                problems.append(
                    "%s:%d calls %s with %d arguments; it is emitted with %d"
                    % (path, line, name, count, arities[name])
                )
    if problems:
        raise ValueError(
            "the generated tree does not link:\n  " + "\n  ".join(problems)
        )


def _tensor_product_proteus_alias(element_name, proteus_name, dim, n_shape):
    extension = current_target().mesh_source_extension()
    return {
        "element_name": element_name,
        "proteus_name": proteus_name,
        "n_shape": int(n_shape),
        "shape_order": tensor_product_cartesian_shape_order(int(dim), int(n_shape)),
        "source_prefix": "d%d/%s/" % (int(dim), element_name),
        "target_prefix": "d%d/%s/" % (int(dim), proteus_name),
        # The inexact apply is a second source file for the same element, and
        # it is why the pass now selects per file which aliases it writes: the
        # whole list into every rewritten file was invisible while each element
        # had one source and produces duplicate symbols with two.
        # The extension is the bound target's, not `cpp`.  While it was
        # spelled here, no device source ever matched, so the mesh-order
        # element kept its own kernels on a device target instead of
        # delegating to the PROTEUS twin the way it does on the host.
        "source_suffixes": tuple(
            "_%s_%s.%s" % (element_name, kind, extension)
            for kind in ("operator", "boundary_operator", "inexact_apply_operator")
        ),
        "suffix_pairs": tuple(
            ("_%s_%s.%s" % (element_name, kind, extension),
             "_%s_%s.%s" % (proteus_name, kind, extension))
            for kind in ("inexact_apply_operator", "operator", "boundary_operator")
        ),
    }


def _extern_c_declarations(source):
    return tuple(
        {
            "return_type": header["return_type"],
            "name": header["name"],
            "params": header["params"],
        }
        for header in _extern_c_headers(source)
        if not header["defines"]
    )


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


def _layout_codegen_files(unit, context, files, subdirectory=""):
    local_headers = tuple(
        generated.path
        for generated in files
        if _is_codegen_local_header(generated.path)
    )
    return tuple(
        GeneratedKernelFile(
            _layout_codegen_path(unit, context, generated.path, subdirectory),
            _layout_codegen_source(
                unit,
                context,
                generated.path,
                generated.source,
                local_headers,
                subdirectory,
            ),
        )
        for generated in files
    )


def _in_target_subdirectory(directory, subdirectory):
    """`d3/tet4` becomes `d3/tet4/cuda`, and the tree root becomes `cuda`.

    The whole of the target's placement decision: a device file sits in a local
    folder inside the directory its host twin occupies, which is how the rest
    of the repository is laid out and what stops the two trees colliding on the
    files that are `.hpp` on both -- the element API and the reference tables.
    """
    if not subdirectory:
        return directory
    return os.path.join(directory, subdirectory) if directory else subdirectory


def _layout_codegen_path(unit, context, filename, subdirectory=""):
    if _is_target_independent_codegen_file(filename):
        subdirectory = ""
    if _is_shared_reference_header(filename):
        # The name already carries `reference/`, so the target's folder goes
        # inside it -- beside the host tables, not above them.
        reference_directory, name = os.path.split(filename)
        return os.path.join(
            _in_target_subdirectory(reference_directory, subdirectory), name
        )
    directory = _codegen_file_directory(unit, context, filename, subdirectory)
    if not directory:
        return filename
    return os.path.join(directory, filename)


def _layout_codegen_source(
    unit, context, filename, source, local_headers, subdirectory=""
):
    if _is_target_independent_codegen_file(filename):
        subdirectory = ""
    directory = _codegen_file_directory(unit, context, filename, subdirectory)
    replacements = {}
    for header in _CODEGEN_COMMON_HEADERS:
        # Each included header is looked up where *it* was placed, which is not
        # always where this file was: `matrix_formats.hpp` keeps the host
        # position under every target.
        header_directory = _in_target_subdirectory(
            "", "" if _is_target_independent_codegen_file(header) else subdirectory
        )
        replacements[header] = _relative_codegen_include(
            directory, os.path.join(header_directory, header)
        )
    for local in local_headers:
        replacements[local] = _relative_codegen_include(
            directory,
            os.path.join(
                _in_target_subdirectory(
                    _codegen_dimension_directory(context), subdirectory
                ),
                local,
            ),
        )

    relocated = source
    for header, relative in replacements.items():
        if header != relative:
            relocated = relocated.replace(
                '#include "%s"' % header,
                '#include "%s"' % relative,
            )
    # The reference headers are a family, so their includes move by path rather
    # than by a lookup of each name.
    reference_directory = _in_target_subdirectory(REFERENCE_DIRECTORY, subdirectory)
    if directory or reference_directory != REFERENCE_DIRECTORY:
        relocated = relocated.replace(
            '#include "%s/' % REFERENCE_DIRECTORY,
            '#include "%s/' % _relative_codegen_include(directory, reference_directory),
        )
    return relocated


def _is_shared_primitive_header(filename):
    """`kernel_math.hpp`, or `cuda/kernel_math.cuh` under a device target.

    The set names the files; the target decides which folder they sit in, so
    membership is tested on the name and whatever directory the layout gave it
    is carried along when the header is hoisted.
    """
    normalized = str(filename).replace(os.sep, "/")
    return normalized.rsplit("/", 1)[-1] in _CODEGEN_SHARED_PRIMITIVE_HEADERS


def _is_shared_reference_header(filename):
    """`reference/<key>.hpp` -- one basis's or one rule's tables.

    Matched by shape rather than by membership in a frozenset, because unlike
    `kernel_math.hpp` this is a *family*: a new element or a new quadrature order
    adds a file, and a fixed list of names could not follow.
    """
    normalized = str(filename).replace(os.sep, "/")
    return normalized.startswith(
        "%s/" % REFERENCE_DIRECTORY
    ) and normalized.endswith((".hpp", ".cuh"))


#: Files a device generation emits unchanged, so there is one copy of each.
#:
#: The matrix-format record and the assembly operator it describes are host
#: C++ whatever the target -- no `__global__`, no `__device__`, no stream --
#: because matrix assembly has no device lowering.  Putting them in the
#: target's folder would make a second copy of a file that is byte-identical
#: apart from the include path the copy itself forced, which is the redundant
#: path rather than a device variant of anything.
def _is_target_independent_codegen_file(filename):
    name = str(filename).replace(os.sep, "/").rsplit("/", 1)[-1]
    return name == "matrix_formats.hpp" or name.endswith("_matrix_format_operator.cpp")


def _codegen_file_directory(unit, context, filename, subdirectory=""):
    if _is_target_independent_codegen_file(filename):
        subdirectory = ""
    if _is_shared_reference_header(filename):
        # The path already carries its directory, and the file is shared by every
        # material, so it must not be pushed down into d3/tet4/.
        return ""
    if filename in _CODEGEN_COMMON_HEADERS:
        return _in_target_subdirectory("", subdirectory)
    if _is_codegen_local_header(filename):
        return _in_target_subdirectory(
            _codegen_dimension_directory(context), subdirectory
        )
    return _in_target_subdirectory(
        _codegen_output_directory(unit, context), subdirectory
    )


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
    for filename in files:
        if _is_shared_primitive_header(filename) or _is_shared_reference_header(filename):
            header_targets[filename] = os.path.join("..", filename)

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
    # One tree per material, whatever the target: the device files sit in
    # `cuda/` folders inside it, the way the rest of the repository lays out
    # its device sources.  The sibling `<material>_cuda` tree this used to
    # accept was a second place for the same material and is gone.
    return generated_root == "generated" and output_name == material_name


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


def _total_residual_unit(material_system, mixed_order=False):
    """One more unit carrying the material's whole residual, where it needs one.

    A residual merit squares a node's complete value, so it has to be
    contracted by a single kernel over a single form.  A material written as
    several units has no such form until they are summed, which is what
    `total_residual_collection` does.

    Absent for a material that already is its own total residual.  A single
    unit's residual *is* the material's, so emitting a second unit computing
    the same thing would be a redundant path -- two kernels that must agree,
    with nothing making them.  So this is exactly the multi-unit case, and
    materials like navier_stokes, written as one coupled residual, are
    untouched.
    """
    equations = material_system.equations
    if len(equations) < 2:
        return None
    if not any(equation.is_residual for equation in equations):
        return None
    if mixed_order:
        # Not yet.  A mixed-order system's fields do not share a shape count,
        # so its kernels are emitted per field element type, and the combined
        # residual arrives as lowered components -- `u0`, `u1`, `u2` -- with no
        # record of which element each came from.  The emitter asks for exactly
        # that map and refuses without it.  Carrying the grouping through the
        # lowering is the fix and it is a separate piece of work; until then a
        # Taylor-Hood material keeps every kernel it has, and its residual
        # merit falls back to assembling per trial step, which is correct and
        # slow rather than absent.
        return None
    return LoweredEquationEvaluation(
        TOTAL_RESIDUAL_UNIT_NAME,
        # Both orders.  The unit publishes neither a residual nor a Jacobian
        # action -- `published_residual_forms` refuses them -- but lowering the
        # 2-form is measured at about two seconds against the 1-form's cost,
        # and requiring only what is published would make the 2-form optional
        # through the backend, the emitter and every signature that names it.
        # Not worth that for two seconds.
        total_residual_collection(
            material_system, orders=(FormOrder.ONE, FormOrder.TWO)
        ),
        kernels=(),
        diagnostics=False,
    )


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
            kernels=equation.kernels,
            diagnostics=equation.diagnostics,
            matrix_format_plan=matrix_format_plan,
        )
    raise TypeError("unsupported equation form %s" % equation.form)


def _equation_form_orders(equation):
    if equation.is_energy:
        return _energy_form_orders(equation.kernels)
    if equation.is_residual:
        return _residual_form_orders(equation.kernels)
    raise TypeError("unsupported equation form %s" % equation.form)


def _energy_data_symbols(dim, variables):
    """The energy's variables as `n_field_components x dim`.

    An energy differentiates against a field gradient, which has one row per
    field component and one column per spatial direction.  The count of
    variables and `dim` determine the shape between them: nine variables in
    three dimensions is a displacement's deformation gradient, three is a scalar
    field's gradient.

    Anything that does not divide by `dim` is not a gradient at all and keeps
    the column shape it had, which is what the callers that pass a list of
    unrelated variables rely on.
    """
    if len(variables) % dim == 0 and variables:
        return sp.Matrix(len(variables) // dim, dim, variables)
    return sp.Matrix(len(variables), 1, variables)


def _unit_generated_prefix(unit):
    name = _unit_output_name(unit)
    return name


def _unit_output_name(unit):
    return unit_output_name(unit.material_name, unit.unit_name)


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


def _residual_form_orders(kernels):
    order_by_kernel = {
        "value": FormOrder.ZERO,
        "objective": FormOrder.ZERO,
        "gradient": FormOrder.ONE,
        "residual": FormOrder.ONE,
        "apply": FormOrder.TWO,
        "jacobian_action": FormOrder.TWO,
    }
    orders = []
    for kernel in kernels:
        order = order_by_kernel.get(kernel)
        if order is not None and order not in orders:
            orders.append(order)
    if not orders:
        orders.extend((FormOrder.ONE, FormOrder.TWO))
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


#: The lexicographically numbered twin of each mesh-order tensor-product family.
_PROTEUS_TWIN_TYPES = {
    "QUAD4": "PROTEUS_QUAD4",
    "HEX8": "PROTEUS_HEX8",
    "HEX27": "PROTEUS_HEX27",
}


def _proteus_twin(element):
    """The same element with its nodes numbered lexicographically, or None.

    A mixed element has to be answered field by field: `HEX27_HEX8` becomes
    `PROTEUS_HEX27_PROTEUS_HEX8`, keeping the field names the material chose.
    Matching on the selection name alone could not see this, so the HEX27 half
    of the pairing never fired and `d3/hex27*` carried real kernels that
    permuted their own connectivity instead of forwarding to a Cartesian twin.
    """
    if not isinstance(element, SfemCompatibleElement):
        return _PROTEUS_TWIN_TYPES.get(_element_selection_name(element))
    cell = _PROTEUS_TWIN_TYPES.get(element.cell_element_type)
    fields = tuple(
        (field, _PROTEUS_TWIN_TYPES.get(family))
        for field, family in element.field_element_types
    )
    if cell is None or any(family is None for _, family in fields):
        return None
    return SfemCompatibleElement(
        "_".join(dict.fromkeys(family for _, family in fields)),
        cell,
        fields,
    )


def _generation_available_elements(elements):
    available = tuple(elements or sfem_supported_element_types())
    present = {_element_selection_name(element) for element in available}
    additions = []
    for element in available:
        twin = _proteus_twin(element)
        if twin is None:
            continue
        name = _element_selection_name(twin)
        if name not in present:
            present.add(name)
            additions.append(twin)
    return available + tuple(additions)


def _with_tensor_product_proteus_alias_dependencies(selected, available):
    selected = tuple(selected)
    additions = []
    selected_names = {_element_selection_name(element) for element in selected}
    available_by_name = {
        _element_selection_name(element): element
        for element in available
    }
    for element in selected:
        twin = _proteus_twin(element)
        if twin is None:
            continue
        name = _element_selection_name(twin)
        if name in available_by_name and name not in selected_names:
            selected_names.add(name)
            additions.append(available_by_name[name])
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


def _collapse_duplicate_operators(outputs):
    """Keep one copy of an operator source that several units emitted alike.

    Emission runs once per unit, so a coupled system produces one operator per
    form, each genuinely different.  The tensor-product alias rewrite then
    replaces every one of them with the same forwarding source, because a HEX8
    operator is an alias onto the PROTEUS_HEX8 one regardless of which form it
    was emitted for.  Two-phase flow ends with seven byte-identical translation
    units -- six forms plus the material's own -- each defining the same
    ``extern "C"`` symbols, so linking any two together fails with duplicate
    definitions and the material cannot be linked at all.

    This runs after that rewrite for exactly that reason: before it the sources
    still differ, and there is nothing to collapse.

    Nothing names these files: the manifest does not mention them and every
    consumer globs ``*_operator.cpp``, so collapsing the duplicates is invisible
    except that the material now links.  Only exact duplicates within one
    directory are collapsed, which is the per-form case and nothing else; two
    elements' operators differ in their symbols and are left alone.  The name
    kept is the shortest, which is the one without a form segment -- the same
    ``<material>_<element>_operator.cpp`` every single-form material produces.

    The deeper fix is that a form-specific name should not survive a rewrite
    that erases everything form-specific about the content.  This is the narrow
    correction, placed where the duplication becomes true rather than inside the
    alias rewrite that creates it.
    """
    by_directory = {}
    for filename, source in outputs.items():
        if not filename.endswith("_operator.cpp"):
            continue
        directory = os.path.dirname(filename)
        # _write_files normalises trailing whitespace, so two sources that
        # differ only there land on disk identical; compare what will be
        # written rather than what was produced.
        by_directory.setdefault((directory, source.rstrip()), []).append(filename)

    dropped = set()
    for filenames in by_directory.values():
        if len(filenames) < 2:
            continue
        keep = min(filenames, key=lambda name: (len(os.path.basename(name)), name))
        dropped.update(name for name in filenames if name != keep)

    if not dropped:
        return outputs
    return {
        filename: source
        for filename, source in outputs.items()
        if filename not in dropped
    }


def _merge_files(outputs, files):
    for generated in files:
        existing = outputs.get(generated.path)
        if existing is not None and existing != generated.source:
            raise RuntimeError("conflicting generated source for %s" % generated.path)
        outputs[generated.path] = generated.source


def _write_files(out_dir, files):
    """Write the generated sources, each appearing complete or not at all.

    Written to a temporary name in the same directory and renamed into place,
    because `os.replace` is atomic within a filesystem.  Every material writes
    the shared primitive headers -- `kernel_math.hpp` and its four siblings --
    with identical content, so two materials generating at once could otherwise
    interleave inside one of them and leave a reader with a half-written file.
    Materials are independent processes and there is no reason to run them one
    at a time; this is what makes running them together safe.
    """
    paths = []
    for filename, source in sorted(files.items()):
        path = os.path.join(out_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=os.path.dirname(path), prefix=".%s." % os.path.basename(path)
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as output:
                output.write(source.rstrip() + "\n")
            os.replace(temporary, path)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
        paths.append(path)
    return tuple(paths)


def _target_subdirectories():
    """Every folder a target claims for its own sources.

    Derived from the registered backends rather than listed, so adding a target
    cannot leave a folder that some other target's clean will walk into.
    """
    return frozenset(
        backend.target.source_subdirectory() for backend in BACKENDS_BY_TARGET.values()
    ) - {""}


def _belongs_to_target_tree(path, out_dir, subdirectory):
    """Is this file part of the tree the run being cleaned owns?

    The host and the device trees share a directory now, so a clean has to be
    able to tell them apart: without this, the host run's `d*/*` glob reaches
    `d3/cuda/` and deletes device headers that no later step rewrites.
    """
    segments = os.path.relpath(path, out_dir).split(os.sep)[:-1]
    claimed = _target_subdirectories()
    present = [segment for segment in segments if segment in claimed]
    return present == ([subdirectory] if subdirectory else [])


def _clean_outputs(out_dir, name, subdirectory=""):
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
        # The shared primitive headers are deliberately absent.  Every material
        # rewrites them with identical content, so removing them here achieved
        # nothing -- and with materials generating concurrently it is a race:
        # one can delete the header another has just written and is about to be
        # compiled against.  A stale one cannot survive unnoticed, because
        # `tools/codegen_snapshot.py check-tree` compares the committed tree
        # against a fresh generation and reports anything extra.
    )
    def _remove(*parts):
        #: The target's folder sits between the directory and the file name,
        #: and the predicate then rejects anything that belongs to a different
        #: target -- the `d*/*` glob below would otherwise reach `d3/cuda/`.
        directories = parts[:-1] + ((subdirectory,) if subdirectory else ())
        for path in glob.glob(os.path.join(out_dir, *directories, parts[-1])):
            if _belongs_to_target_tree(path, out_dir, subdirectory):
                os.remove(path)

    for pattern in patterns:
        _remove(pattern)
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
        _remove("d*", pattern)
    for pattern in nested_patterns:
        _remove("d*", "*", pattern)
    for pattern in ("sfem_*.hpp", "sfem_*.cuh", "sfem_*.cpp", "sfem_*.cu", "sfem_*.o", "sfem_*_manifest.json"):
        _remove("op", pattern)


def _repo_root():
    """The repository root, found by walking up rather than counting `..`.

    This used to be ``dirname(__file__)/../..``, which was correct only while
    this module lived in ``python/sfem/``.  Moving it into the framework broke
    that silently -- a path-shaped assumption of exactly the kind this
    refactoring set out to remove -- so it now looks for a marker instead.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(path, "base")) and os.path.isdir(
            os.path.join(path, "python")
        ):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            raise RuntimeError("could not locate the SFEM repository root from %s" % __file__)
        path = parent


def _compile_operators(paths):
    compiler = _operator_compiler()
    if compiler is None:
        raise RuntimeError("C++ compiler is not available")
    repo_root = _repo_root()
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


def _validate_reserved_names(name, systems, parameter_defaults):
    """Refuse a material that declares a name the generator has claimed.

    Checked here, once, at specification time -- not in a kernel, and not left to
    coincidence.  The generator's own vocabulary and the material author's
    overlap in exactly the places a physicist would reach for first: `T` is a
    temperature and a template parameter, `S` a second Piola-Kirchhoff stress,
    `G` a shear modulus.  `plans/conventions.py` keeps the generator off those,
    and this makes the reservation real rather than a note in a document.
    """
    declared = [str(parameter) for parameter, _default in parameter_defaults]
    for system in systems:
        for field in getattr(system, "fields", ()) or ():
            field_name = getattr(field, "name", None)
            if field_name:
                declared.append(str(field_name))
    conventions.check_material(name, declared)


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
    "dt",
    "old",
    "TimeRate",
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
