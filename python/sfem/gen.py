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
    CoupledResidualSystem,
    EquationForm,
    EquationSystem,
    GeneratedKernelFile,
    KernelExpressions,
    TwoPhaseFlowConstitutiveModel,
    FormEvaluation,
    FormOrder,
    PipelineStage,
    coupled_residual_weak_coefficients,
    energy_form_pipeline,
    generate_coupled_residual_sfem_files,
    generate_mixed_residual_sfem_files,
    generate_sfem_soa_cpp_files_for_element,
    matrix_inner,
    residual_form_pipeline,
    sfem_soa_kernel_form,
    sfem_soa_weak_form,
)
from codegen.framework.fem import (
    SfemCompatibleElement,
    sfem_supported_element_types,
    sfem_soa_element_specialization,
    sfem_taylor_hood_element_types,
)


DEFAULT_VECTOR_SIZE = 16


@dataclass(frozen=True)
class HyperelasticMaterial:
    name: str
    energy: object
    elements: tuple = ()
    kernels: tuple = ("objective", "gradient", "apply")
    diagnostics: bool = True
    op_name: str = None
    parameter_defaults: tuple = ()

    def __post_init__(self):
        _validate_name(self.name)
        if not callable(self.energy):
            raise TypeError("energy must be callable")
        _validate_op(self.op_name, self.parameter_defaults)


@dataclass(frozen=True)
class CoupledResidualMaterial:
    name: str
    define: object
    elements: tuple
    op_name: str = None
    parameter_defaults: tuple = ()

    def __post_init__(self):
        _validate_name(self.name)
        if not callable(self.define):
            raise TypeError("define must be callable")
        if not self.elements:
            raise ValueError("coupled residual materials require supported elements")
        _validate_op(self.op_name, self.parameter_defaults)


@dataclass(frozen=True)
class UnifiedMaterial:
    name: str
    define: object
    elements: tuple
    op_name: str = None
    parameter_defaults: tuple = ()

    def __post_init__(self):
        _validate_name(self.name)
        if not callable(self.define):
            raise TypeError("define must be callable")
        if not self.elements:
            raise ValueError("unified materials require supported elements")
        _validate_op(self.op_name, self.parameter_defaults)


@dataclass(frozen=True)
class GenerationResult:
    sources: tuple
    objects: tuple = ()


@dataclass(frozen=True)
class ElementGenerationContext:
    material_name: str
    element_type: str
    label: str
    specialization: object
    compatible_element: object = None

    @classmethod
    def create(cls, material_name, element, vector_size, quadrature_order):
        compatible = element if isinstance(element, SfemCompatibleElement) else None
        element_type = compatible.cell_element_type if compatible else str(element).upper()
        label = compatible.name.lower() if compatible else element_type.lower()
        return cls(
            material_name,
            element_type,
            label,
            sfem_soa_element_specialization(
                element_type,
                vector_size,
                quadrature_order,
            ),
            compatible,
        )

    @property
    def generated_prefix(self):
        return "generated_%s" % self.material_name

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
        if self.specialization.quadrature_rule.is_tensor_product:
            return "tensor_product"
        return "simplex"

    @property
    def is_mixed_order(self):
        return bool(self.compatible_element and self.compatible_element.is_mixed_order)


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
        contexts = tuple(
            ElementGenerationContext.create(
                material.name,
                element,
                vector_size,
                quadrature_order,
            )
            for element in elements
        )
        return cls(material, tuple(elements), vector_size, quadrature_order, contexts)


@dataclass(frozen=True)
class HyperelasticDimensionEvaluation:
    name: str
    form_evaluation: FormEvaluation
    deformation_gradient: object
    kernels: tuple
    diagnostics: bool


@dataclass(frozen=True)
class ResidualDimensionEvaluation:
    name: str
    system: CoupledResidualSystem
    form_evaluation: FormEvaluation
    fields: tuple


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
    HYPERELASTIC_SOA = "hyperelastic_soa"
    RESIDUAL_SOA = "residual_soa"


@dataclass(frozen=True)
class CodeGenerationUnit:
    kind: CodeGenerationKind
    material_name: str
    unit_name: str
    dim: int
    payload: object

    def matches(self, context):
        return self.dim == context.specialization.dim


@dataclass(frozen=True)
class HyperelasticCodeGenerationPayload:
    kernel_forms: tuple
    diagnostic_graph: object
    diagnostics: bool


@dataclass(frozen=True)
class ResidualCodeGenerationPayload:
    system: CoupledResidualSystem
    residual_coeffs: tuple
    action_coeffs: tuple
    equation_fields: tuple


@dataclass(frozen=True)
class CodeGenerationPlan:
    units: tuple

    def units_for_context(self, context):
        return tuple(unit for unit in self.units if unit.matches(context))


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
            for unit in self.codegen_plan.units_for_context(context):
                _merge_files(outputs, _emit_codegen_unit(unit, context))
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
    files = CodeGenerationStage(user_input, codegen_plan).run()

    if material.op_name:
        if isinstance(material, UnifiedMaterial):
            raise ValueError(
                "generated Op wrappers are only available for legacy single-form "
                "HyperelasticMaterial and CoupledResidualMaterial inputs"
            )
        files.update(generate_op_files(material, selected))

    source_paths = _write_files(out_dir, files)
    object_paths = _compile_operators(source_paths) if compile else ()
    return GenerationResult(source_paths, object_paths)


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
    return result


def _evaluate_forms(user_input):
    by_dim = {}
    for context in user_input.element_contexts:
        dim = context.specialization.dim
        if dim in by_dim:
            continue
        units = tuple(
            _evaluate_equation(dim, equation)
            for equation in _material_equations(user_input.material, dim)
        )
        if not units:
            raise ValueError("material '%s' did not define any equations" % user_input.material.name)
        by_dim[dim] = DimensionFormEvaluation(dim, units)
    return UnifiedFormEvaluation(user_input.material, by_dim)


def _codegen_plan_from_form_evaluation(form_evaluation):
    units = []
    for dim_eval in form_evaluation.by_dim.values():
        for evaluated in dim_eval.units:
            if isinstance(evaluated, HyperelasticDimensionEvaluation):
                units.append(
                    _hyperelastic_codegen_unit(
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
    return CodeGenerationPlan(tuple(units))


def _hyperelastic_codegen_unit(material_name, dim, evaluated):
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
        CodeGenerationKind.HYPERELASTIC_SOA,
        material_name,
        evaluated.name,
        dim,
        HyperelasticCodeGenerationPayload(
            kernel_forms,
            diagnostic_graph,
            evaluated.diagnostics,
        ),
    )


def _residual_codegen_unit(material_name, dim, evaluated):
    return CodeGenerationUnit(
        CodeGenerationKind.RESIDUAL_SOA,
        material_name,
        evaluated.name,
        dim,
        ResidualCodeGenerationPayload(
            evaluated.system,
            coupled_residual_weak_coefficients(evaluated.system, False),
            coupled_residual_weak_coefficients(evaluated.system, True),
            evaluated.fields,
        ),
    )


def _emit_codegen_unit(unit, context):
    if unit.kind is CodeGenerationKind.HYPERELASTIC_SOA:
        return _emit_hyperelastic_soa(unit, context)
    if unit.kind is CodeGenerationKind.RESIDUAL_SOA:
        return _emit_residual_soa(unit, context)
    raise ValueError("unsupported code generation unit kind %s" % unit.kind)


def _emit_hyperelastic_soa(unit, context):
    payload = unit.payload
    generated_prefix = _unit_generated_prefix(unit)
    files = list(
        generate_sfem_soa_cpp_files_for_element(
            payload.kernel_forms,
            prefix="%s_%s" % (generated_prefix, context.element_type.lower()),
            local_prefix="%s_d%d_%s"
            % (generated_prefix, context.specialization.dim, context.family),
            specialization=context.specialization,
        )
    )
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
    if context.is_mixed_order:
        payload = unit.payload
        return generate_mixed_residual_sfem_files(
            payload.system,
            prefix=_unit_generated_prefix(unit),
            compatible_element=context.compatible_element,
            vector_size=context.specialization.vector_size,
            quadrature_order=context.specialization.quadrature_rule.order,
            residual_coeffs=payload.residual_coeffs,
            action_coeffs=payload.action_coeffs,
            field_element_types=_field_element_types_for_context(
                payload.equation_fields,
                context,
            ),
        )
    payload = unit.payload
    return generate_coupled_residual_sfem_files(
        payload.system,
        prefix=_unit_generated_prefix(unit),
        element_type=context.element_type,
        vector_size=context.specialization.vector_size,
        quadrature_order=context.specialization.quadrature_rule.order,
        specialization=context.specialization,
        residual_coeffs=payload.residual_coeffs,
        action_coeffs=payload.action_coeffs,
    )


def _material_equations(material, dim):
    system = EquationSystem(dim)
    if isinstance(material, HyperelasticMaterial):
        system.energy(
            "",
            material.energy,
            kernels=material.kernels,
            diagnostics=material.diagnostics,
        )
        return system.equations
    if isinstance(material, CoupledResidualMaterial):
        system.residual("", material.define)
        return system.equations
    if isinstance(material, UnifiedMaterial):
        material.define(system)
        return system.equations
    raise TypeError("unsupported material type %s" % type(material).__name__)


def _evaluate_equation(dim, equation):
    if equation.is_energy:
        return _evaluate_energy_equation(dim, equation)
    if equation.is_residual:
        return _evaluate_residual_equation(dim, equation)
    raise TypeError("unsupported equation form %s" % equation.form)


def _evaluate_energy_equation(dim, equation):
    orders = _hyperelastic_form_orders(equation.kernels)
    deformation_gradient = sp.Matrix(
        dim,
        dim,
        tuple(sp.symbols("F[%d]" % i) for i in range(dim * dim)),
    )
    directions = (
        tuple(sp.symbols("dF[%d]" % i) for i in range(dim * dim))
        if FormOrder.TWO in orders
        else None
    )
    energy_pipeline = energy_form_pipeline(
        equation.define(deformation_gradient),
        tuple(deformation_gradient),
        directions,
    )
    return HyperelasticDimensionEvaluation(
        equation.name,
        energy_pipeline.evaluate(orders),
        deformation_gradient,
        equation.kernels,
        equation.diagnostics,
    )


def _evaluate_residual_equation(dim, equation):
    system = CoupledResidualSystem(dim)
    equation.define(system)
    residual_vector = sp.Matrix(
        [system.residual(field) for field in system.fields]
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
    return ResidualDimensionEvaluation(
        equation.name,
        system,
        residual_form_pipeline(
            residual_vector,
            variables,
            directions,
        ).evaluate((FormOrder.ZERO, FormOrder.ONE, FormOrder.TWO)),
        equation.fields,
    )


def _field_element_types_for_context(equation_fields, context):
    ret = {}
    for field in equation_fields:
        family = field.family or field.name
        element_type = context.compatible_element.element_for_field(family)
        if field.components == 1:
            ret[field.name] = element_type
        else:
            for component in range(field.components):
                ret["%s%d" % (field.name, component)] = element_type
    return ret


def _unit_generated_prefix(unit):
    name = _unit_output_name(unit)
    return "generated_%s" % name


def _unit_output_name(unit):
    if unit.unit_name:
        return "%s_%s" % (unit.material_name, unit.unit_name)
    return unit.material_name


def _unit_report_name(unit):
    return _unit_output_name(unit)


def _hyperelastic_form_orders(kernels):
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
    default_by_name = {_element_selection_name(element): element for element in default_entries}
    supported = set(sfem_supported_element_types())
    supported.update(default_by_name)
    defaults = tuple(default_by_name)
    enabled = set(defaults)
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


def _element_selection_name(element):
    if isinstance(element, SfemCompatibleElement):
        return element.name
    return str(element).upper()


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
        "%s_*_summary.md" % name,
        "%s_*_reduced_outputs.txt" % name,
        "kernel_math.hpp",
        "kernel_diagnostics.hpp",
    )
    for pattern in patterns:
        for path in glob.glob(os.path.join(out_dir, pattern)):
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
    "CoupledResidualMaterial",
    "CoupledResidualSystem",
    "DEFAULT_VECTOR_SIZE",
    "EquationForm",
    "EquationSystem",
    "GenerationResult",
    "HyperelasticMaterial",
    "TwoPhaseFlowConstitutiveModel",
    "UnifiedMaterial",
    "generate",
    "matrix_inner",
    "run",
    "sfem_taylor_hood_element_types",
]
