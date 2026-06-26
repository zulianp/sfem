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
    GeneratedKernelFile,
    KernelExpressions,
    TwoPhaseFlowConstitutiveModel,
    FormEvaluation,
    FormOrder,
    PipelineStage,
    coupled_residual_weak_coefficients,
    energy_form_pipeline,
    generate_coupled_residual_sfem_files,
    generate_sfem_soa_cpp_files_for_element,
    matrix_inner,
    residual_form_pipeline,
    sfem_soa_kernel_form,
    sfem_soa_weak_form,
)
from codegen.framework.fem import (
    sfem_supported_element_types,
    sfem_soa_element_specialization,
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
class GenerationResult:
    sources: tuple
    objects: tuple = ()


@dataclass(frozen=True)
class ElementGenerationContext:
    material_name: str
    element_type: str
    specialization: object

    @classmethod
    def create(cls, material_name, element_type, vector_size, quadrature_order):
        return cls(
            material_name,
            str(element_type).upper(),
            sfem_soa_element_specialization(
                element_type,
                vector_size,
                quadrature_order,
            ),
        )

    @property
    def generated_prefix(self):
        return "generated_%s" % self.material_name

    @property
    def element_prefix(self):
        return "%s_%s" % (self.generated_prefix, self.element_type.lower())

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
    form_evaluation: FormEvaluation
    weak_form: object
    kernel_forms: tuple
    diagnostic_graph: object


@dataclass(frozen=True)
class HyperelasticFormEvaluation:
    material: HyperelasticMaterial
    by_dim: dict

    @property
    def stage(self):
        return PipelineStage.FORM_EVALUATION


@dataclass(frozen=True)
class ResidualDimensionEvaluation:
    system: CoupledResidualSystem
    form_evaluation: FormEvaluation
    residual_coeffs: tuple
    action_coeffs: tuple


@dataclass(frozen=True)
class CoupledResidualFormEvaluation:
    material: CoupledResidualMaterial
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
        if isinstance(self.form_evaluation, HyperelasticFormEvaluation):
            return _hyperelastic_codegen_plan(self.form_evaluation)
        if isinstance(self.form_evaluation, CoupledResidualFormEvaluation):
            return _coupled_residual_codegen_plan(self.form_evaluation)
        raise TypeError(
            "unsupported form evaluation type %s"
            % type(self.form_evaluation).__name__
        )


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
    material = user_input.material
    if isinstance(material, HyperelasticMaterial):
        return _evaluate_hyperelastic_forms(user_input)
    if isinstance(material, CoupledResidualMaterial):
        return _evaluate_coupled_residual_forms(user_input)
    raise TypeError("unsupported material type %s" % type(material).__name__)


def _hyperelastic_codegen_plan(form_evaluation):
    units = []
    for dim, evaluated in form_evaluation.by_dim.items():
        units.append(
            CodeGenerationUnit(
                CodeGenerationKind.HYPERELASTIC_SOA,
                form_evaluation.material.name,
                dim,
                HyperelasticCodeGenerationPayload(
                    evaluated.kernel_forms,
                    evaluated.diagnostic_graph,
                    form_evaluation.material.diagnostics,
                ),
            )
        )
    return CodeGenerationPlan(tuple(units))


def _coupled_residual_codegen_plan(form_evaluation):
    units = []
    for dim, evaluated in form_evaluation.by_dim.items():
        units.append(
            CodeGenerationUnit(
                CodeGenerationKind.RESIDUAL_SOA,
                form_evaluation.material.name,
                dim,
                ResidualCodeGenerationPayload(
                    evaluated.system,
                    evaluated.residual_coeffs,
                    evaluated.action_coeffs,
                ),
            )
        )
    return CodeGenerationPlan(tuple(units))


def _emit_codegen_unit(unit, context):
    if unit.kind is CodeGenerationKind.HYPERELASTIC_SOA:
        return _emit_hyperelastic_soa(unit, context)
    if unit.kind is CodeGenerationKind.RESIDUAL_SOA:
        return _emit_residual_soa(unit, context)
    raise ValueError("unsupported code generation unit kind %s" % unit.kind)


def _emit_hyperelastic_soa(unit, context):
    payload = unit.payload
    files = list(
        generate_sfem_soa_cpp_files_for_element(
            payload.kernel_forms,
            prefix=context.element_prefix,
            local_prefix=context.local_prefix,
            specialization=context.specialization,
        )
    )
    if payload.diagnostics:
        report_prefix = "%s_%s" % (
            unit.material_name,
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
    payload = unit.payload
    return generate_coupled_residual_sfem_files(
        payload.system,
        prefix=context.generated_prefix,
        element_type=context.element_type,
        vector_size=context.specialization.vector_size,
        quadrature_order=context.specialization.quadrature_rule.order,
        specialization=context.specialization,
        residual_coeffs=payload.residual_coeffs,
        action_coeffs=payload.action_coeffs,
    )


def _evaluate_hyperelastic_forms(user_input):
    material = user_input.material
    by_dim = {}
    orders = _hyperelastic_form_orders(material.kernels)
    for context in user_input.element_contexts:
        dim = context.specialization.dim
        if dim in by_dim:
            continue
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
            material.energy(deformation_gradient),
            tuple(deformation_gradient),
            directions,
        )
        form_evaluation = energy_pipeline.evaluate(orders)
        weak_form = sfem_soa_weak_form(
            form_evaluation.form(FormOrder.ZERO).expression,
            deformation_gradient,
        )
        kernel_forms = tuple(
            sfem_soa_kernel_form(
                kernel,
                weak_form=weak_form,
                has_direction=kernel == "apply",
                output_mode="accumulate",
            )
            for kernel in material.kernels
        )
        diagnostic_graph = None
        if material.diagnostics:
            diagnostic_graph = (
                KernelExpressions()
                .add(
                    "operator_evaluation",
                    weak_form.diagnostic_expressions(has_direction=True),
                )
                .build_graph(
                    data_symbols=weak_form.deformation_gradient,
                    temporary_prefix="%s_inspect_tmp" % material.name,
                )
            )
        by_dim[dim] = HyperelasticDimensionEvaluation(
            form_evaluation,
            weak_form,
            kernel_forms,
            diagnostic_graph,
        )
    return HyperelasticFormEvaluation(material, by_dim)


def _evaluate_coupled_residual_forms(user_input):
    material = user_input.material
    by_dim = {}
    for context in user_input.element_contexts:
        dim = context.specialization.dim
        if dim in by_dim:
            continue
        system = CoupledResidualSystem(dim)
        material.define(system)
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
        form_evaluation = residual_form_pipeline(
            residual_vector,
            variables,
            directions,
        ).evaluate((FormOrder.ZERO, FormOrder.ONE, FormOrder.TWO))
        by_dim[dim] = ResidualDimensionEvaluation(
            system,
            form_evaluation,
            coupled_residual_weak_coefficients(system, False),
            coupled_residual_weak_coefficients(system, True),
        )
    return CoupledResidualFormEvaluation(material, by_dim)


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
    supported = set(sfem_supported_element_types())
    defaults = tuple(str(element).upper() for element in defaults)
    enabled = set(defaults)
    if not values:
        selected = defaults
    else:
        selected = []
        for value in values:
            for item in str(value).split(","):
                element = item.strip().upper()
                if not element:
                    continue
                if element == "ALL":
                    selected = list(defaults)
                    break
                selected.append(element)
        selected = tuple(dict.fromkeys(selected))

    invalid = tuple(element for element in selected if element not in supported)
    if invalid:
        raise ValueError(
            "unsupported element %s; expected one of %s"
            % (", ".join(invalid), ", ".join(sorted(supported)))
        )
    disabled = tuple(element for element in selected if element not in enabled)
    if disabled:
        raise ValueError(
            "element %s is not enabled for this material; expected one of %s"
            % (", ".join(disabled), ", ".join(defaults))
        )
    return tuple(selected)


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
    "GenerationResult",
    "HyperelasticMaterial",
    "TwoPhaseFlowConstitutiveModel",
    "generate",
    "matrix_inner",
    "run",
]
