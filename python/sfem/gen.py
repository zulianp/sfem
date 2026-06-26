import argparse
import glob
import os
import shutil
import subprocess
from dataclasses import dataclass

import sympy as sp

from ._gen_op import generate_op_files
from codegen.framework import (
    CoupledResidualSystem,
    KernelExpressions,
    TwoPhaseFlowConstitutiveModel,
    FormOrder,
    energy_form_pipeline,
    generate_coupled_residual_sfem_files,
    generate_sfem_soa_cpp_files_for_element,
    matrix_inner,
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

    if isinstance(material, HyperelasticMaterial):
        files = _generate_hyperelastic(
            material,
            selected,
            vector_size,
            quadrature_order,
        )
    elif isinstance(material, CoupledResidualMaterial):
        files = _generate_coupled_residual(
            material,
            selected,
            vector_size,
            quadrature_order,
        )
    else:
        raise TypeError("unsupported material type %s" % type(material).__name__)

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


def _generate_hyperelastic(material, elements, vector_size, quadrature_order):
    outputs = {}
    for element in elements:
        context = ElementGenerationContext.create(
            material.name,
            element,
            vector_size,
            quadrature_order,
        )
        dim = context.specialization.dim
        deformation_gradient = sp.Matrix(
            dim,
            dim,
            tuple(sp.symbols("F[%d]" % i) for i in range(dim * dim)),
        )
        energy_pipeline = energy_form_pipeline(
            material.energy(deformation_gradient),
            tuple(deformation_gradient),
        )
        weak_form = sfem_soa_weak_form(
            energy_pipeline.form(FormOrder.ZERO).expression,
            deformation_gradient,
        )
        forms = tuple(
            sfem_soa_kernel_form(
                kernel,
                weak_form=weak_form,
                has_direction=kernel == "apply",
                output_mode="accumulate",
            )
            for kernel in material.kernels
        )
        generated = generate_sfem_soa_cpp_files_for_element(
            forms,
            prefix=context.element_prefix,
            local_prefix=context.local_prefix,
            specialization=context.specialization,
        )
        _merge_files(outputs, generated)
        if material.diagnostics:
            graph = (
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
            report_prefix = "%s_%s" % (material.name, element.lower())
            outputs["%s_summary.md" % report_prefix] = _summary(
                material.name,
                graph,
                context.specialization,
            )
            outputs["%s_reduced_outputs.txt" % report_prefix] = (
                "\n\n".join(str(output) for output in graph.reduced_outputs) + "\n"
            )
    return outputs


def _generate_coupled_residual(
    material,
    elements,
    vector_size,
    quadrature_order,
):
    outputs = {}
    systems = {}
    for element in elements:
        context = ElementGenerationContext.create(
            material.name,
            element,
            vector_size,
            quadrature_order,
        )
        system = systems.get(context.specialization.dim)
        if system is None:
            system = CoupledResidualSystem(context.specialization.dim)
            material.define(system)
            systems[context.specialization.dim] = system
        _merge_files(
            outputs,
            generate_coupled_residual_sfem_files(
                system,
                prefix=context.generated_prefix,
                element_type=element,
                vector_size=vector_size,
                quadrature_order=quadrature_order,
            ),
        )
    return outputs


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
