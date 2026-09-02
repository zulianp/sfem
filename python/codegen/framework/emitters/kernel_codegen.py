"""Procedural C, OpenMP, and CUDA kernel emission from an expression graph.

These entry points turn a ``KernelExpressions`` graph into source text.  They
were part of ``symbolic/core.py`` until the layering work; they live here because
emitting text is this layer's job and because they select syntax through a target
object, which the specification layer must not name.
"""

from dataclasses import dataclass

import sympy as sp

from codegen.framework.symbolic.core import (
    DataLayout,
    KernelExpressions,
    LayoutKind,
    ScopeKind,
    layout_offset,
)
from codegen.framework.plans.scheduling import (
    EvaluationPlan,
    ExpressionGraph,
    build_expression_graph,
)
from codegen.framework.emitters.cprinter import (
    _cpp_argument_name,
    _cpp_lvalue,
    _cpp_macro_name,
    _cpp_symbol,
    _direct_output_targets,
    _group_kernel_symbols,
    _indexed_symbol,
    _sfem_ccode,
    _sfem_math_function_lines,
)
from codegen.framework.emitters.artifacts import GeneratedKernelCode
from codegen.framework.backends.targets import CUDATarget, OpenMPTarget




def _sfem_math_inline_source_lines(
    inline_qualifier="SFEM_INLINE",
    define_sfem_inline=True,
):
    lines = [
    ]
    if define_sfem_inline:
        lines.extend(
            [
                "#ifndef SFEM_INLINE",
                "#define SFEM_INLINE inline",
                "#endif",
                "",
            ]
        )
    lines.extend(_sfem_math_function_lines(inline_qualifier))
    return lines


def generate_cpp_kernel(
    expression_graph,
    function_name="generated_kernel",
    scalar_type="double",
    output_name="out",
):
    statements = expression_graph.evaluation_plan.statements
    temporary_symbols = set(expression_graph.evaluation_plan.temporary_symbols)
    input_symbols, output_targets = _kernel_io_symbols(statements, temporary_symbols)
    arguments = _kernel_arguments(input_symbols, output_targets, scalar_type, output_name)

    lines = [
        "#include <math.h>",
        "",
    ]
    lines.extend(_sfem_math_inline_source_lines())
    lines.extend(["", 'extern "C" void %s(%s) {' % (function_name, ", ".join(arguments))])

    _append_statement_lines(lines, statements, scalar_type, output_name, indent="    ")

    lines.append("}")
    lines.append("")
    return GeneratedKernelCode("c++", function_name, "\n".join(lines))


def generate_openmp_cpp_kernel(
    expression_graph,
    function_name="generated_openmp_kernel",
    wrapper_name=None,
    scalar_type="double",
    index_type="ptrdiff_t",
    output_name="out",
    target=None,
):
    target = OpenMPTarget() if target is None else target
    wrapper_name = wrapper_name or _cpp_wrapper_name(function_name)
    element_function_name = "%s_element" % function_name
    statements = expression_graph.evaluation_plan.statements
    temporary_symbols = set(expression_graph.evaluation_plan.temporary_symbols)
    input_symbols, output_targets = _kernel_io_symbols(statements, temporary_symbols)
    element_arguments = _kernel_arguments(
        input_symbols,
        output_targets,
        scalar_type,
        output_name,
    )
    batch_arguments = _openmp_kernel_arguments(
        input_symbols,
        output_targets,
        scalar_type,
        index_type,
        output_name,
    )
    element_call_arguments = _openmp_element_call_arguments(
        input_symbols,
        output_targets,
        output_name,
    )

    lines = [
        "#include <stddef.h>",
        "#include <math.h>",
        "",
    ]
    lines.extend(_sfem_math_inline_source_lines())
    lines.extend(
        [
            "",
            'extern "C" void %s(%s)' % (element_function_name, ", ".join(element_arguments)),
            "{",
        ]
    )
    _append_statement_lines(lines, statements, scalar_type, output_name, indent="    ")
    lines.extend(
        [
            "}",
            "",
            'extern "C" void %s(%s) {' % (function_name, ", ".join(batch_arguments)),
        ]
    )
    pragma = target.parallel_for_pragma()
    if pragma:
        lines.append(pragma)
    lines.extend(
        [
            "    for (%s e = 0; e < nelements; ++e) {" % index_type,
            "        %s(%s);" % (element_function_name, ", ".join(element_call_arguments)),
            "    }",
            "}",
            "",
            "struct %s {" % wrapper_name,
            "    void apply(%s) const {" % ", ".join(batch_arguments),
            "        %s(%s);" % (function_name, ", ".join(_openmp_wrapper_call_arguments(batch_arguments))),
            "    }",
            "};",
            "",
        ]
    )
    return GeneratedKernelCode(target.generated_language, function_name, "\n".join(lines))


def generate_cuda_kernel(
    expression_graph,
    function_name="generated_cuda_kernel",
    scalar_type="double",
    index_type="ptrdiff_t",
    output_name="out",
    target=None,
):
    target = CUDATarget() if target is None else target
    element_function_name = "%s_element" % function_name
    global_function_name = "%s_global" % function_name
    statements = expression_graph.evaluation_plan.statements
    temporary_symbols = set(expression_graph.evaluation_plan.temporary_symbols)
    input_symbols, output_targets = _kernel_io_symbols(statements, temporary_symbols)
    element_arguments = _kernel_arguments(
        input_symbols,
        output_targets,
        scalar_type,
        output_name,
    )
    kernel_arguments = _openmp_kernel_arguments(
        input_symbols,
        output_targets,
        scalar_type,
        index_type,
        output_name,
    )
    element_call_arguments = _openmp_element_call_arguments(
        input_symbols,
        output_targets,
        output_name,
    )
    wrapper_call_arguments = _openmp_wrapper_call_arguments(kernel_arguments)
    launch_arguments = tuple(arg for arg in wrapper_call_arguments if arg != "nelements")

    lines = ["#include <stddef.h>"]
    lines.extend(target.includes())
    lines.append("")
    lines.extend(
        _sfem_math_inline_source_lines(
            target.function_qualifier(),
            define_sfem_inline=False,
        )
    )
    lines.extend(
        [
            "",
            target.function_qualifier(),
            "void %s(%s)" % (element_function_name, ", ".join(element_arguments)),
            "{",
        ]
    )
    _append_statement_lines(lines, statements, scalar_type, output_name, indent="    ")
    lines.extend(
        [
            "}",
            "",
            'extern "C" __global__ void %s(%s)' % (global_function_name, ", ".join(kernel_arguments)),
            "{",
            "    for (%s e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {" % index_type,
            "        %s(%s);" % (element_function_name, ", ".join(element_call_arguments)),
            "    }",
            "}",
            "",
            'extern "C" void %s(%s)' % (function_name, ", ".join(kernel_arguments)),
            "{",
            "    const int block_size = 256;",
            "    const int grid_size = (int)((nelements + block_size - 1) / block_size);",
            "    %s<<<grid_size, block_size>>>(nelements%s%s);" % (
                global_function_name,
                ", " if launch_arguments else "",
                ", ".join(launch_arguments),
            ),
            "}",
            "",
        ]
    )
    return GeneratedKernelCode(target.generated_language, function_name, "\n".join(lines))


def _append_statement_lines(lines, statements, scalar_type, output_name, indent):
    for statement in statements:
        expression = _sfem_ccode(statement.expression, scalar_type)
        if statement.kind == "intermediate":
            target = _cpp_symbol(statement.target, output_name)
            lines.append("%sconst %s %s = %s;" % (indent, scalar_type, target, expression))
        elif statement.augmented:
            target = _cpp_lvalue(statement.target, output_name)
            lines.append("%s%s += %s;" % (indent, target, expression))
        else:
            target = _cpp_lvalue(statement.target, output_name)
            lines.append("%s%s = %s;" % (indent, target, expression))


def _kernel_io_symbols(statements, temporary_symbols):
    inputs = set()
    outputs = []
    output_set = set()

    for statement in statements:
        if statement.target not in temporary_symbols and statement.target not in output_set:
            outputs.append(statement.target)
            output_set.add(statement.target)

    for statement in statements:
        for dependency in statement.dependencies:
            if dependency not in temporary_symbols and dependency not in output_set:
                inputs.add(dependency)

    return tuple(sorted(inputs, key=str)), tuple(outputs)


def _kernel_arguments(input_symbols, output_targets, scalar_type, output_name):
    input_arrays, input_scalars = _group_kernel_symbols(input_symbols)
    direct_output_targets = tuple(
        target
        for target in output_targets
        if not (isinstance(target, str) and target.startswith("output:"))
    )
    needs_output_array = len(direct_output_targets) != len(output_targets)
    output_arrays, output_scalars = _group_kernel_symbols(direct_output_targets)
    arguments = []

    for base in sorted(input_arrays):
        arguments.append("const %s * const %s" % (scalar_type, base))
    for symbol in sorted(input_scalars, key=str):
        arguments.append("%s %s" % (scalar_type, _cpp_symbol(symbol, output_name)))
    for base in sorted(output_arrays):
        arguments.append("%s * const %s" % (scalar_type, base))
    for symbol in sorted(output_scalars, key=str):
        arguments.append("%s * const %s" % (scalar_type, _cpp_symbol(symbol, output_name)))

    if needs_output_array or (not output_arrays and not output_scalars):
        arguments.append("%s * const %s" % (scalar_type, output_name))

    return arguments


def _openmp_kernel_arguments(
    input_symbols,
    output_targets,
    scalar_type,
    index_type,
    output_name,
):
    input_arrays, input_scalars = _group_kernel_symbols(input_symbols)
    direct_output_targets, needs_output_array = _direct_output_targets(
        output_targets,
    )
    output_arrays, output_scalars = _group_kernel_symbols(direct_output_targets)
    arguments = ["%s nelements" % index_type]

    for base in sorted(input_arrays):
        arguments.append("const %s * const %s" % (scalar_type, base))
        arguments.append("%s %s_stride" % (index_type, base))
    for symbol in sorted(input_scalars, key=str):
        arguments.append("%s %s" % (scalar_type, _cpp_symbol(symbol, output_name)))
    for base in sorted(output_arrays):
        arguments.append("%s * const %s" % (scalar_type, base))
        arguments.append("%s %s_stride" % (index_type, base))
    for symbol in sorted(output_scalars, key=str):
        arguments.append("%s * const %s" % (scalar_type, _cpp_symbol(symbol, output_name)))

    if needs_output_array or (not output_arrays and not output_scalars):
        arguments.append("%s * const %s" % (scalar_type, output_name))
        arguments.append("%s %s_stride" % (index_type, output_name))

    return arguments


def _openmp_element_call_arguments(input_symbols, output_targets, output_name):
    input_arrays, input_scalars = _group_kernel_symbols(input_symbols)
    direct_output_targets, needs_output_array = _direct_output_targets(
        output_targets,
    )
    output_arrays, output_scalars = _group_kernel_symbols(direct_output_targets)
    arguments = []

    for base in sorted(input_arrays):
        arguments.append("%s + e * %s_stride" % (base, base))
    for symbol in sorted(input_scalars, key=str):
        arguments.append(_cpp_symbol(symbol, output_name))
    for base in sorted(output_arrays):
        arguments.append("%s + e * %s_stride" % (base, base))
    for symbol in sorted(output_scalars, key=str):
        arguments.append(_cpp_symbol(symbol, output_name))

    if needs_output_array or (not output_arrays and not output_scalars):
        arguments.append("%s + e * %s_stride" % (output_name, output_name))

    return arguments


def _openmp_wrapper_call_arguments(arguments):
    return tuple(_cpp_argument_name(argument) for argument in arguments)


def _cpp_wrapper_name(function_name):
    words = []
    for word in str(function_name).replace("-", "_").split("_"):
        if word:
            words.append(word[0].upper() + word[1:])
    return "%sOperator" % "".join(words)
