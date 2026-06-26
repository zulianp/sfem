from dataclasses import dataclass

import sympy as sp

from .residual import CoupledResidualSystem
from .tensor_product_geometry import (
    isoparametric_adjugate_lines,
    streams_in_shape_order,
    tensor_product_cartesian_shape_order,
    tensor_product_isoparametric_geometry_lines,
)
from .fem import sfem_element_quadrature_rule, sfem_soa_element_specialization
from .symbolic import (
    GeneratedKernelFile,
    KernelExpressions,
    _cpp_scalar_initializer_list,
    _sfem_ccode,
    _sfem_soa_diagnostic_print_wrapper_lines,
    _sfem_soa_diagnostics_header,
    _sfem_math_header_source,
)


@dataclass(frozen=True)
class WeakResidualCoefficients:
    row_field: str
    value: sp.Expr
    gradient: tuple


@dataclass(frozen=True)
class ResidualCodegenDependencies:
    current: bool
    previous: bool
    direction: bool
    parameters: tuple
    current_value: bool
    current_gradient: bool
    previous_value: bool
    previous_gradient: bool
    direction_value: bool
    direction_gradient: bool
    value_coefficients: tuple
    gradient_coefficients: tuple

    @property
    def uses_trial_gradients(self):
        return self.current_gradient or self.previous_gradient or self.direction_gradient

    @property
    def uses_test_gradients(self):
        return any(any(row) for row in self.gradient_coefficients)

    @property
    def uses_reference_gradients(self):
        return self.uses_trial_gradients or self.uses_test_gradients

    @property
    def uses_adjugate(self):
        return self.uses_reference_gradients


def weak_residual_coefficients(system, expression, row_field):
    field = system.field(row_field)
    expression = sp.sympify(expression)
    value = sp.diff(expression, field.test_value)
    gradient = tuple(
        sp.diff(expression, symbol) for symbol in field.test_gradient
    )
    test_symbols = set(field.test_symbols)
    if value.free_symbols.intersection(test_symbols) or any(
        coefficient.free_symbols.intersection(test_symbols)
        for coefficient in gradient
    ):
        raise ValueError(
            "residual for field '%s' must be linear in its test value and gradient"
            % field.name
        )
    if expression.xreplace({symbol: sp.S.Zero for symbol in test_symbols}) != 0:
        raise ValueError(
            "residual for field '%s' must not contain a test-independent term"
            % field.name
        )
    return WeakResidualCoefficients(field.name, value, gradient)


def coupled_residual_weak_coefficients(system, jacobian_action=False):
    coefficients = []
    if jacobian_action:
        blocks = {
            (block.row_field, block.column_field): block.expression
            for block in system.jacobian_blocks()
        }
        for row in system.fields:
            expression = sum(
                blocks[(row.name, column.name)] for column in system.fields
            )
            coefficients.append(
                weak_residual_coefficients(system, expression, row.name)
            )
    else:
        for row in system.fields:
            coefficients.append(
                weak_residual_coefficients(
                    system,
                    system.residual_expression(row),
                    row.name,
                )
            )
    return tuple(coefficients)


def _codegen_dependencies(system, coefficients, dependencies):
    free_symbols = set()
    for coefficient in coefficients:
        free_symbols.update(sp.sympify(coefficient.value).free_symbols)
        for expression in coefficient.gradient:
            free_symbols.update(sp.sympify(expression).free_symbols)
    current_value = any(field.value in free_symbols for field in system.fields)
    current_gradient = any(
        free_symbols.intersection(field.gradient) for field in system.fields
    )
    previous_value = any(
        field.previous_value is not None and field.previous_value in free_symbols
        for field in system.fields
    )
    previous_gradient = any(
        free_symbols.intersection(field.previous_gradient) for field in system.fields
    )
    direction_value = any(
        field.direction_value in free_symbols for field in system.fields
    )
    direction_gradient = any(
        free_symbols.intersection(field.direction_gradient) for field in system.fields
    )
    return ResidualCodegenDependencies(
        current=current_value or current_gradient,
        previous=previous_value or previous_gradient,
        direction=direction_value or direction_gradient,
        parameters=tuple(
            parameter for parameter in dependencies.parameters if parameter in free_symbols
        ),
        current_value=current_value,
        current_gradient=current_gradient,
        previous_value=previous_value,
        previous_gradient=previous_gradient,
        direction_value=direction_value,
        direction_gradient=direction_gradient,
        value_coefficients=tuple(
            not _is_zero(coefficient.value) for coefficient in coefficients
        ),
        gradient_coefficients=tuple(
            tuple(not _is_zero(expression) for expression in coefficient.gradient)
            for coefficient in coefficients
        ),
    )


def _is_zero(expression):
    expression = sp.sympify(expression)
    return expression == 0 or expression.is_zero is True


def generate_coupled_residual_sfem_files(
    system,
    *,
    prefix,
    element_type,
    vector_size=16,
    quadrature_order=None,
    specialization=None,
    residual_coeffs=None,
    action_coeffs=None,
):
    if not isinstance(system, CoupledResidualSystem):
        raise TypeError("system must be CoupledResidualSystem")
    if specialization is None:
        specialization = sfem_soa_element_specialization(
            element_type,
            vector_size,
            quadrature_order,
        )
    if system.dim != specialization.dim:
        raise ValueError("residual system dimension does not match element dimension")
    if residual_coeffs is None:
        residual_coeffs = coupled_residual_weak_coefficients(system, False)
    if action_coeffs is None:
        action_coeffs = coupled_residual_weak_coefficients(system, True)
    family = (
        "tensor_product"
        if specialization.quadrature_rule.is_tensor_product
        else "simplex"
    )
    local_prefix = "%s_d%d_%s" % (prefix, system.dim, family)
    element_prefix = "%s_%s" % (prefix, element_type.lower())
    local_name = "%s_local.hpp" % local_prefix
    operator_name = "%s_operator.cpp" % element_prefix
    local_source = _local_header(
        system,
        local_prefix,
        specialization,
        residual_coeffs,
        action_coeffs,
    )
    operator_source = _operator_source(
        system,
        element_prefix,
        local_prefix,
        specialization,
        local_name,
    )
    diagnostics_name = "kernel_diagnostics.hpp"
    return (
        GeneratedKernelFile("kernel_math.hpp", _sfem_math_header_source()),
        GeneratedKernelFile(
            diagnostics_name,
            "\n".join(_sfem_soa_diagnostics_header()),
        ),
        GeneratedKernelFile(local_name, local_source),
        GeneratedKernelFile(operator_name, operator_source),
    )


def generate_mixed_residual_sfem_files(
    system,
    *,
    prefix,
    compatible_element,
    vector_size=16,
    quadrature_order=None,
    residual_coeffs=None,
    action_coeffs=None,
    field_element_types=None,
):
    if not isinstance(system, CoupledResidualSystem):
        raise TypeError("system must be CoupledResidualSystem")
    cell_specialization = sfem_soa_element_specialization(
        compatible_element.cell_element_type,
        vector_size,
        quadrature_order,
    )
    if system.dim != cell_specialization.dim:
        raise ValueError("residual system dimension does not match element dimension")
    field_element_types = dict(field_element_types or ())
    missing_fields = tuple(
        field.name for field in system.fields if field.name not in field_element_types
    )
    if missing_fields:
        raise ValueError(
            "mixed residual code generation requires element types for field(s): %s"
            % ", ".join(missing_fields)
        )
    if residual_coeffs is None:
        residual_coeffs = coupled_residual_weak_coefficients(system, False)
    if action_coeffs is None:
        action_coeffs = coupled_residual_weak_coefficients(system, True)
    family = (
        "tensor_product"
        if cell_specialization.quadrature_rule.is_tensor_product
        else "simplex"
    )
    local_prefix = "%s_d%d_%s_mixed" % (prefix, system.dim, family)
    operator_name = "%s_%s_operator.cpp" % (prefix, compatible_element.name.lower())
    operator_source = _mixed_operator_source(
        system,
        prefix,
        local_prefix,
        cell_specialization,
        compatible_element,
        field_element_types,
        residual_coeffs,
        action_coeffs,
    )
    return (
        GeneratedKernelFile("kernel_math.hpp", _sfem_math_header_source()),
        GeneratedKernelFile(
            "kernel_diagnostics.hpp",
            "\n".join(_sfem_soa_diagnostics_header()),
        ),
        GeneratedKernelFile(operator_name, operator_source),
    )


def _local_header(system, local_prefix, specialization, residual_coeffs, action_coeffs):
    rule = specialization.quadrature_rule
    residual_dependencies = _codegen_dependencies(
        system,
        residual_coeffs,
        system.residual_dependencies(),
    )
    action_dependencies = _codegen_dependencies(
        system,
        action_coeffs,
        system.jacobian_action_dependencies(),
    )
    guard = ("%s_LOCAL_HPP" % local_prefix).upper()
    lines = [
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "",
        "#include <math.h>",
        "#include <stddef.h>",
        "#if defined(__has_include)",
        '#if __has_include("sfem_base.hpp")',
        '#include "sfem_base.hpp"',
        "#define SFEM_GENERATED_SCALAR_T",
        "#endif",
        "#endif",
        '#include "kernel_math.hpp"',
        "",
        "#ifndef SFEM_INLINE",
        "#define SFEM_INLINE inline",
        "#endif",
        "#ifndef SFEM_RESTRICT",
        "#define SFEM_RESTRICT",
        "#endif",
        "#ifndef SFEM_GENERATED_SCALAR_T",
        "#define SFEM_GENERATED_SCALAR_T",
        "typedef double real_t;",
        "typedef ptrdiff_t idx_t;",
        "typedef double geom_t;",
        "#endif",
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
    ]
    if rule.is_tensor_product:
        lines.extend(_tensor_helpers(local_prefix, system.dim))
        lines.append("")
    lines.extend(
        _local_function(
            system,
            "%s_residual_block" % local_prefix,
            specialization,
            residual_coeffs,
            dependencies=residual_dependencies,
            local_prefix=local_prefix,
        )
    )
    lines.append("")
    lines.extend(
        _local_function(
            system,
            "%s_jacobian_action_block" % local_prefix,
            specialization,
            action_coeffs,
            dependencies=action_dependencies,
            local_prefix=local_prefix,
        )
    )
    lines.extend(
        ["", "} // namespace codegen", "} // namespace sfem", "", "#endif", ""]
    )
    return "\n".join(lines)


def _local_function(
    system,
    function_name,
    specialization,
    coefficients,
    dependencies,
    local_prefix,
):
    rule = specialization.quadrature_rule
    dim = system.dim
    n_fields = len(system.fields)
    params = [
        "const ptrdiff_t nelems",
        "const ptrdiff_t geometry_stride",
        "const scalar_t *const SFEM_RESTRICT determinant",
    ]
    if dependencies.uses_adjugate:
        params.append(
            "const scalar_t *const SFEM_RESTRICT adjugate[%d]" % (dim * dim)
        )
    if rule.is_tensor_product:
        params.append("const scalar_t *const SFEM_RESTRICT shape_1d")
        if dependencies.uses_reference_gradients:
            params.append("const scalar_t *const SFEM_RESTRICT grad_1d")
        params.append("const scalar_t *const SFEM_RESTRICT q_weight_1d")
    else:
        params.append("const scalar_t *const SFEM_RESTRICT shape")
        if dependencies.uses_reference_gradients:
            params.extend(
                "const scalar_t *const SFEM_RESTRICT %s"
                % _simplex_grad_ref_name("grad_ref", d)
                for d in range(dim)
            )
        params.append("const scalar_t *const SFEM_RESTRICT q_weight")
    if dependencies.current:
        params.append(
            "const scalar_t *const SFEM_RESTRICT current[%d * N_SHAPE]"
            % n_fields
        )
    if dependencies.previous:
        params.append(
            "const scalar_t *const SFEM_RESTRICT previous[%d * N_SHAPE]"
            % n_fields
        )
    if dependencies.direction:
        params.append(
            "const scalar_t *const SFEM_RESTRICT direction[%d * N_SHAPE]"
            % n_fields
        )
    params.extend(
        "const scalar_t %s" % parameter for parameter in dependencies.parameters
    )
    params.append(
        "scalar_t *const SFEM_RESTRICT output[%d * N_SHAPE]" % n_fields
    )
    lines = [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>",
        "static SFEM_INLINE void %s(" % function_name,
    ]
    for index, param in enumerate(params):
        lines.append("        %s%s" % (param, "," if index + 1 < len(params) else ""))
    lines.extend(
        [
            ") {",
            "    static constexpr int DIM = %d;" % dim,
            "    static constexpr int N_FIELDS = %d;" % n_fields,
        ]
    )
    if rule.is_tensor_product:
        lines.extend(
            _tensor_local_body(
                system,
                local_prefix,
                coefficients,
                dependencies,
            )
        )
    else:
        lines.extend(_simplex_local_body(system, coefficients, dependencies))
    lines.append("}")
    return lines


def _simplex_local_body(system, coefficients, dependencies):
    dim = system.dim
    lines = [
        "    for (int q = 0; q < N_QP; ++q) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            const ptrdiff_t geometry_offset = q * geometry_stride + lane;",
        "            const scalar_t det = determinant[geometry_offset];",
    ]
    if dependencies.uses_adjugate:
        for i in range(dim * dim):
            lines.append(
                "            const scalar_t adj%d = adjugate[%d][geometry_offset];"
                % (i, i)
            )
    lines.extend(_field_evaluation_lines(system, dependencies, "            ", False))
    lines.extend(
        _coefficient_evaluation_lines(
            system,
            coefficients,
            "            ",
            "q_weight[q]",
            dependencies,
        )
    )
    lines.extend(
        [
            "            for (int test = 0; test < N_SHAPE; ++test) {",
            "                const scalar_t test_value = shape[q * N_SHAPE + test];",
        ]
    )
    for d in range(dim):
        if not any(row[d] for row in dependencies.gradient_coefficients):
            continue
        terms = [
            "%s[q * N_SHAPE + test] * adj%d"
            % (_simplex_grad_ref_name("grad_ref", k), k * dim + d)
            for k in range(dim)
        ]
        lines.append(
            "                const scalar_t test_grad%d = (%s) / det;"
            % (d, " + ".join(terms))
        )
    for row in range(len(system.fields)):
        terms = []
        if dependencies.value_coefficients[row]:
            terms.append("value_coeff%d * test_value" % row)
        terms.extend(
            "grad_coeff%d_%d * test_grad%d" % (row, d, d)
            for d in range(dim)
            if dependencies.gradient_coefficients[row][d]
        )
        if terms:
            lines.append(
                "                output[test * N_FIELDS + %d][lane] += q_weight[q] * det * (%s);"
                % (row, " + ".join(terms))
            )
    lines.extend(["            }", "        }", "    }"])
    return lines


def _tensor_local_body(system, prefix, coefficients, dependencies):
    dim = system.dim
    n_fields = len(system.fields)
    lines = []
    for group, enabled in (
        ("current", dependencies.current),
        ("previous", dependencies.previous),
        ("direction", dependencies.direction),
    ):
        if not enabled:
            continue
        uses_gradient = getattr(dependencies, "%s_gradient" % group)
        lines.extend(
            [
                "    scalar_t %s_value[N_FIELDS * N_QP * VECTOR_SIZE];" % group,
            ]
        )
        if uses_gradient:
            lines.append(
                "    scalar_t %s_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];"
                % group
            )
        if uses_gradient:
            lines.extend(
                [
                    "    %s_tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>("
                    % prefix,
                    "            nelems, shape_1d, grad_1d, %s, %s_value, %s_grad_ref);"
                    % (group, group, group),
                ]
            )
        else:
            lines.extend(
                [
                    "    %s_tensor_evaluate_value<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>("
                    % prefix,
                    "            nelems, shape_1d, %s, %s_value);" % (group, group),
                ]
            )
    lines.append("    scalar_t value_coeff[N_FIELDS * N_QP * VECTOR_SIZE];")
    if dependencies.uses_test_gradients:
        lines.append("    scalar_t grad_coeff_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];")
    lines.extend(
        [
            "    static constexpr int Q = %s_integer_root(N_QP, DIM);" % prefix,
            "    for (int q = 0; q < N_QP; ++q) {",
        ]
    )
    if dim == 2:
        lines.extend(
            [
                "        const int qx = q % Q;",
                "        const int qy = q / Q;",
                "        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy];",
            ]
        )
    else:
        lines.extend(
            [
                "        const int qx = q % Q;",
                "        const int qy = (q / Q) % Q;",
                "        const int qz = q / (Q * Q);",
                "        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];",
            ]
        )
    lines.extend(
        [
            "#pragma omp simd",
            "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
            "            const ptrdiff_t geometry_offset = q * geometry_stride + lane;",
            "            const scalar_t det = determinant[geometry_offset];",
        ]
    )
    if dependencies.uses_adjugate:
        for i in range(dim * dim):
            lines.append(
                "            const scalar_t adj%d = adjugate[%d][geometry_offset];"
                % (i, i)
            )
    lines.extend(_tensor_field_alias_lines(system, dependencies))
    lines.extend(_coefficient_evaluation_lines(system, coefficients, "            ", "qw", dependencies))
    for row in range(n_fields):
        if dependencies.value_coefficients[row]:
            value = "qw * det * value_coeff%d" % row
        else:
            value = "scalar_t(0)"
        lines.append(
            "            value_coeff[(%d * N_QP + q) * VECTOR_SIZE + lane] = %s;"
            % (row, value)
        )
        if dependencies.uses_test_gradients:
            for k in range(dim):
                terms = [
                    "adj%d * grad_coeff%d_%d" % (k * dim + d, row, d)
                    for d in range(dim)
                    if dependencies.gradient_coefficients[row][d]
                ]
                value = "qw * (%s)" % " + ".join(terms) if terms else "scalar_t(0)"
                lines.append(
                    "            grad_coeff_ref[((%d * N_QP + q) * DIM + %d) * VECTOR_SIZE + lane] = %s;"
                    % (row, k, value)
                )
    lines.extend(["        }", "    }"])
    if dependencies.uses_test_gradients:
        lines.extend(
            [
                "    %s_tensor_integrate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>("
                % prefix,
                "            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);",
            ]
        )
    else:
        lines.extend(
            [
                "    %s_tensor_integrate_value<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>("
                % prefix,
                "            nelems, shape_1d, value_coeff, output);",
            ]
        )
    return lines


def _field_evaluation_lines(system, dependencies, indent, tensor):
    if tensor:
        raise AssertionError("tensor aliases are emitted separately")
    dim = system.dim
    lines = []
    for field_index, field in enumerate(system.fields):
        groups = []
        if dependencies.current:
            groups.append(
                (
                    "",
                    "current",
                    dependencies.current_value,
                    dependencies.current_gradient,
                )
            )
        if dependencies.previous:
            groups.append(
                (
                    "_old",
                    "previous",
                    dependencies.previous_value,
                    dependencies.previous_gradient,
                )
            )
        for stem, stream, uses_value, uses_gradient in groups:
            if uses_value:
                lines.append("%sscalar_t %s%s = scalar_t(0);" % (indent, field.name, stem))
            if uses_gradient:
                for d in range(dim):
                    lines.append(
                        "%sscalar_t %s%s_grad_%d_ref = scalar_t(0);"
                        % (indent, field.name, stem, d)
                    )
            lines.append("%sfor (int trial = 0; trial < N_SHAPE; ++trial) {" % indent)
            lines.append(
                "%s    const scalar_t coeff = %s[trial * N_FIELDS + %d][lane];"
                % (indent, stream, field_index)
            )
            if uses_value:
                lines.append(
                    "%s    %s%s += coeff * shape[q * N_SHAPE + trial];"
                    % (indent, field.name, stem)
                )
            if uses_gradient:
                for d in range(dim):
                    lines.append(
                        "%s    %s%s_grad_%d_ref += coeff * %s[q * N_SHAPE + trial];"
                        % (
                            indent,
                            field.name,
                            stem,
                            d,
                            _simplex_grad_ref_name("grad_ref", d),
                        )
                    )
            lines.append("%s}" % indent)
            if uses_gradient:
                lines.extend(
                    _physical_gradient_lines(field.name + stem, dim, indent)
                )
        if dependencies.direction:
            if dependencies.direction_value:
                lines.append("%sscalar_t %s_direction = scalar_t(0);" % (indent, field.name))
            if dependencies.direction_gradient:
                for d in range(dim):
                    lines.append(
                        "%sscalar_t %s_direction_grad_%d_ref = scalar_t(0);"
                        % (indent, field.name, d)
                    )
            lines.append("%sfor (int trial = 0; trial < N_SHAPE; ++trial) {" % indent)
            lines.append(
                "%s    const scalar_t coeff = direction[trial * N_FIELDS + %d][lane];"
                % (indent, field_index)
            )
            if dependencies.direction_value:
                lines.append(
                    "%s    %s_direction += coeff * shape[q * N_SHAPE + trial];"
                    % (indent, field.name)
                )
            if dependencies.direction_gradient:
                for d in range(dim):
                    lines.append(
                        "%s    %s_direction_grad_%d_ref += coeff * %s[q * N_SHAPE + trial];"
                        % (
                            indent,
                            field.name,
                            d,
                            _simplex_grad_ref_name("grad_ref", d),
                        )
                    )
            lines.append("%s}" % indent)
            if dependencies.direction_gradient:
                lines.extend(
                    _physical_gradient_lines(field.name + "_direction", dim, indent)
                )
    return lines


def _physical_gradient_lines(stem, dim, indent):
    lines = []
    for d in range(dim):
        terms = [
            "%s_grad_%d_ref * adj%d" % (stem, k, k * dim + d)
            for k in range(dim)
        ]
        lines.append(
            "%sconst scalar_t %s_grad_%d = (%s) / det;"
            % (indent, stem, d, " + ".join(terms))
        )
    return lines


def _tensor_field_alias_lines(system, dependencies):
    dim = system.dim
    lines = []
    for field_index, field in enumerate(system.fields):
        groups = []
        if dependencies.current:
            groups.append(
                (
                    "",
                    "current",
                    dependencies.current_value,
                    dependencies.current_gradient,
                )
            )
        if dependencies.previous:
            groups.append(
                (
                    "_old",
                    "previous",
                    dependencies.previous_value,
                    dependencies.previous_gradient,
                )
            )
        for stem, array, uses_value, uses_gradient in groups:
            if uses_value:
                lines.append(
                    "            const scalar_t %s%s = %s_value[(%d * N_QP + q) * VECTOR_SIZE + lane];"
                    % (field.name, stem, array, field_index)
                )
            if uses_gradient:
                for k in range(dim):
                    lines.append(
                        "            const scalar_t %s%s_grad_%d_ref = %s_grad_ref[((%d * N_QP + q) * DIM + %d) * VECTOR_SIZE + lane];"
                        % (field.name, stem, k, array, field_index, k)
                    )
                lines.extend(
                    _physical_gradient_lines(field.name + stem, dim, "            ")
                )
        if dependencies.direction:
            if dependencies.direction_value:
                lines.append(
                    "            const scalar_t %s_direction = direction_value[(%d * N_QP + q) * VECTOR_SIZE + lane];"
                    % (field.name, field_index)
                )
            if dependencies.direction_gradient:
                for k in range(dim):
                    lines.append(
                        "            const scalar_t %s_direction_grad_%d_ref = direction_grad_ref[((%d * N_QP + q) * DIM + %d) * VECTOR_SIZE + lane];"
                        % (field.name, k, field_index, k)
                    )
                lines.extend(
                    _physical_gradient_lines(
                        field.name + "_direction", dim, "            "
                    )
                )
    return lines


def _coefficient_evaluation_lines(system, coefficients, indent, weight, dependencies=None):
    expressions = []
    targets = []
    for row, coefficient in enumerate(coefficients):
        if dependencies is None or dependencies.value_coefficients[row]:
            expressions.append(coefficient.value)
            targets.append("value_coeff%d" % row)
        for d, expression in enumerate(coefficient.gradient):
            if dependencies is None or dependencies.gradient_coefficients[row][d]:
                expressions.append(expression)
                targets.append("grad_coeff%d_%d" % (row, d))
    if not expressions:
        return []
    temporaries, reduced = sp.cse(
        expressions,
        symbols=sp.numbered_symbols("residual_tmp"),
    )
    lines = [
        "%sconst scalar_t %s = %s;" % (indent, symbol, _sfem_ccode(expression))
        for symbol, expression in temporaries
    ]
    lines.extend(
        "%sconst scalar_t %s = %s;"
        % (indent, target, _sfem_ccode(expression))
        for target, expression in zip(targets, reduced)
    )
    return lines


def _operator_source(system, prefix, local_prefix, specialization, local_name):
    rule = specialization.quadrature_rule
    dim = system.dim
    n_fields = len(system.fields)
    n_shape = rule.n_shape
    n_qp = rule.n_qp
    vector_size = specialization.vector_size
    element = rule.element_type.lower()
    lines = [
        '#include "%s"' % local_name,
        '#include "kernel_diagnostics.hpp"',
        "",
        "#ifndef SFEM_SUCCESS",
        "#define SFEM_SUCCESS 0",
        "#endif",
        "#ifndef MIN",
        "#define MIN(a, b) ((a) < (b) ? (a) : (b))",
        "#endif",
        "#ifdef _OPENMP",
        "#include <omp.h>",
        "#endif",
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
    ]
    lines.extend(_reference_data_lines(prefix, rule))
    lines.extend(["", "} // namespace codegen", "} // namespace sfem", ""])
    lines.extend(_residual_diagnostics_lines(system, prefix, specialization))
    lines.append("")
    form_dependencies = {
        "residual": _codegen_dependencies(
            system,
            coupled_residual_weak_coefficients(system, False),
            system.residual_dependencies(),
        ),
        "jacobian_action": _codegen_dependencies(
            system,
            coupled_residual_weak_coefficients(system, True),
            system.jacobian_action_dependencies(),
        ),
    }
    for form in ("residual", "jacobian_action"):
        dependencies = form_dependencies[form]
        function = "%s_%s_element_soa" % (prefix, form)
        block = "%s_%s_block" % (local_prefix, form)
        for scalar_type, suffix in (("double", ""), ("float", "_float")):
            reference_suffix = "f64" if scalar_type == "double" else "f32"
            params = [
                "const ptrdiff_t nelems",
                "const ptrdiff_t geometry_stride",
                "const %s *const SFEM_RESTRICT determinant" % scalar_type,
            ]
            if dependencies.uses_adjugate:
                params.append(
                    "const %s *const SFEM_RESTRICT adjugate[%d]"
                    % (scalar_type, dim * dim)
                )
            if dependencies.current:
                params.append(
                    "const %s *const SFEM_RESTRICT current[%d]"
                    % (scalar_type, n_fields * n_shape)
                )
            if dependencies.previous:
                params.append(
                    "const %s *const SFEM_RESTRICT previous[%d]"
                    % (scalar_type, n_fields * n_shape)
                )
            if dependencies.direction:
                params.append(
                    "const %s *const SFEM_RESTRICT direction[%d]"
                    % (scalar_type, n_fields * n_shape)
                )
            params.extend(
                "const %s %s" % (scalar_type, parameter)
                for parameter in dependencies.parameters
            )
            params.append(
                "%s *const SFEM_RESTRICT output[%d]"
                % (scalar_type, n_fields * n_shape)
            )
            lines.append('extern "C" int %s%s(' % (function, suffix))
            for index, param in enumerate(params):
                lines.append(
                    "        %s%s"
                    % (param, "," if index + 1 < len(params) else "")
                )
            call_args = ["nelems", "geometry_stride", "determinant"]
            if dependencies.uses_adjugate:
                call_args.append("adjugate")
            if rule.is_tensor_product:
                call_args.append(
                    "sfem::codegen::%s_%s_shape_1d_%s"
                    % (prefix, element, reference_suffix)
                )
                if dependencies.uses_reference_gradients:
                    call_args.append(
                        "sfem::codegen::%s_%s_grad_1d_%s"
                        % (prefix, element, reference_suffix)
                    )
                call_args.append(
                    "sfem::codegen::%s_%s_q_weight_1d_%s"
                    % (prefix, element, reference_suffix)
                )
            else:
                call_args.append(
                    "sfem::codegen::%s_%s_shape_%s"
                    % (prefix, element, reference_suffix)
                )
                if dependencies.uses_reference_gradients:
                    call_args.extend(
                        "sfem::codegen::%s_%s_%s_%s"
                        % (
                            prefix,
                            element,
                            _simplex_grad_ref_name("grad_ref", d),
                            reference_suffix,
                        )
                        for d in range(dim)
                    )
                call_args.append(
                    "sfem::codegen::%s_%s_q_weight_%s"
                    % (prefix, element, reference_suffix)
                )
            if dependencies.current:
                call_args.append("current")
            if dependencies.previous:
                call_args.append("previous")
            if dependencies.direction:
                call_args.append("direction")
            call_args.extend(map(str, dependencies.parameters))
            call_args.append("output")
            lines.extend(
                [
                    ") {",
                    "    sfem::codegen::%s<%s, %d, %d, %d>(%s);"
                    % (
                        block,
                        scalar_type,
                        n_qp,
                        n_shape,
                        vector_size,
                        ", ".join(call_args),
                    ),
                    "    return SFEM_SUCCESS;",
                    "}",
                    "",
                ]
            )
        lines.extend(
            _mesh_operator_source(
                system,
                prefix,
                local_prefix,
                specialization,
                form,
                dependencies,
            )
        )
        lines.extend(
            _aos_dispatch_source(
                system,
                prefix,
                form,
                dependencies,
            )
        )
    return "\n".join(lines)


def _mixed_operator_source(
    system,
    prefix,
    local_prefix,
    cell_specialization,
    compatible_element,
    field_element_types,
    residual_coeffs,
    action_coeffs,
):
    rule = cell_specialization.quadrature_rule
    element = compatible_element.name.lower()
    lines = [
        "#include <math.h>",
        "#include <stddef.h>",
        '#include "kernel_math.hpp"',
        '#include "kernel_diagnostics.hpp"',
        "",
        "#ifndef SFEM_SUCCESS",
        "#define SFEM_SUCCESS 0",
        "#endif",
        "#ifndef MIN",
        "#define MIN(a, b) ((a) < (b) ? (a) : (b))",
        "#endif",
        "#ifndef SFEM_RESTRICT",
        "#define SFEM_RESTRICT",
        "#endif",
        "#ifndef SFEM_INLINE",
        "#define SFEM_INLINE inline",
        "#endif",
        "#ifndef SFEM_GENERATED_SCALAR_T",
        "#define SFEM_GENERATED_SCALAR_T",
        "typedef double real_t;",
        "typedef ptrdiff_t idx_t;",
        "typedef double geom_t;",
        "#endif",
        "#ifdef _OPENMP",
        "#include <omp.h>",
        "#endif",
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
    ]
    lines.extend(_mixed_reference_data_lines(prefix, rule, system, field_element_types))
    lines.extend(["", "} // namespace codegen", "} // namespace sfem", ""])
    form_data = (
        ("residual", residual_coeffs, system.residual_dependencies()),
        ("jacobian_action", action_coeffs, system.jacobian_action_dependencies()),
    )
    for form, coefficients, dependencies in form_data:
        lines.extend(
            _mixed_isoparametric_function(
                system,
                prefix,
                element,
                rule,
                field_element_types,
                form,
                coefficients,
                dependencies,
            )
        )
    return "\n".join(lines)


def _mixed_reference_data_lines(prefix, cell_rule, system, field_element_types):
    data = list(_simplex_reference_gradient_data("cell_grad_ref", cell_rule))
    data.append(("q_weight", cell_rule.weights))
    for field in system.fields:
        element_type = field_element_types.get(field.name, cell_rule.element_type)
        shape, grad = _shape_data_for_element_at_cell_rule(element_type, cell_rule)
        data.append(("%s_shape" % field.name, shape))
        data.extend(
            _split_reference_gradient_data(
                "%s_grad_ref" % field.name,
                grad,
                cell_rule.n_qp,
                len(shape) // cell_rule.n_qp,
                cell_rule.dim,
            )
        )
    lines = []
    for scalar_type, suffix in (("double", "f64"), ("float", "f32")):
        for name, values in data:
            lines.append(
                "static const %s %s_%s_%s[%d] = {%s};"
                % (
                    scalar_type,
                    prefix,
                    name,
                    suffix,
                    len(values),
                    _cpp_scalar_initializer_list(values, scalar_type),
                )
            )
    return lines


def _mixed_isoparametric_function(
    system,
    prefix,
    element,
    cell_rule,
    field_element_types,
    form,
    coefficients,
    dependencies,
):
    dim = system.dim
    params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
        "const geom_t *const *const SFEM_RESTRICT points",
    ]
    params.extend("const scalar_t %s" % parameter for parameter in dependencies.parameters)
    if dependencies.current:
        params.append("const ptrdiff_t current_stride")
        params.extend("const scalar_t *const SFEM_RESTRICT %s_data" % field.name for field in system.fields)
    if dependencies.previous:
        params.append("const ptrdiff_t previous_stride")
        params.extend("const scalar_t *const SFEM_RESTRICT %s_old_data" % field.name for field in system.fields)
    if dependencies.direction:
        params.append("const ptrdiff_t direction_stride")
        params.extend("const scalar_t *const SFEM_RESTRICT %s_direction_data" % field.name for field in system.fields)
    params.append("const ptrdiff_t out_stride")
    params.extend("scalar_t *const SFEM_RESTRICT %s_out" % field.name for field in system.fields)
    impl = "%s_%s_%s_isoparametric_mesh_mixed_impl" % (prefix, element, form)
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename scalar_t>",
        "static SFEM_INLINE int %s(" % impl,
    ]
    for index, param in enumerate(params):
        lines.append("        %s%s" % (param, "," if index + 1 < len(params) else ""))
    lines.extend(
        [
            ") {",
            "    static constexpr int DIM = %d;" % dim,
            "    static constexpr int N_QP = %d;" % cell_rule.n_qp,
            "    static constexpr int CELL_N_SHAPE = %d;" % cell_rule.n_shape,
            "    (void)nnodes;",
            "#pragma omp parallel for schedule(static)",
            "    for (ptrdiff_t e = 0; e < nelements; ++e) {",
            "        idx_t ev[CELL_N_SHAPE];",
            "        for (int s = 0; s < CELL_N_SHAPE; ++s) ev[s] = elements[s][e];",
            "        for (int q = 0; q < N_QP; ++q) {",
        ]
    )
    lines.extend(_mixed_geometry_lines(prefix, cell_rule, dim))
    lines.extend(
        _mixed_field_eval_lines(
            prefix,
            system,
            cell_rule,
            field_element_types,
            dependencies,
        )
    )
    lines.extend(_coefficient_evaluation_lines(system, coefficients, "            ", "1"))
    lines.append("            const scalar_t qw_det = %s_q_weight_f64[q] * det;" % prefix)
    for row, field in enumerate(system.fields):
        n_shape = _field_n_shape(field, cell_rule, field_element_types)
        lines.append("            for (int test = 0; test < %d; ++test) {" % n_shape)
        lines.append(
            "                const scalar_t test_value = %s_%s_shape_f64[q * %d + test];"
            % (prefix, field.name, n_shape)
        )
        for d in range(dim):
            terms = [
                "%s_%s_%s_f64[q * %d + test] * adj%d"
                % (
                    prefix,
                    field.name,
                    _simplex_grad_ref_name("grad_ref", k),
                    n_shape,
                    k * dim + d,
                )
                for k in range(dim)
            ]
            lines.append(
                "                const scalar_t test_grad%d = (%s) / det;"
                % (d, " + ".join(terms))
            )
        terms = ["value_coeff%d * test_value" % row] + [
            "grad_coeff%d_%d * test_grad%d" % (row, d, d)
            for d in range(dim)
        ]
        lines.extend(
            [
                "#pragma omp atomic update",
                "                %s_out[ev[test] * out_stride] += qw_det * (%s);"
                % (field.name, " + ".join(terms)),
                "            }",
            ]
        )
    lines.extend(
        [
            "        }",
            "    }",
            "    return SFEM_SUCCESS;",
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )
    function = "%s_%s_%s_isoparametric_mesh_soa" % (prefix, element, form)
    for scalar_type, suffix in (("double", ""), ("float", "_float")):
        typed_params = [param.replace("scalar_t", scalar_type) for param in params]
        lines.append('extern "C" int %s%s(' % (function, suffix))
        for index, param in enumerate(typed_params):
            lines.append("        %s%s" % (param, "," if index + 1 < len(typed_params) else ""))
        call_args = ["nelements", "nnodes", "elements", "points"]
        call_args.extend(map(str, dependencies.parameters))
        if dependencies.current:
            call_args.append("current_stride")
            call_args.extend("%s_data" % field.name for field in system.fields)
        if dependencies.previous:
            call_args.append("previous_stride")
            call_args.extend("%s_old_data" % field.name for field in system.fields)
        if dependencies.direction:
            call_args.append("direction_stride")
            call_args.extend("%s_direction_data" % field.name for field in system.fields)
        call_args.append("out_stride")
        call_args.extend("%s_out" % field.name for field in system.fields)
        lines.extend(
            [
                ") {",
                "    return sfem::codegen::%s<%s>(%s);"
                % (impl, scalar_type, ", ".join(call_args)),
                "}",
                "",
            ]
        )
    return lines


def _mixed_geometry_lines(prefix, cell_rule, dim):
    lines = []
    for i in range(dim):
        for j in range(dim):
            terms = [
                "points[%d][ev[%d]] * %s_%s_f64[q * CELL_N_SHAPE + %d]"
                % (
                    i,
                    shape,
                    prefix,
                    _simplex_grad_ref_name("cell_grad_ref", j),
                    shape,
                )
                for shape in range(cell_rule.n_shape)
            ]
            lines.append(
                "            const scalar_t J%d%d = %s;"
                % (i, j, " + ".join(terms))
            )
    if dim == 2:
        lines.extend(
            [
                "            const scalar_t det = J00 * J11 - J01 * J10;",
                "            const scalar_t adj0 = J11;",
                "            const scalar_t adj1 = -J01;",
                "            const scalar_t adj2 = -J10;",
                "            const scalar_t adj3 = J00;",
            ]
        )
    elif dim == 3:
        lines.extend(
            [
                "            const scalar_t adj0 = J11 * J22 - J12 * J21;",
                "            const scalar_t adj1 = J02 * J21 - J01 * J22;",
                "            const scalar_t adj2 = J01 * J12 - J02 * J11;",
                "            const scalar_t adj3 = J12 * J20 - J10 * J22;",
                "            const scalar_t adj4 = J00 * J22 - J02 * J20;",
                "            const scalar_t adj5 = J02 * J10 - J00 * J12;",
                "            const scalar_t adj6 = J10 * J21 - J11 * J20;",
                "            const scalar_t adj7 = J01 * J20 - J00 * J21;",
                "            const scalar_t adj8 = J00 * J11 - J01 * J10;",
                "            const scalar_t det = J00 * adj0 + J01 * adj3 + J02 * adj6;",
            ]
        )
    else:
        raise ValueError("mixed residual codegen supports dimensions 2 and 3")
    return lines


def _mixed_field_eval_lines(prefix, system, cell_rule, field_element_types, dependencies):
    dim = system.dim
    lines = []
    groups = []
    if dependencies.current:
        groups.append(("", "_data", "current_stride"))
    if dependencies.previous:
        groups.append(("_old", "_old_data", "previous_stride"))
    if dependencies.direction:
        groups.append(("_direction", "_direction_data", "direction_stride"))
    for field in system.fields:
        n_shape = _field_n_shape(field, cell_rule, field_element_types)
        for suffix, pointer_suffix, stride in groups:
            value_terms = [
                "%s%s[ev[%d] * %s] * %s_%s_shape_f64[q * %d + %d]"
                % (field.name, pointer_suffix, s, stride, prefix, field.name, n_shape, s)
                for s in range(n_shape)
            ]
            lines.append(
                "            const scalar_t %s%s = %s;"
                % (field.name, suffix, " + ".join(value_terms))
            )
            for k in range(dim):
                grad_terms = [
                    "%s%s[ev[%d] * %s] * %s_%s_%s_f64[q * %d + %d]"
                    % (
                        field.name,
                        pointer_suffix,
                        s,
                        stride,
                        prefix,
                        field.name,
                        _simplex_grad_ref_name("grad_ref", k),
                        n_shape,
                        s,
                    )
                    for s in range(n_shape)
                ]
                lines.append(
                    "            const scalar_t %s%s_grad_%d_ref = %s;"
                    % (field.name, suffix, k, " + ".join(grad_terms))
                )
            lines.extend(
                _physical_gradient_lines(field.name + suffix, dim, "            ")
            )
    return lines


def _field_n_shape(field, cell_rule, field_element_types):
    element_type = field_element_types[field.name]
    return sfem_element_quadrature_rule(
        element_type,
        cell_rule.order if element_type in ("QUAD4", "HEX8") else None,
    ).n_shape


def _shape_data_for_element_at_cell_rule(element_type, cell_rule):
    element_type = str(element_type).upper()
    points = _cell_rule_points(cell_rule)
    if element_type == "TRI3":
        shape = []
        grad = []
        for x, y in points:
            shape.extend((1.0 - x - y, x, y))
            grad.extend((-1.0, -1.0, 1.0, 0.0, 0.0, 1.0))
        return tuple(shape), tuple(grad)
    if element_type == "TRI6":
        shape = []
        grad = []
        for x, y in points:
            l0 = 1.0 - x - y
            shape.extend(
                (
                    l0 * (2.0 * l0 - 1.0),
                    x * (2.0 * x - 1.0),
                    y * (2.0 * y - 1.0),
                    4.0 * x * l0,
                    4.0 * x * y,
                    4.0 * y * l0,
                )
            )
            grad.extend(_tri6_gradients_at(x, y))
        return tuple(shape), tuple(grad)
    if element_type == "TET4":
        shape = []
        grad = []
        for x, y, z in points:
            shape.extend((1.0 - x - y - z, x, y, z))
            grad.extend(
                (
                    -1.0, -1.0, -1.0,
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                )
            )
        return tuple(shape), tuple(grad)
    if element_type == "TET10":
        shape = []
        grad = []
        for x, y, z in points:
            l0 = 1.0 - x - y - z
            shape.extend(
                (
                    l0 * (2.0 * l0 - 1.0),
                    x * (2.0 * x - 1.0),
                    y * (2.0 * y - 1.0),
                    z * (2.0 * z - 1.0),
                    4.0 * x * l0,
                    4.0 * x * y,
                    4.0 * y * l0,
                    4.0 * z * l0,
                    4.0 * x * z,
                    4.0 * y * z,
                )
            )
            grad.extend(_tet10_gradients_at(x, y, z))
        return tuple(shape), tuple(grad)
    if element_type in ("HEX8", "HEX27"):
        order = 1 if element_type == "HEX8" else 2
        return _hex_lagrange_shape_gradients(points, order)
    raise ValueError("unsupported mixed residual field element '%s'" % element_type)


def _cell_rule_points(rule):
    if rule.element_type in ("TRI3",):
        return ((1.0 / 3.0, 1.0 / 3.0),)
    if rule.element_type == "TRI6":
        return (
            (1.0 / 6.0, 1.0 / 6.0),
            (2.0 / 3.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0),
        )
    if rule.element_type == "TET4":
        return ((0.25, 0.25, 0.25),)
    if rule.element_type == "TET10":
        a = 0.5854101966249685
        b = 0.1381966011250105
        return ((b, b, b), (a, b, b), (b, a, b), (b, b, a))
    if rule.element_type in ("HEX8", "HEX27"):
        pts_1d = _unit_interval_gauss_points(rule.order)
        return tuple((x, y, z) for z in pts_1d for y in pts_1d for x in pts_1d)
    raise ValueError("unsupported cell rule '%s'" % rule.element_type)


def _unit_interval_gauss_points(order):
    if order == 1:
        return (0.5,)
    if order == 2:
        offset = 0.5 / 3.0 ** 0.5
        return (0.5 - offset, 0.5 + offset)
    if order == 3:
        offset = 0.5 * (3.0 / 5.0) ** 0.5
        return (0.5 - offset, 0.5, 0.5 + offset)
    raise ValueError("unsupported tensor-product order %d" % order)


def _tri6_gradients_at(x, y):
    return (
        -3.0 + 4.0 * x + 4.0 * y,
        -3.0 + 4.0 * x + 4.0 * y,
        4.0 * x - 1.0,
        0.0,
        0.0,
        4.0 * y - 1.0,
        4.0 - 8.0 * x - 4.0 * y,
        -4.0 * x,
        4.0 * y,
        4.0 * x,
        -4.0 * y,
        4.0 - 4.0 * x - 8.0 * y,
    )


def _tet10_gradients_at(x, y, z):
    dx = (
        4.0 * x + 4.0 * y + 4.0 * z - 3.0,
        4.0 * x - 1.0,
        0.0,
        0.0,
        -8.0 * x - 4.0 * y - 4.0 * z + 4.0,
        4.0 * y,
        -4.0 * y,
        -4.0 * z,
        4.0 * z,
        0.0,
    )
    dy = (
        4.0 * x + 4.0 * y + 4.0 * z - 3.0,
        0.0,
        4.0 * y - 1.0,
        0.0,
        -4.0 * x,
        4.0 * x,
        -8.0 * y - 4.0 * x - 4.0 * z + 4.0,
        -4.0 * z,
        0.0,
        4.0 * z,
    )
    dz = (
        4.0 * x + 4.0 * y + 4.0 * z - 3.0,
        0.0,
        0.0,
        4.0 * z - 1.0,
        -4.0 * x,
        0.0,
        -4.0 * y,
        -8.0 * z - 4.0 * x - 4.0 * y + 4.0,
        4.0 * x,
        4.0 * y,
    )
    gradients = []
    for i in range(10):
        gradients.extend((dx[i], dy[i], dz[i]))
    return tuple(gradients)


def _hex_lagrange_shape_gradients(points, order):
    n = order + 1
    shape = []
    grad = []
    for x, y, z in points:
        values_x, grads_x = _lagrange_1d_at(x, order)
        values_y, grads_y = _lagrange_1d_at(y, order)
        values_z, grads_z = _lagrange_1d_at(z, order)
        shape_q = [None] * (n * n * n)
        grad_q = [None] * (n * n * n)
        for sz in range(n):
            for sy in range(n):
                for sx in range(n):
                    idx = _tensor_hex_shape_index(n, sx, sy, sz)
                    vx = values_x[sx]
                    vy = values_y[sy]
                    vz = values_z[sz]
                    shape_q[idx] = vx * vy * vz
                    grad_q[idx] = (
                        grads_x[sx] * vy * vz,
                        vx * grads_y[sy] * vz,
                        vx * vy * grads_z[sz],
                    )
        shape.extend(shape_q)
        for item in grad_q:
            grad.extend(item)
    return tuple(shape), tuple(grad)


def _lagrange_1d_at(x, order):
    if order == 1:
        return (1.0 - x, x), (-1.0, 1.0)
    if order == 2:
        return (
            2.0 * x * x - 3.0 * x + 1.0,
            4.0 * x - 4.0 * x * x,
            2.0 * x * x - x,
        ), (
            4.0 * x - 3.0,
            4.0 - 8.0 * x,
            4.0 * x - 1.0,
        )
    raise ValueError("unsupported 1D Lagrange order %d" % order)


def _tensor_hex_shape_index(n_shape_1d, sx, sy, sz):
    if n_shape_1d == 2:
        return (sx if sy == 0 else (3 if sx == 0 else 2)) + 4 * sz
    if n_shape_1d == 3:
        cartesian_to_hex27 = (
            0, 8, 1,
            11, 24, 9,
            3, 10, 2,
            16, 20, 17,
            23, 26, 21,
            19, 22, 18,
            4, 12, 5,
            15, 25, 13,
            7, 14, 6,
        )
        return cartesian_to_hex27[sx + 3 * (sy + 3 * sz)]
    raise ValueError("unsupported tensor-product hex order")


def _residual_diagnostics_lines(system, prefix, specialization):
    rule = specialization.quadrature_rule
    diagnostics = [
        (
            "%s_residual_element_soa" % prefix,
            system.build_residual_graph("residual_diagnostics_tmp").cost,
            system.residual_dependencies(),
        )
    ]
    block_expressions = system.jacobian_blocks()
    for block in block_expressions:
        graph = (
            KernelExpressions()
            .jacobian_action(block.expression, block.name)
            .build_graph(
                data_symbols=system.jacobian_action_data_symbols(),
                temporary_prefix="%s_diagnostics_tmp" % block.name,
            )
        )
        diagnostics.append(
            (
                "%s_%s" % (prefix, block.name),
                graph.cost,
                system.dependencies_for_expressions((block.expression,)),
            )
        )
    diagnostics.append(
        (
            "%s_jacobian_action_element_soa" % prefix,
            system.build_jacobian_action_graph(
                temporary_prefix="jacobian_action_diagnostics_tmp"
            ).cost,
            system.jacobian_action_dependencies(),
        )
    )
    lines = []
    for public_name, cost, dependencies in diagnostics:
        if lines:
            lines.append("")
        lines.extend(
            _kernel_diagnostics_lines(
                system,
                public_name,
                cost,
                specialization,
                dependencies,
            )
        )
    return lines


def _kernel_diagnostics_lines(
    system,
    public_name,
    cost,
    specialization,
    dependencies,
):
    rule = specialization.quadrature_rule
    n_fields = len(system.fields)
    field_streams = n_fields * rule.n_shape
    geometry_streams = system.dim * system.dim + 1
    if rule.is_tensor_product:
        reference_scalars = (
            len(rule.tensor_product_shape_values_1d)
            + len(rule.tensor_product_shape_gradients_1d)
        )
        quadrature_weight_scalars = len(rule.tensor_product_weights_1d)
    else:
        reference_scalars = (
            len(_simplex_shape_values(rule)) + len(rule.reference_gradients)
        )
        quadrature_weight_scalars = len(rule.weights)
    variable_name = "%s_diagnostics_data" % public_name
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "static const KernelDiagnostics %s = {" % variable_name,
        '    "%s",' % public_name,
        '    "%s",' % rule.element_type,
        "    %d," % system.dim,
        "    %d," % rule.n_qp,
        "    %d," % rule.n_shape,
        "    %d," % specialization.vector_size,
        "    %d," % rule.order,
        "    %d," % cost.adds,
        "    %d," % cost.muls,
        "    %d," % cost.divs,
        "    %d," % cost.sqrts,
        "    %d," % cost.pows,
        "    %d," % cost.exps,
        "    %d," % cost.logs,
        "    %d," % cost.trigs,
        "    %d," % cost.loads,
        "    %d," % cost.stores,
        "    %d," % cost.flops,
        "    %d," % cost.temporaries,
        "    %d," % cost.estimated_registers,
        "    %d," % geometry_streams,
        "    %d," % reference_scalars,
        "    %d," % quadrature_weight_scalars,
        "    %d," % len(dependencies.parameters),
        "    %d,"
        % (
            field_streams
            * (int(dependencies.current) + int(dependencies.previous))
        ),
        "    %d," % (field_streams if dependencies.direction else 0),
        "    %d," % field_streams,
        "    1,",
        "    1,",
        "    1.0,",
        "    1.0,",
        "    8.0,",
        "    12.0,",
        "    16.0,",
        "    20.0,",
        "    20.0,",
        "    24.0,",
        "    1.0,",
        "    1.0",
        "};",
        "",
        "} // namespace codegen",
        "} // namespace sfem",
        "",
        'extern "C" const sfem::codegen::KernelDiagnostics *%s_diagnostics(void) {'
        % public_name,
        "    return &sfem::codegen::%s;" % variable_name,
        "}",
        "",
        'extern "C" double %s_arithmetic_intensity(' % public_name,
        "        const ptrdiff_t nelements,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(",
        "            &sfem::codegen::%s," % variable_name,
        "            nelements, scalar_bytes, real_bytes, accumulator_bytes);",
        "}",
    ]
    function_names = [public_name]
    if public_name.endswith("_element_soa"):
        function_names.extend(
            (
                public_name.replace("_element_soa", "_affine_mesh_soa"),
                public_name.replace(
                    "_element_soa", "_isoparametric_mesh_soa"
                ),
            )
        )
    for function_name in function_names:
        for scalar_type in ("double", "float"):
            lines.append("")
            lines.extend(
                _sfem_soa_diagnostic_print_wrapper_lines(
                    function_name,
                    variable_name,
                    scalar_type,
                )
            )
    return lines


def _mesh_operator_source(
    system,
    prefix,
    local_prefix,
    specialization,
    form,
    dependencies,
):
    rule = specialization.quadrature_rule
    dim = system.dim
    n_fields = len(system.fields)
    n_shape = rule.n_shape
    n_qp = rule.n_qp
    vector_size = specialization.vector_size
    shape_order = (
        tensor_product_cartesian_shape_order(dim, n_shape)
        if rule.is_tensor_product
        else tuple(range(n_shape))
    )
    field_stream_order = streams_in_shape_order(
        tuple(range(n_fields * n_shape)),
        n_fields,
        shape_order,
    )
    impl = "%s_%s_affine_mesh_soa_impl" % (prefix, form)
    block = "%s_%s_block" % (local_prefix, form)
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename scalar_t>",
        "static SFEM_INLINE int %s(" % impl,
    ]
    params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
    ]
    if dependencies.uses_adjugate:
        params.extend(
            "const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate%d" % i
            for i in range(dim * dim)
        )
    params.append(
        "const scalar_t *const SFEM_RESTRICT g_jacobian_determinant0"
    )
    params.extend(
        "const scalar_t %s" % parameter for parameter in dependencies.parameters
    )
    if dependencies.current:
        params.append("const ptrdiff_t current_stride")
        params.extend(
            "const scalar_t *const SFEM_RESTRICT %s" % field.name
            for field in system.fields
        )
    if dependencies.previous:
        params.append("const ptrdiff_t previous_stride")
        params.extend(
            "const scalar_t *const SFEM_RESTRICT %s_old" % field.name
            for field in system.fields
        )
    if dependencies.direction:
        params.append("const ptrdiff_t direction_stride")
        params.extend(
            "const scalar_t *const SFEM_RESTRICT %s_direction" % field.name
            for field in system.fields
        )
    params.append("const ptrdiff_t out_stride")
    params.extend(
        "scalar_t *const SFEM_RESTRICT %s_out" % field.name
        for field in system.fields
    )
    for index, param in enumerate(params):
        lines.append(
            "        %s%s" % (param, "," if index + 1 < len(params) else "")
        )
    lines.extend(
        [
            ") {",
            "    static constexpr int DIM = %d;" % dim,
            "    static constexpr int N_QP = %d;" % n_qp,
            "    static constexpr int N_SHAPE = %d;" % n_shape,
            "    static constexpr int N_FIELDS = %d;" % n_fields,
            "    static constexpr int VECTOR_SIZE = %d;" % vector_size,
            "    (void)nnodes;",
        ]
    )
    lines.extend(_mesh_reference_data_lines(rule))
    lines.extend(
        [
            "",
            "#pragma omp parallel for schedule(static)",
            "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            "        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
            "        idx_t ev[VECTOR_SIZE * N_SHAPE];",
        ]
    )
    if dependencies.current:
        lines.append(
            "        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];"
        )
    if dependencies.previous:
        lines.append(
            "        scalar_t block_previous[N_FIELDS * N_SHAPE][VECTOR_SIZE];"
        )
    if dependencies.direction:
        lines.append(
            "        scalar_t block_direction[N_FIELDS * N_SHAPE][VECTOR_SIZE];"
        )
    lines.extend(
        [
            "        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];",
            "",
            "#pragma omp simd",
            "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        ]
    )
    for shape in range(n_shape):
        lines.append(
            "            ev[lane * N_SHAPE + %d] = elements[%d][evbegin + lane];"
            % (shape, shape)
        )
    lines.extend(["        }", "", "#pragma omp simd", "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {"])
    for shape in range(n_shape):
        for field_index, field in enumerate(system.fields):
            stream = shape * n_fields + field_index
            node = "ev[lane * N_SHAPE + %d]" % shape
            if dependencies.current:
                lines.append(
                    "            block_current[%d][lane] = %s[%s * current_stride];"
                    % (stream, field.name, node)
                )
            if dependencies.previous:
                lines.append(
                    "            block_previous[%d][lane] = %s_old[%s * previous_stride];"
                    % (stream, field.name, node)
                )
            if dependencies.direction:
                lines.append(
                    "            block_direction[%d][lane] = %s_direction[%s * direction_stride];"
                    % (stream, field.name, node)
                )
            lines.append("            block_output[%d][lane] = scalar_t(0);" % stream)
    lines.extend(["        }", ""])
    if dependencies.current:
        lines.append(
            "        const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {%s};"
            % ", ".join("block_current[%d]" % i for i in field_stream_order)
        )
    if dependencies.previous:
        lines.append(
            "        const scalar_t *const block_previous_streams[N_FIELDS * N_SHAPE] = {%s};"
            % ", ".join("block_previous[%d]" % i for i in field_stream_order)
        )
    if dependencies.direction:
        lines.append(
            "        const scalar_t *const block_direction_streams[N_FIELDS * N_SHAPE] = {%s};"
            % ", ".join(
                "block_direction[%d]" % i
                for i in field_stream_order
            )
        )
    lines.append(
        "        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {%s};"
        % ", ".join("block_output[%d]" % i for i in field_stream_order)
    )
    if dependencies.uses_adjugate:
        lines.append(
            "        const scalar_t *const block_adjugate[%d] = {%s};"
            % (
                dim * dim,
                ", ".join(
                    "g_jacobian_adjugate%d + evbegin" % i
                    for i in range(dim * dim)
                ),
            )
        )
    call_args = [
        "nelems",
        "0",
        "g_jacobian_determinant0 + evbegin",
    ]
    if dependencies.uses_adjugate:
        call_args.append("block_adjugate")
    if rule.is_tensor_product:
        call_args.append("shape_1d")
        if dependencies.uses_reference_gradients:
            call_args.append("grad_1d")
        call_args.append("q_weight_1d")
    else:
        call_args.append("shape")
        if dependencies.uses_reference_gradients:
            call_args.extend(
                _simplex_grad_ref_name("grad_ref", d) for d in range(dim)
            )
        call_args.append("q_weight")
    if dependencies.current:
        call_args.append("block_current_streams")
    if dependencies.previous:
        call_args.append("block_previous_streams")
    if dependencies.direction:
        call_args.append("block_direction_streams")
    call_args.extend(map(str, dependencies.parameters))
    call_args.append("block_output_streams")
    lines.extend(
        [
            "",
            "        %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
            % (block, ", ".join(call_args)),
            "",
            "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        ]
    )
    for shape in range(n_shape):
        for field_index, field in enumerate(system.fields):
            stream = shape * n_fields + field_index
            lines.extend(
                [
                    "#pragma omp atomic update",
                    "            %s_out[ev[lane * N_SHAPE + %d] * out_stride] += block_output[%d][lane];"
                    % (field.name, shape, stream),
                ]
            )
    lines.extend(
        [
            "        }",
            "    }",
            "",
            "    return SFEM_SUCCESS;",
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )
    function = "%s_%s_affine_mesh_soa" % (prefix, form)
    for scalar_type, suffix in (("double", ""), ("float", "_float")):
        typed_params = [
            param.replace("scalar_t", scalar_type) for param in params
        ]
        lines.append('extern "C" int %s%s(' % (function, suffix))
        for index, param in enumerate(typed_params):
            lines.append(
                "        %s%s"
                % (param, "," if index + 1 < len(typed_params) else "")
            )
        call_args = ["nelements", "nnodes", "elements"]
        if dependencies.uses_adjugate:
            call_args.extend(
                "g_jacobian_adjugate%d" % i for i in range(dim * dim)
            )
        call_args.append("g_jacobian_determinant0")
        call_args.extend(map(str, dependencies.parameters))
        if dependencies.current:
            call_args.append("current_stride")
            call_args.extend(field.name for field in system.fields)
        if dependencies.previous:
            call_args.append("previous_stride")
            call_args.extend("%s_old" % field.name for field in system.fields)
        if dependencies.direction:
            call_args.append("direction_stride")
            call_args.extend(
                "%s_direction" % field.name for field in system.fields
            )
        call_args.append("out_stride")
        call_args.extend("%s_out" % field.name for field in system.fields)
        lines.extend(
            [
                ") {",
                "    return sfem::codegen::%s<%s>(%s);"
                % (impl, scalar_type, ", ".join(call_args)),
                "}",
                "",
            ]
        )
    lines.extend(
        _isoparametric_mesh_operator_source(
            system,
            prefix,
            local_prefix,
            specialization,
            form,
            dependencies,
        )
    )
    return lines


def _aos_dispatch_source(system, prefix, form, dependencies):
    target = "%s_%s_isoparametric_mesh_soa" % (prefix, form)
    function = "%s_%s_isoparametric_mesh_aos" % (prefix, form)
    n_fields = len(system.fields)
    lines = []
    for scalar_type, suffix in (("double", ""), ("float", "_float")):
        params = [
            "const ptrdiff_t nelements",
            "const ptrdiff_t nnodes",
            "idx_t **const SFEM_RESTRICT elements",
            "const geom_t *const *const SFEM_RESTRICT points",
            "const %s *const SFEM_RESTRICT parameters" % scalar_type,
        ]
        if dependencies.current:
            params.append("const %s *const SFEM_RESTRICT current" % scalar_type)
        if dependencies.previous:
            params.append("const %s *const SFEM_RESTRICT previous" % scalar_type)
        if dependencies.direction:
            params.append(
                "const %s *const SFEM_RESTRICT direction" % scalar_type
            )
        params.append("%s *const SFEM_RESTRICT output" % scalar_type)
        lines.append('extern "C" int %s%s(' % (function, suffix))
        for index, param in enumerate(params):
            lines.append(
                "        %s%s" % (param, "," if index + 1 < len(params) else "")
            )
        call_args = ["nelements", "nnodes", "elements", "points"]
        call_args.extend(
            "parameters[%d]" % index
            for index, parameter in enumerate(system.parameters)
            if parameter in dependencies.parameters
        )
        if dependencies.current:
            call_args.append(str(n_fields))
            call_args.extend(
                "current + %d" % index
                for index in range(n_fields)
            )
        if dependencies.previous:
            call_args.append(str(n_fields))
            call_args.extend(
                "previous + %d" % index
                for index in range(n_fields)
            )
        if dependencies.direction:
            call_args.append(str(n_fields))
            call_args.extend(
                "direction + %d" % index
                for index in range(n_fields)
            )
        call_args.append(str(n_fields))
        call_args.extend(
            "output + %d" % index
            for index in range(n_fields)
        )
        lines.extend(
            [
                ") {",
                "    return %s%s(%s);"
                % (target, suffix, ", ".join(call_args)),
                "}",
                "",
            ]
        )
    return lines


def _isoparametric_mesh_operator_source(
    system,
    prefix,
    local_prefix,
    specialization,
    form,
    dependencies,
):
    rule = specialization.quadrature_rule
    dim = system.dim
    n_fields = len(system.fields)
    n_shape = rule.n_shape
    n_qp = rule.n_qp
    vector_size = specialization.vector_size
    shape_order = (
        tensor_product_cartesian_shape_order(dim, n_shape)
        if rule.is_tensor_product
        else tuple(range(n_shape))
    )
    field_stream_order = streams_in_shape_order(
        tuple(range(n_fields * n_shape)),
        n_fields,
        shape_order,
    )
    coordinate_stream_order = streams_in_shape_order(
        tuple(range(dim * n_shape)),
        dim,
        shape_order,
    )
    impl = "%s_%s_isoparametric_mesh_soa_impl" % (prefix, form)
    block = "%s_%s_block" % (local_prefix, form)
    params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
        "const geom_t *const *const SFEM_RESTRICT points",
    ]
    params.extend(
        "const scalar_t %s" % parameter for parameter in dependencies.parameters
    )
    if dependencies.current:
        params.append("const ptrdiff_t current_stride")
        params.extend(
            "const scalar_t *const SFEM_RESTRICT %s" % field.name
            for field in system.fields
        )
    if dependencies.previous:
        params.append("const ptrdiff_t previous_stride")
        params.extend(
            "const scalar_t *const SFEM_RESTRICT %s_old" % field.name
            for field in system.fields
        )
    if dependencies.direction:
        params.append("const ptrdiff_t direction_stride")
        params.extend(
            "const scalar_t *const SFEM_RESTRICT %s_direction" % field.name
            for field in system.fields
        )
    params.append("const ptrdiff_t out_stride")
    params.extend(
        "scalar_t *const SFEM_RESTRICT %s_out" % field.name
        for field in system.fields
    )
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename scalar_t>",
        "static SFEM_INLINE int %s(" % impl,
    ]
    for index, param in enumerate(params):
        lines.append(
            "        %s%s" % (param, "," if index + 1 < len(params) else "")
        )
    lines.extend(
        [
            ") {",
            "    static constexpr int DIM = %d;" % dim,
            "    static constexpr int N_QP = %d;" % n_qp,
            "    static constexpr int N_SHAPE = %d;" % n_shape,
            "    static constexpr int N_FIELDS = %d;" % n_fields,
            "    static constexpr int VECTOR_SIZE = %d;" % vector_size,
            "    (void)nnodes;",
        ]
    )
    lines.extend(_mesh_reference_data_lines(rule))
    lines.extend(
        [
            "",
            "#pragma omp parallel for schedule(static)",
            "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            "        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
            "        idx_t ev[VECTOR_SIZE * N_SHAPE];",
            "        scalar_t block_coordinates[%d * N_SHAPE][VECTOR_SIZE];"
            % dim,
            "        scalar_t block_adjugate_data[%d][N_QP * VECTOR_SIZE];"
            % (dim * dim),
            "        scalar_t block_determinant[N_QP * VECTOR_SIZE];",
        ]
    )
    if dependencies.current:
        lines.append(
            "        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];"
        )
    if dependencies.previous:
        lines.append(
            "        scalar_t block_previous[N_FIELDS * N_SHAPE][VECTOR_SIZE];"
        )
    if dependencies.direction:
        lines.append(
            "        scalar_t block_direction[N_FIELDS * N_SHAPE][VECTOR_SIZE];"
        )
    lines.extend(
        [
            "        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];",
            "",
            "#pragma omp simd",
            "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        ]
    )
    for shape in range(n_shape):
        lines.append(
            "            ev[lane * N_SHAPE + %d] = elements[%d][evbegin + lane];"
            % (shape, shape)
        )
    lines.extend(["        }", "", "#pragma omp simd", "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {"])
    for shape in range(n_shape):
        node = "ev[lane * N_SHAPE + %d]" % shape
        for d in range(dim):
            lines.append(
                "            block_coordinates[%d][lane] = points[%d][%s];"
                % (shape * dim + d, d, node)
            )
        for field_index, field in enumerate(system.fields):
            stream = shape * n_fields + field_index
            if dependencies.current:
                lines.append(
                    "            block_current[%d][lane] = %s[%s * current_stride];"
                    % (stream, field.name, node)
                )
            if dependencies.previous:
                lines.append(
                    "            block_previous[%d][lane] = %s_old[%s * previous_stride];"
                    % (stream, field.name, node)
                )
            if dependencies.direction:
                lines.append(
                    "            block_direction[%d][lane] = %s_direction[%s * direction_stride];"
                    % (stream, field.name, node)
                )
            lines.append("            block_output[%d][lane] = scalar_t(0);" % stream)
    lines.extend(
        [
            "        }",
        ]
    )
    if rule.is_tensor_product:
        def evaluator_lines(streams, gradient, indent):
            return [
                "%sscalar_t coordinate_value[DIM * N_QP * VECTOR_SIZE];"
                % indent,
                "%s%s_tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM>("
                % (indent, local_prefix),
                "%s        nelems, shape_1d, grad_1d, %s,"
                % (indent, streams),
                "%s        coordinate_value, %s);" % (indent, gradient),
            ]

        lines.append("")
        lines.extend(
            tensor_product_isoparametric_geometry_lines(
                dim=dim,
                n_shape=n_shape,
                coordinate_streams=[
                    "block_coordinates[%d]" % i
                    for i in coordinate_stream_order
                ],
                evaluator_lines=evaluator_lines,
                adjugate_target=lambda component, index: (
                    "block_adjugate_data[%d][%s]" % (component, index)
                ),
                determinant_target=lambda index: (
                    "block_determinant[%s]" % index
                ),
            )
        )
    else:
        lines.extend(
            [
                "",
                "        for (int q = 0; q < N_QP; ++q) {",
                "#pragma omp simd",
                "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
            ]
        )
        for i in range(dim):
            for j in range(dim):
                terms = [
                    "block_coordinates[%d][lane] * %s[q * N_SHAPE + %d]"
                    % (
                        shape * dim + i,
                        _simplex_grad_ref_name("grad_ref", j),
                        shape,
                    )
                    for shape in range(n_shape)
                ]
                lines.append(
                    "                const scalar_t J%d%d = %s;"
                    % (i, j, " + ".join(terms))
                )
        lines.extend(_isoparametric_geometry_assignment_lines(dim, "                "))
        lines.extend(["            }", "        }"])
    lines.append("")
    if dependencies.current:
        lines.append(
            "        const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {%s};"
            % ", ".join("block_current[%d]" % i for i in field_stream_order)
        )
    if dependencies.previous:
        lines.append(
            "        const scalar_t *const block_previous_streams[N_FIELDS * N_SHAPE] = {%s};"
            % ", ".join("block_previous[%d]" % i for i in field_stream_order)
        )
    if dependencies.direction:
        lines.append(
            "        const scalar_t *const block_direction_streams[N_FIELDS * N_SHAPE] = {%s};"
            % ", ".join(
                "block_direction[%d]" % i
                for i in field_stream_order
            )
        )
    lines.append(
        "        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {%s};"
        % ", ".join("block_output[%d]" % i for i in field_stream_order)
    )
    if dependencies.uses_adjugate:
        lines.append(
            "        const scalar_t *const block_adjugate[%d] = {%s};"
            % (
                dim * dim,
                ", ".join(
                    "block_adjugate_data[%d]" % i for i in range(dim * dim)
                ),
            )
        )
    call_args = [
        "nelems",
        "VECTOR_SIZE",
        "block_determinant",
    ]
    if dependencies.uses_adjugate:
        call_args.append("block_adjugate")
    if rule.is_tensor_product:
        call_args.append("shape_1d")
        if dependencies.uses_reference_gradients:
            call_args.append("grad_1d")
        call_args.append("q_weight_1d")
    else:
        call_args.append("shape")
        if dependencies.uses_reference_gradients:
            call_args.extend(
                _simplex_grad_ref_name("grad_ref", d) for d in range(dim)
            )
        call_args.append("q_weight")
    if dependencies.current:
        call_args.append("block_current_streams")
    if dependencies.previous:
        call_args.append("block_previous_streams")
    if dependencies.direction:
        call_args.append("block_direction_streams")
    call_args.extend(map(str, dependencies.parameters))
    call_args.append("block_output_streams")
    lines.extend(
        [
            "",
            "        %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
            % (block, ", ".join(call_args)),
            "",
            "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        ]
    )
    for shape in range(n_shape):
        for field_index, field in enumerate(system.fields):
            stream = shape * n_fields + field_index
            lines.extend(
                [
                    "#pragma omp atomic update",
                    "            %s_out[ev[lane * N_SHAPE + %d] * out_stride] += block_output[%d][lane];"
                    % (field.name, shape, stream),
                ]
            )
    lines.extend(
        [
            "        }",
            "    }",
            "",
            "    return SFEM_SUCCESS;",
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )
    function = "%s_%s_isoparametric_mesh_soa" % (prefix, form)
    for scalar_type, suffix in (("double", ""), ("float", "_float")):
        typed_params = [
            param.replace("scalar_t", scalar_type) for param in params
        ]
        lines.append('extern "C" int %s%s(' % (function, suffix))
        for index, param in enumerate(typed_params):
            lines.append(
                "        %s%s"
                % (param, "," if index + 1 < len(typed_params) else "")
            )
        call_args = ["nelements", "nnodes", "elements", "points"]
        call_args.extend(map(str, dependencies.parameters))
        if dependencies.current:
            call_args.append("current_stride")
            call_args.extend(field.name for field in system.fields)
        if dependencies.previous:
            call_args.append("previous_stride")
            call_args.extend("%s_old" % field.name for field in system.fields)
        if dependencies.direction:
            call_args.append("direction_stride")
            call_args.extend(
                "%s_direction" % field.name for field in system.fields
            )
        call_args.append("out_stride")
        call_args.extend("%s_out" % field.name for field in system.fields)
        lines.extend(
            [
                ") {",
                "    return sfem::codegen::%s<%s>(%s);"
                % (impl, scalar_type, ", ".join(call_args)),
                "}",
                "",
            ]
        )
    return lines


def _isoparametric_geometry_assignment_lines(dim, indent):
    return isoparametric_adjugate_lines(
        dim,
        indent,
        "q * VECTOR_SIZE + lane",
        lambda component, index: (
            "block_adjugate_data[%d][%s]" % (component, index)
        ),
        lambda index: "block_determinant[%s]" % index,
    )


def _mesh_reference_data_lines(rule):
    if rule.is_tensor_product:
        data = (
            ("shape_1d", rule.tensor_product_shape_values_1d),
            ("grad_1d", rule.tensor_product_shape_gradients_1d),
            ("q_weight_1d", rule.tensor_product_weights_1d),
        )
    else:
        data = (
            (("shape", _simplex_shape_values(rule)),)
            + _simplex_reference_gradient_data("grad_ref", rule)
            + (("q_weight", rule.weights),)
        )
    return [
        "    static const scalar_t %s[%d] = {%s};"
        % (name, len(values), _cpp_scalar_initializer_list(values, "scalar_t"))
        for name, values in data
    ]


def _reference_data_lines(prefix, rule):
    element = rule.element_type.lower()
    if rule.is_tensor_product:
        data = (
            ("shape_1d", rule.tensor_product_shape_values_1d),
            ("grad_1d", rule.tensor_product_shape_gradients_1d),
            ("q_weight_1d", rule.tensor_product_weights_1d),
        )
    else:
        data = (
            (("shape", _simplex_shape_values(rule)),)
            + _simplex_reference_gradient_data("grad_ref", rule)
            + (("q_weight", rule.weights),)
        )
    lines = []
    for scalar_type, suffix in (("double", "f64"), ("float", "f32")):
        for name, values in data:
            lines.append(
                "static const %s %s_%s_%s_%s[%d] = {%s};"
                % (
                    scalar_type,
                    prefix,
                    element,
                    name,
                    suffix,
                    len(values),
                    _cpp_scalar_initializer_list(values, scalar_type),
                )
            )
    return lines


def _simplex_shape_values(rule):
    if rule.element_type == "TRI3":
        return (1.0 / 3.0,) * 3
    if rule.element_type == "TET4":
        return (0.25,) * 4
    raise ValueError(
        "simplex residual lowering currently supports TRI3 and TET4"
    )


def _simplex_grad_ref_name(prefix, component):
    names = ("x", "y", "z")
    if component < 0 or component >= len(names):
        raise ValueError("unsupported simplex gradient component %d" % component)
    return "%s_%s" % (prefix, names[component])


def _split_reference_gradient_data(prefix, gradients, n_qp, n_shape, dim):
    components = []
    for d in range(dim):
        values = []
        for q in range(n_qp):
            for shape in range(n_shape):
                values.append(gradients[(q * n_shape + shape) * dim + d])
        components.append((_simplex_grad_ref_name(prefix, d), tuple(values)))
    return tuple(components)


def _simplex_reference_gradient_data(prefix, rule):
    return _split_reference_gradient_data(
        prefix,
        rule.reference_gradients,
        rule.n_qp,
        rule.n_shape,
        rule.dim,
    )


def _tensor_helpers(prefix, dim):
    lines = [
        "static constexpr int %s_ipow(const int b, const int e) {" % prefix,
        "    return e == 0 ? 1 : b * %s_ipow(b, e - 1);" % prefix,
        "}",
        "static constexpr int %s_integer_root_search(const int v, const int e, const int c) {"
        % prefix,
        "    return %s_ipow(c, e) >= v ? c : %s_integer_root_search(v, e, c + 1);"
        % (prefix, prefix),
        "}",
        "static constexpr int %s_integer_root(const int v, const int e) {"
        % prefix,
        "    return %s_integer_root_search(v, e, 1);" % prefix,
        "}",
    ]
    lines.append("")
    lines.extend(_tensor_evaluate_helper(prefix, dim))
    lines.append("")
    lines.extend(_tensor_value_evaluate_helper(prefix, dim))
    lines.append("")
    lines.extend(_tensor_integrate_helper(prefix, dim))
    lines.append("")
    lines.extend(_tensor_value_integrate_helper(prefix, dim))
    return lines


def _tensor_evaluate_helper(prefix, dim):
    if dim == 2:
        return _tensor_evaluate_2d(prefix)
    return _tensor_evaluate_3d(prefix)


def _tensor_value_evaluate_helper(prefix, dim):
    if dim == 2:
        return _tensor_value_evaluate_2d(prefix)
    return _tensor_value_evaluate_3d(prefix)


def _tensor_evaluate_2d(prefix):
    return [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int N_FIELDS>",
        "static SFEM_INLINE void %s_tensor_evaluate(" % prefix,
        "        const ptrdiff_t nelems, const scalar_t *const shape_1d, const scalar_t *const grad_1d,",
        "        const scalar_t *const SFEM_RESTRICT streams[N_FIELDS * N_SHAPE], scalar_t *const value, scalar_t *const gradient) {",
        "    static constexpr int Q = %s_integer_root(N_QP, 2);" % prefix,
        "    static constexpr int S = %s_integer_root(N_SHAPE, 2);" % prefix,
        "    scalar_t vx[N_FIELDS * Q * S * VECTOR_SIZE];",
        "    scalar_t gx[N_FIELDS * Q * S * VECTOR_SIZE];",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t v = scalar_t(0), g = scalar_t(0);",
        "            for (int sx = 0; sx < S; ++sx) {",
        "                const int s = sx + S * sy;",
        "                const scalar_t u = streams[s * N_FIELDS + f][lane];",
        "                v += u * shape_1d[qx * S + sx]; g += u * grad_1d[qx * S + sx];",
        "            }",
        "            const int i = ((f * Q + qx) * S + sy) * VECTOR_SIZE + lane;",
        "            vx[i] = v; gx[i] = g;",
        "        }",
        "    }",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qy = 0; qy < Q; ++qy) for (int qx = 0; qx < Q; ++qx) {",
        "        const int q = qx + Q * qy;",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t v = scalar_t(0), g0 = scalar_t(0), g1 = scalar_t(0);",
        "            for (int sy = 0; sy < S; ++sy) {",
        "                const int i = ((f * Q + qx) * S + sy) * VECTOR_SIZE + lane;",
        "                v += vx[i] * shape_1d[qy * S + sy];",
        "                g0 += gx[i] * shape_1d[qy * S + sy];",
        "                g1 += vx[i] * grad_1d[qy * S + sy];",
        "            }",
        "            value[(f * N_QP + q) * VECTOR_SIZE + lane] = v;",
        "            gradient[((f * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane] = g0;",
        "            gradient[((f * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane] = g1;",
        "        }",
        "    }",
        "}",
    ]


def _tensor_value_evaluate_2d(prefix):
    return [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int N_FIELDS>",
        "static SFEM_INLINE void %s_tensor_evaluate_value(" % prefix,
        "        const ptrdiff_t nelems, const scalar_t *const shape_1d,",
        "        const scalar_t *const SFEM_RESTRICT streams[N_FIELDS * N_SHAPE], scalar_t *const value) {",
        "    static constexpr int Q = %s_integer_root(N_QP, 2);" % prefix,
        "    static constexpr int S = %s_integer_root(N_SHAPE, 2);" % prefix,
        "    scalar_t vx[N_FIELDS * Q * S * VECTOR_SIZE];",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t v = scalar_t(0);",
        "            for (int sx = 0; sx < S; ++sx) {",
        "                const int s = sx + S * sy;",
        "                v += streams[s * N_FIELDS + f][lane] * shape_1d[qx * S + sx];",
        "            }",
        "            vx[((f * Q + qx) * S + sy) * VECTOR_SIZE + lane] = v;",
        "        }",
        "    }",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qy = 0; qy < Q; ++qy) for (int qx = 0; qx < Q; ++qx) {",
        "        const int q = qx + Q * qy;",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t v = scalar_t(0);",
        "            for (int sy = 0; sy < S; ++sy) {",
        "                v += vx[((f * Q + qx) * S + sy) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy];",
        "            }",
        "            value[(f * N_QP + q) * VECTOR_SIZE + lane] = v;",
        "        }",
        "    }",
        "}",
    ]


def _tensor_evaluate_3d(prefix):
    return [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int N_FIELDS>",
        "static SFEM_INLINE void %s_tensor_evaluate(" % prefix,
        "        const ptrdiff_t nelems, const scalar_t *const shape_1d, const scalar_t *const grad_1d,",
        "        const scalar_t *const SFEM_RESTRICT streams[N_FIELDS * N_SHAPE], scalar_t *const value, scalar_t *const gradient) {",
        "    static constexpr int Q = %s_integer_root(N_QP, 3);" % prefix,
        "    static constexpr int S = %s_integer_root(N_SHAPE, 3);" % prefix,
        "    scalar_t vx[N_FIELDS * Q * S * S * VECTOR_SIZE], gx[N_FIELDS * Q * S * S * VECTOR_SIZE];",
        "    scalar_t vxy[N_FIELDS * Q * Q * S * VECTOR_SIZE], g0xy[N_FIELDS * Q * Q * S * VECTOR_SIZE], g1xy[N_FIELDS * Q * Q * S * VECTOR_SIZE];",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) for (int sz = 0; sz < S; ++sz) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t v = scalar_t(0), g = scalar_t(0);",
        "            for (int sx = 0; sx < S; ++sx) {",
        "                const int s = sx + S * (sy + S * sz);",
        "                const scalar_t u = streams[s * N_FIELDS + f][lane];",
        "                v += u * shape_1d[qx * S + sx]; g += u * grad_1d[qx * S + sx];",
        "            }",
        "            const int i = (((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane;",
        "            vx[i] = v; gx[i] = g;",
        "        }",
        "    }",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int qy = 0; qy < Q; ++qy) for (int sz = 0; sz < S; ++sz) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t v = scalar_t(0), g0 = scalar_t(0), g1 = scalar_t(0);",
        "            for (int sy = 0; sy < S; ++sy) {",
        "                const int i = (((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane;",
        "                v += vx[i] * shape_1d[qy * S + sy]; g0 += gx[i] * shape_1d[qy * S + sy]; g1 += vx[i] * grad_1d[qy * S + sy];",
        "            }",
        "            const int j = (((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane;",
        "            vxy[j] = v; g0xy[j] = g0; g1xy[j] = g1;",
        "        }",
        "    }",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qz = 0; qz < Q; ++qz) for (int qy = 0; qy < Q; ++qy) for (int qx = 0; qx < Q; ++qx) {",
        "        const int q = qx + Q * (qy + Q * qz);",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t v = scalar_t(0), g0 = scalar_t(0), g1 = scalar_t(0), g2 = scalar_t(0);",
        "            for (int sz = 0; sz < S; ++sz) {",
        "                const int j = (((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane;",
        "                v += vxy[j] * shape_1d[qz * S + sz]; g0 += g0xy[j] * shape_1d[qz * S + sz];",
        "                g1 += g1xy[j] * shape_1d[qz * S + sz]; g2 += vxy[j] * grad_1d[qz * S + sz];",
        "            }",
        "            value[(f * N_QP + q) * VECTOR_SIZE + lane] = v;",
        "            gradient[((f * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = g0;",
        "            gradient[((f * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = g1;",
        "            gradient[((f * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = g2;",
        "        }",
        "    }",
        "}",
    ]


def _tensor_value_evaluate_3d(prefix):
    return [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int N_FIELDS>",
        "static SFEM_INLINE void %s_tensor_evaluate_value(" % prefix,
        "        const ptrdiff_t nelems, const scalar_t *const shape_1d,",
        "        const scalar_t *const SFEM_RESTRICT streams[N_FIELDS * N_SHAPE], scalar_t *const value) {",
        "    static constexpr int Q = %s_integer_root(N_QP, 3);" % prefix,
        "    static constexpr int S = %s_integer_root(N_SHAPE, 3);" % prefix,
        "    scalar_t vx[N_FIELDS * Q * S * S * VECTOR_SIZE];",
        "    scalar_t vxy[N_FIELDS * Q * Q * S * VECTOR_SIZE];",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) for (int sz = 0; sz < S; ++sz) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t v = scalar_t(0);",
        "            for (int sx = 0; sx < S; ++sx) {",
        "                const int s = sx + S * (sy + S * sz);",
        "                v += streams[s * N_FIELDS + f][lane] * shape_1d[qx * S + sx];",
        "            }",
        "            vx[(((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane] = v;",
        "        }",
        "    }",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int qy = 0; qy < Q; ++qy) for (int sz = 0; sz < S; ++sz) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t v = scalar_t(0);",
        "            for (int sy = 0; sy < S; ++sy) {",
        "                v += vx[(((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy];",
        "            }",
        "            vxy[(((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane] = v;",
        "        }",
        "    }",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qz = 0; qz < Q; ++qz) for (int qy = 0; qy < Q; ++qy) for (int qx = 0; qx < Q; ++qx) {",
        "        const int q = qx + Q * (qy + Q * qz);",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t v = scalar_t(0);",
        "            for (int sz = 0; sz < S; ++sz) {",
        "                v += vxy[(((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz];",
        "            }",
        "            value[(f * N_QP + q) * VECTOR_SIZE + lane] = v;",
        "        }",
        "    }",
        "}",
    ]


def _tensor_integrate_helper(prefix, dim):
    if dim == 2:
        return _tensor_integrate_2d(prefix)
    return _tensor_integrate_3d(prefix)


def _tensor_value_integrate_helper(prefix, dim):
    if dim == 2:
        return _tensor_value_integrate_2d(prefix)
    return _tensor_value_integrate_3d(prefix)


def _tensor_integrate_2d(prefix):
    return [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int N_FIELDS>",
        "static SFEM_INLINE void %s_tensor_integrate(" % prefix,
        "        const ptrdiff_t nelems, const scalar_t *const shape_1d, const scalar_t *const grad_1d,",
        "        const scalar_t *const value_coeff, const scalar_t *const grad_coeff, scalar_t *const SFEM_RESTRICT output[N_FIELDS * N_SHAPE]) {",
        "    static constexpr int Q = %s_integer_root(N_QP, 2), S = %s_integer_root(N_SHAPE, 2);"
        % (prefix, prefix),
        "    scalar_t sv[N_FIELDS * Q * S * VECTOR_SIZE], sg[N_FIELDS * Q * S * VECTOR_SIZE];",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t a = scalar_t(0), b = scalar_t(0);",
        "            for (int qy = 0; qy < Q; ++qy) { const int q = qx + Q * qy;",
        "                a += value_coeff[(f * N_QP + q) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy]",
        "                   + grad_coeff[((f * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane] * grad_1d[qy * S + sy];",
        "                b += grad_coeff[((f * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy]; }",
        "            const int i = ((f * Q + qx) * S + sy) * VECTOR_SIZE + lane; sv[i] = a; sg[i] = b;",
        "        }",
        "    }",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int sy = 0; sy < S; ++sy) for (int sx = 0; sx < S; ++sx) {",
        "        const int s = sx + S * sy;",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) { scalar_t v = scalar_t(0);",
        "            for (int qx = 0; qx < Q; ++qx) { const int i = ((f * Q + qx) * S + sy) * VECTOR_SIZE + lane;",
        "                v += sv[i] * shape_1d[qx * S + sx] + sg[i] * grad_1d[qx * S + sx]; }",
        "            output[s * N_FIELDS + f][lane] += v;",
        "        }",
        "    }",
        "}",
    ]


def _tensor_value_integrate_2d(prefix):
    return [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int N_FIELDS>",
        "static SFEM_INLINE void %s_tensor_integrate_value(" % prefix,
        "        const ptrdiff_t nelems, const scalar_t *const shape_1d,",
        "        const scalar_t *const value_coeff, scalar_t *const SFEM_RESTRICT output[N_FIELDS * N_SHAPE]) {",
        "    static constexpr int Q = %s_integer_root(N_QP, 2), S = %s_integer_root(N_SHAPE, 2);"
        % (prefix, prefix),
        "    scalar_t sv[N_FIELDS * Q * S * VECTOR_SIZE];",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t a = scalar_t(0);",
        "            for (int qy = 0; qy < Q; ++qy) {",
        "                const int q = qx + Q * qy;",
        "                a += value_coeff[(f * N_QP + q) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy];",
        "            }",
        "            sv[((f * Q + qx) * S + sy) * VECTOR_SIZE + lane] = a;",
        "        }",
        "    }",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int sy = 0; sy < S; ++sy) for (int sx = 0; sx < S; ++sx) {",
        "        const int s = sx + S * sy;",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t v = scalar_t(0);",
        "            for (int qx = 0; qx < Q; ++qx) {",
        "                v += sv[((f * Q + qx) * S + sy) * VECTOR_SIZE + lane] * shape_1d[qx * S + sx];",
        "            }",
        "            output[s * N_FIELDS + f][lane] += v;",
        "        }",
        "    }",
        "}",
    ]


def _tensor_integrate_3d(prefix):
    return [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int N_FIELDS>",
        "static SFEM_INLINE void %s_tensor_integrate(" % prefix,
        "        const ptrdiff_t nelems, const scalar_t *const shape_1d, const scalar_t *const grad_1d,",
        "        const scalar_t *const value_coeff, const scalar_t *const grad_coeff, scalar_t *const SFEM_RESTRICT output[N_FIELDS * N_SHAPE]) {",
        "    static constexpr int Q = %s_integer_root(N_QP, 3), S = %s_integer_root(N_SHAPE, 3);"
        % (prefix, prefix),
        "    scalar_t z0[N_FIELDS * Q * Q * S * VECTOR_SIZE], z1[N_FIELDS * Q * Q * S * VECTOR_SIZE], z2[N_FIELDS * Q * Q * S * VECTOR_SIZE];",
        "    scalar_t yz0[N_FIELDS * Q * S * S * VECTOR_SIZE], yz1[N_FIELDS * Q * S * S * VECTOR_SIZE];",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int qy = 0; qy < Q; ++qy) for (int sz = 0; sz < S; ++sz) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) { scalar_t a = scalar_t(0), b = scalar_t(0), c = scalar_t(0);",
        "            for (int qz = 0; qz < Q; ++qz) { const int q = qx + Q * (qy + Q * qz);",
        "                a += value_coeff[(f * N_QP + q) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz]",
        "                   + grad_coeff[((f * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] * grad_1d[qz * S + sz];",
        "                b += grad_coeff[((f * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz];",
        "                c += grad_coeff[((f * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz]; }",
        "            const int i = (((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane; z0[i] = a; z1[i] = b; z2[i] = c;",
        "        }",
        "    }",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) for (int sz = 0; sz < S; ++sz) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) { scalar_t a = scalar_t(0), b = scalar_t(0);",
        "            for (int qy = 0; qy < Q; ++qy) { const int i = (((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane;",
        "                a += z0[i] * shape_1d[qy * S + sy] + z2[i] * grad_1d[qy * S + sy];",
        "                b += z1[i] * shape_1d[qy * S + sy]; }",
        "            const int j = (((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane; yz0[j] = a; yz1[j] = b;",
        "        }",
        "    }",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int sz = 0; sz < S; ++sz) for (int sy = 0; sy < S; ++sy) for (int sx = 0; sx < S; ++sx) {",
        "        const int s = sx + S * (sy + S * sz);",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) { scalar_t v = scalar_t(0);",
        "            for (int qx = 0; qx < Q; ++qx) { const int j = (((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane;",
        "                v += yz0[j] * shape_1d[qx * S + sx] + yz1[j] * grad_1d[qx * S + sx]; }",
        "            output[s * N_FIELDS + f][lane] += v;",
        "        }",
        "    }",
        "}",
    ]


def _tensor_value_integrate_3d(prefix):
    return [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int N_FIELDS>",
        "static SFEM_INLINE void %s_tensor_integrate_value(" % prefix,
        "        const ptrdiff_t nelems, const scalar_t *const shape_1d,",
        "        const scalar_t *const value_coeff, scalar_t *const SFEM_RESTRICT output[N_FIELDS * N_SHAPE]) {",
        "    static constexpr int Q = %s_integer_root(N_QP, 3), S = %s_integer_root(N_SHAPE, 3);"
        % (prefix, prefix),
        "    scalar_t z0[N_FIELDS * Q * Q * S * VECTOR_SIZE];",
        "    scalar_t yz0[N_FIELDS * Q * S * S * VECTOR_SIZE];",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int qy = 0; qy < Q; ++qy) for (int sz = 0; sz < S; ++sz) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t a = scalar_t(0);",
        "            for (int qz = 0; qz < Q; ++qz) {",
        "                const int q = qx + Q * (qy + Q * qz);",
        "                a += value_coeff[(f * N_QP + q) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz];",
        "            }",
        "            z0[(((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane] = a;",
        "        }",
        "    }",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) for (int sz = 0; sz < S; ++sz) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t a = scalar_t(0);",
        "            for (int qy = 0; qy < Q; ++qy) {",
        "                a += z0[(((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy];",
        "            }",
        "            yz0[(((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane] = a;",
        "        }",
        "    }",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int sz = 0; sz < S; ++sz) for (int sy = 0; sy < S; ++sy) for (int sx = 0; sx < S; ++sx) {",
        "        const int s = sx + S * (sy + S * sz);",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t v = scalar_t(0);",
        "            for (int qx = 0; qx < Q; ++qx) {",
        "                v += yz0[(((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane] * shape_1d[qx * S + sx];",
        "            }",
        "            output[s * N_FIELDS + f][lane] += v;",
        "        }",
        "    }",
        "}",
    ]
