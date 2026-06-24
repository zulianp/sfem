from dataclasses import dataclass

import sympy as sp

from .residual import CoupledResidualSystem
from .symbolic import (
    GeneratedKernelFile,
    _cpp_scalar_initializer_list,
    _sfem_ccode,
    _sfem_math_header_source,
    sfem_soa_element_specialization,
)


@dataclass(frozen=True)
class WeakResidualCoefficients:
    row_field: str
    value: sp.Expr
    gradient: tuple


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
                    system.residual(row),
                    row.name,
                )
            )
    return tuple(coefficients)


def generate_coupled_residual_sfem_files(
    system,
    *,
    prefix,
    element_type,
    vector_size=16,
    quadrature_order=None,
):
    if not isinstance(system, CoupledResidualSystem):
        raise TypeError("system must be CoupledResidualSystem")
    specialization = sfem_soa_element_specialization(
        element_type,
        vector_size,
        quadrature_order,
    )
    if system.dim != specialization.dim:
        raise ValueError("residual system dimension does not match element dimension")
    residual_coeffs = coupled_residual_weak_coefficients(system, False)
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
    return (
        GeneratedKernelFile("kernel_math.hpp", _sfem_math_header_source()),
        GeneratedKernelFile(local_name, local_source),
        GeneratedKernelFile(operator_name, operator_source),
    )


def _local_header(system, local_prefix, specialization, residual_coeffs, action_coeffs):
    rule = specialization.quadrature_rule
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
            has_direction=False,
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
            has_direction=True,
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
    has_direction,
    local_prefix,
):
    rule = specialization.quadrature_rule
    dim = system.dim
    n_fields = len(system.fields)
    params = [
        "const ptrdiff_t nelems",
        "const ptrdiff_t geometry_stride",
        "const scalar_t *const SFEM_RESTRICT adjugate[%d]" % (dim * dim),
        "const scalar_t *const SFEM_RESTRICT determinant",
    ]
    if rule.is_tensor_product:
        params.extend(
            (
                "const scalar_t *const SFEM_RESTRICT shape_1d",
                "const scalar_t *const SFEM_RESTRICT grad_1d",
                "const scalar_t *const SFEM_RESTRICT q_weight_1d",
            )
        )
    else:
        params.extend(
            (
                "const scalar_t *const SFEM_RESTRICT shape",
                "const scalar_t *const SFEM_RESTRICT grad_ref",
                "const scalar_t *const SFEM_RESTRICT q_weight",
            )
        )
    params.extend(
        (
            "const scalar_t *const SFEM_RESTRICT current[%d * N_SHAPE]"
            % n_fields,
            "const scalar_t *const SFEM_RESTRICT previous[%d * N_SHAPE]"
            % n_fields,
        )
    )
    if has_direction:
        params.append(
            "const scalar_t *const SFEM_RESTRICT direction[%d * N_SHAPE]"
            % n_fields
        )
    params.extend(
        "const scalar_t %s" % parameter for parameter in system.parameters
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
                has_direction,
            )
        )
    else:
        lines.extend(_simplex_local_body(system, coefficients, has_direction))
    lines.append("}")
    return lines


def _simplex_local_body(system, coefficients, has_direction):
    dim = system.dim
    lines = [
        "    for (int q = 0; q < N_QP; ++q) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            const ptrdiff_t geometry_offset = q * geometry_stride + lane;",
        "            const scalar_t det = determinant[geometry_offset];",
    ]
    for i in range(dim * dim):
        lines.append(
            "            const scalar_t adj%d = adjugate[%d][geometry_offset];"
            % (i, i)
        )
    lines.extend(_field_evaluation_lines(system, has_direction, "            ", False))
    lines.extend(
        _coefficient_evaluation_lines(
            system,
            coefficients,
            "            ",
            "q_weight[q]",
        )
    )
    lines.extend(
        [
            "            for (int test = 0; test < N_SHAPE; ++test) {",
            "                const scalar_t test_value = shape[q * N_SHAPE + test];",
        ]
    )
    for d in range(dim):
        terms = [
            "grad_ref[(q * N_SHAPE + test) * DIM + %d] * adj%d"
            % (k, k * dim + d)
            for k in range(dim)
        ]
        lines.append(
            "                const scalar_t test_grad%d = (%s) / det;"
            % (d, " + ".join(terms))
        )
    for row in range(len(system.fields)):
        terms = ["value_coeff%d * test_value" % row] + [
            "grad_coeff%d_%d * test_grad%d" % (row, d, d)
            for d in range(dim)
        ]
        lines.append(
            "                output[test * N_FIELDS + %d][lane] += q_weight[q] * det * (%s);"
            % (row, " + ".join(terms))
        )
    lines.extend(["            }", "        }", "    }"])
    return lines


def _tensor_local_body(system, prefix, coefficients, has_direction):
    dim = system.dim
    n_fields = len(system.fields)
    lines = [
        "    scalar_t current_value[N_FIELDS * N_QP * VECTOR_SIZE];",
        "    scalar_t current_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];",
        "    scalar_t previous_value[N_FIELDS * N_QP * VECTOR_SIZE];",
        "    scalar_t previous_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];",
        "    %s_tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>("
        % prefix,
        "            nelems, shape_1d, grad_1d, current, current_value, current_grad_ref);",
        "    %s_tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>("
        % prefix,
        "            nelems, shape_1d, grad_1d, previous, previous_value, previous_grad_ref);",
    ]
    if has_direction:
        lines.extend(
            [
                "    scalar_t direction_value[N_FIELDS * N_QP * VECTOR_SIZE];",
                "    scalar_t direction_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];",
                "    %s_tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>("
                % prefix,
                "            nelems, shape_1d, grad_1d, direction, direction_value, direction_grad_ref);",
            ]
        )
    lines.extend(
        [
            "    scalar_t value_coeff[N_FIELDS * N_QP * VECTOR_SIZE];",
            "    scalar_t grad_coeff_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];",
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
    for i in range(dim * dim):
        lines.append(
            "            const scalar_t adj%d = adjugate[%d][geometry_offset];"
            % (i, i)
        )
    lines.extend(_tensor_field_alias_lines(system, has_direction))
    lines.extend(_coefficient_evaluation_lines(system, coefficients, "            ", "qw"))
    for row in range(n_fields):
        lines.append(
            "            value_coeff[(%d * N_QP + q) * VECTOR_SIZE + lane] = qw * det * value_coeff%d;"
            % (row, row)
        )
        for k in range(dim):
            terms = [
                "adj%d * grad_coeff%d_%d" % (k * dim + d, row, d)
                for d in range(dim)
            ]
            lines.append(
                "            grad_coeff_ref[((%d * N_QP + q) * DIM + %d) * VECTOR_SIZE + lane] = qw * (%s);"
                % (row, k, " + ".join(terms))
            )
    lines.extend(
        [
            "        }",
            "    }",
            "    %s_tensor_integrate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>("
            % prefix,
            "            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);",
        ]
    )
    return lines


def _field_evaluation_lines(system, has_direction, indent, tensor):
    if tensor:
        raise AssertionError("tensor aliases are emitted separately")
    dim = system.dim
    lines = []
    for field_index, field in enumerate(system.fields):
        for stem, stream in (("", "current"), ("_old", "previous")):
            lines.append("%sscalar_t %s%s = 0;" % (indent, field.name, stem))
            for d in range(dim):
                lines.append(
                    "%sscalar_t %s%s_grad_%d_ref = 0;"
                    % (indent, field.name, stem, d)
                )
            lines.append("%sfor (int trial = 0; trial < N_SHAPE; ++trial) {" % indent)
            lines.append(
                "%s    const scalar_t coeff = %s[trial * N_FIELDS + %d][lane];"
                % (indent, stream, field_index)
            )
            lines.append(
                "%s    %s%s += coeff * shape[q * N_SHAPE + trial];"
                % (indent, field.name, stem)
            )
            for d in range(dim):
                lines.append(
                    "%s    %s%s_grad_%d_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + %d];"
                    % (indent, field.name, stem, d, d)
                )
            lines.append("%s}" % indent)
            lines.extend(
                _physical_gradient_lines(field.name + stem, dim, indent)
            )
        if has_direction:
            lines.append("%sscalar_t %s_direction = 0;" % (indent, field.name))
            for d in range(dim):
                lines.append(
                    "%sscalar_t %s_direction_grad_%d_ref = 0;"
                    % (indent, field.name, d)
                )
            lines.append("%sfor (int trial = 0; trial < N_SHAPE; ++trial) {" % indent)
            lines.append(
                "%s    const scalar_t coeff = direction[trial * N_FIELDS + %d][lane];"
                % (indent, field_index)
            )
            lines.append(
                "%s    %s_direction += coeff * shape[q * N_SHAPE + trial];"
                % (indent, field.name)
            )
            for d in range(dim):
                lines.append(
                    "%s    %s_direction_grad_%d_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + %d];"
                    % (indent, field.name, d, d)
                )
            lines.append("%s}" % indent)
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


def _tensor_field_alias_lines(system, has_direction):
    dim = system.dim
    lines = []
    for field_index, field in enumerate(system.fields):
        for stem, array in (("", "current"), ("_old", "previous")):
            lines.append(
                "            const scalar_t %s%s = %s_value[(%d * N_QP + q) * VECTOR_SIZE + lane];"
                % (field.name, stem, array, field_index)
            )
            for k in range(dim):
                lines.append(
                    "            const scalar_t %s%s_grad_%d_ref = %s_grad_ref[((%d * N_QP + q) * DIM + %d) * VECTOR_SIZE + lane];"
                    % (field.name, stem, k, array, field_index, k)
                )
            lines.extend(
                _physical_gradient_lines(field.name + stem, dim, "            ")
            )
        if has_direction:
            lines.append(
                "            const scalar_t %s_direction = direction_value[(%d * N_QP + q) * VECTOR_SIZE + lane];"
                % (field.name, field_index)
            )
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


def _coefficient_evaluation_lines(system, coefficients, indent, weight):
    expressions = []
    targets = []
    for row, coefficient in enumerate(coefficients):
        expressions.append(coefficient.value)
        targets.append("value_coeff%d" % row)
        for d, expression in enumerate(coefficient.gradient):
            expressions.append(expression)
            targets.append("grad_coeff%d_%d" % (row, d))
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
    for form, has_direction in (("residual", False), ("jacobian_action", True)):
        function = "%s_%s_element_soa" % (prefix, form)
        block = "%s_%s_block" % (local_prefix, form)
        for scalar_type, suffix in (("double", ""), ("float", "_float")):
            reference_suffix = "f64" if scalar_type == "double" else "f32"
            params = [
                "const ptrdiff_t nelems",
                "const ptrdiff_t geometry_stride",
                "const %s *const SFEM_RESTRICT adjugate[%d]"
                % (scalar_type, dim * dim),
                "const %s *const SFEM_RESTRICT determinant" % scalar_type,
                "const %s *const SFEM_RESTRICT current[%d]"
                % (scalar_type, n_fields * n_shape),
                "const %s *const SFEM_RESTRICT previous[%d]"
                % (scalar_type, n_fields * n_shape),
            ]
            if has_direction:
                params.append(
                    "const %s *const SFEM_RESTRICT direction[%d]"
                    % (scalar_type, n_fields * n_shape)
                )
            params.extend(
                "const %s %s" % (scalar_type, parameter)
                for parameter in system.parameters
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
            call_args = ["nelems", "geometry_stride", "adjugate", "determinant"]
            if rule.is_tensor_product:
                call_args.extend(
                    (
                        "sfem::codegen::%s_%s_shape_1d_%s"
                        % (prefix, element, reference_suffix),
                        "sfem::codegen::%s_%s_grad_1d_%s"
                        % (prefix, element, reference_suffix),
                        "sfem::codegen::%s_%s_q_weight_1d_%s"
                        % (prefix, element, reference_suffix),
                    )
                )
            else:
                call_args.extend(
                    (
                        "sfem::codegen::%s_%s_shape_%s"
                        % (prefix, element, reference_suffix),
                        "sfem::codegen::%s_%s_grad_ref_%s"
                        % (prefix, element, reference_suffix),
                        "sfem::codegen::%s_%s_q_weight_%s"
                        % (prefix, element, reference_suffix),
                    )
                )
            call_args.extend(("current", "previous"))
            if has_direction:
                call_args.append("direction")
            call_args.extend(map(str, system.parameters))
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
                has_direction,
            )
        )
    return "\n".join(lines)


def _mesh_operator_source(
    system,
    prefix,
    local_prefix,
    specialization,
    form,
    has_direction,
):
    rule = specialization.quadrature_rule
    dim = system.dim
    n_fields = len(system.fields)
    n_shape = rule.n_shape
    n_qp = rule.n_qp
    vector_size = specialization.vector_size
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
    params.extend(
        "const scalar_t *const SFEM_RESTRICT g_jacobian_adjugate%d" % i
        for i in range(dim * dim)
    )
    params.append(
        "const scalar_t *const SFEM_RESTRICT g_jacobian_determinant0"
    )
    params.extend(
        "const scalar_t %s" % parameter for parameter in system.parameters
    )
    params.append("const ptrdiff_t current_stride")
    params.extend(
        "const scalar_t *const SFEM_RESTRICT %s" % field.name
        for field in system.fields
    )
    params.append("const ptrdiff_t previous_stride")
    params.extend(
        "const scalar_t *const SFEM_RESTRICT %s_old" % field.name
        for field in system.fields
    )
    if has_direction:
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
            "        scalar_t block_current[N_FIELDS * N_SHAPE][VECTOR_SIZE];",
            "        scalar_t block_previous[N_FIELDS * N_SHAPE][VECTOR_SIZE];",
        ]
    )
    if has_direction:
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
            lines.append(
                "            block_current[%d][lane] = %s[%s * current_stride];"
                % (stream, field.name, node)
            )
            lines.append(
                "            block_previous[%d][lane] = %s_old[%s * previous_stride];"
                % (stream, field.name, node)
            )
            if has_direction:
                lines.append(
                    "            block_direction[%d][lane] = %s_direction[%s * direction_stride];"
                    % (stream, field.name, node)
                )
            lines.append("            block_output[%d][lane] = 0;" % stream)
    lines.extend(
        [
            "        }",
            "",
            "        const scalar_t *const block_current_streams[N_FIELDS * N_SHAPE] = {%s};"
            % ", ".join(
                "block_current[%d]" % i for i in range(n_fields * n_shape)
            ),
            "        const scalar_t *const block_previous_streams[N_FIELDS * N_SHAPE] = {%s};"
            % ", ".join(
                "block_previous[%d]" % i for i in range(n_fields * n_shape)
            ),
        ]
    )
    if has_direction:
        lines.append(
            "        const scalar_t *const block_direction_streams[N_FIELDS * N_SHAPE] = {%s};"
            % ", ".join(
                "block_direction[%d]" % i
                for i in range(n_fields * n_shape)
            )
        )
    lines.extend(
        [
            "        scalar_t *const block_output_streams[N_FIELDS * N_SHAPE] = {%s};"
            % ", ".join(
                "block_output[%d]" % i for i in range(n_fields * n_shape)
            ),
            "        const scalar_t *const block_adjugate[%d] = {%s};"
            % (
                dim * dim,
                ", ".join(
                    "g_jacobian_adjugate%d + evbegin" % i
                    for i in range(dim * dim)
                ),
            ),
        ]
    )
    call_args = [
        "nelems",
        "0",
        "block_adjugate",
        "g_jacobian_determinant0 + evbegin",
    ]
    if rule.is_tensor_product:
        call_args.extend(("shape_1d", "grad_1d", "q_weight_1d"))
    else:
        call_args.extend(("shape", "grad_ref", "q_weight"))
    call_args.extend(("block_current_streams", "block_previous_streams"))
    if has_direction:
        call_args.append("block_direction_streams")
    call_args.extend(map(str, system.parameters))
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
        call_args.extend(
            "g_jacobian_adjugate%d" % i for i in range(dim * dim)
        )
        call_args.append("g_jacobian_determinant0")
        call_args.extend(map(str, system.parameters))
        call_args.append("current_stride")
        call_args.extend(field.name for field in system.fields)
        call_args.append("previous_stride")
        call_args.extend("%s_old" % field.name for field in system.fields)
        if has_direction:
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


def _mesh_reference_data_lines(rule):
    if rule.is_tensor_product:
        data = (
            ("shape_1d", rule.tensor_product_shape_values_1d),
            ("grad_1d", rule.tensor_product_shape_gradients_1d),
            ("q_weight_1d", rule.tensor_product_weights_1d),
        )
    else:
        data = (
            ("shape", _simplex_shape_values(rule)),
            ("grad_ref", rule.reference_gradients),
            ("q_weight", rule.weights),
        )
    return [
        "    static const scalar_t %s[%d] = {%s};"
        % (name, len(values), _cpp_scalar_initializer_list(values))
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
            ("shape", _simplex_shape_values(rule)),
            ("grad_ref", rule.reference_gradients),
            ("q_weight", rule.weights),
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
                    _cpp_scalar_initializer_list(values),
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


def _tensor_helpers(prefix, dim):
    # The generated helper performs staged 1D contractions with lane-contiguous
    # scratch arrays. It supports Q1 and lexicographically ordered higher order
    # tensor elements.
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
        "",
        "template <int S>",
        "static SFEM_INLINE int %s_tensor_index(const int sx, const int sy, const int sz = 0) {"
        % prefix,
    ]
    if dim == 2:
        lines.append(
            "    return S == 2 ? sx + sy * (3 - 2 * sx) : sx + S * sy;"
        )
    else:
        lines.append(
            "    return S == 2 ? sx + sy * (3 - 2 * sx) + 4 * sz : sx + S * (sy + S * sz);"
        )
    lines.extend(["}", ""])
    lines.extend(_tensor_evaluate_helper(prefix, dim))
    lines.append("")
    lines.extend(_tensor_integrate_helper(prefix, dim))
    return lines


def _tensor_evaluate_helper(prefix, dim):
    if dim == 2:
        return _tensor_evaluate_2d(prefix)
    return _tensor_evaluate_3d(prefix)


def _tensor_evaluate_2d(prefix):
    return [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int N_FIELDS>",
        "static SFEM_INLINE void %s_tensor_evaluate(" % prefix,
        "        const ptrdiff_t nelems, const scalar_t *const shape_1d, const scalar_t *const grad_1d,",
        "        const scalar_t *const streams[N_FIELDS * N_SHAPE], scalar_t *const value, scalar_t *const gradient) {",
        "    static constexpr int Q = %s_integer_root(N_QP, 2);" % prefix,
        "    static constexpr int S = %s_integer_root(N_SHAPE, 2);" % prefix,
        "    scalar_t vx[N_FIELDS * Q * S * VECTOR_SIZE];",
        "    scalar_t gx[N_FIELDS * Q * S * VECTOR_SIZE];",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t v = 0, g = 0;",
        "            for (int sx = 0; sx < S; ++sx) {",
        "                const int s = %s_tensor_index<S>(sx, sy);" % prefix,
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
        "            scalar_t v = 0, g0 = 0, g1 = 0;",
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


def _tensor_evaluate_3d(prefix):
    return [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int N_FIELDS>",
        "static SFEM_INLINE void %s_tensor_evaluate(" % prefix,
        "        const ptrdiff_t nelems, const scalar_t *const shape_1d, const scalar_t *const grad_1d,",
        "        const scalar_t *const streams[N_FIELDS * N_SHAPE], scalar_t *const value, scalar_t *const gradient) {",
        "    static constexpr int Q = %s_integer_root(N_QP, 3);" % prefix,
        "    static constexpr int S = %s_integer_root(N_SHAPE, 3);" % prefix,
        "    scalar_t vx[N_FIELDS * Q * S * S * VECTOR_SIZE], gx[N_FIELDS * Q * S * S * VECTOR_SIZE];",
        "    scalar_t vxy[N_FIELDS * Q * Q * S * VECTOR_SIZE], g0xy[N_FIELDS * Q * Q * S * VECTOR_SIZE], g1xy[N_FIELDS * Q * Q * S * VECTOR_SIZE];",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) for (int sz = 0; sz < S; ++sz) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t v = 0, g = 0;",
        "            for (int sx = 0; sx < S; ++sx) {",
        "                const int s = %s_tensor_index<S>(sx, sy, sz);" % prefix,
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
        "            scalar_t v = 0, g0 = 0, g1 = 0;",
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
        "            scalar_t v = 0, g0 = 0, g1 = 0, g2 = 0;",
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


def _tensor_integrate_helper(prefix, dim):
    if dim == 2:
        return _tensor_integrate_2d(prefix)
    return _tensor_integrate_3d(prefix)


def _tensor_integrate_2d(prefix):
    return [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int N_FIELDS>",
        "static SFEM_INLINE void %s_tensor_integrate(" % prefix,
        "        const ptrdiff_t nelems, const scalar_t *const shape_1d, const scalar_t *const grad_1d,",
        "        const scalar_t *const value_coeff, const scalar_t *const grad_coeff, scalar_t *const output[N_FIELDS * N_SHAPE]) {",
        "    static constexpr int Q = %s_integer_root(N_QP, 2), S = %s_integer_root(N_SHAPE, 2);"
        % (prefix, prefix),
        "    scalar_t sv[N_FIELDS * Q * S * VECTOR_SIZE], sg[N_FIELDS * Q * S * VECTOR_SIZE];",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
        "            scalar_t a = 0, b = 0;",
        "            for (int qy = 0; qy < Q; ++qy) { const int q = qx + Q * qy;",
        "                a += value_coeff[(f * N_QP + q) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy]",
        "                   + grad_coeff[((f * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane] * grad_1d[qy * S + sy];",
        "                b += grad_coeff[((f * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy]; }",
        "            const int i = ((f * Q + qx) * S + sy) * VECTOR_SIZE + lane; sv[i] = a; sg[i] = b;",
        "        }",
        "    }",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int sy = 0; sy < S; ++sy) for (int sx = 0; sx < S; ++sx) {",
        "        const int s = %s_tensor_index<S>(sx, sy);" % prefix,
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) { scalar_t v = 0;",
        "            for (int qx = 0; qx < Q; ++qx) { const int i = ((f * Q + qx) * S + sy) * VECTOR_SIZE + lane;",
        "                v += sv[i] * shape_1d[qx * S + sx] + sg[i] * grad_1d[qx * S + sx]; }",
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
        "        const scalar_t *const value_coeff, const scalar_t *const grad_coeff, scalar_t *const output[N_FIELDS * N_SHAPE]) {",
        "    static constexpr int Q = %s_integer_root(N_QP, 3), S = %s_integer_root(N_SHAPE, 3);"
        % (prefix, prefix),
        "    scalar_t z0[N_FIELDS * Q * Q * S * VECTOR_SIZE], z1[N_FIELDS * Q * Q * S * VECTOR_SIZE], z2[N_FIELDS * Q * Q * S * VECTOR_SIZE];",
        "    scalar_t yz0[N_FIELDS * Q * S * S * VECTOR_SIZE], yz1[N_FIELDS * Q * S * S * VECTOR_SIZE];",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int qy = 0; qy < Q; ++qy) for (int sz = 0; sz < S; ++sz) {",
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) { scalar_t a = 0, b = 0, c = 0;",
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
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) { scalar_t a = 0, b = 0;",
        "            for (int qy = 0; qy < Q; ++qy) { const int i = (((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane;",
        "                a += z0[i] * shape_1d[qy * S + sy] + z2[i] * grad_1d[qy * S + sy];",
        "                b += z1[i] * shape_1d[qy * S + sy]; }",
        "            const int j = (((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane; yz0[j] = a; yz1[j] = b;",
        "        }",
        "    }",
        "    for (int f = 0; f < N_FIELDS; ++f) for (int sz = 0; sz < S; ++sz) for (int sy = 0; sy < S; ++sy) for (int sx = 0; sx < S; ++sx) {",
        "        const int s = %s_tensor_index<S>(sx, sy, sz);" % prefix,
        "#pragma omp simd",
        "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) { scalar_t v = 0;",
        "            for (int qx = 0; qx < Q; ++qx) { const int j = (((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane;",
        "                v += yz0[j] * shape_1d[qx * S + sx] + yz1[j] * grad_1d[qx * S + sx]; }",
        "            output[s * N_FIELDS + f][lane] += v;",
        "        }",
        "    }",
        "}",
    ]
