from dataclasses import dataclass

import sympy as sp

from codegen.framework.symbolic.residual import CoupledResidualSystem
from codegen.framework.fem.tensor_product_geometry import (
    isoparametric_adjugate_call_lines,
    isoparametric_adjugate_stream_array_lines,
    sfem_geometry_kernels_header_source,
    streams_in_shape_order,
    tensor_product_cartesian_shape_order,
    tensor_product_gradient_isoparametric_geometry_lines,
)
from codegen.framework.fem.tensor_product_kernels import sfem_tensor_product_kernels_header_source
from codegen.framework.backends.targets import OpenMPTarget
from codegen.framework.fem.reference import (
    sfem_element_quadrature_rule,
    sfem_field_n_shape,
    sfem_is_tensor_product_hex_element,
    sfem_mesh_reference_data,
    sfem_reference_data,
    sfem_simplex_grad_ref_name,
    sfem_simplex_field_reference_data,
    sfem_soa_element_specialization,
    sfem_tensor_product_field_reference_data,
    sfem_tensor_product_hex_uses_cartesian_ordering,
    sfem_tensor_product_quad_uses_cartesian_ordering,
    SfemReferenceData,
)
from codegen.framework.emitters.quadrature_codegen import (
    quadrature_reference_accessor,
    quadrature_reference_struct_lines,
)
from codegen.framework.plans.reference_data import validate_reference_data_plan
from codegen.framework.plans.diagnostics import validate_diagnostics_plan_names
from codegen.framework.plans.form_transformations import (
    constant_p1_simplex_reference_gradients,
    simplex_gradient_metric_transformation,
    symmetric_metric_component_count,
    symmetric_metric_component_index,
    symmetric_metric_storage_component_index,
)
from codegen.framework.symbolic.core import (
    GeneratedKernelFile,
    KernelExpressions,
    _prune_dead_cse_intermediates,
    _sfem_ccode,
    _sfem_math_header_source,
)
from codegen.framework.emitters.energy_codegen import (
    _sfem_soa_diagnostic_print_wrapper_lines,
    _sfem_soa_diagnostics_header,
    _sfem_packed_thread_scratch_header_source,
)


def _target():
    return OpenMPTarget()


def _tensor_product_coordinate_shape_order(dim, n_shape, element_type):
    if sfem_tensor_product_hex_uses_cartesian_ordering(element_type) or sfem_tensor_product_quad_uses_cartesian_ordering(element_type):
        return tuple(range(n_shape))
    return tensor_product_cartesian_shape_order(dim, n_shape)


def _identity_order(order):
    return tuple(order) == tuple(range(len(order)))


def _single_field_shape_order(n_shape, n_fields, field_stream_order):
    return tuple(field_stream_order[shape * n_fields] // n_fields for shape in range(n_shape))


def _stream_to_tensor_order(field_stream_order):
    ordered = [0] * len(field_stream_order)
    for tensor_stream, mesh_stream in enumerate(field_stream_order):
        ordered[mesh_stream] = tensor_stream
    return tuple(ordered)


def _linear_index_offset(values):
    values = tuple(values)
    if not values:
        return 0
    offset = values[0]
    if values == tuple(offset + i for i in range(len(values))):
        return offset
    return None


def _local_index_mapping_expr(name, values, index_expr):
    offset = _linear_index_offset(values)
    if offset is None:
        return "%s(%s)" % (name, index_expr)
    if offset == 0:
        return index_expr
    return "(%s + %d)" % (index_expr, offset)


def _local_index_mapping_lambda_lines(name, values, indent):
    if _linear_index_offset(values) is not None:
        return []
    lines = [
        "%sconst auto %s = [](const int local) -> int {" % (indent, name),
        "%s    switch (local) {" % indent,
    ]
    lines.extend(
        "%s        case %d: return %d;" % (indent, local, value)
        for local, value in enumerate(values)
    )
    lines.extend(
        [
            "%s        default: return 0;" % indent,
            "%s    }" % indent,
            "%s};" % indent,
        ]
    )
    return lines


def _single_field_element_alias_lines(n_shape, shape_order, indent, array_name="field_elements", pointer_type="idx_t"):
    if _identity_order(shape_order):
        return [], "elements"
    return [
        "%sconst %s *const SFEM_RESTRICT %s[%d] = {%s};"
        % (
            indent,
            pointer_type,
            array_name,
            n_shape,
            ", ".join("elements[%d]" % shape for shape in shape_order),
        )
    ], array_name


def _coordinate_element_alias_lines(dim, n_shape, element_type, indent, pointer_type="idx_t"):
    return _single_field_element_alias_lines(
        n_shape,
        _tensor_product_coordinate_shape_order(dim, n_shape, element_type),
        indent,
        array_name="coordinate_elements",
        pointer_type=pointer_type,
    )


def _inline_qualifier():
    return _target().inline_qualifier()


def _function_qualifier():
    return _target().function_qualifier()


def _inline_definition_lines():
    return _target().inline_definition_lines()


def _vectorize_pragma():
    return _target().vectorize_pragma()


def _parallel_for_pragma(schedule=None):
    return _target().parallel_for_pragma(schedule)


def _atomic_update_pragma():
    return _target().atomic_update_pragma()


def _work_item_index():
    return _target().work_item_index()


def _work_item_loop_lines(indent):
    return _target().work_item_loop_lines(indent)


def _affine_geometry_stream_conversion_lines(streams, indent):
    streams = tuple(streams)
    n_streams = len(streams)
    return [
        "%sconst jacobian_t *const affine_geometry_sources[%d] = {%s};"
        % (
            indent,
            n_streams,
            ", ".join("g_%s + evbegin" % stream for stream in streams),
        ),
        "%sscalar_t block_affine_geometry_data[%d][VECTOR_SIZE];"
        % (indent, n_streams),
        "%sconst scalar_t *block_affine_geometry_streams[%d];"
        % (indent, n_streams),
        "%sfor (int geometry_stream = 0; geometry_stream < %d; ++geometry_stream) {"
        % (indent, n_streams),
        "%s    block_affine_geometry_streams[geometry_stream] = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>("
        % indent,
        "%s            nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<jacobian_t, scalar_t>());"
        % indent,
        "%s}" % indent,
    ]


def _affine_geometry_stream_helper_lines():
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename scalar_t, typename jacobian_t, int VECTOR_SIZE>",
        "%s const scalar_t *affine_geometry_stream(" % _inline_qualifier(),
        "        const int,",
        "        const jacobian_t *const SFEM_RESTRICT source,",
        "        scalar_t *const SFEM_RESTRICT,",
        "        std::true_type) {",
        "    return source;",
        "}",
        "",
        "template <typename scalar_t, typename jacobian_t, int VECTOR_SIZE>",
        "%s const scalar_t *affine_geometry_stream(" % _inline_qualifier(),
        "        const int nelems,",
        "        const jacobian_t *const SFEM_RESTRICT source,",
        "        scalar_t *const SFEM_RESTRICT converted,",
        "        std::false_type) {",
    ]
    pragma = _vectorize_pragma()
    if pragma:
        lines.append("    %s" % pragma)
    lines.extend(
        [
            "    for (int lane = 0; lane < nelems; ++lane) {",
            "        converted[lane] = scalar_t(source[lane]);",
            "    }",
            "    return converted;",
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
        ]
    )
    return lines


def _uses_cached_affine_metric(gradient_metric):
    return (
        gradient_metric is not None
        and getattr(gradient_metric, "affine_geometry_storage", "")
        == "symmetric_metric_soa"
    )


def _direct_atomic_scatter_lines(pointer, node_expr, value_expr, indent):
    lines = [
        "%s{" % indent,
        "%s    for (int scatter = 0; scatter < nelems; ++scatter) {" % indent,
        "%s        %s" % (indent, _atomic_update_pragma()),
        "%s        %s[%s] += %s;"
        % (indent, pointer, node_expr % "scatter", value_expr % "scatter"),
        "%s    }" % indent,
        "%s}" % indent,
    ]
    return lines


def _metric_component_load(component, scale, source=None):
    if source is None:
        source = "scalar_t(g_geom_metric%d[i])" % component
    if scale == "1":
        return source
    return "metric_factor * %s" % source


def _simplex_metric_scalar_affine_loop_lines(
    system,
    rule,
    input_name,
    input_stride,
    output_name,
    scale,
    indent,
    unit_stride,
    unit_scale,
    metric_layout="soa",
):
    if (
        system.dim == 3
        and rule.n_shape == 4
        and metric_layout == "aos"
        and unit_stride
        and unit_scale
    ):
        lines = [
            "",
            "%s%s" % (indent, _parallel_for_pragma("static").strip()),
            "%sfor (ptrdiff_t i = 0; i < nelements; ++i) {" % indent,
            "%s    scalar_t element_vector[4];" % indent,
            "%s    scalar_t fff[6];" % indent,
            "%s    for (int k = 0; k < 6; ++k) {" % indent,
            "%s        fff[k] = scalar_t(g_geom_metric[i * 6 + k]);" % indent,
            "%s    }" % indent,
            "%s    const idx_t ev0 = elements[0][i];" % indent,
            "%s    const idx_t ev1 = elements[1][i];" % indent,
            "%s    const idx_t ev2 = elements[2][i];" % indent,
            "%s    const idx_t ev3 = elements[3][i];" % indent,
            "%s    const scalar_t u0 = %s[ev0];" % (indent, input_name),
            "%s    const scalar_t u1 = %s[ev1];" % (indent, input_name),
            "%s    const scalar_t u2 = %s[ev2];" % (indent, input_name),
            "%s    const scalar_t u3 = %s[ev3];" % (indent, input_name),
            "%s    const scalar_t x0 = fff[0] + fff[1] + fff[2];" % indent,
            "%s    const scalar_t x1 = fff[1] + fff[3] + fff[4];" % indent,
            "%s    const scalar_t x2 = fff[2] + fff[4] + fff[5];" % indent,
            "%s    const scalar_t x3 = fff[1] * u0;" % indent,
            "%s    const scalar_t x4 = fff[2] * u0;" % indent,
            "%s    const scalar_t x5 = fff[4] * u0;" % indent,
            "%s    element_vector[0] = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;" % indent,
            "%s    element_vector[1] = -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;" % indent,
            "%s    element_vector[2] = fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;" % indent,
            "%s    element_vector[3] = fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;" % indent,
            "%s        %s" % (indent, _atomic_update_pragma()),
            "%s        %s[ev0] += element_vector[0];" % (indent, output_name),
            "%s        %s" % (indent, _atomic_update_pragma()),
            "%s        %s[ev1] += element_vector[1];" % (indent, output_name),
            "%s        %s" % (indent, _atomic_update_pragma()),
            "%s        %s[ev2] += element_vector[2];" % (indent, output_name),
            "%s        %s" % (indent, _atomic_update_pragma()),
            "%s        %s[ev3] += element_vector[3];" % (indent, output_name),
            "%s}" % indent,
        ]
        return lines

    lines = [
        "",
        "%s%s" % (indent, _parallel_for_pragma("static").strip()),
        "%sfor (ptrdiff_t i = 0; i < nelements; ++i) {" % indent,
    ]
    for shape in range(rule.n_shape):
        lines.append(
            "%s    const idx_t ev%d = elements[%d][i];" % (indent, shape, shape)
        )
    for shape in range(rule.n_shape):
        index = "ev%d" % shape if unit_stride else "ev%d * %s" % (shape, input_stride)
        lines.append(
            "%s    const scalar_t u%d = %s[%s];"
            % (indent, shape, input_name, index)
        )
    if not unit_scale and scale != "1":
        lines.append("%s    const scalar_t metric_factor = %s;" % (indent, scale))
    component_scale = "1" if unit_scale else scale
    metric_components = system.dim * (system.dim + 1) // 2
    for component in range(metric_components):
        if metric_layout == "aos":
            source = "scalar_t(g_geom_metric[i * %d + %d])" % (
                metric_components,
                component,
            )
        else:
            source = "scalar_t(g_geom_metric%d[i])" % component
        lines.append(
            "%s    const scalar_t fff%d = %s;"
            % (
                indent,
                component,
                _metric_component_load(component, component_scale, source),
            )
        )
    for d in range(system.dim):
        lines.append("%s    const scalar_t grad%d = u%d - u0;" % (indent, d, d + 1))

    if system.dim == 2:
        lines.extend(
            [
                "%s    const scalar_t e1 = fff0 * grad0 + fff1 * grad1;" % indent,
                "%s    const scalar_t e2 = fff1 * grad0 + fff2 * grad1;" % indent,
                "%s    const scalar_t e0 = -(e1) - e2;" % indent,
            ]
        )
    else:
        lines.extend(
            [
                "%s    const scalar_t x0 = fff0 + fff1 + fff2;" % indent,
                "%s    const scalar_t x1 = fff1 + fff3 + fff4;" % indent,
                "%s    const scalar_t x2 = fff2 + fff4 + fff5;" % indent,
                "%s    const scalar_t x3 = fff1 * u0;" % indent,
                "%s    const scalar_t x4 = fff2 * u0;" % indent,
                "%s    const scalar_t x5 = fff4 * u0;" % indent,
                "%s    const scalar_t e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;" % indent,
                "%s    const scalar_t e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;" % indent,
                "%s    const scalar_t e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;" % indent,
                "%s    const scalar_t e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;" % indent,
            ]
        )
    for shape in range(rule.n_shape):
        index = "ev%d" % shape if unit_stride else "ev%d * out_stride" % shape
        lines.extend(
            [
                "%s    %s" % (indent, _atomic_update_pragma()),
                "%s    %s[%s] += e%d;" % (indent, output_name, index, shape),
            ]
        )
    lines.append("%s}" % indent)
    return lines


def _simplex_metric_scalar_affine_fast_path_body(
    system,
    rule,
    dependencies,
    gradient_metric,
):
    if not _uses_cached_affine_metric(gradient_metric):
        return None
    if len(system.fields) != 1 or rule.n_qp != 1:
        return None
    if system.dim not in (2, 3) or rule.n_shape != system.dim + 1:
        return None
    if dependencies.previous or dependencies.current == dependencies.direction:
        return None

    stream_group_name = "current" if dependencies.current else "direction"
    if gradient_metric.stream_group_name != stream_group_name:
        return None

    field = system.fields[0]
    input_name = field.name if dependencies.current else "%s_direction" % field.name
    input_stride = "current_stride" if dependencies.current else "direction_stride"
    output_name = "%s_out" % field.name
    scale = _sfem_ccode(gradient_metric.scale)
    lines = []
    if scale == "1":
        lines.extend(
            [
                "    if (%s == 1 && out_stride == 1) {" % input_stride,
                *_simplex_metric_scalar_affine_loop_lines(
                    system,
                    rule,
                    input_name,
                    input_stride,
                    output_name,
                    scale,
                    "        ",
                    unit_stride=True,
                    unit_scale=True,
                    metric_layout="soa",
                ),
                "    } else {",
                *_simplex_metric_scalar_affine_loop_lines(
                    system,
                    rule,
                    input_name,
                    input_stride,
                    output_name,
                    scale,
                    "        ",
                    unit_stride=False,
                    unit_scale=True,
                    metric_layout="soa",
                ),
                "    }",
            ]
        )
    else:
        lines.extend(
            [
                "    if (%s == 1 && out_stride == 1) {" % input_stride,
                "        if ((%s) == scalar_t(1)) {" % scale,
                *_simplex_metric_scalar_affine_loop_lines(
                    system,
                    rule,
                    input_name,
                    input_stride,
                    output_name,
                    scale,
                    "            ",
                    unit_stride=True,
                    unit_scale=True,
                    metric_layout="soa",
                ),
                "        } else {",
                *_simplex_metric_scalar_affine_loop_lines(
                    system,
                    rule,
                    input_name,
                    input_stride,
                    output_name,
                    scale,
                    "            ",
                    unit_stride=True,
                    unit_scale=False,
                    metric_layout="soa",
                ),
                "        }",
                "    } else {",
                "        if ((%s) == scalar_t(1)) {" % scale,
                *_simplex_metric_scalar_affine_loop_lines(
                    system,
                    rule,
                    input_name,
                    input_stride,
                    output_name,
                    scale,
                    "            ",
                    unit_stride=False,
                    unit_scale=True,
                    metric_layout="soa",
                ),
                "        } else {",
                *_simplex_metric_scalar_affine_loop_lines(
                    system,
                    rule,
                    input_name,
                    input_stride,
                    output_name,
                    scale,
                    "            ",
                    unit_stride=False,
                    unit_scale=False,
                    metric_layout="soa",
                ),
                "        }",
                "    }",
            ]
        )
    return lines


def _simplex_metric_scalar_affine_aos_wrapper_lines(
    function,
    system,
    rule,
    dependencies,
    gradient_metric,
):
    field = system.fields[0]
    input_kind = "current" if dependencies.current else "direction"
    input_name = field.name if dependencies.current else "%s_direction" % field.name
    input_stride = "%s_stride" % input_kind
    output_name = "%s_out" % field.name
    scale = _sfem_ccode(gradient_metric.scale)
    lines = []
    for scalar_type, suffix in (("double", ""), ("float", "_float")):
        params = [
            "const ptrdiff_t nelements",
            "const ptrdiff_t nnodes",
            "idx_t **const SFEM_RESTRICT elements",
            "const geom_t *const SFEM_RESTRICT g_geom_metric",
        ]
        params.extend(
            "const %s %s" % (scalar_type, parameter)
            for parameter in dependencies.parameters
        )
        if dependencies.current:
            params.append("const ptrdiff_t current_stride")
            params.extend(
                "const %s *const SFEM_RESTRICT %s" % (scalar_type, field.name)
                for field in system.fields
            )
        if dependencies.direction:
            params.append("const ptrdiff_t direction_stride")
            params.extend(
                "const %s *const SFEM_RESTRICT %s_direction"
                % (scalar_type, field.name)
                for field in system.fields
            )
        params.append("const ptrdiff_t out_stride")
        params.extend(
            "%s *const SFEM_RESTRICT %s_out" % (scalar_type, field.name)
            for field in system.fields
        )
        lines.append('extern "C" int %s_aos%s(' % (function, suffix))
        for index, param in enumerate(params):
            lines.append(
                "        %s%s"
                % (param, "," if index + 1 < len(params) else "")
            )
        lines.extend(
            [
                ") {",
                "    using scalar_t = %s;" % scalar_type,
                "    static constexpr int DIM = %d;" % system.dim,
                "    static constexpr int N_QP = %d;" % rule.n_qp,
                "    static constexpr int N_SHAPE = %d;" % rule.n_shape,
                "    (void)DIM;",
                "    (void)N_QP;",
                "    (void)N_SHAPE;",
                "    (void)nnodes;",
                "    if (%s == 1 && out_stride == 1) {" % input_stride,
                "        if ((%s) == scalar_t(1)) {" % scale,
                *_simplex_metric_scalar_affine_loop_lines(
                    system,
                    rule,
                    input_name,
                    input_stride,
                    output_name,
                    scale,
                    "            ",
                    unit_stride=True,
                    unit_scale=True,
                    metric_layout="aos",
                ),
                "        } else {",
                *_simplex_metric_scalar_affine_loop_lines(
                    system,
                    rule,
                    input_name,
                    input_stride,
                    output_name,
                    scale,
                    "            ",
                    unit_stride=True,
                    unit_scale=False,
                    metric_layout="aos",
                ),
                "        }",
                "    } else {",
                "        if ((%s) == scalar_t(1)) {" % scale,
                *_simplex_metric_scalar_affine_loop_lines(
                    system,
                    rule,
                    input_name,
                    input_stride,
                    output_name,
                    scale,
                    "            ",
                    unit_stride=False,
                    unit_scale=True,
                    metric_layout="aos",
                ),
                "        } else {",
                *_simplex_metric_scalar_affine_loop_lines(
                    system,
                    rule,
                    input_name,
                    input_stride,
                    output_name,
                    scale,
                    "            ",
                    unit_stride=False,
                    unit_scale=False,
                    metric_layout="aos",
                ),
                "        }",
                "    }",
                "    return SFEM_SUCCESS;",
                "}",
                "",
            ]
        )
        if len(dependencies.parameters) == 1 and scale == str(dependencies.parameters[0]):
            unit_params = [
                "const ptrdiff_t nelements",
                "const ptrdiff_t nnodes",
                "idx_t **const SFEM_RESTRICT elements",
                "const geom_t *const SFEM_RESTRICT g_geom_metric",
            ]
            if dependencies.current:
                unit_params.extend(
                    "const %s *const SFEM_RESTRICT %s" % (scalar_type, field.name)
                    for field in system.fields
                )
            if dependencies.direction:
                unit_params.extend(
                    "const %s *const SFEM_RESTRICT %s_direction"
                    % (scalar_type, field.name)
                    for field in system.fields
                )
            unit_params.extend(
                "%s *const SFEM_RESTRICT %s_out" % (scalar_type, field.name)
                for field in system.fields
            )
            lines.append('extern "C" int %s_aos_unit%s(' % (function, suffix))
            for index, param in enumerate(unit_params):
                lines.append(
                    "        %s%s"
                    % (param, "," if index + 1 < len(unit_params) else "")
                )
            lines.extend(
                [
                    ") {",
                    "    using scalar_t = %s;" % scalar_type,
                    "    (void)nnodes;",
                    *_simplex_metric_scalar_affine_loop_lines(
                        system,
                        rule,
                        input_name,
                        input_stride,
                        output_name,
                        scale,
                        "    ",
                        unit_stride=True,
                        unit_scale=True,
                        metric_layout="aos",
                    ),
                    "    return SFEM_SUCCESS;",
                    "}",
                    "",
                ]
            )
    return lines


def _affine_mesh_public_wrapper_lines(
    function,
    impl,
    params,
    uses_cached_affine_metric,
    gradient_metric,
    dependencies,
    system,
    dim,
):
    lines = []
    for scalar_type, suffix in (("double", ""), ("float", "_float")):
        typed_params = [
            param.replace("jacobian_t", "geom_t").replace("scalar_t", scalar_type)
            for param in params
        ]
        lines.append('extern "C" int %s%s(' % (function, suffix))
        for index, param in enumerate(typed_params):
            lines.append(
                "        %s%s"
                % (param, "," if index + 1 < len(typed_params) else "")
            )
        call_args = ["nelements", "nnodes", "elements"]
        if uses_cached_affine_metric:
            call_args.extend(
                "g_geom_metric%d" % i
                for i in range(gradient_metric.metric_components)
            )
        elif dependencies.uses_adjugate:
            call_args.extend(
                "g_jacobian_adjugate%d" % i for i in range(dim * dim)
            )
        if not uses_cached_affine_metric:
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
                "    return sfem::codegen::%s<%s, geom_t>(%s);"
                % (impl, scalar_type, ", ".join(call_args)),
                "}",
                "",
            ]
        )
    return lines


def _zero_block_output_lines(name, n_streams, indent):
    return [
        "%sfor (int stream = 0; stream < %d; ++stream) {" % (indent, n_streams),
        *_work_item_loop_lines("%s    " % indent),
        "%s        %s[stream][lane] = scalar_t(0);" % (indent, name),
        "%s    }" % indent,
        "%s}" % indent,
    ]


def _stream_pointer_array_lines(pointer_type, array_name, storage_name, n_streams, stream_order, indent):
    if tuple(stream_order) == tuple(range(n_streams)):
        return []

    return [
        "%s%sconst %s[%d] = {%s};"
        % (
            indent,
            pointer_type,
            array_name,
            n_streams,
            ", ".join("%s[%d]" % (storage_name, i) for i in stream_order),
        )
    ]


def _block_stream_argument(
    pointer_type,
    array_name,
    storage_name,
    n_streams,
    stream_order,
    indent,
    mutable,
):
    lines = _stream_pointer_array_lines(
        pointer_type,
        array_name,
        storage_name,
        n_streams,
        stream_order,
        indent,
    )
    if lines:
        return lines, array_name
    return [], storage_name


@dataclass(frozen=True)
class MeshBlockGatherLoop:
    shape_var: str
    shape_count: str
    lane_indent: str
    setup_lines: tuple = ()
    close_lines: tuple = ()
    element_index: str = None
    element_array: str = "elements"
    element_pointer_type: str = "idx_t"


@dataclass(frozen=True)
class MeshBlockScatterLoop:
    shape_var: str
    shape_count: str
    scatter_indent: str
    setup_lines: tuple = ()
    close_lines: tuple = ()
    element_index: str = None
    element_array: str = "elements"
    element_pointer_type: str = "idx_t"


def _mesh_block_gather_loop_lines(indent, loop, assignment_lines):
    element_index = loop.shape_var if loop.element_index is None else loop.element_index
    lines = [
        "%sfor (int %s = 0; %s < %s; ++%s) {"
        % (indent, loop.shape_var, loop.shape_var, loop.shape_count, loop.shape_var),
        "%s    const %s *const SFEM_RESTRICT element_shape = %s[%s];"
        % (indent, loop.element_pointer_type, loop.element_array, element_index),
    ]
    lines.extend(loop.setup_lines)
    lines.extend(_work_item_loop_lines(loop.lane_indent))
    lines.append(
        "%s    const idx_t node = element_shape[evbegin + lane];"
        % loop.lane_indent
    )
    lines.extend(assignment_lines)
    lines.append("%s}" % loop.lane_indent)
    lines.extend(loop.close_lines)
    lines.append("%s}" % indent)
    return lines


def _mesh_block_scatter_loop_lines(indent, loop, accumulation_line):
    element_index = loop.shape_var if loop.element_index is None else loop.element_index
    lines = [
        "%sfor (int %s = 0; %s < %s; ++%s) {"
        % (indent, loop.shape_var, loop.shape_var, loop.shape_count, loop.shape_var),
        "%s    const %s *const SFEM_RESTRICT element_shape = %s[%s];"
        % (indent, loop.element_pointer_type, loop.element_array, element_index),
    ]
    lines.extend(loop.setup_lines)
    lines.extend(
        [
            "%sfor (int scatter = 0; scatter < nelems; ++scatter) {"
            % loop.scatter_indent,
            "%s    %s" % (loop.scatter_indent, _atomic_update_pragma()),
            accumulation_line,
            "%s}" % loop.scatter_indent,
        ]
    )
    lines.extend(loop.close_lines)
    lines.append("%s}" % indent)
    return lines


def _field_gather_lines(system, dependencies, indent, element_array="elements"):
    lines = []
    if dependencies.current:
        lines.append(
            "%sconst scalar_t *const current_components[N_FIELDS] = {%s};"
            % (indent, ", ".join(field.name for field in system.fields))
        )
    if dependencies.previous:
        lines.append(
            "%sconst scalar_t *const previous_components[N_FIELDS] = {%s};"
            % (indent, ", ".join("%s_old" % field.name for field in system.fields))
        )
    if dependencies.direction:
        lines.append(
            "%sconst scalar_t *const direction_components[N_FIELDS] = {%s};"
            % (indent, ", ".join("%s_direction" % field.name for field in system.fields))
        )

    if not lines:
        return lines

    assignment_lines = []
    if dependencies.current:
        assignment_lines.append(
            "%s            block_current[stream][lane] = current_components[field][node * current_stride];"
            % indent
        )
    if dependencies.previous:
        assignment_lines.append(
            "%s            block_previous[stream][lane] = previous_components[field][node * previous_stride];"
            % indent
        )
    if dependencies.direction:
        assignment_lines.append(
            "%s            block_direction[stream][lane] = direction_components[field][node * direction_stride];"
            % indent
        )
    lines.extend(
        [
            "",
            *_mesh_block_gather_loop_lines(
                indent,
                MeshBlockGatherLoop(
                    shape_var="shape",
                    shape_count="N_SHAPE",
                    element_array=element_array,
                    setup_lines=(
                        "%s    for (int field = 0; field < N_FIELDS; ++field)" % indent
                        + " {",
                        "%s        const int stream = shape * N_FIELDS + field;" % indent,
                    ),
                    close_lines=("%s    }" % indent,),
                    lane_indent="%s        " % indent,
                ),
                assignment_lines,
            ),
        ]
    )
    return lines


def _coordinate_gather_lines(dim, indent, element_array="elements", pointer_type="idx_t"):
    return [
        "%sconst geom_t *const coordinate_components[DIM] = {%s};"
        % (indent, ", ".join("points[%d]" % d for d in range(dim))),
        *_mesh_block_gather_loop_lines(
            indent,
            MeshBlockGatherLoop(
                shape_var="shape",
                shape_count="N_SHAPE",
                element_array=element_array,
                element_pointer_type=pointer_type,
                setup_lines=("%s    for (int d = 0; d < DIM; ++d) {" % indent,),
                close_lines=("%s    }" % indent,),
                lane_indent="%s        " % indent,
            ),
            [
                "%s            block_coordinates[shape * DIM + d][lane] = coordinate_components[d][node];"
                % indent
            ],
        ),
    ]


def _field_atomic_scatter_lines(system, indent, element_array="elements"):
    return [
        "%sscalar_t *const output_components[N_FIELDS] = {%s};"
        % (indent, ", ".join("%s_out" % field.name for field in system.fields)),
        *_mesh_block_scatter_loop_lines(
            indent,
                MeshBlockScatterLoop(
                    shape_var="shape",
                    shape_count="N_SHAPE",
                    element_array=element_array,
                    setup_lines=(
                        "%s    for (int field = 0; field < N_FIELDS; ++field) {" % indent,
                    "%s        const int stream = shape * N_FIELDS + field;" % indent,
                    "%s        scalar_t *const SFEM_RESTRICT out = output_components[field];" % indent,
                ),
                close_lines=("%s    }" % indent,),
                scatter_indent="%s        " % indent,
            ),
            "%s            out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];"
            % indent,
        ),
    ]


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
    def uses_test_coefficients(self):
        return any(self.value_coefficients) or self.uses_test_gradients

    @property
    def uses_reference_gradients(self):
        return self.uses_trial_gradients or self.uses_test_gradients

    @property
    def uses_adjugate(self):
        return self.uses_reference_gradients


@dataclass(frozen=True)
class MixedFieldGroup:
    name: str
    components: int
    field_indices: tuple
    shape_count: int
    offset: int


@dataclass(frozen=True)
class MixedFieldLayout:
    fields: tuple
    groups: tuple
    field_group_indices: tuple
    shape_counts: tuple
    offsets: tuple
    total_streams: int

    @classmethod
    def create(cls, system, cell_rule, field_element_types):
        fields = tuple(system.fields)
        group_entries = []
        group_index_by_name = {}
        field_group_indices = []
        for field_index, field in enumerate(fields):
            group_name = _residual_parent_field_name(field)
            try:
                group_index = group_index_by_name[group_name]
            except KeyError:
                group_index = len(group_entries)
                group_index_by_name[group_name] = group_index
                group_entries.append(
                    {
                        "name": group_name,
                        "components": int(getattr(field, "components", 1)),
                        "field_indices": [],
                    }
                )
            group_entries[group_index]["field_indices"].append(field_index)
            field_group_indices.append(group_index)

        group_shape_counts = tuple(
            _field_n_shape_by_name(entry["name"], cell_rule, field_element_types)
            for entry in group_entries
        )
        shape_counts = tuple(
            group_shape_counts[group_index]
            for group_index in field_group_indices
        )
        offsets = []
        offset = 0
        for n_shape in shape_counts:
            offsets.append(offset)
            offset += n_shape
        groups = []
        group_offsets = [None] * len(group_entries)
        for field_index, group_index in enumerate(field_group_indices):
            if group_offsets[group_index] is None:
                group_offsets[group_index] = offsets[field_index]
        for entry, n_shape, group_offset in zip(group_entries, group_shape_counts, group_offsets):
            groups.append(
                MixedFieldGroup(
                    entry["name"],
                    entry["components"],
                    tuple(entry["field_indices"]),
                    n_shape,
                    group_offset,
                )
            )
        return cls(
            fields,
            tuple(groups),
            tuple(field_group_indices),
            shape_counts,
            tuple(offsets),
            offset,
        )

    @property
    def n_reference_fields(self):
        return len(self.groups)

    def n_shape(self, field_index):
        return self.shape_counts[field_index]

    def offset(self, field_index):
        return self.offsets[field_index]

    def reference_index(self, field_index):
        return self.field_group_indices[field_index]

    def group(self, group_index):
        return self.groups[group_index]

    def group_for_field(self, field_index):
        return self.groups[self.reference_index(field_index)]

    def stream_index(self, field_index, local_shape):
        return self.offsets[field_index] + local_shape

    def n_shape_constant(self, field):
        return "%s_N_SHAPE" % _residual_parent_field_name(field).upper()

    def n_shape_1d_constant(self, field):
        return "%s_N_SHAPE_1D" % _residual_parent_field_name(field).upper()


@dataclass(frozen=True)
class DependencyStreamGroup:
    name: str
    symbol_suffix: str
    pointer_suffix: str
    stride: str
    uses_value: bool
    uses_gradient: bool


def _dependency_stream_groups(dependencies, *, mesh=False):
    groups = []
    if dependencies.current:
        groups.append(
            DependencyStreamGroup(
                "current",
                "",
                "_data" if mesh else "",
                "current_stride",
                dependencies.current_value,
                dependencies.current_gradient,
            )
        )
    if dependencies.previous:
        groups.append(
            DependencyStreamGroup(
                "previous",
                "_old",
                "_old_data" if mesh else "",
                "previous_stride",
                dependencies.previous_value,
                dependencies.previous_gradient,
            )
        )
    if dependencies.direction:
        groups.append(
            DependencyStreamGroup(
                "direction",
                "_direction",
                "_direction_data" if mesh else "",
                "direction_stride",
                dependencies.direction_value,
                dependencies.direction_gradient,
            )
        )
    return tuple(groups)


def _dependency_stream_group_by_name(dependencies, name):
    name = str(name)
    for group in _dependency_stream_groups(dependencies):
        if group.name == name:
            return group
    raise ValueError("dependency stream group '%s' is not active" % name)


def _indexed_stream_initializer(name, count):
    return ", ".join("%s[%d]" % (name, i) for i in range(count))


def _indexed_stream_range_initializer(name, begin, count):
    return ", ".join("%s[%d]" % (name, begin + i) for i in range(count))


def _field_stream_initializer(layout, field_index, array_name):
    return ", ".join(
        "%s[%d]" % (array_name, layout.stream_index(field_index, s))
        for s in range(layout.n_shape(field_index))
    )


def _mixed_local_reference_params(cell_rule, n_fields, dim, dependencies, basis_family=None):
    if _is_tensor_product_family(cell_rule, basis_family):
        params = [
            "const scalar_t *const SFEM_RESTRICT field_shape_1d[%d]" % n_fields
        ]
        if dependencies.uses_reference_gradients:
            params.append(
                "const scalar_t *const SFEM_RESTRICT field_grad_1d[%d]" % n_fields
            )
        params.append("const scalar_t *const SFEM_RESTRICT q_weight_1d")
        return params

    params = ["const scalar_t *const SFEM_RESTRICT field_shape[%d]" % n_fields]
    if dependencies.uses_reference_gradients:
        params.append(
            "const scalar_t *const SFEM_RESTRICT field_grad_ref[%d]"
            % (n_fields * dim)
        )
    params.append("const scalar_t *const SFEM_RESTRICT q_weight")
    return params


def _mixed_reference_pointer_lines(
    reference_data,
    system,
    cell_rule,
    dependencies,
    indent,
    field_element_types=None,
    basis_family=None,
):
    dim = system.dim
    field_element_types = {} if field_element_types is None else field_element_types
    layout = MixedFieldLayout.create(system, cell_rule, field_element_types)
    lines = []
    if _is_tensor_product_family(cell_rule, basis_family):
        lines.append(
            "%sconst scalar_t *const field_shape_1d[N_FIELDS] = {%s};"
            % (
                indent,
                ", ".join(
                    "sfem::codegen::%s::%s_shape_1d()"
                    % (
                        reference_data,
                        _tensor_reference_prefix(
                            _field_element_type(group.name, cell_rule, field_element_types)
                        ),
                    )
                    for group in layout.groups
                ),
            )
        )
        if dependencies.uses_reference_gradients:
            lines.append(
                "%sconst scalar_t *const field_grad_1d[N_FIELDS] = {%s};"
                % (
                    indent,
                    ", ".join(
                        "sfem::codegen::%s::%s_grad_1d()"
                        % (
                            reference_data,
                            _tensor_reference_prefix(
                                _field_element_type(group.name, cell_rule, field_element_types)
                            ),
                        )
                        for group in layout.groups
                    ),
                )
            )
        return lines

    lines.append(
        "%sconst scalar_t *const field_shape[N_FIELDS] = {%s};"
        % (
            indent,
            ", ".join(
                "sfem::codegen::%s::%s_shape()"
                % (
                    reference_data,
                    _simplex_reference_prefix(
                        _field_element_type(group.name, cell_rule, field_element_types)
                    ),
                )
                for group in layout.groups
            ),
        )
    )
    if dependencies.uses_reference_gradients:
        grad_refs = []
        for group in layout.groups:
            for d in range(dim):
                reference_prefix = _simplex_reference_prefix(
                    _field_element_type(group.name, cell_rule, field_element_types)
                )
                grad_refs.append(
                    "sfem::codegen::%s::%s()"
                    % (
                        reference_data,
                        sfem_simplex_grad_ref_name(
                            "%s_grad_ref" % reference_prefix,
                            d,
                        ),
                    )
                )
        lines.append(
            "%sconst scalar_t *const field_grad_ref[N_FIELDS * DIM] = {%s};"
            % (indent, ", ".join(grad_refs))
        )
    return lines


def _mixed_reference_call_args(cell_rule, dependencies, reference_data, basis_family=None):
    if _is_tensor_product_family(cell_rule, basis_family):
        args = ["field_shape_1d"]
        if dependencies.uses_reference_gradients:
            args.append("field_grad_1d")
        args.append("sfem::codegen::%s::q_weight_1d()" % reference_data)
        return args

    args = ["field_shape"]
    if dependencies.uses_reference_gradients:
        args.append("field_grad_ref")
    args.append("sfem::codegen::%s::q_weight()" % reference_data)
    return args


def _mesh_reference_name(geometry_mode, name):
    return "%s_%s" % (geometry_mode, name)


def _mixed_mesh_dependency_params(layout, dependencies):
    params = []
    for group in _dependency_stream_groups(dependencies, mesh=True):
        params.append("const ptrdiff_t %s" % group.stride)
        for field_group in layout.groups:
            if field_group.components == 1:
                params.append(
                    "const scalar_t *const SFEM_RESTRICT %s%s"
                    % (field_group.name, group.pointer_suffix)
                )
            else:
                params.append(
                    "const scalar_t *const SFEM_RESTRICT %s%s[%d]"
                    % (field_group.name, group.pointer_suffix, field_group.components)
                )
    return params


def _mixed_mesh_dependency_call_args(layout, dependencies):
    args = []
    for group in _dependency_stream_groups(dependencies, mesh=True):
        args.append(group.stride)
        args.extend(
            "%s%s" % (field_group.name, group.pointer_suffix)
            for field_group in layout.groups
        )
    return args


def _mixed_mesh_field_pointer(field, group):
    name = _residual_parent_field_name(field)
    if int(getattr(field, "components", 1)) == 1:
        return "%s%s" % (name, group.pointer_suffix)
    return "%s%s[%d]" % (name, group.pointer_suffix, int(getattr(field, "component", 0)))


def _mixed_mesh_output_pointer(field):
    name = _residual_parent_field_name(field)
    if int(getattr(field, "components", 1)) == 1:
        return "%s_out" % name
    return "%s_out[%d]" % (name, int(getattr(field, "component", 0)))


def _mixed_tensor_product_field_stream_order(
    layout,
    cell_rule,
    field_element_types,
    basis_family,
):
    if not _is_tensor_product_family(cell_rule, basis_family):
        return tuple(range(layout.total_streams))

    field_element_types = {} if field_element_types is None else field_element_types
    order = []
    for field_index, field in enumerate(layout.fields):
        n_shape = layout.n_shape(field_index)
        element_type = _field_element_type(
            _residual_parent_field_name(field),
            cell_rule,
            field_element_types,
        )
        shape_order = (
            tuple(range(n_shape))
            if sfem_tensor_product_hex_uses_cartesian_ordering(element_type)
            else tensor_product_cartesian_shape_order(cell_rule.dim, n_shape)
        )
        order.extend(layout.stream_index(field_index, shape) for shape in shape_order)
    return tuple(order)


def _mixed_field_shape_orders(
    layout,
    cell_rule,
    field_element_types,
    basis_family,
):
    if not _is_tensor_product_family(cell_rule, basis_family):
        return tuple(tuple(range(layout.n_shape(field_index))) for field_index in range(len(layout.fields)))

    field_element_types = {} if field_element_types is None else field_element_types
    orders = []
    for field_index, field in enumerate(layout.fields):
        n_shape = layout.n_shape(field_index)
        element_type = _field_element_type(
            _residual_parent_field_name(field),
            cell_rule,
            field_element_types,
        )
        orders.append(
            tuple(range(n_shape))
            if sfem_tensor_product_hex_uses_cartesian_ordering(element_type)
            else tensor_product_cartesian_shape_order(cell_rule.dim, n_shape)
        )
    return tuple(orders)


def _mixed_field_element_alias_lines(
    layout,
    cell_rule,
    field_element_types,
    basis_family,
    indent,
):
    shape_orders = _mixed_field_shape_orders(
        layout,
        cell_rule,
        field_element_types,
        basis_family,
    )
    lines = []
    arrays = []
    for field_index, shape_order in enumerate(shape_orders):
        if _identity_order(shape_order):
            arrays.append("elements")
            continue
        array_name = "field_%d_elements" % field_index
        lines.append(
            "%sconst idx_t *const SFEM_RESTRICT %s[%d] = {%s};"
            % (
                indent,
                array_name,
                layout.n_shape(field_index),
                ", ".join("elements[%d]" % shape for shape in shape_order),
            )
        )
        arrays.append(array_name)
    return lines, tuple(arrays)


def _mixed_block_stream_pointer_lines(
    layout,
    dependencies,
    indent,
    cell_rule,
    field_element_types,
    basis_family,
    force_contiguous=False,
):
    if force_contiguous:
        args = {
            group.name: "block_%s" % group.name
            for group in _dependency_stream_groups(dependencies)
        }
        args["output"] = "block_output"
        return [], args

    field_stream_order = _mixed_tensor_product_field_stream_order(
        layout,
        cell_rule,
        field_element_types,
        basis_family,
    )
    lines = []
    args = {}
    for group in _dependency_stream_groups(dependencies):
        group_lines, group_arg = _block_stream_argument(
            "const scalar_t *",
            "block_%s_streams" % group.name,
            "block_%s" % group.name,
            layout.total_streams,
            field_stream_order,
            indent,
            mutable=False,
        )
        lines.extend(group_lines)
        args[group.name] = group_arg
    output_lines, output_arg = _block_stream_argument(
        "scalar_t *",
        "block_output_streams",
        "block_output",
        layout.total_streams,
        field_stream_order,
        indent,
        mutable=True,
    )
    lines.extend(output_lines)
    args["output"] = output_arg
    return lines, args


def _single_field_block_stream_arguments(
    dependencies,
    n_streams,
    field_stream_order,
    indent,
    force_contiguous=False,
):
    if force_contiguous:
        args = {
            group.name: "block_%s" % group.name
            for group in _dependency_stream_groups(dependencies)
        }
        args["output"] = "block_output"
        return [], args

    lines = []
    args = {}
    for group in _dependency_stream_groups(dependencies):
        group_lines, group_arg = _block_stream_argument(
            "const scalar_t *",
            "block_%s_streams" % group.name,
            "block_%s" % group.name,
            n_streams,
            field_stream_order,
            indent,
            mutable=False,
        )
        lines.extend(group_lines)
        args[group.name] = group_arg
    output_lines, output_arg = _block_stream_argument(
        "scalar_t *",
        "block_output_streams",
        "block_output",
        n_streams,
        field_stream_order,
        indent,
        mutable=True,
    )
    lines.extend(output_lines)
    args["output"] = output_arg
    return lines, args


def _mixed_field_gather_lines(system, layout, dependencies, indent, field_element_arrays=None):
    lines = []
    dependency_groups = tuple(_dependency_stream_groups(dependencies, mesh=True))
    if field_element_arrays is None:
        field_element_arrays = tuple("elements" for _ in system.fields)
    for field_index, field in enumerate(system.fields):
        n_shape = layout.n_shape(field_index)
        offset = layout.offset(field_index)
        field_lines = []
        for group in dependency_groups:
            field_lines.append(
                "%sblock_%s[stream][lane] = %s[node * %s];"
                % (
                    indent + "        ",
                    group.name,
                    _mixed_mesh_field_pointer(field, group),
                    group.stride,
                )
            )
        if not field_lines:
            continue
        lines.extend(
            _mesh_block_gather_loop_lines(
                indent,
                MeshBlockGatherLoop(
                    shape_var="local_shape",
                    shape_count=str(n_shape),
                    element_array=field_element_arrays[field_index],
                    setup_lines=(
                        "%s    const int stream = %d + local_shape;" % (indent, offset),
                    ),
                    lane_indent="%s    " % indent,
                ),
                field_lines,
            )
        )
    return lines


def _mixed_field_atomic_scatter_lines(system, layout, indent, field_element_arrays=None):
    lines = []
    if field_element_arrays is None:
        field_element_arrays = tuple("elements" for _ in system.fields)
    for field_index, field in enumerate(system.fields):
        n_shape = layout.n_shape(field_index)
        offset = layout.offset(field_index)
        lines.extend(
            [
                "%s{" % indent,
                "%s    scalar_t *const SFEM_RESTRICT out = %s;"
                % (indent, _mixed_mesh_output_pointer(field)),
                *_mesh_block_scatter_loop_lines(
                    "%s    " % indent,
                    MeshBlockScatterLoop(
                        shape_var="local_shape",
                        shape_count=str(n_shape),
                        element_array=field_element_arrays[field_index],
                        setup_lines=(
                            "%s        const int stream = %d + local_shape;"
                            % (indent, offset),
                        ),
                        scatter_indent="%s        " % indent,
                    ),
                    "%s            out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];"
                    % indent,
                ),
                "%s}" % indent,
            ]
        )
    return lines


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
    candidate_parameters = tuple(
        dict.fromkeys(tuple(dependencies.parameters) + tuple(system.parameters))
    )
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
            parameter for parameter in candidate_parameters if parameter in free_symbols
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


def _is_tensor_product_family(rule, basis_family=None):
    if basis_family is None:
        raise ValueError("basis family must be provided by the emission plan")
    return str(basis_family) == "tensor_product"


def generate_coupled_residual_sfem_files(
    system,
    *,
    prefix,
    emission_plan,
    residual_coeffs=None,
    action_coeffs=None,
    local_prefix=None,
    local_name=None,
    operator_prefix=None,
    operator_name=None,
    reference_data_plan=None,
    diagnostics_plan=None,
    matrix_format_plan=None,
):
    if not isinstance(system, CoupledResidualSystem):
        raise TypeError("system must be CoupledResidualSystem")
    if emission_plan is None:
        raise ValueError("residual code generation requires an ElementEmissionPlan")
    element_type = emission_plan.element_type
    specialization = emission_plan.isoparametric_specialization
    affine_specialization = emission_plan.affine_specialization
    if system.dim != specialization.dim:
        raise ValueError("residual system dimension does not match element dimension")
    if affine_specialization.dim != specialization.dim:
        raise ValueError("affine and isoparametric residual specializations must have the same dimension")
    if affine_specialization.n_shape != specialization.n_shape:
        raise ValueError("affine and isoparametric residual specializations must have the same shape count")
    if residual_coeffs is None:
        residual_coeffs = coupled_residual_weak_coefficients(system, False)
    if action_coeffs is None:
        action_coeffs = coupled_residual_weak_coefficients(system, True)
    family = emission_plan.basis_family
    geometry_family = emission_plan.geometry_family
    local_prefix = "%s_d%d_%s" % (prefix, system.dim, family) if local_prefix is None else str(local_prefix)
    element_prefix = "%s_%s" % (prefix, element_type.lower()) if operator_prefix is None else str(operator_prefix)
    if reference_data_plan is not None:
        validate_reference_data_plan(
            reference_data_plan,
            element_prefix,
            affine_specialization.quadrature_rule,
            specialization.quadrature_rule,
            family,
        )
    if diagnostics_plan is not None:
        expected_diagnostics = [
            "%s_residual_element_soa" % element_prefix,
        ]
        expected_diagnostics.extend(
            "%s_%s" % (element_prefix, block.name)
            for block in system.jacobian_blocks()
        )
        expected_diagnostics.append("%s_jacobian_action_element_soa" % element_prefix)
        validate_diagnostics_plan_names(diagnostics_plan, expected_diagnostics)
    local_name = "%s_local.hpp" % local_prefix if local_name is None else str(local_name)
    operator_name = "%s_operator.cpp" % element_prefix if operator_name is None else str(operator_name)
    local_source = _local_header(
        system,
        local_prefix,
        specialization,
        residual_coeffs,
        action_coeffs,
        basis_family=family,
    )
    operator_source = _operator_source(
        system,
        element_prefix,
        local_prefix,
        specialization,
        affine_specialization,
        local_name,
        residual_coeffs,
        action_coeffs,
        basis_family=family,
        geometry_family=geometry_family,
        matrix_format_plan=matrix_format_plan,
    )
    diagnostics_name = "kernel_diagnostics.hpp"
    return (
        GeneratedKernelFile("kernel_math.hpp", _sfem_math_header_source()),
        GeneratedKernelFile(
            "tensor_product_kernels.hpp",
            sfem_tensor_product_kernels_header_source(),
        ),
        GeneratedKernelFile(
            "geometry_kernels.hpp",
            sfem_geometry_kernels_header_source(),
        ),
        GeneratedKernelFile(
            diagnostics_name,
            "\n".join(_sfem_soa_diagnostics_header()),
        ),
        GeneratedKernelFile(
            "packed_thread_scratch.hpp",
            _sfem_packed_thread_scratch_header_source(),
        ),
        GeneratedKernelFile(local_name, local_source),
        GeneratedKernelFile(operator_name, operator_source),
    )


def generate_mixed_residual_sfem_files(
    system,
    *,
    prefix,
    compatible_element,
    emission_plan,
    residual_coeffs=None,
    action_coeffs=None,
    field_element_types=None,
    local_prefix=None,
    local_name=None,
    operator_prefix=None,
    operator_name=None,
    reference_data_plan=None,
    diagnostics_plan=None,
):
    if not isinstance(system, CoupledResidualSystem):
        raise TypeError("system must be CoupledResidualSystem")
    if emission_plan is None:
        raise ValueError("mixed residual code generation requires an ElementEmissionPlan")
    cell_specialization = emission_plan.isoparametric_specialization
    affine_specialization = emission_plan.affine_specialization
    if system.dim != cell_specialization.dim:
        raise ValueError("residual system dimension does not match element dimension")
    field_element_types = dict(field_element_types or ())
    missing_fields = tuple(
        field_name
        for field_name in dict.fromkeys(
            _residual_parent_field_name(field) for field in system.fields
        )
        if field_name not in field_element_types
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
    family = emission_plan.basis_family
    geometry_family = emission_plan.geometry_family
    if reference_data_plan is not None:
        validate_reference_data_plan(
            reference_data_plan,
            prefix,
            affine_specialization.quadrature_rule,
            cell_specialization.quadrature_rule,
            family,
        )
    local_prefix = "%s_d%d_%s_mixed" % (prefix, system.dim, family) if local_prefix is None else str(local_prefix)
    element_prefix = "%s_%s" % (prefix, compatible_element.name.lower()) if operator_prefix is None else str(operator_prefix)
    if diagnostics_plan is not None:
        validate_diagnostics_plan_names(
            diagnostics_plan,
            (
                "%s_residual_element_soa" % element_prefix,
                "%s_jacobian_action_element_soa" % element_prefix,
            ),
        )
    local_name = "%s_local.hpp" % local_prefix if local_name is None else str(local_name)
    operator_name = "%s_operator.cpp" % element_prefix if operator_name is None else str(operator_name)
    local_source = _mixed_local_header(
        system,
        local_prefix,
        cell_specialization,
        field_element_types,
        residual_coeffs,
        action_coeffs,
        basis_family=family,
    )
    operator_source = _mixed_operator_source(
        system,
        prefix,
        local_prefix,
        local_name,
        cell_specialization,
        compatible_element,
        field_element_types,
        residual_coeffs,
        action_coeffs,
        basis_family=family,
        geometry_family=geometry_family,
    )
    return (
        GeneratedKernelFile("kernel_math.hpp", _sfem_math_header_source()),
        GeneratedKernelFile(
            "tensor_product_kernels.hpp",
            sfem_tensor_product_kernels_header_source(),
        ),
        GeneratedKernelFile(
            "geometry_kernels.hpp",
            sfem_geometry_kernels_header_source(),
        ),
        GeneratedKernelFile(
            "kernel_diagnostics.hpp",
            "\n".join(_sfem_soa_diagnostics_header()),
        ),
        GeneratedKernelFile(
            "packed_thread_scratch.hpp",
            _sfem_packed_thread_scratch_header_source(),
        ),
        GeneratedKernelFile(local_name, local_source),
        GeneratedKernelFile(operator_name, operator_source),
    )


def _local_header(system, local_prefix, specialization, residual_coeffs, action_coeffs, basis_family=None):
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
        '#include "tensor_product_kernels.hpp"',
        "",
        *_inline_definition_lines(),
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
    lines.extend(
        _local_function(
            system,
            "%s_residual_block" % local_prefix,
            specialization,
            residual_coeffs,
            dependencies=residual_dependencies,
            local_prefix=local_prefix,
            basis_family=basis_family,
            allow_simplex_gradient_metric=False,
            constant_p1_gradient_expansion=False,
        )
    )
    lines.append("")
    lines.extend(
        _local_function(
            system,
            "%s_residual_block_contiguous" % local_prefix,
            specialization,
            residual_coeffs,
            dependencies=residual_dependencies,
            local_prefix=local_prefix,
            basis_family=basis_family,
            allow_simplex_gradient_metric=False,
            constant_p1_gradient_expansion=False,
            stream_layout="contiguous",
        )
    )
    lines.append("")
    specialized = _constant_p1_affine_specialized_local(
        local_prefix,
        specialization,
    )
    specialized_prefix = specialized[0] if specialized is not None else None
    specialized_specialization = specialized[1] if specialized is not None else None
    if specialized_prefix is not None:
        lines.extend(
            _local_function(
                system,
                "%s_residual_block" % specialized_prefix,
                specialized_specialization,
                residual_coeffs,
                dependencies=residual_dependencies,
                local_prefix=local_prefix,
                basis_family=basis_family,
                allow_simplex_gradient_metric=True,
                constant_p1_gradient_expansion=True,
            )
        )
        lines.append("")
        lines.extend(
            _local_function(
                system,
                "%s_residual_block_contiguous" % specialized_prefix,
                specialized_specialization,
                residual_coeffs,
                dependencies=residual_dependencies,
                local_prefix=local_prefix,
                basis_family=basis_family,
                allow_simplex_gradient_metric=True,
                constant_p1_gradient_expansion=True,
                stream_layout="contiguous",
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
            basis_family=basis_family,
            allow_simplex_gradient_metric=False,
            constant_p1_gradient_expansion=False,
        )
    )
    lines.append("")
    lines.extend(
        _local_function(
            system,
            "%s_jacobian_action_block_contiguous" % local_prefix,
            specialization,
            action_coeffs,
            dependencies=action_dependencies,
            local_prefix=local_prefix,
            basis_family=basis_family,
            allow_simplex_gradient_metric=False,
            constant_p1_gradient_expansion=False,
            stream_layout="contiguous",
        )
    )
    if specialized_prefix is not None:
        lines.append("")
        lines.extend(
            _local_function(
                system,
                "%s_jacobian_action_block" % specialized_prefix,
                specialized_specialization,
                action_coeffs,
                dependencies=action_dependencies,
                local_prefix=local_prefix,
                basis_family=basis_family,
                allow_simplex_gradient_metric=True,
                constant_p1_gradient_expansion=True,
            )
        )
        lines.append("")
        lines.extend(
            _local_function(
                system,
                "%s_jacobian_action_block_contiguous" % specialized_prefix,
                specialized_specialization,
                action_coeffs,
                dependencies=action_dependencies,
                local_prefix=local_prefix,
                basis_family=basis_family,
                allow_simplex_gradient_metric=True,
                constant_p1_gradient_expansion=True,
                stream_layout="contiguous",
            )
        )
    lines.extend(
        ["", "} // namespace codegen", "} // namespace sfem", "", "#endif", ""]
    )
    return "\n".join(lines)


def _constant_p1_affine_specialized_local(local_prefix, specialization):
    rule = specialization.quadrature_rule
    specialized_prefix = _constant_p1_affine_specialized_local_prefix(
        local_prefix,
        rule,
    )
    if specialized_prefix is not None:
        return specialized_prefix, specialization

    if rule is None or not str(local_prefix).endswith("_simplex"):
        return None

    element_type = {2: "TRI3", 3: "TET4"}.get(int(getattr(rule, "dim", 0)))
    if element_type is None:
        return None

    p1_specialization = sfem_soa_element_specialization(
        element_type,
        vector_size=specialization.vector_size,
    )
    p1_prefix = _constant_p1_affine_specialized_local_prefix(
        local_prefix,
        p1_specialization.quadrature_rule,
    )
    if p1_prefix is None:
        return None
    return p1_prefix, p1_specialization


def _constant_p1_affine_specialized_local_prefix(local_prefix, rule):
    if rule is None:
        return None
    if constant_p1_simplex_reference_gradients(rule) is None:
        return None
    element_type = str(getattr(rule, "element_type", "")).lower()
    if element_type not in ("tri3", "tet4"):
        return None
    return "%s_%s" % (local_prefix, element_type)


def _mixed_local_header(
    system,
    local_prefix,
    specialization,
    field_element_types,
    residual_coeffs,
    action_coeffs,
    basis_family=None,
):
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
        '#include "tensor_product_kernels.hpp"',
        "",
        *_inline_definition_lines(),
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
    lines.extend(
        _mixed_local_function(
            system,
            "%s_residual_block" % local_prefix,
            specialization,
            field_element_types,
            residual_coeffs,
            dependencies=residual_dependencies,
            basis_family=basis_family,
        )
    )
    lines.append("")
    lines.extend(
        _mixed_local_function(
            system,
            "%s_residual_block_contiguous" % local_prefix,
            specialization,
            field_element_types,
            residual_coeffs,
            dependencies=residual_dependencies,
            basis_family=basis_family,
            stream_layout="contiguous",
        )
    )
    lines.append("")
    lines.extend(
        _mixed_local_function(
            system,
            "%s_jacobian_action_block" % local_prefix,
            specialization,
            field_element_types,
            action_coeffs,
            dependencies=action_dependencies,
            basis_family=basis_family,
        )
    )
    lines.append("")
    lines.extend(
        _mixed_local_function(
            system,
            "%s_jacobian_action_block_contiguous" % local_prefix,
            specialization,
            field_element_types,
            action_coeffs,
            dependencies=action_dependencies,
            basis_family=basis_family,
            stream_layout="contiguous",
        )
    )
    lines.extend(
        ["", "} // namespace codegen", "} // namespace sfem", "", "#endif", ""]
    )
    return "\n".join(lines)


def _mixed_local_function(
    system,
    function_name,
    specialization,
    field_element_types,
    coefficients,
    dependencies,
    basis_family=None,
    stream_layout="pointer",
):
    rule = specialization.quadrature_rule
    dim = system.dim
    layout = MixedFieldLayout.create(system, rule, field_element_types)
    params = [
        "const int nelems",
        "const ptrdiff_t geometry_stride",
        "const scalar_t *const SFEM_RESTRICT determinant",
    ]
    if dependencies.uses_adjugate:
        params.append(
            "const scalar_t *const SFEM_RESTRICT adjugate[%d]" % (dim * dim)
        )
    params.extend(_mixed_local_reference_params(rule, layout.n_reference_fields, dim, dependencies, basis_family))
    if dependencies.current:
        if stream_layout == "contiguous":
            params.append(
                "const scalar_t current[%d][VECTOR_SIZE]" % layout.total_streams
            )
        else:
            params.append(
                "const scalar_t *const SFEM_RESTRICT current[%d]"
                % layout.total_streams
            )
    if dependencies.previous:
        if stream_layout == "contiguous":
            params.append(
                "const scalar_t previous[%d][VECTOR_SIZE]" % layout.total_streams
            )
        else:
            params.append(
                "const scalar_t *const SFEM_RESTRICT previous[%d]"
                % layout.total_streams
            )
    if dependencies.direction:
        if stream_layout == "contiguous":
            params.append(
                "const scalar_t direction[%d][VECTOR_SIZE]" % layout.total_streams
            )
        else:
            params.append(
                "const scalar_t *const SFEM_RESTRICT direction[%d]"
                % layout.total_streams
            )
    params.extend(
        "const scalar_t %s" % parameter for parameter in dependencies.parameters
    )
    if stream_layout == "contiguous":
        params.append("scalar_t output[%d][VECTOR_SIZE]" % layout.total_streams)
    else:
        params.append(
            "scalar_t *const SFEM_RESTRICT output[%d]" % layout.total_streams
        )
    template_params = [
        "typename scalar_t",
        "int N_QP",
        "int CELL_N_SHAPE",
        "int VECTOR_SIZE",
    ]
    lines = [
        "template <%s>" % ", ".join(template_params),
        "%s void %s(" % (_function_qualifier(), function_name),
    ]
    for index, param in enumerate(params):
        lines.append("        %s%s" % (param, "," if index + 1 < len(params) else ""))
    lines.extend(
        [
            ") {",
            "    static constexpr int DIM = %d;" % dim,
            "    static constexpr int N_FIELDS = %d;" % layout.n_reference_fields,
            "    static constexpr int N_FIELD_STREAMS = %d;" % layout.total_streams,
            "    (void)CELL_N_SHAPE;",
            "    (void)N_FIELD_STREAMS;",
        ]
    )
    lines.extend(
        "    static constexpr int %s_N_SHAPE = %d;"
        % (group.name.upper(), group.shape_count)
        for group in layout.groups
    )
    if _is_tensor_product_family(rule, basis_family):
        lines.extend(
            _mixed_tensor_local_body(
                system,
                layout,
                coefficients,
                dependencies,
                stream_layout=stream_layout,
            )
        )
    else:
        lines.extend(_mixed_simplex_local_body(system, layout, coefficients, dependencies))
    lines.append("}")
    return lines


def _mixed_simplex_local_body(system, layout, coefficients, dependencies):
    dim = system.dim
    lines = [
        "    for (int q = 0; q < N_QP; ++q) {",
        *_work_item_loop_lines("        "),
        "            const ptrdiff_t geometry_offset = q * geometry_stride + lane;",
        "            const scalar_t det = determinant[geometry_offset];",
    ]
    if dependencies.uses_adjugate:
        for i in range(dim * dim):
            lines.append(
                "            const scalar_t adj%d = adjugate[%d][geometry_offset];"
                % (i, i)
            )
    lines.extend(
        _mixed_local_field_evaluation_lines(
            system,
            layout,
            dependencies,
            "            ",
            unroll=True,
        )
    )
    lines.extend(
        _coefficient_evaluation_lines(
            system,
            coefficients,
            "            ",
            "q_weight[q]",
            dependencies,
        )
    )
    for row, field in enumerate(system.fields):
        offset = layout.offset(row)
        n_shape_name = layout.n_shape_constant(field)
        reference_index = layout.reference_index(row)
        for test in range(layout.n_shape(row)):
            if dependencies.value_coefficients[row]:
                lines.append(
                    "            const scalar_t test_value_%s_%d = field_shape[%d][q * %s + %d];"
                    % (field.name, test, reference_index, n_shape_name, test)
                )
            for d in range(dim):
                if not dependencies.gradient_coefficients[row][d]:
                    continue
                terms = [
                    "field_grad_ref[%d * DIM + %d][q * %s + %d] * adj%d"
                    % (reference_index, k, n_shape_name, test, k * dim + d)
                    for k in range(dim)
                ]
                lines.append(
                    "            const scalar_t test_grad%d_%s_%d = (%s) / det;"
                    % (d, field.name, test, " + ".join(terms))
                )
            terms = []
            if dependencies.value_coefficients[row]:
                terms.append("value_coeff%d * test_value_%s_%d" % (row, field.name, test))
            terms.extend(
                "grad_coeff%d_%d * test_grad%d_%s_%d" % (row, d, d, field.name, test)
                for d in range(dim)
                if dependencies.gradient_coefficients[row][d]
            )
            if terms:
                lines.append(
                    "            output[%d][lane] += q_weight[q] * det * (%s);"
                    % (offset + test, " + ".join(terms))
                )
    lines.extend(["        }", "    }"])
    return lines


def _mixed_tensor_local_body(system, layout, coefficients, dependencies, stream_layout="pointer"):
    dim = system.dim
    groups = _dependency_stream_groups(dependencies)
    uses_determinant = any(dependencies.value_coefficients) or dependencies.uses_adjugate
    uses_geometry_offset = uses_determinant or dependencies.uses_adjugate

    lines = [
        "    static constexpr int N_QP_1D = integer_root(N_QP, DIM);",
        "    static_assert(ipow(N_QP_1D, DIM) == N_QP, \"N_QP must be tensor-product compatible\");",
    ]
    for group in layout.groups:
        shape_name = "%s_N_SHAPE" % group.name.upper()
        shape_1d_name = "%s_N_SHAPE_1D" % group.name.upper()
        lines.extend(
            [
                "    static constexpr int %s = integer_root(%s, DIM);"
                % (shape_1d_name, shape_name),
                "    static_assert(ipow(%s, DIM) == %s, \"%s must be tensor-product compatible\");"
                % (shape_1d_name, shape_name, shape_name),
            ]
        )
    if not dependencies.uses_test_coefficients:
        return lines
    tensor_evaluate_name = (
        "tensor_evaluate_contiguous" if stream_layout == "contiguous" else "tensor_evaluate"
    )
    tensor_evaluate_value_name = (
        "tensor_evaluate_value_contiguous"
        if stream_layout == "contiguous"
        else "tensor_evaluate_value"
    )
    tensor_integrate_name = (
        "tensor_integrate_contiguous" if stream_layout == "contiguous" else "tensor_integrate"
    )
    tensor_integrate_value_name = (
        "tensor_integrate_value_contiguous"
        if stream_layout == "contiguous"
        else "tensor_integrate_value"
    )

    for field_index, field in enumerate(system.fields):
        reference_index = layout.reference_index(field_index)
        shape_name = layout.n_shape_constant(field)
        for group in groups:
            if group.uses_value or group.uses_gradient:
                lines.append(
                    "    scalar_t %s_%s_value[N_QP * VECTOR_SIZE];"
                    % (group.name, field.name)
                )
            if group.uses_gradient:
                lines.append(
                    "    scalar_t %s_%s_grad_ref[N_QP * DIM * VECTOR_SIZE];"
                    % (group.name, field.name)
                )
            if stream_layout == "contiguous":
                stream_arg = "%s + %d" % (group.name, layout.offset(field_index))
            else:
                stream_arg = "%s_%s_streams" % (group.name, field.name)
                lines.append(
                    "    const scalar_t *const %s[%s] = {%s};"
                    % (
                        stream_arg,
                        shape_name,
                        _field_stream_initializer(layout, field_index, group.name),
                    )
                )
            if group.uses_gradient:
                lines.extend(
                    [
                        "    %s<scalar_t, N_QP, %s, VECTOR_SIZE, DIM, 1>("
                        % (tensor_evaluate_name, shape_name),
                        "            nelems, field_shape_1d[%d], field_grad_1d[%d], %s, %s_%s_value, %s_%s_grad_ref);"
                        % (
                            reference_index,
                            reference_index,
                            stream_arg,
                            group.name,
                            field.name,
                            group.name,
                            field.name,
                        ),
                    ]
                )
            elif group.uses_value:
                lines.extend(
                    [
                        "    %s<scalar_t, N_QP, %s, VECTOR_SIZE, DIM, 1>("
                        % (tensor_evaluate_value_name, shape_name),
                        "            nelems, field_shape_1d[%d], %s, %s_%s_value);"
                        % (reference_index, stream_arg, group.name, field.name),
                    ]
                )

    for row, field in enumerate(system.fields):
        lines.append(
            "    scalar_t %s_value_coeff[N_QP * VECTOR_SIZE];" % field.name
        )
        if dependencies.uses_test_gradients:
            lines.append(
                "    scalar_t %s_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];"
                % field.name
            )

    lines.extend(["    for (int q = 0; q < N_QP; ++q) {"])
    if dim == 2:
        lines.extend(
            [
                "        const int qx = q % N_QP_1D;",
                "        const int qy = q / N_QP_1D;",
                "        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy];",
            ]
        )
    else:
        lines.extend(
            [
                "        const int qx = q % N_QP_1D;",
                "        const int qy = (q / N_QP_1D) % N_QP_1D;",
                "        const int qz = q / (N_QP_1D * N_QP_1D);",
                "        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];",
            ]
        )
    lines.extend(
        [
            *_work_item_loop_lines("        "),
        ]
    )
    if uses_geometry_offset:
        lines.append("            const ptrdiff_t geometry_offset = q * geometry_stride + lane;")
    if uses_determinant:
        lines.append("            const scalar_t det = determinant[geometry_offset];")
    if dependencies.uses_adjugate:
        for i in range(dim * dim):
            lines.append(
                "            const scalar_t adj%d = adjugate[%d][geometry_offset];"
                % (i, i)
            )
    for field_index, field in enumerate(system.fields):
        for group in groups:
            if group.uses_value:
                lines.append(
                    "            const scalar_t %s%s = %s_%s_value[q * VECTOR_SIZE + lane];"
                    % (field.name, group.symbol_suffix, group.name, field.name)
                )
            if group.uses_gradient:
                for k in range(dim):
                    lines.append(
                        "            const scalar_t %s%s_grad_%d_ref = %s_%s_grad_ref[(q * DIM + %d) * VECTOR_SIZE + lane];"
                        % (field.name, group.symbol_suffix, k, group.name, field.name, k)
                    )
                lines.extend(
                    _physical_gradient_lines(field.name + group.symbol_suffix, dim, "            ")
                )
    lines.extend(
        _coefficient_evaluation_lines(
            system,
            coefficients,
            "            ",
            "qw",
            dependencies,
        )
    )
    for row, field in enumerate(system.fields):
        if dependencies.value_coefficients[row]:
            value = "qw * det * value_coeff%d" % row
        else:
            value = "scalar_t(0)"
        lines.append(
            "            %s_value_coeff[q * VECTOR_SIZE + lane] = %s;"
            % (field.name, value)
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
                    "            %s_grad_coeff_ref[(q * DIM + %d) * VECTOR_SIZE + lane] = %s;"
                    % (field.name, k, value)
                )
    lines.extend(["        }", "    }"])
    for row, field in enumerate(system.fields):
        shape_name = layout.n_shape_constant(field)
        offset = layout.offset(row)
        reference_index = layout.reference_index(row)
        if stream_layout == "contiguous":
            output_arg = "output + %d" % offset
        else:
            output_arg = "%s_output_streams" % field.name
            lines.append(
                "    scalar_t *const %s[%s] = {%s};"
                % (
                    output_arg,
                    shape_name,
                    _indexed_stream_range_initializer(
                        "output", offset, layout.n_shape(row)
                    ),
                )
            )
        if dependencies.uses_test_gradients:
            lines.extend(
                [
                    "    %s<scalar_t, N_QP, %s, VECTOR_SIZE, DIM, 1>("
                    % (tensor_integrate_name, shape_name),
                    "            nelems, field_shape_1d[%d], field_grad_1d[%d], %s_value_coeff, %s_grad_coeff_ref, %s);"
                    % (reference_index, reference_index, field.name, field.name, output_arg),
                ]
            )
        else:
            lines.extend(
                [
                    "    %s<scalar_t, N_QP, %s, VECTOR_SIZE, DIM, 1>("
                    % (tensor_integrate_value_name, shape_name),
                    "            nelems, field_shape_1d[%d], %s_value_coeff, %s);"
                    % (reference_index, field.name, output_arg),
                ]
            )
    return lines


def _mixed_local_field_evaluation_lines(
    system,
    layout,
    dependencies,
    indent,
    *,
    unroll=False,
):
    dim = system.dim
    lines = []
    groups = _dependency_stream_groups(dependencies)
    for field_index, field in enumerate(system.fields):
        n_shape_name = layout.n_shape_constant(field)
        offset = layout.offset(field_index)
        reference_index = layout.reference_index(field_index)
        for group in groups:
            if group.uses_value:
                lines.append("%sscalar_t %s%s = scalar_t(0);" % (indent, field.name, group.symbol_suffix))
            if group.uses_gradient:
                for d in range(dim):
                    lines.append(
                        "%sscalar_t %s%s_grad_%d_ref = scalar_t(0);"
                        % (indent, field.name, group.symbol_suffix, d)
                    )
            if unroll:
                for trial in range(layout.n_shape(field_index)):
                    coeff_name = "coeff_%s_%s_%d" % (group.name, field.name, trial)
                    lines.append(
                        "%sconst scalar_t %s = %s[%d][lane];"
                        % (indent, coeff_name, group.name, offset + trial)
                    )
                    if group.uses_value:
                        lines.append(
                            "%s%s%s += %s * field_shape[%d][q * %s + %d];"
                            % (
                                indent,
                                field.name,
                                group.symbol_suffix,
                                coeff_name,
                                reference_index,
                                n_shape_name,
                                trial,
                            )
                        )
                    if group.uses_gradient:
                        for d in range(dim):
                            lines.append(
                                "%s%s%s_grad_%d_ref += %s * field_grad_ref[%d * DIM + %d][q * %s + %d];"
                                % (
                                    indent,
                                    field.name,
                                    group.symbol_suffix,
                                    d,
                                    coeff_name,
                                    reference_index,
                                    d,
                                    n_shape_name,
                                    trial,
                                )
                            )
            else:
                lines.append("%sfor (int trial = 0; trial < %s; ++trial) {" % (indent, n_shape_name))
                lines.append(
                    "%s    const scalar_t coeff = %s[%d + trial][lane];"
                    % (indent, group.name, offset)
                )
                if group.uses_value:
                    lines.append(
                        "%s    %s%s += coeff * field_shape[%d][q * %s + trial];"
                        % (indent, field.name, group.symbol_suffix, reference_index, n_shape_name)
                    )
                if group.uses_gradient:
                    for d in range(dim):
                        lines.append(
                            "%s    %s%s_grad_%d_ref += coeff * field_grad_ref[%d * DIM + %d][q * %s + trial];"
                            % (
                                indent,
                                field.name,
                                group.symbol_suffix,
                                d,
                                reference_index,
                                d,
                                n_shape_name,
                            )
                        )
                lines.append("%s}" % indent)
            if group.uses_gradient:
                lines.extend(
                    _physical_gradient_lines(field.name + group.symbol_suffix, dim, indent)
                )
    return lines


def _local_function(
    system,
    function_name,
    specialization,
    coefficients,
    dependencies,
    local_prefix,
    basis_family=None,
    allow_simplex_gradient_metric=True,
    constant_p1_gradient_expansion=True,
    stream_layout="pointer",
):
    rule = specialization.quadrature_rule
    dim = system.dim
    n_fields = len(system.fields)
    tensor_product = _is_tensor_product_family(rule, basis_family)
    gradient_metric = (
        None
        if tensor_product or not allow_simplex_gradient_metric
        else simplex_gradient_metric_transformation(system, rule, coefficients, dependencies)
    )
    omit_simplex_reference_basis_inputs = gradient_metric is not None
    params = [
        "const int nelems",
        "const ptrdiff_t geometry_stride",
    ]
    if gradient_metric is not None:
        params.append(
            "const scalar_t *const SFEM_RESTRICT geom_metric[%d]"
            % symmetric_metric_component_count(dim)
        )
    else:
        params.append("const scalar_t *const SFEM_RESTRICT determinant")
    if dependencies.uses_adjugate and gradient_metric is None:
        params.append(
            "const scalar_t *const SFEM_RESTRICT adjugate[%d]" % (dim * dim)
        )
    if tensor_product:
        params.append("const scalar_t *const SFEM_RESTRICT shape_1d")
        if dependencies.uses_reference_gradients:
            params.append("const scalar_t *const SFEM_RESTRICT grad_1d")
        params.append("const scalar_t *const SFEM_RESTRICT q_weight_1d")
    else:
        if not omit_simplex_reference_basis_inputs:
            params.append("const scalar_t *const SFEM_RESTRICT shape")
            if dependencies.uses_reference_gradients:
                params.extend(
                    "const scalar_t *const SFEM_RESTRICT %s"
                    % sfem_simplex_grad_ref_name("grad_ref", d)
                    for d in range(dim)
                )
        params.append("const scalar_t *const SFEM_RESTRICT q_weight")
    if dependencies.current:
        if stream_layout == "contiguous":
            params.append(
                "const scalar_t current[%d * N_SHAPE][VECTOR_SIZE]"
                % n_fields
            )
        else:
            params.append(
                "const scalar_t *const SFEM_RESTRICT current[%d * N_SHAPE]"
                % n_fields
            )
    if dependencies.previous:
        if stream_layout == "contiguous":
            params.append(
                "const scalar_t previous[%d * N_SHAPE][VECTOR_SIZE]"
                % n_fields
            )
        else:
            params.append(
                "const scalar_t *const SFEM_RESTRICT previous[%d * N_SHAPE]"
                % n_fields
            )
    if dependencies.direction:
        if stream_layout == "contiguous":
            params.append(
                "const scalar_t direction[%d * N_SHAPE][VECTOR_SIZE]"
                % n_fields
            )
        else:
            params.append(
                "const scalar_t *const SFEM_RESTRICT direction[%d * N_SHAPE]"
                % n_fields
            )
    params.extend(
        "const scalar_t %s" % parameter for parameter in dependencies.parameters
    )
    if stream_layout == "contiguous":
        params.append("scalar_t output[%d * N_SHAPE][VECTOR_SIZE]" % n_fields)
    else:
        params.append(
            "scalar_t *const SFEM_RESTRICT output[%d * N_SHAPE]" % n_fields
        )
    template_params = [
        "typename scalar_t",
        "int N_QP",
        "int N_SHAPE",
        "int VECTOR_SIZE",
    ]
    lines = [
        "template <%s>" % ", ".join(template_params),
        "%s void %s(" % (_function_qualifier(), function_name),
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
    if tensor_product:
        lines.extend(
            _tensor_local_body(
                system,
                local_prefix,
                coefficients,
                dependencies,
                stream_layout=stream_layout,
            )
        )
    else:
        lines.extend(
            _simplex_local_body(
                system,
                rule,
                coefficients,
                dependencies,
                gradient_metric,
                allow_gradient_metric=allow_simplex_gradient_metric,
                constant_p1_gradient_expansion=constant_p1_gradient_expansion,
            )
        )
    lines.append("}")
    return lines


def _simplex_local_body(
    system,
    rule,
    coefficients,
    dependencies,
    gradient_metric=None,
    allow_gradient_metric=True,
    constant_p1_gradient_expansion=True,
):
    if gradient_metric is None and allow_gradient_metric:
        gradient_metric = simplex_gradient_metric_transformation(system, rule, coefficients, dependencies)
    if gradient_metric is not None:
        return _simplex_gradient_metric_body(system, rule, dependencies, gradient_metric)
    reference_gradients = constant_p1_simplex_reference_gradients(rule)
    if constant_p1_gradient_expansion and _uses_constant_p1_gradient_expansion(system, dependencies, reference_gradients):
        return _constant_p1_gradient_expanded_body(
            system,
            coefficients,
            dependencies,
            reference_gradients,
        )

    dim = system.dim
    groups = _dependency_stream_groups(dependencies)
    lines = ["    for (int q = 0; q < N_QP; ++q) {"]
    for field in system.fields:
        for group in groups:
            if group.uses_value:
                lines.append("        scalar_t %s%s_values[VECTOR_SIZE];" % (field.name, group.symbol_suffix))
            if group.uses_gradient:
                for d in range(dim):
                    lines.append(
                        "        scalar_t %s%s_grad_%d_ref_values[VECTOR_SIZE];"
                        % (field.name, group.symbol_suffix, d)
                    )
    for row, _field in enumerate(system.fields):
        if dependencies.value_coefficients[row]:
            lines.append("        scalar_t value_coeff%d_values[VECTOR_SIZE];" % row)
        for d in range(dim):
            if dependencies.gradient_coefficients[row][d]:
                lines.append("        scalar_t grad_coeff%d_%d_values[VECTOR_SIZE];" % (row, d))
    for field_index, field in enumerate(system.fields):
        for group in groups:
            if group.uses_value:
                lines.extend(_work_item_loop_lines("        "))
                lines.append("            %s%s_values[lane] = scalar_t(0);" % (field.name, group.symbol_suffix))
                lines.append("        }")
            if group.uses_gradient:
                for d in range(dim):
                    lines.extend(_work_item_loop_lines("        "))
                    lines.append(
                        "            %s%s_grad_%d_ref_values[lane] = scalar_t(0);"
                        % (field.name, group.symbol_suffix, d)
                    )
                    lines.append("        }")
            lines.append("        for (int trial = 0; trial < N_SHAPE; ++trial) {")
            lines.extend(_work_item_loop_lines("            "))
            lines.append(
                "                const scalar_t coeff = %s[trial * N_FIELDS + %d][lane];"
                % (group.name, field_index)
            )
            if group.uses_value:
                lines.append(
                    "                %s%s_values[lane] += coeff * shape[q * N_SHAPE + trial];"
                    % (field.name, group.symbol_suffix)
                )
            if group.uses_gradient:
                for d in range(dim):
                    lines.append(
                        "                %s%s_grad_%d_ref_values[lane] += coeff * %s[q * N_SHAPE + trial];"
                        % (
                            field.name,
                            group.symbol_suffix,
                            d,
                            sfem_simplex_grad_ref_name("grad_ref", d),
                        )
                    )
            lines.append("            }")
            lines.append("        }")
    lines.extend(_work_item_loop_lines("        "))
    lines.extend(
        [
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
    for field in system.fields:
        for group in groups:
            if group.uses_value:
                lines.append(
                    "            const scalar_t %s%s = %s%s_values[lane];"
                    % (field.name, group.symbol_suffix, field.name, group.symbol_suffix)
                )
            if group.uses_gradient:
                for d in range(dim):
                    lines.append(
                        "            const scalar_t %s%s_grad_%d_ref = %s%s_grad_%d_ref_values[lane];"
                        % (
                            field.name,
                            group.symbol_suffix,
                            d,
                            field.name,
                            group.symbol_suffix,
                            d,
                        )
                    )
                lines.extend(
                    _physical_gradient_lines(
                        field.name + group.symbol_suffix, dim, "            "
                    )
                )
    lines.extend(
        _coefficient_evaluation_lines(
            system,
            coefficients,
            "            ",
            "q_weight[q]",
            dependencies,
        )
    )
    for row, _field in enumerate(system.fields):
        if dependencies.value_coefficients[row]:
            lines.append("            value_coeff%d_values[lane] = value_coeff%d;" % (row, row))
        for d in range(dim):
            if dependencies.gradient_coefficients[row][d]:
                lines.append(
                    "            grad_coeff%d_%d_values[lane] = grad_coeff%d_%d;"
                    % (row, d, row, d)
                )
    lines.append("        }")
    lines.extend(
        [
            "        for (int test = 0; test < N_SHAPE; ++test) {",
            *_work_item_loop_lines("            "),
            "                const ptrdiff_t geometry_offset = q * geometry_stride + lane;",
            "                const scalar_t det = determinant[geometry_offset];",
            "                const scalar_t test_value = shape[q * N_SHAPE + test];",
        ]
    )
    if dependencies.uses_adjugate:
        for i in range(dim * dim):
            lines.append(
                "                const scalar_t adj%d = adjugate[%d][geometry_offset];"
                % (i, i)
            )
    for d in range(dim):
        if not any(row[d] for row in dependencies.gradient_coefficients):
            continue
        terms = [
            "%s[q * N_SHAPE + test] * adj%d"
            % (sfem_simplex_grad_ref_name("grad_ref", k), k * dim + d)
            for k in range(dim)
        ]
        lines.append(
            "                const scalar_t test_grad%d = (%s) / det;"
            % (d, " + ".join(terms))
        )
    for row in range(len(system.fields)):
        terms = []
        if dependencies.value_coefficients[row]:
            terms.append("value_coeff%d_values[lane] * test_value" % row)
        terms.extend(
            "grad_coeff%d_%d_values[lane] * test_grad%d" % (row, d, d)
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


def _uses_constant_p1_gradient_expansion(system, dependencies, reference_gradients):
    if reference_gradients is None:
        return False
    if any(dependencies.value_coefficients):
        return False
    if not dependencies.uses_test_gradients:
        return False
    if dependencies.current_value or dependencies.previous_value or dependencies.direction_value:
        return False
    if not dependencies.uses_trial_gradients:
        return False
    return len(reference_gradients) == system.dim + 1


def _reference_gradient_expr(reference_gradients, shape, component):
    return sp.sympify(reference_gradients[int(shape)][int(component)])


def _constant_reference_gradient_sum(reference_gradients, n_shape, dim, expr_for_term):
    terms = []
    for shape in range(n_shape):
        factor = _reference_gradient_expr(reference_gradients, shape, dim)
        if factor == 0:
            continue
        terms.append(_scaled_cpp_term(factor, expr_for_term(shape)))
    return _sum_cpp_terms(terms)


def _constant_p1_gradient_expanded_body(system, coefficients, dependencies, reference_gradients):
    dim = system.dim
    n_fields = len(system.fields)
    groups = _dependency_stream_groups(dependencies)
    lines = ["    for (int q = 0; q < N_QP; ++q) {"]
    lines.extend(_work_item_loop_lines("        "))
    lines.extend(
        [
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
    for field_index, field in enumerate(system.fields):
        for group in groups:
            if not group.uses_gradient:
                continue
            for d in range(dim):
                value = _constant_reference_gradient_sum(
                    reference_gradients,
                    dim + 1,
                    d,
                    lambda shape, group=group, field_index=field_index: "%s[%d][lane]"
                    % (group.name, shape * n_fields + field_index),
                )
                lines.append(
                    "            const scalar_t %s%s_grad_%d_ref = %s;"
                    % (field.name, group.symbol_suffix, d, value)
                )
            lines.extend(
                _physical_gradient_lines(
                    field.name + group.symbol_suffix, dim, "            "
                )
            )
    lines.extend(
        _coefficient_evaluation_lines(
            system,
            coefficients,
            "            ",
            "q_weight[q]",
            dependencies,
        )
    )
    for row in range(n_fields):
        for d in range(dim):
            if dependencies.gradient_coefficients[row][d]:
                lines.append(
                    "            const scalar_t grad_coeff%d_%d_value = grad_coeff%d_%d;"
                    % (row, d, row, d)
                )
    test_grad_names = {}
    for test in range(dim + 1):
        for d in range(dim):
            if not any(row[d] for row in dependencies.gradient_coefficients):
                continue
            name = "test%d_grad%d" % (test, d)
            test_grad_names[(test, d)] = name
            terms = []
            for k in range(dim):
                factor = _reference_gradient_expr(reference_gradients, test, k)
                if factor == 0:
                    continue
                terms.append(_scaled_cpp_term(factor, "adj%d" % (k * dim + d)))
            lines.append(
                "            const scalar_t %s = (%s) / det;"
                % (name, _sum_cpp_terms(terms))
            )
    for test in range(dim + 1):
        for row in range(n_fields):
            terms = []
            for d in range(dim):
                if dependencies.gradient_coefficients[row][d]:
                    terms.append(
                        "grad_coeff%d_%d_value * %s"
                        % (row, d, test_grad_names[(test, d)])
                    )
            if terms:
                lines.append(
                    "            output[%d][lane] += q_weight[q] * det * (%s);"
                    % (test * n_fields + row, _sum_cpp_terms(terms))
                )
    lines.extend(["        }", "    }"])
    return lines


def _simplex_gradient_metric_body(system, rule, dependencies, specialization):
    dim = system.dim
    field = system.fields[0]
    group = _dependency_stream_group_by_name(dependencies, specialization.stream_group_name)
    field_index = specialization.field_index
    scale = _sfem_ccode(specialization.scale)
    lines = ["    for (int q = 0; q < N_QP; ++q) {"]
    lines.extend(_work_item_loop_lines("        "))
    for trial in range(rule.n_shape):
        lines.append(
            "            const scalar_t coeff_%s_%s_%d = %s[%d][lane];"
            % (
                group.name,
                field.name,
                trial,
                group.name,
                trial * len(system.fields) + field_index,
            )
        )
    for d in range(dim):
        terms = []
        for trial in range(rule.n_shape):
            factor = specialization.reference_gradient(trial, d)
            if factor == 0:
                continue
            terms.append(
                _scaled_cpp_term(
                    factor,
                    "coeff_%s_%s_%d" % (group.name, field.name, trial),
                )
            )
        value = _sum_cpp_terms(terms)
        lines.append(
            "            const scalar_t %s%s_grad_%d_ref_value = %s;"
            % (field.name, group.symbol_suffix, d, value)
        )
    lines.append("            const ptrdiff_t geometry_offset = q * geometry_stride + lane;")
    if scale == "1":
        lines.append("            const scalar_t metric_factor = q_weight[q];")
    else:
        lines.append("            const scalar_t metric_factor = q_weight[q] * (%s);" % scale)
    for left in range(dim):
        for right in range(left, dim):
            lines.append(
                "            const scalar_t geom_metric%d%d = metric_factor * geom_metric[%d][geometry_offset];"
                % (
                    left,
                    right,
                    specialization.metric_component(left, right),
                )
            )
    for left in range(dim):
        terms = []
        for right in range(dim):
            metric = "geom_metric%d%d" % (
                min(left, right),
                max(left, right),
            )
            trial_grad = "%s%s_grad_%d_ref_value" % (
                field.name,
                group.symbol_suffix,
                right,
            )
            terms.append("%s * %s" % (metric, trial_grad))
        lines.append(
            "            const scalar_t %s%s_metric_grad_%d_ref_value = %s;"
            % (
                field.name,
                group.symbol_suffix,
                left,
                _sum_cpp_terms(terms),
            )
        )
    for test in range(rule.n_shape):
        terms = []
        for left in range(dim):
            test_factor = specialization.reference_gradient(test, left)
            if test_factor == 0:
                continue
            terms.append(
                _scaled_cpp_term(
                    test_factor,
                    "%s%s_metric_grad_%d_ref_value"
                    % (
                        field.name,
                        group.symbol_suffix,
                        left,
                    ),
                )
            )
        if terms:
            lines.append(
                "            output[%d][lane] += %s;"
                % (test * len(system.fields) + field_index, _sum_cpp_terms(terms))
            )
    lines.extend(["        }", "    }"])
    return lines


def _scaled_cpp_term(factor, expression):
    factor = sp.sympify(factor)
    if factor == 1:
        return expression
    if factor == -1:
        return "-(%s)" % expression
    return "(%s) * (%s)" % (_sfem_ccode(factor), expression)


def _sum_cpp_terms(terms):
    terms = tuple(term for term in terms if term)
    if not terms:
        return "scalar_t(0)"
    expression = terms[0]
    for term in terms[1:]:
        if term.startswith("-("):
            expression += " - " + term[2:-1]
        else:
            expression += " + " + term
    return expression


def _tensor_local_body(system, prefix, coefficients, dependencies, stream_layout="pointer"):
    dim = system.dim
    n_fields = len(system.fields)
    uses_determinant = any(dependencies.value_coefficients) or dependencies.uses_adjugate
    uses_geometry_offset = uses_determinant or dependencies.uses_adjugate
    lines = []
    if not dependencies.uses_test_coefficients:
        return lines
    tensor_evaluate_name = (
        "tensor_evaluate_contiguous" if stream_layout == "contiguous" else "tensor_evaluate"
    )
    tensor_evaluate_value_name = (
        "tensor_evaluate_value_contiguous"
        if stream_layout == "contiguous"
        else "tensor_evaluate_value"
    )
    tensor_integrate_name = (
        "tensor_integrate_contiguous" if stream_layout == "contiguous" else "tensor_integrate"
    )
    tensor_integrate_value_name = (
        "tensor_integrate_value_contiguous"
        if stream_layout == "contiguous"
        else "tensor_integrate_value"
    )
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
                    "    %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>("
                    % tensor_evaluate_name,
                    "            nelems, shape_1d, grad_1d, %s, %s_value, %s_grad_ref);"
                    % (group, group, group),
                ]
            )
        else:
            lines.extend(
                [
                    "    %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>("
                    % tensor_evaluate_value_name,
                    "            nelems, shape_1d, %s, %s_value);" % (group, group),
                ]
            )
    lines.append("    scalar_t value_coeff[N_FIELDS * N_QP * VECTOR_SIZE];")
    if dependencies.uses_test_gradients:
        lines.append("    scalar_t grad_coeff_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];")
    lines.extend(
        [
            "    static constexpr int Q = integer_root(N_QP, DIM);",
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
            *_work_item_loop_lines("        "),
        ]
    )
    if uses_geometry_offset:
        lines.append("            const ptrdiff_t geometry_offset = q * geometry_stride + lane;")
    if uses_determinant:
        lines.append("            const scalar_t det = determinant[geometry_offset];")
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
                "    %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>("
                % tensor_integrate_name,
                "            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);",
            ]
        )
    else:
        lines.extend(
            [
                "    %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>("
                % tensor_integrate_value_name,
                "            nelems, shape_1d, value_coeff, output);",
            ]
        )
    return lines


def _field_evaluation_lines(system, dependencies, indent, tensor):
    if tensor:
        raise AssertionError("tensor aliases are emitted separately")
    dim = system.dim
    lines = []
    groups = _dependency_stream_groups(dependencies)
    for field_index, field in enumerate(system.fields):
        for group in groups:
            if group.uses_value:
                lines.append(
                    "%sscalar_t %s%s = scalar_t(0);"
                    % (indent, field.name, group.symbol_suffix)
                )
            if group.uses_gradient:
                for d in range(dim):
                    lines.append(
                        "%sscalar_t %s%s_grad_%d_ref = scalar_t(0);"
                        % (indent, field.name, group.symbol_suffix, d)
                    )
            lines.append("%sfor (int trial = 0; trial < N_SHAPE; ++trial) {" % indent)
            lines.append(
                "%s    const scalar_t coeff = %s[trial * N_FIELDS + %d][lane];"
                % (indent, group.name, field_index)
            )
            if group.uses_value:
                lines.append(
                    "%s    %s%s += coeff * shape[q * N_SHAPE + trial];"
                    % (indent, field.name, group.symbol_suffix)
                )
            if group.uses_gradient:
                for d in range(dim):
                    lines.append(
                        "%s    %s%s_grad_%d_ref += coeff * %s[q * N_SHAPE + trial];"
                        % (
                            indent,
                            field.name,
                            group.symbol_suffix,
                            d,
                            sfem_simplex_grad_ref_name("grad_ref", d),
                        )
                    )
            lines.append("%s}" % indent)
            if group.uses_gradient:
                lines.extend(
                    _physical_gradient_lines(
                        field.name + group.symbol_suffix, dim, indent
                    )
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
    groups = _dependency_stream_groups(dependencies)
    for field_index, field in enumerate(system.fields):
        for group in groups:
            if group.uses_value:
                lines.append(
                    "            const scalar_t %s%s = %s_value[(%d * N_QP + q) * VECTOR_SIZE + lane];"
                    % (field.name, group.symbol_suffix, group.name, field_index)
                )
            if group.uses_gradient:
                for k in range(dim):
                    lines.append(
                        "            const scalar_t %s%s_grad_%d_ref = %s_grad_ref[((%d * N_QP + q) * DIM + %d) * VECTOR_SIZE + lane];"
                        % (
                            field.name,
                            group.symbol_suffix,
                            k,
                            group.name,
                            field_index,
                            k,
                        )
                    )
                lines.extend(
                    _physical_gradient_lines(
                        field.name + group.symbol_suffix, dim, "            "
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
    temporaries = _prune_dead_cse_intermediates(temporaries, reduced)
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


def _geometry_metric_stream_initializer(name, dim):
    return "{%s}" % ", ".join(
        "%s[%d]" % (name, index)
        for index in range(symmetric_metric_component_count(dim))
    )


def _numbered_geometry_metric_stream_initializer(prefix, dim, component_order):
    entries = []
    for left in range(dim):
        for right in range(left, dim):
            entries.append(
                (
                    symmetric_metric_component_index(left, right),
                    symmetric_metric_storage_component_index(
                        dim, left, right, component_order
                    ),
                )
            )
    return ", ".join(
        "%s%d" % (prefix, storage_index)
        for _, storage_index in sorted(entries)
    )


def _indexed_geometry_metric_stream_initializer(name, dim, component_order):
    entries = []
    for left in range(dim):
        for right in range(left, dim):
            entries.append(
                (
                    symmetric_metric_component_index(left, right),
                    symmetric_metric_storage_component_index(
                        dim, left, right, component_order
                    ),
                )
            )
    return ", ".join(
        "%s[%d]" % (name, storage_index)
        for _, storage_index in sorted(entries)
    )


def _geometry_metric_grouping_lines(
    dim,
    determinant_expr,
    adjugate_expr,
    metric_target,
    indent,
    prefix,
    scalar_type="scalar_t",
):
    lines = ["%sconst %s %s_det = %s;" % (indent, scalar_type, prefix, determinant_expr)]
    for component in range(dim * dim):
        lines.append(
            "%sconst %s %s_adj%d = %s;"
            % (indent, scalar_type, prefix, component, adjugate_expr(component))
        )
    for left in range(dim):
        for right in range(left, dim):
            dot = " + ".join(
                "%s_adj%d * %s_adj%d"
                % (prefix, left * dim + d, prefix, right * dim + d)
                for d in range(dim)
            )
            lines.append(
                "%s%s = (%s) / %s_det;"
                % (
                    indent,
                    metric_target(symmetric_metric_component_index(left, right)),
                    dot,
                    prefix,
                )
            )
    return lines


def _operator_source(
    system,
    prefix,
    local_prefix,
    specialization,
    affine_specialization,
    local_name,
    residual_coeffs,
    action_coeffs,
    basis_family=None,
    geometry_family=None,
    matrix_format_plan=None,
):
    rule = specialization.quadrature_rule
    dim = system.dim
    n_fields = len(system.fields)
    n_shape = rule.n_shape
    n_qp = rule.n_qp
    vector_size = specialization.vector_size
    element = rule.element_type.lower()
    tensor_product = _is_tensor_product_family(rule, basis_family)
    lines = [
        "#include <type_traits>",
        "#include <cstdint>",
        "#include <cstdlib>",
        '#include "%s"' % local_name,
        '#include "geometry_kernels.hpp"',
        '#include "kernel_diagnostics.hpp"',
        '#include "packed_thread_scratch.hpp"',
        "",
        "#ifndef SFEM_SUCCESS",
        "#define SFEM_SUCCESS 0",
        "#endif",
        "#ifndef SFEM_FAILURE",
        "#define SFEM_FAILURE 1",
        "#endif",
        "#ifndef MIN",
        "#define MIN(a, b) ((a) < (b) ? (a) : (b))",
        "#endif",
        "#ifdef _OPENMP",
        "#include <omp.h>",
        "#endif",
        "#include <cstdio>",
        "",
    ]
    lines.extend(_affine_geometry_stream_helper_lines())
    lines.extend(
        [
            "",
            "namespace sfem {",
            "namespace codegen {",
            "",
        ]
    )
    lines.extend(
        quadrature_reference_struct_lines(
            prefix,
            "affine",
            sfem_mesh_reference_data(affine_specialization.quadrature_rule),
        )
    )
    lines.extend(
        quadrature_reference_struct_lines(
            prefix,
            "isoparametric",
            sfem_mesh_reference_data(rule),
        )
    )
    lines.extend(["", "} // namespace codegen", "} // namespace sfem", ""])
    lines.extend(_residual_diagnostics_lines(system, prefix, specialization))
    lines.append("")
    form_dependencies = {
        "residual": _codegen_dependencies(
            system,
            residual_coeffs,
            system.residual_dependencies(),
        ),
        "jacobian_action": _codegen_dependencies(
            system,
            action_coeffs,
            system.jacobian_action_dependencies(),
        ),
    }
    for form in ("residual", "jacobian_action"):
        dependencies = form_dependencies[form]
        coefficients = residual_coeffs if form == "residual" else action_coeffs
        gradient_metric = None
        function = "%s_%s_element_soa" % (prefix, form)
        block = "%s_%s_block" % (local_prefix, form)
        for scalar_type, suffix in (("double", ""), ("float", "_float")):
            params = [
                "const int nelems",
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
            call_args = ["nelems", "geometry_stride"]
            pre_call_lines = []
            if gradient_metric is not None:
                pre_call_lines.extend(
                    [
                        "    static constexpr int N_QP = %d;" % n_qp,
                        "    static constexpr int VECTOR_SIZE = %d;" % vector_size,
                        "    %s geom_metric_data[%d][N_QP * VECTOR_SIZE];"
                        % (scalar_type, gradient_metric.metric_components),
                        "    const %s *const geom_metric[%d] = %s;"
                        % (
                            scalar_type,
                            gradient_metric.metric_components,
                            _geometry_metric_stream_initializer(
                                "geom_metric_data",
                                dim,
                            ),
                        ),
                        "    for (int q = 0; q < N_QP; ++q) {",
                        *_work_item_loop_lines("        "),
                        "            const ptrdiff_t geometry_offset = q * geometry_stride + lane;",
                    ]
                )
                pre_call_lines.extend(
                    _geometry_metric_grouping_lines(
                        dim,
                        "determinant[geometry_offset]",
                        lambda component: "adjugate[%d][geometry_offset]" % component,
                        lambda component: "geom_metric_data[%d][q * VECTOR_SIZE + lane]"
                        % component,
                        "            ",
                        "metric",
                        scalar_type,
                    )
                )
                pre_call_lines.extend(["        }", "    }"])
                call_args.append("geom_metric")
            else:
                call_args.append("determinant")
            if dependencies.uses_adjugate and gradient_metric is None:
                call_args.append("adjugate")
            if tensor_product:
                call_args.append(
                    quadrature_reference_accessor(
                        prefix, "isoparametric", "shape_1d", scalar_type
                    )
                )
                if dependencies.uses_reference_gradients:
                    call_args.append(
                        quadrature_reference_accessor(
                            prefix, "isoparametric", "grad_1d", scalar_type
                        )
                    )
                call_args.append(
                    quadrature_reference_accessor(
                        prefix, "isoparametric", "q_weight_1d", scalar_type
                    )
                )
            else:
                call_args.append(
                    quadrature_reference_accessor(prefix, "isoparametric", "shape", scalar_type)
                )
                if dependencies.uses_reference_gradients:
                    call_args.extend(
                        quadrature_reference_accessor(
                            prefix,
                            "isoparametric",
                            sfem_simplex_grad_ref_name("grad_ref", d),
                            scalar_type,
                        )
                        for d in range(dim)
                    )
                call_args.append(
                    quadrature_reference_accessor(prefix, "isoparametric", "q_weight", scalar_type)
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
                    *pre_call_lines,
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
                affine_specialization,
                specialization,
                form,
                dependencies,
                coefficients,
                basis_family,
                geometry_family,
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
    lines.extend(
        _scalar_crs_matrix_assembly_source(
            system,
            prefix,
            local_prefix,
            specialization,
            form_dependencies["jacobian_action"],
            basis_family,
            geometry_family,
            matrix_format_plan,
        )
    )
    lines.extend(
        _scalar_coo_triplet_matrix_assembly_source(
            system,
            prefix,
            local_prefix,
            specialization,
            form_dependencies["jacobian_action"],
            basis_family,
            geometry_family,
            matrix_format_plan,
        )
    )
    lines.extend(
        _scalar_dia_matrix_assembly_source(
            system,
            prefix,
            local_prefix,
            specialization,
            form_dependencies["jacobian_action"],
            basis_family,
            geometry_family,
            matrix_format_plan,
        )
    )
    return "\n".join(lines)


def _mixed_operator_source(
    system,
    prefix,
    local_prefix,
    local_name,
    cell_specialization,
    compatible_element,
    field_element_types,
    residual_coeffs,
    action_coeffs,
    basis_family=None,
    geometry_family=None,
):
    rule = cell_specialization.quadrature_rule
    element = compatible_element.name.lower()
    lines = [
        "#include <type_traits>",
        '#include "%s"' % local_name,
        '#include "kernel_math.hpp"',
        '#include "geometry_kernels.hpp"',
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
        *_inline_definition_lines(),
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
    ]
    lines.extend(_affine_geometry_stream_helper_lines())
    lines.extend(
        [
            "",
            "namespace sfem {",
            "namespace codegen {",
            "",
        ]
    )
    lines.extend(_mixed_reference_data_lines(prefix, "affine", rule, system, field_element_types, basis_family))
    lines.extend(_mixed_reference_data_lines(prefix, "isoparametric", rule, system, field_element_types, basis_family))
    lines.extend(["", "} // namespace codegen", "} // namespace sfem", ""])
    lines.extend(
        _mixed_residual_diagnostics_lines(
            system,
            prefix,
            element,
            cell_specialization,
            field_element_types,
            residual_coeffs,
            action_coeffs,
            basis_family,
        )
    )
    lines.append("")
    form_data = (
        (
            "residual",
            residual_coeffs,
            _codegen_dependencies(
                system,
                residual_coeffs,
                system.residual_dependencies(),
            ),
        ),
        (
            "jacobian_action",
            action_coeffs,
            _codegen_dependencies(
                system,
                action_coeffs,
                system.jacobian_action_dependencies(),
            ),
        ),
    )
    for form, coefficients, dependencies in form_data:
        lines.extend(
            _mixed_affine_function(
                system,
                prefix,
                local_prefix,
                element,
                rule,
                "affine",
                cell_specialization.vector_size,
                field_element_types,
                form,
                dependencies,
                basis_family,
            )
        )
        lines.extend(
            _mixed_isoparametric_function(
                system,
                prefix,
                local_prefix,
                element,
                rule,
                "isoparametric",
                cell_specialization.vector_size,
                field_element_types,
                form,
                coefficients,
                dependencies,
                basis_family,
                geometry_family,
            )
        )
    lines.extend(
        _mixed_coo_triplet_matrix_assembly_source(
            system,
            prefix,
            local_prefix,
            element,
            rule,
            field_element_types,
            _codegen_dependencies(
                system,
                action_coeffs,
                system.jacobian_action_dependencies(),
            ),
            basis_family,
            geometry_family,
        )
    )
    return "\n".join(lines)


def _mixed_reference_data_lines(prefix, reference_stage, cell_rule, system, field_element_types, basis_family=None):
    return quadrature_reference_struct_lines(
        prefix,
        reference_stage,
        _mixed_reference_data(cell_rule, system, field_element_types, basis_family),
    )


def _mixed_reference_data(cell_rule, system, field_element_types, basis_family=None):
    if _is_tensor_product_family(cell_rule, basis_family):
        data = (
            SfemReferenceData("q_weight_1d", cell_rule.tensor_product_weights_1d),
        )
        for element_type in _mixed_tensor_reference_element_types(
            cell_rule,
            system,
            field_element_types,
        ):
            data += sfem_tensor_product_field_reference_data(
                element_type,
                cell_rule,
                _tensor_reference_prefix(element_type),
            )
    else:
        data = (SfemReferenceData("q_weight", cell_rule.weights),)
        for element_type in _mixed_simplex_reference_element_types(
            cell_rule,
            system,
            field_element_types,
        ):
            data += sfem_simplex_field_reference_data(
                element_type,
                cell_rule,
                _simplex_reference_prefix(element_type),
            )
    return data


def _mixed_tensor_reference_element_types(cell_rule, system, field_element_types):
    seen = set()
    element_types = []
    for element_type in (cell_rule.element_type,) + tuple(
        _field_element_type(field, cell_rule, field_element_types)
        for field in system.fields
    ):
        element_type = str(element_type).upper()
        if element_type in seen:
            continue
        seen.add(element_type)
        element_types.append(element_type)
    return tuple(element_types)


def _mixed_simplex_reference_element_types(cell_rule, system, field_element_types):
    seen = set()
    element_types = []
    for element_type in (cell_rule.element_type,) + tuple(
        _field_element_type(field, cell_rule, field_element_types)
        for field in system.fields
    ):
        element_type = str(element_type).upper()
        if element_type in seen:
            continue
        seen.add(element_type)
        element_types.append(element_type)
    return tuple(element_types)


def _tensor_reference_prefix(element_type):
    return str(element_type).lower()


def _simplex_reference_prefix(element_type):
    return str(element_type).lower()


def _mixed_tensor_cell_reference_alias_lines(prefix, reference_stage, cell_rule):
    reference_data = "%s_%s_reference_data<scalar_t>" % (prefix, reference_stage)
    reference_prefix = _tensor_reference_prefix(cell_rule.element_type)
    return [
        "    const scalar_t *const %s_shape_1d = sfem::codegen::%s::%s_shape_1d();"
        % (reference_stage, reference_data, reference_prefix),
        "    const scalar_t *const %s_grad_1d = sfem::codegen::%s::%s_grad_1d();"
        % (reference_stage, reference_data, reference_prefix),
    ]


def _mixed_simplex_cell_reference_alias_lines(prefix, reference_stage, cell_rule):
    reference_data = "%s_%s_reference_data<scalar_t>" % (prefix, reference_stage)
    reference_prefix = _simplex_reference_prefix(cell_rule.element_type)
    return [
        "    const scalar_t *const %s_cell_grad_ref_%d = sfem::codegen::%s::%s();"
        % (
            reference_stage,
            d,
            reference_data,
            sfem_simplex_grad_ref_name("%s_grad_ref" % reference_prefix, d),
        )
        for d in range(cell_rule.dim)
    ]


def _mixed_affine_function(
    system,
    prefix,
    local_prefix,
    element,
    cell_rule,
    reference_stage,
    vector_size,
    field_element_types,
    form,
    dependencies,
    basis_family=None,
):
    dim = system.dim
    layout = MixedFieldLayout.create(system, cell_rule, field_element_types)
    params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
    ]
    if dependencies.uses_adjugate:
        params.extend(
            "const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate%d" % i
            for i in range(dim * dim)
        )
    params.append("const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0")
    params.extend("const scalar_t %s" % parameter for parameter in dependencies.parameters)
    params.extend(_mixed_mesh_dependency_params(layout, dependencies))
    params.append("const ptrdiff_t out_stride")
    for field_group in layout.groups:
        if field_group.components == 1:
            params.append("scalar_t *const SFEM_RESTRICT %s_out" % field_group.name)
        else:
            params.append(
                "scalar_t *const SFEM_RESTRICT %s_out[%d]"
                % (field_group.name, field_group.components)
            )

    impl = "%s_%s_%s_affine_mesh_mixed_impl" % (prefix, element, form)
    block = "%s_%s_block" % (local_prefix, form)
    reference_data = "%s_%s_reference_data<scalar_t>" % (prefix, reference_stage)
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename scalar_t, typename jacobian_t>",
        "%s int %s(" % (_function_qualifier(), impl),
    ]
    for index, param in enumerate(params):
        lines.append("        %s%s" % (param, "," if index + 1 < len(params) else ""))
    lines.extend(
        [
            ") {",
            "    static constexpr int DIM = %d;" % dim,
            "    static constexpr int N_QP = %d;" % cell_rule.n_qp,
            "    static constexpr int CELL_N_SHAPE = %d;" % cell_rule.n_shape,
            "    static constexpr int N_SHAPE = CELL_N_SHAPE;",
            "    static constexpr int N_FIELDS = %d;" % layout.n_reference_fields,
            "    static constexpr int N_FIELD_STREAMS = %d;" % layout.total_streams,
            "    static constexpr int VECTOR_SIZE = %d;" % vector_size,
            "    (void)nnodes;",
        ]
    )
    if not dependencies.uses_test_coefficients:
        lines.extend(
            [
                "    return SFEM_SUCCESS;",
                "}",
                "",
                "} // namespace codegen",
                "} // namespace sfem",
                "",
            ]
        )
        return lines
    lines.extend(
        _mixed_reference_pointer_lines(
            reference_data,
            system,
            cell_rule,
            dependencies,
            "    ",
            field_element_types,
            basis_family,
        )
    )
    field_element_lines, field_element_arrays = _mixed_field_element_alias_lines(
        layout,
        cell_rule,
        field_element_types,
        basis_family,
        "    ",
    )
    lines.extend(field_element_lines)
    lines.extend(
        [
            "",
            _parallel_for_pragma("static"),
            "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            "        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
        ]
    )
    if dependencies.current:
        lines.append("        scalar_t block_current[N_FIELD_STREAMS][VECTOR_SIZE];")
    if dependencies.previous:
        lines.append("        scalar_t block_previous[N_FIELD_STREAMS][VECTOR_SIZE];")
    if dependencies.direction:
        lines.append("        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];")
    lines.extend(
        [
            "        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];",
        ]
    )
    field_gather = _mixed_field_gather_lines(system, layout, dependencies, "        ", field_element_arrays)
    if field_gather:
        lines.extend(["", *field_gather])
    lines.extend(["", *_zero_block_output_lines("block_output", layout.total_streams, "        ")])
    affine_geometry_streams = (
        (
            tuple("jacobian_adjugate%d" % i for i in range(dim * dim))
            if dependencies.uses_adjugate
            else ()
        )
        + ("jacobian_determinant0",)
    )
    affine_geometry_stream_indices = {
        stream: index for index, stream in enumerate(affine_geometry_streams)
    }
    lines.extend(_affine_geometry_stream_conversion_lines(affine_geometry_streams, "        "))
    if dependencies.uses_adjugate:
        lines.extend(
            [
                "        const scalar_t *block_adjugate[DIM * DIM];",
                "        for (int component = 0; component < DIM * DIM; ++component) {",
                "            block_adjugate[component] = block_affine_geometry_streams[component];",
                "        }",
            ]
        )
    block_stream_lines, block_stream_args = _mixed_block_stream_pointer_lines(
        layout,
        dependencies,
        "        ",
        cell_rule,
        field_element_types,
        basis_family,
        force_contiguous=True,
    )
    lines.extend(block_stream_lines)
    block_function = "%s_contiguous" % block if not block_stream_lines else block
    call_args = [
        "nelems",
        "0",
        "block_affine_geometry_streams[%d]"
        % affine_geometry_stream_indices["jacobian_determinant0"],
    ]
    if dependencies.uses_adjugate:
        call_args.append("block_adjugate")
    call_args.extend(_mixed_reference_call_args(cell_rule, dependencies, reference_data, basis_family))
    call_args.extend(
        block_stream_args[group.name]
        for group in _dependency_stream_groups(dependencies)
    )
    call_args.extend(map(str, dependencies.parameters))
    call_args.append(block_stream_args["output"])
    lines.extend(
        [
            "",
            "        %s<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(%s);"
            % (block_function, ", ".join(call_args)),
            "",
        ]
    )
    lines.extend(_mixed_field_atomic_scatter_lines(system, layout, "        ", field_element_arrays))
    lines.extend(
        [
            "    }",
            "    return SFEM_SUCCESS;",
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )

    function = "%s_%s_%s_affine_mesh_soa" % (prefix, element, form)
    for scalar_type, suffix in (("double", ""), ("float", "_float")):
        typed_params = [
            param.replace("jacobian_t", "geom_t").replace("scalar_t", scalar_type)
            for param in params
        ]
        lines.append('extern "C" int %s%s(' % (function, suffix))
        for index, param in enumerate(typed_params):
            lines.append("        %s%s" % (param, "," if index + 1 < len(typed_params) else ""))
        call_args = ["nelements", "nnodes", "elements"]
        if dependencies.uses_adjugate:
            call_args.extend("g_jacobian_adjugate%d" % i for i in range(dim * dim))
        call_args.append("g_jacobian_determinant0")
        call_args.extend(map(str, dependencies.parameters))
        call_args.extend(_mixed_mesh_dependency_call_args(layout, dependencies))
        call_args.append("out_stride")
        call_args.extend("%s_out" % group.name for group in layout.groups)
        lines.extend(
            [
                ") {",
                "    return sfem::codegen::%s<%s, geom_t>(%s);"
                % (impl, scalar_type, ", ".join(call_args)),
                "}",
                "",
            ]
        )
    return lines


def _mixed_isoparametric_function(
    system,
    prefix,
    local_prefix,
    element,
    cell_rule,
    reference_stage,
    vector_size,
    field_element_types,
    form,
    coefficients,
    dependencies,
    basis_family=None,
    geometry_family=None,
):
    dim = system.dim
    layout = MixedFieldLayout.create(system, cell_rule, field_element_types)
    params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
        "const geom_t *const *const SFEM_RESTRICT points",
    ]
    params.extend("const scalar_t %s" % parameter for parameter in dependencies.parameters)
    params.extend(_mixed_mesh_dependency_params(layout, dependencies))
    params.append("const ptrdiff_t out_stride")
    for field_group in layout.groups:
        if field_group.components == 1:
            params.append("scalar_t *const SFEM_RESTRICT %s_out" % field_group.name)
        else:
            params.append(
                "scalar_t *const SFEM_RESTRICT %s_out[%d]"
                % (field_group.name, field_group.components)
            )
    impl = "%s_%s_%s_isoparametric_mesh_mixed_impl" % (prefix, element, form)
    block = "%s_%s_block" % (local_prefix, form)
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename scalar_t>",
        "%s int %s(" % (_function_qualifier(), impl),
    ]
    for index, param in enumerate(params):
        lines.append("        %s%s" % (param, "," if index + 1 < len(params) else ""))
    lines.extend(
        [
            ") {",
            "    static constexpr int DIM = %d;" % dim,
            "    static constexpr int N_QP = %d;" % cell_rule.n_qp,
            "    static constexpr int CELL_N_SHAPE = %d;" % cell_rule.n_shape,
            "    static constexpr int N_SHAPE = CELL_N_SHAPE;",
            "    static constexpr int N_FIELDS = %d;" % layout.n_reference_fields,
            "    static constexpr int N_FIELD_STREAMS = %d;" % layout.total_streams,
            "    static constexpr int VECTOR_SIZE = %d;" % vector_size,
            "    (void)nnodes;",
        ]
    )
    if not dependencies.uses_test_coefficients:
        lines.extend(
            [
                "    return SFEM_SUCCESS;",
                "}",
                "",
                "} // namespace codegen",
                "} // namespace sfem",
                "",
            ]
        )
        return lines
    reference_data = "%s_%s_reference_data<scalar_t>" % (prefix, reference_stage)
    tensor_product = _is_tensor_product_family(cell_rule, basis_family)
    tensor_product_geometry = _is_tensor_product_family(cell_rule, geometry_family)
    if tensor_product:
        lines.extend(_mixed_tensor_cell_reference_alias_lines(prefix, reference_stage, cell_rule))
    else:
        lines.extend(_mixed_simplex_cell_reference_alias_lines(prefix, reference_stage, cell_rule))
    field_element_lines, field_element_arrays = _mixed_field_element_alias_lines(
        layout,
        cell_rule,
        field_element_types,
        basis_family,
        "    ",
    )
    lines.extend(field_element_lines)
    coordinate_element_lines, coordinate_element_array = (
        _coordinate_element_alias_lines(
            dim,
            cell_rule.n_shape,
            cell_rule.element_type,
            "    ",
        )
        if tensor_product_geometry
        else ([], "elements")
    )
    lines.extend(coordinate_element_lines)
    lines.extend(
        [
            _parallel_for_pragma("static"),
            "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            "        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
            "        scalar_t block_coordinates[DIM * CELL_N_SHAPE][VECTOR_SIZE];",
            "        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];",
            "        scalar_t block_determinant[N_QP * VECTOR_SIZE];",
        ]
    )
    if dependencies.current:
        lines.append("        scalar_t block_current[N_FIELD_STREAMS][VECTOR_SIZE];")
    if dependencies.previous:
        lines.append("        scalar_t block_previous[N_FIELD_STREAMS][VECTOR_SIZE];")
    if dependencies.direction:
        lines.append("        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];")
    lines.extend(
        [
            "        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];",
        ]
    )
    lines.extend(["", *_coordinate_gather_lines(dim, "        ", coordinate_element_array)])
    field_gather = _mixed_field_gather_lines(system, layout, dependencies, "        ", field_element_arrays)
    if field_gather:
        lines.extend(["", *field_gather])
    lines.extend(["", *_zero_block_output_lines("block_output", layout.total_streams, "        ")])
    if tensor_product_geometry:
        lines.append("")
        lines.extend(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim=dim,
                n_shape=cell_rule.n_shape,
                n_qp=cell_rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams="block_coordinates",
                contiguous_coordinate_streams=True,
                adjugate_target=lambda component, index: (
                    "block_adjugate_data[%d][%s]" % (component, index)
                ),
                determinant_target=lambda index: (
                    "block_determinant[%s]" % index
                ),
                adjugate_streams=tuple(
                    "block_adjugate_data[%d]" % component
                    for component in range(dim * dim)
                ),
                determinant_stream="block_determinant",
                shape_name="%s_shape_1d" % reference_stage,
                grad_name="%s_grad_1d" % reference_stage,
            )
        )
    else:
        lines.extend(
            [
                "",
                "        scalar_t *block_adjugate_streams[DIM * DIM] = {%s};"
                % ", ".join(
                    "block_adjugate_data[%d]" % component
                    for component in range(dim * dim)
                ),
                "        for (int q = 0; q < N_QP; ++q) {",
                *_work_item_loop_lines("            "),
            ]
        )
        for i in range(dim):
            for j in range(dim):
                terms = [
                    "block_coordinates[%d][lane] * %s_cell_grad_ref_%d[q * CELL_N_SHAPE + %d]"
                    % (
                        shape * dim + i,
                        reference_stage,
                        j,
                        shape,
                    )
                    for shape in range(cell_rule.n_shape)
                ]
                lines.append(
                    "                const scalar_t J%d%d = %s;"
                    % (i, j, " + ".join(terms))
                )
        lines.extend(_isoparametric_geometry_assignment_lines(dim, "                "))
        lines.extend(["            }", "        }"])
    lines.extend([""])
    lines.extend(
        _mixed_reference_pointer_lines(
            reference_data,
            system,
            cell_rule,
            dependencies,
            "        ",
            field_element_types,
            basis_family,
        )
    )
    lines.append(
        "        const scalar_t *const block_adjugate[DIM * DIM] = {%s};"
        % ", ".join("block_adjugate_data[%d]" % i for i in range(dim * dim))
    )
    block_stream_lines, block_stream_args = _mixed_block_stream_pointer_lines(
        layout,
        dependencies,
        "        ",
        cell_rule,
        field_element_types,
        basis_family,
        force_contiguous=True,
    )
    lines.extend(block_stream_lines)
    block_function = "%s_contiguous" % block if not block_stream_lines else block
    call_args = [
        "nelems",
        "VECTOR_SIZE",
        "block_determinant",
    ]
    if dependencies.uses_adjugate:
        call_args.append("block_adjugate")
    call_args.extend(_mixed_reference_call_args(cell_rule, dependencies, reference_data, basis_family))
    call_args.extend(
        block_stream_args[group.name]
        for group in _dependency_stream_groups(dependencies)
    )
    call_args.extend(map(str, dependencies.parameters))
    call_args.append(block_stream_args["output"])
    lines.extend(
        [
            "",
            "        %s<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(%s);"
            % (block_function, ", ".join(call_args)),
            "",
        ]
    )
    lines.extend(_mixed_field_atomic_scatter_lines(system, layout, "        ", field_element_arrays))
    lines.extend(
        [
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
        call_args.extend(_mixed_mesh_dependency_call_args(layout, dependencies))
        call_args.append("out_stride")
        call_args.extend("%s_out" % group.name for group in layout.groups)
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


def _mixed_stream_component_offsets(layout):
    group_component_begin = {}
    component_begin = 0
    for group in layout.groups:
        group_component_begin[group.name] = component_begin
        component_begin += group.components

    offsets = []
    for field_index, field in enumerate(layout.fields):
        group = layout.group_for_field(field_index)
        component = int(getattr(field, "component", 0))
        component_offset = group_component_begin[group.name] + component
        offsets.extend(component_offset for _ in range(layout.n_shape(field_index)))
    return tuple(offsets)


def _mixed_stream_shape_offsets(layout):
    offsets = []
    for field_index, _ in enumerate(layout.fields):
        offsets.extend(range(layout.n_shape(field_index)))
    return tuple(offsets)


def _mixed_triplet_stream_indices(layout, group_names):
    selected = set(group_names)
    indices = []
    for field_index, field in enumerate(layout.fields):
        if _residual_parent_field_name(field) not in selected:
            continue
        indices.extend(
            layout.stream_index(field_index, local_shape)
            for local_shape in range(layout.n_shape(field_index))
        )
    return tuple(indices)


def _mixed_triplet_group_names_from_prefix(prefix, layout):
    marker = "_form_2_"
    groups = tuple(group.name for group in layout.groups)
    if marker not in prefix:
        return groups, groups

    suffix = prefix.split(marker, 1)[1]
    for row in groups:
        row_prefix = "%s_" % row
        if not suffix.startswith(row_prefix):
            continue
        column = suffix[len(row_prefix) :]
        if column in groups:
            return (row,), (column,)
    return (), ()


def _mixed_coo_triplet_scatter_lines(function_base, layout, row_streams, column_streams):
    component_offsets = _mixed_stream_component_offsets(layout)
    shape_offsets = _mixed_stream_shape_offsets(layout)
    return [
        "template <typename scalar_t>",
        "static SFEM_INLINE void %s_scatter_coo_triplets(" % function_base,
        "        idx_t **const SFEM_RESTRICT elements,",
        "        const ptrdiff_t element,",
        "        const ptrdiff_t out_stride,",
        "        const scalar_t *const SFEM_RESTRICT element_matrix,",
        "        idx_t *const SFEM_RESTRICT rows,",
        "        idx_t *const SFEM_RESTRICT cols,",
        "        scalar_t *const SFEM_RESTRICT values) {",
        "    static constexpr int N_ROW_STREAMS = %d;" % len(row_streams),
        "    static constexpr int N_COL_STREAMS = %d;" % len(column_streams),
        "    static constexpr int ROW_COMPONENT[N_ROW_STREAMS] = {%s};"
        % ", ".join(str(component_offsets[stream]) for stream in row_streams),
        "    static constexpr int ROW_SHAPE[N_ROW_STREAMS] = {%s};"
        % ", ".join(str(shape_offsets[stream]) for stream in row_streams),
        "    static constexpr int COL_COMPONENT[N_COL_STREAMS] = {%s};"
        % ", ".join(str(component_offsets[stream]) for stream in column_streams),
        "    static constexpr int COL_SHAPE[N_COL_STREAMS] = {%s};"
        % ", ".join(str(shape_offsets[stream]) for stream in column_streams),
        "    const ptrdiff_t element_offset = element * N_ROW_STREAMS * N_COL_STREAMS;",
        "    for (int row_stream = 0; row_stream < N_ROW_STREAMS; ++row_stream) {",
        "        const idx_t row_node = elements[ROW_SHAPE[row_stream]][element];",
        "        const idx_t global_row = row_node * out_stride + ROW_COMPONENT[row_stream];",
        "        for (int col_stream = 0; col_stream < N_COL_STREAMS; ++col_stream) {",
        "            const idx_t col_node = elements[COL_SHAPE[col_stream]][element];",
        "            const ptrdiff_t entry = element_offset + row_stream * N_COL_STREAMS + col_stream;",
        "            rows[entry] = global_row;",
        "            cols[entry] = col_node * out_stride + COL_COMPONENT[col_stream];",
        "            values[entry] = element_matrix[row_stream * N_COL_STREAMS + col_stream];",
        "        }",
        "    }",
        "}",
        "",
    ]


def _mixed_coo_triplet_matrix_assembly_source(
    system,
    prefix,
    local_prefix,
    element,
    cell_rule,
    field_element_types,
    dependencies,
    basis_family=None,
    geometry_family=None,
):
    if not dependencies.direction:
        return []

    dim = system.dim
    layout = MixedFieldLayout.create(system, cell_rule, field_element_types)
    row_groups, column_groups = _mixed_triplet_group_names_from_prefix(prefix, layout)
    row_streams = _mixed_triplet_stream_indices(
        layout,
        row_groups,
    )
    column_streams = _mixed_triplet_stream_indices(
        layout,
        column_groups,
    )
    if not row_streams or not column_streams:
        return []

    function_base = "%s_%s_hessian_coo_triplet_isoparametric_mesh_soa" % (
        prefix,
        str(element).lower(),
    )
    impl = "%s_impl" % function_base
    block = "%s_jacobian_action_block" % local_prefix
    tensor_product = _is_tensor_product_family(cell_rule, basis_family)
    tensor_product_geometry = _is_tensor_product_family(cell_rule, geometry_family)
    field_stream_order = _mixed_tensor_product_field_stream_order(
        layout,
        cell_rule,
        field_element_types,
        basis_family,
    )
    stream_to_tensor_order = _stream_to_tensor_order(field_stream_order)
    row_tensor_streams = tuple(stream_to_tensor_order[stream] for stream in row_streams)
    column_tensor_streams = tuple(stream_to_tensor_order[stream] for stream in column_streams)
    reference_stage = "isoparametric"
    reference_data = "%s_%s_reference_data<scalar_t>" % (prefix, reference_stage)
    params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
        "const geom_t *const *const SFEM_RESTRICT points",
    ]
    params.extend("const scalar_t %s" % parameter for parameter in dependencies.parameters)
    state_dependencies = ResidualCodegenDependencies(
        current=dependencies.current,
        previous=dependencies.previous,
        direction=False,
        parameters=dependencies.parameters,
        current_value=dependencies.current_value,
        current_gradient=dependencies.current_gradient,
        previous_value=dependencies.previous_value,
        previous_gradient=dependencies.previous_gradient,
        direction_value=False,
        direction_gradient=False,
        value_coefficients=dependencies.value_coefficients,
        gradient_coefficients=dependencies.gradient_coefficients,
    )
    params.extend(_mixed_mesh_dependency_params(layout, state_dependencies))
    params.extend(
        [
            "const ptrdiff_t out_stride",
            "idx_t *const SFEM_RESTRICT rows",
            "idx_t *const SFEM_RESTRICT cols",
            "scalar_t *const SFEM_RESTRICT values",
        ]
    )

    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
    ]
    lines.extend(_mixed_coo_triplet_scatter_lines(function_base, layout, row_streams, column_streams))
    lines.extend(
        [
            "template <typename scalar_t>",
            "%s int %s(" % (_function_qualifier(), impl),
        ]
    )
    for index, param in enumerate(params):
        lines.append("        %s%s" % (param, "," if index + 1 < len(params) else ""))
    lines.extend(
        [
            ") {",
            "    static constexpr int DIM = %d;" % dim,
            "    static constexpr int N_QP = %d;" % cell_rule.n_qp,
            "    static constexpr int CELL_N_SHAPE = %d;" % cell_rule.n_shape,
            "    static constexpr int N_SHAPE = CELL_N_SHAPE;",
            "    static constexpr int N_FIELDS = %d;" % layout.n_reference_fields,
            "    static constexpr int N_FIELD_STREAMS = %d;" % layout.total_streams,
            "    static constexpr int VECTOR_SIZE = 1;",
            "    (void)nnodes;",
        ]
    )
    if tensor_product:
        lines.extend(_mixed_tensor_cell_reference_alias_lines(prefix, reference_stage, cell_rule))
    else:
        lines.extend(_mixed_simplex_cell_reference_alias_lines(prefix, reference_stage, cell_rule))
    field_element_lines, field_element_arrays = _mixed_field_element_alias_lines(
        layout,
        cell_rule,
        field_element_types,
        basis_family,
        "    ",
    )
    lines.extend(field_element_lines)
    coordinate_element_lines, coordinate_element_array = (
        _coordinate_element_alias_lines(
            dim,
            cell_rule.n_shape,
            cell_rule.element_type,
            "    ",
        )
        if tensor_product_geometry
        else ([], "elements")
    )
    lines.extend(coordinate_element_lines)
    lines.extend(
        [
            _parallel_for_pragma("static"),
            "    for (ptrdiff_t element = 0; element < nelements; ++element) {",
            "        const ptrdiff_t evbegin = element;",
            "        const int nelems = 1;",
            "        scalar_t block_coordinates[DIM * CELL_N_SHAPE][VECTOR_SIZE];",
            "        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];",
            "        scalar_t block_determinant[N_QP * VECTOR_SIZE];",
        ]
    )
    if dependencies.current:
        lines.append("        scalar_t block_current[N_FIELD_STREAMS][VECTOR_SIZE];")
    if dependencies.previous:
        lines.append("        scalar_t block_previous[N_FIELD_STREAMS][VECTOR_SIZE];")
    lines.extend(
        [
            "        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];",
            "        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];",
            "        scalar_t element_matrix[%d];" % (len(row_streams) * len(column_streams)),
            "",
        ]
    )
    lines.extend(_coordinate_gather_lines(dim, "        ", coordinate_element_array))
    field_gather = _mixed_field_gather_lines(system, layout, state_dependencies, "        ", field_element_arrays)
    if field_gather:
        lines.extend(["", *field_gather])
    if tensor_product_geometry:
        lines.append("")
        lines.extend(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim=dim,
                n_shape=cell_rule.n_shape,
                n_qp=cell_rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams="block_coordinates",
                contiguous_coordinate_streams=True,
                adjugate_target=lambda component, index: (
                    "block_adjugate_data[%d][%s]" % (component, index)
                ),
                determinant_target=lambda index: (
                    "block_determinant[%s]" % index
                ),
                adjugate_streams=tuple(
                    "block_adjugate_data[%d]" % component
                    for component in range(dim * dim)
                ),
                determinant_stream="block_determinant",
                shape_name="%s_shape_1d" % reference_stage,
                grad_name="%s_grad_1d" % reference_stage,
            )
        )
    else:
        lines.extend(
            [
                "",
                "        scalar_t *block_adjugate_streams[DIM * DIM] = {%s};"
                % ", ".join(
                    "block_adjugate_data[%d]" % component
                    for component in range(dim * dim)
                ),
                "        for (int q = 0; q < N_QP; ++q) {",
                "            const int lane = 0;",
            ]
        )
        for i in range(dim):
            for j in range(dim):
                terms = [
                    "block_coordinates[%d][lane] * %s_cell_grad_ref_%d[q * CELL_N_SHAPE + %d]"
                    % (
                        shape * dim + i,
                        reference_stage,
                        j,
                        shape,
                    )
                    for shape in range(cell_rule.n_shape)
                ]
                lines.append(
                    "            const scalar_t J%d%d = %s;"
                    % (i, j, " + ".join(terms))
                )
        lines.extend(_isoparametric_geometry_assignment_lines(dim, "            "))
        lines.extend(["        }"])
    lines.extend([""])
    lines.extend(
        _mixed_reference_pointer_lines(
            reference_data,
            system,
            cell_rule,
            dependencies,
            "        ",
            field_element_types,
            basis_family,
        )
    )
    lines.append(
        "        const scalar_t *const block_adjugate[DIM * DIM] = {%s};"
        % ", ".join("block_adjugate_data[%d]" % i for i in range(dim * dim))
    )
    block_stream_lines, block_stream_args = _mixed_block_stream_pointer_lines(
        layout,
        dependencies,
        "        ",
        cell_rule,
        field_element_types,
        basis_family,
        force_contiguous=True,
    )
    lines.extend(block_stream_lines)
    block_function = "%s_contiguous" % block if not block_stream_lines else block
    call_args = [
        "nelems",
        "0",
        "block_determinant",
    ]
    if dependencies.uses_adjugate:
        call_args.append("block_adjugate")
    call_args.extend(_mixed_reference_call_args(cell_rule, dependencies, reference_data, basis_family))
    call_args.extend(
        block_stream_args[group.name]
        for group in _dependency_stream_groups(dependencies)
    )
    call_args.extend(map(str, dependencies.parameters))
    call_args.append(block_stream_args["output"])
    lines.extend([""])
    lines.extend(_local_index_mapping_lambda_lines("row_tensor_stream", row_tensor_streams, "        "))
    lines.extend(_local_index_mapping_lambda_lines("col_tensor_stream", column_tensor_streams, "        "))
    lines.extend(
        [
            "        for (int entry = 0; entry < %d; ++entry) {"
            % (len(row_streams) * len(column_streams)),
            "            element_matrix[entry] = scalar_t(0);",
            "        }",
            "        for (int trial_local = 0; trial_local < %d; ++trial_local) {"
            % len(column_streams),
            "            const int trial = %s;"
            % _local_index_mapping_expr("col_tensor_stream", column_tensor_streams, "trial_local"),
            "            for (int stream = 0; stream < N_FIELD_STREAMS; ++stream) {",
            "                block_direction[stream][0] = scalar_t(0);",
            "                block_output[stream][0] = scalar_t(0);",
            "            }",
            "            block_direction[trial][0] = scalar_t(1);",
            "            %s<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(%s);"
            % (block_function, ", ".join(call_args)),
            "            for (int test_local = 0; test_local < %d; ++test_local) {"
            % len(row_streams),
            "                const int test = %s;"
            % _local_index_mapping_expr("row_tensor_stream", row_tensor_streams, "test_local"),
            "                element_matrix[test_local * %d + trial_local] = block_output[test][0];"
            % len(column_streams),
            "            }",
            "        }",
            "",
            "        %s_scatter_coo_triplets(elements, element, out_stride, element_matrix, rows, cols, values);"
            % function_base,
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
    for scalar_type, suffix in (("double", ""), ("float", "_float")):
        typed_params = [param.replace("scalar_t", scalar_type) for param in params]
        lines.append('extern "C" int %s%s(' % (function_base, suffix))
        for index, param in enumerate(typed_params):
            lines.append("        %s%s" % (param, "," if index + 1 < len(typed_params) else ""))
        call_args = ["nelements", "nnodes", "elements", "points"]
        call_args.extend(map(str, dependencies.parameters))
        call_args.extend(_mixed_mesh_dependency_call_args(layout, state_dependencies))
        call_args.extend(("out_stride", "rows", "cols", "values"))
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


def _field_n_shape(field, cell_rule, field_element_types):
    return _field_n_shape_by_name(
        _residual_parent_field_name(field),
        cell_rule,
        field_element_types,
    )


def _field_n_shape_by_name(field_name, cell_rule, field_element_types):
    element_type = _field_element_type(field_name, cell_rule, field_element_types)
    return sfem_field_n_shape(
        element_type,
        cell_rule.order
        if element_type in ("QUAD4", "PROTEUS_QUAD4") or sfem_is_tensor_product_hex_element(element_type)
        else None,
    )


def _field_element_type(field_or_name, cell_rule, field_element_types):
    field_name = _residual_parent_field_name(field_or_name)
    return str(field_element_types.get(field_name, cell_rule.element_type)).upper()


def _residual_parent_field_name(field_or_name):
    return str(getattr(field_or_name, "field_name", field_or_name))


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


def _mixed_residual_diagnostics_lines(
    system,
    prefix,
    element,
    specialization,
    field_element_types,
    residual_coeffs,
    action_coeffs,
    basis_family,
):
    diagnostics = (
        (
            "%s_%s_residual_element_soa" % (prefix, element),
            system.build_residual_graph("residual_diagnostics_tmp").cost,
            _codegen_dependencies(
                system,
                residual_coeffs,
                system.residual_dependencies(),
            ),
        ),
        (
            "%s_%s_jacobian_action_element_soa" % (prefix, element),
            system.build_jacobian_action_graph(
                temporary_prefix="jacobian_action_diagnostics_tmp"
            ).cost,
            _codegen_dependencies(
                system,
                action_coeffs,
                system.jacobian_action_dependencies(),
            ),
        ),
    )
    lines = []
    for public_name, cost, dependencies in diagnostics:
        if lines:
            lines.append("")
        lines.extend(
            _mixed_kernel_diagnostics_lines(
                system,
                public_name,
                cost,
                specialization,
                field_element_types,
                dependencies,
                basis_family,
            )
        )
    return lines


def _mixed_kernel_diagnostics_lines(
    system,
    public_name,
    cost,
    specialization,
    field_element_types,
    dependencies,
    basis_family,
):
    rule = specialization.quadrature_rule
    layout = MixedFieldLayout.create(system, rule, field_element_types)
    return _kernel_diagnostics_lines(
        system,
        public_name,
        cost,
        specialization,
        dependencies,
        field_streams=layout.total_streams,
        reference_data=_mixed_reference_data(rule, system, field_element_types, basis_family),
    )


def _kernel_diagnostics_lines(
    system,
    public_name,
    cost,
    specialization,
    dependencies,
    *,
    field_streams=None,
    reference_data=None,
):
    rule = specialization.quadrature_rule
    n_fields = len(system.fields)
    if field_streams is None:
        field_streams = n_fields * rule.n_shape
    geometry_streams = system.dim * system.dim + 1
    if reference_data is None:
        reference_data = sfem_reference_data(rule)
    reference_scalars = sum(
        len(reference.values)
        for reference in reference_data
        if not reference.name.startswith("q_weight")
    )
    quadrature_weight_scalars = sum(
        len(reference.values)
        for reference in reference_data
        if reference.name.startswith("q_weight")
    )
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
        "    0,",
        "    0,",
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
                (
                    public_name.replace("_element_soa", "_affine_mesh_soa"),
                    "KernelDiagnostics_print_rate_affine_mesh",
                ),
                (
                    public_name.replace(
                        "_element_soa", "_isoparametric_mesh_soa"
                    ),
                    "KernelDiagnostics_print_rate_isoparametric_mesh",
                ),
            )
        )
    for function_name_entry in function_names:
        if isinstance(function_name_entry, tuple):
            function_name, print_rate_helper = function_name_entry
        else:
            function_name = function_name_entry
            print_rate_helper = "KernelDiagnostics_print_rate"
        for scalar_type in ("double", "float"):
            lines.append("")
            lines.extend(
                _sfem_soa_diagnostic_print_wrapper_lines(
                    function_name,
                    variable_name,
                    scalar_type,
                    print_rate_helper,
                )
            )
    return lines


def _mesh_operator_source(
    system,
    prefix,
    local_prefix,
    affine_specialization,
    isoparametric_specialization,
    form,
    dependencies,
    coefficients,
    basis_family=None,
    geometry_family=None,
):
    rule = affine_specialization.quadrature_rule
    dim = system.dim
    n_fields = len(system.fields)
    n_shape = rule.n_shape
    n_qp = rule.n_qp
    vector_size = affine_specialization.vector_size
    tensor_product = _is_tensor_product_family(rule, basis_family)
    specialized_prefix = _constant_p1_affine_specialized_local_prefix(
        local_prefix,
        rule,
    )
    gradient_metric = (
        None
        if tensor_product or specialized_prefix is None
        else simplex_gradient_metric_transformation(
            system,
            rule,
            coefficients,
            dependencies,
        )
    )
    uses_cached_affine_metric = _uses_cached_affine_metric(gradient_metric)
    omit_simplex_reference_basis_inputs = gradient_metric is not None
    shape_order = (
        tuple(range(n_shape))
        if sfem_tensor_product_hex_uses_cartesian_ordering(rule.element_type)
        else tensor_product_cartesian_shape_order(dim, n_shape)
        if tensor_product
        else tuple(range(n_shape))
    )
    field_stream_order = streams_in_shape_order(
        tuple(range(n_fields * n_shape)),
        n_fields,
        shape_order,
    )
    impl = "%s_%s_affine_mesh_soa_impl" % (prefix, form)
    block_prefix = specialized_prefix if gradient_metric is not None else local_prefix
    block = "%s_%s_block" % (block_prefix, form)
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename scalar_t, typename jacobian_t>",
        "%s int %s(" % (_function_qualifier(), impl),
    ]
    params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
    ]
    if uses_cached_affine_metric:
        params.extend(
            "const jacobian_t *const SFEM_RESTRICT g_geom_metric%d" % i
            for i in range(gradient_metric.metric_components)
        )
    elif dependencies.uses_adjugate:
        params.extend(
            "const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate%d" % i
            for i in range(dim * dim)
        )
    if not uses_cached_affine_metric:
        params.append(
            "const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0"
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
    fast_path_body = _simplex_metric_scalar_affine_fast_path_body(
        system,
        rule,
        dependencies,
        gradient_metric,
    )
    if fast_path_body is not None:
        lines.extend(fast_path_body)
        lines.extend(
            [
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
        lines.extend(
            _affine_mesh_public_wrapper_lines(
                function,
                impl,
                params,
                uses_cached_affine_metric,
                gradient_metric,
                dependencies,
                system,
                dim,
            )
        )
        lines.extend(
            _simplex_metric_scalar_affine_aos_wrapper_lines(
                function,
                system,
                rule,
                dependencies,
                gradient_metric,
            )
        )
        lines.extend(
            _isoparametric_mesh_operator_source(
                system,
                prefix,
                local_prefix,
                isoparametric_specialization,
                form,
                dependencies,
                coefficients,
                basis_family,
                geometry_family,
            )
        )
        return lines
    lines.extend(
        _mesh_reference_alias_lines(
            prefix,
            rule,
            "affine",
            emit_reference_basis=not omit_simplex_reference_basis_inputs,
        )
    )
    field_shape_order = _single_field_shape_order(n_shape, n_fields, field_stream_order)
    field_element_lines, field_element_array = _single_field_element_alias_lines(
        n_shape,
        field_shape_order,
        "    ",
    )
    lines.extend(field_element_lines)
    lines.extend(
        [
            "",
            _parallel_for_pragma("static"),
            "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            "        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
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
    if gradient_metric is not None and not uses_cached_affine_metric:
        lines.append(
            "        scalar_t block_geom_metric_data[%d][VECTOR_SIZE];"
            % gradient_metric.metric_components
        )
    lines.extend(
        [
            "        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];",
        ]
    )
    lines.extend(_field_gather_lines(system, dependencies, "        ", field_element_array))
    lines.extend(["", *_zero_block_output_lines("block_output", n_fields * n_shape, "        "), ""])
    block_stream_lines, block_stream_args = _single_field_block_stream_arguments(
        dependencies,
        n_fields * n_shape,
        field_stream_order,
        "        ",
        force_contiguous=bool(field_element_lines),
    )
    lines.extend(block_stream_lines)
    block_function = "%s_contiguous" % block if not block_stream_lines else block
    affine_geometry_streams = (
        (
            tuple(
                "geom_metric%d" % i
                for i in range(gradient_metric.metric_components)
            )
            if uses_cached_affine_metric
            else tuple("jacobian_adjugate%d" % i for i in range(dim * dim))
            if dependencies.uses_adjugate
            else ()
        )
        + (() if uses_cached_affine_metric else ("jacobian_determinant0",))
    )
    affine_geometry_stream_indices = {
        stream: index for index, stream in enumerate(affine_geometry_streams)
    }
    lines.extend(_affine_geometry_stream_conversion_lines(affine_geometry_streams, "        "))
    if dependencies.uses_adjugate and not uses_cached_affine_metric:
        lines.extend(
            [
                "        const scalar_t *block_adjugate[%d];" % (dim * dim),
                "        for (int component = 0; component < %d; ++component) {"
                % (dim * dim),
                "            block_adjugate[component] = block_affine_geometry_streams[component];",
                "        }",
            ]
        )
    if uses_cached_affine_metric:
        lines.append(
            "        const scalar_t *const block_geom_metric[%d] = {%s};"
            % (
                gradient_metric.metric_components,
                _indexed_geometry_metric_stream_initializer(
                    "block_affine_geometry_streams",
                    dim,
                    getattr(
                        gradient_metric,
                        "affine_geometry_component_order",
                        "upper_column_major",
                    ),
                ),
            )
        )
        lines.append(
            "        static const scalar_t cached_affine_metric_q_weight[1] = {scalar_t(1)};"
        )
    elif gradient_metric is not None:
        lines.extend(["", *_work_item_loop_lines("        ")])
        lines.extend(
            _geometry_metric_grouping_lines(
                dim,
                "block_affine_geometry_streams[%d][lane]"
                % affine_geometry_stream_indices["jacobian_determinant0"],
                lambda component: "block_affine_geometry_streams[%d][lane]"
                % affine_geometry_stream_indices["jacobian_adjugate%d" % component],
                lambda component: "block_geom_metric_data[%d][lane]" % component,
                "            ",
                "metric",
            )
        )
        lines.append("        }")
        lines.append(
            "        const scalar_t *const block_geom_metric[%d] = %s;"
            % (
                gradient_metric.metric_components,
                _geometry_metric_stream_initializer(
                    "block_geom_metric_data",
                    dim,
                ),
            )
        )
    call_args = [
        "nelems",
        "0",
    ]
    if gradient_metric is not None:
        call_args.append("block_geom_metric")
    else:
        call_args.append(
            "block_affine_geometry_streams[%d]"
            % affine_geometry_stream_indices["jacobian_determinant0"]
        )
    if dependencies.uses_adjugate and gradient_metric is None:
        call_args.append("block_adjugate")
    if tensor_product:
        call_args.append(_mesh_reference_name("affine", "shape_1d"))
        if dependencies.uses_reference_gradients:
            call_args.append(_mesh_reference_name("affine", "grad_1d"))
        call_args.append(_mesh_reference_name("affine", "q_weight_1d"))
    else:
        if not omit_simplex_reference_basis_inputs:
            call_args.append(_mesh_reference_name("affine", "shape"))
            if dependencies.uses_reference_gradients:
                call_args.extend(
                    _mesh_reference_name(
                        "affine",
                        sfem_simplex_grad_ref_name("grad_ref", d),
                    )
                    for d in range(dim)
                )
        call_args.append(
            "cached_affine_metric_q_weight"
            if uses_cached_affine_metric
            else _mesh_reference_name("affine", "q_weight")
        )
    if dependencies.current:
        call_args.append(block_stream_args["current"])
    if dependencies.previous:
        call_args.append(block_stream_args["previous"])
    if dependencies.direction:
        call_args.append(block_stream_args["direction"])
    call_args.extend(map(str, dependencies.parameters))
    call_args.append(block_stream_args["output"])
    lines.extend(
        [
            "",
            "        %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
            % (block_function, ", ".join(call_args)),
            "",
        ]
    )
    lines.extend(_field_atomic_scatter_lines(system, "        ", field_element_array))
    lines.extend(
        [
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
            param.replace("jacobian_t", "geom_t").replace("scalar_t", scalar_type)
            for param in params
        ]
        lines.append('extern "C" int %s%s(' % (function, suffix))
        for index, param in enumerate(typed_params):
            lines.append(
                "        %s%s"
                % (param, "," if index + 1 < len(typed_params) else "")
            )
        call_args = ["nelements", "nnodes", "elements"]
        if uses_cached_affine_metric:
            call_args.extend(
                "g_geom_metric%d" % i
                for i in range(gradient_metric.metric_components)
            )
        elif dependencies.uses_adjugate:
            call_args.extend(
                "g_jacobian_adjugate%d" % i for i in range(dim * dim)
            )
        if not uses_cached_affine_metric:
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
                "    return sfem::codegen::%s<%s, geom_t>(%s);"
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
            isoparametric_specialization,
            form,
            dependencies,
            coefficients,
            basis_family,
            geometry_family,
        )
    )
    if form == "jacobian_action":
        lines.extend(
            _scalar_packed_jacobian_action_source(
                system,
                prefix,
                local_prefix,
                isoparametric_specialization,
                dependencies,
                coefficients,
                basis_family,
                geometry_family,
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


def _matrix_format_values(matrix_format_plan):
    if matrix_format_plan is None or getattr(matrix_format_plan, "is_empty", True):
        return ()
    values = []
    for matrix_format in matrix_format_plan.formats:
        value = getattr(matrix_format, "value", str(matrix_format)).lower()
        if value not in values:
            values.append(value)
    return tuple(values)


def _packed_crs_passes(matrix_format_plan):
    if matrix_format_plan is None or getattr(matrix_format_plan, "is_empty", True):
        return ()
    passes = []
    for variant in matrix_format_plan.variants:
        matrix_format = getattr(variant.matrix_format, "value", str(variant.matrix_format)).lower()
        mesh_layout = getattr(variant.mesh_layout, "value", str(variant.mesh_layout)).lower()
        if matrix_format != "crs" or mesh_layout != "packed":
            continue
        packed_pass = getattr(variant.packed_pass, "value", str(variant.packed_pass)).lower()
        if packed_pass and packed_pass != "none" and packed_pass not in passes:
            passes.append(packed_pass)
    return tuple(passes)


def _compatible_matrix_field_indices_from_prefix(prefix, system, element_type):
    fields = tuple(system.fields)
    names = tuple(field.name for field in fields)
    marker = "_form_2_"
    if marker not in prefix:
        all_indices = tuple(range(len(fields)))
        return all_indices, all_indices

    suffix = prefix.split(marker, 1)[1]
    element_suffix = "_%s" % str(element_type).lower()
    if suffix.endswith(element_suffix):
        suffix = suffix[: -len(element_suffix)]

    for row_index, row_name in sorted(
        enumerate(names),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        row_prefix = "%s_" % row_name
        if not suffix.startswith(row_prefix):
            continue
        column_name = suffix[len(row_prefix) :]
        if column_name in names:
            return (row_index,), (names.index(column_name),)

    all_indices = tuple(range(len(fields)))
    return all_indices, all_indices


def _compatible_matrix_stream_indices(field_indices, n_shape):
    streams = []
    for field_index in field_indices:
        streams.extend(field_index * n_shape + shape for shape in range(n_shape))
    return tuple(streams)


def _compatible_stream_component_offsets(n_fields, n_shape):
    offsets = []
    for field_index in range(n_fields):
        offsets.extend(field_index for _ in range(n_shape))
    return tuple(offsets)


def _compatible_stream_shape_offsets(n_fields, n_shape):
    offsets = []
    for _ in range(n_fields):
        offsets.extend(range(n_shape))
    return tuple(offsets)


def _crs_find_cols_lines(function_base, n_shape):
    lines = [
        "static SFEM_INLINE void %s_find_cols(" % function_base,
        "        const idx_t *const SFEM_RESTRICT targets,",
        "        const idx_t *const SFEM_RESTRICT row,",
        "        const int lenrow,",
        "        idx_t *const SFEM_RESTRICT ks) {",
    ]
    if n_shape <= 10:
        lines.append("#pragma unroll(%d)" % n_shape)
    lines.extend(
        [
            "    for (int d = 0; d < %d; ++d) {" % n_shape,
            "        ks[d] = 0;",
            "    }",
            "    for (int k = 0; k < lenrow; ++k) {",
        ]
    )
    if n_shape <= 10:
        lines.append("#pragma unroll(%d)" % n_shape)
    lines.extend(
        [
            "        for (int d = 0; d < %d; ++d) {" % n_shape,
            "            ks[d] += row[k] < targets[d];",
            "        }",
            "    }",
            "}",
            "",
        ]
    )
    return lines


def _scalar_crs_matrix_scatter_lines(function_base, n_shape):
    return _crs_find_cols_lines(function_base, n_shape) + [
        "template <typename scalar_t>",
        "static SFEM_INLINE int %s_scatter_crs(" % function_base,
        "        const idx_t *const SFEM_RESTRICT ev,",
        "        const scalar_t *const SFEM_RESTRICT element_matrix,",
        "        const count_t *const SFEM_RESTRICT rowptr,",
        "        const idx_t *const SFEM_RESTRICT colidx,",
        "        scalar_t *const SFEM_RESTRICT values) {",
        "    static constexpr int N_SHAPE = %d;" % n_shape,
        "    count_t entries[N_SHAPE * N_SHAPE];",
        "    idx_t ks[N_SHAPE];",
        "    bool valid_graph = true;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        const count_t row_begin = rowptr[ev[i]];",
        "        const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);",
        "        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];",
        "        %s_find_cols(ev, cols, lenrow, ks);" % function_base,
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            if (ks[j] < 0 || ks[j] >= lenrow || cols[ks[j]] != ev[j]) {",
        "                if (valid_graph) {",
        "                    std::fprintf(stderr, \"%s_scatter_crs missing graph entry (%%ld, %%ld)\\n\", (long)ev[i], (long)ev[j]);"
        % function_base,
        "                }",
        "                entries[i * N_SHAPE + j] = row_begin;",
        "                valid_graph = false;",
        "            } else {",
        "                entries[i * N_SHAPE + j] = row_begin + ks[j];",
        "            }",
        "        }",
        "    }",
        "    if (!valid_graph) return SFEM_FAILURE;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "#pragma omp atomic update",
        "            values[entries[i * N_SHAPE + j]] += element_matrix[i * N_SHAPE + j];",
        "        }",
        "    }",
        "    return SFEM_SUCCESS;",
        "}",
        "",
    ]


def _compatible_crs_matrix_scatter_lines(function_base, n_shape, n_fields, row_streams, column_streams):
    component_offsets = _compatible_stream_component_offsets(n_fields, n_shape)
    shape_offsets = _compatible_stream_shape_offsets(n_fields, n_shape)
    return _crs_find_cols_lines(function_base, n_shape) + [
        "template <typename scalar_t>",
        "static SFEM_INLINE int %s_scatter_crs(" % function_base,
        "        const idx_t *const SFEM_RESTRICT ev,",
        "        const scalar_t *const SFEM_RESTRICT element_matrix,",
        "        const count_t *const SFEM_RESTRICT rowptr,",
        "        const idx_t *const SFEM_RESTRICT colidx,",
        "        scalar_t *const SFEM_RESTRICT values) {",
        "    static constexpr int N_SHAPE = %d;" % n_shape,
        "    static constexpr int N_FIELDS = %d;" % n_fields,
        "    static constexpr int N_ROW_STREAMS = %d;" % len(row_streams),
        "    static constexpr int N_COL_STREAMS = %d;" % len(column_streams),
        "    static constexpr int ROW_COMPONENT[%d] = {%s};"
        % (
            len(row_streams),
            ", ".join(str(component_offsets[stream]) for stream in row_streams),
        ),
        "    static constexpr int ROW_SHAPE[%d] = {%s};"
        % (
            len(row_streams),
            ", ".join(str(shape_offsets[stream]) for stream in row_streams),
        ),
        "    static constexpr int COL_COMPONENT[%d] = {%s};"
        % (
            len(column_streams),
            ", ".join(str(component_offsets[stream]) for stream in column_streams),
        ),
        "    static constexpr int COL_SHAPE[%d] = {%s};"
        % (
            len(column_streams),
            ", ".join(str(shape_offsets[stream]) for stream in column_streams),
        ),
        "    count_t entries[N_SHAPE * N_SHAPE];",
        "    idx_t ks[N_SHAPE];",
        "    bool valid_graph = true;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        const count_t row_begin = rowptr[ev[i]];",
        "        const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);",
        "        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];",
        "        %s_find_cols(ev, cols, lenrow, ks);" % function_base,
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            if (ks[j] < 0 || ks[j] >= lenrow || cols[ks[j]] != ev[j]) {",
        "                if (valid_graph) {",
        "                    std::fprintf(stderr, \"%s_scatter_crs missing graph entry (%%ld, %%ld)\\n\", (long)ev[i], (long)ev[j]);"
        % function_base,
        "                }",
        "                entries[i * N_SHAPE + j] = row_begin;",
        "                valid_graph = false;",
        "            } else {",
        "                entries[i * N_SHAPE + j] = row_begin + ks[j];",
        "            }",
        "        }",
        "    }",
        "    if (!valid_graph) return SFEM_FAILURE;",
        "    for (int row_stream = 0; row_stream < N_ROW_STREAMS; ++row_stream) {",
        "        const int row_shape = ROW_SHAPE[row_stream];",
        "        const int bi = ROW_COMPONENT[row_stream];",
        "        for (int col_stream = 0; col_stream < N_COL_STREAMS; ++col_stream) {",
        "            const int col_shape = COL_SHAPE[col_stream];",
        "            const int bj = COL_COMPONENT[col_stream];",
        "            scalar_t *const block = &values[entries[row_shape * N_SHAPE + col_shape] * N_FIELDS * N_FIELDS];",
        "#pragma omp atomic update",
        "            block[bi * N_FIELDS + bj] += element_matrix[row_stream * N_COL_STREAMS + col_stream];",
        "        }",
        "    }",
        "    return SFEM_SUCCESS;",
        "}",
        "",
    ]


def _scalar_crs_packed_matrix_helpers(function_base, n_shape, n_fields, row_streams, column_streams):
    lines = [
        "static SFEM_INLINE idx_t %s_packed_global_node(" % function_base,
        "        const uint16_t packed_node,",
        "        const ptrdiff_t pack,",
        "        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,",
        "        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,",
        "        const idx_t *const SFEM_RESTRICT ghost_idx) {",
        "    const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];",
        "    return packed_node < n_contiguous ? idx_t(owned_nodes_ptr[pack] + packed_node) : ghost_idx[ghost_ptr[pack] + packed_node - n_contiguous];",
        "}",
        "",
        "template <typename scalar_t>",
        "static SFEM_INLINE int %s_discover_packed_crs_entries(" % function_base,
        "        const idx_t *const SFEM_RESTRICT ev,",
        "        const count_t *const SFEM_RESTRICT rowptr,",
        "        const idx_t *const SFEM_RESTRICT colidx,",
        "        count_t *const SFEM_RESTRICT entries) {",
        "    static constexpr int N_SHAPE = %d;" % n_shape,
        "    idx_t ks[N_SHAPE];",
        "    bool valid_graph = true;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        const count_t row_begin = rowptr[ev[i]];",
        "        const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);",
        "        const idx_t *const SFEM_RESTRICT cols = &colidx[row_begin];",
        "        %s_find_cols(ev, cols, lenrow, ks);" % function_base,
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            if (ks[j] < 0 || ks[j] >= lenrow || cols[ks[j]] != ev[j]) {",
        "                if (valid_graph) {",
        "                    std::fprintf(stderr, \"%s_discover_packed_crs_entries missing graph entry (%%ld, %%ld)\\n\", (long)ev[i], (long)ev[j]);"
        % function_base,
        "                }",
        "                entries[i * N_SHAPE + j] = row_begin;",
        "                valid_graph = false;",
        "            } else {",
        "                entries[i * N_SHAPE + j] = row_begin + ks[j];",
        "            }",
        "        }",
        "    }",
        "    return valid_graph ? SFEM_SUCCESS : SFEM_FAILURE;",
        "}",
        "",
    ]
    if n_fields == 1:
        lines.extend(
            [
                "template <typename scalar_t>",
                "static SFEM_INLINE void %s_scatter_packed_crs_entries(" % function_base,
                "        const scalar_t *const SFEM_RESTRICT element_matrix,",
                "        const count_t *const SFEM_RESTRICT entries,",
                "        scalar_t *const SFEM_RESTRICT values) {",
                "    static constexpr int N_SHAPE = %d;" % n_shape,
                "    for (int i = 0; i < N_SHAPE; ++i) {",
                "        for (int j = 0; j < N_SHAPE; ++j) {",
                "#pragma omp atomic update",
                "            values[entries[i * N_SHAPE + j]] += element_matrix[i * N_SHAPE + j];",
                "        }",
                "    }",
                "}",
                "",
            ]
        )
        return lines

    component_offsets = _compatible_stream_component_offsets(n_fields, n_shape)
    shape_offsets = _compatible_stream_shape_offsets(n_fields, n_shape)
    lines.extend(
        [
            "template <typename scalar_t>",
            "static SFEM_INLINE void %s_scatter_packed_crs_entries(" % function_base,
            "        const scalar_t *const SFEM_RESTRICT element_matrix,",
            "        const count_t *const SFEM_RESTRICT entries,",
            "        scalar_t *const SFEM_RESTRICT values) {",
            "    static constexpr int N_SHAPE = %d;" % n_shape,
            "    static constexpr int N_FIELDS = %d;" % n_fields,
            "    static constexpr int N_ROW_STREAMS = %d;" % len(row_streams),
            "    static constexpr int N_COL_STREAMS = %d;" % len(column_streams),
            "    static constexpr int ROW_COMPONENT[%d] = {%s};"
            % (
                len(row_streams),
                ", ".join(str(component_offsets[stream]) for stream in row_streams),
            ),
            "    static constexpr int ROW_SHAPE[%d] = {%s};"
            % (
                len(row_streams),
                ", ".join(str(shape_offsets[stream]) for stream in row_streams),
            ),
            "    static constexpr int COL_COMPONENT[%d] = {%s};"
            % (
                len(column_streams),
                ", ".join(str(component_offsets[stream]) for stream in column_streams),
            ),
            "    static constexpr int COL_SHAPE[%d] = {%s};"
            % (
                len(column_streams),
                ", ".join(str(shape_offsets[stream]) for stream in column_streams),
            ),
            "    for (int row_stream = 0; row_stream < N_ROW_STREAMS; ++row_stream) {",
            "        const int row_shape = ROW_SHAPE[row_stream];",
            "        const int bi = ROW_COMPONENT[row_stream];",
            "        for (int col_stream = 0; col_stream < N_COL_STREAMS; ++col_stream) {",
            "            const int col_shape = COL_SHAPE[col_stream];",
            "            const int bj = COL_COMPONENT[col_stream];",
            "            scalar_t *const block = &values[entries[row_shape * N_SHAPE + col_shape] * N_FIELDS * N_FIELDS];",
            "#pragma omp atomic update",
            "            block[bi * N_FIELDS + bj] += element_matrix[row_stream * N_COL_STREAMS + col_stream];",
            "        }",
            "    }",
            "}",
            "",
        ]
    )
    return lines


def _scalar_crs_matrix_assembly_source(
    system,
    prefix,
    local_prefix,
    specialization,
    dependencies,
    basis_family,
    geometry_family,
    matrix_format_plan,
):
    matrix_formats = _matrix_format_values(matrix_format_plan)
    if not {"crs", "bsr"}.intersection(matrix_formats):
        return []
    if not dependencies.direction:
        return []
    packed_crs_passes = _packed_crs_passes(matrix_format_plan)

    rule = specialization.quadrature_rule
    dim = system.dim
    n_fields = len(system.fields)
    n_shape = rule.n_shape
    n_streams = n_fields * n_shape
    n_qp = rule.n_qp
    tensor_product = _is_tensor_product_family(rule, basis_family)
    tensor_product_geometry = _is_tensor_product_family(rule, geometry_family)
    row_fields, column_fields = _compatible_matrix_field_indices_from_prefix(
        prefix,
        system,
        rule.element_type,
    )
    row_streams = _compatible_matrix_stream_indices(row_fields, n_shape)
    column_streams = _compatible_matrix_stream_indices(column_fields, n_shape)
    shape_order = (
        tuple(range(n_shape))
        if sfem_tensor_product_hex_uses_cartesian_ordering(rule.element_type)
        else tensor_product_cartesian_shape_order(dim, n_shape)
        if tensor_product
        else tuple(range(n_shape))
    )
    field_stream_order = streams_in_shape_order(tuple(range(n_shape)), 1, shape_order)
    if n_fields != 1:
        field_stream_order = streams_in_shape_order(
            tuple(range(n_streams)),
            n_fields,
            shape_order,
        )
    field_shape_order = _single_field_shape_order(n_shape, n_fields, field_stream_order)
    field_streams_in_tensor_order = _stream_to_tensor_order(field_stream_order)
    row_tensor_streams = tuple(field_streams_in_tensor_order[stream] for stream in row_streams)
    column_tensor_streams = tuple(field_streams_in_tensor_order[stream] for stream in column_streams)
    field_element_lines, field_element_array = _single_field_element_alias_lines(
        n_shape,
        field_shape_order,
        "    ",
    )
    coordinate_element_lines, coordinate_element_array = (
        _coordinate_element_alias_lines(
            dim,
            n_shape,
            rule.element_type,
            "    ",
        )
        if tensor_product_geometry
        else ([], "elements")
    )
    function_base = "%s_hessian_crs_isoparametric_mesh_soa" % prefix
    impl = "%s_impl" % function_base
    block = "%s_jacobian_action_block" % local_prefix
    params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
        "const geom_t *const *const SFEM_RESTRICT points",
    ]
    params.extend(
        "const scalar_t %s" % parameter for parameter in dependencies.parameters
    )
    state_dependencies = ResidualCodegenDependencies(
        current=dependencies.current,
        previous=dependencies.previous,
        direction=False,
        parameters=dependencies.parameters,
        current_value=dependencies.current_value,
        current_gradient=dependencies.current_gradient,
        previous_value=dependencies.previous_value,
        previous_gradient=dependencies.previous_gradient,
        direction_value=False,
        direction_gradient=False,
        value_coefficients=dependencies.value_coefficients,
        gradient_coefficients=dependencies.gradient_coefficients,
    )
    if state_dependencies.current:
        params.append("const ptrdiff_t current_stride")
        params.extend(
            "const scalar_t *const SFEM_RESTRICT %s" % field.name
            for field in system.fields
        )
    if state_dependencies.previous:
        params.append("const ptrdiff_t previous_stride")
        params.extend(
            "const scalar_t *const SFEM_RESTRICT %s_old" % field.name
            for field in system.fields
        )
    params.extend(
        [
            "const count_t *const SFEM_RESTRICT rowptr",
            "const idx_t *const SFEM_RESTRICT colidx",
            "scalar_t *const SFEM_RESTRICT values",
        ]
    )

    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
    ]
    if n_fields == 1:
        lines.extend(_scalar_crs_matrix_scatter_lines(function_base, n_shape))
    else:
        lines.extend(
            _compatible_crs_matrix_scatter_lines(
                function_base,
                n_shape,
                n_fields,
                row_streams,
                column_streams,
            )
        )
    if packed_crs_passes:
        lines.extend(
            _scalar_crs_packed_matrix_helpers(
                function_base,
                n_shape,
                n_fields,
                row_streams,
                column_streams,
            )
        )
    lines.extend(
        [
            "template <typename scalar_t>",
            "%s int %s(" % (_function_qualifier(), impl),
        ]
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
            "    static constexpr int N_STREAMS = N_FIELDS * N_SHAPE;",
            "    static constexpr int VECTOR_SIZE = 1;",
            "    (void)nnodes;",
        ]
    )
    lines.extend(_mesh_reference_alias_lines(prefix, rule, "isoparametric"))
    lines.extend(field_element_lines)
    lines.extend(coordinate_element_lines)
    lines.extend(
        [
            "",
            "    int invalid_matrix_graph = 0;",
            "#pragma omp parallel for schedule(static) reduction(|:invalid_matrix_graph)",
            "    for (ptrdiff_t element = 0; element < nelements; ++element) {",
            "        const ptrdiff_t evbegin = element;",
            "        const int nelems = 1;",
            "        idx_t ev[N_SHAPE];",
            "        scalar_t element_matrix[%d];" % (len(row_streams) * len(column_streams)),
            "        scalar_t block_coordinates[DIM * N_SHAPE][VECTOR_SIZE];",
            "        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];",
            "        scalar_t block_determinant[N_QP * VECTOR_SIZE];",
        ]
    )
    if state_dependencies.current:
        lines.append("        scalar_t block_current[N_STREAMS][VECTOR_SIZE];")
    if state_dependencies.previous:
        lines.append("        scalar_t block_previous[N_STREAMS][VECTOR_SIZE];")
    lines.extend(
        [
            "        scalar_t block_direction[N_STREAMS][VECTOR_SIZE];",
            "        scalar_t block_output[N_STREAMS][VECTOR_SIZE];",
            "        const geom_t *const coordinate_components[DIM] = {%s};"
            % ", ".join("points[%d]" % d for d in range(dim)),
            "",
            "        for (int shape = 0; shape < N_SHAPE; ++shape) {",
            "            const idx_t node = elements[shape][element];",
            "            const idx_t coordinate_node = %s[shape][element];" % coordinate_element_array,
            "            ev[shape] = node;",
            "            for (int d = 0; d < DIM; ++d) {",
            "                block_coordinates[shape * DIM + d][0] = scalar_t(coordinate_components[d][coordinate_node]);",
            "            }",
        ]
    )
    if state_dependencies.current:
        if field_element_lines:
            lines.append("            const idx_t field_node = %s[shape][element];" % field_element_array)
            for field_index, field in enumerate(system.fields):
                lines.append(
                    "            block_current[shape * N_FIELDS + %d][0] = %s[field_node * current_stride];"
                    % (field_index, field.name)
                )
        else:
            for field_index, field in enumerate(system.fields):
                lines.append(
                    "            block_current[%d * N_SHAPE + shape][0] = %s[node * current_stride];"
                    % (field_index, field.name)
                )
    if state_dependencies.previous:
        if field_element_lines and not state_dependencies.current:
            lines.append("            const idx_t field_node = %s[shape][element];" % field_element_array)
        if field_element_lines:
            for field_index, field in enumerate(system.fields):
                lines.append(
                    "            block_previous[shape * N_FIELDS + %d][0] = %s_old[field_node * previous_stride];"
                    % (field_index, field.name)
                )
        else:
            for field_index, field in enumerate(system.fields):
                lines.append(
                    "            block_previous[%d * N_SHAPE + shape][0] = %s_old[node * previous_stride];"
                    % (field_index, field.name)
                )
    lines.extend(
        [
            "        }",
            "",
        ]
    )
    if tensor_product_geometry:
        lines.extend(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim=dim,
                n_shape=n_shape,
                n_qp=rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams="block_coordinates",
                contiguous_coordinate_streams=True,
                adjugate_target=lambda component, index: (
                    "block_adjugate_data[%d][%s]" % (component, index)
                ),
                determinant_target=lambda index: (
                    "block_determinant[%s]" % index
                ),
                adjugate_streams=tuple(
                    "block_adjugate_data[%d]" % component
                    for component in range(dim * dim)
                ),
                determinant_stream="block_determinant",
                shape_name=_mesh_reference_name("isoparametric", "shape_1d"),
                grad_name=_mesh_reference_name("isoparametric", "grad_1d"),
            )
        )
    else:
        lines.extend(
            [
                "        scalar_t *block_adjugate_streams[DIM * DIM] = {%s};"
                % ", ".join(
                    "block_adjugate_data[%d]" % component
                    for component in range(dim * dim)
                ),
                "        for (int q = 0; q < N_QP; ++q) {",
                "            const int lane = 0;",
            ]
        )
        for i in range(dim):
            for j in range(dim):
                terms = [
                    "block_coordinates[%d][lane] * %s[q * N_SHAPE + %d]"
                    % (
                        shape * dim + i,
                        _mesh_reference_name(
                            "isoparametric",
                            sfem_simplex_grad_ref_name("grad_ref", j),
                        ),
                        shape,
                    )
                    for shape in range(n_shape)
                ]
                lines.append(
                    "            const scalar_t J%d%d = %s;"
                    % (i, j, " + ".join(terms))
                )
        lines.extend(_isoparametric_geometry_assignment_lines(dim, "            "))
        lines.extend(["        }"])
    if field_element_lines:
        state_stream_args = {}
        if state_dependencies.current:
            state_stream_args["current"] = "block_current"
        if state_dependencies.previous:
            state_stream_args["previous"] = "block_previous"
        direction_arg = "block_direction"
        output_arg = "block_output"
        block_function = "%s_contiguous" % block
    else:
        state_stream_args = {}
        if state_dependencies.current:
            current_lines, current_arg = _block_stream_argument(
                "const scalar_t *",
                "block_current_streams",
                "block_current",
                n_streams,
                field_stream_order,
                "        ",
                mutable=False,
            )
            lines.extend(current_lines)
            state_stream_args["current"] = current_arg
        if state_dependencies.previous:
            previous_lines, previous_arg = _block_stream_argument(
                "const scalar_t *",
                "block_previous_streams",
                "block_previous",
                n_streams,
                field_stream_order,
                "        ",
                mutable=False,
            )
            lines.extend(previous_lines)
            state_stream_args["previous"] = previous_arg
        direction_lines, direction_arg = _block_stream_argument(
            "const scalar_t *",
            "block_direction_streams",
            "block_direction",
            n_streams,
            field_stream_order,
            "        ",
            mutable=False,
        )
        output_lines, output_arg = _block_stream_argument(
            "scalar_t *",
            "block_output_streams",
            "block_output",
            n_streams,
            field_stream_order,
            "        ",
            mutable=True,
        )
        lines.extend(direction_lines)
        lines.extend(output_lines)
        block_function = (
            block
            if direction_lines or output_lines
            else "%s_contiguous" % block
        )
    lines.append(
        "        const scalar_t *const block_adjugate[DIM * DIM] = {%s};"
        % ", ".join("block_adjugate_data[%d]" % i for i in range(dim * dim))
    )
    call_args = [
        "1",
        "1",
        "block_determinant",
    ]
    if dependencies.uses_adjugate:
        call_args.append("block_adjugate")
    if tensor_product:
        call_args.append(_mesh_reference_name("isoparametric", "shape_1d"))
        if dependencies.uses_reference_gradients:
            call_args.append(_mesh_reference_name("isoparametric", "grad_1d"))
        call_args.append(_mesh_reference_name("isoparametric", "q_weight_1d"))
    else:
        call_args.append(_mesh_reference_name("isoparametric", "shape"))
        if dependencies.uses_reference_gradients:
            call_args.extend(
                _mesh_reference_name(
                    "isoparametric",
                    sfem_simplex_grad_ref_name("grad_ref", d),
                )
                for d in range(dim)
            )
        call_args.append(_mesh_reference_name("isoparametric", "q_weight"))
    if state_dependencies.current:
        call_args.append(state_stream_args["current"])
    if state_dependencies.previous:
        call_args.append(state_stream_args["previous"])
    call_args.append(direction_arg)
    call_args.extend(map(str, dependencies.parameters))
    call_args.append(output_arg)
    lines.extend([""])
    lines.extend(_local_index_mapping_lambda_lines("row_tensor_stream", row_tensor_streams, "        "))
    lines.extend(_local_index_mapping_lambda_lines("col_tensor_stream", column_tensor_streams, "        "))
    lines.extend(
        [
            "        for (int entry = 0; entry < %d; ++entry) {"
            % (len(row_streams) * len(column_streams)),
            "            element_matrix[entry] = scalar_t(0);",
            "        }",
            "        for (int trial_local = 0; trial_local < %d; ++trial_local) {"
            % len(column_streams),
            "            const int trial = %s;"
            % _local_index_mapping_expr("col_tensor_stream", column_tensor_streams, "trial_local"),
            "            for (int stream = 0; stream < N_STREAMS; ++stream) {",
            "                block_direction[stream][0] = scalar_t(0);",
            "                block_output[stream][0] = scalar_t(0);",
            "            }",
            "            block_direction[trial][0] = scalar_t(1);",
            "            %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
            % (block_function, ", ".join(call_args)),
            "            for (int test_local = 0; test_local < %d; ++test_local) {"
            % len(row_streams),
            "                const int test = %s;"
            % _local_index_mapping_expr("row_tensor_stream", row_tensor_streams, "test_local"),
            "                element_matrix[test_local * %d + trial_local] = block_output[test][0];"
            % len(column_streams),
            "            }",
            "        }",
            "",
            "        invalid_matrix_graph |= (%s_scatter_crs(ev, element_matrix, rowptr, colidx, values) != SFEM_SUCCESS);"
            % function_base,
            "    }",
            "",
            "    return invalid_matrix_graph ? SFEM_FAILURE : SFEM_SUCCESS;",
            "}",
            "",
        ]
    )
    if packed_crs_passes:
        packed_fill_impl = "%s_packed_fill_impl" % function_base
        packed_discover_impl = "%s_packed_discover_impl" % function_base
        packed_params = [
            "const ptrdiff_t n_packs",
            "const ptrdiff_t n_elements_per_pack",
            "const ptrdiff_t nelements",
            "const ptrdiff_t nnodes",
            "const ptrdiff_t max_nodes_per_pack",
            "uint16_t **const SFEM_RESTRICT elements",
            "const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr",
            "const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes",
            "const ptrdiff_t *const SFEM_RESTRICT ghost_ptr",
            "const idx_t *const SFEM_RESTRICT ghost_idx",
            "const geom_t *const *const SFEM_RESTRICT points",
        ]
        packed_params.extend(
            "const scalar_t %s" % parameter for parameter in dependencies.parameters
        )
        if state_dependencies.current:
            packed_params.append("const ptrdiff_t current_stride")
            packed_params.extend(
                "const scalar_t *const SFEM_RESTRICT %s" % field.name
                for field in system.fields
            )
        if state_dependencies.previous:
            packed_params.append("const ptrdiff_t previous_stride")
            packed_params.extend(
                "const scalar_t *const SFEM_RESTRICT %s_old" % field.name
                for field in system.fields
            )
        packed_fill_params = tuple(
            packed_params
            + [
                "const count_t *const SFEM_RESTRICT packed_element_entries",
                "scalar_t *const SFEM_RESTRICT values",
            ]
        )
        packed_discover_params = tuple(
            [
                "const ptrdiff_t n_packs",
                "const ptrdiff_t n_elements_per_pack",
                "const ptrdiff_t nelements",
                "const ptrdiff_t nnodes",
                "const ptrdiff_t max_nodes_per_pack",
                "uint16_t **const SFEM_RESTRICT elements",
                "const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr",
                "const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes",
                "const ptrdiff_t *const SFEM_RESTRICT ghost_ptr",
                "const idx_t *const SFEM_RESTRICT ghost_idx",
                "const count_t *const SFEM_RESTRICT rowptr",
                "const idx_t *const SFEM_RESTRICT colidx",
                "count_t *const SFEM_RESTRICT packed_element_entries",
            ]
        )
        lines.extend(
            [
                "template <typename scalar_t>",
                "%s int %s(" % (_function_qualifier(), packed_discover_impl),
            ]
        )
        for index, param in enumerate(packed_discover_params):
            lines.append("        %s%s" % (param, "," if index + 1 < len(packed_discover_params) else ""))
        lines.extend(
            [
                ") {",
                "    static constexpr int N_SHAPE = %d;" % n_shape,
                "    (void)nnodes;",
                "    (void)max_nodes_per_pack;",
                "    (void)n_shared_nodes;",
                "    int invalid_matrix_graph = 0;",
                "#pragma omp parallel for schedule(static) reduction(|:invalid_matrix_graph)",
                "    for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {",
                "        const ptrdiff_t e_start = pack * n_elements_per_pack;",
                "        const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);",
                "        for (ptrdiff_t element = e_start; element < e_end; ++element) {",
                "            idx_t ev[N_SHAPE];",
                "            for (int shape = 0; shape < N_SHAPE; ++shape) {",
                "                ev[shape] = %s_packed_global_node(elements[shape][element], pack, owned_nodes_ptr, ghost_ptr, ghost_idx);" % function_base,
                "            }",
                "            count_t *const entries = &packed_element_entries[element * N_SHAPE * N_SHAPE];",
                "            invalid_matrix_graph |= (%s_discover_packed_crs_entries<scalar_t>(ev, rowptr, colidx, entries) != SFEM_SUCCESS);" % function_base,
                "        }",
                "    }",
                "    return invalid_matrix_graph ? SFEM_FAILURE : SFEM_SUCCESS;",
                "}",
                "",
                "template <typename scalar_t>",
                "%s int %s(" % (_function_qualifier(), packed_fill_impl),
            ]
        )
        for index, param in enumerate(packed_fill_params):
            lines.append("        %s%s" % (param, "," if index + 1 < len(packed_fill_params) else ""))
        lines.extend(
            [
                ") {",
                "    static constexpr int DIM = %d;" % dim,
                "    static constexpr int N_QP = %d;" % n_qp,
                "    static constexpr int N_SHAPE = %d;" % n_shape,
                "    static constexpr int N_FIELDS = %d;" % n_fields,
                "    static constexpr int N_STREAMS = N_FIELDS * N_SHAPE;",
                "    static constexpr int VECTOR_SIZE = 1;",
                "    (void)nnodes;",
                "    (void)n_shared_nodes;",
            ]
        )
        lines.extend(_mesh_reference_alias_lines(prefix, rule, "isoparametric"))
        packed_coordinate_element_lines, packed_coordinate_element_array = (
            _coordinate_element_alias_lines(
                dim,
                n_shape,
                rule.element_type,
                "    ",
                pointer_type="uint16_t",
            )
            if tensor_product_geometry
            else ([], "elements")
        )
        lines.extend(packed_coordinate_element_lines)
        packed_field_element_lines, packed_field_element_array = _single_field_element_alias_lines(
            n_shape,
            field_shape_order,
            "    ",
            pointer_type="uint16_t",
        )
        lines.extend(packed_field_element_lines)
        lines.extend(
            [
                "",
                "#pragma omp parallel",
                "    {",
                "        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);",
            ]
        )
        if state_dependencies.current:
            lines.append(
                "        scalar_t *const SFEM_RESTRICT pack_current = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)N_FIELDS * (size_t)max_nodes_per_pack);"
            )
        if state_dependencies.previous:
            lines.append(
                "        scalar_t *const SFEM_RESTRICT pack_previous = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)N_FIELDS * (size_t)max_nodes_per_pack);"
            )
        lines.extend(
            [
                "",
                "#pragma omp for schedule(static)",
                "        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {",
                "            const ptrdiff_t e_start = pack * n_elements_per_pack;",
                "            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);",
                "            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];",
                "            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];",
                "            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];",
                "            const geom_t *const coordinate_components[DIM] = {%s};"
                % ", ".join("points[%d]" % d for d in range(dim)),
                "            for (int d = 0; d < DIM; ++d) {",
                "                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;",
                "                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];",
                "                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
                "                    pack_coordinate[k] = scalar_t(coordinate_component[owned_nodes_ptr[pack] + k]);",
                "                }",
                "                for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                "                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[ghosts[k]]);",
                "                }",
                "            }",
            ]
        )
        if state_dependencies.current:
            for field_index, field in enumerate(system.fields):
                lines.extend(
                    [
                        "            {",
                        "                scalar_t *const SFEM_RESTRICT pack_field = pack_current + %d * max_nodes_per_pack;" % field_index,
                        "                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
                        "                    pack_field[k] = %s[(owned_nodes_ptr[pack] + k) * current_stride];" % field.name,
                        "                }",
                        "                for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                        "                    pack_field[n_contiguous + k] = %s[ghosts[k] * current_stride];" % field.name,
                        "                }",
                        "            }",
                    ]
                )
        if state_dependencies.previous:
            for field_index, field in enumerate(system.fields):
                lines.extend(
                    [
                        "            {",
                        "                scalar_t *const SFEM_RESTRICT pack_field = pack_previous + %d * max_nodes_per_pack;" % field_index,
                        "                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
                        "                    pack_field[k] = %s_old[(owned_nodes_ptr[pack] + k) * previous_stride];" % field.name,
                        "                }",
                        "                for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                        "                    pack_field[n_contiguous + k] = %s_old[ghosts[k] * previous_stride];" % field.name,
                        "                }",
                        "            }",
                    ]
                )
        lines.extend(
            [
                "",
                "            for (ptrdiff_t element = e_start; element < e_end; ++element) {",
                "                const int nelems = 1;",
                "                scalar_t element_matrix[%d];" % (len(row_streams) * len(column_streams)),
                "                scalar_t block_coordinates[DIM * N_SHAPE][VECTOR_SIZE];",
                "                scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];",
                "                scalar_t block_determinant[N_QP * VECTOR_SIZE];",
            ]
        )
        if state_dependencies.current:
            lines.append("                scalar_t block_current[N_STREAMS][VECTOR_SIZE];")
        if state_dependencies.previous:
            lines.append("                scalar_t block_previous[N_STREAMS][VECTOR_SIZE];")
        lines.extend(
            [
                "                scalar_t block_direction[N_STREAMS][VECTOR_SIZE];",
                "                scalar_t block_output[N_STREAMS][VECTOR_SIZE];",
                "",
                "                for (int shape = 0; shape < N_SHAPE; ++shape) {",
                "                    const uint16_t packed_node = elements[shape][element];",
                "                    const uint16_t coordinate_packed_node = %s[shape][element];" % packed_coordinate_element_array,
                "                    for (int d = 0; d < DIM; ++d) {",
                "                        block_coordinates[shape * DIM + d][0] = pack_coordinates[d * max_nodes_per_pack + coordinate_packed_node];",
                "                }",
            ]
        )
        if state_dependencies.current:
            if packed_field_element_lines:
                lines.append("                    const uint16_t field_packed_node = %s[shape][element];" % packed_field_element_array)
                for field_index, field in enumerate(system.fields):
                    lines.append(
                        "                    block_current[shape * N_FIELDS + %d][0] = pack_current[%d * max_nodes_per_pack + field_packed_node];"
                        % (field_index, field_index)
                    )
            else:
                for field_index, field in enumerate(system.fields):
                    lines.append(
                        "                    block_current[%d * N_SHAPE + shape][0] = pack_current[%d * max_nodes_per_pack + packed_node];"
                        % (field_index, field_index)
                    )
        if state_dependencies.previous:
            if packed_field_element_lines and not state_dependencies.current:
                lines.append("                    const uint16_t field_packed_node = %s[shape][element];" % packed_field_element_array)
            if packed_field_element_lines:
                for field_index, field in enumerate(system.fields):
                    lines.append(
                        "                    block_previous[shape * N_FIELDS + %d][0] = pack_previous[%d * max_nodes_per_pack + field_packed_node];"
                        % (field_index, field_index)
                    )
            else:
                for field_index, field in enumerate(system.fields):
                    lines.append(
                        "                    block_previous[%d * N_SHAPE + shape][0] = pack_previous[%d * max_nodes_per_pack + packed_node];"
                        % (field_index, field_index)
                    )
        lines.extend(["                }", ""])
        if tensor_product_geometry:
            lines.extend(
                tensor_product_gradient_isoparametric_geometry_lines(
                    dim=dim,
                    n_shape=n_shape,
                    n_qp=rule.n_qp,
                    local_prefix=local_prefix,
                    coordinate_streams="block_coordinates",
                    contiguous_coordinate_streams=True,
                    adjugate_target=lambda component, index: (
                        "block_adjugate_data[%d][%s]" % (component, index)
                    ),
                    determinant_target=lambda index: (
                        "block_determinant[%s]" % index
                    ),
                    adjugate_streams=tuple(
                        "block_adjugate_data[%d]" % component
                        for component in range(dim * dim)
                    ),
                    determinant_stream="block_determinant",
                    shape_name=_mesh_reference_name("isoparametric", "shape_1d"),
                    grad_name=_mesh_reference_name("isoparametric", "grad_1d"),
                )
            )
        else:
            lines.extend(
                [
                    "            scalar_t *block_adjugate_streams[DIM * DIM] = {%s};"
                    % ", ".join(
                        "block_adjugate_data[%d]" % component
                        for component in range(dim * dim)
                    ),
                    "            for (int q = 0; q < N_QP; ++q) {",
                    "                const int lane = 0;",
                ]
            )
            for i in range(dim):
                for j in range(dim):
                    terms = [
                        "block_coordinates[%d][lane] * %s[q * N_SHAPE + %d]"
                        % (
                            shape * dim + i,
                            _mesh_reference_name(
                                "isoparametric",
                                sfem_simplex_grad_ref_name("grad_ref", j),
                            ),
                            shape,
                        )
                        for shape in range(n_shape)
                    ]
                    lines.append(
                        "                const scalar_t J%d%d = %s;"
                        % (i, j, " + ".join(terms))
                    )
            lines.extend(_isoparametric_geometry_assignment_lines(dim, "                "))
            lines.extend(["            }"])
        if packed_field_element_lines:
            if state_dependencies.current:
                state_stream_args["current"] = "block_current"
            if state_dependencies.previous:
                state_stream_args["previous"] = "block_previous"
            direction_arg = "block_direction"
            output_arg = "block_output"
        else:
            if state_dependencies.current:
                current_lines, current_arg = _block_stream_argument(
                    "const scalar_t *",
                    "block_current_streams",
                    "block_current",
                    n_streams,
                    field_stream_order,
                    "            ",
                    mutable=False,
                )
                lines.extend(current_lines)
                state_stream_args["current"] = current_arg
            if state_dependencies.previous:
                previous_lines, previous_arg = _block_stream_argument(
                    "const scalar_t *",
                    "block_previous_streams",
                    "block_previous",
                    n_streams,
                    field_stream_order,
                    "            ",
                    mutable=False,
                )
                lines.extend(previous_lines)
                state_stream_args["previous"] = previous_arg
            direction_lines, direction_arg = _block_stream_argument(
                "const scalar_t *",
                "block_direction_streams",
                "block_direction",
                n_streams,
                field_stream_order,
                "            ",
                mutable=False,
            )
            output_lines, output_arg = _block_stream_argument(
                "scalar_t *",
                "block_output_streams",
                "block_output",
                n_streams,
                field_stream_order,
                "            ",
                mutable=True,
            )
            lines.extend(direction_lines)
            lines.extend(output_lines)
        lines.append(
            "            const scalar_t *const block_adjugate[DIM * DIM] = {%s};"
            % ", ".join("block_adjugate_data[%d]" % i for i in range(dim * dim))
        )
        packed_call_args = list(call_args)
        if state_dependencies.current:
            current_index = packed_call_args.index(state_stream_args["current"])
            packed_call_args[current_index] = state_stream_args["current"]
        if state_dependencies.previous:
            previous_index = packed_call_args.index(state_stream_args["previous"])
            packed_call_args[previous_index] = state_stream_args["previous"]
        packed_call_args[packed_call_args.index(direction_arg)] = direction_arg
        packed_call_args[-1] = output_arg
        lines.extend([""])
        lines.extend(_local_index_mapping_lambda_lines("row_tensor_stream", row_tensor_streams, "            "))
        lines.extend(_local_index_mapping_lambda_lines("col_tensor_stream", column_tensor_streams, "            "))
        lines.extend(
            [
                "            for (int entry = 0; entry < %d; ++entry) {"
                % (len(row_streams) * len(column_streams)),
                "                element_matrix[entry] = scalar_t(0);",
                "            }",
                "            for (int trial_local = 0; trial_local < %d; ++trial_local) {"
                % len(column_streams),
                "                const int trial = %s;"
                % _local_index_mapping_expr("col_tensor_stream", column_tensor_streams, "trial_local"),
                "                for (int stream = 0; stream < N_STREAMS; ++stream) {",
                "                    block_direction[stream][0] = scalar_t(0);",
                "                    block_output[stream][0] = scalar_t(0);",
                "                }",
                "                block_direction[trial][0] = scalar_t(1);",
                "                %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
                % (block_function, ", ".join(packed_call_args)),
                "                for (int test_local = 0; test_local < %d; ++test_local) {"
                % len(row_streams),
                "                    const int test = %s;"
                % _local_index_mapping_expr("row_tensor_stream", row_tensor_streams, "test_local"),
                "                    element_matrix[test_local * %d + trial_local] = block_output[test][0];"
                % len(column_streams),
                "                }",
                "            }",
                "",
                "            const count_t *const entries = &packed_element_entries[element * N_SHAPE * N_SHAPE];",
                "            %s_scatter_packed_crs_entries(element_matrix, entries, values);" % function_base,
                "            }",
                "        }",
            ]
        )
        lines.extend(
            [
                "    }",
                "    return SFEM_SUCCESS;",
                "}",
                "",
            ]
        )
    lines.extend(
        [
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )
    for scalar_type, suffix in (("double", ""), ("float", "_float")):
        typed_params = [
            param.replace("scalar_t", scalar_type) for param in params
        ]
        if "crs" in matrix_formats:
            lines.append('extern "C" int %s%s(' % (function_base, suffix))
            for index, param in enumerate(typed_params):
                lines.append(
                    "        %s%s" % (param, "," if index + 1 < len(typed_params) else "")
                )
            call_args = ["nelements", "nnodes", "elements", "points"]
            call_args.extend(map(str, dependencies.parameters))
            if state_dependencies.current:
                call_args.append("current_stride")
                call_args.extend(field.name for field in system.fields)
            if state_dependencies.previous:
                call_args.append("previous_stride")
                call_args.extend("%s_old" % field.name for field in system.fields)
            call_args.extend(("rowptr", "colidx", "values"))
            lines.extend(
                [
                    ") {",
                    "    return sfem::codegen::%s<%s>(%s);"
                    % (impl, scalar_type, ", ".join(call_args)),
                    "}",
                    "",
                ]
            )
        if "bsr" in matrix_formats:
            bsr_name = function_base.replace("_hessian_crs_", "_hessian_bsr_")
            lines.append('extern "C" int %s%s(' % (bsr_name, suffix))
            for index, param in enumerate(typed_params):
                lines.append(
                    "        %s%s" % (param, "," if index + 1 < len(typed_params) else "")
                )
            call_args = ["nelements", "nnodes", "elements", "points"]
            call_args.extend(map(str, dependencies.parameters))
            if state_dependencies.current:
                call_args.append("current_stride")
                call_args.extend(field.name for field in system.fields)
            if state_dependencies.previous:
                call_args.append("previous_stride")
                call_args.extend("%s_old" % field.name for field in system.fields)
            call_args.extend(("rowptr", "colidx", "values"))
            lines.extend(
                [
                    ") {",
                    "    return sfem::codegen::%s<%s>(%s);"
                    % (impl, scalar_type, ", ".join(call_args)),
                    "}",
                    "",
                ]
            )
        if "crs" in matrix_formats and "one_pass" in packed_crs_passes:
            one_pass_name = function_base.replace(
                "_hessian_crs_",
                "_hessian_crs_packed_one_pass_",
            )
            typed_packed_fill_params = [
                param.replace("scalar_t", scalar_type) for param in packed_fill_params
            ]
            lines.append('extern "C" int %s%s(' % (one_pass_name, suffix))
            for index, param in enumerate(typed_packed_fill_params):
                lines.append(
                    "        %s%s" % (param, "," if index + 1 < len(typed_packed_fill_params) else "")
                )
            call_args = [
                "n_packs",
                "n_elements_per_pack",
                "nelements",
                "nnodes",
                "max_nodes_per_pack",
                "elements",
                "owned_nodes_ptr",
                "n_shared_nodes",
                "ghost_ptr",
                "ghost_idx",
                "points",
            ]
            call_args.extend(map(str, dependencies.parameters))
            if state_dependencies.current:
                call_args.append("current_stride")
                call_args.extend(field.name for field in system.fields)
            if state_dependencies.previous:
                call_args.append("previous_stride")
                call_args.extend("%s_old" % field.name for field in system.fields)
            call_args.extend(("packed_element_entries", "values"))
            lines.extend(
                [
                    ") {",
                    "    return sfem::codegen::%s<%s>(%s);"
                    % (packed_fill_impl, scalar_type, ", ".join(call_args)),
                    "}",
                    "",
                ]
            )
        if "crs" in matrix_formats and "two_pass" in packed_crs_passes:
            two_pass_name = function_base.replace(
                "_hessian_crs_",
                "_hessian_crs_packed_two_pass_",
            )
            typed_two_pass_params = [
                param.replace("scalar_t", scalar_type)
                for param in (
                    packed_params
                    + [
                        "const count_t *const SFEM_RESTRICT rowptr",
                        "const idx_t *const SFEM_RESTRICT colidx",
                        "count_t *const SFEM_RESTRICT packed_element_entries",
                        "scalar_t *const SFEM_RESTRICT values",
                    ]
                )
            ]
            lines.append('extern "C" int %s%s(' % (two_pass_name, suffix))
            for index, param in enumerate(typed_two_pass_params):
                lines.append(
                    "        %s%s" % (param, "," if index + 1 < len(typed_two_pass_params) else "")
                )
            common_args = [
                "n_packs",
                "n_elements_per_pack",
                "nelements",
                "nnodes",
                "max_nodes_per_pack",
                "elements",
                "owned_nodes_ptr",
                "n_shared_nodes",
                "ghost_ptr",
                "ghost_idx",
            ]
            fill_args = common_args + ["points"]
            fill_args.extend(map(str, dependencies.parameters))
            if state_dependencies.current:
                fill_args.append("current_stride")
                fill_args.extend(field.name for field in system.fields)
            if state_dependencies.previous:
                fill_args.append("previous_stride")
                fill_args.extend("%s_old" % field.name for field in system.fields)
            fill_args.extend(("packed_element_entries", "values"))
            lines.extend(
                [
                    ") {",
                    "    const int graph_status = sfem::codegen::%s<%s>(%s);"
                    % (
                        packed_discover_impl,
                        scalar_type,
                        ", ".join(common_args + ["rowptr", "colidx", "packed_element_entries"]),
                    ),
                    "    if (graph_status != SFEM_SUCCESS) return graph_status;",
                    "    return sfem::codegen::%s<%s>(%s);"
                    % (packed_fill_impl, scalar_type, ", ".join(fill_args)),
                    "}",
                    "",
                ]
            )
    return lines


def _scalar_coo_triplet_matrix_scatter_lines(function_base, n_shape, n_fields=1):
    if n_fields == 1:
        return [
            "template <typename scalar_t>",
            "static SFEM_INLINE void %s_scatter_coo_triplets(" % function_base,
            "        const idx_t *const SFEM_RESTRICT ev,",
            "        const scalar_t *const SFEM_RESTRICT element_matrix,",
            "        const ptrdiff_t element,",
            "        idx_t *const SFEM_RESTRICT rows,",
            "        idx_t *const SFEM_RESTRICT cols,",
            "        scalar_t *const SFEM_RESTRICT values) {",
            "    static constexpr int N_SHAPE = %d;" % n_shape,
            "    const ptrdiff_t element_offset = element * N_SHAPE * N_SHAPE;",
            "    for (int i = 0; i < N_SHAPE; ++i) {",
            "        const idx_t global_row = ev[i];",
            "        for (int j = 0; j < N_SHAPE; ++j) {",
            "            const ptrdiff_t entry = element_offset + i * N_SHAPE + j;",
            "            rows[entry] = global_row;",
            "            cols[entry] = ev[j];",
            "            values[entry] = element_matrix[i * N_SHAPE + j];",
            "        }",
            "    }",
            "}",
            "",
        ]
    return [
        "template <typename scalar_t>",
        "static SFEM_INLINE void %s_scatter_coo_triplets(" % function_base,
        "        const idx_t *const SFEM_RESTRICT ev,",
        "        const ptrdiff_t out_stride,",
        "        const scalar_t *const SFEM_RESTRICT element_matrix,",
        "        const ptrdiff_t element,",
        "        idx_t *const SFEM_RESTRICT rows,",
        "        idx_t *const SFEM_RESTRICT cols,",
        "        scalar_t *const SFEM_RESTRICT values) {",
        "    static constexpr int N_SHAPE = %d;" % n_shape,
        "    static constexpr int N_FIELDS = %d;" % n_fields,
        "    static constexpr int N_STREAMS = N_FIELDS * N_SHAPE;",
        "    const ptrdiff_t element_offset = element * N_STREAMS * N_STREAMS;",
        "    for (int row_field = 0; row_field < N_FIELDS; ++row_field) {",
        "        for (int row_shape = 0; row_shape < N_SHAPE; ++row_shape) {",
        "            const int row_stream = row_field * N_SHAPE + row_shape;",
        "            const idx_t global_row = ev[row_shape] * out_stride + row_field;",
        "            for (int col_field = 0; col_field < N_FIELDS; ++col_field) {",
        "                for (int col_shape = 0; col_shape < N_SHAPE; ++col_shape) {",
        "                    const int col_stream = col_field * N_SHAPE + col_shape;",
        "                    const ptrdiff_t entry = element_offset + row_stream * N_STREAMS + col_stream;",
        "                    rows[entry] = global_row;",
        "                    cols[entry] = ev[col_shape] * out_stride + col_field;",
        "                    values[entry] = element_matrix[row_stream * N_STREAMS + col_stream];",
        "                }",
        "            }",
        "        }",
        "    }",
        "}",
        "",
    ]


def _scalar_coo_triplet_matrix_assembly_source(
    system,
    prefix,
    local_prefix,
    specialization,
    dependencies,
    basis_family,
    geometry_family,
    matrix_format_plan,
):
    matrix_formats = _matrix_format_values(matrix_format_plan)
    if "coo" not in matrix_formats:
        return []
    if not dependencies.direction:
        return []

    rule = specialization.quadrature_rule
    dim = system.dim
    n_fields = len(system.fields)
    n_shape = rule.n_shape
    n_streams = n_fields * n_shape
    n_qp = rule.n_qp
    tensor_product = _is_tensor_product_family(rule, basis_family)
    tensor_product_geometry = _is_tensor_product_family(rule, geometry_family)
    shape_order = (
        tuple(range(n_shape))
        if sfem_tensor_product_hex_uses_cartesian_ordering(rule.element_type)
        else tensor_product_cartesian_shape_order(dim, n_shape)
        if tensor_product
        else tuple(range(n_shape))
    )
    field_stream_order = streams_in_shape_order(tuple(range(n_shape)), 1, shape_order)
    if n_fields != 1:
        field_stream_order = streams_in_shape_order(
            tuple(range(n_streams)),
            n_fields,
            shape_order,
        )
    field_shape_order = _single_field_shape_order(n_shape, n_fields, field_stream_order)
    field_streams_in_tensor_order = _stream_to_tensor_order(field_stream_order)
    function_base = "%s_hessian_coo_triplet_isoparametric_mesh_soa" % prefix
    impl = "%s_impl" % function_base
    block = "%s_jacobian_action_block" % local_prefix
    params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
        "const geom_t *const *const SFEM_RESTRICT points",
    ]
    params.extend(
        "const scalar_t %s" % parameter for parameter in dependencies.parameters
    )
    state_dependencies = ResidualCodegenDependencies(
        current=dependencies.current,
        previous=dependencies.previous,
        direction=False,
        parameters=dependencies.parameters,
        current_value=dependencies.current_value,
        current_gradient=dependencies.current_gradient,
        previous_value=dependencies.previous_value,
        previous_gradient=dependencies.previous_gradient,
        direction_value=False,
        direction_gradient=False,
        value_coefficients=dependencies.value_coefficients,
        gradient_coefficients=dependencies.gradient_coefficients,
    )
    if state_dependencies.current:
        params.append("const ptrdiff_t current_stride")
        params.extend(
            "const scalar_t *const SFEM_RESTRICT %s" % field.name
            for field in system.fields
        )
    if state_dependencies.previous:
        params.append("const ptrdiff_t previous_stride")
        params.extend(
            "const scalar_t *const SFEM_RESTRICT %s_old" % field.name
            for field in system.fields
        )
    if n_fields != 1:
        params.append("const ptrdiff_t out_stride")
    params.extend(
        [
            "idx_t *const SFEM_RESTRICT rows",
            "idx_t *const SFEM_RESTRICT cols",
            "scalar_t *const SFEM_RESTRICT values",
        ]
    )

    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
    ]
    lines.extend(_scalar_coo_triplet_matrix_scatter_lines(function_base, n_shape, n_fields))
    lines.extend(
        [
            "template <typename scalar_t>",
            "%s int %s(" % (_function_qualifier(), impl),
        ]
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
            "    static constexpr int N_STREAMS = N_FIELDS * N_SHAPE;",
            "    static constexpr int VECTOR_SIZE = 1;",
            "    (void)nnodes;",
        ]
    )
    lines.extend(_mesh_reference_alias_lines(prefix, rule, "isoparametric"))
    field_element_lines, field_element_array = _single_field_element_alias_lines(
        n_shape,
        field_shape_order,
        "    ",
    )
    lines.extend(field_element_lines)
    coordinate_element_lines, coordinate_element_array = (
        _coordinate_element_alias_lines(
            dim,
            n_shape,
            rule.element_type,
            "    ",
        )
        if tensor_product_geometry
        else ([], "elements")
    )
    lines.extend(coordinate_element_lines)
    lines.extend(
        [
            "",
            "#pragma omp parallel for schedule(static)",
            "    for (ptrdiff_t element = 0; element < nelements; ++element) {",
            "        const ptrdiff_t evbegin = element;",
            "        const int nelems = 1;",
            "        idx_t ev[N_SHAPE];",
            "        scalar_t element_matrix[N_STREAMS * N_STREAMS];",
            "        scalar_t block_coordinates[DIM * N_SHAPE][VECTOR_SIZE];",
            "        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];",
            "        scalar_t block_determinant[N_QP * VECTOR_SIZE];",
        ]
    )
    if state_dependencies.current:
        lines.append("        scalar_t block_current[N_STREAMS][VECTOR_SIZE];")
    if state_dependencies.previous:
        lines.append("        scalar_t block_previous[N_STREAMS][VECTOR_SIZE];")
    lines.extend(
        [
            "        scalar_t block_direction[N_STREAMS][VECTOR_SIZE];",
            "        scalar_t block_output[N_STREAMS][VECTOR_SIZE];",
            "        const geom_t *const coordinate_components[DIM] = {%s};"
            % ", ".join("points[%d]" % d for d in range(dim)),
            "",
            "        for (int shape = 0; shape < N_SHAPE; ++shape) {",
            "            const idx_t node = elements[shape][element];",
            "            const idx_t coordinate_node = %s[shape][element];" % coordinate_element_array,
            "            ev[shape] = node;",
            "            for (int d = 0; d < DIM; ++d) {",
            "                block_coordinates[shape * DIM + d][0] = scalar_t(coordinate_components[d][coordinate_node]);",
            "            }",
        ]
    )
    if state_dependencies.current:
        if field_element_lines:
            lines.append("            const idx_t field_node = %s[shape][element];" % field_element_array)
            for field_index, field in enumerate(system.fields):
                lines.append(
                    "            block_current[shape * N_FIELDS + %d][0] = %s[field_node * current_stride];"
                    % (field_index, field.name)
                )
        else:
            for field_index, field in enumerate(system.fields):
                lines.append(
                    "            block_current[%d * N_SHAPE + shape][0] = %s[node * current_stride];"
                    % (field_index, field.name)
                )
    if state_dependencies.previous:
        if field_element_lines and not state_dependencies.current:
            lines.append("            const idx_t field_node = %s[shape][element];" % field_element_array)
        if field_element_lines:
            for field_index, field in enumerate(system.fields):
                lines.append(
                    "            block_previous[shape * N_FIELDS + %d][0] = %s_old[field_node * previous_stride];"
                    % (field_index, field.name)
                )
        else:
            for field_index, field in enumerate(system.fields):
                lines.append(
                    "            block_previous[%d * N_SHAPE + shape][0] = %s_old[node * previous_stride];"
                    % (field_index, field.name)
                )
    lines.extend(
        [
            "        }",
            "",
        ]
    )
    if tensor_product_geometry:
        lines.extend(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim=dim,
                n_shape=n_shape,
                n_qp=rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams="block_coordinates",
                contiguous_coordinate_streams=True,
                adjugate_target=lambda component, index: (
                    "block_adjugate_data[%d][%s]" % (component, index)
                ),
                determinant_target=lambda index: (
                    "block_determinant[%s]" % index
                ),
                adjugate_streams=tuple(
                    "block_adjugate_data[%d]" % component
                    for component in range(dim * dim)
                ),
                determinant_stream="block_determinant",
                shape_name=_mesh_reference_name("isoparametric", "shape_1d"),
                grad_name=_mesh_reference_name("isoparametric", "grad_1d"),
            )
        )
    else:
        lines.extend(
            [
                "        scalar_t *block_adjugate_streams[DIM * DIM] = {%s};"
                % ", ".join(
                    "block_adjugate_data[%d]" % component
                    for component in range(dim * dim)
                ),
                "        for (int q = 0; q < N_QP; ++q) {",
                "            const int lane = 0;",
            ]
        )
        for i in range(dim):
            for j in range(dim):
                terms = [
                    "block_coordinates[%d][lane] * %s[q * N_SHAPE + %d]"
                    % (
                        shape * dim + i,
                        _mesh_reference_name(
                            "isoparametric",
                            sfem_simplex_grad_ref_name("grad_ref", j),
                        ),
                        shape,
                    )
                    for shape in range(n_shape)
                ]
                lines.append(
                    "            const scalar_t J%d%d = %s;"
                    % (i, j, " + ".join(terms))
                )
        lines.extend(_isoparametric_geometry_assignment_lines(dim, "            "))
        lines.extend(["        }"])
    if field_element_lines:
        state_stream_args = {}
        if state_dependencies.current:
            state_stream_args["current"] = "block_current"
        if state_dependencies.previous:
            state_stream_args["previous"] = "block_previous"
        direction_arg = "block_direction"
        output_arg = "block_output"
        block_function = "%s_contiguous" % block
    else:
        state_stream_args = {}
        if state_dependencies.current:
            current_lines, current_arg = _block_stream_argument(
                "const scalar_t *",
                "block_current_streams",
                "block_current",
                n_streams,
                field_stream_order,
                "        ",
                mutable=False,
            )
            lines.extend(current_lines)
            state_stream_args["current"] = current_arg
        if state_dependencies.previous:
            previous_lines, previous_arg = _block_stream_argument(
                "const scalar_t *",
                "block_previous_streams",
                "block_previous",
                n_streams,
                field_stream_order,
                "        ",
                mutable=False,
            )
            lines.extend(previous_lines)
            state_stream_args["previous"] = previous_arg
        direction_lines, direction_arg = _block_stream_argument(
            "const scalar_t *",
            "block_direction_streams",
            "block_direction",
            n_streams,
            field_stream_order,
            "        ",
            mutable=False,
        )
        output_lines, output_arg = _block_stream_argument(
            "scalar_t *",
            "block_output_streams",
            "block_output",
            n_streams,
            field_stream_order,
            "        ",
            mutable=True,
        )
        lines.extend(direction_lines)
        lines.extend(output_lines)
        block_function = (
            block
            if direction_lines or output_lines
            else "%s_contiguous" % block
        )
    lines.append(
        "        const scalar_t *const block_adjugate[DIM * DIM] = {%s};"
        % ", ".join("block_adjugate_data[%d]" % i for i in range(dim * dim))
    )
    call_args = [
        "1",
        "1",
        "block_determinant",
    ]
    if dependencies.uses_adjugate:
        call_args.append("block_adjugate")
    if tensor_product:
        call_args.append(_mesh_reference_name("isoparametric", "shape_1d"))
        if dependencies.uses_reference_gradients:
            call_args.append(_mesh_reference_name("isoparametric", "grad_1d"))
        call_args.append(_mesh_reference_name("isoparametric", "q_weight_1d"))
    else:
        call_args.append(_mesh_reference_name("isoparametric", "shape"))
        if dependencies.uses_reference_gradients:
            call_args.extend(
                _mesh_reference_name(
                    "isoparametric",
                    sfem_simplex_grad_ref_name("grad_ref", d),
                )
                for d in range(dim)
            )
        call_args.append(_mesh_reference_name("isoparametric", "q_weight"))
    if state_dependencies.current:
        call_args.append(state_stream_args["current"])
    if state_dependencies.previous:
        call_args.append(state_stream_args["previous"])
    call_args.append(direction_arg)
    call_args.extend(map(str, dependencies.parameters))
    call_args.append(output_arg)
    lines.extend(
        [
            "",
            "        for (int entry = 0; entry < N_STREAMS * N_STREAMS; ++entry) {",
            "            element_matrix[entry] = scalar_t(0);",
            "        }",
            "        static constexpr int TENSOR_STREAMS[N_STREAMS] = {%s};"
            % ", ".join(str(stream) for stream in field_streams_in_tensor_order),
            "        for (int trial = 0; trial < N_STREAMS; ++trial) {",
            "            const int tensor_trial = TENSOR_STREAMS[trial];",
            "            for (int stream = 0; stream < N_STREAMS; ++stream) {",
            "                block_direction[stream][0] = scalar_t(0);",
            "                block_output[stream][0] = scalar_t(0);",
            "            }",
            "            block_direction[tensor_trial][0] = scalar_t(1);",
            "            %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
            % (block_function, ", ".join(call_args)),
            "            for (int test = 0; test < N_STREAMS; ++test) {",
            "                const int tensor_test = TENSOR_STREAMS[test];",
            "                element_matrix[test * N_STREAMS + trial] = block_output[tensor_test][0];",
            "            }",
            "        }",
            "",
            (
                "        %s_scatter_coo_triplets(ev, element_matrix, element, rows, cols, values);"
                % function_base
                if n_fields == 1
                else "        %s_scatter_coo_triplets(ev, out_stride, element_matrix, element, rows, cols, values);"
                % function_base
            ),
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
    for scalar_type, suffix in (("double", ""), ("float", "_float")):
        typed_params = [
            param.replace("scalar_t", scalar_type) for param in params
        ]
        lines.append('extern "C" int %s%s(' % (function_base, suffix))
        for index, param in enumerate(typed_params):
            lines.append(
                "        %s%s" % (param, "," if index + 1 < len(typed_params) else "")
            )
        call_args = ["nelements", "nnodes", "elements", "points"]
        call_args.extend(map(str, dependencies.parameters))
        if state_dependencies.current:
            call_args.append("current_stride")
            call_args.extend(field.name for field in system.fields)
        if state_dependencies.previous:
            call_args.append("previous_stride")
            call_args.extend("%s_old" % field.name for field in system.fields)
        if n_fields != 1:
            call_args.append("out_stride")
        call_args.extend(("rows", "cols", "values"))
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


def _scalar_dia_matrix_scatter_lines(function_base, n_shape):
    return [
        "template <typename scalar_t>",
        "static SFEM_INLINE int %s_scatter_dia(" % function_base,
        "        const idx_t *const SFEM_RESTRICT ev,",
        "        const scalar_t *const SFEM_RESTRICT element_matrix,",
        "        const ptrdiff_t nnodes,",
        "        const int *const SFEM_RESTRICT diag_offsets,",
        "        const ptrdiff_t ndiag,",
        "        scalar_t *const SFEM_RESTRICT values) {",
        "    static constexpr int N_SHAPE = %d;" % n_shape,
        "    ptrdiff_t diagonals[N_SHAPE * N_SHAPE];",
        "    bool valid_diagonal_offsets = true;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            const int offset = (int)(ev[j] - ev[i]);",
        "            ptrdiff_t diagonal = 0;",
        "            while (diagonal < ndiag && diag_offsets[diagonal] != offset) ++diagonal;",
        "            if (diagonal == ndiag) {",
        "                if (valid_diagonal_offsets) {",
        "                    std::fprintf(stderr, \"%s_scatter_dia missing diagonal offset %%d\\n\", offset);"
        % function_base,
        "                }",
        "                diagonals[i * N_SHAPE + j] = 0;",
        "                valid_diagonal_offsets = false;",
        "            } else {",
        "                diagonals[i * N_SHAPE + j] = diagonal;",
        "            }",
        "        }",
        "    }",
        "    if (!valid_diagonal_offsets) return SFEM_FAILURE;",
        "    for (int i = 0; i < N_SHAPE; ++i) {",
        "        for (int j = 0; j < N_SHAPE; ++j) {",
        "            const ptrdiff_t diagonal = diagonals[i * N_SHAPE + j];",
        "#pragma omp atomic update",
        "            values[diagonal * nnodes + ev[i]] += element_matrix[i * N_SHAPE + j];",
        "        }",
        "    }",
        "    return SFEM_SUCCESS;",
        "}",
        "",
    ]


def _scalar_dia_matrix_assembly_source(
    system,
    prefix,
    local_prefix,
    specialization,
    dependencies,
    basis_family,
    geometry_family,
    matrix_format_plan,
):
    matrix_formats = _matrix_format_values(matrix_format_plan)
    if "dia" not in matrix_formats:
        return []
    if len(system.fields) != 1:
        return []
    if dependencies.current or dependencies.previous or not dependencies.direction:
        return []

    rule = specialization.quadrature_rule
    dim = system.dim
    n_shape = rule.n_shape
    n_qp = rule.n_qp
    tensor_product = _is_tensor_product_family(rule, basis_family)
    tensor_product_geometry = _is_tensor_product_family(rule, geometry_family)
    shape_order = (
        tuple(range(n_shape))
        if sfem_tensor_product_hex_uses_cartesian_ordering(rule.element_type)
        else tensor_product_cartesian_shape_order(dim, n_shape)
        if tensor_product
        else tuple(range(n_shape))
    )
    field_stream_order = streams_in_shape_order(tuple(range(n_shape)), 1, shape_order)
    stream_to_tensor_order = _stream_to_tensor_order(field_stream_order)
    use_contiguous_field_streams = not _identity_order(field_stream_order)
    function_base = "%s_hessian_dia_isoparametric_mesh_soa" % prefix
    impl = "%s_impl" % function_base
    block = "%s_jacobian_action_block" % local_prefix
    params = [
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "idx_t **const SFEM_RESTRICT elements",
        "const geom_t *const *const SFEM_RESTRICT points",
    ]
    params.extend(
        "const scalar_t %s" % parameter for parameter in dependencies.parameters
    )
    params.extend(
        [
            "const int *const SFEM_RESTRICT diag_offsets",
            "const ptrdiff_t ndiag",
            "scalar_t *const SFEM_RESTRICT values",
        ]
    )

    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
    ]
    lines.extend(_scalar_dia_matrix_scatter_lines(function_base, n_shape))
    lines.extend(
        [
            "template <typename scalar_t>",
            "%s int %s(" % (_function_qualifier(), impl),
        ]
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
            "    static constexpr int VECTOR_SIZE = 1;",
            "    (void)nnodes;",
        ]
    )
    lines.extend(_mesh_reference_alias_lines(prefix, rule, "isoparametric"))
    coordinate_element_lines, coordinate_element_array = (
        _coordinate_element_alias_lines(
            dim,
            n_shape,
            rule.element_type,
            "    ",
        )
        if tensor_product_geometry
        else ([], "elements")
    )
    lines.extend(coordinate_element_lines)
    lines.extend(
        [
            "",
            "    int invalid_matrix_graph = 0;",
            "#pragma omp parallel for schedule(static) reduction(|:invalid_matrix_graph)",
            "    for (ptrdiff_t element = 0; element < nelements; ++element) {",
            "        const ptrdiff_t evbegin = element;",
            "        const int nelems = 1;",
            "        idx_t ev[N_SHAPE];",
            "        scalar_t element_matrix[N_SHAPE * N_SHAPE];",
            "        scalar_t block_coordinates[DIM * N_SHAPE][VECTOR_SIZE];",
            "        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];",
            "        scalar_t block_determinant[N_QP * VECTOR_SIZE];",
            "        scalar_t block_direction[N_SHAPE][VECTOR_SIZE];",
            "        scalar_t block_output[N_SHAPE][VECTOR_SIZE];",
            "        const geom_t *const coordinate_components[DIM] = {%s};"
            % ", ".join("points[%d]" % d for d in range(dim)),
            "",
            "        for (int shape = 0; shape < N_SHAPE; ++shape) {",
            "            const idx_t node = elements[shape][element];",
            "            const idx_t coordinate_node = %s[shape][element];" % coordinate_element_array,
            "            ev[shape] = node;",
            "            for (int d = 0; d < DIM; ++d) {",
            "                block_coordinates[shape * DIM + d][0] = scalar_t(coordinate_components[d][coordinate_node]);",
            "            }",
            "        }",
            "",
        ]
    )
    if tensor_product_geometry:
        lines.extend(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim=dim,
                n_shape=n_shape,
                n_qp=rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams="block_coordinates",
                contiguous_coordinate_streams=True,
                adjugate_target=lambda component, index: (
                    "block_adjugate_data[%d][%s]" % (component, index)
                ),
                determinant_target=lambda index: (
                    "block_determinant[%s]" % index
                ),
                adjugate_streams=tuple(
                    "block_adjugate_data[%d]" % component
                    for component in range(dim * dim)
                ),
                determinant_stream="block_determinant",
                shape_name=_mesh_reference_name("isoparametric", "shape_1d"),
                grad_name=_mesh_reference_name("isoparametric", "grad_1d"),
            )
        )
    else:
        lines.extend(
            [
                "        scalar_t *block_adjugate_streams[DIM * DIM] = {%s};"
                % ", ".join(
                    "block_adjugate_data[%d]" % component
                    for component in range(dim * dim)
                ),
                "        for (int q = 0; q < N_QP; ++q) {",
                "            const int lane = 0;",
            ]
        )
        for i in range(dim):
            for j in range(dim):
                terms = [
                    "block_coordinates[%d][lane] * %s[q * N_SHAPE + %d]"
                    % (
                        shape * dim + i,
                        _mesh_reference_name(
                            "isoparametric",
                            sfem_simplex_grad_ref_name("grad_ref", j),
                        ),
                        shape,
                    )
                    for shape in range(n_shape)
                ]
                lines.append(
                    "            const scalar_t J%d%d = %s;"
                    % (i, j, " + ".join(terms))
                )
        lines.extend(_isoparametric_geometry_assignment_lines(dim, "            "))
        lines.extend(["        }"])
    if use_contiguous_field_streams:
        direction_arg = "block_direction"
        output_arg = "block_output"
        block_function = "%s_contiguous" % block
    else:
        direction_lines, direction_arg = _block_stream_argument(
            "const scalar_t *",
            "block_direction_streams",
            "block_direction",
            n_shape,
            field_stream_order,
            "        ",
            mutable=False,
        )
        output_lines, output_arg = _block_stream_argument(
            "scalar_t *",
            "block_output_streams",
            "block_output",
            n_shape,
            field_stream_order,
            "        ",
            mutable=True,
        )
        lines.extend(direction_lines)
        lines.extend(output_lines)
        block_function = (
            block
            if direction_lines or output_lines
            else "%s_contiguous" % block
        )
    lines.append(
        "        const scalar_t *const block_adjugate[DIM * DIM] = {%s};"
        % ", ".join("block_adjugate_data[%d]" % i for i in range(dim * dim))
    )
    call_args = [
        "1",
        "1",
        "block_determinant",
        "block_adjugate",
    ]
    if tensor_product:
        call_args.append(_mesh_reference_name("isoparametric", "shape_1d"))
        if dependencies.uses_reference_gradients:
            call_args.append(_mesh_reference_name("isoparametric", "grad_1d"))
        call_args.append(_mesh_reference_name("isoparametric", "q_weight_1d"))
    else:
        call_args.append(_mesh_reference_name("isoparametric", "shape"))
        if dependencies.uses_reference_gradients:
            call_args.extend(
                _mesh_reference_name(
                    "isoparametric",
                    sfem_simplex_grad_ref_name("grad_ref", d),
                )
                for d in range(dim)
            )
        call_args.append(_mesh_reference_name("isoparametric", "q_weight"))
    call_args.append(direction_arg)
    call_args.extend(map(str, dependencies.parameters))
    call_args.append(output_arg)
    lines.extend(
        [
            "",
            "        for (int entry = 0; entry < N_SHAPE * N_SHAPE; ++entry) {",
            "            element_matrix[entry] = scalar_t(0);",
            "        }",
            "        static constexpr int TENSOR_STREAMS[N_SHAPE] = {%s};"
            % ", ".join(str(stream) for stream in stream_to_tensor_order),
            "        for (int trial = 0; trial < N_SHAPE; ++trial) {",
            "            const int tensor_trial = TENSOR_STREAMS[trial];",
            "            for (int stream = 0; stream < N_SHAPE; ++stream) {",
            "                block_direction[stream][0] = scalar_t(0);",
            "                block_output[stream][0] = scalar_t(0);",
            "            }",
            "            block_direction[tensor_trial][0] = scalar_t(1);",
            "            %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
            % (block_function, ", ".join(call_args)),
            "            for (int test = 0; test < N_SHAPE; ++test) {",
            "                const int tensor_test = TENSOR_STREAMS[test];",
            "                element_matrix[test * N_SHAPE + trial] = block_output[tensor_test][0];",
            "            }",
            "        }",
            "",
            "        invalid_matrix_graph |= (%s_scatter_dia(ev, element_matrix, nnodes, diag_offsets, ndiag, values) != SFEM_SUCCESS);"
            % function_base,
            "    }",
            "",
            "    return invalid_matrix_graph ? SFEM_FAILURE : SFEM_SUCCESS;",
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )
    for scalar_type, suffix in (("double", ""), ("float", "_float")):
        typed_params = [
            param.replace("scalar_t", scalar_type) for param in params
        ]
        lines.append('extern "C" int %s%s(' % (function_base, suffix))
        for index, param in enumerate(typed_params):
            lines.append(
                "        %s%s" % (param, "," if index + 1 < len(typed_params) else "")
            )
        call_args = ["nelements", "nnodes", "elements", "points"]
        call_args.extend(map(str, dependencies.parameters))
        call_args.extend(("diag_offsets", "ndiag", "values"))
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


def _isoparametric_mesh_operator_source(
    system,
    prefix,
    local_prefix,
    specialization,
    form,
    dependencies,
    coefficients,
    basis_family=None,
    geometry_family=None,
):
    rule = specialization.quadrature_rule
    dim = system.dim
    n_fields = len(system.fields)
    n_shape = rule.n_shape
    n_qp = rule.n_qp
    vector_size = specialization.vector_size
    tensor_product = _is_tensor_product_family(rule, basis_family)
    tensor_product_geometry = _is_tensor_product_family(rule, geometry_family)
    gradient_metric = None
    shape_order = (
        tuple(range(n_shape))
        if sfem_tensor_product_hex_uses_cartesian_ordering(rule.element_type)
        else tensor_product_cartesian_shape_order(dim, n_shape)
        if tensor_product
        else tuple(range(n_shape))
    )
    field_stream_order = streams_in_shape_order(
        tuple(range(n_fields * n_shape)),
        n_fields,
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
        "%s int %s(" % (_function_qualifier(), impl),
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
    lines.extend(_mesh_reference_alias_lines(prefix, rule, "isoparametric"))
    field_shape_order = _single_field_shape_order(n_shape, n_fields, field_stream_order)
    field_element_lines, field_element_array = _single_field_element_alias_lines(
        n_shape,
        field_shape_order,
        "    ",
    )
    lines.extend(field_element_lines)
    coordinate_element_lines, coordinate_element_array = (
        _coordinate_element_alias_lines(
            dim,
            n_shape,
            rule.element_type,
            "    ",
        )
        if tensor_product_geometry
        else ([], "elements")
    )
    lines.extend(coordinate_element_lines)
    lines.extend(
        [
            "",
            _parallel_for_pragma("static"),
            "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            "        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
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
    if gradient_metric is not None:
        lines.append(
            "        scalar_t block_geom_metric_data[%d][N_QP * VECTOR_SIZE];"
            % gradient_metric.metric_components
        )
    lines.extend(
        [
            "        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];",
        ]
    )
    lines.extend(["", *_coordinate_gather_lines(dim, "        ", coordinate_element_array)])
    lines.extend(_field_gather_lines(system, dependencies, "        ", field_element_array))
    lines.extend(["", *_zero_block_output_lines("block_output", n_fields * n_shape, "        ")])
    if tensor_product_geometry:
        lines.append("")
        lines.extend(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim=dim,
                n_shape=n_shape,
                n_qp=rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams="block_coordinates",
                contiguous_coordinate_streams=True,
                adjugate_target=lambda component, index: (
                    "block_adjugate_data[%d][%s]" % (component, index)
                ),
                determinant_target=lambda index: (
                    "block_determinant[%s]" % index
                ),
                adjugate_streams=tuple(
                    "block_adjugate_data[%d]" % component
                    for component in range(dim * dim)
                ),
                determinant_stream="block_determinant",
                shape_name=_mesh_reference_name("isoparametric", "shape_1d"),
                grad_name=_mesh_reference_name("isoparametric", "grad_1d"),
            )
        )
    else:
        lines.extend(
            [
                "",
                "        scalar_t *block_adjugate_streams[DIM * DIM] = {%s};"
                % ", ".join(
                    "block_adjugate_data[%d]" % component
                    for component in range(dim * dim)
                ),
                "        for (int q = 0; q < N_QP; ++q) {",
                *_work_item_loop_lines("            "),
            ]
        )
        for i in range(dim):
            for j in range(dim):
                terms = [
                    "block_coordinates[%d][lane] * %s[q * N_SHAPE + %d]"
                    % (
                        shape * dim + i,
                        _mesh_reference_name(
                            "isoparametric",
                            sfem_simplex_grad_ref_name("grad_ref", j),
                        ),
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
    if gradient_metric is not None:
        lines.extend(
            [
                "",
                "        for (int q = 0; q < N_QP; ++q) {",
                *_work_item_loop_lines("            "),
                "                const ptrdiff_t geometry_offset = q * VECTOR_SIZE + lane;",
            ]
        )
        lines.extend(
            _geometry_metric_grouping_lines(
                dim,
                "block_determinant[geometry_offset]",
                lambda component: "block_adjugate_data[%d][geometry_offset]" % component,
                lambda component: "block_geom_metric_data[%d][geometry_offset]" % component,
                "                ",
                "metric",
            )
        )
        lines.extend(["            }", "        }"])
    lines.append("")
    block_stream_lines, block_stream_args = _single_field_block_stream_arguments(
        dependencies,
        n_fields * n_shape,
        field_stream_order,
        "        ",
        force_contiguous=bool(field_element_lines),
    )
    lines.extend(block_stream_lines)
    block_function = "%s_contiguous" % block if not block_stream_lines else block
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
    if gradient_metric is not None:
        lines.append(
            "        const scalar_t *const block_geom_metric[%d] = %s;"
            % (
                gradient_metric.metric_components,
                _geometry_metric_stream_initializer(
                    "block_geom_metric_data",
                    dim,
                ),
            )
        )
    call_args = [
        "nelems",
        "VECTOR_SIZE",
    ]
    if gradient_metric is not None:
        call_args.append("block_geom_metric")
    else:
        call_args.append("block_determinant")
    if dependencies.uses_adjugate and gradient_metric is None:
        call_args.append("block_adjugate")
    if tensor_product:
        call_args.append(_mesh_reference_name("isoparametric", "shape_1d"))
        if dependencies.uses_reference_gradients:
            call_args.append(_mesh_reference_name("isoparametric", "grad_1d"))
        call_args.append(_mesh_reference_name("isoparametric", "q_weight_1d"))
    else:
        call_args.append(_mesh_reference_name("isoparametric", "shape"))
        if dependencies.uses_reference_gradients:
            call_args.extend(
                _mesh_reference_name(
                    "isoparametric",
                    sfem_simplex_grad_ref_name("grad_ref", d),
                )
                for d in range(dim)
            )
        call_args.append(_mesh_reference_name("isoparametric", "q_weight"))
    if dependencies.current:
        call_args.append(block_stream_args["current"])
    if dependencies.previous:
        call_args.append(block_stream_args["previous"])
    if dependencies.direction:
        call_args.append(block_stream_args["direction"])
    call_args.extend(map(str, dependencies.parameters))
    call_args.append(block_stream_args["output"])
    lines.extend(
        [
            "",
            "        %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
            % (block_function, ", ".join(call_args)),
            "",
        ]
    )
    lines.extend(_field_atomic_scatter_lines(system, "        ", field_element_array))
    lines.extend(
        [
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


def _scalar_packed_jacobian_action_source(
    system,
    prefix,
    local_prefix,
    specialization,
    dependencies,
    coefficients,
    basis_family=None,
    geometry_family=None,
):
    if len(system.fields) != 1 or not dependencies.direction:
        return []
    rule = specialization.quadrature_rule
    dim = system.dim
    n_fields = len(system.fields)
    n_shape = rule.n_shape
    n_qp = rule.n_qp
    tensor_product = _is_tensor_product_family(rule, basis_family)
    tensor_product_geometry = _is_tensor_product_family(rule, geometry_family)
    shape_order = (
        tuple(range(n_shape))
        if sfem_tensor_product_hex_uses_cartesian_ordering(rule.element_type)
        else tensor_product_cartesian_shape_order(dim, n_shape)
        if tensor_product
        else tuple(range(n_shape))
    )
    field_stream_order = streams_in_shape_order(
        tuple(range(n_fields * n_shape)),
        n_fields,
        shape_order,
    )
    function = "%s_jacobian_action_packed_isoparametric_mesh_soa" % prefix
    impl = "%s_impl" % function
    block = "%s_jacobian_action_block" % local_prefix
    block_function = "%s_contiguous" % block
    params = [
        "const ptrdiff_t n_packs",
        "const ptrdiff_t n_elements_per_pack",
        "const ptrdiff_t nelements",
        "const ptrdiff_t nnodes",
        "const ptrdiff_t max_nodes_per_pack",
        "uint16_t **const SFEM_RESTRICT elements",
        "const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr",
        "const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes",
        "const ptrdiff_t *const SFEM_RESTRICT ghost_ptr",
        "const idx_t *const SFEM_RESTRICT ghost_idx",
        "const geom_t *const *const SFEM_RESTRICT points",
    ]
    params.extend("const scalar_t %s" % parameter for parameter in dependencies.parameters)
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
    coordinate_element_lines, coordinate_element_array = (
        _coordinate_element_alias_lines(
            dim,
            n_shape,
            rule.element_type,
            "    ",
            pointer_type="uint16_t",
        )
        if tensor_product_geometry
        else ([], "elements")
    )
    field_shape_order = _single_field_shape_order(n_shape, n_fields, field_stream_order)
    field_element_lines, field_element_array = _single_field_element_alias_lines(
        n_shape,
        field_shape_order,
        "    ",
        pointer_type="uint16_t",
    )
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename scalar_t>",
        "%s int %s(" % (_function_qualifier(), impl),
    ]
    for index, param in enumerate(params):
        lines.append("        %s%s" % (param, "," if index + 1 < len(params) else ""))
    lines.extend(
        [
            ") {",
            "    static constexpr int DIM = %d;" % dim,
            "    static constexpr int N_QP = %d;" % n_qp,
            "    static constexpr int N_SHAPE = %d;" % n_shape,
            "    static constexpr int N_FIELDS = %d;" % n_fields,
            "    static constexpr int N_STREAMS = N_FIELDS * N_SHAPE;",
            "    static constexpr int VECTOR_SIZE = 1;",
            "    (void)nnodes;",
        ]
    )
    lines.extend(_mesh_reference_alias_lines(prefix, rule, "isoparametric"))
    lines.extend(coordinate_element_lines)
    lines.extend(field_element_lines)
    lines.extend(
        [
            "",
            "#pragma omp parallel",
            "    {",
            "        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);",
        ]
    )
    if dependencies.current:
        lines.append(
            "        scalar_t *const SFEM_RESTRICT pack_current = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)max_nodes_per_pack);"
        )
    if dependencies.previous:
        lines.append(
            "        scalar_t *const SFEM_RESTRICT pack_previous = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)max_nodes_per_pack);"
        )
    lines.extend(
        [
            "        scalar_t *const SFEM_RESTRICT pack_direction = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)max_nodes_per_pack);",
            "        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)max_nodes_per_pack);",
            "",
            "#pragma omp for schedule(static)",
            "        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {",
            "            const ptrdiff_t e_start = pack * n_elements_per_pack;",
            "            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);",
            "            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];",
            "            const ptrdiff_t n_shared = n_shared_nodes[pack];",
            "            const ptrdiff_t n_not_shared = n_contiguous - n_shared;",
            "            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];",
            "            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];",
            "            const geom_t *const coordinate_components[DIM] = {%s};"
            % ", ".join("points[%d]" % d for d in range(dim)),
            "            for (int d = 0; d < DIM; ++d) {",
            "                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;",
            "                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];",
            "                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
            "                    pack_coordinate[k] = scalar_t(coordinate_component[owned_nodes_ptr[pack] + k]);",
            "                }",
            "                for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
            "                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[ghosts[k]]);",
            "                }",
            "            }",
        ]
    )
    field = system.fields[0]
    if dependencies.current:
        lines.extend(
            [
                "            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
                "                const idx_t node = owned_nodes_ptr[pack] + k;",
                "                pack_current[k] = %s[node * current_stride];" % field.name,
                "            }",
                "            for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                "                pack_current[n_contiguous + k] = %s[ghosts[k] * current_stride];" % field.name,
                "            }",
            ]
        )
    if dependencies.previous:
        lines.extend(
            [
                "            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
                "                const idx_t node = owned_nodes_ptr[pack] + k;",
                "                pack_previous[k] = %s_old[node * previous_stride];" % field.name,
                "            }",
                "            for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
                "                pack_previous[n_contiguous + k] = %s_old[ghosts[k] * previous_stride];" % field.name,
                "            }",
            ]
        )
    lines.extend(
        [
            "            for (ptrdiff_t k = 0; k < n_contiguous; ++k) {",
            "                const idx_t node = owned_nodes_ptr[pack] + k;",
            "                pack_direction[k] = %s_direction[node * direction_stride];" % field.name,
            "            }",
            "            for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
            "                pack_direction[n_contiguous + k] = %s_direction[ghosts[k] * direction_stride];" % field.name,
            "            }",
            "",
            "            for (ptrdiff_t element = e_start; element < e_end; ++element) {",
            "                const int nelems = 1;",
            "                scalar_t block_coordinates[DIM * N_SHAPE][VECTOR_SIZE];",
            "                scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];",
            "                scalar_t block_determinant[N_QP * VECTOR_SIZE];",
        ]
    )
    if dependencies.current:
        lines.append("                scalar_t block_current[N_STREAMS][VECTOR_SIZE];")
    if dependencies.previous:
        lines.append("                scalar_t block_previous[N_STREAMS][VECTOR_SIZE];")
    lines.extend(
        [
            "                scalar_t block_direction[N_STREAMS][VECTOR_SIZE];",
            "                scalar_t block_output[N_STREAMS][VECTOR_SIZE];",
            "",
            "                for (int shape = 0; shape < N_SHAPE; ++shape) {",
            "                    const uint16_t packed_node = elements[shape][element];",
            "                    const uint16_t coordinate_packed_node = %s[shape][element];" % coordinate_element_array,
            "                    const uint16_t field_packed_node = %s[shape][element];" % field_element_array,
            "                    for (int d = 0; d < DIM; ++d) {",
            "                        block_coordinates[shape * DIM + d][0] = pack_coordinates[d * max_nodes_per_pack + coordinate_packed_node];",
            "                    }",
        ]
    )
    if dependencies.current:
        lines.append("                    block_current[shape][0] = pack_current[field_packed_node];")
    if dependencies.previous:
        lines.append("                    block_previous[shape][0] = pack_previous[field_packed_node];")
    lines.extend(
        [
            "                    block_direction[shape][0] = pack_direction[field_packed_node];",
            "                    block_output[shape][0] = scalar_t(0);",
            "                }",
        ]
    )
    if tensor_product_geometry:
        lines.append("")
        lines.extend(
            tensor_product_gradient_isoparametric_geometry_lines(
                dim=dim,
                n_shape=n_shape,
                n_qp=rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams="block_coordinates",
                contiguous_coordinate_streams=True,
                adjugate_target=lambda component, index: (
                    "block_adjugate_data[%d][%s]" % (component, index)
                ),
                determinant_target=lambda index: ("block_determinant[%s]" % index),
                adjugate_streams=tuple(
                    "block_adjugate_data[%d]" % component
                    for component in range(dim * dim)
                ),
                determinant_stream="block_determinant",
                shape_name=_mesh_reference_name("isoparametric", "shape_1d"),
                grad_name=_mesh_reference_name("isoparametric", "grad_1d"),
            )
        )
    else:
        lines.extend(
            [
                "",
                "                scalar_t *block_adjugate_streams[DIM * DIM] = {%s};"
                % ", ".join(
                    "block_adjugate_data[%d]" % component
                    for component in range(dim * dim)
                ),
                "                for (int q = 0; q < N_QP; ++q) {",
                "                    const int lane = 0;",
            ]
        )
        for i in range(dim):
            for j in range(dim):
                terms = [
                    "block_coordinates[%d][0] * %s[q * N_SHAPE + %d]"
                    % (
                        shape * dim + i,
                        _mesh_reference_name(
                            "isoparametric",
                            sfem_simplex_grad_ref_name("grad_ref", j),
                        ),
                        shape,
                    )
                    for shape in range(n_shape)
                ]
                lines.append(
                    "                    const scalar_t J%d%d = %s;"
                    % (i, j, " + ".join(terms))
                )
        lines.extend(_isoparametric_geometry_assignment_lines(dim, "                    "))
        lines.extend(["                }"])
    if dependencies.uses_adjugate:
        lines.append(
            "                const scalar_t *const block_adjugate[DIM * DIM] = {%s};"
            % ", ".join(
                "block_adjugate_data[%d]" % component
                for component in range(dim * dim)
            )
        )
    call_args = ["nelems", "VECTOR_SIZE", "block_determinant"]
    if dependencies.uses_adjugate:
        call_args.append("block_adjugate")
    if tensor_product:
        call_args.append(_mesh_reference_name("isoparametric", "shape_1d"))
        if dependencies.uses_reference_gradients:
            call_args.append(_mesh_reference_name("isoparametric", "grad_1d"))
        call_args.append(_mesh_reference_name("isoparametric", "q_weight_1d"))
    else:
        call_args.append(_mesh_reference_name("isoparametric", "shape"))
        if dependencies.uses_reference_gradients:
            call_args.extend(
                _mesh_reference_name(
                    "isoparametric",
                    sfem_simplex_grad_ref_name("grad_ref", d),
                )
                for d in range(dim)
            )
        call_args.append(_mesh_reference_name("isoparametric", "q_weight"))
    if dependencies.current:
        call_args.append("block_current")
    if dependencies.previous:
        call_args.append("block_previous")
    call_args.append("block_direction")
    call_args.extend(map(str, dependencies.parameters))
    call_args.append("block_output")
    lines.extend(
        [
            "",
            "                %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
            % (block_function, ", ".join(call_args)),
            "",
            "                for (int shape = 0; shape < N_SHAPE; ++shape) {",
            "                    const uint16_t field_packed_node = %s[shape][element];"
            % field_element_array,
            "                    pack_out[field_packed_node] += block_output[shape][0];",
            "                }",
            "            }",
            "",
            "            for (ptrdiff_t k = 0; k < n_not_shared; ++k) {",
            "                %s_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_out[k];" % field.name,
            "                pack_out[k] = scalar_t(0);",
            "            }",
            "            for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {",
            "#pragma omp atomic update",
            "                %s_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_out[k];" % field.name,
            "                pack_out[k] = scalar_t(0);",
            "            }",
            "            for (ptrdiff_t k = 0; k < n_ghost; ++k) {",
            "#pragma omp atomic update",
            "                %s_out[ghosts[k] * out_stride] += pack_out[n_contiguous + k];" % field.name,
            "                pack_out[n_contiguous + k] = scalar_t(0);",
            "            }",
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
    for scalar_type, suffix in (("double", ""), ("float", "_float")):
        typed_params = [param.replace("scalar_t", scalar_type) for param in params]
        lines.append('extern "C" int %s%s(' % (function, suffix))
        for index, param in enumerate(typed_params):
            lines.append("        %s%s" % (param, "," if index + 1 < len(typed_params) else ""))
        call_args = [
            "n_packs",
            "n_elements_per_pack",
            "nelements",
            "nnodes",
            "max_nodes_per_pack",
            "elements",
            "owned_nodes_ptr",
            "n_shared_nodes",
            "ghost_ptr",
            "ghost_idx",
            "points",
        ]
        call_args.extend(map(str, dependencies.parameters))
        if dependencies.current:
            call_args.append("current_stride")
            call_args.extend(field.name for field in system.fields)
        if dependencies.previous:
            call_args.append("previous_stride")
            call_args.extend("%s_old" % field.name for field in system.fields)
        call_args.append("direction_stride")
        call_args.extend("%s_direction" % field.name for field in system.fields)
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
    return isoparametric_adjugate_call_lines(
        dim=dim,
        indent=indent,
        index="q * VECTOR_SIZE + lane",
        stream_array_name="block_adjugate_streams",
        determinant_stream="block_determinant",
    )


def _mesh_reference_alias_lines(prefix, rule, geometry_mode, emit_reference_basis=True):
    references = tuple(sfem_mesh_reference_data(rule))
    if not emit_reference_basis:
        references = tuple(
            reference for reference in references if reference.name.startswith("q_weight")
        )
    return [
        "    const scalar_t *const %s = %s;"
        % (
            _mesh_reference_name(geometry_mode, reference.name),
            quadrature_reference_accessor(prefix, geometry_mode, reference.name),
        )
        for reference in references
    ]
