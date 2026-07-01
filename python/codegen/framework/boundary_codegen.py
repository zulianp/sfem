import math

import sympy as sp

from .symbolic import GeneratedKernelFile, _sfem_ccode, _sfem_math_header_source
from .fem import (
    sfem_is_proteus_hex_element,
    sfem_tensor_product_hex_order,
    _sfem_lagrange_1d_at,
    _sfem_triangle_quadrature_rule,
    _sfem_unit_interval_gauss_rule,
    _tri6_reference_gradients,
)
from .targets import OpenMPTarget


def _target():
    return OpenMPTarget()


def _function_qualifier():
    return _target().function_qualifier()


def _parallel_for_pragma(schedule=None):
    return _target().parallel_for_pragma(schedule)


def _atomic_update_pragma():
    return _target().atomic_update_pragma()


_CELL_TO_SURFACE = {
    "TRI3": "EDGESHELL2",
    "QUAD4": "EDGESHELL2",
    "TET4": "TRISHELL3",
    "TET10": "TRISHELL6",
    "HEX8": "QUADSHELL4",
    "HEX27": "QUADSHELL9",
    "PROTEUS_HEX8": "PROTEUS_QUADSHELL4",
    "PROTEUS_HEX27": "PROTEUS_QUADSHELL9",
    "PROTEUS_HEX64": "PROTEUS_QUADSHELL16",
    "PROTEUS_HEX125": "PROTEUS_QUADSHELL25",
    "PROTEUS_HEX216": "PROTEUS_QUADSHELL36",
    "PROTEUS_HEX343": "PROTEUS_QUADSHELL49",
    "PROTEUS_HEX512": "PROTEUS_QUADSHELL64",
    "PROTEUS_HEX729": "PROTEUS_QUADSHELL81",
}


def generate_boundary_residual_sfem_files(
    collection,
    *,
    prefix,
    emission_plan,
    expression_plan,
    reference_data_plan=None,
    diagnostics_plan=None,
):
    if emission_plan is None:
        raise ValueError("boundary residual codegen requires an ElementEmissionPlan")
    if expression_plan is None:
        raise ValueError("boundary residual codegen requires a KernelExpressionPlan")
    element_type = emission_plan.element_type
    surface = _surface_element(element_type)
    if collection.measure != "ds":
        raise ValueError("boundary residual codegen requires measure 'ds'")
    if len(tuple(collection.fields)) != 1:
        raise ValueError("boundary residual codegen currently supports one field")
    system = collection.source
    if system is None:
        raise ValueError("boundary residual form collection requires a lowered residual system")
    field = tuple(collection.fields)[0]
    components = int(field.components)
    coefficients = _boundary_coefficients_from_expression_plan(system, expression_plan)
    _validate_boundary_coefficients(system, coefficients)
    _validate_boundary_metadata(expression_plan.dependencies, coefficients)
    parameters = _dependency_parameters(expression_plan.dependencies)
    function = "%s_%s_boundary_residual_soa" % (prefix, surface.lower())
    source = _boundary_source(
        function,
        element_type,
        surface,
        components,
        parameters,
        coefficients,
        system,
        use_tensor_product=_uses_tensor_product_boundary_surface(surface),
    )
    return (
        GeneratedKernelFile("kernel_math.hpp", _sfem_math_header_source()),
        GeneratedKernelFile("%s_boundary_operator.cpp" % prefix, source),
    )


def _surface_element(element_type):
    try:
        return _CELL_TO_SURFACE[element_type]
    except KeyError as exc:
        raise ValueError("unsupported boundary cell element '%s'" % element_type) from exc


def _uses_tensor_product_boundary_surface(surface):
    surface = str(surface).upper()
    return (
        surface in ("QUADSHELL4", "QUADSHELL9")
        or surface.startswith("PROTEUS_QUADSHELL")
    )


def _cell_side_nodes(element_type):
    element_type = str(element_type).upper()
    if sfem_is_proteus_hex_element(element_type):
        return _proteus_hex_side_nodes(sfem_tensor_product_hex_order(element_type))
    tables = {
        "TRI3": (
            (0, 1),
            (1, 2),
            (2, 0),
        ),
        "QUAD4": (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
        ),
        "TET4": (
            (0, 1, 3),
            (1, 2, 3),
            (0, 3, 2),
            (0, 2, 1),
        ),
        "TET10": (
            (0, 1, 3, 4, 8, 7),
            (1, 2, 3, 5, 9, 8),
            (0, 3, 2, 7, 9, 6),
            (0, 2, 1, 6, 5, 4),
        ),
        "HEX8": (
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
            (3, 2, 1, 0),
            (4, 5, 6, 7),
        ),
        "HEX27": (
            (0, 1, 5, 4, 8, 17, 12, 16, 20),
            (1, 2, 6, 5, 9, 18, 13, 17, 21),
            (2, 3, 7, 6, 10, 19, 14, 18, 22),
            (3, 0, 4, 7, 11, 16, 15, 19, 23),
            (3, 2, 1, 0, 10, 9, 8, 11, 24),
            (4, 5, 6, 7, 12, 13, 14, 15, 25),
        ),
    }
    try:
        return tables[element_type]
    except KeyError as exc:
        raise ValueError("unsupported boundary sideset cell element '%s'" % element_type) from exc


def _proteus_hex_side_nodes(order):
    L = int(order)

    def lidx(x, y, z):
        n = L + 1
        return z * n * n + y * n + x

    sides = []
    sides.append(tuple(lidx(xi, 0, zi) for zi in range(L + 1) for xi in range(L + 1)))
    sides.append(tuple(lidx(L, yi, zi) for zi in range(L + 1) for yi in range(L + 1)))
    sides.append(tuple(lidx(xi, L, zi) for zi in range(L + 1) for xi in range(L, -1, -1)))
    sides.append(tuple(lidx(0, yi, zi) for zi in range(L + 1) for yi in range(L, -1, -1)))
    sides.append(tuple(lidx(xi, yi, 0) for yi in range(L, -1, -1) for xi in range(L + 1)))
    sides.append(tuple(lidx(xi, yi, L) for yi in range(L + 1) for xi in range(L + 1)))
    return tuple(sides)


def _boundary_coefficients_from_expression_plan(system, expression_plan):
    coefficients = tuple(expression_plan.coefficients)
    if not coefficients:
        raise ValueError("boundary expression plan '%s' has no coefficients" % expression_plan.name)
    if len(coefficients) != len(system.fields):
        raise ValueError(
            "boundary expression plan '%s' has %d coefficient entries for %d fields"
            % (expression_plan.name, len(coefficients), len(system.fields))
        )
    values = []
    by_name = {field.name: field for field in system.fields}
    for coefficient in coefficients:
        row_field = getattr(coefficient, "row_field", None)
        if row_field is None:
            values.append(sp.simplify(coefficient))
            continue
        if row_field not in by_name:
            raise ValueError("boundary coefficient row field '%s' is not in the residual system" % row_field)
        gradient = tuple(getattr(coefficient, "gradient", ()))
        if any(sp.sympify(value) != 0 for value in gradient):
            raise ValueError(
                "boundary residual for field '%s' must be linear in the test value only"
                % row_field
            )
        values.append(sp.simplify(getattr(coefficient, "value")))
    return tuple(values)


def _validate_boundary_coefficients(system, coefficients):
    forbidden = set()
    for field in system.fields:
        forbidden.update(field.gradient)
        forbidden.update(field.previous_symbols)
        forbidden.update(field.direction_symbols)
        forbidden.update(field.test_symbols)
    for coeff in coefficients:
        invalid = sorted(sp.sympify(coeff).free_symbols.intersection(forbidden), key=str)
        if invalid:
            raise ValueError(
                "boundary residual coefficients cannot depend on field/test/direction symbols: %s"
                % ", ".join(map(str, invalid))
            )


def _validate_boundary_metadata(dependencies, coefficients):
    declared = set(_dependency_symbols(dependencies))
    required = set()
    for coeff in coefficients:
        required.update(sp.sympify(coeff).free_symbols)
    missing = tuple(sorted(required.difference(declared), key=str))
    if missing:
        raise ValueError(
            "boundary residual metadata does not declare coefficient dependencies: %s"
            % ", ".join(map(str, missing))
        )


def _dependency_parameters(dependencies):
    return tuple(getattr(dependencies, "parameters", ()))


def _dependency_symbols(dependencies):
    symbols = getattr(dependencies, "symbols", None)
    if symbols is not None:
        return tuple(symbols)
    ret = []
    for attr in (
        "current_symbols",
        "previous_symbols",
        "direction_symbols",
        "geometry_symbols",
        "parameters",
    ):
        ret.extend(tuple(getattr(dependencies, attr, ())))
    return tuple(dict.fromkeys(ret))


def _boundary_source(function, element_type, surface, components, parameters, coefficients, system, use_tensor_product):
    use_tensor_product = bool(use_tensor_product)
    if use_tensor_product:
        return _boundary_tensor_product_source(
            function,
            element_type,
            surface,
            components,
            parameters,
            coefficients,
            system,
        )

    data = _surface_reference_data(surface)
    side_nodes = _cell_side_nodes(element_type)
    n_shape = data["n_shape"]
    n_qp = data["n_qp"]
    ref_dim = data["ref_dim"]
    physical_dim = 2 if ref_dim == 1 else 3
    coordinate_symbols = _coefficient_coordinate_symbols(coefficients, physical_dim)
    current_symbols = _coefficient_current_symbols(system, coefficients)
    codegen_coefficients = _replace_current_symbols(coefficients, current_symbols)
    parameters = _filter_coordinate_parameters(parameters, physical_dim)
    if any(len(side) != n_shape for side in side_nodes):
        raise ValueError(
            "sideset side table for '%s' does not match surface shape count %d"
            % (element_type, n_shape)
        )
    side_node_values = tuple(node for side in side_nodes for node in side)
    sideset_function = function.replace("_boundary_residual_soa", "_boundary_residual_sideset_soa")
    param_decls = "".join(", const scalar_t %s" % parameter for parameter in parameters)
    extern_param_decls = "".join(", const real_t %s" % parameter for parameter in parameters)
    extern_float_param_decls = "".join(", const float %s" % parameter for parameter in parameters)
    param_args = "".join(", %s" % parameter for parameter in parameters)
    current_decls = _current_declarations(current_symbols, "scalar_t")
    extern_current_decls = _current_declarations(current_symbols, "real_t")
    extern_float_current_decls = _current_declarations(current_symbols, "float")
    current_args = "".join(", %s" % symbol for symbol in current_symbols)
    component_uses_coordinates = tuple(
        bool(sp.sympify(coefficient).free_symbols.intersection(coordinate_symbols))
        for coefficient in coefficients
    )
    component_uses_current = tuple(
        bool(sp.sympify(coefficient).free_symbols.intersection(current_symbols))
        for coefficient in coefficients
    )
    coeff_lines = [
        "        const scalar_t coeff%d = %s;" % (i, _sfem_ccode(codegen_coefficients[i]))
        for i in range(components)
        if not component_uses_coordinates[i] and not component_uses_current[i]
    ]
    qp_coeff_lines = _value_eval_lines(coordinate_symbols, current_symbols) + [
        "        const scalar_t coeff%d = %s;" % (i, _sfem_ccode(codegen_coefficients[i]))
        for i in range(components)
        if component_uses_coordinates[i] or component_uses_current[i]
    ]
    scatter_streams = ", ".join("out%d" % i for i in range(components))
    out_params = "\n".join(
        "        scalar_t *const SFEM_RESTRICT out%d%s" % (i, "," if i + 1 < components else "")
        for i in range(components)
    )
    extern_out_params = "\n".join(
        "        real_t *const SFEM_RESTRICT out%d%s" % (i, "," if i + 1 < components else "")
        for i in range(components)
    )
    extern_float_out_params = "\n".join(
        "        float *const SFEM_RESTRICT out%d%s" % (i, "," if i + 1 < components else "")
        for i in range(components)
    )
    return """#include "sfem_base.hpp"
#include "sfem_defs.hpp"
#include "sfem_macros.hpp"

#include <math.h>
#include "kernel_math.hpp"

namespace sfem {{
namespace codegen {{

template <typename scalar_t>
struct {function}_reference_data {{
    static constexpr int N_SHAPE = {n_shape};
    static constexpr int N_QP = {n_qp};
    static constexpr int REF_DIM = {ref_dim};
    static constexpr int PHYSICAL_DIM = {physical_dim};

    static const scalar_t *shape() {{
        static const scalar_t data[{shape_count}] = {{
{shape_values}
        }};
        return data;
    }}

    static const scalar_t *grad() {{
        static const scalar_t data[{grad_count}] = {{
{grad_values}
        }};
        return data;
    }}

    static const scalar_t *weight() {{
        static const scalar_t data[{weight_count}] = {{
{weight_values}
        }};
        return data;
    }}
}};

template <typename scalar_t>
{function_qualifier} scalar_t {function}_measure(
        const int q,
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points) {{
    const scalar_t *const grad = {function}_reference_data<scalar_t>::grad();
    const int n_shape = {function}_reference_data<scalar_t>::N_SHAPE;
{measure_body}
}}

{function_qualifier} const int *{function}_side_nodes() {{
    static const int data[{side_node_count}] = {{
{side_node_values}
    }};
    return data;
}}

{function_qualifier} void {function}_gather_sideset_element(
        const element_idx_t parent_element,
        const int side,
        idx_t **const SFEM_RESTRICT elements,
        idx_t *const SFEM_RESTRICT ev) {{
    const int *const SFEM_RESTRICT side_nodes = {function}_side_nodes();
    constexpr int n_shape = {n_shape};
    for (int i = 0; i < n_shape; ++i) {{
        ev[i] = elements[side_nodes[side * n_shape + i]][parent_element];
    }}
}}

template <typename scalar_t>
{function_qualifier} void {function}_element(
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points{current_decls}{param_decls},
        scalar_t element_vector[{components}][{n_shape}]) {{
    const scalar_t *const shape = {function}_reference_data<scalar_t>::shape();
    const scalar_t *const weight = {function}_reference_data<scalar_t>::weight();
    const int n_shape = {function}_reference_data<scalar_t>::N_SHAPE;
    const int n_qp = {function}_reference_data<scalar_t>::N_QP;

{coeff_lines}

    for (int q = 0; q < n_qp; ++q) {{
        const scalar_t dS = {function}_measure<scalar_t>(q, ev, points);
        const scalar_t qw = weight[q] * dS;
{qp_coeff_lines}
        for (int i = 0; i < n_shape; ++i) {{
            const scalar_t test = shape[q * n_shape + i] * qw;
{accum_lines}
        }}
    }}
}}

template <typename scalar_t>
{function_qualifier} void {function}_scatter_element(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t element_vector[{components}][{n_shape}],
        const int out_stride,
{out_params}) {{
    constexpr int n_shape = {n_shape};
    for (int i = 0; i < n_shape; ++i) {{
        const idx_t node = ev[i];
{scatter_lines}
    }}
}}

template <typename scalar_t>
{function_qualifier} int {function}_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points{current_decls}{param_decls},
        const int out_stride,
{out_params}) {{
{parallel_for_pragma}
    for (ptrdiff_t e = 0; e < nelements; ++e) {{
        idx_t ev[{n_shape}];
        scalar_t element_vector[{components}][{n_shape}];
        for (int i = 0; i < {n_shape}; ++i) {{
            ev[i] = elements[i][e];
        }}
        for (int c = 0; c < {components}; ++c) {{
            for (int i = 0; i < {n_shape}; ++i) {{
                element_vector[c][i] = scalar_t(0);
            }}
        }}
        {function}_element<scalar_t>(ev, points{current_args}{param_args}, element_vector);
        {function}_scatter_element<scalar_t>(ev, element_vector, out_stride, {scatter_streams});
    }}

    return SFEM_SUCCESS;
}}

template <typename scalar_t>
{function_qualifier} int {sideset_function}_impl(
        const ptrdiff_t nsides,
        const ptrdiff_t,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points{current_decls}{param_decls},
        const int out_stride,
{out_params}) {{
{parallel_for_pragma}
    for (ptrdiff_t s = 0; s < nsides; ++s) {{
        idx_t ev[{n_shape}];
        scalar_t element_vector[{components}][{n_shape}];
        {function}_gather_sideset_element(parent[s], side_idx[s], elements, ev);
        for (int c = 0; c < {components}; ++c) {{
            for (int i = 0; i < {n_shape}; ++i) {{
                element_vector[c][i] = scalar_t(0);
            }}
        }}
        {function}_element<scalar_t>(ev, points{current_args}{param_args}, element_vector);
        {function}_scatter_element<scalar_t>(ev, element_vector, out_stride, {scatter_streams});
    }}

    return SFEM_SUCCESS;
}}

}}  // namespace codegen
}}  // namespace sfem

extern "C" int {function}(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points{extern_current_decls}{extern_param_decls},
        const int out_stride,
{extern_out_params}) {{
    return sfem::codegen::{function}_impl<real_t>(
            nelements, nnodes, elements, points{current_args}{param_args}, out_stride, {scatter_streams});
}}

extern "C" int {function}_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points{extern_float_current_decls}{extern_float_param_decls},
        const int out_stride,
{extern_float_out_params}) {{
    return sfem::codegen::{function}_impl<float>(
            nelements, nnodes, elements, points{current_args}{param_args}, out_stride, {scatter_streams});
}}

extern "C" int {sideset_function}(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points{extern_current_decls}{extern_param_decls},
        const int out_stride,
{extern_out_params}) {{
    return sfem::codegen::{sideset_function}_impl<real_t>(
            nsides, nnodes, elements, parent, side_idx, points{current_args}{param_args}, out_stride, {scatter_streams});
}}

extern "C" int {sideset_function}_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points{extern_float_current_decls}{extern_float_param_decls},
        const int out_stride,
{extern_float_out_params}) {{
    return sfem::codegen::{sideset_function}_impl<float>(
            nsides, nnodes, elements, parent, side_idx, points{current_args}{param_args}, out_stride, {scatter_streams});
}}
""".format(
        function=function,
        function_qualifier=_function_qualifier(),
        parallel_for_pragma=_parallel_for_pragma(),
        sideset_function=sideset_function,
        n_shape=n_shape,
        n_qp=n_qp,
        ref_dim=ref_dim,
        physical_dim=physical_dim,
        side_node_count=len(side_node_values),
        side_node_values=_cpp_int_array_values(side_node_values),
        shape_count=len(data["shape"]),
        grad_count=len(data["grad"]),
        weight_count=len(data["weight"]),
        shape_values=_cpp_array_values(data["shape"]),
        grad_values=_cpp_array_values(data["grad"]),
        weight_values=_cpp_array_values(data["weight"]),
        measure_body=_measure_body(ref_dim, physical_dim),
        current_decls=current_decls,
        extern_current_decls=extern_current_decls,
        extern_float_current_decls=extern_float_current_decls,
        current_args=current_args,
        param_decls=param_decls,
        extern_param_decls=extern_param_decls,
        extern_float_param_decls=extern_float_param_decls,
        param_args=param_args,
        out_params=out_params,
        extern_out_params=extern_out_params,
        extern_float_out_params=extern_float_out_params,
        coeff_lines="\n".join(coeff_lines),
        qp_coeff_lines="\n".join(qp_coeff_lines),
        components=components,
        accum_lines="\n".join(
            "                element_vector[{c}][i] += coeff{c} * test;".format(c=c)
            for c in range(components)
        ),
        scatter_lines="\n".join(
            "{atomic_update_pragma}\n            out{c}[node * out_stride] += element_vector[{c}][i];".format(
                atomic_update_pragma=_atomic_update_pragma(),
                c=c,
            )
            for c in range(components)
        ),
        scatter_streams=scatter_streams,
    )


def _coordinate_symbol_tuple(physical_dim):
    return tuple(sp.Symbol("x%d" % d) for d in range(int(physical_dim)))


def _coefficient_coordinate_symbols(coefficients, physical_dim):
    coordinate_symbols = set(_coordinate_symbol_tuple(physical_dim))
    used = set()
    for coefficient in coefficients:
        used.update(sp.sympify(coefficient).free_symbols.intersection(coordinate_symbols))
    return tuple(symbol for symbol in _coordinate_symbol_tuple(physical_dim) if symbol in used)


def _filter_coordinate_parameters(parameters, physical_dim):
    coordinate_symbols = set(_coordinate_symbol_tuple(physical_dim))
    return tuple(parameter for parameter in parameters if parameter not in coordinate_symbols)


def _coefficient_current_symbols(system, coefficients):
    current_symbols = tuple(field.value for field in system.fields)
    used = set()
    for coefficient in coefficients:
        used.update(sp.sympify(coefficient).free_symbols.intersection(current_symbols))
    return tuple(symbol for symbol in current_symbols if symbol in used)


def _replace_current_symbols(coefficients, current_symbols):
    substitutions = {
        symbol: sp.Symbol("%s_q" % symbol)
        for symbol in current_symbols
    }
    if not substitutions:
        return tuple(coefficients)
    return tuple(sp.sympify(coefficient).xreplace(substitutions) for coefficient in coefficients)


def _current_declarations(current_symbols, scalar_type):
    return "".join(
        ", const %s *const SFEM_RESTRICT %s" % (scalar_type, symbol)
        for symbol in current_symbols
    )


def _value_eval_lines(coordinate_symbols, current_symbols):
    if not coordinate_symbols and not current_symbols:
        return []
    lines = [
        "        scalar_t %s = scalar_t(0);" % symbol
        for symbol in coordinate_symbols
    ]
    lines.extend(
        "        scalar_t %s_q = scalar_t(0);" % symbol
        for symbol in current_symbols
    )
    lines.extend(
        [
            "        for (int j = 0; j < n_shape; ++j) {",
            "            const scalar_t phi = shape[q * n_shape + j];",
            "            const idx_t node = ev[j];",
        ]
    )
    for symbol in coordinate_symbols:
        component = int(str(symbol)[1:])
        lines.append(
            "            %s += scalar_t(points[%d][node]) * phi;" % (symbol, component)
        )
    for symbol in current_symbols:
        lines.append("            %s_q += %s[node] * phi;" % (symbol, symbol))
    lines.append("        }")
    return lines


def _tensor_value_eval_lines(coordinate_symbols, current_symbols):
    if not coordinate_symbols and not current_symbols:
        return []
    lines = [
        "            scalar_t %s = scalar_t(0);" % symbol
        for symbol in coordinate_symbols
    ]
    lines.extend(
        "            scalar_t %s_q = scalar_t(0);" % symbol
        for symbol in current_symbols
    )
    lines.extend(
        [
            "            for (int cy = 0; cy < S; ++cy) {",
            "                const scalar_t vy_coord = shape_1d[qy * S + cy];",
            "                for (int cx = 0; cx < S; ++cx) {",
            "                    const int j = shape_index[cy * S + cx];",
            "                    const idx_t node = ev[j];",
            "                    const scalar_t phi = shape_1d[qx * S + cx] * vy_coord;",
        ]
    )
    for symbol in coordinate_symbols:
        component = int(str(symbol)[1:])
        lines.append(
            "                    %s += scalar_t(points[%d][node]) * phi;"
            % (symbol, component)
        )
    for symbol in current_symbols:
        lines.append("                    %s_q += %s[node] * phi;" % (symbol, symbol))
    lines.extend(
        [
            "                }",
            "            }",
        ]
    )
    return lines


def _boundary_tensor_product_source(function, element_type, surface, components, parameters, coefficients, system):
    proteus = surface.startswith("PROTEUS_QUADSHELL")
    if proteus:
        n_shape = int(surface.replace("PROTEUS_QUADSHELL", ""))
        n_shape_1d = int(round(math.sqrt(n_shape)))
    elif surface == "QUADSHELL4":
        n_shape_1d = 2
    elif surface == "QUADSHELL9":
        n_shape_1d = 3
    else:
        raise ValueError("unsupported tensor-product boundary surface '%s'" % surface)

    data = _quad_tensor_reference_data(n_shape_1d, proteus)
    side_nodes = _cell_side_nodes(element_type)
    n_shape = n_shape_1d * n_shape_1d
    if any(len(side) != n_shape for side in side_nodes):
        raise ValueError(
            "sideset side table for '%s' does not match surface shape count %d"
            % (element_type, n_shape)
        )
    side_node_values = tuple(node for side in side_nodes for node in side)
    sideset_function = function.replace("_boundary_residual_soa", "_boundary_residual_sideset_soa")
    coordinate_symbols = _coefficient_coordinate_symbols(coefficients, 3)
    current_symbols = _coefficient_current_symbols(system, coefficients)
    codegen_coefficients = _replace_current_symbols(coefficients, current_symbols)
    parameters = _filter_coordinate_parameters(parameters, 3)
    param_decls = "".join(", const scalar_t %s" % parameter for parameter in parameters)
    extern_param_decls = "".join(", const real_t %s" % parameter for parameter in parameters)
    extern_float_param_decls = "".join(", const float %s" % parameter for parameter in parameters)
    param_args = "".join(", %s" % parameter for parameter in parameters)
    current_decls = _current_declarations(current_symbols, "scalar_t")
    extern_current_decls = _current_declarations(current_symbols, "real_t")
    extern_float_current_decls = _current_declarations(current_symbols, "float")
    current_args = "".join(", %s" % symbol for symbol in current_symbols)
    component_uses_coordinates = tuple(
        bool(sp.sympify(coefficient).free_symbols.intersection(coordinate_symbols))
        for coefficient in coefficients
    )
    component_uses_current = tuple(
        bool(sp.sympify(coefficient).free_symbols.intersection(current_symbols))
        for coefficient in coefficients
    )
    coeff_lines = [
        "    const scalar_t coeff%d = %s;" % (i, _sfem_ccode(codegen_coefficients[i]))
        for i in range(components)
        if not component_uses_coordinates[i] and not component_uses_current[i]
    ]
    qp_coeff_lines = _tensor_value_eval_lines(coordinate_symbols, current_symbols) + [
        "            const scalar_t coeff%d = %s;" % (i, _sfem_ccode(codegen_coefficients[i]))
        for i in range(components)
        if component_uses_coordinates[i] or component_uses_current[i]
    ]
    scatter_streams = ", ".join("out%d" % i for i in range(components))
    out_params = "\n".join(
        "        scalar_t *const SFEM_RESTRICT out%d%s" % (i, "," if i + 1 < components else "")
        for i in range(components)
    )
    extern_out_params = "\n".join(
        "        real_t *const SFEM_RESTRICT out%d%s" % (i, "," if i + 1 < components else "")
        for i in range(components)
    )
    extern_float_out_params = "\n".join(
        "        float *const SFEM_RESTRICT out%d%s" % (i, "," if i + 1 < components else "")
        for i in range(components)
    )
    return """#include "sfem_base.hpp"
#include "sfem_defs.hpp"
#include "sfem_macros.hpp"

#include <math.h>
#include "kernel_math.hpp"

namespace sfem {{
namespace codegen {{

template <typename scalar_t>
struct {function}_reference_data {{
    static constexpr int N_SHAPE_1D = {n_shape_1d};
    static constexpr int N_QP_1D = {n_qp_1d};
    static constexpr int N_SHAPE = {n_shape};
    static constexpr int N_QP = {n_qp};
    static constexpr int REF_DIM = 2;
    static constexpr int PHYSICAL_DIM = 3;

    static const scalar_t *shape_1d() {{
        static const scalar_t data[{shape_1d_count}] = {{
{shape_1d_values}
        }};
        return data;
    }}

    static const scalar_t *grad_1d() {{
        static const scalar_t data[{grad_1d_count}] = {{
{grad_1d_values}
        }};
        return data;
    }}

    static const scalar_t *weight_1d() {{
        static const scalar_t data[{weight_1d_count}] = {{
{weight_1d_values}
        }};
        return data;
    }}

    static const int *shape_index() {{
        static const int data[{n_shape}] = {{
{shape_index_values}
        }};
        return data;
    }}
}};

template <typename scalar_t>
{function_qualifier} scalar_t {function}_measure(
        const int qx,
        const int qy,
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points) {{
    const scalar_t *const SFEM_RESTRICT shape_1d = {function}_reference_data<scalar_t>::shape_1d();
    const scalar_t *const SFEM_RESTRICT grad_1d = {function}_reference_data<scalar_t>::grad_1d();
    const int *const SFEM_RESTRICT shape_index = {function}_reference_data<scalar_t>::shape_index();
    constexpr int S = {n_shape_1d};
    scalar_t dxdr0 = scalar_t(0);
    scalar_t dxdr1 = scalar_t(0);
    scalar_t dxdr2 = scalar_t(0);
    scalar_t dxds0 = scalar_t(0);
    scalar_t dxds1 = scalar_t(0);
    scalar_t dxds2 = scalar_t(0);
    for (int sy = 0; sy < S; ++sy) {{
        const scalar_t vy = shape_1d[qy * S + sy];
        const scalar_t gy = grad_1d[qy * S + sy];
        for (int sx = 0; sx < S; ++sx) {{
            const int i = shape_index[sy * S + sx];
            const idx_t node = ev[i];
            const scalar_t vx = shape_1d[qx * S + sx];
            const scalar_t gx = grad_1d[qx * S + sx];
            const scalar_t gr = gx * vy;
            const scalar_t gs = vx * gy;
            const scalar_t x = scalar_t(points[0][node]);
            const scalar_t y = scalar_t(points[1][node]);
            const scalar_t z = scalar_t(points[2][node]);
            dxdr0 += x * gr;
            dxdr1 += y * gr;
            dxdr2 += z * gr;
            dxds0 += x * gs;
            dxds1 += y * gs;
            dxds2 += z * gs;
        }}
    }}
    const scalar_t c0 = dxdr1 * dxds2 - dxdr2 * dxds1;
    const scalar_t c1 = dxdr2 * dxds0 - dxdr0 * dxds2;
    const scalar_t c2 = dxdr0 * dxds1 - dxdr1 * dxds0;
    return sqrt(c0 * c0 + c1 * c1 + c2 * c2);
}}

{function_qualifier} const int *{function}_side_nodes() {{
    static const int data[{side_node_count}] = {{
{side_node_values}
    }};
    return data;
}}

{function_qualifier} void {function}_gather_sideset_element(
        const element_idx_t parent_element,
        const int side,
        idx_t **const SFEM_RESTRICT elements,
        idx_t *const SFEM_RESTRICT ev) {{
    const int *const SFEM_RESTRICT side_nodes = {function}_side_nodes();
    constexpr int n_shape = {n_shape};
    for (int i = 0; i < n_shape; ++i) {{
        ev[i] = elements[side_nodes[side * n_shape + i]][parent_element];
    }}
}}

template <typename scalar_t>
{function_qualifier} void {function}_element(
        const idx_t *const SFEM_RESTRICT ev,
        const geom_t *const *const SFEM_RESTRICT points{current_decls}{param_decls},
        scalar_t element_vector[{components}][{n_shape}]) {{
    const scalar_t *const SFEM_RESTRICT shape_1d = {function}_reference_data<scalar_t>::shape_1d();
    const scalar_t *const SFEM_RESTRICT weight_1d = {function}_reference_data<scalar_t>::weight_1d();
    const int *const SFEM_RESTRICT shape_index = {function}_reference_data<scalar_t>::shape_index();
    constexpr int S = {n_shape_1d};
    constexpr int Q = {n_qp_1d};

{coeff_lines}

    for (int qy = 0; qy < Q; ++qy) {{
        for (int qx = 0; qx < Q; ++qx) {{
            const scalar_t dS = {function}_measure<scalar_t>(qx, qy, ev, points);
            const scalar_t qw = weight_1d[qx] * weight_1d[qy] * dS;
{qp_coeff_lines}
            for (int sy = 0; sy < S; ++sy) {{
                const scalar_t vy = shape_1d[qy * S + sy];
                for (int sx = 0; sx < S; ++sx) {{
                    const int i = shape_index[sy * S + sx];
                    const scalar_t test = shape_1d[qx * S + sx] * vy * qw;
{accum_lines}
                }}
            }}
        }}
    }}
}}

template <typename scalar_t>
{function_qualifier} void {function}_scatter_element(
        const idx_t *const SFEM_RESTRICT ev,
        const scalar_t element_vector[{components}][{n_shape}],
        const int out_stride,
{out_params}) {{
    constexpr int n_shape = {n_shape};
    for (int i = 0; i < n_shape; ++i) {{
        const idx_t node = ev[i];
{scatter_lines}
    }}
}}

template <typename scalar_t>
{function_qualifier} int {function}_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points{current_decls}{param_decls},
        const int out_stride,
{out_params}) {{
{parallel_for_pragma}
    for (ptrdiff_t e = 0; e < nelements; ++e) {{
        idx_t ev[{n_shape}];
        scalar_t element_vector[{components}][{n_shape}];
        for (int i = 0; i < {n_shape}; ++i) {{
            ev[i] = elements[i][e];
        }}
        for (int c = 0; c < {components}; ++c) {{
            for (int i = 0; i < {n_shape}; ++i) {{
                element_vector[c][i] = scalar_t(0);
            }}
        }}
        {function}_element<scalar_t>(ev, points{current_args}{param_args}, element_vector);
        {function}_scatter_element<scalar_t>(ev, element_vector, out_stride, {scatter_streams});
    }}

    return SFEM_SUCCESS;
}}

template <typename scalar_t>
{function_qualifier} int {sideset_function}_impl(
        const ptrdiff_t nsides,
        const ptrdiff_t,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points{current_decls}{param_decls},
        const int out_stride,
{out_params}) {{
{parallel_for_pragma}
    for (ptrdiff_t s = 0; s < nsides; ++s) {{
        idx_t ev[{n_shape}];
        scalar_t element_vector[{components}][{n_shape}];
        {function}_gather_sideset_element(parent[s], side_idx[s], elements, ev);
        for (int c = 0; c < {components}; ++c) {{
            for (int i = 0; i < {n_shape}; ++i) {{
                element_vector[c][i] = scalar_t(0);
            }}
        }}
        {function}_element<scalar_t>(ev, points{current_args}{param_args}, element_vector);
        {function}_scatter_element<scalar_t>(ev, element_vector, out_stride, {scatter_streams});
    }}

    return SFEM_SUCCESS;
}}

}}  // namespace codegen
}}  // namespace sfem

extern "C" int {function}(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points{extern_current_decls}{extern_param_decls},
        const int out_stride,
{extern_out_params}) {{
    return sfem::codegen::{function}_impl<real_t>(
            nelements, nnodes, elements, points{current_args}{param_args}, out_stride, {scatter_streams});
}}

extern "C" int {function}_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points{extern_float_current_decls}{extern_float_param_decls},
        const int out_stride,
{extern_float_out_params}) {{
    return sfem::codegen::{function}_impl<float>(
            nelements, nnodes, elements, points{current_args}{param_args}, out_stride, {scatter_streams});
}}

extern "C" int {sideset_function}(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points{extern_current_decls}{extern_param_decls},
        const int out_stride,
{extern_out_params}) {{
    return sfem::codegen::{sideset_function}_impl<real_t>(
            nsides, nnodes, elements, parent, side_idx, points{current_args}{param_args}, out_stride, {scatter_streams});
}}

extern "C" int {sideset_function}_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const element_idx_t *const SFEM_RESTRICT parent,
        const int16_t *const SFEM_RESTRICT side_idx,
        const geom_t *const *const SFEM_RESTRICT points{extern_float_current_decls}{extern_float_param_decls},
        const int out_stride,
{extern_float_out_params}) {{
    return sfem::codegen::{sideset_function}_impl<float>(
            nsides, nnodes, elements, parent, side_idx, points{current_args}{param_args}, out_stride, {scatter_streams});
}}
""".format(
        function=function,
        function_qualifier=_function_qualifier(),
        parallel_for_pragma=_parallel_for_pragma(),
        sideset_function=sideset_function,
        n_shape=n_shape,
        n_shape_1d=n_shape_1d,
        n_qp_1d=len(data["weight_1d"]),
        n_qp=len(data["weight_1d"]) * len(data["weight_1d"]),
        shape_1d_count=len(data["shape_1d"]),
        grad_1d_count=len(data["grad_1d"]),
        weight_1d_count=len(data["weight_1d"]),
        shape_1d_values=_cpp_array_values(data["shape_1d"]),
        grad_1d_values=_cpp_array_values(data["grad_1d"]),
        weight_1d_values=_cpp_array_values(data["weight_1d"]),
        shape_index_values=_cpp_int_array_values(data["shape_index"]),
        side_node_count=len(side_node_values),
        side_node_values=_cpp_int_array_values(side_node_values),
        current_decls=current_decls,
        extern_current_decls=extern_current_decls,
        extern_float_current_decls=extern_float_current_decls,
        current_args=current_args,
        param_decls=param_decls,
        extern_param_decls=extern_param_decls,
        extern_float_param_decls=extern_float_param_decls,
        param_args=param_args,
        out_params=out_params,
        extern_out_params=extern_out_params,
        extern_float_out_params=extern_float_out_params,
        coeff_lines="\n".join(coeff_lines),
        qp_coeff_lines="\n".join(qp_coeff_lines),
        components=components,
        accum_lines="\n".join(
            "                    element_vector[{c}][i] += coeff{c} * test;".format(c=c)
            for c in range(components)
        ),
        scatter_lines="\n".join(
            "{atomic_update_pragma}\n            out{c}[node * out_stride] += element_vector[{c}][i];".format(
                atomic_update_pragma=_atomic_update_pragma(),
                c=c,
            )
            for c in range(components)
        ),
        scatter_streams=scatter_streams,
    )


def _surface_reference_data(surface):
    if surface in ("EDGESHELL2", "EDGE2", "BEAM2"):
        return _edge_reference_data(1)
    if surface == "TRISHELL3":
        return _tri_reference_data(1)
    if surface == "TRISHELL6":
        return _tri_reference_data(2)
    if surface.startswith("PROTEUS_QUADSHELL"):
        n_shape = int(surface.replace("PROTEUS_QUADSHELL", ""))
        return _quad_reference_data(int(round(math.sqrt(n_shape))), proteus=True)
    if surface == "QUADSHELL4":
        return _quad_reference_data(2, proteus=False)
    if surface == "QUADSHELL9":
        return _quad_reference_data(3, proteus=False)
    raise ValueError("unsupported boundary surface element '%s'" % surface)


def _edge_reference_data(order):
    points, weights = _sfem_unit_interval_gauss_rule(order + 1)
    shape = []
    grad = []
    for x in points:
        values, gradients = _sfem_lagrange_1d_at(x, order)
        shape.extend(values)
        grad.extend(gradients)
    return {"n_shape": order + 1, "n_qp": len(weights), "ref_dim": 1, "shape": tuple(shape), "grad": tuple(grad), "weight": tuple(weights)}


def _tri_reference_data(order):
    points, weights = _sfem_triangle_quadrature_rule(2 * order)
    shape = []
    grad = []
    for x, y in points:
        if order == 1:
            shape.extend((1.0 - x - y, x, y))
            grad.extend((-1.0, -1.0, 1.0, 0.0, 0.0, 1.0))
        else:
            l0 = 1.0 - x - y
            shape.extend((l0 * (2.0 * l0 - 1.0), x * (2.0 * x - 1.0), y * (2.0 * y - 1.0), 4.0 * x * l0, 4.0 * x * y, 4.0 * y * l0))
            grad.extend(_tri6_reference_gradients(x, y))
    return {"n_shape": 3 if order == 1 else 6, "n_qp": len(weights), "ref_dim": 2, "shape": tuple(shape), "grad": tuple(grad), "weight": tuple(weights)}


def _quad_reference_data(n, proteus):
    order = n - 1
    points, weights_1d = _sfem_unit_interval_gauss_rule(order + 1)
    shape = []
    grad = []
    weights = []
    for qy, wy in zip(points, weights_1d):
        vy, gy = _sfem_lagrange_1d_at(qy, order)
        for qx, wx in zip(points, weights_1d):
            vx, gx = _sfem_lagrange_1d_at(qx, order)
            shape_q = [0.0] * (n * n)
            grad_q = [(0.0, 0.0)] * (n * n)
            for sy in range(n):
                for sx in range(n):
                    idx = _quad_shape_index(n, sx, sy, proteus)
                    shape_q[idx] = vx[sx] * vy[sy]
                    grad_q[idx] = (gx[sx] * vy[sy], vx[sx] * gy[sy])
            weights.append(wx * wy)
            shape.extend(shape_q)
            for gx_q, gy_q in grad_q:
                grad.extend((gx_q, gy_q))
    return {"n_shape": n * n, "n_qp": len(weights), "ref_dim": 2, "shape": tuple(shape), "grad": tuple(grad), "weight": tuple(weights)}


def _quad_tensor_reference_data(n, proteus):
    order = n - 1
    points, weights_1d = _sfem_unit_interval_gauss_rule(order + 1)
    shape = []
    grad = []
    for x in points:
        values, gradients = _sfem_lagrange_1d_at(x, order)
        shape.extend(values)
        grad.extend(gradients)
    shape_index = []
    for sy in range(n):
        for sx in range(n):
            shape_index.append(_quad_shape_index(n, sx, sy, proteus))
    return {
        "shape_1d": tuple(shape),
        "grad_1d": tuple(grad),
        "weight_1d": tuple(weights_1d),
        "shape_index": tuple(shape_index),
    }


def _quad_shape_index(n, sx, sy, proteus):
    if proteus:
        return sx + n * sy
    if n == 2:
        return (0, 1, 3, 2)[sx + 2 * sy]
    if n == 3:
        cartesian_to_quad9 = (0, 4, 1, 7, 8, 5, 3, 6, 2)
        return cartesian_to_quad9[sx + 3 * sy]
    return sx + n * sy


def _measure_body(ref_dim, physical_dim):
    if ref_dim == 1:
        return """    scalar_t dx0 = scalar_t(0);
    scalar_t dx1 = scalar_t(0);
    for (int i = 0; i < n_shape; ++i) {
        const scalar_t gi = grad[q * n_shape + i];
        const idx_t node = ev[i];
        dx0 += scalar_t(points[0][node]) * gi;
        dx1 += scalar_t(points[1][node]) * gi;
    }
    return sqrt(dx0 * dx0 + dx1 * dx1);"""
    return """    scalar_t dxdr0 = scalar_t(0);
    scalar_t dxdr1 = scalar_t(0);
    scalar_t dxdr2 = scalar_t(0);
    scalar_t dxds0 = scalar_t(0);
    scalar_t dxds1 = scalar_t(0);
    scalar_t dxds2 = scalar_t(0);
    for (int i = 0; i < n_shape; ++i) {
        const scalar_t gr = grad[(q * n_shape + i) * 2 + 0];
        const scalar_t gs = grad[(q * n_shape + i) * 2 + 1];
        const idx_t node = ev[i];
        const scalar_t x = scalar_t(points[0][node]);
        const scalar_t y = scalar_t(points[1][node]);
        const scalar_t z = scalar_t(points[2][node]);
        dxdr0 += x * gr;
        dxdr1 += y * gr;
        dxdr2 += z * gr;
        dxds0 += x * gs;
        dxds1 += y * gs;
        dxds2 += z * gs;
    }
    const scalar_t c0 = dxdr1 * dxds2 - dxdr2 * dxds1;
    const scalar_t c1 = dxdr2 * dxds0 - dxdr0 * dxds2;
    const scalar_t c2 = dxdr0 * dxds1 - dxdr1 * dxds0;
    return sqrt(c0 * c0 + c1 * c1 + c2 * c2);"""


def _cpp_array_values(values):
    return ",\n".join("            scalar_t(%.17g)" % float(value) for value in values)


def _cpp_int_array_values(values):
    return ",\n".join("        %d" % int(value) for value in values)
