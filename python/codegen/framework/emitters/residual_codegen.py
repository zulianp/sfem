from dataclasses import dataclass

import sympy as sp

from codegen.framework.symbolic.residual import CoupledResidualSystem
from codegen.framework.fem.tensor_product_geometry import (
    isoparametric_adjugate_call_lines,
    isoparametric_adjugate_stream_array_lines,
    sfem_geometry_kernels_header_source,
    streams_in_shape_order,
    tensor_product_cartesian_shape_order,
    tensor_product_evaluated_isoparametric_geometry_lines,
    tensor_product_ordered_coordinate_streams,
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
)


def _target():
    return OpenMPTarget()


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
    lines = []
    for stream in streams:
        lines.extend(
            [
                "%sscalar_t block_%s_data[VECTOR_SIZE];" % (indent, stream),
                "%sconst scalar_t *const block_%s = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>("
                % (indent, stream),
                "%s        nelems, g_%s + evbegin, block_%s_data, std::is_same<jacobian_t, scalar_t>());"
                % (indent, stream, stream),
            ]
        )
    return lines


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


def _zero_block_output_lines(name, n_streams, indent):
    lines = [*_work_item_loop_lines(indent)]
    for stream in range(n_streams):
        lines.append("%s    %s[%d][lane] = scalar_t(0);" % (indent, name, stream))
    lines.append("%s}" % indent)
    return lines


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


def _mixed_block_stream_pointer_lines(layout, dependencies, indent):
    lines = []
    for group in _dependency_stream_groups(dependencies):
        lines.append(
            "%sconst scalar_t *const block_%s_streams[N_FIELD_STREAMS] = {%s};"
            % (
                indent,
                group.name,
                _indexed_stream_initializer(
                    "block_%s" % group.name, layout.total_streams
                ),
            )
        )
    lines.append(
        "%sscalar_t *const block_output_streams[N_FIELD_STREAMS] = {%s};"
        % (indent, _indexed_stream_initializer("block_output", layout.total_streams))
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

    element_type = {3: "TET4"}.get(int(getattr(rule, "dim", 0)))
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
    if element_type != "tet4":
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
            "%s_jacobian_action_block" % local_prefix,
            specialization,
            field_element_types,
            action_coeffs,
            dependencies=action_dependencies,
            basis_family=basis_family,
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
        params.append(
            "const scalar_t *const SFEM_RESTRICT current[%d]" % layout.total_streams
        )
    if dependencies.previous:
        params.append(
            "const scalar_t *const SFEM_RESTRICT previous[%d]" % layout.total_streams
        )
    if dependencies.direction:
        params.append(
            "const scalar_t *const SFEM_RESTRICT direction[%d]" % layout.total_streams
        )
    params.extend(
        "const scalar_t %s" % parameter for parameter in dependencies.parameters
    )
    params.append(
        "scalar_t *const SFEM_RESTRICT output[%d]" % layout.total_streams
    )
    lines = [
        "template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>",
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


def _mixed_tensor_local_body(system, layout, coefficients, dependencies):
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
            lines.append(
                "    const scalar_t *const %s_%s_streams[%s] = {%s};"
                % (
                    group.name,
                    field.name,
                    shape_name,
                    _field_stream_initializer(layout, field_index, group.name),
                )
            )
            if group.uses_gradient:
                lines.extend(
                    [
                        "    tensor_evaluate<scalar_t, N_QP, %s, VECTOR_SIZE, DIM, 1>("
                        % shape_name,
                        "            nelems, field_shape_1d[%d], field_grad_1d[%d], %s_%s_streams, %s_%s_value, %s_%s_grad_ref);"
                        % (
                            reference_index,
                            reference_index,
                            group.name,
                            field.name,
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
                        "    tensor_evaluate_value<scalar_t, N_QP, %s, VECTOR_SIZE, DIM, 1>("
                        % shape_name,
                        "            nelems, field_shape_1d[%d], %s_%s_streams, %s_%s_value);"
                        % (reference_index, group.name, field.name, group.name, field.name),
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
        lines.append(
            "    scalar_t *const %s_output_streams[%s] = {%s};"
            % (
                field.name,
                shape_name,
                _indexed_stream_range_initializer(
                    "output", offset, layout.n_shape(row)
                ),
            )
        )
        if dependencies.uses_test_gradients:
            lines.extend(
                [
                    "    tensor_integrate<scalar_t, N_QP, %s, VECTOR_SIZE, DIM, 1>("
                    % shape_name,
                    "            nelems, field_shape_1d[%d], field_grad_1d[%d], %s_value_coeff, %s_grad_coeff_ref, %s_output_streams);"
                    % (reference_index, reference_index, field.name, field.name, field.name),
                ]
            )
        else:
            lines.extend(
                [
                    "    tensor_integrate_value<scalar_t, N_QP, %s, VECTOR_SIZE, DIM, 1>("
                    % shape_name,
                    "            nelems, field_shape_1d[%d], %s_value_coeff, %s_output_streams);"
                    % (reference_index, field.name, field.name),
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
    if scale == "1":
        metric_factor = "q_weight[q]"
    else:
        metric_factor = "q_weight[q] * (%s)" % scale
    lines.append("            const ptrdiff_t geometry_offset = q * geometry_stride + lane;")
    for left in range(dim):
        for right in range(left, dim):
            lines.append(
                "            const scalar_t geom_metric%d%d = (%s) * geom_metric[%d][geometry_offset];"
                % (
                    left,
                    right,
                    metric_factor,
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


def _tensor_local_body(system, prefix, coefficients, dependencies):
    dim = system.dim
    n_fields = len(system.fields)
    uses_determinant = any(dependencies.value_coefficients) or dependencies.uses_adjugate
    uses_geometry_offset = uses_determinant or dependencies.uses_adjugate
    lines = []
    if not dependencies.uses_test_coefficients:
        return lines
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
                    "    tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(",
                    "            nelems, shape_1d, grad_1d, %s, %s_value, %s_grad_ref);"
                    % (group, group, group),
                ]
            )
        else:
            lines.extend(
                [
                    "    tensor_evaluate_value<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(",
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
                "    tensor_integrate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(",
                "            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);",
            ]
        )
    else:
        lines.extend(
            [
                "    tensor_integrate_value<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(",
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
        '#include "%s"' % local_name,
        '#include "geometry_kernels.hpp"',
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
    lines.extend(
        [
            "",
            _parallel_for_pragma("static"),
            "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            "        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
            "        idx_t ev[VECTOR_SIZE * CELL_N_SHAPE];",
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
            "",
            *_work_item_loop_lines("        "),
        ]
    )
    for shape in range(cell_rule.n_shape):
        lines.append(
            "            ev[%d * VECTOR_SIZE + lane] = elements[%d][evbegin + lane];"
            % (shape, shape)
        )
    lines.append("        }")
    dependency_groups = tuple(_dependency_stream_groups(dependencies, mesh=True))
    if dependency_groups:
        lines.extend(["", *_work_item_loop_lines("        ")])
    for field_index, field in enumerate(system.fields):
        for local_shape in range(layout.n_shape(field_index)):
            stream = layout.stream_index(field_index, local_shape)
            node = "ev[%d * VECTOR_SIZE + lane]" % local_shape
            for group in dependency_groups:
                lines.append(
                    "            block_%s[%d][lane] = %s[%s * %s];"
                    % (
                        group.name,
                        stream,
                        _mixed_mesh_field_pointer(field, group),
                        node,
                        group.stride,
                    )
                )
    if dependency_groups:
        lines.append("        }")
    lines.extend(["", *_zero_block_output_lines("block_output", layout.total_streams, "        ")])
    lines.extend(
        _affine_geometry_stream_conversion_lines(
            (
                tuple("jacobian_adjugate%d" % i for i in range(dim * dim))
                if dependencies.uses_adjugate
                else ()
            )
            + ("jacobian_determinant0",),
            "        ",
        )
    )
    if dependencies.uses_adjugate:
        lines.append(
            "        const scalar_t *const block_adjugate[DIM * DIM] = {%s};"
            % ", ".join("block_jacobian_adjugate%d" % i for i in range(dim * dim))
        )
    lines.extend(_mixed_block_stream_pointer_lines(layout, dependencies, "        "))
    call_args = [
        "nelems",
        "0",
        "block_jacobian_determinant0",
    ]
    if dependencies.uses_adjugate:
        call_args.append("block_adjugate")
    call_args.extend(_mixed_reference_call_args(cell_rule, dependencies, reference_data, basis_family))
    call_args.extend(
        "block_%s_streams" % group.name
        for group in _dependency_stream_groups(dependencies)
    )
    call_args.extend(map(str, dependencies.parameters))
    call_args.append("block_output_streams")
    lines.extend(
        [
            "",
            "        %s<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(%s);"
            % (block, ", ".join(call_args)),
            "",
        ]
    )
    for field_index, field in enumerate(system.fields):
        for local_shape in range(layout.n_shape(field_index)):
            stream = layout.stream_index(field_index, local_shape)
            lines.extend(
                _direct_atomic_scatter_lines(
                    _mixed_mesh_output_pointer(field),
                    "ev[%d * VECTOR_SIZE + %%s] * out_stride" % local_shape,
                    "block_output[%d][%%s]" % stream,
                    "        ",
                )
            )
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
    lines.extend(
        [
            _parallel_for_pragma("static"),
            "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            "        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
            "        idx_t ev[VECTOR_SIZE * CELL_N_SHAPE];",
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
            "",
            *_work_item_loop_lines("        "),
        ]
    )
    for shape in range(cell_rule.n_shape):
        lines.append(
            "            ev[%d * VECTOR_SIZE + lane] = elements[%d][evbegin + lane];"
            % (shape, shape)
        )
    lines.append("        }")
    dependency_groups = tuple(_dependency_stream_groups(dependencies, mesh=True))
    lines.extend(["", *_work_item_loop_lines("        ")])
    for shape in range(cell_rule.n_shape):
        node = "ev[%d * VECTOR_SIZE + lane]" % shape
        for d in range(dim):
            lines.append(
                "            block_coordinates[%d][lane] = points[%d][%s];"
                % (shape * dim + d, d, node)
            )
    for field_index, field in enumerate(system.fields):
        for local_shape in range(layout.n_shape(field_index)):
            stream = layout.stream_index(field_index, local_shape)
            node = "ev[%d * VECTOR_SIZE + lane]" % local_shape
            for group in dependency_groups:
                lines.append(
                    "            block_%s[%d][lane] = %s[%s * %s];"
                    % (
                        group.name,
                        stream,
                        _mixed_mesh_field_pointer(field, group),
                        node,
                        group.stride,
                    )
                )
    lines.append("        }")
    lines.extend(["", *_zero_block_output_lines("block_output", layout.total_streams, "        ")])
    if tensor_product_geometry:
        lines.append("")
        lines.extend(
            tensor_product_evaluated_isoparametric_geometry_lines(
                dim=dim,
                n_shape=cell_rule.n_shape,
                n_qp=cell_rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams=tensor_product_ordered_coordinate_streams(
                    dim,
                    cell_rule.n_shape,
                    tuple(range(dim * cell_rule.n_shape)),
                    lambda stream: "block_coordinates[%d]" % stream,
                    shape_order=tuple(range(cell_rule.n_shape))
                    if sfem_tensor_product_hex_uses_cartesian_ordering(cell_rule.element_type)
                    else None,
                ),
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
    lines.extend(_mixed_block_stream_pointer_lines(layout, dependencies, "        "))
    call_args = [
        "nelems",
        "VECTOR_SIZE",
        "block_determinant",
    ]
    if dependencies.uses_adjugate:
        call_args.append("block_adjugate")
    call_args.extend(_mixed_reference_call_args(cell_rule, dependencies, reference_data, basis_family))
    call_args.extend(
        "block_%s_streams" % group.name
        for group in _dependency_stream_groups(dependencies)
    )
    call_args.extend(map(str, dependencies.parameters))
    call_args.append("block_output_streams")
    lines.extend(
        [
            "",
            "        %s<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(%s);"
            % (block, ", ".join(call_args)),
            "",
        ]
    )
    for field_index, field in enumerate(system.fields):
        for local_shape in range(layout.n_shape(field_index)):
            stream = layout.stream_index(field_index, local_shape)
            lines.extend(
                _direct_atomic_scatter_lines(
                    _mixed_mesh_output_pointer(field),
                    "ev[%d * VECTOR_SIZE + %%s] * out_stride" % local_shape,
                    "block_output[%d][%%s]" % stream,
                    "        ",
                )
            )
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
        if element_type == "QUAD4" or sfem_is_tensor_product_hex_element(element_type)
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
    lines.extend(
        _mesh_reference_alias_lines(
            prefix,
            rule,
            "affine",
            emit_reference_basis=not omit_simplex_reference_basis_inputs,
        )
    )
    lines.extend(
        [
            "",
            _parallel_for_pragma("static"),
            "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            "        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
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
    if gradient_metric is not None and not uses_cached_affine_metric:
        lines.append(
            "        scalar_t block_geom_metric_data[%d][VECTOR_SIZE];"
            % gradient_metric.metric_components
        )
    lines.extend(
        [
            "        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];",
            "",
            *_work_item_loop_lines("        "),
        ]
    )
    for shape in range(n_shape):
        lines.append(
            "            ev[%d * VECTOR_SIZE + lane] = elements[%d][evbegin + lane];"
            % (shape, shape)
        )
    lines.append("        }")
    if dependencies.current or dependencies.previous or dependencies.direction:
        lines.extend(["", *_work_item_loop_lines("        ")])
        for shape in range(n_shape):
            for field_index, field in enumerate(system.fields):
                stream = shape * n_fields + field_index
                node = "ev[%d * VECTOR_SIZE + lane]" % shape
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
        lines.append("        }")
    lines.extend(["", *_zero_block_output_lines("block_output", n_fields * n_shape, "        "), ""])
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
    lines.extend(
        _affine_geometry_stream_conversion_lines(
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
            + (() if uses_cached_affine_metric else ("jacobian_determinant0",)),
            "        ",
        )
    )
    if dependencies.uses_adjugate and not uses_cached_affine_metric:
        lines.append(
            "        const scalar_t *const block_adjugate[%d] = {%s};"
            % (
                dim * dim,
                ", ".join(
                    "block_jacobian_adjugate%d" % i
                    for i in range(dim * dim)
                ),
            )
        )
    if uses_cached_affine_metric:
        lines.append(
            "        const scalar_t *const block_geom_metric[%d] = {%s};"
            % (
                gradient_metric.metric_components,
                _numbered_geometry_metric_stream_initializer(
                    "block_geom_metric",
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
                "block_jacobian_determinant0[lane]",
                lambda component: "block_jacobian_adjugate%d[lane]" % component,
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
        call_args.append("block_jacobian_determinant0")
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
        ]
    )
    for shape in range(n_shape):
        for field_index, field in enumerate(system.fields):
            stream = shape * n_fields + field_index
            lines.extend(
                _direct_atomic_scatter_lines(
                    "%s_out" % field.name,
                    "ev[%d * VECTOR_SIZE + %%s] * out_stride" % shape,
                    "block_output[%d][%%s]" % stream,
                    "        ",
                )
            )
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
    lines.extend(
        [
            "",
            _parallel_for_pragma("static"),
            "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            "        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
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
    if gradient_metric is not None:
        lines.append(
            "        scalar_t block_geom_metric_data[%d][N_QP * VECTOR_SIZE];"
            % gradient_metric.metric_components
        )
    lines.extend(
        [
            "        scalar_t block_output[N_FIELDS * N_SHAPE][VECTOR_SIZE];",
            "",
            *_work_item_loop_lines("        "),
        ]
    )
    for shape in range(n_shape):
        lines.append(
            "            ev[%d * VECTOR_SIZE + lane] = elements[%d][evbegin + lane];"
            % (shape, shape)
        )
    lines.extend(["        }", "", *_work_item_loop_lines("        ")])
    for shape in range(n_shape):
        node = "ev[%d * VECTOR_SIZE + lane]" % shape
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
    lines.extend(
        [
            "        }",
        ]
    )
    lines.extend(["", *_zero_block_output_lines("block_output", n_fields * n_shape, "        ")])
    if tensor_product_geometry:
        lines.append("")
        lines.extend(
            tensor_product_evaluated_isoparametric_geometry_lines(
                dim=dim,
                n_shape=n_shape,
                n_qp=rule.n_qp,
                local_prefix=local_prefix,
                coordinate_streams=tensor_product_ordered_coordinate_streams(
                    dim,
                    n_shape,
                    tuple(range(dim * n_shape)),
                    lambda stream: "block_coordinates[%d]" % stream,
                    shape_order=tuple(range(n_shape))
                    if sfem_tensor_product_hex_uses_cartesian_ordering(rule.element_type)
                    else None,
                ),
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
        ]
    )
    for shape in range(n_shape):
        for field_index, field in enumerate(system.fields):
            stream = shape * n_fields + field_index
            lines.extend(
                _direct_atomic_scatter_lines(
                    "%s_out" % field.name,
                    "ev[%d * VECTOR_SIZE + %%s] * out_stride" % shape,
                    "block_output[%d][%%s]" % stream,
                    "        ",
                )
            )
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
