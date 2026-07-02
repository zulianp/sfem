from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class GradientMetricTransformation:
    field_index: int
    stream_group_name: str
    scale: sp.Expr
    reference_gradients: tuple
    affine_geometry_storage: str = "symmetric_metric_soa"
    affine_geometry_component_order: str = "upper_row_major"

    @property
    def dim(self):
        return len(self.reference_gradients[0]) if self.reference_gradients else 0

    @property
    def n_shape(self):
        return len(self.reference_gradients)

    @property
    def metric_components(self):
        return symmetric_metric_component_count(self.dim)

    def reference_gradient(self, shape, component):
        return self.reference_gradients[int(shape)][int(component)]

    def metric_component(self, left, right):
        return symmetric_metric_component_index(left, right)


def simplex_gradient_metric_transformation(system, rule, coefficients, dependencies):
    if not _is_constant_p1_simplex_rule(rule):
        return None
    if len(system.fields) != 1 or len(coefficients) != 1:
        return None
    if any(dependencies.value_coefficients):
        return None
    if dependencies.previous or dependencies.current_value or dependencies.direction_value:
        return None
    if dependencies.current_gradient == dependencies.direction_gradient:
        return None

    stream_group_name = "current" if dependencies.current_gradient else "direction"
    field = system.fields[0]
    gradient_symbols = (
        field.gradient
        if stream_group_name == "current"
        else field.direction_gradient
    )
    scale = _uniform_gradient_scale(
        coefficients[0].gradient,
        gradient_symbols,
        tuple(
            field.current_symbols
            + field.previous_symbols
            + field.direction_symbols
        ),
    )
    if scale is None:
        return None

    return GradientMetricTransformation(
        field_index=0,
        stream_group_name=stream_group_name,
        scale=scale,
        reference_gradients=_constant_reference_gradients(rule),
    )


def constant_p1_simplex_reference_gradients(rule):
    return _constant_reference_gradients(rule) if _is_constant_p1_simplex_rule(rule) else None


def symmetric_metric_component_count(dim):
    dim = int(dim)
    return dim * (dim + 1) // 2


def symmetric_metric_component_index(left, right):
    left = int(left)
    right = int(right)
    if right < left:
        left, right = right, left
    return right * (right + 1) // 2 + left


def symmetric_metric_storage_component_index(dim, left, right, order):
    dim = int(dim)
    left = int(left)
    right = int(right)
    if right < left:
        left, right = right, left
    if order == "upper_column_major":
        return symmetric_metric_component_index(left, right)
    if order == "upper_row_major":
        return left * dim - (left * (left - 1)) // 2 + (right - left)
    raise ValueError("unsupported symmetric metric component order %r" % (order,))


def _uniform_gradient_scale(expressions, gradient_symbols, forbidden_symbols):
    scale = None
    forbidden = set(forbidden_symbols)
    for symbol, expression in zip(gradient_symbols, expressions):
        expression = sp.sympify(expression)
        ratio = sp.simplify(expression / symbol)
        if sp.simplify(expression - ratio * symbol) != 0:
            return None
        if ratio.free_symbols.intersection(forbidden):
            return None
        if scale is None:
            scale = ratio
        elif sp.simplify(ratio - scale) != 0:
            return None
    return None if scale is None else sp.simplify(scale)


def _is_constant_p1_simplex_rule(rule):
    return (
        str(rule.element_type).upper() in ("TRI3", "TET4")
        and int(rule.n_shape) == int(rule.dim) + 1
        and _constant_reference_gradients(rule) is not None
    )


def _constant_reference_gradients(rule):
    gradients = []
    for shape in range(rule.n_shape):
        row = []
        for d in range(rule.dim):
            first = rule.reference_gradients[(shape * rule.dim) + d]
            for q in range(1, rule.n_qp):
                value = rule.reference_gradients[
                    (q * rule.n_shape + shape) * rule.dim + d
                ]
                if value != first:
                    return None
            row.append(sp.nsimplify(first))
        gradients.append(tuple(row))
    return tuple(gradients)
