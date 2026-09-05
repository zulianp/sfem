from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Mapping, Optional, Tuple, Union

import sympy as sp
import sympy.codegen.ast as ast
from sympy.printing.c import C99CodePrinter

def _validate_diagnostics_plan_names(plan, expected_names):
    if plan is None:
        return None
    expected_names = tuple(str(name) for name in expected_names)
    public_names = tuple(getattr(plan, "public_names", ()))
    missing = tuple(name for name in expected_names if name not in public_names)
    if missing:
        raise ValueError(
            "diagnostics plan is missing entries: %s" % ", ".join(missing)
        )
    return plan



SympyExpr = Union[sp.Expr, ast.Assignment, ast.AddAugmentedAssignment]




















class ExpressionRole(str, Enum):
    ENERGY = "energy"
    RESIDUAL = "residual"
    GRADIENT = "gradient"
    JACOBIAN_ACTION = "jacobian_action"
    HESSIAN_ACTION = "hessian_action"
    MERIT = "merit"
    POTENTIAL = "potential"
    OPERATOR_EVALUATION = "operator_evaluation"


class PatternKind(str, Enum):
    REPEATED_SUBEXPRESSION = "repeated_subexpression"
    DISPLACEMENT_GRADIENT = "displacement_gradient"
    DEFORMATION_GRADIENT = "deformation_gradient"
    GEOMETRIC_JACOBIAN = "geometric_jacobian"
    GEOMETRIC_ADJUGATE = "geometric_adjugate"
    FIRST_PIOLA_STRESS = "first_piola_stress"
    TRANSFORMED_FIRST_PIOLA = "transformed_first_piola"
    LINEARIZED_TRANSFORMED_FIRST_PIOLA = "linearized_transformed_first_piola"
    REFERENCE_SHAPE_VALUE = "reference_shape_value"
    REFERENCE_SHAPE_GRADIENT = "reference_shape_gradient"


class ScopeKind(str, Enum):
    MESH = "mesh"
    PATCH = "patch"
    ELEMENT = "element"
    QUADRATURE = "quadrature"
    TRIAL = "trial"
    TEST = "test"
    VECTOR_LANE = "vector_lane"
    WARP = "warp"
    THREAD = "thread"


class LayoutKind(str, Enum):
    SOA = "soa"
    AOS = "aos"
    AOSOA = "aosoa"


@dataclass(frozen=True)
class DataLayout:
    kind: LayoutKind = LayoutKind.SOA
    block_size: Optional[int] = None
    components: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "kind", LayoutKind(self.kind))
        if self.kind != LayoutKind.AOSOA and self.block_size is not None:
            raise ValueError("block_size is only valid for AoSoA layout")
        if self.kind == LayoutKind.AOSOA:
            if self.block_size is None or self.block_size <= 0:
                raise ValueError("AoSoA layout requires a positive block_size")


def data_layout(kind=LayoutKind.SOA, block_size=None, components=None):
    return DataLayout(kind, block_size, components)


@dataclass(frozen=True)
class ExecutionScope:
    kind: ScopeKind
    symbols: Tuple[sp.Symbol, ...] = ()
    name: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "kind", ScopeKind(self.kind))
        object.__setattr__(self, "symbols", _as_symbol_tuple(self.symbols))
        if self.name is None:
            object.__setattr__(self, "name", self.kind.value)


def execution_scope(kind, symbols=(), name=None):
    return ExecutionScope(kind, symbols, name)


@dataclass(frozen=True)
class KernelTemplateParameter:
    name: str
    value: int
    source: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "value", int(self.value))
        if self.value <= 0:
            raise ValueError("kernel template parameters must be positive")


def kernel_template_parameter(name, value, source=None):
    return KernelTemplateParameter(name, value, source)


@dataclass(frozen=True)
class DimensionSpecialization:
    dim: int
    source: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "dim", int(self.dim))
        if self.dim not in (1, 2, 3):
            raise ValueError("dimension specialization must be 1, 2, or 3")


def dimension_specialization(dim, source=None):
    return DimensionSpecialization(dim, source)


@dataclass(frozen=True)
class KernelExpression:
    role: ExpressionRole
    expression: SympyExpr
    name: Optional[str] = None


@dataclass(frozen=True)
class ExpressionPattern:
    kind: PatternKind
    node: object
    expression: Optional[sp.Expr]
    symbols: Tuple[sp.Symbol, ...]
    source: str
    matched_expressions: Tuple[sp.Expr, ...] = ()
    symbolic_object: Optional["SymbolicObject"] = None


@dataclass(frozen=True)
class SymbolicObject:
    kind: PatternKind
    name: str
    entries: Tuple[sp.Expr, ...]
    shape: Tuple[int, ...]
    direct_symbols: Tuple[sp.Symbol, ...] = ()
    definitions: Tuple[sp.Expr, ...] = ()
    layout: DataLayout = DataLayout()
    metadata: Mapping[str, object] = None

    @property
    def symbols(self):
        symbols = set(self.direct_symbols)
        for entry in self.entries:
            symbols.update(entry.free_symbols)
        for definition in self.definitions:
            symbols.update(definition.free_symbols)
        return tuple(sorted(symbols, key=str))

    def as_matrix(self):
        if len(self.shape) != 2:
            raise ValueError("object is not rank-2")
        return sp.Matrix(self.shape[0], self.shape[1], self.entries)

    def match(self, expression):
        direct_symbol_set = set(self.direct_symbols)
        matched_symbols = tuple(
            sorted(expression.free_symbols.intersection(direct_symbol_set), key=str)
        )
        matched_expressions = tuple(
            entry
            for entry in self.entries + self.definitions
            if (not isinstance(entry, sp.Symbol) or entry not in direct_symbol_set)
            and _contains_expression(expression, entry)
        )
        return matched_symbols, matched_expressions

    @property
    def has_definitions(self):
        return len(self.definitions) != 0

    def definition_matrix(self):
        if not self.has_definitions:
            return self.as_matrix()
        if len(self.shape) != 2:
            raise ValueError("object is not rank-2")
        return sp.Matrix(self.shape[0], self.shape[1], self.definitions)

    def definition_assignments(self):
        if not self.has_definitions:
            return ()
        return tuple(
            ast.Assignment(target, definition)
            for target, definition in zip(self.entries, self.definitions)
        )

    def component_index(self, entry):
        for idx, candidate in enumerate(self.entries):
            if candidate == entry:
                return idx
        raise ValueError("entry does not belong to symbolic object %s" % self.name)

    def layout_offset(self, entry, item_index, stride=None):
        return layout_offset(
            self.layout,
            self.component_index(entry),
            item_index,
            components=len(self.entries),
            stride=stride,
        )

    def as_vector(self):
        return sp.Matrix(len(self.entries), 1, self.entries)

    @property
    def template_parameters(self):
        return tuple((self.metadata or {}).get("template_parameters", ()))


def layout_offset(layout, component, item_index, components=None, stride=None):
    layout = _normalize_layout(layout)
    component = sp.sympify(component)
    item_index = sp.sympify(item_index)
    components = layout.components if components is None else components

    if components is None:
        raise ValueError("components must be provided for layout offset")

    if layout.kind == LayoutKind.SOA:
        stride = sp.sympify(stride if stride is not None else "stride")
        return component * stride + item_index

    if layout.kind == LayoutKind.AOS:
        return item_index * components + component

    block_size = sp.Integer(layout.block_size)
    block = sp.floor(item_index / block_size)
    lane = sp.Mod(item_index, block_size)
    return block * (components * block_size) + component * block_size + lane


class DisplacementGradient(SymbolicObject):
    def __init__(self, name, dim, entries=None, layout=None):
        entries = (
            _matrix_entries(name, dim, dim)
            if entries is None
            else _flatten_entries(entries)
        )
        _init_symbolic_object(
            self,
            PatternKind.DISPLACEMENT_GRADIENT,
            name,
            entries,
            (dim, dim),
            tuple(entry for entry in entries if isinstance(entry, sp.Symbol)),
            layout=_normalize_layout(layout),
        )


class DeformationGradient(SymbolicObject):
    def __init__(self, name, dim, entries=None, layout=None):
        entries = (
            _matrix_entries(name, dim, dim)
            if entries is None
            else _flatten_entries(entries)
        )
        _init_symbolic_object(
            self,
            PatternKind.DEFORMATION_GRADIENT,
            name,
            entries,
            (dim, dim),
            tuple(entry for entry in entries if isinstance(entry, sp.Symbol)),
            layout=_normalize_layout(layout),
        )

    @classmethod
    def from_displacement_gradient(cls, name, displacement_gradient, layout=None):
        dim = displacement_gradient.shape[0]
        F = sp.eye(dim) + displacement_gradient.as_matrix()
        obj = cls.__new__(cls)
        _init_symbolic_object(
            obj,
            PatternKind.DEFORMATION_GRADIENT,
            name,
            _flatten_entries(F),
            (dim, dim),
            (),
            layout=_normalize_layout(layout, displacement_gradient.layout),
        )
        return obj


class GeometricJacobian(SymbolicObject):
    def __init__(self, name, spatial_dim, manifold_dim=None, entries=None, layout=None):
        manifold_dim = spatial_dim if manifold_dim is None else manifold_dim
        entries = (
            _matrix_entries(name, spatial_dim, manifold_dim)
            if entries is None
            else _flatten_entries(entries)
        )
        _init_symbolic_object(
            self,
            PatternKind.GEOMETRIC_JACOBIAN,
            name,
            entries,
            (spatial_dim, manifold_dim),
            tuple(entry for entry in entries if isinstance(entry, sp.Symbol)),
            layout=_normalize_layout(layout),
        )


class GeometricAdjugate(SymbolicObject):
    def __init__(self, name, dim, entries=None, layout=None):
        entries = (
            _matrix_entries(name, dim, dim)
            if entries is None
            else _flatten_entries(entries)
        )
        _init_symbolic_object(
            self,
            PatternKind.GEOMETRIC_ADJUGATE,
            name,
            entries,
            (dim, dim),
            tuple(entry for entry in entries if isinstance(entry, sp.Symbol)),
            layout=_normalize_layout(layout),
        )

    @classmethod
    def from_jacobian(cls, name, jacobian, layout=None):
        J = jacobian.as_matrix()
        if J.shape[0] != J.shape[1]:
            raise ValueError("adjugate requires a square geometric Jacobian")
        obj = cls.__new__(cls)
        _init_symbolic_object(
            obj,
            PatternKind.GEOMETRIC_ADJUGATE,
            name,
            _flatten_entries(_adjugate(J)),
            J.shape,
            (),
            layout=_normalize_layout(layout, jacobian.layout),
        )
        return obj


class ReferenceShapeGradient(SymbolicObject):
    def __init__(self, name, dim, entries=None, layout=None):
        entries = (
            _matrix_entries(name, dim, dim)
            if entries is None
            else _flatten_entries(entries)
        )
        _init_symbolic_object(
            self,
            PatternKind.REFERENCE_SHAPE_GRADIENT,
            name,
            entries,
            (dim, dim),
            tuple(entry for entry in entries if isinstance(entry, sp.Symbol)),
            layout=_normalize_layout(layout),
        )


class ReferenceShapeValues(SymbolicObject):
    def __init__(self, name, n_nodes, entries=None, layout=None):
        entries = (
            _matrix_entries(name, n_nodes, 1)
            if entries is None
            else _flatten_entries(entries)
        )
        if len(entries) != n_nodes:
            raise ValueError("reference shape value array size must equal n_nodes")
        _init_symbolic_object(
            self,
            PatternKind.REFERENCE_SHAPE_VALUE,
            name,
            entries,
            (n_nodes,),
            tuple(entry for entry in entries if isinstance(entry, sp.Symbol)),
            layout=_normalize_layout(layout),
            metadata={
                "n_nodes": n_nodes,
                "dim": 1,
                "template_parameters": (
                    kernel_template_parameter("%s_n_nodes" % name, n_nodes, name),
                ),
            },
        )

    def value(self, node):
        return self.entries[node]


class ReferenceShapeGradients(SymbolicObject):
    def __init__(self, name, n_nodes, dim, entries=None, layout=None):
        entries = (
            _matrix_entries(name, n_nodes, dim)
            if entries is None
            else _flatten_entries(entries)
        )
        if len(entries) != n_nodes * dim:
            raise ValueError("reference shape gradient array size must equal n_nodes * dim")
        _init_symbolic_object(
            self,
            PatternKind.REFERENCE_SHAPE_GRADIENT,
            name,
            entries,
            (n_nodes, dim),
            tuple(entry for entry in entries if isinstance(entry, sp.Symbol)),
            layout=_normalize_layout(layout),
            metadata={
                "n_nodes": n_nodes,
                "dim": dim,
                "template_parameters": (
                    kernel_template_parameter("%s_n_nodes" % name, n_nodes, name),
                    kernel_template_parameter("%s_dim" % name, dim, name),
                ),
            },
        )

    def gradient(self, node, component):
        return self.entries[node * self.dim + component]

    def node_gradient(self, node):
        return sp.Matrix(self.dim, 1, [self.gradient(node, d) for d in range(self.dim)])

    def tensor_gradient(self, node, row):
        ret = sp.zeros(self.dim, self.dim)
        for col in range(self.dim):
            ret[row, col] = self.gradient(node, col)
        return ret

    @property
    def n_nodes(self):
        return self.metadata["n_nodes"]

    @property
    def dim(self):
        return self.metadata["dim"]


class FirstPiolaStress(SymbolicObject):
    def __init__(self, name, dim, entries=None, layout=None):
        entries = (
            _matrix_entries(name, dim, dim)
            if entries is None
            else _flatten_entries(entries)
        )
        _init_symbolic_object(
            self,
            PatternKind.FIRST_PIOLA_STRESS,
            name,
            entries,
            (dim, dim),
            tuple(entry for entry in entries if isinstance(entry, sp.Symbol)),
            layout=_normalize_layout(layout),
        )

    @classmethod
    def from_linear_elasticity(cls, name, displacement_gradient, mu, lmbda, layout=None):
        P = linear_elastic_first_piola(displacement_gradient, mu, lmbda)
        entries = _matrix_entries(name, P.shape[0], P.shape[1])
        obj = cls.__new__(cls)
        _init_symbolic_object(
            obj,
            PatternKind.FIRST_PIOLA_STRESS,
            name,
            entries,
            P.shape,
            entries,
            _flatten_entries(P),
            layout=_normalize_layout(layout, displacement_gradient.layout),
        )
        return obj


class TransformedFirstPiola(SymbolicObject):
    def __init__(self, name, dim, entries=None, layout=None):
        entries = (
            _matrix_entries(name, dim, dim)
            if entries is None
            else _flatten_entries(entries)
        )
        _init_symbolic_object(
            self,
            PatternKind.TRANSFORMED_FIRST_PIOLA,
            name,
            entries,
            (dim, dim),
            tuple(entry for entry in entries if isinstance(entry, sp.Symbol)),
            layout=_normalize_layout(layout),
        )

    @classmethod
    def from_first_piola(cls, name, first_piola, jacobian_inverse, measure=1, layout=None):
        transformed = transformed_first_piola(first_piola, jacobian_inverse, measure)
        entries = _matrix_entries(name, transformed.shape[0], transformed.shape[1])
        obj = cls.__new__(cls)
        _init_symbolic_object(
            obj,
            PatternKind.TRANSFORMED_FIRST_PIOLA,
            name,
            entries,
            transformed.shape,
            entries,
            _flatten_entries(transformed),
            layout=_normalize_layout(layout, first_piola.layout),
        )
        return obj


class LinearizedTransformedFirstPiola(SymbolicObject):
    def __init__(self, name, dim, entries=None, layout=None):
        entries = (
            _matrix_entries(name, dim, dim)
            if entries is None
            else _flatten_entries(entries)
        )
        _init_symbolic_object(
            self,
            PatternKind.LINEARIZED_TRANSFORMED_FIRST_PIOLA,
            name,
            entries,
            (dim, dim),
            tuple(entry for entry in entries if isinstance(entry, sp.Symbol)),
            layout=_normalize_layout(layout),
        )

    @classmethod
    def from_first_piola(
        cls,
        name,
        first_piola,
        displacement_gradient,
        trial_reference_gradient,
        jacobian_inverse,
        measure=1,
        layout=None,
    ):
        linearized = linearized_transformed_first_piola(
            first_piola,
            displacement_gradient,
            trial_reference_gradient,
            jacobian_inverse,
            measure,
        )
        entries = _matrix_entries(name, linearized.shape[0], linearized.shape[1])
        obj = cls.__new__(cls)
        _init_symbolic_object(
            obj,
            PatternKind.LINEARIZED_TRANSFORMED_FIRST_PIOLA,
            name,
            entries,
            linearized.shape,
            entries,
            _flatten_entries(linearized),
            layout=_normalize_layout(layout, first_piola.layout),
        )
        return obj


def matrix_symbols(name, rows, cols):
    return sp.Matrix(rows, cols, _matrix_entries(name, rows, cols))


def vector_symbols(name, size):
    return sp.Matrix(size, 1, _matrix_entries(name, size, 1))


def matrix_inner(left, right):
    left = _as_matrix(left, "left")
    right = _as_matrix(right, "right")
    if left.shape != right.shape:
        raise ValueError("matrix shapes must match for inner product")

    ret = 0
    rows, cols = left.shape
    for i in range(rows):
        for j in range(cols):
            ret += left[i, j] * right[i, j]
    return ret


def displacement_gradient_from_reference(
    displacement,
    reference_shape_gradients,
    jacobian_inverse,
):
    displacement = _as_vector(displacement)
    reference_shape_gradients = tuple(
        _as_matrix(grad, "reference_shape_gradient")
        for grad in reference_shape_gradients
    )
    _check_same_length(
        displacement,
        reference_shape_gradients,
        "displacement",
        "reference_shape_gradients",
    )

    jacobian_inverse = _as_matrix(jacobian_inverse, "jacobian_inverse")
    rows, cols = reference_shape_gradients[0].shape
    eval_grad = sp.zeros(rows, cols)

    for coeff, grad in zip(displacement, reference_shape_gradients):
        if grad.shape != (rows, cols):
            raise ValueError("all reference shape gradients must have the same shape")
        eval_grad += coeff * grad

    return eval_grad * jacobian_inverse


def small_strain(displacement_gradient):
    grad = _as_matrix(displacement_gradient, "displacement_gradient")
    return (grad + grad.T) / 2


def linear_elastic_energy(displacement_gradient, mu, lmbda):
    strain = small_strain(displacement_gradient)
    trace = _matrix_trace(strain)
    return mu * matrix_inner(strain, strain) + (lmbda / 2) * trace * trace


def linear_elastic_first_piola(displacement_gradient, mu, lmbda):
    grad = _as_matrix(displacement_gradient, "displacement_gradient")
    strain = small_strain(grad)
    trace = _matrix_trace(strain)
    return 2 * mu * strain + lmbda * trace * sp.eye(grad.shape[0])


def transformed_first_piola(first_piola, jacobian_inverse, measure=1):
    P = _as_matrix(first_piola, "first_piola")
    jacobian_inverse = _as_matrix(jacobian_inverse, "jacobian_inverse")
    if P.shape[1] != jacobian_inverse.shape[0]:
        raise ValueError("first_piola and jacobian_inverse shapes are incompatible")
    return P * jacobian_inverse.T * measure


def linearized_first_piola(first_piola, displacement_gradient, direction_gradient):
    P = _as_matrix(first_piola, "first_piola")
    grad = _as_matrix(displacement_gradient, "displacement_gradient")
    direction = _as_matrix(direction_gradient, "direction_gradient")
    if P.shape != grad.shape or P.shape != direction.shape:
        raise ValueError(
            "first_piola, displacement_gradient, and direction_gradient shapes must match"
        )

    rows, cols = P.shape
    ret = sp.zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            ret[i, j] = directional_derivative(
                P[i, j],
                _flatten_entries(grad),
                _flatten_entries(direction),
            )
    return ret


def linearized_transformed_first_piola(
    first_piola,
    displacement_gradient,
    trial_reference_gradient,
    jacobian_inverse,
    measure=1,
):
    direction_gradient = _as_matrix(trial_reference_gradient, "trial_reference_gradient")
    jacobian_inverse = _as_matrix(jacobian_inverse, "jacobian_inverse")
    direction_gradient = direction_gradient * jacobian_inverse
    dP = linearized_first_piola(first_piola, displacement_gradient, direction_gradient)
    return transformed_first_piola(dP, jacobian_inverse, measure)


def weak_gradient_from_transformed_first_piola(
    transformed_first_piola,
    reference_shape_gradients,
):
    transformed = _as_matrix(transformed_first_piola, "transformed_first_piola")
    return sp.Matrix(
        len(reference_shape_gradients),
        1,
        [
            matrix_inner(transformed, _as_matrix(grad, "reference_shape_gradient"))
            for grad in reference_shape_gradients
        ],
    )


def weak_hessian_action_from_linearized_transformed_first_piola(
    linearized_transformed_first_piola,
    test_reference_shape_gradients,
):
    linearized = _as_matrix(
        linearized_transformed_first_piola,
        "linearized_transformed_first_piola",
    )
    return sp.Matrix(
        len(test_reference_shape_gradients),
        1,
        [
            matrix_inner(linearized, _as_matrix(grad, "test_reference_shape_gradient"))
            for grad in test_reference_shape_gradients
        ],
    )


def gradient_from_energy(energy, variables):
    _check_scalar_expression(energy, "energy")
    variables = _as_vector(variables)
    _check_variables(variables)
    return sp.Matrix(len(variables), 1, [sp.diff(energy, var) for var in variables])


def residual_from_energy(energy, variables):
    return gradient_from_energy(energy, variables)


def directional_derivative(expression, variables, directions):
    _check_scalar_expression(expression, "expression")
    variables = _as_vector(variables)
    directions = _as_vector(directions)
    _check_same_length(variables, directions, "variables", "directions")
    _check_variables(variables)

    ret = 0
    for var, direction in zip(variables, directions):
        ret += sp.diff(expression, var) * direction
    return ret


def jacobian_action_from_residual(residual, variables, directions):
    residual = _as_vector(residual)
    return sp.Matrix(
        len(residual),
        1,
        [directional_derivative(expr, variables, directions) for expr in residual],
    )


def hessian_action_from_energy(energy, variables, directions):
    residual = residual_from_energy(energy, variables)
    return jacobian_action_from_residual(residual, variables, directions)



























class KernelExpressions:
    def __init__(self, expressions: Optional[Iterable[KernelExpression]] = None):
        self._expressions = []
        if expressions is not None:
            for expr in expressions:
                self.add(expr.role, expr.expression, expr.name)

    @property
    def expressions(self):
        """The collected expressions, in insertion order.

        Scheduling them is the planning layer's job -- see
        ``codegen.framework.plans.scheduling.build_expression_graph``.  This
        class deliberately has no ``build_graph`` method: reaching the scheduler
        from here would point an import from the specification layer down into
        the planning layer.
        """
        return tuple(self._expressions)

    def add(self, role, expression, name=None):
        role = ExpressionRole(role)
        for expr in _flatten_expression(expression):
            self._expressions.append(KernelExpression(role, expr, name))
        return self

    def energy(self, expression, name=None):
        return self.add(ExpressionRole.ENERGY, expression, name)

    def residual(self, expression, name=None):
        return self.add(ExpressionRole.RESIDUAL, expression, name)

    def gradient(self, expression, name=None):
        return self.add(ExpressionRole.GRADIENT, expression, name)

    def jacobian_action(self, expression, name=None):
        return self.add(ExpressionRole.JACOBIAN_ACTION, expression, name)

    def hessian_action(self, expression, name=None):
        return self.add(ExpressionRole.HESSIAN_ACTION, expression, name)

    def merit(self, expression, name=None):
        return self.add(ExpressionRole.MERIT, expression, name)

    def operator_evaluation(self, symbolic_object, name=None):
        if not symbolic_object.has_definitions:
            raise ValueError("symbolic object has no definitions to evaluate")
        return self.add(
            ExpressionRole.OPERATOR_EVALUATION,
            symbolic_object.definition_assignments(),
            name if name is not None else symbolic_object.name,
        )

    def residual_from_energy(self, energy, variables, name=None):
        return self.residual(residual_from_energy(energy, variables), name)

    def gradient_from_energy(self, energy, variables, name=None):
        return self.gradient(gradient_from_energy(energy, variables), name)

    def jacobian_action_from_residual(
        self,
        residual,
        variables,
        directions,
        name=None,
    ):
        return self.jacobian_action(
            jacobian_action_from_residual(residual, variables, directions),
            name,
        )

    def hessian_action_from_energy(self, energy, variables, directions, name=None):
        return self.hessian_action(
            hessian_action_from_energy(energy, variables, directions),
            name,
        )


    def __iter__(self):
        return iter(self._expressions)

    def __len__(self):
        return len(self._expressions)













def _normalize_kernel_expression(expr):
    if isinstance(expr, KernelExpression):
        return expr
    if isinstance(expr, tuple) and len(expr) in (2, 3):
        role = ExpressionRole(expr[0])
        name = expr[2] if len(expr) == 3 else None
        return KernelExpression(role, expr[1], name)
    raise TypeError("Expected KernelExpression or (role, expression[, name]) tuple")




























def _flatten_expression(expression):
    if isinstance(expression, sp.MatrixBase):
        for value in expression:
            yield value
        return

    if isinstance(expression, (list, tuple)):
        for value in expression:
            yield from _flatten_expression(value)
        return

    yield expression


def _as_symbol_tuple(symbols):
    symbols = (symbols,) if isinstance(symbols, sp.Symbol) else tuple(symbols)
    for symbol in symbols:
        if not isinstance(symbol, sp.Symbol):
            raise TypeError("scope symbols must be SymPy symbols")
    return symbols


def _normalize_layout(layout, fallback=None):
    if layout is None:
        return fallback if fallback is not None else DataLayout()
    if isinstance(layout, DataLayout):
        return layout
    return DataLayout(layout)


def _normalize_template_parameter(parameter):
    if isinstance(parameter, KernelTemplateParameter):
        return parameter
    if isinstance(parameter, tuple) and len(parameter) in (2, 3):
        source = parameter[2] if len(parameter) == 3 else None
        return KernelTemplateParameter(parameter[0], parameter[1], source)
    raise TypeError("template parameters must be KernelTemplateParameter or tuple")


def _normalize_dimension_specialization(specialization):
    if specialization is None:
        return None
    if isinstance(specialization, DimensionSpecialization):
        return specialization
    if isinstance(specialization, tuple) and len(specialization) in (1, 2):
        source = specialization[1] if len(specialization) == 2 else None
        return DimensionSpecialization(specialization[0], source)
    return DimensionSpecialization(specialization)


def _symbolic_object_dimension(symbolic_object):
    if symbolic_object.kind == PatternKind.REFERENCE_SHAPE_GRADIENT:
        dim = symbolic_object.metadata.get("dim")
        if dim is not None:
            return dim
    if len(symbolic_object.shape) == 2 and symbolic_object.shape[0] == symbolic_object.shape[1]:
        return symbolic_object.shape[0]
    return None


def _dimension_specialization(symbolic_objects, explicit_specialization):
    specialization = _normalize_dimension_specialization(explicit_specialization)

    for symbolic_object in symbolic_objects:
        dim = _symbolic_object_dimension(symbolic_object)
        if dim is None:
            continue

        candidate = DimensionSpecialization(dim, symbolic_object.name)
        if specialization is None:
            specialization = candidate
            continue

        if specialization.dim != candidate.dim:
            raise ValueError(
                "conflicting dimension specializations: %d from %s and %d from %s"
                % (
                    specialization.dim,
                    specialization.source,
                    candidate.dim,
                    candidate.source,
                )
            )

    return specialization


def _template_parameters(symbolic_objects, explicit_parameters):
    parameters = []
    for symbolic_object in symbolic_objects:
        parameters.extend(symbolic_object.template_parameters)
    parameters.extend(explicit_parameters or ())

    merged = {}
    ordered = []
    for raw_parameter in parameters:
        parameter = _normalize_template_parameter(raw_parameter)
        existing = merged.get(parameter.name)
        if existing is not None:
            if existing.value != parameter.value:
                raise ValueError(
                    "conflicting values for template parameter %s" % parameter.name
                )
            continue
        merged[parameter.name] = parameter
        ordered.append(parameter)

    return tuple(ordered)




















def _as_vector(values):
    vector = _flatten_entries(values)
    for value in vector:
        if not isinstance(value, sp.Expr):
            raise TypeError("expected SymPy expression, got %s" % type(value).__name__)
    return vector


def _as_matrix(value, name):
    if isinstance(value, SymbolicObject):
        return value.as_matrix()
    if isinstance(value, sp.MatrixBase):
        return value
    raise TypeError("%s must be a SymPy Matrix or SymbolicObject" % name)


def _matrix_trace(matrix):
    matrix = _as_matrix(matrix, "matrix")
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("trace requires a square matrix")

    ret = 0
    for i in range(rows):
        ret += matrix[i, i]
    return ret


def _check_same_length(left, right, left_name, right_name):
    if len(left) != len(right):
        raise ValueError(
            "%s and %s must have the same length (%d != %d)"
            % (left_name, right_name, len(left), len(right))
        )


def _check_scalar_expression(expression, name):
    if isinstance(expression, sp.MatrixBase) or isinstance(expression, (list, tuple)):
        raise TypeError("%s must be a scalar SymPy expression" % name)
    if not isinstance(expression, sp.Expr):
        raise TypeError(
            "%s must be a scalar SymPy expression, got %s"
            % (name, type(expression).__name__)
        )


def _check_variables(variables):
    for variable in variables:
        if not isinstance(variable, sp.Symbol):
            raise TypeError(
                "differentiation variables must be SymPy symbols, got %s"
                % type(variable).__name__
            )


def _init_symbolic_object(
    obj,
    kind,
    name,
    entries,
    shape,
    direct_symbols,
    definitions=(),
    layout=None,
    metadata=None,
):
    object.__setattr__(obj, "kind", kind)
    object.__setattr__(obj, "name", name)
    object.__setattr__(obj, "entries", tuple(entries))
    object.__setattr__(obj, "shape", tuple(shape))
    object.__setattr__(obj, "direct_symbols", tuple(direct_symbols))
    object.__setattr__(obj, "definitions", tuple(definitions))
    object.__setattr__(obj, "layout", _normalize_layout(layout))
    object.__setattr__(obj, "metadata", dict(metadata or {}))


def _matrix_entries(name, rows, cols):
    return tuple(sp.symbols("%s[%d]" % (name, i), real=True) for i in range(rows * cols))


def _flatten_entries(entries):
    if isinstance(entries, sp.MatrixBase):
        return tuple(entries)
    if isinstance(entries, (list, tuple)):
        values = []
        for entry in entries:
            values.extend(_flatten_entries(entry))
        return tuple(values)
    return (entries,)


def _contains_expression(expression, needle):
    if expression == needle:
        return True
    for node in sp.preorder_traversal(expression):
        if node == needle:
            return True
    return False


def _adjugate(matrix):
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("adjugate requires a square matrix")

    if rows == 1:
        return sp.Matrix(1, 1, [1])

    ret = sp.zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            minor = matrix.minor_submatrix(j, i)
            ret[i, j] = (-1) ** (i + j) * minor.det()
    return ret


def _rhs(expression):
    if isinstance(expression, (ast.Assignment, ast.AddAugmentedAssignment)):
        return expression.rhs
    return expression


def _lhs(expression):
    if isinstance(expression, (ast.Assignment, ast.AddAugmentedAssignment)):
        return expression.lhs
    return None


def _reattach_lhs(outputs, reduced_rhs):
    reduced_outputs = []
    for kernel_expr, rhs in zip(outputs, reduced_rhs):
        lhs = _lhs(kernel_expr.expression)
        if lhs is None:
            reduced_outputs.append(rhs)
        elif isinstance(kernel_expr.expression, ast.AddAugmentedAssignment):
            reduced_outputs.append(ast.AddAugmentedAssignment(lhs, rhs))
        else:
            reduced_outputs.append(ast.Assignment(lhs, rhs))
    return tuple(reduced_outputs)
































