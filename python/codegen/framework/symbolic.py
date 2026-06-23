from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Iterable, Mapping, Optional, Tuple, Union

import sympy as sp
import sympy.codegen.ast as ast
from sympy.printing.c import C99CodePrinter

try:
    import networkx as nx
except ModuleNotFoundError:
    try:
        from . import _networkx_compat as nx
    except ImportError:
        import _networkx_compat as nx


SympyExpr = Union[sp.Expr, ast.Assignment, ast.AddAugmentedAssignment]
_SFEM_SPECIALIZED_POW_MAX_EXPONENT = 16


class _SfemCCodePrinter(C99CodePrinter):
    def _print_Pow(self, expr):
        base, exponent = expr.as_base_exp()
        if exponent.is_Integer:
            exponent_value = int(exponent)
            if abs(exponent_value) <= _SFEM_SPECIALIZED_POW_MAX_EXPONENT:
                if exponent_value == 0:
                    return "1"
                if exponent_value == 1:
                    return self._print(base)
                suffix = "m%d" % abs(exponent_value) if exponent_value < 0 else "%d" % exponent_value
                return "pow_%s(%s)" % (suffix, self._print(base))
        return super()._print_Pow(expr)


_SFEM_CCODE_PRINTER = _SfemCCodePrinter()


def _sfem_ccode(expression):
    return _SFEM_CCODE_PRINTER.doprint(expression)


def _sfem_pow_function_name(exponent):
    exponent = int(exponent)
    if exponent < 0:
        return "pow_m%d" % abs(exponent)
    return "pow_%d" % exponent


def _sfem_pow_product_expression(exponent):
    exponent = int(exponent)
    if exponent == 0:
        return "T(1)"
    return " * ".join("x" for _ in range(exponent))


def _sfem_math_function_lines():
    lines = []
    for exponent in range(2, _SFEM_SPECIALIZED_POW_MAX_EXPONENT + 1):
        lines.extend(
            [
                "template <typename T>",
                "static SFEM_INLINE T %s(const T x) {" % _sfem_pow_function_name(exponent),
                "    return %s;" % _sfem_pow_product_expression(exponent),
                "}",
                "",
            ]
        )
    for exponent in range(1, _SFEM_SPECIALIZED_POW_MAX_EXPONENT + 1):
        lines.extend(
            [
                "template <typename T>",
                "static SFEM_INLINE T %s(const T x) {" % _sfem_pow_function_name(-exponent),
                "    return T(1) / %s(x);" % _sfem_pow_function_name(exponent)
                if exponent > 1
                else "    return T(1) / x;",
                "}",
                "",
            ]
        )
    return lines


def _sfem_math_header_source():
    lines = [
        "#ifndef SFEM_CODEGEN_KERNEL_MATH_HPP",
        "#define SFEM_CODEGEN_KERNEL_MATH_HPP",
        "",
        "#ifndef SFEM_INLINE",
        "#define SFEM_INLINE inline",
        "#endif",
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
    ]
    lines.extend(_sfem_math_function_lines())
    lines.extend(["} // namespace codegen", "} // namespace sfem", "", "#endif", ""])
    return "\n".join(lines)


def _sfem_math_inline_source_lines():
    lines = [
        "#ifndef SFEM_INLINE",
        "#define SFEM_INLINE inline",
        "#endif",
        "",
    ]
    lines.extend(_sfem_math_function_lines())
    return lines


class ExpressionRole(str, Enum):
    ENERGY = "energy"
    RESIDUAL = "residual"
    GRADIENT = "gradient"
    JACOBIAN_ACTION = "jacobian_action"
    HESSIAN_ACTION = "hessian_action"
    MERIT = "merit"
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


@dataclass(frozen=True)
class GeneratedKernelCode:
    language: str
    function_name: str
    source: str


@dataclass(frozen=True)
class GeneratedKernelFile:
    path: str
    source: str


@dataclass(frozen=True)
class SfemSoAArrayInput:
    name: str
    size: int
    scalar_type: str = "scalar_t"
    layout: str = "element_stream"
    n_qp: int = 1
    n_shape: int = 1
    components: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "n_qp", int(self.n_qp))
        object.__setattr__(self, "n_shape", int(self.n_shape))
        components = self.size if self.components is None else int(self.components)
        object.__setattr__(self, "components", components)
        if self.size <= 0:
            raise ValueError("SoA array input size must be positive")
        if self.n_qp <= 0:
            raise ValueError("SoA array input n_qp must be positive")
        if self.n_shape <= 0:
            raise ValueError("SoA array input n_shape must be positive")
        if self.components <= 0:
            raise ValueError("SoA array input components must be positive")
        if self.layout not in ("element_stream", "reference_qp_shape"):
            raise ValueError("SoA array input layout must be 'element_stream' or 'reference_qp_shape'")
        if self.layout == "reference_qp_shape":
            expected_size = self.n_qp * self.n_shape * self.components
            if self.size != expected_size:
                raise ValueError(
                    "reference_qp_shape input size must equal n_qp * n_shape * components"
                )

    @property
    def is_reference_qp_shape(self):
        return self.layout == "reference_qp_shape"

    @property
    def local_size(self):
        if self.is_reference_qp_shape:
            return self.n_shape * self.components
        return self.size


def sfem_soa_array_input(name, size, scalar_type="scalar_t"):
    return SfemSoAArrayInput(name, size, scalar_type)


def sfem_soa_reference_input(name, n_qp, n_shape, components, scalar_type="scalar_t"):
    return SfemSoAArrayInput(
        name,
        int(n_qp) * int(n_shape) * int(components),
        scalar_type,
        "reference_qp_shape",
        n_qp,
        n_shape,
        components,
    )


@dataclass(frozen=True)
class SfemSoAKernelForm:
    name: str
    expression_graph: Optional["ExpressionGraph"] = None
    has_direction: bool = False
    output_mode: str = "accumulate"
    weak_form: Optional["SfemSoAWeakForm"] = None

    def __post_init__(self):
        if self.output_mode not in ("assign", "accumulate"):
            raise ValueError("output_mode must be 'assign' or 'accumulate'")
        if self.expression_graph is None and self.weak_form is None:
            raise ValueError("SfemSoAKernelForm requires expression_graph or weak_form")


def sfem_soa_kernel_form(
    name,
    expression_graph=None,
    has_direction=False,
    output_mode="accumulate",
    weak_form=None,
):
    return SfemSoAKernelForm(name, expression_graph, has_direction, output_mode, weak_form)


@dataclass(frozen=True)
class SfemSoAWeakForm:
    energy_density: sp.Expr
    deformation_gradient: Tuple[sp.Expr, ...]
    dim: int

    def __post_init__(self):
        dim = int(self.dim)
        deformation_gradient = tuple(self.deformation_gradient)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "energy_density", sp.sympify(self.energy_density))
        object.__setattr__(self, "deformation_gradient", deformation_gradient)
        if dim <= 0:
            raise ValueError("weak form dim must be positive")
        if len(deformation_gradient) != dim * dim:
            raise ValueError("deformation_gradient must have dim * dim entries")

    def deformation_gradient_matrix(self):
        return sp.Matrix(self.dim, self.dim, self.deformation_gradient)

    def first_piola(self):
        variables = self.deformation_gradient
        return sp.Matrix(
            self.dim,
            self.dim,
            [sp.diff(self.energy_density, variable) for variable in variables],
        )

    def linearized_first_piola(self, trial_gradient):
        trial_gradient = tuple(trial_gradient)
        if len(trial_gradient) != self.dim * self.dim:
            raise ValueError("trial_gradient must have dim * dim entries")
        P = self.first_piola()
        variables = self.deformation_gradient
        return sp.Matrix(
            self.dim,
            self.dim,
            [
                directional_derivative(P[i, j], variables, trial_gradient)
                for i in range(self.dim)
                for j in range(self.dim)
            ],
        )

    def diagnostic_expressions(self, has_direction=False):
        expressions = [self.energy_density]
        expressions.extend(tuple(self.first_piola()))
        if has_direction:
            trial_gradient = tuple(
                sp.symbols("trial_grad[%d]" % i)
                for i in range(self.dim * self.dim)
            )
            expressions.extend(tuple(self.linearized_first_piola(trial_gradient)))
        return tuple(expressions)


def sfem_soa_weak_form(energy_density, deformation_gradient):
    deformation_gradient = _as_matrix(deformation_gradient, "deformation_gradient")
    if deformation_gradient.shape[0] != deformation_gradient.shape[1]:
        raise ValueError("deformation_gradient must be square")
    return SfemSoAWeakForm(
        energy_density,
        tuple(deformation_gradient),
        deformation_gradient.shape[0],
    )


@dataclass(frozen=True)
class SfemElementQuadratureRule:
    element_type: str
    dim: int
    n_shape: int
    weights: Tuple[float, ...]
    reference_gradients: Tuple[float, ...]
    order: int = 1
    tensor_product_shape_values_1d: Tuple[float, ...] = ()
    tensor_product_shape_gradients_1d: Tuple[float, ...] = ()
    tensor_product_weights_1d: Tuple[float, ...] = ()
    tensor_product_dim: int = 0

    def __post_init__(self):
        element_type = str(self.element_type).upper()
        dim = int(self.dim)
        n_shape = int(self.n_shape)
        weights = tuple(float(value) for value in self.weights)
        reference_gradients = tuple(float(value) for value in self.reference_gradients)
        tensor_product_shape_values_1d = tuple(
            float(value) for value in self.tensor_product_shape_values_1d
        )
        tensor_product_shape_gradients_1d = tuple(
            float(value) for value in self.tensor_product_shape_gradients_1d
        )
        tensor_product_weights_1d = tuple(
            float(value) for value in self.tensor_product_weights_1d
        )
        tensor_product_dim = int(self.tensor_product_dim)
        object.__setattr__(self, "element_type", element_type)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "n_shape", n_shape)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "reference_gradients", reference_gradients)
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(
            self,
            "tensor_product_shape_values_1d",
            tensor_product_shape_values_1d,
        )
        object.__setattr__(
            self,
            "tensor_product_shape_gradients_1d",
            tensor_product_shape_gradients_1d,
        )
        object.__setattr__(self, "tensor_product_weights_1d", tensor_product_weights_1d)
        object.__setattr__(self, "tensor_product_dim", tensor_product_dim)
        if dim <= 0:
            raise ValueError("element quadrature dim must be positive")
        if n_shape <= 0:
            raise ValueError("element quadrature n_shape must be positive")
        if not weights:
            raise ValueError("element quadrature rule must have at least one point")
        expected = len(weights) * n_shape * dim
        if len(reference_gradients) != expected:
            raise ValueError(
                "reference_gradients must have N_QP * N_SHAPE * dim entries"
            )
        has_tensor_product_data = (
            bool(tensor_product_shape_values_1d)
            or bool(tensor_product_shape_gradients_1d)
            or bool(tensor_product_weights_1d)
            or tensor_product_dim != 0
        )
        if has_tensor_product_data:
            if tensor_product_dim != dim:
                raise ValueError("tensor_product_dim must match dim")
            if not tensor_product_weights_1d:
                raise ValueError("tensor-product quadrature must provide 1D weights")
            n_qp_1d = len(tensor_product_weights_1d)
            expected_1d = n_qp_1d * self.tensor_product_n_shape_1d
            if len(tensor_product_shape_values_1d) != expected_1d:
                raise ValueError("tensor-product shape values must be N_QP_1D * N_SHAPE_1D")
            if len(tensor_product_shape_gradients_1d) != expected_1d:
                raise ValueError("tensor-product shape gradients must be N_QP_1D * N_SHAPE_1D")
            if n_shape != self.tensor_product_n_shape_1d ** dim:
                raise ValueError("tensor-product n_shape does not match 1D shape count")
            if len(weights) != n_qp_1d ** dim:
                raise ValueError("tensor-product weights do not match 1D quadrature count")

    @property
    def n_qp(self):
        return len(self.weights)

    @property
    def is_tensor_product(self):
        return bool(self.tensor_product_weights_1d)

    @property
    def tensor_product_n_qp_1d(self):
        return len(self.tensor_product_weights_1d)

    @property
    def tensor_product_n_shape_1d(self):
        if not self.tensor_product_weights_1d:
            return 0
        return len(self.tensor_product_shape_values_1d) // len(self.tensor_product_weights_1d)


@dataclass(frozen=True)
class SfemSoAElementSpecialization:
    quadrature_rule: SfemElementQuadratureRule
    vector_size: int = 8

    def __post_init__(self):
        object.__setattr__(self, "vector_size", int(self.vector_size))
        if self.vector_size <= 0:
            raise ValueError("vector_size must be positive")

    @property
    def element_type(self):
        return self.quadrature_rule.element_type

    @property
    def dim(self):
        return self.quadrature_rule.dim

    @property
    def n_shape(self):
        return self.quadrature_rule.n_shape

    @property
    def n_qp(self):
        return self.quadrature_rule.n_qp

    def reference_shape_gradient_input(self, name="grad_ref"):
        return sfem_soa_reference_input(name, self.n_qp, self.n_shape, self.dim)

    def adjugate_geometry_inputs(
        self,
        grad_ref_name="grad_ref",
        adjugate_name="jacobian_adjugate",
        determinant_name="jacobian_determinant",
    ):
        return sfem_soa_adjugate_geometry_inputs(
            self,
            grad_ref_name=grad_ref_name,
            adjugate_name=adjugate_name,
            determinant_name=determinant_name,
        )


def sfem_element_quadrature_rule(element_type, order=None):
    element_type = str(element_type).upper()
    if order is None:
        order = (
            3
            if element_type == "HEX27"
            else 2
            if element_type in ("QUAD4", "HEX8", "TRI6", "TET10")
            else 1
        )
    order = int(order)

    if element_type == "TRI3":
        weights = (0.5,)
        gradients = (-1.0, -1.0, 1.0, 0.0, 0.0, 1.0)
        return SfemElementQuadratureRule(element_type, 2, 3, weights, gradients, order)

    if element_type == "TET4":
        weights = (1.0 / 6.0,)
        gradients = (
            -1.0,
            -1.0,
            -1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        return SfemElementQuadratureRule(element_type, 3, 4, weights, gradients, order)

    if element_type == "TRI6":
        if order != 2:
            raise ValueError("TRI6 currently supports quadrature order 2")
        points = (
            (1.0 / 6.0, 1.0 / 6.0),
            (2.0 / 3.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0),
        )
        weights = (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
        gradients = []
        for x, y in points:
            gradients.extend(_tri6_reference_gradients(x, y))
        return SfemElementQuadratureRule(element_type, 2, 6, weights, gradients, order)

    if element_type == "TET10":
        if order != 2:
            raise ValueError("TET10 currently supports quadrature order 2")
        a = 0.5854101966249685
        b = 0.1381966011250105
        points = (
            (b, b, b),
            (a, b, b),
            (b, a, b),
            (b, b, a),
        )
        weights = (1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0)
        gradients = []
        for x, y, z in points:
            gradients.extend(_tet10_reference_gradients(x, y, z))
        return SfemElementQuadratureRule(element_type, 3, 10, weights, gradients, order)

    if element_type == "QUAD4":
        points, weights_1d = _sfem_unit_interval_gauss_rule(order)
        shape_values_1d, shape_gradients_1d = _sfem_lagrange_q1_1d_shapes(points)
        gradients = []
        weights = []
        for y, wy in zip(points, weights_1d):
            for x, wx in zip(points, weights_1d):
                weights.append(wx * wy)
                gradients.extend(
                    (
                        -(1.0 - y),
                        -(1.0 - x),
                        1.0 - y,
                        -x,
                        y,
                        x,
                        -y,
                        1.0 - x,
                    )
                )
        return SfemElementQuadratureRule(
            element_type,
            2,
            4,
            weights,
            gradients,
            order,
            shape_values_1d,
            shape_gradients_1d,
            weights_1d,
            2,
        )

    if element_type == "HEX8":
        points, weights_1d = _sfem_unit_interval_gauss_rule(order)
        shape_values_1d, shape_gradients_1d = _sfem_lagrange_q1_1d_shapes(points)
        gradients = []
        weights = []
        for z, wz in zip(points, weights_1d):
            for y, wy in zip(points, weights_1d):
                for x, wx in zip(points, weights_1d):
                    weights.append(wx * wy * wz)
                    xm = 1.0 - x
                    ym = 1.0 - y
                    zm = 1.0 - z
                    gradients.extend(
                        (
                            -ym * zm,
                            -xm * zm,
                            -xm * ym,
                            ym * zm,
                            -x * zm,
                            -x * ym,
                            y * zm,
                            x * zm,
                            -x * y,
                            -y * zm,
                            xm * zm,
                            -xm * y,
                            -ym * z,
                            -xm * z,
                            xm * ym,
                            ym * z,
                            -x * z,
                            x * ym,
                            y * z,
                            x * z,
                            x * y,
                            -y * z,
                            xm * z,
                            xm * y,
                        )
                    )
        return SfemElementQuadratureRule(
            element_type,
            3,
            8,
            weights,
            gradients,
            order,
            shape_values_1d,
            shape_gradients_1d,
            weights_1d,
            3,
        )

    if element_type == "HEX27":
        if order != 3:
            raise ValueError("HEX27 currently supports quadrature order 3")
        points, weights_1d = _sfem_unit_interval_gauss_rule(order)
        shape_values_1d, shape_gradients_1d = _sfem_lagrange_q2_1d_shapes(points)
        gradients, weights = _sfem_tensor_product_hex_gradients_and_weights(
            shape_values_1d,
            shape_gradients_1d,
            weights_1d,
            3,
        )
        return SfemElementQuadratureRule(
            element_type,
            3,
            27,
            weights,
            gradients,
            order,
            shape_values_1d,
            shape_gradients_1d,
            weights_1d,
            3,
        )

    raise ValueError("unsupported element type '%s'" % element_type)


def sfem_supported_element_types():
    return ("TRI3", "TRI6", "QUAD4", "TET4", "TET10", "HEX8", "HEX27")


def sfem_soa_element_specializations(element_types=None, vector_size=8, quadrature_order=None):
    element_types = sfem_supported_element_types() if element_types is None else tuple(element_types)
    return tuple(
        sfem_soa_element_specialization(element_type, vector_size, quadrature_order)
        for element_type in element_types
    )


def sfem_soa_element_specialization(element_type, vector_size=8, quadrature_order=None):
    return SfemSoAElementSpecialization(
        sfem_element_quadrature_rule(element_type, quadrature_order),
        vector_size,
    )


def _tri6_reference_gradients(x, y):
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


def _tet10_reference_gradients(x, y, z):
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


def sfem_soa_adjugate_geometry_inputs(
    specialization,
    grad_ref_name="grad_ref",
    adjugate_name="jacobian_adjugate",
    determinant_name="jacobian_determinant",
):
    if isinstance(specialization, SfemSoAElementSpecialization):
        dim = specialization.dim
        n_qp = specialization.n_qp
        n_shape = specialization.n_shape
    elif isinstance(specialization, SfemElementQuadratureRule):
        dim = specialization.dim
        n_qp = specialization.n_qp
        n_shape = specialization.n_shape
    else:
        raise TypeError("specialization must be SfemSoAElementSpecialization or SfemElementQuadratureRule")
    return (
        sfem_soa_reference_input(grad_ref_name, n_qp, n_shape, dim),
        sfem_soa_array_input(adjugate_name, dim * dim),
        sfem_soa_array_input(determinant_name, 1),
    )


def _sfem_unit_interval_gauss_rule(order):
    if order == 1:
        return (0.5,), (1.0,)
    if order == 2:
        offset = 0.5 / math.sqrt(3.0)
        return (0.5 - offset, 0.5 + offset), (0.5, 0.5)
    if order == 3:
        offset = 0.5 * math.sqrt(3.0 / 5.0)
        return (0.5 - offset, 0.5, 0.5 + offset), (5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0)
    raise ValueError("unsupported quadrature order %d" % order)


def _sfem_lagrange_q1_1d_shapes(points):
    shape_values = []
    shape_gradients = []
    for x in points:
        shape_values.extend((1.0 - x, x))
        shape_gradients.extend((-1.0, 1.0))
    return tuple(shape_values), tuple(shape_gradients)


def _sfem_lagrange_q2_1d_shapes(points):
    shape_values = []
    shape_gradients = []
    for x in points:
        shape_values.extend(
            (
                2.0 * x * x - 3.0 * x + 1.0,
                4.0 * x - 4.0 * x * x,
                2.0 * x * x - x,
            )
        )
        shape_gradients.extend((4.0 * x - 3.0, 4.0 - 8.0 * x, 4.0 * x - 1.0))
    return tuple(shape_values), tuple(shape_gradients)


def _sfem_tensor_product_hex_gradients_and_weights(
    shape_values_1d,
    shape_gradients_1d,
    weights_1d,
    n_shape_1d,
):
    n_qp_1d = len(weights_1d)
    gradients = []
    weights = []
    for qz in range(n_qp_1d):
        for qy in range(n_qp_1d):
            for qx in range(n_qp_1d):
                weights.append(weights_1d[qx] * weights_1d[qy] * weights_1d[qz])
                for sz in range(n_shape_1d):
                    for sy in range(n_shape_1d):
                        for sx in range(n_shape_1d):
                            sxv = shape_values_1d[qx * n_shape_1d + sx]
                            syv = shape_values_1d[qy * n_shape_1d + sy]
                            szv = shape_values_1d[qz * n_shape_1d + sz]
                            dx = shape_gradients_1d[qx * n_shape_1d + sx] * syv * szv
                            dy = sxv * shape_gradients_1d[qy * n_shape_1d + sy] * szv
                            dz = sxv * syv * shape_gradients_1d[qz * n_shape_1d + sz]
                            gradients.extend((dx, dy, dz))
    return tuple(gradients), tuple(weights)


@dataclass(frozen=True)
class ExpressionCost:
    adds: int = 0
    muls: int = 0
    divs: int = 0
    sqrts: int = 0
    pows: int = 0
    logs: int = 0
    trigs: int = 0
    loads: int = 0
    stores: int = 0
    temporaries: int = 0
    estimated_registers: int = 0

    @property
    def flops(self):
        return (
            self.adds
            + self.muls
            + 8 * self.divs
            + 12 * self.sqrts
            + self.pows
            + 20 * self.logs
            + 24 * self.trigs
        )


@dataclass(frozen=True)
class EvaluationStatement:
    target: object
    expression: sp.Expr
    kind: str
    dependencies: Tuple[sp.Symbol, ...]
    cost: ExpressionCost
    role: Optional[ExpressionRole] = None
    output_index: Optional[int] = None
    augmented: bool = False
    scopes: Tuple[ScopeKind, ...] = ()
    hoist_scope: ScopeKind = ScopeKind.MESH


@dataclass(frozen=True)
class LivenessState:
    statement_index: int
    target: object
    live_temporaries_after: Tuple[sp.Symbol, ...]
    register_pressure: int


@dataclass(frozen=True)
class EvaluationMetrics:
    total_flops: int
    total_loads: int
    total_stores: int
    peak_registers: int
    peak_live_temporaries: int
    liveness: Tuple[LivenessState, ...]


@dataclass(frozen=True)
class EvaluationPlan:
    statements: Tuple[EvaluationStatement, ...]
    intermediates: Tuple[EvaluationStatement, ...]
    outputs: Tuple[EvaluationStatement, ...]
    metrics: EvaluationMetrics

    @property
    def temporary_symbols(self):
        return tuple(statement.target for statement in self.intermediates)


@dataclass(frozen=True)
class ExpressionGraph:
    graph: nx.DiGraph
    outputs: Tuple[KernelExpression, ...]
    intermediates: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    reduced_outputs: Tuple[SympyExpr, ...]
    patterns: Tuple[ExpressionPattern, ...]
    evaluation_plan: EvaluationPlan
    cost: ExpressionCost
    scopes: Tuple[ExecutionScope, ...] = ()
    template_parameters: Tuple[KernelTemplateParameter, ...] = ()
    specialization: Optional[DimensionSpecialization] = None

    def topological_nodes(self):
        return tuple(nx.topological_sort(self.graph))

    def patterns_by_kind(self, kind):
        kind = PatternKind(kind)
        return tuple(pattern for pattern in self.patterns if pattern.kind == kind)

    def scope_symbols(self, kind):
        kind = ScopeKind(kind)
        symbols = []
        for scope in self.scopes:
            if scope.kind == kind:
                symbols.extend(scope.symbols)
        return tuple(symbols)


class KernelExpressions:
    def __init__(self, expressions: Optional[Iterable[KernelExpression]] = None):
        self._expressions = []
        if expressions is not None:
            for expr in expressions:
                self.add(expr.role, expr.expression, expr.name)

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

    def build_graph(
        self,
        *,
        data_symbols: Optional[Iterable[sp.Symbol]] = None,
        loop_symbols: Optional[Mapping[str, Iterable[sp.Symbol]]] = None,
        scopes: Optional[Iterable[ExecutionScope]] = None,
        symbolic_objects: Optional[Iterable[SymbolicObject]] = None,
        template_parameters: Optional[Iterable[KernelTemplateParameter]] = None,
        specialization: Optional[DimensionSpecialization] = None,
        temporary_prefix="t",
        temporary_symbols=None,
        optimizations="basic",
    ):
        return build_expression_graph(
            self._expressions,
            data_symbols=data_symbols,
            loop_symbols=loop_symbols,
            scopes=scopes,
            symbolic_objects=symbolic_objects,
            template_parameters=template_parameters,
            specialization=specialization,
            temporary_prefix=temporary_prefix,
            temporary_symbols=temporary_symbols,
            optimizations=optimizations,
        )

    def __iter__(self):
        return iter(self._expressions)

    def __len__(self):
        return len(self._expressions)


def build_expression_graph(
    expressions: Iterable[KernelExpression],
    *,
    data_symbols: Optional[Iterable[sp.Symbol]] = None,
    loop_symbols: Optional[Mapping[str, Iterable[sp.Symbol]]] = None,
    scopes: Optional[Iterable[ExecutionScope]] = None,
    symbolic_objects: Optional[Iterable[SymbolicObject]] = None,
    template_parameters: Optional[Iterable[KernelTemplateParameter]] = None,
    specialization: Optional[DimensionSpecialization] = None,
    temporary_prefix="t",
    temporary_symbols=None,
    optimizations="basic",
):
    outputs = tuple(_normalize_kernel_expression(expr) for expr in expressions)
    reduced_inputs = [_rhs(expr.expression) for expr in outputs]
    cse_symbols = (
        temporary_symbols
        if temporary_symbols is not None
        else sp.numbered_symbols(temporary_prefix)
    )
    intermediates, reduced_rhs = sp.cse(
        reduced_inputs,
        symbols=cse_symbols,
        optimizations=optimizations,
    )
    reduced_outputs = _reattach_lhs(outputs, reduced_rhs)

    graph = nx.DiGraph()
    data_symbol_set = set(data_symbols or ())
    symbolic_objects = tuple(symbolic_objects or ())
    kernel_template_parameters = _template_parameters(
        symbolic_objects,
        template_parameters,
    )
    graph.graph["template_parameters"] = kernel_template_parameters
    kernel_specialization = _dimension_specialization(
        symbolic_objects,
        specialization,
    )
    graph.graph["specialization"] = kernel_specialization
    layout_symbol_map = _layout_symbol_map(symbolic_objects)
    execution_scopes = _normalize_scopes(loop_symbols, scopes)
    scope_symbol_map = _scope_symbol_map(execution_scopes)

    for scope in execution_scopes:
        for symbol in scope.symbols:
            _add_node(
                graph,
                symbol,
                "loop_index",
                scope=scope.kind.value,
                scope_name=scope.name,
                scope_kind=scope.kind,
            )

    for symbol in data_symbol_set:
        _add_node(graph, symbol, "data")
        _annotate_data_layout(graph, symbol, layout_symbol_map)

    for var, expr in intermediates:
        _add_expression_node(
            graph,
            var,
            expr,
            data_symbol_set,
            scope_symbol_map,
            layout_symbol_map,
            "intermediate",
        )
        graph.nodes[var]["scopes"] = _expression_scopes(expr, scope_symbol_map)

    output_nodes = []
    for idx, (kernel_expr, reduced_expr) in enumerate(zip(outputs, reduced_outputs)):
        output_node = _output_node_name(kernel_expr, idx)
        output_nodes.append(output_node)
        graph.add_node(
            output_node,
            kind="output",
            role=kernel_expr.role.value,
            name=kernel_expr.name,
            expression=reduced_expr,
            scopes=_expression_scopes(_rhs(reduced_expr), scope_symbol_map),
        )
        for dep in _dependencies(_rhs(reduced_expr)):
            _ensure_dependency_node(graph, dep, data_symbol_set, scope_symbol_map, layout_symbol_map)
            graph.add_edge(dep, output_node)

    patterns = _detect_patterns(
        graph,
        intermediates,
        reduced_outputs,
        output_nodes,
        symbolic_objects,
    )
    evaluation_plan = _build_evaluation_plan(
        intermediates,
        reduced_outputs,
        output_nodes,
        outputs,
        data_symbol_set,
        scope_symbol_map,
    )
    _annotate_graph_scope_placements(graph, evaluation_plan, output_nodes)
    cost = _expression_cost(
        intermediates,
        reduced_outputs,
        data_symbol_set,
        evaluation_plan.metrics.peak_registers,
    )
    return ExpressionGraph(
        graph,
        outputs,
        tuple(intermediates),
        reduced_outputs,
        patterns,
        evaluation_plan,
        cost,
        execution_scopes,
        kernel_template_parameters,
        kernel_specialization,
    )


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
):
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
            "#pragma omp parallel for",
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
    return GeneratedKernelCode("c++", function_name, "\n".join(lines))


def generate_sfem_soa_cpp_files(
    forms,
    *,
    prefix,
    dim,
    n_nodes,
    n_qp=1,
    vector_size=8,
    array_inputs=None,
    element_type=None,
    quadrature_order=None,
    quadrature_rule=None,
    local_prefix=None,
):
    forms = tuple(forms)
    if quadrature_rule is None and element_type is not None:
        quadrature_rule = sfem_element_quadrature_rule(element_type, quadrature_order)
    if quadrature_rule is not None:
        dim = quadrature_rule.dim
        n_nodes = quadrature_rule.n_shape
        n_qp = quadrature_rule.n_qp
    n_qp = int(n_qp)
    array_inputs = tuple(
        array_inputs
        if array_inputs is not None
        else (sfem_soa_reference_input("grad_ref", n_qp, n_nodes, dim),)
    )
    if dim < 1 or dim > 3:
        raise ValueError("SoA backend currently supports dimensions 1, 2, and 3")
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if n_qp <= 0:
        raise ValueError("n_qp must be positive")
    if vector_size <= 0:
        raise ValueError("vector_size must be positive")
    for array_input in _sfem_soa_reference_inputs(array_inputs):
        if array_input.n_qp != n_qp:
            raise ValueError(
                "reference input '%s' has n_qp=%d, expected %d"
                % (array_input.name, array_input.n_qp, n_qp)
            )
        if array_input.n_shape != n_nodes:
            raise ValueError(
                "reference input '%s' has n_shape=%d, expected %d"
                % (array_input.name, array_input.n_shape, n_nodes)
            )
    if quadrature_rule is not None:
        _validate_sfem_soa_quadrature_rule(quadrature_rule, dim, n_nodes, n_qp, array_inputs)

    local_prefix = prefix if local_prefix is None else str(local_prefix)
    use_shared_weak_local = local_prefix != prefix
    local_name = "%s_local.hpp" % local_prefix
    math_name = "kernel_math.hpp"
    diagnostics_name = "kernel_diagnostics.hpp"
    operator_name = "%s_operator.cpp" % prefix
    return (
        GeneratedKernelFile(
            math_name,
            _sfem_math_header_source(),
        ),
        GeneratedKernelFile(
            diagnostics_name,
            "\n".join(_sfem_soa_diagnostics_header()),
        ),
        GeneratedKernelFile(
            local_name,
            _sfem_soa_local_header(
                forms,
                local_prefix,
                dim,
                n_nodes,
                array_inputs,
                quadrature_rule,
                use_shared_weak_local,
                math_name,
            ),
        ),
        GeneratedKernelFile(
            operator_name,
            _sfem_soa_operator_source(
                forms,
                prefix,
                dim,
                n_nodes,
                n_qp,
                vector_size,
                local_prefix,
                local_name,
                diagnostics_name,
                array_inputs,
                quadrature_rule,
                use_shared_weak_local,
            ),
        ),
    )


def generate_sfem_soa_cpp_files_for_element(
    forms,
    *,
    prefix,
    specialization,
    array_inputs=None,
    local_prefix=None,
):
    if isinstance(specialization, SfemElementQuadratureRule):
        specialization = SfemSoAElementSpecialization(specialization)
    if not isinstance(specialization, SfemSoAElementSpecialization):
        raise TypeError("specialization must be an SfemSoAElementSpecialization")
    array_inputs = (
        specialization.adjugate_geometry_inputs()
        if array_inputs is None
        else tuple(array_inputs)
    )
    return generate_sfem_soa_cpp_files(
        forms,
        prefix=prefix,
        dim=specialization.dim,
        n_nodes=specialization.n_shape,
        n_qp=specialization.n_qp,
        vector_size=specialization.vector_size,
        array_inputs=array_inputs,
        quadrature_rule=specialization.quadrature_rule,
        local_prefix=local_prefix,
    )


def _sfem_soa_local_header(
    forms,
    prefix,
    dim,
    n_nodes,
    array_inputs,
    quadrature_rule,
    use_shared_weak_local=False,
    math_name="kernel_math.hpp",
):
    guard = "%s_LOCAL_HPP" % _cpp_macro_name(prefix)
    lines = [
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "",
        "#include <math.h>",
        "#include <stddef.h>",
        '#include "%s"' % math_name,
        "",
        "#ifndef SFEM_INLINE",
        "#define SFEM_INLINE inline",
        "#endif",
        "",
        "#ifndef SFEM_RESTRICT",
        "#define SFEM_RESTRICT",
        "#endif",
        "",
        "#ifndef SFEM_GENERATED_SCALAR_T",
        "#define SFEM_GENERATED_SCALAR_T",
        "typedef double real_t;",
        "#endif",
        "",
    ]
    if use_shared_weak_local and quadrature_rule is not None and quadrature_rule.is_tensor_product:
        lines.extend(
            [
                "#ifndef SFEM_GENERATED_INTEGER_ROOT",
                "#define SFEM_GENERATED_INTEGER_ROOT",
                "static constexpr int sfem_generated_ipow(const int base, const int exponent) {",
                "    return exponent == 0 ? 1 : base * sfem_generated_ipow(base, exponent - 1);",
                "}",
                "static constexpr int sfem_generated_integer_root_search(const int value, const int exponent, const int candidate) {",
                "    return sfem_generated_ipow(candidate, exponent) >= value ? candidate : sfem_generated_integer_root_search(value, exponent, candidate + 1);",
                "}",
                "static constexpr int sfem_generated_integer_root(const int value, const int exponent) {",
                "    return sfem_generated_integer_root_search(value, exponent, 1);",
                "}",
                "#endif",
                "",
            ]
        )
    lines.extend(["namespace sfem {", "namespace codegen {", ""])

    for form in forms:
        lines.extend(
            _sfem_soa_block_function(
                form,
                prefix,
                dim,
                n_nodes,
                array_inputs,
                quadrature_rule,
                use_shared_weak_local,
            )
        )
        lines.append("")

    lines.extend(["} // namespace codegen", "} // namespace sfem", "", "#endif", ""])
    return "\n".join(lines)


def _sfem_soa_block_function(
    form,
    prefix,
    dim,
    n_nodes,
    array_inputs,
    quadrature_rule,
    use_shared_weak_local=False,
):
    name = "%s_%s_block" % (prefix, form.name)
    element_inputs = _sfem_soa_element_inputs(array_inputs)
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    use_tensor_product_reference = (
        quadrature_rule is not None
        and quadrature_rule.is_tensor_product
        and len(reference_inputs) == 1
        and reference_inputs[0].name == "grad_ref"
    )
    use_reference_gradient_vectors = (
        form.weak_form is not None
        and not use_tensor_product_reference
        and len(reference_inputs) == 1
        and reference_inputs[0].name == "grad_ref"
    )
    use_stream_arrays = use_shared_weak_local and form.weak_form is not None
    params = ["const ptrdiff_t nelems", "const int q"]
    params.extend(
        "const %s *const SFEM_RESTRICT %s" % (array_input.scalar_type, stream)
        for array_input in element_inputs
        for stream in _soa_array_stream_names(array_input)
    )
    if use_tensor_product_reference:
        params.extend(
            (
                "const scalar_t *const SFEM_RESTRICT shape_1d",
                "const scalar_t *const SFEM_RESTRICT grad_1d",
            )
        )
    elif use_reference_gradient_vectors:
        params.extend(_sfem_reference_gradient_vector_params(dim))
    else:
        params.extend(
            "const %s *const SFEM_RESTRICT %s"
            % (array_input.scalar_type, _sfem_soa_reference_param_name(array_input))
            for array_input in reference_inputs
        )
    params.extend(("const scalar_t qw", "const scalar_t mu", "const scalar_t lmbda"))
    if use_stream_arrays:
        params.extend(
            (
                "const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * %d]" % dim,
            )
        )
        if form.has_direction:
            params.extend(
                (
                    "const scalar_t *const SFEM_RESTRICT h_streams[N_SHAPE * %d]"
                    % dim,
                )
            )
        if form.name == "objective":
            params.extend(("scalar_t *const SFEM_RESTRICT value",))
        else:
            params.extend(
                (
                    "scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * %d]"
                    % dim,
                )
            )
    else:
        params.extend(
            "const scalar_t *const SFEM_RESTRICT %s" % name
            for name in _field_stream_names("u", dim, n_nodes)
        )
        if form.has_direction:
            params.extend(
                "const scalar_t *const SFEM_RESTRICT %s" % name
                for name in _field_stream_names("h", dim, n_nodes)
            )
        params.extend(
            "scalar_t *const SFEM_RESTRICT %s" % name
            for name in _output_stream_names(form, dim, n_nodes)
        )

    lines = [
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>",
        "static SFEM_INLINE void %s(" % name,
    ]
    for idx, param in enumerate(params):
        comma = "," if idx + 1 < len(params) else ""
        lines.append("        %s%s" % (param, comma))
    lines.extend(
        [
            ") {",
            "    static_assert(N_QP > 0, \"N_QP must be positive\");",
            "    static_assert(VECTOR_SIZE > 0, \"VECTOR_SIZE must be positive\");",
        ]
    )
    if not use_stream_arrays:
        lines.append(
            "    static_assert(N_SHAPE == %d, \"N_SHAPE does not match generated expression\");"
            % n_nodes
        )
    if use_tensor_product_reference:
        if use_stream_arrays:
            lines.extend(
                [
                    "    static constexpr int N_QP_1D = sfem_generated_integer_root(N_QP, %d);"
                    % quadrature_rule.dim,
                    "    static constexpr int N_SHAPE_1D = sfem_generated_integer_root(N_SHAPE, %d);"
                    % quadrature_rule.dim,
                    "    static_assert(sfem_generated_ipow(N_QP_1D, %d) == N_QP, \"N_QP must be tensor-product compatible\");"
                    % quadrature_rule.dim,
                    "    static_assert(sfem_generated_ipow(N_SHAPE_1D, %d) == N_SHAPE, \"N_SHAPE must be tensor-product compatible\");"
                    % quadrature_rule.dim,
                ]
            )
        else:
            lines.extend(
                [
                    "    static constexpr int N_QP_1D = %d;"
                    % quadrature_rule.tensor_product_n_qp_1d,
                    "    static constexpr int N_SHAPE_1D = %d;"
                    % quadrature_rule.tensor_product_n_shape_1d,
                ]
            )
        lines.extend(_tensor_product_q_index_lines(quadrature_rule.dim, "    "))
    lines.extend(
        [
            "#pragma omp simd",
            "    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {",
            "        scalar_t u[N_SHAPE * %d];" % dim,
        ]
    )
    for array_input in array_inputs:
        if form.weak_form is not None and array_input.is_reference_qp_shape:
            continue
        if array_input.is_reference_qp_shape:
            array_decl = "%s %s[N_SHAPE * %d];" % (
                array_input.scalar_type,
                array_input.name,
                array_input.components,
            )
        else:
            array_decl = "%s %s[%d];" % (
                array_input.scalar_type,
                array_input.name,
                array_input.local_size,
            )
        lines.insert(
            -1,
            "        %s" % array_decl,
        )
    if form.has_direction:
        lines.append("        scalar_t du[N_SHAPE * %d];" % dim)

    for array_input in element_inputs:
        for i, stream in enumerate(_soa_array_stream_names(array_input)):
            lines.append("        %s[%d] = %s[lane];" % (array_input.name, i, stream))
    if form.weak_form is not None:
        pass
    elif use_tensor_product_reference:
        _append_tensor_product_reference_gradient_lines(
            lines,
            reference_inputs[0].name,
            quadrature_rule,
        )
    else:
        for array_input in reference_inputs:
            source = _sfem_soa_reference_param_name(array_input)
            for shape in range(array_input.n_shape):
                for component in range(array_input.components):
                    local_idx = shape * array_input.components + component
                    lines.append(
                        "        %s[%d] = %s[(q * N_SHAPE + %d) * %d + %d];"
                        % (
                            array_input.name,
                            local_idx,
                            source,
                            shape,
                            array_input.components,
                            component,
                        )
                    )

    if use_stream_arrays:
        lines.extend(["        for (int shape = 0; shape < N_SHAPE; ++shape) {"])
        for d in range(dim):
            lines.append(
                "            u[shape * %d + %d] = u_streams[shape * %d + %d][lane];"
                % (dim, d, dim, d)
            )
            if form.has_direction:
                lines.append(
                    "            du[shape * %d + %d] = h_streams[shape * %d + %d][lane];"
                    % (dim, d, dim, d)
                )
        lines.append("        }")
    else:
        for node in range(n_nodes):
            for d in range(dim):
                idx = node * dim + d
                component = _component_name(d)
                lines.append("        u[%d] = u%s%d[lane];" % (idx, component, node))
                if form.has_direction:
                    lines.append("        du[%d] = h%s%d[lane];" % (idx, component, node))

    if form.weak_form is not None:
        _append_sfem_soa_weak_form_lines(
            lines,
            form,
            dim,
            n_nodes,
            reference_inputs,
            use_tensor_product_reference,
            quadrature_rule,
            use_stream_arrays,
        )
        lines.extend(["    }", "}"])
        return lines

    output_count = len(form.expression_graph.evaluation_plan.outputs)
    lines.append("        scalar_t element_vector[%d];" % max(1, output_count))
    _append_sfem_soa_statement_lines(lines, form.expression_graph, "element_vector")
    _append_sfem_soa_output_lines(lines, form, dim, n_nodes)
    lines.extend(["    }", "}"])
    return lines


def _append_sfem_soa_weak_form_lines(
    lines,
    form,
    dim,
    n_nodes,
    reference_inputs,
    use_tensor_product_reference,
    quadrature_rule,
    use_stream_arrays=False,
):
    weak_form = form.weak_form
    if weak_form.dim != dim:
        raise ValueError("weak form dim does not match SoA kernel dim")
    if form.name not in ("objective", "gradient", "apply"):
        raise ValueError("weak form kernel name must be objective, gradient, or apply")
    if form.name == "apply" and not form.has_direction:
        raise ValueError("weak form apply kernel requires has_direction=True")
    if len(reference_inputs) != 1 or reference_inputs[0].name != "grad_ref":
        raise ValueError("weak form kernels require one grad_ref reference input")

    def reference_gradient(component):
        if use_tensor_product_reference:
            return _tensor_product_dynamic_reference_gradient_expr(dim, component)
        return "%s[q * N_SHAPE + shape]" % _sfem_reference_gradient_vector_name(component)

    lines.extend(
        [
            "        scalar_t grad_u_ref[%d];" % (dim * dim),
            "        scalar_t grad_u[%d];" % (dim * dim),
        ]
    )
    if form.has_direction:
        lines.extend(
            [
                "        scalar_t grad_h_ref[%d];" % (dim * dim),
                "        scalar_t trial_grad[%d];" % (dim * dim),
            ]
        )
    for row in range(dim):
        for col in range(dim):
            idx = row * dim + col
            lines.append("        grad_u_ref[%d] = 0.0;" % idx)
            if form.has_direction:
                lines.append("        grad_h_ref[%d] = 0.0;" % idx)

    lines.extend(["        for (int shape = 0; shape < N_SHAPE; ++shape) {"])
    if use_tensor_product_reference:
        if use_stream_arrays:
            lines.extend(_tensor_product_generic_shape_index_lines(dim, "            "))
        else:
            lines.extend(_tensor_product_shape_index_lines(quadrature_rule, "            "))
    for row in range(dim):
        for col in range(dim):
            lines.append(
                "            grad_u_ref[%d] += u[shape * %d + %d] * %s;"
                % (row * dim + col, dim, row, reference_gradient(col))
            )
            if form.has_direction:
                lines.append(
                    "            grad_h_ref[%d] += du[shape * %d + %d] * %s;"
                    % (row * dim + col, dim, row, reference_gradient(col))
                )
    lines.append("        }")

    lines.append("        const scalar_t inv_jacobian_determinant = 1.0 / jacobian_determinant[0];")
    for row in range(dim):
        for col in range(dim):
            terms = [
                "grad_u_ref[%d] * jacobian_adjugate[%d]"
                % (row * dim + k, k * dim + col)
                for k in range(dim)
            ]
            lines.append(
                "        grad_u[%d] = (%s) * inv_jacobian_determinant;"
                % (row * dim + col, " + ".join(terms))
            )
            if form.has_direction:
                terms = [
                    "grad_h_ref[%d] * jacobian_adjugate[%d]"
                    % (row * dim + k, k * dim + col)
                    for k in range(dim)
                ]
                lines.append(
                    "        trial_grad[%d] = (%s) * inv_jacobian_determinant;"
                    % (row * dim + col, " + ".join(terms))
                )

    deformation_gradient_substitutions = _weak_form_deformation_gradient_substitutions(
        weak_form,
        "grad_u",
    )

    if form.name == "objective":
        _append_cse_array_assignments(
            lines,
            [weak_form.energy_density.xreplace(deformation_gradient_substitutions)],
            ["value[lane] %s" % ("+=" if form.output_mode == "accumulate" else "=")],
            "weak_obj_tmp",
            scale="qw * jacobian_determinant[0]",
        )
        return

    material = (
        weak_form.linearized_first_piola(
            tuple(sp.symbols("trial_grad[%d]" % i) for i in range(dim * dim))
        )
        if form.name == "apply"
        else weak_form.first_piola()
    ).xreplace(deformation_gradient_substitutions)
    lines.append("        scalar_t element_vector[N_SHAPE * %d];" % dim)
    lines.append("        scalar_t loperand[%d];" % (dim * dim))
    _append_transformed_loperand_lines(lines, material, dim, "weak_mat_tmp")

    lines.extend(["        for (int shape = 0; shape < N_SHAPE; ++shape) {"])
    if use_tensor_product_reference:
        if use_stream_arrays:
            lines.extend(_tensor_product_generic_shape_index_lines(dim, "            "))
        else:
            lines.extend(_tensor_product_shape_index_lines(quadrature_rule, "            "))
    for row in range(dim):
        terms = [
            "loperand[%d] * %s" % (row * dim + col, reference_gradient(col))
            for col in range(dim)
        ]
        lines.append(
            "            element_vector[shape * %d + %d] = %s;"
            % (dim, row, " + ".join(terms))
        )
    lines.append("        }")
    if use_stream_arrays:
        lines.extend(["        for (int shape = 0; shape < N_SHAPE; ++shape) {"])
        for row in range(dim):
            op = "+=" if form.output_mode == "accumulate" else "="
            lines.append(
                "            out_streams[shape * %d + %d][lane] %s element_vector[shape * %d + %d];"
                % (dim, row, op, dim, row)
            )
        lines.append("        }")
    else:
        for node in range(n_nodes):
            for row in range(dim):
                stream = "out%s%d" % (_component_name(row), node)
                op = "+=" if form.output_mode == "accumulate" else "="
                lines.append(
                    "        %s[lane] %s element_vector[%d];"
                    % (stream, op, node * dim + row)
                )


def _append_transformed_loperand_lines(lines, material, dim, temporary_prefix):
    material_exprs = tuple(material)
    material_names = ["material[%d] =" % i for i in range(dim * dim)]
    lines.append("        scalar_t material[%d];" % (dim * dim))
    _append_cse_array_assignments(lines, material_exprs, material_names, temporary_prefix)
    for row in range(dim):
        for col in range(dim):
            terms = [
                "material[%d] * jacobian_adjugate[%d]"
                % (row * dim + k, col * dim + k)
                for k in range(dim)
            ]
            lines.append(
                "        loperand[%d] = qw * (%s);"
                % (row * dim + col, " + ".join(terms))
            )


def _weak_form_deformation_gradient_substitutions(weak_form, gradient_name):
    substitutions = {}
    for row in range(weak_form.dim):
        for col in range(weak_form.dim):
            idx = row * weak_form.dim + col
            value = sp.Symbol("%s[%d]" % (gradient_name, idx))
            if row == col:
                value = sp.Integer(1) + value
            substitutions[weak_form.deformation_gradient[idx]] = value
    return substitutions


def _append_cse_array_assignments(lines, expressions, targets, temporary_prefix, scale=None):
    temps, reduced = sp.cse(
        tuple(expressions),
        symbols=sp.numbered_symbols("%s" % temporary_prefix),
    )
    for symbol, expression in temps:
        lines.append("        const scalar_t %s = %s;" % (symbol, _sfem_ccode(expression)))
    for target, expression in zip(targets, reduced):
        if scale is not None:
            lines.append("        %s %s * (%s);" % (target, scale, _sfem_ccode(expression)))
        else:
            lines.append("        %s %s;" % (target, _sfem_ccode(expression)))


def _append_sfem_soa_statement_lines(lines, expression_graph, output_name):
    output_index = 0
    for statement in expression_graph.evaluation_plan.statements:
        expression = _sfem_ccode(statement.expression)
        if statement.kind == "intermediate":
            lines.append("        const scalar_t %s = %s;" % (statement.target, expression))
        else:
            lines.append("        %s[%d] = %s;" % (output_name, output_index, expression))
            output_index += 1


def _tensor_product_q_index_lines(dim, indent):
    if dim == 2:
        return (
            "%sconst int qx = q %% N_QP_1D;" % indent,
            "%sconst int qy = q / N_QP_1D;" % indent,
        )
    if dim == 3:
        return (
            "%sconst int qx = q %% N_QP_1D;" % indent,
            "%sconst int qy = (q / N_QP_1D) %% N_QP_1D;" % indent,
            "%sconst int qz = q / (N_QP_1D * N_QP_1D);" % indent,
        )
    raise ValueError("tensor-product reference generation requires dim 2 or 3")


def _tensor_product_node_coords(quadrature_rule):
    dim = quadrature_rule.dim
    n_shape_1d = quadrature_rule.tensor_product_n_shape_1d
    if n_shape_1d == 2 and dim == 2:
        return ((0, 0), (1, 0), (1, 1), (0, 1))
    if n_shape_1d == 2 and dim == 3:
        return (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        )
    if dim == 2:
        return tuple(
            (sx, sy)
            for sy in range(n_shape_1d)
            for sx in range(n_shape_1d)
        )
    if dim == 3:
        return tuple(
            (sx, sy, sz)
            for sz in range(n_shape_1d)
            for sy in range(n_shape_1d)
            for sx in range(n_shape_1d)
        )
    raise ValueError("tensor-product node ordering requires dim 2 or 3")


def _tensor_product_shape_index_lines(quadrature_rule, indent):
    dim = quadrature_rule.dim
    n_shape_1d = quadrature_rule.tensor_product_n_shape_1d
    if n_shape_1d == 2 and dim == 2:
        return (
            "%sconst int sx = ((shape + 1) >> 1) & 1;" % indent,
            "%sconst int sy = shape >> 1;" % indent,
        )
    if n_shape_1d == 2 and dim == 3:
        return (
            "%sconst int sx = ((shape + 1) >> 1) & 1;" % indent,
            "%sconst int sy = (shape >> 1) & 1;" % indent,
            "%sconst int sz = shape >> 2;" % indent,
        )
    if dim == 2:
        return (
            "%sconst int sx = shape %% N_SHAPE_1D;" % indent,
            "%sconst int sy = shape / N_SHAPE_1D;" % indent,
        )
    if dim == 3:
        return (
            "%sconst int sx = shape %% N_SHAPE_1D;" % indent,
            "%sconst int sy = (shape / N_SHAPE_1D) %% N_SHAPE_1D;" % indent,
            "%sconst int sz = shape / (N_SHAPE_1D * N_SHAPE_1D);" % indent,
        )
    raise ValueError("tensor-product shape indices require dim 2 or 3")


def _tensor_product_generic_shape_index_lines(dim, indent):
    if dim == 2:
        return (
            "%sconst int sx = shape %% N_SHAPE_1D;" % indent,
            "%sconst int sy = shape / N_SHAPE_1D;" % indent,
        )
    if dim == 3:
        return (
            "%sconst int sx = shape %% N_SHAPE_1D;" % indent,
            "%sconst int sy = (shape / N_SHAPE_1D) %% N_SHAPE_1D;" % indent,
            "%sconst int sz = shape / (N_SHAPE_1D * N_SHAPE_1D);" % indent,
        )
    raise ValueError("tensor-product shape indices require dim 2 or 3")


def _sfem_reference_gradient_vector_name(component):
    return "grad_ref_%s" % _component_name(component)


def _sfem_reference_gradient_vector_params(dim):
    return tuple(
        "const scalar_t *const SFEM_RESTRICT %s"
        % _sfem_reference_gradient_vector_name(component)
        for component in range(dim)
    )


def _tensor_product_1d_factor(axis, node_axis, derivative_axis):
    qp_name = ("qx", "qy", "qz")[axis]
    table_name = "grad_1d" if axis == derivative_axis else "shape_1d"
    return "%s[%s * N_SHAPE_1D + %d]" % (table_name, qp_name, node_axis)


def _tensor_product_dynamic_reference_gradient_expr(dim, derivative_axis):
    factors = []
    for axis in range(dim):
        qp_name = ("qx", "qy", "qz")[axis]
        shape_name = ("sx", "sy", "sz")[axis]
        table_name = "grad_1d" if axis == derivative_axis else "shape_1d"
        factors.append("%s[%s * N_SHAPE_1D + %s]" % (table_name, qp_name, shape_name))
    return " * ".join(factors)


def _tensor_product_quadrature_weight_expr(dim):
    factors = ["q_weight_1d[%s]" % name for name in ("qx", "qy", "qz")[:dim]]
    return " * ".join(factors)


def _append_tensor_product_reference_gradient_lines(lines, name, quadrature_rule):
    for shape, coords in enumerate(_tensor_product_node_coords(quadrature_rule)):
        for component in range(quadrature_rule.dim):
            factors = [
                _tensor_product_1d_factor(axis, coords[axis], component)
                for axis in range(quadrature_rule.dim)
            ]
            lines.append(
                "        %s[%d] = %s;"
                % (name, shape * quadrature_rule.dim + component, " * ".join(factors))
            )


def _sfem_soa_isoparametric_geometry_lines(
    dim,
    n_nodes,
    quadrature_rule,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    reference_inputs,
):
    lines = ["#pragma omp simd", "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {"]
    for row in range(dim):
        for col in range(dim):
            lines.append("                scalar_t J%d%d = 0.0;" % (row, col))
    lines.append("                for (int shape = 0; shape < N_SHAPE; ++shape) {")
    if use_tensor_product_reference:
        lines.extend(_tensor_product_shape_index_lines(quadrature_rule, "                    "))
    for col in range(dim):
        lines.append(
            "                    const scalar_t g%d = %s;"
            % (
                col,
                _sfem_soa_isoparametric_reference_gradient_expr(
                    dim,
                    col,
                    use_tensor_product_reference,
                    use_reference_gradient_vectors,
                    reference_inputs,
                ),
            )
        )
    for row in range(dim):
        for col in range(dim):
            lines.append(
                "                    J%d%d += block_coordinate_streams[shape * %d + %d][lane] * g%d;"
                % (row, col, dim, row, col)
            )
    lines.append("                }")
    lines.extend(_sfem_soa_adjugate_assignment_lines(dim, "                "))
    lines.append("            }")
    return lines


def _sfem_soa_isoparametric_reference_gradient_expr(
    dim,
    component,
    use_tensor_product_reference,
    use_reference_gradient_vectors,
    reference_inputs,
):
    if use_tensor_product_reference:
        return _tensor_product_dynamic_reference_gradient_expr(dim, component)
    if use_reference_gradient_vectors:
        return "%s[q * N_SHAPE + shape]" % _sfem_reference_gradient_vector_name(component)
    if len(reference_inputs) == 1 and reference_inputs[0].name == "grad_ref":
        return "%s[(q * N_SHAPE + shape) * %d + %d]" % (
            reference_inputs[0].name,
            dim,
            component,
        )
    raise ValueError("isoparametric geometry generation requires one grad_ref reference input")


def _sfem_soa_adjugate_assignment_lines(dim, indent):
    if dim == 1:
        return (
            "%sblock_jacobian_determinant0[lane] = J00;" % indent,
            "%sblock_jacobian_adjugate0[lane] = 1.0;" % indent,
        )
    if dim == 2:
        return (
            "%sblock_jacobian_determinant0[lane] = J00 * J11 - J01 * J10;" % indent,
            "%sblock_jacobian_adjugate0[lane] = J11;" % indent,
            "%sblock_jacobian_adjugate1[lane] = -J01;" % indent,
            "%sblock_jacobian_adjugate2[lane] = -J10;" % indent,
            "%sblock_jacobian_adjugate3[lane] = J00;" % indent,
        )
    if dim == 3:
        return (
            "%sblock_jacobian_determinant0[lane] = J00 * (J11 * J22 - J12 * J21) - J01 * (J10 * J22 - J12 * J20) + J02 * (J10 * J21 - J11 * J20);" % indent,
            "%sblock_jacobian_adjugate0[lane] = J11 * J22 - J12 * J21;" % indent,
            "%sblock_jacobian_adjugate1[lane] = J02 * J21 - J01 * J22;" % indent,
            "%sblock_jacobian_adjugate2[lane] = J01 * J12 - J02 * J11;" % indent,
            "%sblock_jacobian_adjugate3[lane] = J12 * J20 - J10 * J22;" % indent,
            "%sblock_jacobian_adjugate4[lane] = J00 * J22 - J02 * J20;" % indent,
            "%sblock_jacobian_adjugate5[lane] = J02 * J10 - J00 * J12;" % indent,
            "%sblock_jacobian_adjugate6[lane] = J10 * J21 - J11 * J20;" % indent,
            "%sblock_jacobian_adjugate7[lane] = J01 * J20 - J00 * J21;" % indent,
            "%sblock_jacobian_adjugate8[lane] = J00 * J11 - J01 * J10;" % indent,
        )
    raise ValueError("isoparametric geometry supports dimensions 1, 2, and 3")


def _append_sfem_soa_output_lines(lines, form, dim, n_nodes):
    output_count = len(form.expression_graph.evaluation_plan.outputs)
    if output_count == 1:
        op = "+=" if form.output_mode == "accumulate" else "="
        lines.append("        value[lane] %s element_vector[0];" % op)
        return

    if output_count != dim * n_nodes:
        raise ValueError(
            "SoA form '%s' has %d outputs, expected 1 or dim*n_nodes=%d"
            % (form.name, output_count, dim * n_nodes)
        )

    for node in range(n_nodes):
        for d in range(dim):
            stream = "out%s%d" % (_component_name(d), node)
            idx = node * dim + d
            op = "+=" if form.output_mode == "accumulate" else "="
            lines.append("        %s[lane] %s element_vector[%d];" % (stream, op, idx))


def _sfem_soa_operator_source(
    forms,
    prefix,
    dim,
    n_nodes,
    n_qp,
    vector_size,
    local_prefix,
    local_name,
    diagnostics_name,
    array_inputs,
    quadrature_rule,
    use_shared_weak_local=False,
):
    lines = [
        '#include "%s"' % local_name,
        '#include "%s"' % diagnostics_name,
        "",
        "#ifndef SFEM_SUCCESS",
        "#define SFEM_SUCCESS 0",
        "#endif",
        "",
        "#ifndef MIN",
        "#define MIN(a, b) ((a) < (b) ? (a) : (b))",
        "#endif",
        "",
        "#ifdef _OPENMP",
        "#include <omp.h>",
        "#endif",
        "",
    ]
    if quadrature_rule is not None:
        lines.extend(["namespace sfem {", "namespace codegen {", ""])
        lines.extend(_sfem_soa_quadrature_rule_lines(prefix, quadrature_rule))
        lines.extend(["", "} // namespace codegen", "} // namespace sfem"])
        lines.append("")

    for form in forms:
        lines.extend(
            _sfem_soa_diagnostics_lines(
                form,
                prefix,
                dim,
                n_nodes,
                n_qp,
                vector_size,
                array_inputs,
                quadrature_rule,
            )
        )
        lines.append("")
        lines.extend(
            _sfem_soa_operator_function(
                form,
                prefix,
                dim,
                n_nodes,
                n_qp,
                vector_size,
                local_prefix,
                array_inputs,
                quadrature_rule,
                use_shared_weak_local,
            )
        )
        if _sfem_soa_has_adjugate_geometry_inputs(array_inputs, dim):
            lines.append("")
            lines.extend(
                _sfem_soa_operator_function(
                    form,
                    prefix,
                    dim,
                    n_nodes,
                    n_qp,
                    vector_size,
                    local_prefix,
                    array_inputs,
                    quadrature_rule,
                    use_shared_weak_local,
                    isoparametric_geometry=True,
                )
            )
        lines.append("")

    return "\n".join(lines)


def _sfem_soa_operator_function(
    form,
    prefix,
    dim,
    n_nodes,
    n_qp,
    vector_size,
    local_prefix,
    array_inputs,
    quadrature_rule,
    use_shared_weak_local=False,
    isoparametric_geometry=False,
):
    function_name = (
        _sfem_soa_isoparametric_public_function_name(prefix, form.name, quadrature_rule)
        if isoparametric_geometry
        else _sfem_soa_public_function_name(prefix, form.name, quadrature_rule)
    )
    implementation_name = "%s_impl" % function_name
    block_name = "%s_%s_block" % (local_prefix, form.name)
    element_inputs = _sfem_soa_element_inputs(array_inputs)
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    use_tensor_product_reference = (
        quadrature_rule is not None
        and quadrature_rule.is_tensor_product
        and len(reference_inputs) == 1
        and reference_inputs[0].name == "grad_ref"
    )
    use_reference_gradient_vectors = (
        form.weak_form is not None
        and not use_tensor_product_reference
        and len(reference_inputs) == 1
        and reference_inputs[0].name == "grad_ref"
    )
    use_stream_arrays = use_shared_weak_local and form.weak_form is not None
    base_params = ["const ptrdiff_t nelements"]
    if isoparametric_geometry:
        base_params.extend(
            "const real_t *const SFEM_RESTRICT %s" % stream
            for stream in _coordinate_stream_names(dim, n_nodes)
        )
    else:
        base_params.extend(
            "const %s *const SFEM_RESTRICT %s" % (array_input.scalar_type, stream)
            for array_input in element_inputs
            for stream in _soa_array_stream_names(array_input)
        )
    if use_tensor_product_reference:
        reference_params = (
            "const scalar_t *const SFEM_RESTRICT shape_1d",
            "const scalar_t *const SFEM_RESTRICT grad_1d",
        )
        quadrature_weight_param = "const scalar_t *const SFEM_RESTRICT q_weight_1d"
    elif use_reference_gradient_vectors:
        reference_params = _sfem_reference_gradient_vector_params(dim)
        quadrature_weight_param = "const scalar_t *const SFEM_RESTRICT q_weight"
    else:
        reference_params = tuple(
            "const %s *const SFEM_RESTRICT %s" % (array_input.scalar_type, array_input.name)
            for array_input in reference_inputs
        )
        quadrature_weight_param = "const scalar_t *const SFEM_RESTRICT q_weight"
    material_params = ["const scalar_t mu", "const scalar_t lmbda"]
    field_params = []
    field_params.extend(
        "const real_t *const SFEM_RESTRICT %s" % name
        for name in _field_stream_names("u", dim, n_nodes)
    )
    if form.has_direction:
        field_params.extend(
            "const real_t *const SFEM_RESTRICT %s" % name
            for name in _field_stream_names("h", dim, n_nodes)
        )
    output_params = tuple(
        "real_t *const SFEM_RESTRICT %s" % name
        for name in _output_stream_names(form, dim, n_nodes)
    )
    if isoparametric_geometry and quadrature_rule is None:
        raise ValueError("isoparametric SoA wrappers require an element quadrature rule")
    if quadrature_rule is None:
        impl_params = tuple(base_params) + reference_params + (
            "const scalar_t qw",
        ) + tuple(material_params) + tuple(field_params) + output_params
        wrapper_params = impl_params
    else:
        impl_params = tuple(base_params) + reference_params + (
            quadrature_weight_param,
        ) + tuple(material_params) + tuple(field_params) + output_params
        wrapper_params = tuple(base_params) + tuple(material_params) + tuple(field_params) + output_params
    wrapper_params = _sfem_soa_public_wrapper_params(wrapper_params)

    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>",
        "static SFEM_INLINE int %s(" % implementation_name,
    ]
    for idx, param in enumerate(impl_params):
        comma = "," if idx + 1 < len(impl_params) else ""
        lines.append("        %s%s" % (param, comma))
    lines.extend(
        [
            ") {",
            "    static_assert(N_QP == %d, \"N_QP does not match generated geometry streams\");" % n_qp,
            "    static_assert(N_SHAPE == %d, \"N_SHAPE does not match generated expression\");" % n_nodes,
            "    static_assert(VECTOR_SIZE > 0, \"VECTOR_SIZE must be positive\");",
        ]
    )
    if use_tensor_product_reference:
        lines.extend(
            [
                "    static constexpr int N_QP_1D = %d;"
                % quadrature_rule.tensor_product_n_qp_1d,
                "    static constexpr int N_SHAPE_1D = %d;"
                % quadrature_rule.tensor_product_n_shape_1d,
            ]
        )
    lines.extend(
        [
            "",
            "#pragma omp parallel for schedule(static)",
            "    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {",
            "        const ptrdiff_t nelems = MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);",
        ]
    )

    block_streams = []
    if isoparametric_geometry:
        for stream in _coordinate_stream_names(dim, n_nodes):
            block_streams.append(stream)
            lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
    for array_input in element_inputs:
        for stream in _soa_array_stream_names(array_input):
            block_streams.append(stream)
            lines.append("        %s block_%s[VECTOR_SIZE];" % (array_input.scalar_type, stream))
    for stream in _field_stream_names("u", dim, n_nodes):
        block_streams.append(stream)
        lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
    if form.has_direction:
        for stream in _field_stream_names("h", dim, n_nodes):
            block_streams.append(stream)
            lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)
    for stream in _output_stream_names(form, dim, n_nodes):
        block_streams.append(stream)
        lines.append("        scalar_t block_%s[VECTOR_SIZE];" % stream)

    lines.extend(["", "#pragma omp simd", "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {"])
    if isoparametric_geometry:
        for stream in _coordinate_stream_names(dim, n_nodes):
            lines.append("            block_%s[lane] = %s[evbegin + lane];" % (stream, stream))
    for stream in _field_stream_names("u", dim, n_nodes):
        lines.append("            block_%s[lane] = %s[evbegin + lane];" % (stream, stream))
    if form.has_direction:
        for stream in _field_stream_names("h", dim, n_nodes):
            lines.append("            block_%s[lane] = %s[evbegin + lane];" % (stream, stream))
    for stream in _output_stream_names(form, dim, n_nodes):
        if form.output_mode == "accumulate":
            lines.append("            block_%s[lane] = %s[evbegin + lane];" % (stream, stream))
        else:
            lines.append("            block_%s[lane] = 0;" % stream)
    lines.append("        }")

    if use_stream_arrays:
        lines.append("")
        lines.append(
            "        const scalar_t *const block_u_streams[N_SHAPE * %d] = {%s};"
            % (
                dim,
                ", ".join(
                    "block_%s" % stream
                    for stream in _field_stream_names("u", dim, n_nodes)
                ),
            )
        )
        if form.has_direction:
            lines.append(
                "        const scalar_t *const block_h_streams[N_SHAPE * %d] = {%s};"
                % (
                    dim,
                    ", ".join(
                        "block_%s" % stream
                        for stream in _field_stream_names("h", dim, n_nodes)
                    ),
                )
            )
        if form.name == "objective":
            pass
        else:
            lines.append(
                "        scalar_t *const block_out_streams[N_SHAPE * %d] = {%s};"
                % (
                    dim,
                    ", ".join(
                        "block_%s" % stream
                        for stream in _output_stream_names(form, dim, n_nodes)
                    ),
            )
        )

    if isoparametric_geometry:
        lines.append("")
        lines.append(
            "        const scalar_t *const block_coordinate_streams[N_SHAPE * %d] = {%s};"
            % (
                dim,
                ", ".join(
                    "block_%s" % stream
                    for stream in _coordinate_stream_names(dim, n_nodes)
                ),
            )
        )

    lines.extend(["", "        for (int q = 0; q < N_QP; ++q) {"])
    if use_tensor_product_reference:
        lines.extend(_tensor_product_q_index_lines(dim, "            "))
        lines.append(
            "            const scalar_t tensor_q_weight = %s;"
            % _tensor_product_quadrature_weight_expr(dim)
        )
    if isoparametric_geometry:
        lines.extend(
            _sfem_soa_isoparametric_geometry_lines(
                dim,
                n_nodes,
                quadrature_rule,
                use_tensor_product_reference,
                use_reference_gradient_vectors,
                reference_inputs,
            )
        )
    elif element_inputs:
        lines.extend(["#pragma omp simd", "            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {"])
        for array_input in element_inputs:
            for stream in _soa_array_stream_names(array_input):
                lines.append(
                    "                block_%s[lane] = %s[(ptrdiff_t)q * nelements + evbegin + lane];"
                    % (stream, stream)
                )
        lines.append("            }")

    call_args = ["nelems", "q"]
    call_args.extend(
        "block_%s" % stream
        for array_input in element_inputs
        for stream in _soa_array_stream_names(array_input)
    )
    if use_tensor_product_reference:
        call_args.extend(("shape_1d", "grad_1d"))
    elif use_reference_gradient_vectors:
        call_args.extend(
            _sfem_reference_gradient_vector_name(component)
            for component in range(dim)
        )
    else:
        call_args.extend(array_input.name for array_input in reference_inputs)
    if quadrature_rule is None:
        call_args.extend(("qw", "mu", "lmbda"))
    elif use_tensor_product_reference:
        call_args.extend(("tensor_q_weight", "mu", "lmbda"))
    else:
        call_args.extend(("q_weight[q]", "mu", "lmbda"))
    if use_stream_arrays:
        call_args.append("block_u_streams")
        if form.has_direction:
            call_args.append("block_h_streams")
        if form.name == "objective":
            call_args.append("block_value")
        else:
            call_args.append("block_out_streams")
    else:
        call_args.extend("block_%s" % stream for stream in _field_stream_names("u", dim, n_nodes))
        if form.has_direction:
            call_args.extend("block_%s" % stream for stream in _field_stream_names("h", dim, n_nodes))
        call_args.extend("block_%s" % stream for stream in _output_stream_names(form, dim, n_nodes))
    lines.extend(
        [
            "",
            "            %s<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(%s);"
            % (block_name, ", ".join(call_args)),
            "        }",
            "",
        ]
    )

    lines.extend(["#pragma omp simd", "        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {"])
    for stream in _output_stream_names(form, dim, n_nodes):
        lines.append("            %s[evbegin + lane] = block_%s[lane];" % (stream, stream))
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

    lines.append('extern "C" int %s(' % function_name)
    for idx, param in enumerate(wrapper_params):
        comma = "," if idx + 1 < len(wrapper_params) else ""
        lines.append("        %s%s" % (param, comma))
    if quadrature_rule is None:
        wrapper_args = tuple(_cpp_argument_name(param) for param in wrapper_params)
    else:
        wrapper_args = _sfem_soa_specialized_wrapper_arguments(
            prefix,
            quadrature_rule,
            wrapper_params,
            reference_inputs,
            use_reference_gradient_vectors,
        )
    lines.extend(
        [
            ") {",
            "    return sfem::codegen::%s<real_t, %d, %d, %d>(%s);"
            % (
                implementation_name,
                n_qp,
                n_nodes,
                vector_size,
                ", ".join(wrapper_args),
            ),
            "}",
        ]
    )
    return lines


def _field_stream_names(prefix, dim, n_nodes):
    return tuple(
        "%s%s%d" % (prefix, _component_name(d), node)
        for node in range(n_nodes)
        for d in range(dim)
    )


def _coordinate_stream_names(dim, n_nodes):
    return tuple(
        "%s%d" % (_component_name(d), node)
        for node in range(n_nodes)
        for d in range(dim)
    )


def _soa_array_stream_names(array_input):
    return tuple("%s%d" % (array_input.name, i) for i in range(array_input.size))


def _sfem_soa_element_inputs(array_inputs):
    return tuple(array_input for array_input in array_inputs if not array_input.is_reference_qp_shape)


def _sfem_soa_reference_inputs(array_inputs):
    return tuple(array_input for array_input in array_inputs if array_input.is_reference_qp_shape)


def _sfem_soa_reference_param_name(array_input):
    return "%s_data" % array_input.name


def _sfem_soa_has_adjugate_geometry_inputs(array_inputs, dim):
    element_inputs = _sfem_soa_element_inputs(array_inputs)
    names_and_sizes = {(array_input.name, array_input.size) for array_input in element_inputs}
    return (
        ("jacobian_adjugate", dim * dim) in names_and_sizes
        and ("jacobian_determinant", 1) in names_and_sizes
    )


def _sfem_soa_diagnostics_header():
    struct_name = _sfem_soa_diagnostics_struct_name()
    return [
        "#ifndef SFEM_CODEGEN_KERNEL_DIAGNOSTICS_HPP",
        "#define SFEM_CODEGEN_KERNEL_DIAGNOSTICS_HPP",
        "",
        "#include <stddef.h>",
        "",
        "#ifndef SFEM_INLINE",
        "#define SFEM_INLINE inline",
        "#endif",
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
        "struct %s {" % struct_name,
        "    const char *kernel_name;",
        "    const char *element_type;",
        "    int dim;",
        "    int n_qp;",
        "    int n_shape;",
        "    int vector_size;",
        "    int quadrature_order;",
        "    long add_instructions_per_qp_lane;",
        "    long mul_instructions_per_qp_lane;",
        "    long div_instructions_per_qp_lane;",
        "    long sqrt_instructions_per_qp_lane;",
        "    long pow_instructions_per_qp_lane;",
        "    long log_instructions_per_qp_lane;",
        "    long trig_instructions_per_qp_lane;",
        "    long load_instructions_per_qp_lane;",
        "    long store_instructions_per_qp_lane;",
        "    long flops_per_qp_lane;",
        "    long temporaries;",
        "    long estimated_registers;",
        "    int geometry_streams;",
        "    int reference_scalars;",
        "    int quadrature_weight_scalars;",
        "    int material_scalars;",
        "    int u_streams;",
        "    int h_streams;",
        "    int output_streams;",
        "    int output_reads_per_element;",
        "    int output_writes_per_element;",
        "    double add_cpi;",
        "    double mul_cpi;",
        "    double div_cpi;",
        "    double sqrt_cpi;",
        "    double pow_cpi;",
        "    double log_cpi;",
        "    double trig_cpi;",
        "    double load_cpi;",
        "    double store_cpi;",
        "};",
        "",
        "static SFEM_INLINE double %s_total_flops(" % struct_name,
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements) {",
        "    const double n = nelements > 0 ? (double)nelements : 0.0;",
        "    return n * (double)d->n_qp * (double)d->flops_per_qp_lane;",
        "}",
        "",
        "static SFEM_INLINE size_t %s_total_bytes(" % struct_name,
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    (void)accumulator_bytes;",
        "    const size_t n = nelements > 0 ? (size_t)nelements : (size_t)0;",
        "    const size_t geometry_bytes = n * (size_t)d->n_qp * (size_t)d->geometry_streams * scalar_bytes;",
        "    const size_t field_bytes = n * (size_t)(d->u_streams + d->h_streams) * real_bytes;",
        "    const size_t output_bytes = n * (size_t)d->output_streams * (size_t)(d->output_reads_per_element + d->output_writes_per_element) * real_bytes;",
        "    const size_t reference_bytes = ((size_t)d->reference_scalars + (size_t)d->quadrature_weight_scalars + (size_t)d->material_scalars) * scalar_bytes;",
        "    return geometry_bytes + field_bytes + output_bytes + reference_bytes;",
        "}",
        "",
        "static SFEM_INLINE double %s_arithmetic_intensity(" % struct_name,
        "        const %s *const d," % struct_name,
        "        const ptrdiff_t nelements,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    const size_t bytes = %s_total_bytes(d, nelements, scalar_bytes, real_bytes, accumulator_bytes);" % struct_name,
        "    return bytes ? %s_total_flops(d, nelements) / (double)bytes : 0.0;" % struct_name,
        "}",
        "",
        "} // namespace codegen",
        "} // namespace sfem",
        "",
        "#endif",
    ]


def _sfem_soa_diagnostics_lines(
    form,
    prefix,
    dim,
    n_nodes,
    n_qp,
    vector_size,
    array_inputs,
    quadrature_rule,
):
    public_name = _sfem_soa_public_function_name(prefix, form.name, quadrature_rule)
    struct_name = _sfem_soa_diagnostics_struct_name()
    variable_name = "%s_diagnostics_data" % public_name
    if form.expression_graph is not None:
        cost = form.expression_graph.cost
    elif form.weak_form is not None:
        diagnostic_graph = (
            KernelExpressions()
            .add(ExpressionRole.OPERATOR_EVALUATION, form.weak_form.diagnostic_expressions(form.has_direction))
            .build_graph(
                data_symbols=form.weak_form.deformation_gradient,
                temporary_prefix="weak_diag_tmp",
            )
        )
        cost = diagnostic_graph.cost
    else:
        cost = ExpressionCost()
    element_inputs = _sfem_soa_element_inputs(array_inputs)
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    geometry_streams = sum(array_input.size for array_input in element_inputs)
    if quadrature_rule is not None and quadrature_rule.is_tensor_product:
        reference_scalars = (
            len(quadrature_rule.tensor_product_shape_values_1d)
            + len(quadrature_rule.tensor_product_shape_gradients_1d)
        )
        quadrature_weight_scalars = len(quadrature_rule.tensor_product_weights_1d)
    else:
        reference_scalars = sum(array_input.size for array_input in reference_inputs)
        quadrature_weight_scalars = n_qp if quadrature_rule is not None else 1
    output_streams = len(_output_stream_names(form, dim, n_nodes))
    output_reads = output_streams if form.output_mode == "accumulate" else 0
    output_writes = output_streams
    u_streams = dim * n_nodes
    h_streams = dim * n_nodes if form.has_direction else 0
    element_type = quadrature_rule.element_type if quadrature_rule is not None else "GENERIC"
    quadrature_order = quadrature_rule.order if quadrature_rule is not None else 0
    lines = [
        "namespace sfem {",
        "namespace codegen {",
        "",
        "static const %s %s = {" % (struct_name, variable_name),
        '    "%s",' % public_name,
        '    "%s",' % element_type,
        "    %d," % dim,
        "    %d," % n_qp,
        "    %d," % n_nodes,
        "    %d," % vector_size,
        "    %d," % quadrature_order,
        "    %d," % cost.adds,
        "    %d," % cost.muls,
        "    %d," % cost.divs,
        "    %d," % cost.sqrts,
        "    %d," % cost.pows,
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
        "    2,",
        "    %d," % u_streams,
        "    %d," % h_streams,
        "    %d," % output_streams,
        "    %d," % output_reads,
        "    %d," % output_writes,
        "    1.0,",
        "    1.0,",
        "    8.0,",
        "    12.0,",
        "    16.0,",
        "    20.0,",
        "    24.0,",
        "    1.0,",
        "    1.0",
        "};",
        "",
        "} // namespace codegen",
        "} // namespace sfem",
        "",
        'extern "C" const sfem::codegen::%s *%s_diagnostics(void) {' % (struct_name, public_name),
        "    return &sfem::codegen::%s;" % variable_name,
        "}",
        "",
        'extern "C" double %s_arithmetic_intensity(' % public_name,
        "        const ptrdiff_t nelements,",
        "        const size_t scalar_bytes,",
        "        const size_t real_bytes,",
        "        const size_t accumulator_bytes) {",
        "    return sfem::codegen::%s_arithmetic_intensity(&sfem::codegen::%s, nelements, scalar_bytes, real_bytes, accumulator_bytes);" % (struct_name, variable_name),
        "}",
    ]
    return lines


def _sfem_soa_diagnostics_struct_name():
    return "KernelDiagnostics"


def _validate_sfem_soa_quadrature_rule(quadrature_rule, dim, n_nodes, n_qp, array_inputs):
    if quadrature_rule.dim != dim:
        raise ValueError(
            "quadrature rule dimension %d does not match dim=%d"
            % (quadrature_rule.dim, dim)
        )
    if quadrature_rule.n_shape != n_nodes:
        raise ValueError(
            "quadrature rule n_shape %d does not match n_nodes=%d"
            % (quadrature_rule.n_shape, n_nodes)
        )
    if quadrature_rule.n_qp != n_qp:
        raise ValueError(
            "quadrature rule n_qp %d does not match n_qp=%d"
            % (quadrature_rule.n_qp, n_qp)
        )
    reference_inputs = _sfem_soa_reference_inputs(array_inputs)
    if len(reference_inputs) != 1 or reference_inputs[0].name != "grad_ref":
        raise ValueError("element-specialized wrappers currently require one grad_ref reference input")
    grad_ref = reference_inputs[0]
    if grad_ref.components != dim or grad_ref.n_shape != n_nodes or grad_ref.n_qp != n_qp:
        raise ValueError("grad_ref reference input does not match quadrature rule")


def _sfem_soa_quadrature_rule_lines(prefix, quadrature_rule):
    if quadrature_rule.is_tensor_product:
        shape_name = _sfem_soa_quadrature_array_name(prefix, quadrature_rule, "shape_1d")
        grad_name = _sfem_soa_quadrature_array_name(prefix, quadrature_rule, "grad_1d")
        weight_name = _sfem_soa_quadrature_array_name(prefix, quadrature_rule, "q_weight_1d")
        return [
            "static const real_t %s[%d] = {%s};"
            % (
                shape_name,
                len(quadrature_rule.tensor_product_shape_values_1d),
                _cpp_scalar_initializer_list(quadrature_rule.tensor_product_shape_values_1d),
            ),
            "static const real_t %s[%d] = {%s};"
            % (
                grad_name,
                len(quadrature_rule.tensor_product_shape_gradients_1d),
                _cpp_scalar_initializer_list(quadrature_rule.tensor_product_shape_gradients_1d),
            ),
            "static const real_t %s[%d] = {%s};"
            % (
                weight_name,
                len(quadrature_rule.tensor_product_weights_1d),
                _cpp_scalar_initializer_list(quadrature_rule.tensor_product_weights_1d),
            ),
        ]
    grad_name = _sfem_soa_quadrature_array_name(prefix, quadrature_rule, "grad_ref")
    weight_name = _sfem_soa_quadrature_array_name(prefix, quadrature_rule, "q_weight")
    lines = [
        "static const real_t %s[%d] = {%s};"
        % (
            grad_name,
            len(quadrature_rule.reference_gradients),
            _cpp_scalar_initializer_list(quadrature_rule.reference_gradients),
        ),
        "static const real_t %s[%d] = {%s};"
        % (
            weight_name,
            len(quadrature_rule.weights),
            _cpp_scalar_initializer_list(quadrature_rule.weights),
        ),
    ]
    for component in range(quadrature_rule.dim):
        vector_name = _sfem_soa_quadrature_array_name(
            prefix,
            quadrature_rule,
            _sfem_reference_gradient_vector_name(component),
        )
        vector_values = _sfem_reference_gradient_component_values(quadrature_rule, component)
        lines.append(
            "static const real_t %s[%d] = {%s};"
            % (
                vector_name,
                len(vector_values),
                _cpp_scalar_initializer_list(vector_values),
            )
        )
    return lines


def _sfem_reference_gradient_component_values(quadrature_rule, component):
    values = []
    for q in range(quadrature_rule.n_qp):
        for shape in range(quadrature_rule.n_shape):
            values.append(
                quadrature_rule.reference_gradients[
                    (q * quadrature_rule.n_shape + shape) * quadrature_rule.dim
                    + component
                ]
            )
    return tuple(values)


def _sfem_soa_specialized_wrapper_arguments(
    prefix,
    quadrature_rule,
    wrapper_params,
    reference_inputs,
    use_reference_gradient_vectors=False,
):
    arguments = [_cpp_argument_name(param) for param in wrapper_params]
    if quadrature_rule.is_tensor_product:
        offset = 1 + _sfem_soa_element_stream_count_from_params(wrapper_params)
        arguments.insert(
            offset,
            _sfem_codegen_qualified_name(
                _sfem_soa_quadrature_array_name(prefix, quadrature_rule, "shape_1d")
            ),
        )
        arguments.insert(
            offset + 1,
            _sfem_codegen_qualified_name(
                _sfem_soa_quadrature_array_name(prefix, quadrature_rule, "grad_1d")
            ),
        )
        arguments.insert(
            offset + 2,
            _sfem_codegen_qualified_name(
                _sfem_soa_quadrature_array_name(prefix, quadrature_rule, "q_weight_1d")
            ),
        )
        return tuple(arguments)
    if use_reference_gradient_vectors:
        offset = 1 + _sfem_soa_element_stream_count_from_params(wrapper_params)
        for component in range(quadrature_rule.dim):
            arguments.insert(
                offset + component,
                _sfem_codegen_qualified_name(
                    _sfem_soa_quadrature_array_name(
                        prefix,
                        quadrature_rule,
                        _sfem_reference_gradient_vector_name(component),
                    )
                ),
            )
        arguments.insert(
            offset + quadrature_rule.dim,
            _sfem_codegen_qualified_name(
                _sfem_soa_quadrature_array_name(prefix, quadrature_rule, "q_weight")
            ),
        )
        return tuple(arguments)
    for array_input in reference_inputs:
        arguments.insert(
            1 + _sfem_soa_element_stream_count_from_params(wrapper_params),
            _sfem_codegen_qualified_name(
                _sfem_soa_quadrature_array_name(prefix, quadrature_rule, array_input.name)
            ),
        )
    arguments.insert(
        1 + _sfem_soa_element_stream_count_from_params(wrapper_params) + len(reference_inputs),
        _sfem_codegen_qualified_name(
            _sfem_soa_quadrature_array_name(prefix, quadrature_rule, "q_weight")
        ),
    )
    return tuple(arguments)


def _sfem_codegen_qualified_name(name):
    return "sfem::codegen::%s" % name


def _sfem_soa_element_stream_count_from_params(params):
    count = 0
    for param in params[1:]:
        name = _cpp_argument_name(param)
        if name in ("mu", "lmbda"):
            break
        count += 1
    return count


def _sfem_soa_public_wrapper_params(params):
    return tuple(param.replace("scalar_t", "real_t") for param in params)


def _sfem_soa_public_function_name(prefix, form_name, quadrature_rule):
    if quadrature_rule is None:
        return "%s_%s_soa" % (prefix, form_name)
    return "%s_%s_%s_soa" % (
        prefix,
        quadrature_rule.element_type.lower(),
        form_name,
    )


def _sfem_soa_isoparametric_public_function_name(prefix, form_name, quadrature_rule):
    if quadrature_rule is None:
        return "%s_%s_isoparametric_soa" % (prefix, form_name)
    return "%s_%s_%s_isoparametric_soa" % (
        prefix,
        quadrature_rule.element_type.lower(),
        form_name,
    )


def _sfem_soa_quadrature_array_name(prefix, quadrature_rule, name):
    return "%s_%s_%s" % (
        prefix,
        quadrature_rule.element_type.lower(),
        name,
    )


def _cpp_scalar_initializer_list(values):
    return ", ".join(_cpp_scalar_literal(value) for value in values)


def _cpp_scalar_literal(value):
    value = float(value)
    if value == 0.0:
        return "0.0"
    return "%.17g" % value


def _output_stream_names(form, dim, n_nodes):
    if form.weak_form is not None:
        if form.name == "objective":
            return ("value",)
        return tuple(
            "out%s%d" % (_component_name(d), node)
            for node in range(n_nodes)
            for d in range(dim)
        )
    output_count = len(form.expression_graph.evaluation_plan.outputs)
    if output_count == 1:
        return ("value",)
    return tuple(
        "out%s%d" % (_component_name(d), node)
        for node in range(n_nodes)
        for d in range(dim)
    )


def _component_name(component):
    return ("x", "y", "z")[component]


def _append_statement_lines(lines, statements, scalar_type, output_name, indent):
    for statement in statements:
        expression = _sfem_ccode(statement.expression)
        if statement.kind == "intermediate":
            target = _cpp_symbol(statement.target, output_name)
            lines.append("%sconst %s %s = %s;" % (indent, scalar_type, target, expression))
        elif statement.augmented:
            target = _cpp_lvalue(statement.target, output_name)
            lines.append("%s%s += %s;" % (indent, target, expression))
        else:
            target = _cpp_lvalue(statement.target, output_name)
            lines.append("%s%s = %s;" % (indent, target, expression))


def _normalize_kernel_expression(expr):
    if isinstance(expr, KernelExpression):
        return expr
    if isinstance(expr, tuple) and len(expr) in (2, 3):
        role = ExpressionRole(expr[0])
        name = expr[2] if len(expr) == 3 else None
        return KernelExpression(role, expr[1], name)
    raise TypeError("Expected KernelExpression or (role, expression[, name]) tuple")


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


def _cpp_argument_name(argument):
    return argument.replace("*", " ").split()[-1]


def _direct_output_targets(output_targets):
    direct_output_targets = tuple(
        target
        for target in output_targets
        if not (isinstance(target, str) and target.startswith("output:"))
    )
    return direct_output_targets, len(direct_output_targets) != len(output_targets)


def _group_kernel_symbols(symbols):
    arrays = {}
    scalars = []
    for symbol in symbols:
        base, index = _indexed_symbol(symbol)
        if base is None:
            scalars.append(symbol)
        else:
            arrays.setdefault(base, set()).add(index)
    return arrays, tuple(scalars)


def _indexed_symbol(symbol):
    text = str(symbol)
    if not text.endswith("]"):
        return None, None
    bracket = text.rfind("[")
    if bracket <= 0:
        return None, None
    index = text[bracket + 1 : -1]
    if not index.isdigit():
        return None, None
    return text[:bracket], int(index)


def _cpp_symbol(symbol, output_name):
    if isinstance(symbol, str) and symbol.startswith("output:"):
        index = symbol.rsplit(":", 1)[-1]
        return "%s[%s]" % (output_name, index)
    return str(symbol)


def _cpp_lvalue(symbol, output_name):
    base, _ = _indexed_symbol(symbol)
    if isinstance(symbol, str) and symbol.startswith("output:"):
        return _cpp_symbol(symbol, output_name)
    if base is not None:
        return _cpp_symbol(symbol, output_name)
    return "*%s" % _cpp_symbol(symbol, output_name)


def _cpp_wrapper_name(function_name):
    words = []
    for word in str(function_name).replace("-", "_").split("_"):
        if word:
            words.append(word[0].upper() + word[1:])
    return "%sOperator" % "".join(words)


def _cpp_macro_name(name):
    chars = []
    for char in str(name):
        if char.isalnum():
            chars.append(char.upper())
        else:
            chars.append("_")
    return "".join(chars)


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


def _normalize_scopes(loop_symbols, scopes):
    normalized = []

    for raw_scope in scopes or ():
        if isinstance(raw_scope, ExecutionScope):
            normalized.append(raw_scope)
        else:
            raise TypeError("scopes must contain ExecutionScope instances")

    for raw_kind, symbols in (loop_symbols or {}).items():
        normalized.append(ExecutionScope(_scope_kind(raw_kind), symbols))

    return tuple(normalized)


def _scope_kind(value):
    if isinstance(value, ScopeKind):
        return value
    normalized = str(value).replace("-", "_")
    aliases = {
        "mesh_wide": ScopeKind.MESH,
        "meshwide": ScopeKind.MESH,
        "mesh": ScopeKind.MESH,
    }
    if normalized in aliases:
        return aliases[normalized]
    return ScopeKind(normalized)


def _scope_symbol_map(scopes):
    ret = {}
    for scope in scopes:
        for symbol in scope.symbols:
            ret[symbol] = scope
    return ret


def _layout_symbol_map(symbolic_objects):
    ret = {}
    for symbolic_object in symbolic_objects:
        for symbol in symbolic_object.direct_symbols:
            ret[symbol] = symbolic_object
    return ret


def _annotate_data_layout(graph, symbol, layout_symbol_map):
    if symbol not in graph or symbol not in layout_symbol_map:
        return

    symbolic_object = layout_symbol_map[symbol]
    component = symbolic_object.component_index(symbol)
    item_index = sp.symbols("%s_idx" % symbolic_object.name, integer=True)
    graph.nodes[symbol]["layout"] = symbolic_object.layout
    graph.nodes[symbol]["layout_kind"] = symbolic_object.layout.kind
    graph.nodes[symbol]["symbolic_object"] = symbolic_object.name
    graph.nodes[symbol]["component"] = component
    graph.nodes[symbol]["layout_index"] = item_index
    graph.nodes[symbol]["layout_offset"] = symbolic_object.layout_offset(symbol, item_index)
    graph.nodes[symbol]["object_metadata"] = symbolic_object.metadata
    if "n_nodes" in symbolic_object.metadata and "dim" in symbolic_object.metadata:
        dim = symbolic_object.metadata["dim"]
        graph.nodes[symbol]["node"] = component // dim
        graph.nodes[symbol]["dim_component"] = component % dim


def _expression_scopes(expression, scope_symbol_map):
    kinds = {
        scope_symbol_map[symbol].kind
        for symbol in expression.free_symbols
        if symbol in scope_symbol_map
    }
    return _ordered_scopes(kinds)


def _scope_sort_key(kind):
    return tuple(ScopeKind).index(kind)


def _ordered_scopes(scopes):
    return tuple(sorted(scopes, key=_scope_sort_key))


def _hoist_scope(scopes):
    if not scopes:
        return ScopeKind.MESH
    return scopes[-1]


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


def _output_node_name(kernel_expr, idx):
    if kernel_expr.name is not None:
        return "output:%s:%s" % (kernel_expr.role.value, kernel_expr.name)
    lhs = _lhs(kernel_expr.expression)
    if lhs is not None:
        return "output:%s:%s" % (kernel_expr.role.value, lhs)
    return "output:%s:%d" % (kernel_expr.role.value, idx)


def _add_expression_node(
    graph,
    symbol,
    expression,
    data_symbols,
    scope_symbol_map,
    layout_symbol_map,
    kind,
):
    graph.add_node(symbol, kind=kind, expression=expression)
    for dep in _dependencies(expression):
        _ensure_dependency_node(graph, dep, data_symbols, scope_symbol_map, layout_symbol_map)
        graph.add_edge(dep, symbol)


def _add_node(graph, symbol, kind, **attrs):
    if symbol not in graph:
        graph.add_node(symbol, kind=kind, **attrs)
    else:
        graph.nodes[symbol].update(attrs)
        graph.nodes[symbol]["kind"] = kind


def _ensure_dependency_node(graph, symbol, data_symbols, scope_symbol_map, layout_symbol_map):
    if symbol in graph:
        _annotate_data_layout(graph, symbol, layout_symbol_map)
        return

    if symbol in scope_symbol_map:
        scope = scope_symbol_map[symbol]
        graph.add_node(
            symbol,
            kind="loop_index",
            scope=scope.kind.value,
            scope_name=scope.name,
            scope_kind=scope.kind,
        )
        return

    if symbol in data_symbols:
        graph.add_node(symbol, kind="data")
        _annotate_data_layout(graph, symbol, layout_symbol_map)
    else:
        graph.add_node(symbol, kind="symbol")
        _annotate_data_layout(graph, symbol, layout_symbol_map)


def _dependencies(expression):
    return tuple(sorted(expression.free_symbols, key=str))


def _build_evaluation_plan(
    intermediates,
    reduced_outputs,
    output_nodes,
    kernel_outputs,
    data_symbols,
    scope_symbol_map,
):
    intermediate_statements = []
    output_statements = []

    for target, expr in intermediates:
        intermediate_statements.append(
            EvaluationStatement(
                target=target,
                expression=expr,
                kind="intermediate",
                dependencies=_dependencies(expr),
                cost=_statement_cost(expr, data_symbols, stores=1),
                scopes=_expression_scopes(expr, scope_symbol_map),
            )
        )

    for idx, (output, output_node, kernel_output) in enumerate(
        zip(reduced_outputs, output_nodes, kernel_outputs)
    ):
        expr = _rhs(output)
        lhs = _lhs(output)
        output_statements.append(
            EvaluationStatement(
                target=lhs if lhs is not None else output_node,
                expression=expr,
                kind="output",
                dependencies=_dependencies(expr),
                cost=_statement_cost(expr, data_symbols, stores=1),
                role=kernel_output.role,
                output_index=idx,
                augmented=isinstance(output, ast.AddAugmentedAssignment),
                scopes=_expression_scopes(expr, scope_symbol_map),
            )
        )

    statements = _resolve_statement_scopes(tuple(intermediate_statements + output_statements))
    intermediate_count = len(intermediate_statements)
    metrics = _evaluation_metrics(
        statements,
        tuple(stmt.target for stmt in statements[:intermediate_count]),
    )
    return EvaluationPlan(
        statements,
        tuple(statements[:intermediate_count]),
        tuple(statements[intermediate_count:]),
        metrics,
    )


def _annotate_graph_scope_placements(graph, evaluation_plan, output_nodes):
    for statement in evaluation_plan.intermediates:
        if statement.target in graph:
            graph.nodes[statement.target]["scopes"] = statement.scopes
            graph.nodes[statement.target]["hoist_scope"] = statement.hoist_scope

    for statement, output_node in zip(evaluation_plan.outputs, output_nodes):
        if output_node in graph:
            graph.nodes[output_node]["scopes"] = statement.scopes
            graph.nodes[output_node]["hoist_scope"] = statement.hoist_scope


def _resolve_statement_scopes(statements):
    scope_by_target = {}
    resolved = []

    for statement in statements:
        scopes = set(statement.scopes)
        for dependency in statement.dependencies:
            scopes.update(scope_by_target.get(dependency, ()))

        ordered_scopes = _ordered_scopes(scopes)
        resolved_statement = replace(
            statement,
            scopes=ordered_scopes,
            hoist_scope=_hoist_scope(ordered_scopes),
        )
        resolved.append(resolved_statement)
        scope_by_target[statement.target] = ordered_scopes

    return tuple(resolved)


def _evaluation_metrics(statements, temporary_symbols):
    temporary_symbol_set = set(temporary_symbols)
    last_use = {}

    for idx, statement in enumerate(statements):
        for dependency in statement.dependencies:
            last_use[dependency] = idx

    produced_temporaries = set()
    liveness = []
    peak_registers = 0
    peak_live_temporaries = 0
    total_flops = 0
    total_loads = 0
    total_stores = 0

    for idx, statement in enumerate(statements):
        dependencies = set(statement.dependencies)
        live_temporaries_before = {
            symbol
            for symbol in produced_temporaries
            if last_use.get(symbol, -1) >= idx
        }
        live_during = dependencies | live_temporaries_before

        if statement.target in temporary_symbol_set and last_use.get(statement.target, -1) > idx:
            live_during.add(statement.target)

        register_pressure = len(live_during)
        if register_pressure > peak_registers:
            peak_registers = register_pressure

        if statement.target in temporary_symbol_set:
            produced_temporaries.add(statement.target)

        live_temporaries_after = tuple(
            sorted(
                (
                    symbol
                    for symbol in produced_temporaries
                    if last_use.get(symbol, -1) > idx
                ),
                key=str,
            )
        )
        if len(live_temporaries_after) > peak_live_temporaries:
            peak_live_temporaries = len(live_temporaries_after)

        liveness.append(
            LivenessState(
                idx,
                statement.target,
                live_temporaries_after,
                register_pressure,
            )
        )

        total_flops += statement.cost.flops
        total_loads += statement.cost.loads
        total_stores += statement.cost.stores

    return EvaluationMetrics(
        total_flops,
        total_loads,
        total_stores,
        peak_registers,
        peak_live_temporaries,
        tuple(liveness),
    )


def _detect_patterns(graph, intermediates, reduced_outputs, output_nodes, symbolic_objects):
    patterns = []
    symbolic_objects = tuple(symbolic_objects)

    for symbol, expr in intermediates:
        _append_pattern(
            graph,
            patterns,
            ExpressionPattern(
                PatternKind.REPEATED_SUBEXPRESSION,
                symbol,
                expr,
                _dependencies(expr),
                "sympy_cse",
            ),
        )
        _extend_patterns(graph, patterns, _match_objects(symbol, expr, symbolic_objects))

    for output_node, output in zip(output_nodes, reduced_outputs):
        expr = _rhs(output)
        _extend_patterns(graph, patterns, _match_objects(output_node, expr, symbolic_objects))

    return tuple(patterns)


def _extend_patterns(graph, patterns, new_patterns):
    for pattern in new_patterns:
        _append_pattern(graph, patterns, pattern)


def _append_pattern(graph, patterns, pattern):
    patterns.append(pattern)
    if pattern.node in graph:
        node_attrs = graph.nodes[pattern.node]
        node_patterns = list(node_attrs.get("patterns", ()))
        node_patterns.append(pattern)
        node_attrs["patterns"] = tuple(node_patterns)


def _match_objects(node, expression, symbolic_objects):
    matches = []

    for symbolic_object in symbolic_objects:
        matched_symbols, matched_expressions = symbolic_object.match(expression)
        if matched_symbols or matched_expressions:
            matches.append(
                ExpressionPattern(
                    symbolic_object.kind,
                    node,
                    expression,
                    matched_symbols,
                    symbolic_object.name,
                    matched_expressions,
                    symbolic_object,
                )
            )

    return matches


def _expression_cost(intermediates, reduced_outputs, data_symbols, estimated_registers):
    adds = muls = divs = sqrts = pows = logs = trigs = stores = 0
    loaded = set()

    for _, expr in intermediates:
        a, m, d, s, p, log_count, trig_count = _op_counts(expr)
        adds += a
        muls += m
        divs += d
        sqrts += s
        pows += p
        logs += log_count
        trigs += trig_count
        loaded.update(expr.free_symbols)
        stores += 1

    for output in reduced_outputs:
        expr = _rhs(output)
        a, m, d, s, p, log_count, trig_count = _op_counts(expr)
        adds += a
        muls += m
        divs += d
        sqrts += s
        pows += p
        logs += log_count
        trigs += trig_count
        loaded.update(expr.free_symbols)
        stores += 1

    loads = len(loaded.intersection(data_symbols)) if data_symbols else len(loaded)
    temporaries = len(intermediates)
    return ExpressionCost(
        adds=adds,
        muls=muls,
        divs=divs,
        sqrts=sqrts,
        pows=pows,
        logs=logs,
        trigs=trigs,
        loads=loads,
        stores=stores,
        temporaries=temporaries,
        estimated_registers=estimated_registers,
    )


def _statement_cost(expression, data_symbols, stores):
    adds, muls, divs, sqrts, pows, logs, trigs = _op_counts(expression)
    loaded = expression.free_symbols
    loads = len(loaded.intersection(data_symbols)) if data_symbols else len(loaded)
    return ExpressionCost(
        adds=adds,
        muls=muls,
        divs=divs,
        sqrts=sqrts,
        pows=pows,
        logs=logs,
        trigs=trigs,
        loads=loads,
        stores=stores,
        temporaries=0,
        estimated_registers=loads,
    )


def _op_counts(expression):
    adds = muls = divs = sqrts = pows = logs = trigs = 0
    trig_functions = {
        sp.sin,
        sp.cos,
        sp.tan,
        sp.asin,
        sp.acos,
        sp.atan,
        sp.sinh,
        sp.cosh,
        sp.tanh,
        sp.asinh,
        sp.acosh,
        sp.atanh,
    }

    for node in sp.preorder_traversal(expression):
        if isinstance(node, sp.Add):
            adds += max(0, len(node.args) - 1)
        elif isinstance(node, sp.Mul):
            muls += max(0, len(node.args) - 1)
        elif isinstance(node, sp.Pow):
            if node.exp == -1:
                divs += 1
            elif isinstance(node.exp, sp.Number) and float(node.exp) == 0.5:
                sqrts += 1
            else:
                pows += 1
        elif getattr(node, "is_Function", False):
            if node.func == sp.log:
                logs += 1
            elif node.func in trig_functions:
                trigs += 1
            elif node.func == sp.sqrt:
                sqrts += 1

    return adds, muls, divs, sqrts, pows, logs, trigs
