from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Mapping, Optional, Tuple, Union

import sympy as sp
import sympy.codegen.ast as ast
from sympy.printing.c import C99CodePrinter

try:
    from codegen.framework.fem.reference import (
        SfemElementQuadratureRule,
        SfemSoAArrayInput,
        SfemSoAElementSpecialization,
        sfem_element_quadrature_rule,
        sfem_mesh_reference_data,
        sfem_soa_array_input,
        sfem_soa_element_specialization,
        sfem_soa_element_specializations,
        sfem_soa_reference_input,
        sfem_supported_element_types,
        sfem_tensor_product_hex_uses_cartesian_ordering,
    )
    from codegen.framework.plans.reference_data import validate_reference_data_plan
except ImportError:
    from fem import (
        SfemElementQuadratureRule,
        SfemSoAArrayInput,
        SfemSoAElementSpecialization,
        sfem_element_quadrature_rule,
        sfem_mesh_reference_data,
        sfem_soa_array_input,
        sfem_soa_element_specialization,
        sfem_soa_element_specializations,
        sfem_soa_reference_input,
        sfem_supported_element_types,
        sfem_tensor_product_hex_uses_cartesian_ordering,
    )
    def validate_reference_data_plan(*args, **kwargs):
        return None


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

try:
    from codegen.framework.fem.tensor_product_geometry import (
        isoparametric_adjugate_lines,
        isoparametric_adjugate_call_lines,
        isoparametric_adjugate_stream_array_lines,
        streams_in_shape_order,
        tensor_product_cartesian_shape_order,
        tensor_product_coordinate_gradient_lines,
        tensor_product_current_q_isoparametric_geometry_lines,
        tensor_product_gradient_isoparametric_geometry_lines,
        tensor_product_ordered_coordinate_streams,
    )
except ImportError:
    from tensor_product_geometry import (
        isoparametric_adjugate_lines,
        isoparametric_adjugate_call_lines,
        isoparametric_adjugate_stream_array_lines,
        streams_in_shape_order,
        tensor_product_cartesian_shape_order,
        tensor_product_coordinate_gradient_lines,
        tensor_product_current_q_isoparametric_geometry_lines,
        tensor_product_gradient_isoparametric_geometry_lines,
        tensor_product_ordered_coordinate_streams,
    )

try:
    from codegen.framework.emitters.quadrature_codegen import (
        quadrature_reference_accessor,
        quadrature_reference_struct_lines,
    )
except ImportError:
    from quadrature_codegen import (
        quadrature_reference_accessor,
        quadrature_reference_struct_lines,
    )

try:
    from codegen.framework.backends.targets import CUDATarget, OpenMPTarget
except ImportError:
    from targets import CUDATarget, OpenMPTarget

try:
    from codegen.framework.fem.tensor_product_kernels import sfem_tensor_product_kernels_header_source
except ImportError:
    from tensor_product_kernels import sfem_tensor_product_kernels_header_source

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
    def __init__(self, scalar_type="scalar_t", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scalar_type = scalar_type

    def _scalar_literal(self, value):
        return "%s(%s)" % (self._scalar_type, value)

    def _print_Integer(self, expr):
        return self._scalar_literal("%d" % int(expr))

    def _print_Float(self, expr):
        return self._scalar_literal(super()._print_Float(expr))

    def _print_Rational(self, expr):
        return "(%s / %s)" % (
            self._scalar_literal("%d" % int(expr.p)),
            self._scalar_literal("%d" % int(expr.q)),
        )

    def _print_Pow(self, expr):
        base, exponent = expr.as_base_exp()
        if exponent.is_Integer:
            exponent_value = int(exponent)
            if abs(exponent_value) <= _SFEM_SPECIALIZED_POW_MAX_EXPONENT:
                if exponent_value == 0:
                    return self._scalar_literal("1")
                if exponent_value == 1:
                    return self._print(base)
                suffix = "m%d" % abs(exponent_value) if exponent_value < 0 else "%d" % exponent_value
                return "pow_%s(%s)" % (suffix, self._print(base))
        return super()._print_Pow(expr)


_SFEM_CCODE_PRINTERS = {}


def _sfem_ccode(expression, scalar_type="scalar_t"):
    printer = _SFEM_CCODE_PRINTERS.get(scalar_type)
    if printer is None:
        printer = _SfemCCodePrinter(scalar_type)
        _SFEM_CCODE_PRINTERS[scalar_type] = printer
    return printer.doprint(expression)


def _prune_dead_cse_intermediates(intermediates, outputs):
    required_symbols = set()
    for output in outputs:
        required_symbols.update(_rhs(output).free_symbols)

    retained = []
    for symbol, expression in reversed(tuple(intermediates)):
        if symbol not in required_symbols:
            continue
        required_symbols.remove(symbol)
        required_symbols.update(expression.free_symbols)
        retained.append((symbol, expression))

    return tuple(reversed(retained))


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


def _sfem_math_function_lines(inline_qualifier="SFEM_INLINE"):
    lines = []
    for exponent in range(2, _SFEM_SPECIALIZED_POW_MAX_EXPONENT + 1):
        lines.extend(
            [
                "template <typename T>",
                "static %s T %s(const T x) {"
                % (inline_qualifier, _sfem_pow_function_name(exponent)),
                "    return %s;" % _sfem_pow_product_expression(exponent),
                "}",
                "",
            ]
        )
    for exponent in range(1, _SFEM_SPECIALIZED_POW_MAX_EXPONENT + 1):
        lines.extend(
            [
                "template <typename T>",
                "static %s T %s(const T x) {"
                % (inline_qualifier, _sfem_pow_function_name(-exponent)),
                "    return T(1) / %s(x);" % _sfem_pow_function_name(exponent)
                if exponent > 1
                else "    return T(1) / x;",
                "}",
                "",
            ]
        )
    return lines


def _sfem_math_header_source(
    header_guard_suffix="HPP",
    inline_qualifier="SFEM_INLINE",
    define_sfem_inline=True,
):
    guard = "SFEM_CODEGEN_KERNEL_MATH_%s" % header_guard_suffix
    lines = [
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "",
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
    lines.extend(["namespace sfem {", "namespace codegen {", ""])
    lines.extend(_sfem_math_function_lines(inline_qualifier))
    lines.extend(["} // namespace codegen", "} // namespace sfem", "", "#endif", ""])
    return "\n".join(lines)


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
class SfemSoAKernelForm:
    name: str
    expression_graph: Optional["ExpressionGraph"] = None
    has_direction: bool = False
    output_mode: str = "accumulate"
    weak_form: Optional["SfemSoAWeakForm"] = None
    dependencies: object = None

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
    dependencies=None,
):
    return SfemSoAKernelForm(name, expression_graph, has_direction, output_mode, weak_form, dependencies)


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


@dataclass(frozen=True)
class ExpressionCost:
    adds: int = 0
    muls: int = 0
    divs: int = 0
    sqrts: int = 0
    pows: int = 0
    exps: int = 0
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
            + 20 * self.exps
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
    intermediates = _prune_dead_cse_intermediates(intermediates, reduced_outputs)

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

def _component_name(component):
    return ("x", "y", "z")[component]


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
                target=lhs if lhs is not None else "output:%d" % idx,
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
    adds = muls = divs = sqrts = pows = exps = logs = trigs = stores = 0
    loaded = set()

    for _, expr in intermediates:
        a, m, d, s, p, exp_count, log_count, trig_count = _op_counts(expr)
        adds += a
        muls += m
        divs += d
        sqrts += s
        pows += p
        exps += exp_count
        logs += log_count
        trigs += trig_count
        loaded.update(expr.free_symbols)
        stores += 1

    for output in reduced_outputs:
        expr = _rhs(output)
        a, m, d, s, p, exp_count, log_count, trig_count = _op_counts(expr)
        adds += a
        muls += m
        divs += d
        sqrts += s
        pows += p
        exps += exp_count
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
        exps=exps,
        logs=logs,
        trigs=trigs,
        loads=loads,
        stores=stores,
        temporaries=temporaries,
        estimated_registers=estimated_registers,
    )


def _statement_cost(expression, data_symbols, stores):
    adds, muls, divs, sqrts, pows, exps, logs, trigs = _op_counts(expression)
    loaded = expression.free_symbols
    loads = len(loaded.intersection(data_symbols)) if data_symbols else len(loaded)
    return ExpressionCost(
        adds=adds,
        muls=muls,
        divs=divs,
        sqrts=sqrts,
        pows=pows,
        exps=exps,
        logs=logs,
        trigs=trigs,
        loads=loads,
        stores=stores,
        temporaries=0,
        estimated_registers=loads,
    )


def _op_counts(expression):
    adds = muls = divs = sqrts = pows = exps = logs = trigs = 0
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
            elif node.func == sp.exp:
                exps += 1
            elif node.func in trig_functions:
                trigs += 1
            elif node.func == sp.sqrt:
                sqrts += 1

    return adds, muls, divs, sqrts, pows, exps, logs, trigs
