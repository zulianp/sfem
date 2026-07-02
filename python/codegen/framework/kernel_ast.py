from dataclasses import dataclass, field
from enum import Enum


class KernelASTNodeKind(str, Enum):
    LOOP = "loop"
    ASSIGNMENT = "assignment"
    BUFFER_DECL = "buffer_decl"
    CALL = "call"
    GATHER = "gather"
    SCATTER = "scatter"
    GEOMETRY = "geometry"
    LOCAL_COMPUTATION = "local_computation"


class GeometryNodeKind(str, Enum):
    COORDINATE_GATHER = "coordinate_gather"
    TRANSIENT_JACOBIAN = "transient_jacobian"
    ADJUGATE = "adjugate"
    DETERMINANT = "determinant"
    MEASURE = "measure"


class LoopKind(str, Enum):
    KERNEL = "kernel"
    TILE = "tile"
    QUADRATURE = "quadrature"
    SHAPE = "shape"
    DIMENSION = "dimension"
    TENSOR_PRODUCT = "tensor_product"
    SIMD = "simd"
    SCATTER = "scatter"
    REDUCTION = "reduction"


class PartialAssemblyStrategy(str, Enum):
    MATRIX_FREE = "matrix_free"
    AFFINE_GEOMETRY_PA = "affine_geometry_pa"
    ISOPARAMETRIC_GEOMETRY_PA = "isoparametric_geometry_pa"
    APPLY_PA = "apply_pa"


class LoopIncrementKind(str, Enum):
    PRE_INCREMENT = "pre_increment"
    ADD_ASSIGN = "add_assign"


@dataclass(frozen=True)
class TypeRef:
    name: str

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name))

    def to_dict(self):
        return {"kind": "type", "name": self.name}


@dataclass(frozen=True)
class SymbolRef:
    name: str

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name))

    def to_dict(self):
        return {"kind": "symbol", "name": self.name}


@dataclass(frozen=True)
class ExpressionRef:
    expression: str
    role: str = "external_expression"

    def __post_init__(self):
        object.__setattr__(self, "expression", str(self.expression))
        object.__setattr__(self, "role", str(self.role))

    def to_dict(self):
        return {
            "kind": "expression",
            "role": self.role,
            "expression": self.expression,
        }


@dataclass(frozen=True)
class Literal:
    value: object

    def to_dict(self):
        return {"kind": "literal", "value": self.value}


@dataclass(frozen=True)
class Iterator:
    symbol: SymbolRef
    index_type: TypeRef

    def __post_init__(self):
        object.__setattr__(self, "symbol", _as_symbol(self.symbol))
        object.__setattr__(self, "index_type", _as_type(self.index_type))

    def to_dict(self):
        return {
            "kind": "iterator",
            "symbol": self.symbol.to_dict(),
            "index_type": self.index_type.to_dict(),
        }


@dataclass(frozen=True)
class IterationRange:
    begin: object
    end: object

    def __post_init__(self):
        object.__setattr__(self, "begin", _as_expr(self.begin))
        object.__setattr__(self, "end", _as_expr(self.end))

    def to_dict(self):
        return {
            "kind": "range",
            "begin": _entity_to_dict(self.begin),
            "end": _entity_to_dict(self.end),
        }


@dataclass(frozen=True)
class LoopIncrement:
    kind: LoopIncrementKind
    iterator: Iterator
    amount: object = None

    def __post_init__(self):
        object.__setattr__(self, "kind", LoopIncrementKind(self.kind))
        object.__setattr__(self, "iterator", self.iterator)
        if self.amount is not None:
            object.__setattr__(self, "amount", _as_expr(self.amount))

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "iterator": self.iterator.to_dict(),
            "amount": None if self.amount is None else _entity_to_dict(self.amount),
        }


@dataclass(frozen=True)
class BufferAccess:
    base: SymbolRef
    indices: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "base", _as_symbol(self.base))
        object.__setattr__(self, "indices", tuple(_as_expr(index) for index in self.indices))

    def to_dict(self):
        return {
            "kind": "buffer_access",
            "base": self.base.to_dict(),
            "indices": [_entity_to_dict(index) for index in self.indices],
        }


@dataclass(frozen=True)
class VectorizationStrategy:
    name: str = "simd_lane"
    vector_width_symbol: str = "VECTOR_SIZE"
    lane_index: str = "lane"

    def to_dict(self):
        return {
            "name": self.name,
            "vector_width_symbol": self.vector_width_symbol,
            "lane_index": self.lane_index,
        }


@dataclass(frozen=True)
class LoopStrategy:
    name: str
    iterator: Iterator
    iteration_range: IterationRange
    increment: LoopIncrement
    vectorization: VectorizationStrategy = None

    def to_dict(self):
        return {
            "name": self.name,
            "iterator": self.iterator.to_dict(),
            "range": self.iteration_range.to_dict(),
            "increment": self.increment.to_dict(),
            "vectorization": None
            if self.vectorization is None
            else self.vectorization.to_dict(),
        }


@dataclass(frozen=True)
class TabulationStrategy:
    name: str = "none"
    description: str = ""

    def to_dict(self):
        return {"name": self.name, "description": self.description}


@dataclass(frozen=True)
class KernelAST:
    name: str
    nodes: tuple = ()
    loop_strategy: LoopStrategy = None
    vectorization_strategy: VectorizationStrategy = None
    tabulation_strategy: TabulationStrategy = None
    partial_assembly_strategy: PartialAssemblyStrategy = PartialAssemblyStrategy.MATRIX_FREE

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(
            self,
            "partial_assembly_strategy",
            PartialAssemblyStrategy(self.partial_assembly_strategy),
        )

    def to_dict(self):
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes],
            "loop_strategy": None
            if self.loop_strategy is None
            else self.loop_strategy.to_dict(),
            "vectorization_strategy": None
            if self.vectorization_strategy is None
            else self.vectorization_strategy.to_dict(),
            "tabulation_strategy": None
            if self.tabulation_strategy is None
            else self.tabulation_strategy.to_dict(),
            "partial_assembly_strategy": self.partial_assembly_strategy.value,
        }


@dataclass(frozen=True)
class LoopNode:
    loop_kind: LoopKind
    iterator: Iterator
    iteration_range: IterationRange
    increment: LoopIncrement
    body: tuple = ()
    vectorized: bool = False
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.LOOP, init=False)

    def __post_init__(self):
        object.__setattr__(self, "loop_kind", LoopKind(self.loop_kind))
        object.__setattr__(self, "iterator", self.iterator)
        object.__setattr__(self, "iteration_range", self.iteration_range)
        object.__setattr__(self, "increment", self.increment)
        object.__setattr__(self, "body", tuple(self.body))

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "loop_kind": self.loop_kind.value,
            "iterator": self.iterator.to_dict(),
            "range": self.iteration_range.to_dict(),
            "increment": self.increment.to_dict(),
            "vectorized": self.vectorized,
            "body": [node.to_dict() for node in self.body],
        }


@dataclass(frozen=True)
class AssignmentNode:
    lhs: object
    rhs: object
    operator: str = "="
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.ASSIGNMENT, init=False)

    def __post_init__(self):
        object.__setattr__(self, "lhs", _as_expr(self.lhs))
        object.__setattr__(self, "rhs", _as_expr(self.rhs))

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "lhs": _entity_to_dict(self.lhs),
            "operator": self.operator,
            "rhs": _entity_to_dict(self.rhs),
        }


@dataclass(frozen=True)
class BufferDeclNode:
    scalar_type: TypeRef
    name: SymbolRef
    extents: tuple
    initializer: object = None
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.BUFFER_DECL, init=False)

    def __post_init__(self):
        object.__setattr__(self, "scalar_type", _as_type(self.scalar_type))
        object.__setattr__(self, "name", _as_symbol(self.name))
        object.__setattr__(self, "extents", tuple(_as_expr(extent) for extent in self.extents))
        if self.initializer is not None:
            object.__setattr__(self, "initializer", _as_expr(self.initializer))

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "scalar_type": self.scalar_type.to_dict(),
            "name": self.name.to_dict(),
            "extents": [_entity_to_dict(extent) for extent in self.extents],
            "initializer": None
            if self.initializer is None
            else _entity_to_dict(self.initializer),
        }


@dataclass(frozen=True)
class CallNode:
    callee: SymbolRef
    arguments: tuple = ()
    template_arguments: tuple = ()
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.CALL, init=False)

    def __post_init__(self):
        object.__setattr__(self, "callee", _as_symbol(self.callee))
        object.__setattr__(self, "arguments", tuple(_as_expr(arg) for arg in self.arguments))
        object.__setattr__(
            self,
            "template_arguments",
            tuple(_as_expr(arg) for arg in self.template_arguments),
        )

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "callee": self.callee.to_dict(),
            "template_arguments": [_entity_to_dict(arg) for arg in self.template_arguments],
            "arguments": [_entity_to_dict(arg) for arg in self.arguments],
        }


@dataclass(frozen=True)
class GatherNode:
    target: object
    source: object
    index: object
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.GATHER, init=False)

    def __post_init__(self):
        object.__setattr__(self, "target", _as_expr(self.target))
        object.__setattr__(self, "source", _as_expr(self.source))
        object.__setattr__(self, "index", _as_expr(self.index))

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "target": _entity_to_dict(self.target),
            "source": _entity_to_dict(self.source),
            "index": _entity_to_dict(self.index),
        }


@dataclass(frozen=True)
class ScatterNode:
    target: object
    value: object
    operator: str = "+="
    atomic: bool = False
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.SCATTER, init=False)

    def __post_init__(self):
        object.__setattr__(self, "target", _as_expr(self.target))
        object.__setattr__(self, "value", _as_expr(self.value))

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "target": _entity_to_dict(self.target),
            "operator": self.operator,
            "value": _entity_to_dict(self.value),
            "atomic": self.atomic,
        }


@dataclass(frozen=True)
class GeometryNode:
    geometry_kind: GeometryNodeKind
    outputs: tuple = ()
    inputs: tuple = ()
    scope: str = ""
    persist: bool = False
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.GEOMETRY, init=False)

    def __post_init__(self):
        object.__setattr__(self, "geometry_kind", GeometryNodeKind(self.geometry_kind))
        object.__setattr__(self, "outputs", tuple(_as_expr(output) for output in self.outputs))
        object.__setattr__(self, "inputs", tuple(_as_expr(input_) for input_ in self.inputs))

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "geometry_kind": self.geometry_kind.value,
            "inputs": [_entity_to_dict(input_) for input_ in self.inputs],
            "outputs": [_entity_to_dict(output) for output in self.outputs],
            "scope": self.scope,
            "persist": self.persist,
        }


@dataclass(frozen=True)
class LocalComputationNode:
    evaluation_plan: object
    output_name: str
    cost: object = None
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.LOCAL_COMPUTATION, init=False)

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "output_name": self.output_name,
            "statement_count": len(tuple(getattr(self.evaluation_plan, "statements", ()))),
            "cost": None if self.cost is None else _cost_to_dict(self.cost),
        }


@dataclass(frozen=True)
class KernelASTPass:
    name: str
    preconditions: tuple = ()
    expected_impact: dict = field(default_factory=dict)
    parity_preserving: bool = True
    performance_changing: bool = False

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "preconditions", tuple(self.preconditions))
        object.__setattr__(self, "expected_impact", dict(self.expected_impact))
        if self.performance_changing and self.parity_preserving:
            raise ValueError("a performance-changing AST pass must not be marked parity-preserving")

    def apply(self, ast):
        return KernelASTPassResult(self, ast, ast, ())

    def to_dict(self):
        return {
            "name": self.name,
            "preconditions": list(self.preconditions),
            "expected_impact": dict(self.expected_impact),
            "parity_preserving": self.parity_preserving,
            "performance_changing": self.performance_changing,
        }


@dataclass(frozen=True)
class KernelASTPassResult:
    ast_pass: KernelASTPass
    before: KernelAST
    after: KernelAST
    changed_nodes: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "changed_nodes", tuple(self.changed_nodes))

    @property
    def changed(self):
        return self.before is not self.after or bool(self.changed_nodes)

    def to_dict(self):
        return {
            "pass": self.ast_pass.to_dict(),
            "changed": self.changed,
            "changed_nodes": [str(node) for node in self.changed_nodes],
        }


@dataclass(frozen=True)
class KernelASTPassPipeline:
    passes: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "passes", tuple(self.passes))

    def apply(self, ast):
        current = ast
        results = []
        for ast_pass in self.passes:
            result = ast_pass.apply(current)
            results.append(result)
            current = result.after
        return current, tuple(results)


def _cost_to_dict(cost):
    return {
        "adds": getattr(cost, "adds", 0),
        "muls": getattr(cost, "muls", 0),
        "divs": getattr(cost, "divs", 0),
        "sqrts": getattr(cost, "sqrts", 0),
        "pows": getattr(cost, "pows", 0),
        "exps": getattr(cost, "exps", 0),
        "logs": getattr(cost, "logs", 0),
        "trigs": getattr(cost, "trigs", 0),
        "loads": getattr(cost, "loads", 0),
        "stores": getattr(cost, "stores", 0),
        "flops": getattr(cost, "flops", 0),
        "temporaries": getattr(cost, "temporaries", 0),
        "estimated_registers": getattr(cost, "estimated_registers", 0),
    }


def type_ref(name):
    return TypeRef(name)


def symbol_ref(name):
    return SymbolRef(name)


def expr_ref(expression, role="external_expression"):
    return ExpressionRef(expression, role)


def literal(value):
    return Literal(value)


def iterator(name, index_type):
    return Iterator(SymbolRef(name), TypeRef(index_type))


def iteration_range(begin, end):
    return IterationRange(begin, end)


def pre_increment(iterator_):
    return LoopIncrement(LoopIncrementKind.PRE_INCREMENT, iterator_)


def add_assign_increment(iterator_, amount):
    return LoopIncrement(LoopIncrementKind.ADD_ASSIGN, iterator_, amount)


def buffer_access(base, *indices):
    return BufferAccess(base, indices)


def _as_type(value):
    if isinstance(value, TypeRef):
        return value
    return TypeRef(value)


def _as_symbol(value):
    if isinstance(value, SymbolRef):
        return value
    return SymbolRef(value)


def _as_expr(value):
    if isinstance(value, (ExpressionRef, Literal, SymbolRef, BufferAccess)):
        return value
    if isinstance(value, (int, float)):
        return Literal(value)
    return ExpressionRef(value)


def _entity_to_dict(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return {"kind": "unknown", "value": str(value)}
