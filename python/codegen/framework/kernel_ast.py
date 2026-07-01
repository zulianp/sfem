from dataclasses import dataclass, field
from enum import Enum


class KernelASTNodeKind(str, Enum):
    OPAQUE_STATEMENT = "opaque_statement"
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
    index: str
    extent: str
    increment: str = "++%s"
    vectorization: VectorizationStrategy = None

    def to_dict(self):
        return {
            "name": self.name,
            "index": self.index,
            "extent": self.extent,
            "increment": self.increment,
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
class OpaqueStatementNode:
    statement: str
    reason: str = "parity_migration"
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.OPAQUE_STATEMENT, init=False)

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "statement": self.statement,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LoopNode:
    loop_kind: LoopKind
    index_type: str
    index: str
    begin: str
    end: str
    increment: str
    body: tuple = ()
    vectorized: bool = False
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.LOOP, init=False)

    def __post_init__(self):
        object.__setattr__(self, "loop_kind", LoopKind(self.loop_kind))
        object.__setattr__(self, "body", tuple(self.body))

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "loop_kind": self.loop_kind.value,
            "index_type": self.index_type,
            "index": self.index,
            "begin": self.begin,
            "end": self.end,
            "increment": self.increment,
            "vectorized": self.vectorized,
            "body": [node.to_dict() for node in self.body],
        }


@dataclass(frozen=True)
class AssignmentNode:
    lhs: str
    rhs: str
    operator: str = "="
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.ASSIGNMENT, init=False)

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "lhs": self.lhs,
            "operator": self.operator,
            "rhs": self.rhs,
        }


@dataclass(frozen=True)
class BufferDeclNode:
    scalar_type: str
    name: str
    extents: tuple
    initializer: str = ""
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.BUFFER_DECL, init=False)

    def __post_init__(self):
        object.__setattr__(self, "extents", tuple(str(extent) for extent in self.extents))

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "scalar_type": self.scalar_type,
            "name": self.name,
            "extents": list(self.extents),
            "initializer": self.initializer,
        }


@dataclass(frozen=True)
class CallNode:
    callee: str
    arguments: tuple = ()
    template_arguments: tuple = ()
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.CALL, init=False)

    def __post_init__(self):
        object.__setattr__(self, "arguments", tuple(str(arg) for arg in self.arguments))
        object.__setattr__(
            self,
            "template_arguments",
            tuple(str(arg) for arg in self.template_arguments),
        )

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "callee": self.callee,
            "template_arguments": list(self.template_arguments),
            "arguments": list(self.arguments),
        }


@dataclass(frozen=True)
class GatherNode:
    target: str
    source: str
    index: str
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.GATHER, init=False)

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "target": self.target,
            "source": self.source,
            "index": self.index,
        }


@dataclass(frozen=True)
class ScatterNode:
    target: str
    value: str
    operator: str = "+="
    atomic: bool = False
    kind: KernelASTNodeKind = field(default=KernelASTNodeKind.SCATTER, init=False)

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "target": self.target,
            "operator": self.operator,
            "value": self.value,
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
        object.__setattr__(self, "outputs", tuple(str(output) for output in self.outputs))
        object.__setattr__(self, "inputs", tuple(str(input_) for input_ in self.inputs))

    def to_dict(self):
        return {
            "kind": self.kind.value,
            "geometry_kind": self.geometry_kind.value,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
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
