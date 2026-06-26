from dataclasses import dataclass
from enum import Enum

from .basis import BasisPlanNode
from .forms import FormCollection, FormOrder, PipelineStage
from .geometry import GeometryPlanNode


class DataStreamRole(Enum):
    GEOMETRY = "geometry"
    COEFFICIENT = "coefficient"
    FIELD = "field"
    DIRECTION = "direction"
    OUTPUT = "output"
    MATERIAL_PARAMETER = "material_parameter"
    REFERENCE = "reference"
    TEMPORARY = "temporary"


class DataStreamLayout(Enum):
    SCALAR = "scalar"
    SOA = "soa"
    AOS = "aos"
    QP_SOA = "qp_soa"
    TENSOR_PRODUCT_1D = "tensor_product_1d"


class MeshPhase(Enum):
    GATHER = "gather"
    GEOMETRY = "geometry"
    LOCAL_CALL = "local_call"
    SCATTER = "scatter"


class LocalPhase(Enum):
    EVALUATE_TRIAL = "evaluate_trial"
    TRANSFORM_REFERENCE = "transform_reference"
    EVALUATE_MATERIAL = "evaluate_material"
    CONTRACT_TEST = "contract_test"


class KernelTarget(Enum):
    OPENMP = "openmp"
    CUDA = "cuda"


@dataclass(frozen=True)
class DataStreamPlan:
    name: str
    role: DataStreamRole
    layout: DataStreamLayout
    scalar_type: str = "scalar_t"
    components: int = 1
    n_items: int = 1
    source: str = ""

    def __post_init__(self):
        name = str(self.name)
        role = DataStreamRole(self.role)
        layout = DataStreamLayout(self.layout)
        scalar_type = str(self.scalar_type)
        components = int(self.components)
        n_items = int(self.n_items)
        source = str(self.source)
        if not name or not all(ch.isalnum() or ch in "_:[]" for ch in name):
            raise ValueError("data stream plan requires a non-empty stream name")
        if not scalar_type:
            raise ValueError("data stream scalar_type must be non-empty")
        if components <= 0 or n_items <= 0:
            raise ValueError("data stream components and n_items must be positive")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "scalar_type", scalar_type)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "n_items", n_items)
        object.__setattr__(self, "source", source)


@dataclass(frozen=True)
class GeometryPlan:
    node: GeometryPlanNode
    streams: tuple = ()

    def __post_init__(self):
        if not isinstance(self.node, GeometryPlanNode):
            raise TypeError("GeometryPlan requires a GeometryPlanNode")
        streams = tuple(self.streams)
        for stream in streams:
            if not isinstance(stream, DataStreamPlan):
                raise TypeError("GeometryPlan streams must be DataStreamPlan objects")
        object.__setattr__(self, "streams", streams)

    @property
    def mode(self):
        return self.node.mode

    @property
    def uses_sum_factorization(self):
        return self.node.uses_sum_factorization


@dataclass(frozen=True)
class BlockPlan:
    name: str
    row_field: str
    column_field: str = ""
    form_order: FormOrder = FormOrder.ONE
    local_phases: tuple = ()
    streams: tuple = ()
    basis_plans: tuple = ()

    def __post_init__(self):
        name = str(self.name)
        row_field = str(self.row_field)
        column_field = str(self.column_field)
        form_order = FormOrder(self.form_order)
        local_phases = tuple(LocalPhase(phase) for phase in self.local_phases)
        streams = tuple(self.streams)
        basis_plans = tuple(self.basis_plans)
        if not name:
            raise ValueError("block plan requires a name")
        if not row_field:
            raise ValueError("block plan requires a row field")
        for stream in streams:
            if not isinstance(stream, DataStreamPlan):
                raise TypeError("block plan streams must be DataStreamPlan objects")
        for basis in basis_plans:
            if not isinstance(basis, BasisPlanNode):
                raise TypeError("block plan basis_plans must be BasisPlanNode objects")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "row_field", row_field)
        object.__setattr__(self, "column_field", column_field)
        object.__setattr__(self, "form_order", form_order)
        object.__setattr__(self, "local_phases", local_phases)
        object.__setattr__(self, "streams", streams)
        object.__setattr__(self, "basis_plans", basis_plans)

    @property
    def is_diagonal(self):
        return not self.column_field or self.row_field == self.column_field


@dataclass(frozen=True)
class KernelPlan:
    name: str
    kind: object
    form_collection: FormCollection
    dim: int
    mesh_phases: tuple = ()
    geometry: GeometryPlan = None
    blocks: tuple = ()
    streams: tuple = ()
    target: KernelTarget = KernelTarget.OPENMP
    payload: object = None

    def __post_init__(self):
        name = str(self.name)
        dim = int(self.dim)
        mesh_phases = tuple(MeshPhase(phase) for phase in self.mesh_phases)
        blocks = tuple(self.blocks)
        streams = tuple(self.streams)
        target = KernelTarget(self.target)
        if not name:
            raise ValueError("kernel plan requires a name")
        if dim <= 0:
            raise ValueError("kernel plan dimension must be positive")
        if not isinstance(self.form_collection, FormCollection):
            raise TypeError("kernel plan requires a FormCollection")
        if self.geometry is not None and not isinstance(self.geometry, GeometryPlan):
            raise TypeError("kernel plan geometry must be a GeometryPlan")
        for block in blocks:
            if not isinstance(block, BlockPlan):
                raise TypeError("kernel plan blocks must be BlockPlan objects")
        for stream in streams:
            if not isinstance(stream, DataStreamPlan):
                raise TypeError("kernel plan streams must be DataStreamPlan objects")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "mesh_phases", mesh_phases)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "streams", streams)
        object.__setattr__(self, "target", target)

    def matches(self, context):
        return self.dim == context.specialization.dim


@dataclass(frozen=True)
class GenerationPlan:
    kernels: tuple

    @property
    def stage(self):
        return PipelineStage.SPECIALIZED_FORM_MANIPULATION

    def __post_init__(self):
        kernels = tuple(self.kernels)
        if not kernels:
            raise ValueError("generation plan requires at least one kernel")
        for kernel in kernels:
            if not isinstance(kernel, KernelPlan):
                raise TypeError("generation plan kernels must be KernelPlan objects")
        object.__setattr__(self, "kernels", kernels)

    @property
    def units(self):
        return self.kernels

    def units_for_context(self, context):
        return self.kernels_for_context(context)

    def kernels_for_context(self, context):
        return tuple(kernel for kernel in self.kernels if kernel.matches(context))
