import json
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


class KernelScope(Enum):
    MONOLITHIC = "monolithic"
    BLOCK = "block"


class KernelCoupling(Enum):
    SINGLE_FIELD = "single_field"
    COMPLETE_SYSTEM = "complete_system"
    BLOCK = "block"


class KernelEmission(Enum):
    FILES = "files"
    COVERED_BY_PARENT = "covered_by_parent"


@dataclass(frozen=True)
class LocalKernelPlan:
    prefix: str
    dim: int
    family: str
    suffix: str = ""

    def __post_init__(self):
        prefix = str(self.prefix)
        dim = int(self.dim)
        family = str(self.family)
        suffix = str(self.suffix)
        if not prefix:
            raise ValueError("local kernel plan requires a prefix")
        if dim <= 0:
            raise ValueError("local kernel plan dimension must be positive")
        if family not in ("simplex", "tensor_product"):
            raise ValueError("unsupported local kernel family '%s'" % family)
        if suffix and (not suffix.startswith("_") or not suffix[1:].isidentifier()):
            raise ValueError("local kernel suffix must be empty or an identifier prefixed by '_'")
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "suffix", suffix)

    @property
    def name(self):
        return "%s_d%d_%s%s" % (self.prefix, self.dim, self.family, self.suffix)

    @property
    def header(self):
        return "%s_local.hpp" % self.name

    def to_dict(self):
        return {
            "name": self.name,
            "prefix": self.prefix,
            "dim": self.dim,
            "family": self.family,
            "suffix": self.suffix,
            "header": self.header,
        }


@dataclass(frozen=True)
class MeshKernelPlan:
    prefix: str
    element_label: str

    def __post_init__(self):
        prefix = str(self.prefix)
        element_label = str(self.element_label).lower()
        if not prefix:
            raise ValueError("mesh kernel plan requires a prefix")
        if not element_label or not all(ch.isalnum() or ch == "_" for ch in element_label):
            raise ValueError("mesh kernel element label must be a non-empty identifier fragment")
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "element_label", element_label)

    @property
    def name(self):
        return "%s_%s" % (self.prefix, self.element_label)

    @property
    def source(self):
        return "%s_operator.cpp" % self.name

    def to_dict(self):
        return {
            "name": self.name,
            "prefix": self.prefix,
            "element_label": self.element_label,
            "source": self.source,
        }


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

    def to_dict(self):
        return {
            "name": self.name,
            "role": self.role.value,
            "layout": self.layout.value,
            "scalar_type": self.scalar_type,
            "components": self.components,
            "n_items": self.n_items,
            "source": self.source,
        }


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

    def to_dict(self):
        return {
            "mode": self.node.mode.value,
            "element_type": self.node.element_type,
            "dim": self.node.dim,
            "input_layout": self.node.input_layout.value,
            "evaluation": self.node.evaluation.value,
            "jacobian_scope": self.node.jacobian_scope,
            "geometry_points_per_element": self.node.geometry_points_per_element,
            "uses_sum_factorization": self.node.uses_sum_factorization,
            "streams": [stream.to_dict() for stream in self.streams],
        }


@dataclass(frozen=True)
class BlockPlan:
    name: str
    row_field: str
    column_field: str = ""
    form_order: FormOrder = FormOrder.ONE
    local_phases: tuple = ()
    streams: tuple = ()
    basis_plans: tuple = ()
    local_phase_plans: tuple = ()

    def __post_init__(self):
        name = str(self.name)
        row_field = str(self.row_field)
        column_field = str(self.column_field)
        form_order = FormOrder(self.form_order)
        streams = tuple(self.streams)
        basis_plans = tuple(self.basis_plans)
        local_phase_plans = tuple(self.local_phase_plans)
        if local_phase_plans:
            for plan in local_phase_plans:
                if not isinstance(plan, LocalPhasePlan):
                    raise TypeError("block plan local_phase_plans must be LocalPhasePlan objects")
            local_phases = tuple(plan.phase for plan in local_phase_plans)
            requested_local_phases = tuple(LocalPhase(phase) for phase in self.local_phases)
            if requested_local_phases and requested_local_phases != local_phases:
                raise ValueError("block plan local_phases do not match local_phase_plans")
        else:
            local_phase_plans = tuple(LocalPhasePlan(phase) for phase in self.local_phases)
            local_phases = tuple(plan.phase for plan in local_phase_plans)
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
        object.__setattr__(self, "local_phase_plans", local_phase_plans)

    @property
    def is_diagonal(self):
        return not self.column_field or self.row_field == self.column_field

    def to_dict(self):
        return {
            "name": self.name,
            "row_field": self.row_field,
            "column_field": self.column_field,
            "form_order": self.form_order.value,
            "local_phases": [phase.value for phase in self.local_phases],
            "local_phase_plans": [phase.to_dict() for phase in self.local_phase_plans],
            "is_diagonal": self.is_diagonal,
            "streams": [stream.to_dict() for stream in self.streams],
            "basis_plans": [
                {
                    "role": basis.role,
                    "element_type": basis.element_type,
                    "cell_element_type": basis.cell_element_type,
                    "family": basis.family.value,
                    "n_shape": basis.n_shape,
                    "n_qp": basis.n_qp,
                    "uses_sum_factorization": basis.uses_sum_factorization,
                }
                for basis in self.basis_plans
            ],
        }


@dataclass(frozen=True)
class LocalPhasePlan:
    phase: LocalPhase
    streams: tuple = ()
    basis_plans: tuple = ()
    label: str = ""

    def __post_init__(self):
        phase = LocalPhase(self.phase)
        streams = tuple(self.streams)
        basis_plans = tuple(self.basis_plans)
        label = str(self.label)
        for stream in streams:
            if not isinstance(stream, DataStreamPlan):
                raise TypeError("local phase streams must be DataStreamPlan objects")
        for basis in basis_plans:
            if not isinstance(basis, BasisPlanNode):
                raise TypeError("local phase basis_plans must be BasisPlanNode objects")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "streams", streams)
        object.__setattr__(self, "basis_plans", basis_plans)
        object.__setattr__(self, "label", label)

    @property
    def is_evaluate_trial(self):
        return self.phase is LocalPhase.EVALUATE_TRIAL

    @property
    def is_transform_reference(self):
        return self.phase is LocalPhase.TRANSFORM_REFERENCE

    @property
    def is_evaluate_material(self):
        return self.phase is LocalPhase.EVALUATE_MATERIAL

    @property
    def is_contract_test(self):
        return self.phase is LocalPhase.CONTRACT_TEST

    def to_dict(self):
        return {
            "phase": self.phase.value,
            "label": self.label,
            "streams": [stream.name for stream in self.streams],
            "basis_plans": [
                {
                    "role": basis.role,
                    "element_type": basis.element_type,
                    "cell_element_type": basis.cell_element_type,
                    "family": basis.family.value,
                    "n_shape": basis.n_shape,
                    "n_qp": basis.n_qp,
                }
                for basis in self.basis_plans
            ],
        }


@dataclass(frozen=True)
class MeshPhasePlan:
    phase: MeshPhase
    streams: tuple = ()
    geometry: GeometryPlan = None
    blocks: tuple = ()
    label: str = ""

    def __post_init__(self):
        phase = MeshPhase(self.phase)
        streams = tuple(self.streams)
        blocks = tuple(self.blocks)
        label = str(self.label)
        for stream in streams:
            if not isinstance(stream, DataStreamPlan):
                raise TypeError("mesh phase streams must be DataStreamPlan objects")
        if self.geometry is not None and not isinstance(self.geometry, GeometryPlan):
            raise TypeError("mesh phase geometry must be a GeometryPlan")
        for block in blocks:
            if not isinstance(block, BlockPlan):
                raise TypeError("mesh phase blocks must be BlockPlan objects")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "streams", streams)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "label", label)

    @property
    def is_gather(self):
        return self.phase is MeshPhase.GATHER

    @property
    def is_geometry(self):
        return self.phase is MeshPhase.GEOMETRY

    @property
    def is_local_call(self):
        return self.phase is MeshPhase.LOCAL_CALL

    @property
    def is_scatter(self):
        return self.phase is MeshPhase.SCATTER

    def to_dict(self):
        return {
            "phase": self.phase.value,
            "label": self.label,
            "geometry": None if self.geometry is None else self.geometry.to_dict(),
            "blocks": [block.name for block in self.blocks],
            "streams": [stream.name for stream in self.streams],
        }


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
    mesh_phase_plans: tuple = ()
    scope: KernelScope = KernelScope.MONOLITHIC
    coupling: KernelCoupling = KernelCoupling.SINGLE_FIELD
    block: BlockPlan = None
    block_kernels: tuple = ()
    emission: KernelEmission = KernelEmission.FILES

    def __post_init__(self):
        name = str(self.name)
        dim = int(self.dim)
        blocks = tuple(self.blocks)
        streams = tuple(self.streams)
        target = KernelTarget(self.target)
        scope = KernelScope(self.scope)
        coupling = KernelCoupling(self.coupling)
        emission = KernelEmission(self.emission)
        block_kernels = tuple(self.block_kernels)
        mesh_phase_plans = tuple(self.mesh_phase_plans)
        if mesh_phase_plans:
            for plan in mesh_phase_plans:
                if not isinstance(plan, MeshPhasePlan):
                    raise TypeError("kernel plan mesh_phase_plans must be MeshPhasePlan objects")
            mesh_phases = tuple(plan.phase for plan in mesh_phase_plans)
            requested_mesh_phases = tuple(MeshPhase(phase) for phase in self.mesh_phases)
            if requested_mesh_phases and requested_mesh_phases != mesh_phases:
                raise ValueError("kernel plan mesh_phases do not match mesh_phase_plans")
        else:
            mesh_phase_plans = self._mesh_phase_plans_from_phases(
                self.mesh_phases,
                self.geometry,
                blocks,
                streams,
            )
            mesh_phases = tuple(plan.phase for plan in mesh_phase_plans)
        if not name:
            raise ValueError("kernel plan requires a name")
        if dim <= 0:
            raise ValueError("kernel plan dimension must be positive")
        if not isinstance(self.form_collection, FormCollection):
            raise TypeError("kernel plan requires a FormCollection")
        if self.geometry is not None and not isinstance(self.geometry, GeometryPlan):
            raise TypeError("kernel plan geometry must be a GeometryPlan")
        if self.block is not None and not isinstance(self.block, BlockPlan):
            raise TypeError("kernel plan block must be a BlockPlan")
        if scope is KernelScope.BLOCK and self.block is None:
            raise ValueError("block kernel plans require a block")
        if scope is KernelScope.MONOLITHIC and self.block is not None:
            raise ValueError("monolithic kernel plans cannot select a single block")
        if scope is KernelScope.BLOCK and coupling is not KernelCoupling.BLOCK:
            raise ValueError("block kernel plans require BLOCK coupling")
        if scope is KernelScope.MONOLITHIC and coupling is KernelCoupling.BLOCK:
            raise ValueError("monolithic kernel plans cannot use BLOCK coupling")
        if scope is KernelScope.MONOLITHIC and emission is KernelEmission.COVERED_BY_PARENT:
            raise ValueError("monolithic kernel plans must own file emission")
        for block in blocks:
            if not isinstance(block, BlockPlan):
                raise TypeError("kernel plan blocks must be BlockPlan objects")
        for stream in streams:
            if not isinstance(stream, DataStreamPlan):
                raise TypeError("kernel plan streams must be DataStreamPlan objects")
        for kernel in block_kernels:
            if not isinstance(kernel, KernelPlan):
                raise TypeError("kernel plan block_kernels must be KernelPlan objects")
            if kernel.scope is not KernelScope.BLOCK:
                raise ValueError("kernel plan block_kernels must have BLOCK scope")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "mesh_phases", mesh_phases)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "streams", streams)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "mesh_phase_plans", mesh_phase_plans)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "coupling", coupling)
        object.__setattr__(self, "emission", emission)
        object.__setattr__(self, "block_kernels", block_kernels)

    @staticmethod
    def _mesh_phase_plans_from_phases(mesh_phases, geometry, blocks, streams):
        phase_plans = []
        for phase in mesh_phases:
            if isinstance(phase, MeshPhasePlan):
                phase_plans.append(phase)
                continue
            phase = MeshPhase(phase)
            phase_plans.append(
                MeshPhasePlan(
                    phase,
                    streams=streams if phase in (MeshPhase.GATHER, MeshPhase.SCATTER) else (),
                    geometry=geometry if phase is MeshPhase.GEOMETRY else None,
                    blocks=blocks if phase is MeshPhase.LOCAL_CALL else (),
                )
            )
        return tuple(phase_plans)

    def matches(self, context):
        return self.dim == context.specialization.dim

    @property
    def is_monolithic(self):
        return self.scope is KernelScope.MONOLITHIC

    @property
    def is_block(self):
        return self.scope is KernelScope.BLOCK

    @property
    def is_complete_system(self):
        return self.coupling is KernelCoupling.COMPLETE_SYSTEM

    @property
    def is_single_field(self):
        return self.coupling is KernelCoupling.SINGLE_FIELD

    @property
    def emits_files(self):
        return self.emission is KernelEmission.FILES

    def validate_for_context(self, context):
        if not self.matches(context):
            raise ValueError(
                "kernel '%s' dimension %d does not match context dimension %d"
                % (self.name, self.dim, context.specialization.dim)
            )
        if self.target is not KernelTarget.OPENMP:
            raise ValueError("kernel '%s' target '%s' is not supported" % (self.name, self.target.value))
        _validate_phase_sequence(
            "mesh",
            tuple(phase.phase for phase in self.mesh_phase_plans),
            (MeshPhase.GATHER, MeshPhase.GEOMETRY, MeshPhase.LOCAL_CALL, MeshPhase.SCATTER),
        )
        if self.geometry is not None:
            if not any(plan is self.geometry.node for plan in context.geometry_plans):
                raise ValueError("kernel '%s' geometry plan is not available for context" % self.name)
        field_names = _form_field_names(self.form_collection)
        for block in self.blocks:
            _validate_block_fields(self.name, block, field_names)
            _validate_local_phase_sequence(block)
        if self.block is not None:
            _validate_block_fields(self.name, self.block, field_names)
            _validate_local_phase_sequence(self.block)
        for block_kernel in self.block_kernels:
            block_kernel.validate_for_context(context)

    def local_kernel_plan(self, context, prefix, suffix=""):
        return LocalKernelPlan(
            prefix,
            context.specialization.dim,
            context.family,
            suffix,
        )

    def mesh_kernel_plan(self, context, prefix):
        return MeshKernelPlan(prefix, context.label)

    def to_dict(self, include_block_kernels=True):
        return {
            "name": self.name,
            "kind": _value_or_name(self.kind),
            "dim": self.dim,
            "target": self.target.value,
            "scope": self.scope.value,
            "coupling": self.coupling.value,
            "emission": self.emission.value,
            "selected_block": None if self.block is None else self.block.name,
            "form_collection": _form_collection_dump(self.form_collection),
            "mesh_phases": [phase.value for phase in self.mesh_phases],
            "mesh_phase_plans": [phase.to_dict() for phase in self.mesh_phase_plans],
            "geometry": None if self.geometry is None else self.geometry.to_dict(),
            "blocks": [block.to_dict() for block in self.blocks],
            "streams": [stream.to_dict() for stream in self.streams],
            "block_kernels": [
                kernel.to_dict(include_block_kernels=False)
                for kernel in self.block_kernels
            ] if include_block_kernels else [],
        }


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

    def validate_for_context(self, context):
        matched = self.kernels_for_context(context)
        if not matched:
            raise ValueError(
                "generation plan has no kernels for context '%s' dimension %d"
                % (context.element_type, context.specialization.dim)
            )
        for kernel in matched:
            kernel.validate_for_context(context)

    def emission_kernels_for_context(self, context):
        self.validate_for_context(context)
        return tuple(
            kernel
            for kernel in _flatten_kernels(self.kernels_for_context(context))
            if kernel.emits_files
        )

    @property
    def monolithic_kernels(self):
        return tuple(kernel for kernel in self.kernels if kernel.is_monolithic)

    @property
    def block_kernels(self):
        return tuple(
            block_kernel
            for kernel in self.kernels
            for block_kernel in kernel.block_kernels
        )

    @property
    def complete_system_kernels(self):
        return tuple(kernel for kernel in self.kernels if kernel.is_complete_system)

    def to_dict(self):
        return {
            "stage": self.stage.value,
            "n_kernels": len(self.kernels),
            "n_monolithic_kernels": len(self.monolithic_kernels),
            "n_block_kernels": len(self.block_kernels),
            "n_complete_system_kernels": len(self.complete_system_kernels),
            "kernels": [kernel.to_dict() for kernel in self.kernels],
        }

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write_json(self, path, indent=2):
        with open(path, "w") as output:
            output.write(self.to_json(indent=indent))
            output.write("\n")


def _value_or_name(value):
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "name"):
        return value.name
    return str(value)


def _flatten_kernels(kernels):
    ret = []
    for kernel in kernels:
        ret.append(kernel)
        ret.extend(_flatten_kernels(kernel.block_kernels))
    return tuple(ret)


def _validate_phase_sequence(name, phases, allowed_order):
    order = {phase: index for index, phase in enumerate(allowed_order)}
    previous = -1
    seen = set()
    for phase in phases:
        if phase not in order:
            raise ValueError("unsupported %s phase '%s'" % (name, phase.value))
        current = order[phase]
        if current < previous:
            raise ValueError("%s phases are not in canonical order" % name)
        if phase in seen:
            raise ValueError("%s phase '%s' appears more than once" % (name, phase.value))
        seen.add(phase)
        previous = current


def _validate_block_fields(kernel_name, block, field_names):
    if block.row_field not in field_names:
        raise ValueError(
            "kernel '%s' block '%s' row field '%s' is not in the form collection"
            % (kernel_name, block.name, block.row_field)
        )


def _form_field_names(collection):
    names = set()
    for field in collection.fields:
        names.add(field.name)
        for component in range(int(field.components)):
            names.add("%s%d" % (field.name, component))
    return names
    if block.column_field and block.column_field not in field_names:
        raise ValueError(
            "kernel '%s' block '%s' column field '%s' is not in the form collection"
            % (kernel_name, block.name, block.column_field)
        )


def _validate_local_phase_sequence(block):
    _validate_phase_sequence(
        "local",
        tuple(phase.phase for phase in block.local_phase_plans),
        (
            LocalPhase.EVALUATE_TRIAL,
            LocalPhase.TRANSFORM_REFERENCE,
            LocalPhase.EVALUATE_MATERIAL,
            LocalPhase.CONTRACT_TEST,
        ),
    )


def _form_collection_dump(collection):
    return {
        "equation_name": collection.equation_name,
        "kind": collection.kind.value,
        "fields": [
            {
                "name": field.name,
                "components": field.components,
                "family": field.family,
            }
            for field in collection.fields
        ],
        "forms": [
            {
                "order": form.order.value,
                "standard_name": form.standard_name,
                "name": form.name,
                "role": _value_or_name(form.role),
            }
            for form in collection.forms
        ],
        "blocks": [
            {
                "name": block.name,
                "order": block.order.value,
                "row_field": block.row_field,
                "column_field": block.column_field,
                "is_diagonal": block.is_diagonal,
                "is_coupling": block.is_coupling,
            }
            for block in collection.blocks
        ],
    }
