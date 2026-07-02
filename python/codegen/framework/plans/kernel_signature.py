from dataclasses import dataclass
from types import SimpleNamespace

from codegen.framework.symbolic.forms import FormOrder


@dataclass(frozen=True)
class KernelArgument:
    name: str
    declaration: str
    role: str

    def __post_init__(self):
        name = str(self.name)
        declaration = str(self.declaration)
        role = str(self.role)
        if not name:
            raise ValueError("kernel argument requires a name")
        if not declaration:
            raise ValueError("kernel argument requires a declaration")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "declaration", declaration)
        object.__setattr__(self, "role", role)

    def to_dict(self):
        return {
            "name": self.name,
            "declaration": self.declaration,
            "role": self.role,
        }


@dataclass(frozen=True)
class LocalKernelSignature:
    name: str
    form_order: FormOrder
    template_parameters: tuple
    arguments: tuple
    reuse_key: tuple = ()
    reusable: bool = True

    def __post_init__(self):
        name = str(self.name)
        if not name:
            raise ValueError("local kernel signature requires a name")
        arguments = tuple(self.arguments)
        for argument in arguments:
            if not isinstance(argument, KernelArgument):
                raise TypeError("local kernel signature arguments must be KernelArgument objects")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "form_order", FormOrder(self.form_order))
        object.__setattr__(self, "template_parameters", tuple(self.template_parameters))
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "reuse_key", tuple(self.reuse_key))
        object.__setattr__(self, "reusable", bool(self.reusable))

    @property
    def template_line(self):
        return "template <%s>" % ", ".join(self.template_parameters)

    @property
    def argument_names(self):
        return tuple(argument.name for argument in self.arguments)

    def to_dict(self):
        return {
            "name": self.name,
            "form_order": self.form_order.value,
            "template_parameters": list(self.template_parameters),
            "arguments": [argument.to_dict() for argument in self.arguments],
            "reuse_key": list(self.reuse_key),
            "reusable": self.reusable,
        }


@dataclass(frozen=True)
class MeshKernelSignature:
    name: str
    template_parameters: tuple
    arguments: tuple

    def __post_init__(self):
        name = str(self.name)
        if not name:
            raise ValueError("mesh kernel signature requires a name")
        arguments = tuple(self.arguments)
        for argument in arguments:
            if not isinstance(argument, KernelArgument):
                raise TypeError("mesh kernel signature arguments must be KernelArgument objects")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "template_parameters", tuple(self.template_parameters))
        object.__setattr__(self, "arguments", arguments)

    @property
    def argument_names(self):
        return tuple(argument.name for argument in self.arguments)

    def to_dict(self):
        return {
            "name": self.name,
            "template_parameters": list(self.template_parameters),
            "arguments": [argument.to_dict() for argument in self.arguments],
        }


LOCAL_KERNEL_TEMPLATE_PARAMETERS = (
    "typename scalar_t",
    "int N_QP",
    "int N_SHAPE",
    "int VECTOR_SIZE",
)


MESH_KERNEL_TEMPLATE_PARAMETERS = ("typename scalar_t",)


def local_kernel_signatures_from_plan(unit, emission_plan, local_prefix, kind):
    kind = str(kind)
    local_prefix = str(local_prefix)
    expression_plans = tuple(
        expression_plan
        for expression_plan in unit.expression_plans
        if _expression_plan_has_local_kernel(kind, expression_plan)
    )
    signatures = []
    for expression_plan in expression_plans:
        arguments = _local_arguments(unit, emission_plan, kind, expression_plan)
        signatures.append(
            LocalKernelSignature(
                _local_signature_name(local_prefix, expression_plan),
                expression_plan.form_order,
                LOCAL_KERNEL_TEMPLATE_PARAMETERS,
                arguments,
                reuse_key=_local_reuse_key(unit, emission_plan, kind, local_prefix, expression_plan, arguments),
                reusable=bool(local_prefix),
            )
        )
    return tuple(signatures)


def local_kernel_suffix_from_plan(unit, context, kind):
    kind = str(kind)
    if kind == "mixed_residual_soa" and not _is_diagonal_two_form_block(unit):
        return "_mixed"
    return ""


def mesh_kernel_signature_from_plan(unit, emission_plan, operator_prefix, kind):
    return MeshKernelSignature(
        str(operator_prefix),
        MESH_KERNEL_TEMPLATE_PARAMETERS,
        _mesh_arguments(unit, emission_plan, str(kind)),
    )


def _expression_plan_has_local_kernel(kind, expression_plan):
    if kind == "energy_soa":
        return expression_plan.source is not None
    if kind in ("residual_soa", "mixed_residual_soa"):
        return expression_plan.form_order in (FormOrder.ONE, FormOrder.TWO)
    if kind == "boundary_residual_soa":
        return expression_plan.form_order is FormOrder.ONE
    return False


def _local_signature_name(local_prefix, expression_plan):
    if local_prefix:
        return "%s_%s" % (local_prefix, expression_plan.name)
    return expression_plan.name


def _local_reuse_key(unit, emission_plan, kind, local_prefix, expression_plan, arguments):
    return (
        _safe_name(unit.name),
        str(kind),
        "d%d" % int(unit.dim),
        emission_plan.basis_family,
        str(local_prefix),
        FormOrder(expression_plan.form_order).value,
        tuple(argument.declaration for argument in arguments),
    )


def _safe_name(value):
    return str(value)


def _is_diagonal_two_form_block(unit):
    return (
        unit.is_block
        and unit.block is not None
        and unit.block.form_order is FormOrder.TWO
        and unit.block.column_field
        and unit.block.row_field == unit.block.column_field
    )


def _mesh_arguments(unit, emission_plan, kind):
    dim = int(unit.dim)
    arguments = [
        KernelArgument("nelements", "const ptrdiff_t nelements", "control"),
        KernelArgument("nnodes", "const ptrdiff_t nnodes", "control"),
        KernelArgument("elements", "idx_t **const SFEM_RESTRICT elements", "connectivity"),
    ]
    arguments.extend(
        (
            KernelArgument(
                "adjugate",
                "const scalar_t *const SFEM_RESTRICT adjugate[%d]" % (dim * dim),
                "geometry",
            ),
            KernelArgument(
                "determinant",
                "const scalar_t *const SFEM_RESTRICT determinant",
                "geometry",
            ),
            KernelArgument(
                "coordinates",
                "const scalar_t *const SFEM_RESTRICT coordinates[%d]" % dim,
                "geometry",
            ),
        )
    )
    arguments.extend(_mesh_field_arguments(unit, kind))
    arguments.extend(_parameter_arguments(_merged_dependencies(unit.expression_plans)))
    arguments.extend(_mesh_output_arguments(unit, kind))
    return tuple(arguments)


def _mesh_field_arguments(unit, kind):
    dependencies = _merged_dependencies(unit.expression_plans)
    n_components = _field_component_count(unit)
    arguments = []
    if _dependencies_use_current(dependencies, default=False):
        arguments.append(
            KernelArgument(
                "current",
                "const scalar_t *const SFEM_RESTRICT current[%d]" % n_components,
                "field",
            )
        )
    if getattr(dependencies, "previous", False):
        arguments.append(
            KernelArgument(
                "previous",
                "const scalar_t *const SFEM_RESTRICT previous[%d]" % n_components,
                "previous",
            )
        )
    if _dependencies_use_direction(dependencies, default=False):
        arguments.append(
            KernelArgument(
                "direction",
                "const scalar_t *const SFEM_RESTRICT direction[%d]" % n_components,
                "direction",
            )
        )
    return tuple(arguments)


def _mesh_output_arguments(unit, kind):
    if kind == "energy_soa":
        return (KernelArgument("output", "scalar_t *const SFEM_RESTRICT output", "output"),)
    return (
        KernelArgument(
            "output",
            "scalar_t *const SFEM_RESTRICT output[%d]" % _field_component_count(unit),
            "output",
        ),
    )


def _local_arguments(unit, emission_plan, kind, expression_plan):
    dim = int(unit.dim)
    dependencies = expression_plan.dependencies
    arguments = [KernelArgument("nelems", "const ptrdiff_t nelems", "control")]
    if kind != "boundary_residual_soa":
        arguments.extend(_geometry_arguments(dim))
        arguments.extend(_reference_arguments(emission_plan, dim, dependencies))
    else:
        arguments.extend(_boundary_reference_arguments())
    arguments.extend(_field_arguments(unit, kind, expression_plan, dependencies))
    arguments.extend(_parameter_arguments(dependencies))
    arguments.extend(_output_arguments(unit, kind, expression_plan))
    return tuple(arguments)


def _geometry_arguments(dim):
    return (
        KernelArgument(
            "adjugate",
            "const scalar_t *const SFEM_RESTRICT adjugate[%d]" % (dim * dim),
            "geometry",
        ),
        KernelArgument(
            "determinant",
            "const scalar_t *const SFEM_RESTRICT determinant",
            "geometry",
        ),
    )


def _reference_arguments(emission_plan, dim, dependencies):
    if emission_plan.basis_family == "tensor_product":
        return (
            KernelArgument("shape_1d", "const scalar_t *const SFEM_RESTRICT shape_1d", "reference"),
            KernelArgument("grad_1d", "const scalar_t *const SFEM_RESTRICT grad_1d", "reference"),
            KernelArgument("q_weight_1d", "const scalar_t *const SFEM_RESTRICT q_weight_1d", "reference"),
        )
    arguments = [
        KernelArgument("shape", "const scalar_t *const SFEM_RESTRICT shape", "reference")
    ]
    if _uses_reference_gradients(dependencies):
        arguments.extend(
            KernelArgument(
                "grad_ref_%d" % d,
                "const scalar_t *const SFEM_RESTRICT grad_ref_%d" % d,
                "reference",
            )
            for d in range(dim)
        )
    arguments.append(
        KernelArgument("q_weight", "const scalar_t *const SFEM_RESTRICT q_weight", "reference")
    )
    return tuple(arguments)


def _boundary_reference_arguments():
    return (
        KernelArgument("shape", "const scalar_t *const SFEM_RESTRICT shape", "reference"),
        KernelArgument("q_weight", "const scalar_t *const SFEM_RESTRICT q_weight", "reference"),
    )


def _field_arguments(unit, kind, expression_plan, dependencies):
    if kind == "energy_soa":
        dim = int(unit.dim)
        arguments = []
        if _dependencies_use_current(dependencies, default=True):
            arguments.append(
                KernelArgument(
                    "u_streams",
                    "const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * %d]" % dim,
                    "field",
                )
            )
        if _dependencies_use_direction(dependencies, default=expression_plan.has_direction):
            arguments.append(
                KernelArgument(
                    "h_streams",
                    "const scalar_t *const SFEM_RESTRICT h_streams[N_SHAPE * %d]" % dim,
                    "direction",
                )
            )
        return tuple(arguments)

    n_streams = _local_field_stream_count(unit, kind)
    stream_extent = _field_stream_extent(unit, kind)
    arguments = []
    if getattr(dependencies, "current", False):
        arguments.append(
            KernelArgument(
                "current",
                "const scalar_t *const SFEM_RESTRICT current[%s]" % stream_extent,
                "field",
            )
        )
    if getattr(dependencies, "previous", False):
        arguments.append(
            KernelArgument(
                "previous",
                "const scalar_t *const SFEM_RESTRICT previous[%s]" % stream_extent,
                "previous",
            )
        )
    if getattr(dependencies, "direction", False):
        arguments.append(
            KernelArgument(
                "direction",
                "const scalar_t *const SFEM_RESTRICT direction[%s]" % stream_extent,
                "direction",
            )
        )
    return tuple(arguments)


def _parameter_arguments(dependencies):
    return tuple(
        KernelArgument(str(parameter), "const scalar_t %s" % parameter, "parameter")
        for parameter in getattr(dependencies, "parameters", ())
    )


def _merged_dependencies(expression_plans):
    dependencies = tuple(
        expression_plan.dependencies
        for expression_plan in expression_plans
        if expression_plan.dependencies is not None
    )
    parameters = []
    for dependency in dependencies:
        for parameter in getattr(dependency, "parameters", ()):
            if parameter not in parameters:
                parameters.append(parameter)

    return SimpleNamespace(
        current=any(getattr(dependency, "current", False) for dependency in dependencies),
        previous=any(getattr(dependency, "previous", False) for dependency in dependencies),
        direction=any(getattr(dependency, "direction", False) for dependency in dependencies),
        geometry=any(getattr(dependency, "geometry", False) for dependency in dependencies),
        parameters=tuple(parameters),
    )


def _output_arguments(unit, kind, expression_plan):
    if kind == "energy_soa" and expression_plan.form_order is FormOrder.ZERO:
        return (KernelArgument("value", "scalar_t *const SFEM_RESTRICT value", "output"),)
    n_streams = _local_field_stream_count(unit, kind)
    if kind == "energy_soa":
        dim = int(unit.dim)
        declaration = "scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * %d]" % dim
    else:
        declaration = "scalar_t *const SFEM_RESTRICT output[%s]" % _field_stream_extent(unit, kind)
    return (KernelArgument("output", declaration, "output"),)


def _local_field_stream_count(unit, kind):
    if unit.is_block and unit.block is not None:
        field_count = 1
    else:
        field_count = len(tuple(unit.form_collection.fields))
    if kind == "boundary_residual_soa":
        return field_count
    return max(1, field_count)


def _field_stream_extent(unit, kind):
    count = _local_field_stream_count(unit, kind)
    if kind == "boundary_residual_soa":
        return str(count)
    if count == 1:
        return "N_SHAPE"
    return "%d * N_SHAPE" % count


def _local_shape_symbol_factor(unit):
    return 1 if unit.is_block else 1


def _field_component_count(unit):
    fields = tuple(unit.form_collection.fields)
    if unit.is_block and unit.block is not None:
        fields = tuple(field for field in fields if field.name == unit.block.row_field)
    return max(1, sum(int(getattr(field, "components", 1)) for field in fields))


def _uses_reference_gradients(dependencies):
    if dependencies is None:
        return True
    if getattr(dependencies, "geometry", False):
        return True
    symbols = tuple(getattr(dependencies, "current_symbols", ())) + tuple(
        getattr(dependencies, "direction_symbols", ())
    )
    return bool(symbols)


def _dependencies_use_current(dependencies, default):
    if dependencies is None:
        return bool(default)
    return bool(getattr(dependencies, "current", False))


def _dependencies_use_direction(dependencies, default):
    if dependencies is None:
        return bool(default)
    return bool(getattr(dependencies, "direction", False))
