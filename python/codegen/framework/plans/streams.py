"""Which data streams a kernel needs, decided from its dependencies.

A residual kernel reads its fields in up to three roles: the current iterate,
the previous time step, and the direction a Jacobian action is applied to.  Each
role is present only if the form actually depends on it, and each carries values,
gradients, or both -- again only as the form requires.  Deciding that is
planning: it follows from the lowered form's dependency set and from nothing
else.  It fixes how much data crosses the memory boundary per element, which is
the single largest lever on a matrix-free kernel's arithmetic intensity.

It was being decided inside ``emitters/residual_codegen.py``, interleaved with
the code that spells the resulting pointer names.  The two are separable: which
groups exist and what each carries is this module's answer; what they are
*called* in C is the emitter's.

``DataStreamPlan`` in ``plans.generation`` describes one concrete named stream.
``FieldStreamGroup`` here is the level above it -- a role and what it carries --
from which the emitter derives the individual streams.
"""

from dataclasses import dataclass

from codegen.framework.plans.generation import (
    DataStreamLayout,
    DataStreamPlan,
    DataStreamRole,
)


#: Roles in the order the generated kernels have always emitted them.  The order
#: is part of the contract: it determines argument order in every generated
#: signature, so it is fixed here rather than left to the emitter.
FIELD_STREAM_ROLES = ("current", "previous", "direction")

_ROLE_TO_STREAM_ROLE = {
    "current": DataStreamRole.FIELD,
    "previous": DataStreamRole.FIELD,
    "direction": DataStreamRole.DIRECTION,
}


@dataclass(frozen=True)
class FieldStreamGroup:
    """One role's worth of field data, and what the form needs from it."""

    name: str
    role: DataStreamRole
    uses_value: bool
    uses_gradient: bool

    def __post_init__(self):
        name = str(self.name)
        if name not in FIELD_STREAM_ROLES:
            raise ValueError(
                "field stream group name must be one of %s; got '%s'"
                % (", ".join(FIELD_STREAM_ROLES), name)
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "role", DataStreamRole(self.role))
        object.__setattr__(self, "uses_value", bool(self.uses_value))
        object.__setattr__(self, "uses_gradient", bool(self.uses_gradient))

    def to_dict(self):
        return {
            "name": self.name,
            "role": self.role.value,
            "uses_value": self.uses_value,
            "uses_gradient": self.uses_gradient,
        }


def field_stream_groups(dependencies):
    """The stream groups a kernel with these dependencies needs.

    ``dependencies`` is a ``ResidualDependencies``: it records, per role,
    whether the form touches it at all and whether it touches values, gradients
    or both.  A role absent from the dependencies produces no group, and so no
    argument, no gather, and no traffic.
    """
    groups = []
    for name in FIELD_STREAM_ROLES:
        if not getattr(dependencies, name, False):
            continue
        groups.append(
            FieldStreamGroup(
                name=name,
                role=_ROLE_TO_STREAM_ROLE[name],
                uses_value=bool(getattr(dependencies, "%s_value" % name, False)),
                uses_gradient=bool(getattr(dependencies, "%s_gradient" % name, False)),
            )
        )
    return tuple(groups)


def field_stream_group(dependencies, name):
    """One group by role name, or ``None`` if the form does not need it."""
    name = str(name)
    for group in field_stream_groups(dependencies):
        if group.name == name:
            return group
    return None


#: The three field streams a local kernel may be handed, in the order the
#: kernel signature has always listed them.
_FIELD_STREAM_ORDER = ("current", "previous", "direction")


def local_kernel_stream_plans(
    dependencies,
    *,
    dim,
    n_fields,
    tensor_product,
    uses_gradient_metric,
    metric_components,
    stream_layout="pointer",
    grad_ref_name=None,
):
    """Which streams cross a local kernel's boundary, and in what order.

    This is a planning decision, not a spelling one.  Whether the kernel is
    handed a metric or a determinant and adjugate, whether it needs
    one-dimensional or simplex reference basis data, which field streams are
    live and whether they arrive as pointers or contiguous tiles -- all of it
    follows from the dependencies and the geometry, and none of it depends on
    how C declares a parameter.

    It used to be seventy lines of conditionals inside ``_local_function``,
    which meant the signature of every generated kernel was decided in the
    emission layer.  The emitter now spells what this returns.

    The order is part of the contract: it is the kernel's ABI, and every
    caller the framework emits depends on it.
    """
    streams = []

    if uses_gradient_metric:
        streams.append(
            DataStreamPlan(
                name="geom_metric",
                role=DataStreamRole.GEOMETRY,
                layout=DataStreamLayout.SOA,
                n_items=int(metric_components),
            )
        )
    else:
        streams.append(
            DataStreamPlan(
                name="determinant",
                role=DataStreamRole.GEOMETRY,
                layout=DataStreamLayout.SCALAR,
            )
        )
        if dependencies.uses_adjugate:
            streams.append(
                DataStreamPlan(
                    name="adjugate",
                    role=DataStreamRole.GEOMETRY,
                    layout=DataStreamLayout.SOA,
                    n_items=dim * dim,
                )
            )

    if tensor_product:
        streams.append(
            DataStreamPlan(
                name="shape_1d",
                role=DataStreamRole.REFERENCE,
                layout=DataStreamLayout.TENSOR_PRODUCT_1D,
            )
        )
        if dependencies.uses_reference_gradients:
            streams.append(
                DataStreamPlan(
                    name="grad_1d",
                    role=DataStreamRole.REFERENCE,
                    layout=DataStreamLayout.TENSOR_PRODUCT_1D,
                )
            )
        streams.append(
            DataStreamPlan(
                name="q_weight_1d",
                role=DataStreamRole.REFERENCE,
                layout=DataStreamLayout.TENSOR_PRODUCT_1D,
            )
        )
    else:
        # A gradient-metric kernel contracts the basis into the metric before
        # it is called, so it is handed no reference basis at all.
        if not uses_gradient_metric:
            streams.append(
                DataStreamPlan(
                    name="shape",
                    role=DataStreamRole.REFERENCE,
                    layout=DataStreamLayout.SCALAR,
                )
            )
            if dependencies.uses_reference_gradients:
                streams.extend(
                    DataStreamPlan(
                        name=grad_ref_name(d),
                        role=DataStreamRole.REFERENCE,
                        layout=DataStreamLayout.SCALAR,
                    )
                    for d in range(dim)
                )
        streams.append(
            DataStreamPlan(
                name="q_weight",
                role=DataStreamRole.REFERENCE,
                layout=DataStreamLayout.SCALAR,
            )
        )

    contiguous = stream_layout == "contiguous"
    for name in _FIELD_STREAM_ORDER:
        if not getattr(dependencies, name):
            continue
        streams.append(
            DataStreamPlan(
                name=name,
                role=(
                    DataStreamRole.DIRECTION
                    if name == "direction"
                    else DataStreamRole.FIELD
                ),
                layout=DataStreamLayout.AOS if contiguous else DataStreamLayout.SOA,
                components=n_fields,
            )
        )

    streams.extend(
        DataStreamPlan(
            name=parameter,
            role=DataStreamRole.MATERIAL_PARAMETER,
            layout=DataStreamLayout.SCALAR,
        )
        for parameter in dependencies.parameters
    )

    streams.append(
        DataStreamPlan(
            name="output",
            role=DataStreamRole.OUTPUT,
            layout=DataStreamLayout.AOS if contiguous else DataStreamLayout.SOA,
            components=n_fields,
        )
    )
    return tuple(streams)


#: How each field stream's per-field arrays are named at the mesh boundary,
#: and the stride that precedes them.  The suffix is part of the ABI.
MESH_FIELD_STREAMS = (
    ("current", ""),
    ("previous", "_old"),
    ("direction", "_direction"),
)


def mesh_kernel_stream_plans(dependencies, fields, include_output=True):
    """Which field streams cross a mesh kernel's boundary, and in what order.

    Every mesh-level kernel is handed a stride and one pointer per field for
    each live stream, then the same for its output.  Which streams are live
    follows from the lowered form; the order is the kernel's ABI.

    Eight signatures and nine call sites derive this independently, each with
    its own ``if dependencies.current`` chain.  They agree, but nothing makes
    them: a signature and its call could disagree, and seventeen copies is
    seventeen places to miss when a stream is added.

    ``include_output`` is false for matrix assembly, which writes into a
    sparse structure rather than a field output and takes rowptr/colidx/values
    instead.  The emitter adds those, because which matrix format is in play
    is not this plan's decision.
    """
    streams = []
    for name, suffix in MESH_FIELD_STREAMS:
        if not getattr(dependencies, name, False):
            continue
        streams.append(
            DataStreamPlan(
                name="%s_stride" % name,
                role=DataStreamRole.TEMPORARY,
                layout=DataStreamLayout.SCALAR,
                source="stride",
            )
        )
        streams.extend(
            DataStreamPlan(
                name="%s%s" % (field.name, suffix),
                role=(
                    DataStreamRole.DIRECTION
                    if name == "direction"
                    else DataStreamRole.FIELD
                ),
                layout=DataStreamLayout.SOA,
                source=name,
            )
            for field in fields
        )
    if not include_output:
        return tuple(streams)
    streams.append(
        DataStreamPlan(
            name="out_stride",
            role=DataStreamRole.TEMPORARY,
            layout=DataStreamLayout.SCALAR,
            source="stride",
        )
    )
    streams.extend(
        DataStreamPlan(
            name="%s_out" % field.name,
            role=DataStreamRole.OUTPUT,
            layout=DataStreamLayout.SOA,
            source="output",
        )
        for field in fields
    )
    return tuple(streams)
