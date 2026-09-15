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


#: Which of a field's symbols carry its value in a given role, and which carry
#: its gradient.  The roles differ in spelling only, so the table keeps the
#: three answers together rather than spreading them through three branches.
_ROLE_FIELD_SYMBOLS = {
    "current": (
        lambda field: (field.value,),
        lambda field: tuple(field.gradient),
    ),
    "previous": (
        lambda field: () if field.previous_value is None else (field.previous_value,),
        lambda field: tuple(field.previous_gradient),
    ),
    "direction": (
        lambda field: (field.direction_value,),
        lambda field: tuple(field.direction_gradient),
    ),
}


@dataclass(frozen=True)
class FieldStreamUsage:
    """What one field contributes to one role, as opposed to the whole system."""

    uses_value: bool
    uses_gradient: bool
    #: Which physical directions of this field's gradient the form reads.
    #:
    #: `uses_gradient` is `any()` of this, and the gap between the two is what
    #: `emitters/residual_codegen.py` was paying for: `_physical_gradient_nodes`
    #: emitted one chain-rule declaration per direction whenever the gradient
    #: was read at all, so a form reading one component computed `dim` of them.
    #: Navier-Stokes is the clearest case -- its pressure form contracts the
    #: divergence, which needs `u0_grad_0`, `u1_grad_1` and `u2_grad_2` and none
    #: of the six off-diagonal components, and all six were built and dropped.
    #:
    #: The reference components are a separate question and stay whole: physical
    #: direction `d` is a combination of every reference direction `k`, so
    #: narrowing the physical loop does not narrow the staging above it.
    gradient_components: tuple = ()

    @property
    def is_read(self):
        return self.uses_value or self.uses_gradient


def field_stream_usage(dependencies, field, role):
    """Whether this field's value and gradient are read, in this role.

    `field_stream_groups` above answers the same question for the system as a
    whole: does *any* field's value appear in the form, does *any* field's
    gradient.  That is the right question for the kernel's signature and for
    which reference tables it needs, and the wrong one for what to stage per
    field.

    A block of a coupled system reads a subset of the fields it is handed.  The
    (p_w, p_w) block of `two_phase_flow` reads p_c's value -- the residual
    contains `p_c - p_w` -- and never its gradient; staging by the system-wide
    answer zero-filled three gradient accumulators, ran them over every trial
    function, mapped them to the physical element with a divide each, and
    dropped the result.  A transitive dead-store scan over the shipped tree
    found 1557 such assignments.

    The information to avoid it was already on the dependency set: it records
    the exact symbols the expressions use, per role, and a field knows which of
    its symbols are its value and which its gradient.  This intersects the two.
    """
    role = str(role)
    if role not in FIELD_STREAM_ROLES:
        raise ValueError(
            "field stream role must be one of %s; got '%s'"
            % (", ".join(FIELD_STREAM_ROLES), role)
        )
    used = set(getattr(dependencies, "%s_symbols" % role, ()))
    if not used:
        # Some construction paths build a dependency set from the role flags
        # alone, with no symbol detail -- `ResidualDependencies` allows
        # `current=True` with an empty `current_symbols`.  There is nothing to
        # refine per field then, so the group-wide answer stands.
        #
        # The fallback direction is the safe one, and the asymmetry is the
        # point: a field that is staged and not read costs arithmetic, while a
        # field that is read and not staged does not compile.
        group = field_stream_group(dependencies, role)
        if group is None:
            return FieldStreamUsage(uses_value=False, uses_gradient=False)
        gradient_symbols = _ROLE_FIELD_SYMBOLS[role][1]
        return FieldStreamUsage(
            uses_value=group.uses_value,
            uses_gradient=group.uses_gradient,
            # No symbol detail to narrow with, so every direction stands.  Same
            # asymmetry as the flags above: a direction built and not read costs
            # arithmetic, one read and not built does not compile.
            gradient_components=(
                tuple(range(len(gradient_symbols(field))))
                if group.uses_gradient
                else ()
            ),
        )
    value_symbols, gradient_symbols = _ROLE_FIELD_SYMBOLS[role]
    components = tuple(
        d for d, symbol in enumerate(gradient_symbols(field)) if symbol in used
    )
    return FieldStreamUsage(
        uses_value=any(symbol in used for symbol in value_symbols(field)),
        uses_gradient=bool(components),
        gradient_components=components,
    )


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


def field_stream_layout(stream_layout):
    """The layout a caller's stream-layout word means.

    ``"contiguous"`` is a lane-major tile the kernel indexes directly;
    anything else is an array of pointers, one per stream.  The word is the
    caller's and the layout is this layer's, and four sites in
    ``emitters/residual_codegen.py`` compared the word themselves to pick a
    spelling -- a helper-name suffix, two call arguments and a C parameter --
    which is four places for the word to change out from under.
    """
    return (
        DataStreamLayout.AOS
        if stream_layout == "contiguous"
        else DataStreamLayout.SOA
    )


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
    needs_reference_basis=True,
    reads_shape_values=True,
    output=None,
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

    Two of the arguments describe kernels that are not applies.
    ``needs_reference_basis`` is
    ``plans.evaluation_strategy.ElementEvaluationPlan.needs_reference_basis_data``:
    a kernel that evaluates in closed form has the basis gradients folded into
    its arithmetic as constants, so it is handed no shape table, no reference
    gradients and no quadrature weights -- and taking them would be three dead
    parameters at every call site.  ``reads_shape_values`` is the finer question
    for a kernel that does take reference data: the shape *values* are read only
    where a form contracts a test function's value or substitutes a trial
    function's, and a kernel that does neither would otherwise name a table it
    never touches.  ``output`` replaces the per-degree-of-freedom output streams
    with a single stream, which is what a kernel that fills an element matrix
    writes to.
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
        if not uses_gradient_metric and needs_reference_basis:
            if reads_shape_values:
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
        if needs_reference_basis:
            streams.append(
                DataStreamPlan(
                    name="q_weight",
                    role=DataStreamRole.REFERENCE,
                    layout=DataStreamLayout.SCALAR,
                )
            )

    contiguous = field_stream_layout(stream_layout) is DataStreamLayout.AOS
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
        output
        if output is not None
        else DataStreamPlan(
            name="output",
            role=DataStreamRole.OUTPUT,
            layout=DataStreamLayout.AOS if contiguous else DataStreamLayout.SOA,
            components=n_fields,
        )
    )
    return tuple(streams)


def element_matrix_stream_plan(name="element_matrix"):
    """The output of a kernel that fills an element matrix, not a vector.

    An apply writes one value per degree of freedom and takes a stream per
    degree of freedom to write it to.  An assembly writes every entry of a
    dense square, addressed by row and column, and there is no per-degree-of-
    freedom structure at the boundary to describe -- so it is one array.
    """
    return DataStreamPlan(
        name=name,
        role=DataStreamRole.OUTPUT,
        layout=DataStreamLayout.DENSE,
    )


#: How each field stream's per-field arrays are named at the mesh boundary,
#: and the stride that precedes them.  The suffix is part of the ABI.
MESH_FIELD_STREAMS = (
    ("current", ""),
    ("previous", "_old"),
    ("direction", "_direction"),
)


@dataclass(frozen=True)
class MeshFieldRole:
    """One field role that crosses the mesh boundary, with its ABI names.

    ``suffix`` distinguishes the role's per-field arrays and ``stride`` the
    element stride that precedes them; both appear in every generated signature
    and every gather, so they belong with the order rather than being
    re-spelled wherever a role is emitted.  ``index`` is the role's position in
    the fixed sequence, which is what the per-thread scratch buffers are keyed
    on.
    """

    name: str
    suffix: str
    index: int

    @property
    def stride(self):
        return "%s_stride" % self.name

    def field_pointer(self, field_name):
        """The per-element array this role reads for one field."""
        return "%s%s" % (field_name, self.suffix)


#: The roles that carry solution state.  A Jacobian action's direction is not
#: one of them: it is the vector the action is applied to rather than a state
#: the linearization is taken at, and the kernels that gather state separately
#: from direction want to say which they mean.
STATE_FIELD_ROLES = ("current", "previous")


#: The other mesh field naming, for a kernel handed one *vector* field.
#:
#: ``MESH_FIELD_STREAMS`` above names a role by suffixing the field --
#: ``u``, ``u_old``, ``u_direction`` -- which is right when the fields are
#: separate scalars.  A kernel handed one vector field's components as separate
#: streams names them the other way round: the role is a prefix and the
#: component is the suffix, so the displacement's three streams are ``ux``,
#: ``uy``, ``uz`` and the previous state's are ``zx``, ``zy``, ``zz``.
#:
#: Both are ABI and both are frozen -- the dispatch layer builds its calls out
#: of these names, and `package/op_wrappers.py` reads the emitted signature for
#: ``"ux"`` to decide whether a tangent takes the state at all.  What is not
#: acceptable is a third site deriving either, which is why the second one is
#: written here beside the first rather than in the emitter that needs it.
COMPONENT_FIELD_STREAMS = (
    ("current", "u"),
    ("previous", "z"),
    ("direction", "h"),
    ("output", "out"),
)


@dataclass(frozen=True)
class ComponentFieldRole:
    """One vector field's role at the boundary, with its ABI names."""

    name: str
    prefix: str

    @property
    def stride(self):
        return "%s_stride" % self.prefix

    def component_pointer(self, component):
        """The per-element array this role reads for one component."""
        return "%s%s" % (self.prefix, component)


def component_field_role(name):
    """One role by name, or ``None`` when the boundary has no such role."""
    for role_name, prefix in COMPONENT_FIELD_STREAMS:
        if role_name == name:
            return ComponentFieldRole(name=role_name, prefix=prefix)
    return None


def component_stream_names(role, components):
    """A role's boundary names: its stride, then one buffer per component."""
    return (role.stride,) + tuple(
        role.component_pointer(component) for component in components
    )


def live_field_roles(dependencies, roles=None):
    """The field roles this kernel reads, in the order the ABI lists them.

    Emission asks this as an unrolled loop: ``if dependencies.current: ...``
    followed by ``if dependencies.previous: ...``, the same pair written out
    wherever a buffer is declared, a gather is emitted, a scratch slot is taken
    or a stream argument is named.  It is one sequence, and iterating it is the
    same thing the conditionals spell -- with the difference that a role added
    here reaches every site instead of the ones somebody remembered.

    ``roles`` narrows the sequence to a subset, for the kernels that handle one
    group of roles apart from the rest; ``STATE_FIELD_ROLES`` is the one that
    comes up.  The order and the indices are those of the full sequence either
    way, because both are ABI.
    """
    return tuple(
        MeshFieldRole(name=name, suffix=suffix, index=index)
        for index, (name, suffix) in enumerate(MESH_FIELD_STREAMS)
        if getattr(dependencies, name, False)
        and (roles is None or name in roles)
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
