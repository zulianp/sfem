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

from codegen.framework.plans.generation import DataStreamRole


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
