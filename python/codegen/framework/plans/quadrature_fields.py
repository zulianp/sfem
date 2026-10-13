"""Integration-point field streams: what a kernel reads per quadrature point.

An integration-point field is a tensor stored at the points of an element's
rule -- the frozen history of a Prony series is the first -- and is laid out in
memory as

    data[(component * NQ + q) * nelements + element]

so each (component, point) is one contiguous run over the elements, the layout
SFEM's `QuadratureState` holds.  A kernel reads the components of the point it
is at into a local array named like the symbols that stand for them
(`<field>_qp[k]`, see `symbolic.fields.quadrature_field_symbol`), which is why
the material expression needs no substitution: the symbol prints as the read.

This module owns the stream's spellings -- the pointer a block and a mesh
kernel take, the element stride the block is handed, the point's base and the
per-lane read -- so every kernel family emits one shape of read instead of
inventing its own.  Every helper takes a tuple of `QuadratureFieldStream` and
an empty tuple yields nothing, so an emitter iterates the plan and never asks
whether a form has one.
"""

from dataclasses import dataclass

from codegen.framework.plans.form_emission import FormAccumulation, form_accumulation
from codegen.framework.symbolic.fields import quadrature_symbol_parts

#: The element stride a block kernel is handed for every integration-point
#: stream: the streams share one, the mesh's element count.
QUADRATURE_STRIDE = "qp_stride"


@dataclass(frozen=True)
class QuadratureFieldStream:
    """One integration-point field a kernel reads or writes, and how many components."""

    name: str
    n_components: int
    #: Whether the kernel writes it -- a quadrature output -- rather than reads it.
    writes: bool = False

    @property
    def qualifier(self):
        return "" if self.writes else "const "

    @property
    def local(self):
        """The local array the point's components are read into."""
        return "%s_qp" % self.name

    @property
    def point_base(self):
        """The stream advanced to the current quadrature point."""
        return "%s_q" % self.name


def quadrature_field_streams(form):
    """The integration-point fields a form reads, then those it writes.

    The reads are asked of the form's dependency set, which classified the
    symbols, and sorted by name.  The component count is the highest component
    read plus one: a stream's components are addressed by index, so a kernel
    that reads a subset still reads them at their stored offsets.  The writes
    are a per-point form's declared outputs, in declaration order.
    """
    reads = quadrature_read_streams(getattr(form, "dependencies", None))
    return reads + _WRITTEN_STREAMS[form_accumulation(form)](form)


def quadrature_read_streams(dependencies):
    """The integration-point fields a dependency set reads, sorted by name."""
    symbols = getattr(dependencies, "quadrature_symbols", ()) if dependencies is not None else ()
    extents = {}
    for symbol in symbols:
        name, component = quadrature_symbol_parts(symbol)
        extents[name] = max(extents.get(name, 0), component + 1)
    return tuple(QuadratureFieldStream(name, extents[name]) for name in sorted(extents))


def _per_point_outputs(form):
    weak_form = getattr(form, "weak_form", None)
    outputs = getattr(weak_form, "quadrature_outputs", ()) if weak_form is not None else ()
    return tuple(
        QuadratureFieldStream(name, len(components), writes=True) for name, components in outputs
    )


_WRITTEN_STREAMS = {
    FormAccumulation.SCALAR: lambda form: (),
    FormAccumulation.PER_SHAPE: lambda form: (),
    FormAccumulation.PER_POINT: _per_point_outputs,
}


def read_streams(streams):
    return tuple(stream for stream in streams if not stream.writes)


def written_streams(streams):
    return tuple(stream for stream in streams if stream.writes)


def block_parameters(streams, scalar_type="s_t"):
    """A block kernel's parameters: one pointer per stream, then the stride."""
    return tuple(
        "%s%s *const RSTR %s" % (stream.qualifier, scalar_type, stream.name) for stream in streams
    ) + tuple("const ptrdiff_t %s" % QUADRATURE_STRIDE for _ in streams[:1])


def mesh_parameters(streams, scalar_type="s_t"):
    """A mesh kernel's parameters: one pointer per stream.

    The stride is the mesh kernel's own element count, so it is not a
    parameter.
    """
    return tuple(
        "%s%s *const RSTR %s" % (stream.qualifier, scalar_type, stream.name) for stream in streams
    )


def block_call_arguments(streams, element_offset, n_elements):
    """What a mesh kernel hands its block: each stream at the block's first element."""
    return tuple("%s + %s" % (stream.name, element_offset) for stream in streams) + tuple(
        n_elements for _ in streams[:1]
    )


def point_base_lines(streams, indent, point="q", scalar_type="s_t"):
    """Each stream advanced to quadrature point `point`, once per point."""
    return [
        "%s%s%s *const RSTR %s = %s + %s * %s;"
        % (indent, stream.qualifier, scalar_type, stream.point_base, stream.name, point, QUADRATURE_STRIDE)
        for stream in streams
    ]


def store_target(stream, component, lane, n_points="NQ"):
    """Where component `component` of the current point of a written stream goes."""
    return "%s[%s]" % (stream.point_base, _component_offset(component, n_points, lane))


def load_lines(streams, indent, lane, n_points="NQ", scalar_type="s_t"):
    """The point's components into the local array the expression names.

    Needs `point_base_lines` above it.  `n_points` is the rule's point count,
    which separates consecutive components of one point.  Only the streams a
    kernel reads are loaded.
    """
    streams = read_streams(streams)
    return [
        "%sconst %s %s[%d] = {%s};"
        % (
            indent,
            scalar_type,
            stream.local,
            stream.n_components,
            ", ".join(
                "%s[%s]" % (stream.point_base, _component_offset(k, n_points, lane))
                for k in range(stream.n_components)
            ),
        )
        for stream in streams
    ]


#: The lane of a kernel that holds a single element, whose streams are already
#: offset to it: there is no lane to add.
SINGLE_ELEMENT = None


def _component_offset(component, n_points, lane):
    terms = []
    if component:
        terms.append("%d * %s * %s" % (component, n_points, QUADRATURE_STRIDE))
    if lane is not SINGLE_ELEMENT:
        terms.append(str(lane))
    return " + ".join(terms) or "0"
