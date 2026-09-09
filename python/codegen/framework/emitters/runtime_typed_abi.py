"""One C entry point per kernel, carrying its scalar type at run time.

SFEM's C ABI names the scalar with a `smesh::PrimitiveType` and passes buffers
as `void *`, rather than publishing one symbol per precision.  The public
dispatch entry points already do that; everything below them did not.

Every generated kernel published a `double` entry point and a `_float` twin --
1597 thin `extern "C"` forwarders across the shipped tree, 1.18 MB, each one a
full parameter list restated twice over -- and each pair was then reached
through a `switch (resolved_real_type)` in the dispatch source, which also
carried a forward declaration of both.  Counting the prototypes and the inner
switches, that layer was about 2.1 MB, 18% of the tree, for a boundary with
exactly one caller.

Collapsing the pair moves the type test one level down, from the dispatch into
the entry point, and leaves the same single branch on the same call path.  What
disappears is the second signature, the second prototype and the switch.

The functions here are the spelling half: given a parameter list written in
terms of `s_t`, they say how the merged entry point declares it and how each
concrete instantiation casts back.  `package/op_wrappers.py` performs the same
transformation on the public boundary by diffing an already-emitted pair; both
now go through this module so the two boundaries cannot disagree about what a
runtime-typed parameter looks like.
"""

import re

#: The parameter that names the scalar, and the argument that forwards it.
#: First in the list, immediately after any `element_type`: one fixed slot, so
#: a call site never has to work out where the scalar-carrying group begins.
#:
#: An `int` width rather than `smesh::PrimitiveType`, because these entry
#: points live in the operator sources, which include no smesh header and
#: should not start to.  `smesh::PrimitiveType` numbers its scalars by their
#: width -- `SMESH_FLOAT32 = 4`, `SMESH_FLOAT64 = 8` -- so the dispatch source,
#: which does see smesh, forwards the enum unchanged and asserts the
#: correspondence rather than translating it.  The entry point's own case
#: labels are `sizeof` of the types it instantiates, so nothing here is a
#: number copied from somewhere else.
RUNTIME_TYPE_PARAMETER = "const int scalar_bytes"
RUNTIME_TYPE_ARGUMENT = "scalar_bytes"

#: The concrete scalars an entry point instantiates.  A table rather than a
#: pair of branches: adding a precision is a row here and nothing else.
RUNTIME_SCALAR_TYPES = ("double", "float")


def runtime_type_correspondence_lines():
    """Proof, where both are visible, that the widths and the enum agree."""
    return [
        "static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double),",
        '              "the generated kernels select their scalar by width");',
        "static_assert((int)smesh::SMESH_FLOAT32 == (int)sizeof(float),",
        '              "the generated kernels select their scalar by width");',
        "",
    ]

_SCALAR_SPELLINGS = r"\b(s_t|real_t|double|float)\b"


def carries_scalar(param):
    """Whether this parameter's type is the scalar being dispatched.

    `geom_t` is not on this axis -- SFEM keeps geometry at one precision -- and
    neither are the strides, the connectivity or the counts.
    """
    return bool(re.search(r"\b(s_t|real_t)\b", param)) or bool(
        re.search(r"\b(double|float)\b", param)
    )


def runtime_typed_parameter(param):
    """One parameter as the merged entry point declares it.

    A pointer to the dispatched scalar crosses as `void *`; a scalar passed by
    value crosses as `real_t` and is converted on the way in, which is what the
    public entry points already do with material parameters.
    """
    if not carries_scalar(param):
        return param
    if "*" in param:
        return re.sub(_SCALAR_SPELLINGS, "void", param, count=1)
    return re.sub(_SCALAR_SPELLINGS, "real_t", param, count=1)


def runtime_typed_parameters(params):
    """The whole list, with the type parameter first."""
    return (RUNTIME_TYPE_PARAMETER,) + tuple(
        runtime_typed_parameter(param) for param in params
    )


def parameter_name(param):
    """The name a parameter declares, ignoring any array extent."""
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*$", param.strip())
    if not match:
        raise ValueError("could not read a parameter name from %r" % param)
    return match.group(1)


def runtime_typed_argument(param, scalar_type):
    """One argument at a concrete instantiation, cast back from `void *`.

    Two shapes reach here.  A plain buffer is one pointer and casts to one.  A
    field handed as an array of pointers -- `const s_t *const u_data[3]` --
    decays to a pointer-to-pointer, so it casts to `const float *const *`.
    """
    name = parameter_name(param)
    if not carries_scalar(param) or "*" not in param:
        return name
    const = "const " if param.lstrip().startswith("const ") else ""
    if re.search(r"\[\s*\d+\s*\]\s*$", param):
        return "(%s%s *const *)%s" % (const, scalar_type, name)
    return "(%s%s *)%s" % (const, scalar_type, name)


def runtime_typed_arguments(params, scalar_type):
    """Every argument for one instantiation, in declaration order."""
    return tuple(runtime_typed_argument(param, scalar_type) for param in params)


def cast_arguments(params, argument_names, scalar_type):
    """Cast a call's arguments back from `void *`, matched to the parameters.

    The emitters build a call's argument list by name rather than from the
    signature, so the cast is applied by looking each name up rather than by
    position -- a positional match would silently mis-cast a signature whose
    argument order drifted from its parameter order.  A name with no matching
    parameter is passed through, which is what the counts and strides are.
    """
    by_name = {parameter_name(param): param for param in params}
    return tuple(
        runtime_typed_argument(by_name[name], scalar_type)
        if name in by_name
        else name
        for name in argument_names
    )


def runtime_typed_entry_point_lines(
    public_name, params, call, indent="  ", parameter_lines=None
):
    """One `extern "C"` entry point that selects its scalar at run time.

    `call(scalar_type, arguments)` returns the body lines that reach the
    template for that scalar -- the emitters differ in how they spell it, some
    forwarding to one implementation and some to a pair, so the call is the
    caller's to write and only the shape around it is here.

    `parameter_lines` formats the declaration, defaulting to one parameter per
    line at four spaces; a caller with its own indentation passes its own.
    """
    lines = ['extern "C" int %s(' % public_name]
    declared = runtime_typed_parameters(params)
    if parameter_lines is None:
        lines.extend(
            "    %s%s" % (param, "," if index + 1 < len(declared) else "")
            for index, param in enumerate(declared)
        )
    else:
        lines.extend(parameter_lines(declared))
    lines.append(") {")
    lines.append("%sswitch (%s) {" % (indent, RUNTIME_TYPE_ARGUMENT))
    for scalar_type in RUNTIME_SCALAR_TYPES:
        # Each arm gets its own scope.  A body that declares anything -- the
        # two-pass packed assembly declares a graph status before it fills --
        # otherwise redefines that name in the next arm, and C++ refuses to
        # jump past an initialisation into a later case.
        lines.append("%s  case (int)sizeof(%s): {" % (indent, scalar_type))
        for line in call(scalar_type, runtime_typed_arguments(params, scalar_type)):
            lines.append("%s    %s" % (indent, line))
        lines.append("%s  }" % indent)
    lines.extend(
        [
            "%s  default:" % indent,
            "%s    break;" % indent,
            "%s}" % indent,
            '%sreturn sfem::codegen::unsupported_dispatch("%s", -1, (int)%s);'
            % (indent, public_name, RUNTIME_TYPE_ARGUMENT),
            "}",
            "",
        ]
    )
    return lines
