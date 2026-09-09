"""Every name the generator emits, in one editable table.

`CONVENTIONS.md` is the prose; this is the machine-readable form of it, and it is
the only place a spelling should be changed.  Edit a value here and the emitters
follow; edit a literal in an emitter and the two halves of a composed name drift
apart.

**Why this module exists.**  The names the generator emits fall into two classes
that look identical in the emitted text and behave nothing alike:

    literal     spelled once, in one emitter, e.g. `idet`, `goff`
    composed    built at run time from a prefix and a stream name, e.g.
                `"block_%s" % stream` -> `block_adj0`, `"pack_%s" % field`

A composed name has *two* owners -- the prefix and the stream -- and some use
sites in the emitters spell the composed result as a literal anyway.  Renaming
either half alone makes the declaration and the use disagree.  That is not a
theoretical hazard: it is what happened when the temporaries were first
shortened, four separate times, each caught by the C++ compiler on a different
family (`badj0` used but `block_adj0` declared, `pack_u` used but `pk_cur`
declared, and twice more).  Nothing was shipped, but nothing was learned by
grepping either -- the provenance simply is not visible in the emitter text.

So: a name lives here, composition happens here, and the collision check below
runs as a test.  `RESERVED` is checked automatically against every material's
declared fields and parameters, which is the check that would have caught `S`
being both the kernel scalar type and the per-dimension shape count.
"""

from dataclasses import dataclass


# --------------------------------------------------------------------------
# Types and compile-time constants
# --------------------------------------------------------------------------

#: The generated template parameters.  `geom_t`, `idx_t`, `count_t` and
#: `ptrdiff_t` are SFEM's or POSIX's and deliberately absent: they are not ours
#: to rename, and `package/op_wrappers.py` hands `geom_t` across the C ABI.
TYPES = {
    "scalar": "s_t",
    "geometry": "g_t",
}

#: Compile-time constants.  Never a bare single capital -- that namespace
#: belongs to the material author, who writes `T`, `R`, `Z` for temperature,
#: a gas constant and a compressibility factor, and `S` for the second
#: Piola-Kirchhoff stress.
CONSTANTS = {
    "vector_width": "VS",
    "quadrature_points": "NQ",
    "quadrature_points_1d": "NQ1",
    "shape_functions": "NS",
    "shape_functions_1d": "NS1",
    "field_shape_functions": "U_NS",
    "field_components": "NC",
    "spatial_dimension": "ND",
}


# --------------------------------------------------------------------------
# Composed names: a prefix plus a stream name
# --------------------------------------------------------------------------

#: Prefixes that compose with a stream name.  These are the halves that must
#: move together with the stream names below; `compose` is the only correct way
#: to spell the result.
PREFIXES = {
    "block": "b",        # staged into a VS-wide block on the stack
    "pack":  "pk_",      # per-thread scratch for the packed mesh traversal
}

#: Stream names.  A geometry stream's local spelling is owned by
#: `fem/reference.py`; its `g_`-prefixed C ABI spelling is frozen and lives in
#: `plans/geometry_quantities.py`.  Listing them here records the pairing so a
#: rename cannot move one without the other.
STREAMS = {
    "adjugate": "adj",
    "determinant": "det",
    "reference_gradient": "grad_ref",
}


#: Qualifiers the generator emits into every signature.
#:
#: `RSTR` is the generator's spelling of SFEM's `SFEM_RESTRICT`.  The macro is
#: not ours -- `base/sfem_base.hpp` picks `__restrict__` or `__restrict` per
#: compiler -- but its 13 characters appear 48,000 times, which is 4.3% of the
#: generated tree, and the generated headers already emit a fallback definition
#: of their own.  So the generator aliases a name it does not own, once, in the
#: shared prelude, and uses the short form everywhere below.
QUALIFIERS = {
    "restrict": "RSTR",
    "restrict_source": "SFEM_RESTRICT",
}


def restrict_prelude(definition="__restrict__", indent=""):
    """The lines that make the short qualifier available in a generated file.

    Emitted once per file, and the only place either spelling appears in
    generated output.  It defers to SFEM's macro when the real header is
    present, so the generator never second-guesses which of `__restrict__` and
    `__restrict` this compiler wants, and falls back only when compiled
    standalone.
    """
    short = QUALIFIERS["restrict"]
    source = QUALIFIERS["restrict_source"]
    define = ("%s#define %s %s" % (indent, source, definition)).rstrip()
    return [
        "%s#ifndef %s" % (indent, source),
        define,
        "%s#endif" % indent,
        "%s#ifndef %s" % (indent, short),
        "%s#define %s %s" % (indent, short, source),
        "%s#endif" % indent,
    ]


#: The C ABI spelling of each geometry stream, as `g_` plus this.
#:
#: The local name and the ABI name used to be the same string with a `g_` in
#: front, which meant the local one could not be shortened without moving a
#: parameter name `tools/reproducibility.py` matches by regex.  They are two
#: names for one quantity with two different owners, and this is where they are
#: told apart -- which is also what made it safe to shorten both.
#:
#: These are *not* seeded: the binder maps them structurally to mesh data, so a
#: rename changes no input and moves no digest.  It does require the binder's
#: patterns to move with them, and they are derived from this table for exactly
#: that reason.
ABI_GEOMETRY = (
    ("adj", "adj"),
    ("det", "det"),
    ("geom_metric", "met"),
)


def abi_geometry_name(local):
    """The frozen C ABI parameter name for a local geometry stream name.

    `adj0` -> `g_adj0`.  Call this wherever the ABI parameter is
    spelled; never `"g_%s" % stream`, which derives the frozen name from the one
    that is free to change.
    """
    local = str(local)
    for short, frozen in ABI_GEOMETRY:
        rest = local[len(short):]
        # The bare form (no index) is the AoS spelling of the same quantity, and
        # must map like the indexed ones or the binder builds a pattern that
        # matches nothing.
        if local.startswith(short) and (rest == "" or rest.isdigit()):
            return "g_%s%s" % (frozen, rest)
    return "g_%s" % local


#: How a published C ABI name ends.
#:
#: `package/op_wrappers.py` does not compose these names -- the emitters do, and
#: L7 sees them only as text in the sources it is handed.  So L7 *parses* them,
#: and that parser is a second, unwritten copy of the grammar.  When only one
#: copy moved during a rename the parser stopped recognising anything, every
#: lookup returned "not a dispatch kernel", the wrapper emitted its fallback
#: paths, and generation still reported success.
#:
#: The cure is not to delete the parser -- L7 has no other source for these
#: names -- but to make it *total*.  Both halves read the tables below, and
#: `classify_abi_name` puts every published name into a named category.  A name
#: that fits none is a hard error rather than a silent skip, so a half-finished
#: rename fails loudly at the first name it reaches.

#: The geometry a mesh kernel is specialised for.
ABI_GEOMETRY_TOKENS = ("affine", "isoparametric", "sideset")

#: An optional qualifier between the geometry and the layout, naming the
#: geometry *representation* the kernel takes rather than the geometry itself.
ABI_GEOMETRY_QUALIFIERS = ("", "metric_")

#: The mesh layout a kernel reads, longest first so a tail is never eaten by the
#: shorter tail it contains.
ABI_LAYOUT_TAILS = ("mesh_soa_aos_unit", "mesh_soa", "mesh_aos", "soa")

#: Published names that are deliberately not dimension-generic mesh kernels, and
#: what each one is instead.  Every entry here is a reason for `_dispatch_mapping`
#: to decline a name; anything not covered is a defect.
ABI_NON_MESH_TAILS = (
    ("_diagnostics", "diagnostics"),
    ("_arithmetic_intensity", "query"),
    ("_print_rate", "query"),
    ("_print_variant", "query"),
    ("_variant_count", "query"),
    ("_matrix_assembly_variant", "query"),
    ("_boundary_residual_soa", "boundary"),
    ("_element_soa", "local"),
)


#: The tail of a per-element diagnostics accessor, and of the dimension-generic
#: one that dispatches to it.  `ABI_NON_MESH_TAILS` carries the broader
#: `_diagnostics`, which also covers the per-block accessors two_phase_flow
#: publishes; this is the narrower spelling the dispatch is built for.
ABI_DIAGNOSTICS_TAIL = "_soa_diagnostics"

#: The traversal a kernel name announces before its geometry token.
ABI_TRAVERSAL_TOKENS = ("packed",)


def dimension_markers():
    """The fragments a dimension-generic dispatch name splices its `_<n>d` before.

    Longest first: `_affine_mesh_soa` is a prefix of `_affine_mesh_soa_aos_unit`,
    and matching the short one first would insert the dimension in the right
    place but classify the kernel as the wrong variant.
    """
    markers = [
        "_%s_%s%s" % (geometry, qualifier, tail)
        for geometry in ABI_GEOMETRY_TOKENS
        for qualifier in ABI_GEOMETRY_QUALIFIERS
        for tail in ABI_LAYOUT_TAILS
    ]
    return tuple(sorted(markers, key=len, reverse=True))


def classify_abi_name(name):
    """`(kind, marker)` for a published C ABI function name, or `None`.

    `kind` is `"mesh"` for a kernel that wants a dimension-generic dispatch --
    `marker` is then the fragment the dimension goes in front of -- and one of
    the `ABI_NON_MESH_TAILS` kinds otherwise, with `marker` `None`.  `None`
    means the name matches nothing the framework knows how to publish, which is
    a generator defect and never a reason to skip the name quietly.
    """
    stem = name[: -len("_float")] if name.endswith("_float") else name
    # Tails first.  `laplace_hex8_apply_sideset_soa_print_rate` reports on a mesh
    # kernel without being one, and it carries the marker that would otherwise
    # claim it.
    for tail, kind in ABI_NON_MESH_TAILS:
        if stem.endswith(tail):
            return (kind, None)
    best = None
    for marker in dimension_markers():
        index = stem.find(marker)
        if index < 0:
            continue
        if best is None or index < best[0]:
            best = (index, marker)
    if best is not None:
        return ("mesh", best[1])
    return None


def compose(prefix, stream):
    """The one correct spelling of a composed name.

    Call this rather than writing `"block_%s" % stream`: the point is that the
    prefix and the stream are both editable above, and every site that names the
    result agrees by construction instead of by coincidence.
    """
    if prefix not in PREFIXES:
        raise KeyError("no such name prefix: %r" % (prefix,))
    return PREFIXES[prefix] + str(stream)


# --------------------------------------------------------------------------
# Literals: spelled in one place, no composition
# --------------------------------------------------------------------------

#: Temporaries and buffers with no constructor anywhere.  Safe to rename here.
LITERALS = {
    "inverse_determinant": "idet",
    "geometry_offset": "goff",
    "affine_geometry_stream": "ageom_stream",
    "affine_geometry_block": "bageom_streams",
    "field_reference_gradients": "fgref",
    "element_block_base": "evb",
    "element_node": "ev",
    "pack_shape_index": "eshape",
}

#: Loop indices.  `lane` stays spelled out: it costs 145,748 bytes and is the
#: innermost index of every `#pragma omp simd` loop and the most-read token in
#: the arithmetic.  Recorded as a deliberate exception, with its number.
INDICES = {
    "lane": "lane",
    "quadrature": "q",
    "shape": "s",
    "element": "e",
    "pack_node": "k",
    # The element count a local kernel is given.  Local kernels are called
    # positionally, including from hand-written SFEM, so this name is the
    # generator's alone -- unlike the mesh-level `nelements`, which is a C ABI
    # parameter the reproducibility binder matches by name.
    "local_element_count": "ne",
}

#: Memory spaces, for the targets that have them.  `g_` is NOT available for
#: "global": it already means *geometry* at the frozen C ABI.
SPACES = {
    "global": "gl_",
    "shared": "sh_",
    "local": "lo_",
    "register": "rg_",
}


# --------------------------------------------------------------------------
# The reserved namespace, and the collision check
# --------------------------------------------------------------------------

#: Everything the generator claims.  A material may not declare a field or a
#: parameter with any of these names.
def reserved():
    """The full reserved set, derived from the tables above rather than typed twice."""
    names = set(TYPES.values())
    names |= set(CONSTANTS.values())
    names |= set(LITERALS.values())
    names |= set(INDICES.values())
    names |= set(QUALIFIERS.values())
    return frozenset(names)


#: Prefixes a material may not use.  A one-character prefix is deliberately not
#: reserved: `b` is the staged-buffer prefix in emitted kernels, but reserving it
#: would refuse a material parameter called `beta` or `b0`, which is far more of
#: the author's namespace than the generator has any claim to.  The buffers it
#: forms are protected by name instead, and they are all `b` + a stream name the
#: generator owns.
RESERVED_PREFIXES = frozenset(
    p
    for p in list(PREFIXES.values()) + list(SPACES.values()) + ["g_"]
    if len(p) > 1
)


class NameCollision(ValueError):
    """A material declared a name the generator has claimed."""


def check_material(name, declared):
    """Refuse a material whose fields or parameters collide with a reserved name.

    `declared` is every field and parameter name the material publishes.  This
    runs at generation, not in a kernel: a collision is a specification error and
    the material author is the one who can fix it, so it must fail with their
    name in the message rather than produce a kernel that shadows a type.
    """
    claimed = reserved()
    hits = sorted(n for n in declared if n in claimed)
    prefixed = sorted(
        n for n in declared if any(n.startswith(p) for p in RESERVED_PREFIXES)
    )
    if hits or prefixed:
        parts = []
        if hits:
            parts.append("names the generator reserves: %s" % ", ".join(hits))
        if prefixed:
            parts.append(
                "names using a reserved prefix (%s): %s"
                % (", ".join(sorted(RESERVED_PREFIXES)), ", ".join(prefixed))
            )
        raise NameCollision(
            "material %r declares %s.  See CONVENTIONS.md; rename the field or "
            "parameter, or take the name out of the reserved set if the "
            "generator no longer needs it." % (name, " and ".join(parts))
        )
    return True


def check_tables():
    """The tables must be injective: two concepts may not share a spelling.

    Without this the abbreviation table can quietly merge two things -- which is
    exactly how `field_grad_ref` was once folded onto `grad_ref`, the
    reference-basis parameter that already existed, merging the per-field array
    of reference gradients with the single reference-gradient table.
    """
    seen = {}
    for table, label in (
        (TYPES, "TYPES"),
        (CONSTANTS, "CONSTANTS"),
        (LITERALS, "LITERALS"),
        (INDICES, "INDICES"),
    ):
        for concept, spelling in table.items():
            if spelling in seen:
                other_label, other_concept = seen[spelling]
                raise NameCollision(
                    "%r is the spelling of both %s.%s and %s.%s; one concept per "
                    "name" % (spelling, other_label, other_concept, label, concept)
                )
            seen[spelling] = (label, concept)
    return True
