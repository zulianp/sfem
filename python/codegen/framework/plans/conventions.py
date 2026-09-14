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

import re

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

#: How each axis of a published name is spelled, long form to short.
#:
#: The long form stays the value of the enums in `plans/apply_variants.py`,
#: because that is what plan dumps, diagnostics metadata and the `--geometry`
#: flag carry, and what a person reads when asking what a kernel is.  Only the
#: *name* is abbreviated, and only here: `abi_rename_pairs` derives the rename
#: from these tables, and `dimension_markers` derives the parser from the same
#: ones, so the two halves of the grammar cannot be moved separately.  That is
#: the property the previous rename attempt lacked.
#:
#: `a` and `i` are single letters, which is the material author's namespace
#: everywhere else in this file.  They are safe here because a geometry token
#: never stands alone -- it is always `_<geom>_<layout>`, and the composite is
#: the unit both the composer and the parser work in.
ABI_GEOMETRY_SPELLING = (
    ("affine", "a"),
    ("isoparametric", "i"),
    ("sideset", "ss"),
)

#: An optional qualifier between the geometry and the layout, naming the
#: geometry *representation* the kernel takes rather than the geometry itself.
ABI_QUALIFIER_SPELLING = (
    ("", ""),
    ("metric_", "met_"),
)

#: The mesh layout a kernel reads, longest first so a tail is never eaten by the
#: shorter tail it contains.
ABI_LAYOUT_SPELLING = (
    ("mesh_soa_aos_unit", "msoa_aos_unit"),
    ("mesh_soa", "msoa"),
    ("mesh_aos", "maos"),
    ("soa", "soa"),
)

#: The element-local kernels, which have no geometry token because they are
#: handed geometry rather than reading a mesh.
#:
#: The rule is the same one the mesh side follows: the level letter prefixes the
#: layout token -- `m` + `soa` is `msoa`, `e` + `soa` is `esoa` -- and a
#: qualifier sits between them, so `element_coords_soa` becomes `ecoords_soa`.
ABI_LEVEL_LETTER = "e"

#: What may qualify an element-local level.  `coords` marks the kernel that takes
#: coordinates directly rather than through a stream of them.
ABI_LOCAL_QUALIFIERS = ("", "coords_", "geometry_")

ABI_LOCAL_SPELLING = tuple(
    ("element_%ssoa" % qualifier, "%s%ssoa" % (ABI_LEVEL_LETTER, qualifier))
    for qualifier in ABI_LOCAL_QUALIFIERS
)


def abi_local_level(qualifier=""):
    """The element-local level token, e.g. `esoa` or `ecoords_soa`.

    Call this rather than spelling `_element_%ssoa`.  That literal splits the
    level around its qualifier, so `element_soa` never appears contiguously in
    the source and no text rewrite can reach it -- which is how eight element
    API headers kept the long form through a rename that moved everything else.
    """
    return "%s%ssoa" % (ABI_LEVEL_LETTER, qualifier)

ABI_GEOMETRY_TOKENS = tuple(short for _, short in ABI_GEOMETRY_SPELLING)
ABI_GEOMETRY_QUALIFIERS = tuple(short for _, short in ABI_QUALIFIER_SPELLING)
ABI_LAYOUT_TAILS = tuple(short for _, short in ABI_LAYOUT_SPELLING)


def abi_mesh_fragment(geometry, layout="mesh_soa", qualifier=""):
    """The `<geom>[_<qual>]_<layout>` fragment of a published mesh kernel name.

    Call this wherever the geometry arrives as a value rather than as text --
    `emitters/energy_codegen.py` splices it from `geometry_mode` -- so those
    sites move with the table instead of quietly emitting the long form.
    """
    spelling = dict(ABI_GEOMETRY_SPELLING)
    return "%s_%s%s" % (
        spelling.get(str(geometry), str(geometry)),
        dict(ABI_QUALIFIER_SPELLING)[qualifier],
        dict(ABI_LAYOUT_SPELLING)[layout],
    )


def abi_with_geometry_qualifier(name, qualifier="metric_"):
    """`name` with the geometry-representation qualifier spliced before its layout.

    Two elements of one dimension can want different geometry -- an affine
    simplex contracts through the symmetric metric where a hexahedron needs the
    full adjugate -- and that is two entry points, not one.  Naming the geometry
    keeps both; before it did, the second collided with the first and was dropped
    silently, leaving the metric elements with no affine entry point and a
    runtime `default:` as the only sign.

    The splice point is the layout tail, taken from the table, because it used to
    be the literal `_mesh_` and a rename left it matching nothing -- at which
    point the qualifier was appended to the end instead and named a symbol that
    does not exist.
    """
    short = dict(ABI_QUALIFIER_SPELLING)[qualifier]
    for tail in ABI_LAYOUT_TAILS:
        index = name.rfind("_%s" % tail)
        if index >= 0:
            return "%s_%s%s" % (name[:index], short, name[index + 1 :])
    return "%s_%s" % (name, short.rstrip("_"))


def abi_rename_pairs():
    """`(long, short)` for every composite a published name can end with.

    The composite is the unit, never the bare token: `affine` also names a
    geometry *mode* in plan dumps, a `--geometry` value and a JSON field, and
    rewriting it there is how a previous attempt broke `matrix_packed_passes`.
    Anchoring each rewrite to `_<geom>[_<qual>]_<layout>` makes the name-shaped
    occurrences the only ones that match.
    """
    pairs = []
    for geometry, short_geometry in ABI_GEOMETRY_SPELLING:
        for qualifier, short_qualifier in ABI_QUALIFIER_SPELLING:
            for layout, short_layout in ABI_LAYOUT_SPELLING:
                long_form = "_%s_%s%s" % (geometry, qualifier, layout)
                short_form = "_%s_%s%s" % (short_geometry, short_qualifier, short_layout)
                if long_form != short_form:
                    pairs.append((long_form, short_form))
    for local, short_local in ABI_LOCAL_SPELLING:
        if local == short_local:
            continue
        # The dimension-generic accessor used to be spelled by splitting the
        # level around the dimension -- `residual_element_3d_soa_diagnostics`.
        # With the level a single token the dimension goes in front of the whole
        # tail, `residual_3d_esoa_diagnostics`, which is what the mesh dispatch
        # has always done.  Listed here so the shape change is part of the rename
        # rather than an unexplained difference.
        head, _, layout = local.partition("_")
        for dim in (2, 3):
            pairs.append(
                ("_%s_%dd_%s_diagnostics" % (head, dim, layout),
                 "_%dd_%s_diagnostics" % (dim, short_local)),
            )
        pairs.append(("_%s" % local, "_%s" % short_local))
    return tuple(sorted(pairs, key=lambda pair: len(pair[0]), reverse=True))

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
) + tuple(("_%s" % short, "local") for _, short in ABI_LOCAL_SPELLING)


#: The tails a diagnostics accessor can end with, longest first.
#:
#: There are two because there are two levels.  A material whose kernels read a
#: mesh publishes `<mat>_<elem>_<verb>_soa_diagnostics`; one whose kernels are
#: handed elements publishes `<mat>_<elem>_<verb>_esoa_diagnostics`.  Pinning
#: only the first is how the `_element_soa` rename first went wrong: the
#: `endswith` pre-filter in `_diagnostic_dispatch_groups` stopped matching, every
#: navier_stokes accessor was skipped before it could reach the guard, no
#: dimension-generic accessor was emitted, and the wrapper's one-argument call
#: bound to the zero-argument per-element symbol.  The arity check caught it;
#: deriving these from the layout table is what stops it happening again.
#:
#: `ABI_NON_MESH_TAILS` carries the broader `_diagnostics`, which also covers the
#: per-block accessors two_phase_flow publishes; these are the narrower spellings
#: the dimension-generic dispatch is built for.
ABI_DIAGNOSTICS_TAILS = tuple(
    sorted(
        {"_%s_diagnostics" % tail for tail in ABI_LAYOUT_TAILS}
        | {"_%s_diagnostics" % short for _, short in ABI_LOCAL_SPELLING},
        key=len,
        reverse=True,
    )
)


def diagnostics_tail(name):
    """The diagnostics tail `name` ends with, or `None`."""
    for tail in ABI_DIAGNOSTICS_TAILS:
        if name.endswith(tail):
            return tail
    return None

#: The traversal a kernel name announces before its geometry token.  Unabbreviated
#: for now: it is 0.08% of the tree and `packed` is the word every driver flag,
#: plan field and benchmark column already uses.
#: Longest first, because `packed` is a prefix of the others.
#:
#: The three-word forms come from two different plans and mean two different
#: things.  `packed_two_pass` is a matrix-free traversal (`plans/apply_variants.py`
#: `MeshTraversal`); `packed_one_pass` and `packed_two_pass` are also matrix
#: assembly passes (`plans/matrix_formats.py` `PackedAssemblyPass`), which is why
#: the middle spelling appears once and covers both.  They are listed together
#: because this table answers one question -- what may occupy the slot between
#: the verb and the geometry -- and both do.
#:
#: `packed_one_pass` is not emitted by any shipped material, only by
#: `tests/test_m11_matrix_formats.py`.  It was missing from the first version of
#: this table for exactly that reason, and the omission was caught by that test
#: rather than by the tree, which is the argument for deriving a vocabulary from
#: the plans rather than from a sample of the output.
ABI_TRAVERSAL_SPELLING = (
    ("packed_one_pass", "packed_one_pass"),
    ("packed_two_pass", "packed_two_pass"),
    ("packed", "packed"),
)

#: Which dispatch translation unit each traversal belongs in.
#:
#: Both pack, so both go in `*_packed_<geometry>_dispatch.cpp`.  This is a
#: separate mapping from the spelling because a two-pass kernel must not be
#: routed to a unit of its own, and because the file name keeps the long word --
#: file names are outside the abbreviation.
ABI_TRAVERSAL_UNIT = {
    "packed_one_pass": "packed",
    "packed_two_pass": "packed",
    "packed": "packed",
}

#: The verb slot: what kernel a name is, not what mathematics it computes.
#:
#: Longest first, because `objective_steps` contains `objective` and
#: `hessian_bsr` contains `hessian`.  `residual` and `jacobian_action` are the
#: residual path's words for `gradient` and `apply`; `CONVENTIONS.md` records
#: that they are not canonical, and they are listed here because they are what
#: the tree publishes today.
ABI_VERBS = (
    "hessian_block_diag_sym",
    "objective_steps",
    "jacobian_action",
    "matrix_assembly",
    "inexact_apply",
    "hessian_bsr",
    "hessian_crs",
    "objective",
    "gradient",
    "residual",
    "hessian",
    "energy",
    "apply",
)

#: The store an inexact apply reads, which occupies the same slot a traversal
#: does.  See `INEXACT.md`.
ABI_INEXACT_MODES = ("tangent", "stored", "compressed")


def unit_output_name(material_name, unit_name):
    """What a material's unit publishes its kernels under.

    A material with one unit publishes under its own name; a material split into
    units qualifies each with the unit, because two materials can both have an
    `elastic` unit and the ABI has one namespace.

    This runs once, in `pipeline/driver.py`, as a unit is turned into an
    emission kernel.  Everything downstream reads the composed name off that
    kernel rather than composing again -- `_unit_name` in
    `emitters/inexact_apply_codegen.py` looks like a second spelling of this and
    is not one, because what it is handed is the result, not the parts.
    Composing twice yields `neohookean_ogden_neohookean_ogden_tet4`.
    """
    if unit_name:
        return "%s_%s" % (material_name, unit_name)
    return str(material_name)


#: The verb phrase the inexact-apply family publishes under.  Two words, because
#: the kernel is an apply and `inexact` says which apply it is.
ABI_INEXACT_VERB_PHRASE = "inexact_apply"


def inexact_apply_name(prefix, mode, traversal="", geometry="affine"):
    """The published name of one inexact-apply kernel.

    Thirteen sites in `emitters/inexact_apply_codegen.py` built this by
    formatting `"%s_inexact_apply_%s_a_msoa"`, spelling three separate pieces of
    ABI at each of them -- the verb phrase, the mode, and the mesh-SoA tail --
    while `abi_qualifier` below parses that same name back out of the shipped
    tree.  A name written in thirteen places and read in one is a name whose two
    halves can disagree, and the reader is the half that raises.

    `mode` is one of `ABI_INEXACT_MODES` and `traversal` is the optional second
    occupant of the qualifier slot, the pair `abi_traversal` splits apart again.
    Rejecting an unknown mode here is the point: the emitter used to interpolate
    whatever it was handed, so a typo became a kernel nothing could route to.
    """
    if mode not in ABI_INEXACT_MODES:
        raise ValueError(
            "%r is not an inexact-apply mode; the family publishes %s"
            % (mode, ", ".join(ABI_INEXACT_MODES))
        )
    qualifier = "%s_%s" % (mode, traversal) if traversal else mode
    return "%s_%s_%s_%s" % (
        prefix,
        ABI_INEXACT_VERB_PHRASE,
        qualifier,
        abi_mesh_fragment(geometry),
    )


def abi_qualifier(name):
    """What sits between the verb and the geometry token, or `""` for nothing.

    This slot is why `_dispatch_source_kind` used to sniff the name for
    `_packed_`.  A sniff cannot tell "this kernel is not packed" from "the token
    for packed has moved and I no longer recognise it", and the second silently
    merged two dispatch translation units into one.  Parsing the slot instead
    makes the two distinguishable: an occupant this table does not know is an
    error, and an empty slot is a plain kernel.

    The parse is total over the tree -- every published mesh-kernel name has a
    verb from `ABI_VERBS`, and the slot after it holds one of six things.
    """
    classified = classify_abi_name(name)
    if classified is None or classified[0] != "mesh":
        raise ValueError("%s is not a published mesh kernel" % name)
    head = name[: name.find(classified[1])]
    head = re.sub(r"_\d+d$", "", head)  # the dimension a dispatch name inserts
    for verb in ABI_VERBS:
        index = head.rfind("_%s" % verb)
        if index < 0:
            continue
        slot = head[index + len(verb) + 1 :]
        if not slot:
            return ""
        traversals = tuple(short for _, short in ABI_TRAVERSAL_SPELLING)
        # The slot may hold a store, a traversal, or -- since the inexact apply
        # grew a packed variant -- a store followed by a traversal.  Two
        # occupants, because they answer different questions about the same
        # kernel: which store it reads, and how it walks the mesh.
        known = tuple(
            "%s_%s" % (mode, traversal)
            for mode in ABI_INEXACT_MODES
            for traversal in traversals
        ) + traversals + ABI_INEXACT_MODES
        for candidate in known:
            if slot == "_%s" % candidate:
                return candidate
        raise ValueError(
            "%s carries an unknown qualifier %r between its verb and its "
            "geometry; the naming table and the emitters disagree" % (name, slot)
        )
    raise ValueError("no verb from the table appears in %s" % name)


def abi_traversal(qualifier):
    """The traversal half of a qualifier slot, or `""` when it holds none.

    `stored_packed_two_pass` is one slot with two occupants; everything that
    routes by traversal wants only the second.  Splitting it here rather than at
    each consumer is what keeps the pair in one place.
    """
    for mode in ABI_INEXACT_MODES:
        if qualifier == mode:
            return ""
        if qualifier.startswith("%s_" % mode):
            return qualifier[len(mode) + 1 :]
    return qualifier


def abi_geometry(name):
    """The geometry a published mesh-kernel name is specialised for."""
    classified = classify_abi_name(name)
    if classified is None or classified[0] != "mesh":
        raise ValueError("%s is not a published mesh kernel" % name)
    marker = classified[1]
    for geometry, short in ABI_GEOMETRY_SPELLING:
        if marker.startswith("_%s_" % short):
            return geometry
    raise ValueError("no geometry token in %s" % name)


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
