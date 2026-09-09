def cpp_scalar_literal(value, scalar_type="s_t"):
    value = float(value)
    if value == 0.0:
        return "%s(0)" % scalar_type
    return "%s(%.17g)" % (scalar_type, value)


def cpp_scalar_initializer_list(values, scalar_type="s_t"):
    return ", ".join(cpp_scalar_literal(value, scalar_type) for value in values)


def quadrature_reference_accessor(cell_rule, reference_name, scalar_type="s_t"):
    """The call that reads one reference table.

    Named by the rule the table was evaluated at, not by the kernel that reads
    it.  `sfem::codegen::navier_stokes_isoparametric_reference_data<s_t>::q_weight()`
    becomes `sfem::codegen::quad_tet_q11<s_t>::q_weight()` -- shorter, and it says
    what the table *is* rather than which kernel happened to want it.

    The old per-kernel name was also not an identity.  `navier_stokes_form_1_p_affine_reference_data`
    was defined twice in one program -- with a 6-point triangle rule in the 2-D
    translation unit and an 11-point tetrahedron rule in the 3-D one, because the
    mixed path names its struct from the bare material prefix with no element in
    it.  Two bodies, one mangled symbol, and the linker keeps whichever it sees
    first.  A name derived from the rule cannot collide that way: the two are
    `quad_tri_q6` and `quad_tet_q11`.
    """
    owner, accessor = _owner(cell_rule, reference_name)
    return "sfem::codegen::%s<%s>::%s()" % (owner, scalar_type, accessor)


#: Where a shared reference header lives, relative to the generated tree root.
#:
#: One directory rather than the tree root beside `kernel_math.hpp`: the set
#: grows with every element and every quadrature order a material asks for, and
#: the root is where a reader looks for the handful of primitives.
REFERENCE_DIRECTORY = "reference"


def reference_header_path(key):
    """The file `key` lives in -- the same path for every material that needs it.

    Deduplication is the file name doing its job: two materials on the same basis
    and rule emit identical bytes to this path, and `pipeline/driver.py
    _merge_files` refuses two *different* bodies for one path, so a key that is
    too coarse stops generation instead of silently choosing a winner.
    """
    return "%s/%s.hpp" % (REFERENCE_DIRECTORY, key)


def _reference_header_lines(key, struct_name, tables, includes=()):
    guard = "SFEM_CODEGEN_REFERENCE_%s_HPP" % str(key).upper()
    lines = ["#ifndef %s" % guard, "#define %s" % guard, ""]
    for include in includes:
        lines.append('#include "%s"' % include)
    if includes:
        lines.append("")
    lines.extend(["namespace sfem {", "namespace codegen {"])
    lines.extend(_reference_struct_lines(struct_name, tables))
    lines.extend(["", "}  // namespace codegen", "}  // namespace sfem", "", "#endif"])
    return lines


def _reference_struct_lines(struct_name, tables):
    lines = ["", "template <typename s_t>", "struct %s {" % struct_name]
    for table in tables:
        values = tuple(table.values)
        lines.extend(
            [
                "  static const s_t *%s() {" % table.name,
                "    static const s_t data[%d] = {%s};"
                % (len(values), cpp_scalar_initializer_list(values, "s_t")),
                "    return data;",
                "  }",
            ]
        )
    lines.append("};")
    return lines


def reference_basis_header_source(basis, rule_key):
    """The shape and gradient tables of one basis at one quadrature rule.

    The rule's weights are not here.  They belong to the rule alone, so they get
    their own header and this one includes it -- which is what lets a mixed
    element's two bases share one set of weights instead of carrying one each.
    """
    lines = _reference_header_lines(
        basis.key,
        basis.struct_name,
        basis.tables,
        includes=("%s.hpp" % rule_key,),
    )
    return "\n".join(lines) + "\n"


def reference_rule_header_source(dataset):
    """The quadrature weights, with exactly one owner per rule."""
    lines = _reference_header_lines(dataset.rule_key, dataset.rule_key, dataset.weight_tables)
    return "\n".join(lines) + "\n"



def _owner(cell_rule, reference_name):
    """`(struct, accessor)` for one table: which shared struct holds it, and what
    it is called there.

    Resolved from the rule the table was *evaluated at* -- the emission site has
    it in hand together with the values -- so the header a kernel reads is by
    construction the header its numbers came from.  Nothing here infers a rule
    from a geometry mode, which is where this would have gone wrong: the affine
    and isoparametric structs of one kernel are built from different rules, and
    for the coupled residual path they are built from the same one.
    """
    from codegen.framework.plans.reference_data import basis_key_for, rule_key_for

    name = str(reference_name)
    if name.startswith("q_weight"):
        return rule_key_for(cell_rule), name
    prefix, canonical = split_reference_accessor(name)
    element = prefix.upper() if prefix else cell_rule.element_type
    return "ref_%s" % basis_key_for(element, cell_rule), canonical


#: The canonical accessor names, longest first so `shape_1d` is not read as an
#: element-prefixed `shape` and `grad_ref_x` is not read as `grad_ref`.
_CANONICAL_ACCESSORS = (
    "shape_1d",
    "grad_1d",
    "grad_ref_x",
    "grad_ref_y",
    "grad_ref_z",
    "grad_ref",
    "shape",
)


def split_reference_accessor(name):
    """`(element_prefix, canonical_name)`.

    A mixed kernel's struct disambiguates its two bases by prefixing the
    accessor -- `tri6_shape` beside `tri3_shape`.  In a shared header each basis
    has a struct of its own, so the prefix is the struct and the accessor keeps
    its plain name.
    """
    name = str(name)
    for canonical in _CANONICAL_ACCESSORS:
        if name == canonical:
            return "", canonical
        if name.endswith("_%s" % canonical):
            return name[: -len(canonical) - 1], canonical
    raise ValueError("'%s' is not a reference accessor this table knows" % name)


def reference_include_lines(cell_rule, references):
    """The shared headers a kernel's reference struct forwards into."""
    owners = []
    for reference in references:
        owner, _ = _owner(cell_rule, reference.name)
        key = owner[4:] if owner.startswith("ref_") else owner
        if key not in owners:
            owners.append(key)
    return ['#include "%s"' % reference_header_path(key) for key in sorted(owners)]


def reference_header_files(cell_rule, references):
    """The shared headers holding this kernel's tables.

    Emitted by every kernel that reads them, byte for byte the same; the file
    name is the deduplication and `pipeline/driver.py _merge_files` -- which
    refuses two different bodies for one path -- is the check on it.
    """
    from codegen.framework.emitters.artifacts import GeneratedKernelFile
    from codegen.framework.plans.reference_data import rule_weight_accessor

    grouped = {}
    for reference in references:
        owner, accessor = _owner(cell_rule, reference.name)
        key = owner[4:] if owner.startswith("ref_") else owner
        grouped.setdefault((key, owner), []).append(
            SfemReferenceLike(accessor, reference.values)
        )
    files = []
    for (key, owner), tables in sorted(grouped.items()):
        includes = ()
        if owner.startswith("ref_"):
            rule_owner, _ = _owner(cell_rule, rule_weight_accessor(cell_rule))
            includes = ("%s.hpp" % rule_owner,)
        source = "\n".join(_reference_header_lines(key, owner, tables, includes)) + "\n"
        files.append(GeneratedKernelFile(reference_header_path(key), source))
    return tuple(files)


class SfemReferenceLike(object):
    """A table under its shared-header name."""

    __slots__ = ("name", "values")

    def __init__(self, name, values):
        self.name = name
        self.values = tuple(values)
