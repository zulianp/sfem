def cpp_scalar_literal(value, scalar_type="s_t"):
    value = float(value)
    if value == 0.0:
        return "%s(0)" % scalar_type
    return "%s(%.17g)" % (scalar_type, value)


def cpp_scalar_initializer_list(values, scalar_type="s_t"):
    return ", ".join(cpp_scalar_literal(value, scalar_type) for value in values)


def quadrature_reference_struct_name(prefix, stage):
    return "%s_%s_reference_data" % (prefix, stage)


def quadrature_reference_accessor(prefix, stage, reference_name, scalar_type="s_t"):
    return "sfem::codegen::%s<%s>::%s()" % (
        quadrature_reference_struct_name(prefix, stage),
        scalar_type,
        reference_name,
    )


def quadrature_reference_struct_lines(prefix, stage, references):
    struct_name = quadrature_reference_struct_name(prefix, stage)
    lines = [
        "",
        "template <typename s_t>",
        "struct %s {" % struct_name,
    ]
    for reference in references:
        values = tuple(reference.values)
        lines.extend(
            [
                "  static const s_t *%s() {" % reference.name,
                "    static const s_t data[%d] = {%s};"
                % (
                    len(values),
                    cpp_scalar_initializer_list(values, "s_t"),
                ),
                "    return data;",
                "  }",
            ]
        )
    lines.append("};")
    return lines


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

