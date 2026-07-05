def cpp_scalar_literal(value, scalar_type="scalar_t"):
    value = float(value)
    if value == 0.0:
        return "%s(0)" % scalar_type
    return "%s(%.17g)" % (scalar_type, value)


def cpp_scalar_initializer_list(values, scalar_type="scalar_t"):
    return ", ".join(cpp_scalar_literal(value, scalar_type) for value in values)


def quadrature_reference_struct_name(prefix, stage):
    return "%s_%s_reference_data" % (prefix, stage)


def quadrature_reference_accessor(prefix, stage, reference_name, scalar_type="scalar_t"):
    return "sfem::codegen::%s<%s>::%s()" % (
        quadrature_reference_struct_name(prefix, stage),
        scalar_type,
        reference_name,
    )


def quadrature_reference_struct_lines(prefix, stage, references):
    struct_name = quadrature_reference_struct_name(prefix, stage)
    lines = [
        "",
        "template <typename scalar_t>",
        "struct %s {" % struct_name,
    ]
    for reference in references:
        values = tuple(reference.values)
        lines.extend(
            [
                "    static const scalar_t *%s() {" % reference.name,
                "        static const scalar_t data[%d] = {%s};"
                % (
                    len(values),
                    cpp_scalar_initializer_list(values, "scalar_t"),
                ),
                "        return data;",
                "    }",
            ]
        )
    lines.append("};")
    return lines
