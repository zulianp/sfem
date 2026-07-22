from codegen.framework.plans.matrix_formats import MatrixFormatPlan
from codegen.framework.symbolic.core import GeneratedKernelFile


def emit_matrix_format_metadata_files(operator_prefix, matrix_format_plan):
    if matrix_format_plan is None or matrix_format_plan.is_empty:
        return ()
    if not isinstance(matrix_format_plan, MatrixFormatPlan):
        raise TypeError("matrix_format_plan must be a MatrixFormatPlan")
    return (
        GeneratedKernelFile("matrix_formats.hpp", matrix_formats_header_source()),
        GeneratedKernelFile(
            "%s_matrix_format_operator.cpp" % operator_prefix,
            matrix_format_operator_source(operator_prefix, matrix_format_plan),
        ),
    )


def matrix_formats_header_source():
    return """#ifndef SFEM_CODEGEN_MATRIX_FORMATS_HPP
#define SFEM_CODEGEN_MATRIX_FORMATS_HPP

#include <cstddef>
#include <cstdio>

namespace sfem {
namespace codegen {

struct MatrixAssemblyDiagnostics {
    const char *format;
    const char *mesh_layout;
    const char *packed_pass;
    int node_index_filter;
    int format_aware_apply;
    ptrdiff_t row_dofs_per_element;
    ptrdiff_t column_dofs_per_element;
    ptrdiff_t entries_per_element;
    ptrdiff_t index_reads_per_element;
    ptrdiff_t value_writes_per_element;
    double flops_per_element;
    size_t bytes_per_element;
};

static inline double MatrixAssemblyDiagnostics_total_flops(
        const MatrixAssemblyDiagnostics *const d,
        const ptrdiff_t nelements) {
    return d->flops_per_element * double(nelements);
}

static inline size_t MatrixAssemblyDiagnostics_total_bytes(
        const MatrixAssemblyDiagnostics *const d,
        const ptrdiff_t nelements) {
    return size_t(nelements) * d->bytes_per_element;
}

static inline double MatrixAssemblyDiagnostics_arithmetic_intensity(
        const MatrixAssemblyDiagnostics *const d) {
    return d->bytes_per_element == 0 ? 0.0 : d->flops_per_element / double(d->bytes_per_element);
}

static inline void MatrixAssemblyDiagnostics_print(
        const char *const name,
        const MatrixAssemblyDiagnostics *const d,
        const ptrdiff_t nelements) {
    std::printf(
            "%s format=%s mesh=%s pass=%s indexed=%d apply=%d rows=%td cols=%td entries=%td bytes=%zu ai=%g\\n",
            name,
            d->format,
            d->mesh_layout,
            d->packed_pass,
            d->node_index_filter,
            d->format_aware_apply,
            d->row_dofs_per_element,
            d->column_dofs_per_element,
            d->entries_per_element,
            MatrixAssemblyDiagnostics_total_bytes(d, nelements),
            MatrixAssemblyDiagnostics_arithmetic_intensity(d));
}

} // namespace codegen
} // namespace sfem

#endif // SFEM_CODEGEN_MATRIX_FORMATS_HPP
"""


def matrix_format_operator_source(operator_prefix, matrix_format_plan):
    operator_prefix = str(operator_prefix)
    lines = [
        '#include "matrix_formats.hpp"',
        "",
        "namespace sfem {",
        "namespace codegen {",
        "",
    ]
    names = []
    for variant in matrix_format_plan.variants:
        name = "%s_%s_matrix_assembly_diagnostics_data" % (
            operator_prefix,
            variant.name,
        )
        names.append(name)
        lines.extend(_variant_definition_lines(name, variant))
        lines.append("")
    lines.extend(
        [
            "static const MatrixAssemblyDiagnostics *const %s_matrix_assembly_variants[%d] = {"
            % (operator_prefix, len(names)),
        ]
    )
    lines.extend("    &%s," % name for name in names)
    lines.extend(
        [
            "};",
            "",
            "int %s_matrix_assembly_variant_count() {" % operator_prefix,
            "    return %d;" % len(names),
            "}",
            "",
            "const MatrixAssemblyDiagnostics *%s_matrix_assembly_variant(const int variant) {"
            % operator_prefix,
            "    return (variant >= 0 && variant < %d) ? %s_matrix_assembly_variants[variant] : nullptr;"
            % (len(names), operator_prefix),
            "}",
            "",
            "void %s_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {"
            % operator_prefix,
            "    const MatrixAssemblyDiagnostics *const d = %s_matrix_assembly_variant(variant);"
            % operator_prefix,
            "    if (d) {",
            "        MatrixAssemblyDiagnostics_print(\"%s\", d, nelements);" % operator_prefix,
            "    }",
            "}",
            "",
            "} // namespace codegen",
            "} // namespace sfem",
            "",
        ]
    )
    return "\n".join(lines)


def _variant_definition_lines(name, variant):
    return [
        "static const MatrixAssemblyDiagnostics %s = {" % name,
        '    "%s",' % variant.matrix_format.value,
        '    "%s",' % variant.mesh_layout.value,
        '    "%s",' % variant.packed_pass.value,
        "    %d," % int(variant.node_index_filter),
        "    %d," % int(variant.format_aware_apply),
        "    %d," % int(variant.row_dofs_per_element),
        "    %d," % int(variant.column_dofs_per_element),
        "    %d," % int(variant.entries_per_element),
        "    %d," % int(variant.index_reads_per_element),
        "    %d," % int(variant.value_writes_per_element),
        "    %.17g," % float(variant.expected_flops_per_element),
        "    size_t(%d)," % int(variant.expected_bytes_per_element),
        "};",
    ]
