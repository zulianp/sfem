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

struct sfem_MatrixAssemblyDiagnostics {
    const char *format;
    const char *mesh_layout;
    const char *packed_pass;
    const char *mesh_access;
    const char *assembly_kind;
    const char *index_policy;
    const char *value_layout;
    const char *accumulation_policy;
    const char *structural_compatibility;
    const char *reduction_policy;
    int node_index_filter;
    int row_block_size;
    int column_block_size;
    int block_size;
    int block_rows_per_element;
    int block_columns_per_element;
    int block_entries_per_element;
    int compatible_block_size;
    ptrdiff_t row_dofs_per_element;
    ptrdiff_t column_dofs_per_element;
    ptrdiff_t entries_per_element;
    ptrdiff_t index_reads_per_element;
    ptrdiff_t value_writes_per_element;
    double flops_per_element;
    size_t bytes_per_element;
};

namespace sfem {
namespace codegen {

using MatrixAssemblyDiagnostics = ::sfem_MatrixAssemblyDiagnostics;

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
            "%s format=%s mesh=%s pass=%s access=%s layout=%s index=%s values=%s accum=%s compat=%s reduce=%s indexed=%d block=%d rows=%td cols=%td entries=%td bytes=%zu ai=%g\\n",
            name,
            d->format,
            d->mesh_layout,
            d->packed_pass,
            d->mesh_access,
            d->assembly_kind,
            d->index_policy,
            d->value_layout,
            d->accumulation_policy,
            d->structural_compatibility,
            d->reduction_policy,
            d->node_index_filter,
            d->block_size,
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
            'extern "C" int %s_matrix_assembly_variant_count() {' % operator_prefix,
            "    return sfem::codegen::%s_matrix_assembly_variant_count();" % operator_prefix,
            "}",
            "",
            'extern "C" const sfem_MatrixAssemblyDiagnostics *%s_matrix_assembly_variant(const int variant) {'
            % operator_prefix,
            "    return sfem::codegen::%s_matrix_assembly_variant(variant);" % operator_prefix,
            "}",
            "",
            'extern "C" void %s_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {'
            % operator_prefix,
            "    sfem::codegen::%s_matrix_assembly_print_variant(variant, nelements);" % operator_prefix,
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _variant_definition_lines(name, variant):
    layout = _diagnostics_layout_fields(variant)
    return [
        "static const MatrixAssemblyDiagnostics %s = {" % name,
        '    "%s",' % variant.matrix_format.value,
        '    "%s",' % variant.mesh_layout.value,
        '    "%s",' % variant.packed_pass.value,
        '    "%s",' % layout["mesh_access"],
        '    "%s",' % layout["assembly_kind"],
        '    "%s",' % layout["index_policy"],
        '    "%s",' % layout["value_layout"],
        '    "%s",' % layout["accumulation_policy"],
        '    "%s",' % layout["structural_compatibility"],
        '    "%s",' % layout["reduction_policy"],
        "    %d," % int(variant.node_index_filter),
        "    %d," % int(layout["row_block_size"]),
        "    %d," % int(layout["column_block_size"]),
        "    %d," % int(layout["block_size"]),
        "    %d," % int(layout["block_rows_per_element"]),
        "    %d," % int(layout["block_columns_per_element"]),
        "    %d," % int(layout["block_entries_per_element"]),
        "    %d," % int(layout["compatible_block_size"]),
        "    %d," % int(variant.row_dofs_per_element),
        "    %d," % int(variant.column_dofs_per_element),
        "    %d," % int(variant.entries_per_element),
        "    %d," % int(variant.index_reads_per_element),
        "    %d," % int(variant.value_writes_per_element),
        "    %.17g," % float(variant.expected_flops_per_element),
        "    size_t(%d)," % int(variant.expected_bytes_per_element),
        "};",
    ]


def _diagnostics_layout_fields(variant):
    plan = getattr(variant, "assembly_plan", None)
    if plan is None:
        return _fallback_layout_fields(variant)
    data = plan.to_dict()
    kind = data.get("kind", variant.matrix_format.value)
    if kind == "crs":
        return {
            "assembly_kind": "crs",
            "mesh_access": data["mesh_access"],
            "index_policy": "%s_%s" % (data["row_pointer"], data["column_index"]),
            "value_layout": "scalar_element_matrix",
            "accumulation_policy": data["accumulation_policy"],
            "structural_compatibility": data["structural_compatibility"],
            "reduction_policy": data["reduction_policy"],
            "row_block_size": 1,
            "column_block_size": 1,
            "block_size": 1,
            "block_rows_per_element": variant.row_dofs_per_element,
            "block_columns_per_element": variant.column_dofs_per_element,
            "block_entries_per_element": variant.entries_per_element,
            "compatible_block_size": 1,
        }
    if kind == "bsr":
        return {
            "assembly_kind": "bsr",
            "mesh_access": data["mesh_access"],
            "index_policy": "%s_%s" % (data["row_pointer"], data["column_index"]),
            "value_layout": data["block_value_layout"],
            "accumulation_policy": data["accumulation_policy"],
            "structural_compatibility": data["structural_compatibility"],
            "reduction_policy": data["reduction_policy"],
            "row_block_size": data["row_block_size"],
            "column_block_size": data["column_block_size"],
            "block_size": data["block_size"],
            "block_rows_per_element": data["block_rows_per_element"],
            "block_columns_per_element": data["block_columns_per_element"],
            "block_entries_per_element": data["block_entries_per_element"],
            "compatible_block_size": int(data["compatible_block_size"]),
        }
    if kind == "dia":
        return {
            "assembly_kind": "dia",
            "mesh_access": data["mesh_access"],
            "index_policy": data["diagonal_offsets"],
            "value_layout": data["value_layout"],
            "accumulation_policy": data["accumulation_policy"],
            "structural_compatibility": data["structural_compatibility"],
            "reduction_policy": data["reduction_policy"],
            "row_block_size": 1,
            "column_block_size": 1,
            "block_size": 1,
            "block_rows_per_element": data["row_dofs_per_element"],
            "block_columns_per_element": 1,
            "block_entries_per_element": data["values_per_element"],
            "compatible_block_size": 1,
        }
    if kind == "coo":
        return {
            "assembly_kind": "coo",
            "mesh_access": data["mesh_access"],
            "index_policy": "%s_%s" % (data["row_index_stream"], data["column_index_stream"]),
            "value_layout": "triplet_element_matrix",
            "accumulation_policy": data["accumulation_policy"],
            "structural_compatibility": data["structural_compatibility"],
            "reduction_policy": data["duplicate_policy"],
            "row_block_size": 1,
            "column_block_size": 1,
            "block_size": 1,
            "block_rows_per_element": variant.row_dofs_per_element,
            "block_columns_per_element": variant.column_dofs_per_element,
            "block_entries_per_element": data["entries_per_element"],
            "compatible_block_size": 1,
        }
    if kind == "patch":
        return {
            "assembly_kind": "patch",
            "mesh_access": data["mesh_access"],
            "index_policy": data["patch_graph"],
            "value_layout": data["patch_value_layout"],
            "accumulation_policy": data["accumulation_policy"],
            "structural_compatibility": data["structural_compatibility"],
            "reduction_policy": data["reduction_policy"],
            "row_block_size": 1,
            "column_block_size": 1,
            "block_size": 1,
            "block_rows_per_element": data["row_dofs_per_patch"],
            "block_columns_per_element": data["column_dofs_per_patch"],
            "block_entries_per_element": data["entries_per_patch"],
            "compatible_block_size": 1,
        }
    return _fallback_layout_fields(variant)


def _fallback_layout_fields(variant):
    return {
        "assembly_kind": variant.matrix_format.value,
        "mesh_access": "unspecified",
        "index_policy": "unspecified",
        "value_layout": "unspecified",
        "accumulation_policy": "unspecified",
        "structural_compatibility": "unspecified",
        "reduction_policy": "unspecified",
        "row_block_size": 1,
        "column_block_size": 1,
        "block_size": 1,
        "block_rows_per_element": variant.row_dofs_per_element,
        "block_columns_per_element": variant.column_dofs_per_element,
        "block_entries_per_element": variant.entries_per_element,
        "compatible_block_size": 1,
    }
