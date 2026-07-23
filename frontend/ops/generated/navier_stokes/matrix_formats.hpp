#ifndef SFEM_CODEGEN_MATRIX_FORMATS_HPP
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
    int format_aware_apply;
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
            "%s format=%s mesh=%s pass=%s access=%s layout=%s index=%s values=%s accum=%s compat=%s reduce=%s indexed=%d apply=%d block=%d rows=%td cols=%td entries=%td bytes=%zu ai=%g\n",
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
            d->format_aware_apply,
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
