#include "../../matrix_formats.hpp"

namespace sfem {
namespace codegen {

static const MatrixAssemblyDiagnostics modified_mooney_rivlin_tet10_bsr_standard_matrix_assembly_diagnostics_data = {
    "bsr",
    "standard",
    "none",
    "standard_block_elements",
    "bsr",
    "rowptr_colidx",
    "node_major_row_component_column_component",
    "add_scatter",
    "requires_node_block_graph",
    "atomic_add",
    0,
    1,
    3,
    3,
    3,
    10,
    10,
    100,
    1,
    30,
    30,
    900,
    60,
    900,
    1800,
    size_t(14640),
};

static const MatrixAssemblyDiagnostics *const modified_mooney_rivlin_tet10_matrix_assembly_variants[1] = {
    &modified_mooney_rivlin_tet10_bsr_standard_matrix_assembly_diagnostics_data,
};

int modified_mooney_rivlin_tet10_matrix_assembly_variant_count() {
    return 1;
}

const MatrixAssemblyDiagnostics *modified_mooney_rivlin_tet10_matrix_assembly_variant(const int variant) {
    return (variant >= 0 && variant < 1) ? modified_mooney_rivlin_tet10_matrix_assembly_variants[variant] : nullptr;
}

void modified_mooney_rivlin_tet10_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    const MatrixAssemblyDiagnostics *const d = modified_mooney_rivlin_tet10_matrix_assembly_variant(variant);
    if (d) {
        MatrixAssemblyDiagnostics_print("modified_mooney_rivlin_tet10", d, nelements);
    }
}

} // namespace codegen
} // namespace sfem

extern "C" int modified_mooney_rivlin_tet10_matrix_assembly_variant_count() {
    return sfem::codegen::modified_mooney_rivlin_tet10_matrix_assembly_variant_count();
}

extern "C" const sfem_MatrixAssemblyDiagnostics *modified_mooney_rivlin_tet10_matrix_assembly_variant(const int variant) {
    return sfem::codegen::modified_mooney_rivlin_tet10_matrix_assembly_variant(variant);
}

extern "C" void modified_mooney_rivlin_tet10_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    sfem::codegen::modified_mooney_rivlin_tet10_matrix_assembly_print_variant(variant, nelements);
}
