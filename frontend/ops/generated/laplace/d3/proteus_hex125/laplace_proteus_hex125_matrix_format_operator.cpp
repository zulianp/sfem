#include "../../matrix_formats.hpp"

namespace sfem {
namespace codegen {

static const MatrixAssemblyDiagnostics laplace_proteus_hex125_crs_standard_matrix_assembly_diagnostics_data = {
    "crs",
    "standard",
    "none",
    "standard_block_elements",
    "crs",
    "rowptr_colidx",
    "scalar_element_matrix",
    "add_scatter",
    "requires_full_graph",
    "atomic_add",
    0,
    1,
    1,
    1,
    125,
    125,
    15625,
    1,
    125,
    125,
    15625,
    250,
    15625,
    31250,
    size_t(251000),
};

static const MatrixAssemblyDiagnostics laplace_proteus_hex125_bsr_standard_matrix_assembly_diagnostics_data = {
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
    1,
    1,
    125,
    125,
    15625,
    1,
    125,
    125,
    15625,
    250,
    15625,
    31250,
    size_t(251000),
};

static const MatrixAssemblyDiagnostics laplace_proteus_hex125_dia_standard_matrix_assembly_diagnostics_data = {
    "dia",
    "standard",
    "none",
    "standard_block_elements",
    "dia",
    "diagonal_offsets",
    "diagonal_node_block_row_major",
    "fill_diagonal_values",
    "stable_tensor_product_diagonal_offsets",
    "atomic_add",
    0,
    1,
    1,
    1,
    125,
    1,
    125,
    1,
    125,
    125,
    15625,
    250,
    125,
    31250,
    size_t(127000),
};

static const MatrixAssemblyDiagnostics *const laplace_proteus_hex125_matrix_assembly_variants[3] = {
    &laplace_proteus_hex125_crs_standard_matrix_assembly_diagnostics_data,
    &laplace_proteus_hex125_bsr_standard_matrix_assembly_diagnostics_data,
    &laplace_proteus_hex125_dia_standard_matrix_assembly_diagnostics_data,
};

int laplace_proteus_hex125_matrix_assembly_variant_count() {
    return 3;
}

const MatrixAssemblyDiagnostics *laplace_proteus_hex125_matrix_assembly_variant(const int variant) {
    return (variant >= 0 && variant < 3) ? laplace_proteus_hex125_matrix_assembly_variants[variant] : nullptr;
}

void laplace_proteus_hex125_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    const MatrixAssemblyDiagnostics *const d = laplace_proteus_hex125_matrix_assembly_variant(variant);
    if (d) {
        MatrixAssemblyDiagnostics_print("laplace_proteus_hex125", d, nelements);
    }
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_proteus_hex125_matrix_assembly_variant_count() {
    return sfem::codegen::laplace_proteus_hex125_matrix_assembly_variant_count();
}

extern "C" const sfem_MatrixAssemblyDiagnostics *laplace_proteus_hex125_matrix_assembly_variant(const int variant) {
    return sfem::codegen::laplace_proteus_hex125_matrix_assembly_variant(variant);
}

extern "C" void laplace_proteus_hex125_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    sfem::codegen::laplace_proteus_hex125_matrix_assembly_print_variant(variant, nelements);
}
