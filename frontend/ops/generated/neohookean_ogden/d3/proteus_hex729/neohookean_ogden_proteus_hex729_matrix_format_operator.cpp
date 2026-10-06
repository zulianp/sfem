#include "../../matrix_formats.hpp"

namespace sfem {
namespace codegen {

static const MatrixAssemblyDiagnostics neohookean_ogden_proteus_hex729_bsr_standard_matrix_assembly_diagnostics_data = {
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
    3,
    3,
    3,
    729,
    729,
    531441,
    1,
    2187,
    2187,
    4782969,
    4374,
    4782969,
    9565938,
    size_t(76545000),
};

static const MatrixAssemblyDiagnostics *const neohookean_ogden_proteus_hex729_matrix_assembly_variants[1] = {
    &neohookean_ogden_proteus_hex729_bsr_standard_matrix_assembly_diagnostics_data,
};

int neohookean_ogden_proteus_hex729_matrix_assembly_variant_count() {
    return 1;
}

const MatrixAssemblyDiagnostics *neohookean_ogden_proteus_hex729_matrix_assembly_variant(const int variant) {
    return (variant >= 0 && variant < 1) ? neohookean_ogden_proteus_hex729_matrix_assembly_variants[variant] : nullptr;
}

void neohookean_ogden_proteus_hex729_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    const MatrixAssemblyDiagnostics *const d = neohookean_ogden_proteus_hex729_matrix_assembly_variant(variant);
    if (d) {
        MatrixAssemblyDiagnostics_print("neohookean_ogden_proteus_hex729", d, nelements);
    }
}

} // namespace codegen
} // namespace sfem

extern "C" int neohookean_ogden_proteus_hex729_matrix_assembly_variant_count() {
    return sfem::codegen::neohookean_ogden_proteus_hex729_matrix_assembly_variant_count();
}

extern "C" const sfem_MatrixAssemblyDiagnostics *neohookean_ogden_proteus_hex729_matrix_assembly_variant(const int variant) {
    return sfem::codegen::neohookean_ogden_proteus_hex729_matrix_assembly_variant(variant);
}

extern "C" void neohookean_ogden_proteus_hex729_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    sfem::codegen::neohookean_ogden_proteus_hex729_matrix_assembly_print_variant(variant, nelements);
}
