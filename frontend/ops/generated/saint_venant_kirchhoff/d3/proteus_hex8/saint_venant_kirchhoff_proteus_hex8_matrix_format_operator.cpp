#include "../../matrix_formats.hpp"

namespace sfem {
namespace codegen {

static const MatrixAssemblyDiagnostics saint_venant_kirchhoff_proteus_hex8_bsr_standard_matrix_assembly_diagnostics_data = {
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
    8,
    8,
    64,
    1,
    24,
    24,
    576,
    48,
    576,
    1152,
    size_t(9408),
};

static const MatrixAssemblyDiagnostics *const saint_venant_kirchhoff_proteus_hex8_matrix_assembly_variants[1] = {
    &saint_venant_kirchhoff_proteus_hex8_bsr_standard_matrix_assembly_diagnostics_data,
};

int saint_venant_kirchhoff_proteus_hex8_matrix_assembly_variant_count() {
    return 1;
}

const MatrixAssemblyDiagnostics *saint_venant_kirchhoff_proteus_hex8_matrix_assembly_variant(const int variant) {
    return (variant >= 0 && variant < 1) ? saint_venant_kirchhoff_proteus_hex8_matrix_assembly_variants[variant] : nullptr;
}

void saint_venant_kirchhoff_proteus_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    const MatrixAssemblyDiagnostics *const d = saint_venant_kirchhoff_proteus_hex8_matrix_assembly_variant(variant);
    if (d) {
        MatrixAssemblyDiagnostics_print("saint_venant_kirchhoff_proteus_hex8", d, nelements);
    }
}

} // namespace codegen
} // namespace sfem

extern "C" int saint_venant_kirchhoff_proteus_hex8_matrix_assembly_variant_count() {
    return sfem::codegen::saint_venant_kirchhoff_proteus_hex8_matrix_assembly_variant_count();
}

extern "C" const sfem_MatrixAssemblyDiagnostics *saint_venant_kirchhoff_proteus_hex8_matrix_assembly_variant(const int variant) {
    return sfem::codegen::saint_venant_kirchhoff_proteus_hex8_matrix_assembly_variant(variant);
}

extern "C" void saint_venant_kirchhoff_proteus_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    sfem::codegen::saint_venant_kirchhoff_proteus_hex8_matrix_assembly_print_variant(variant, nelements);
}
