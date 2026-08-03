#include "../../matrix_formats.hpp"

namespace sfem {
namespace codegen {

static const MatrixAssemblyDiagnostics linear_elasticity_proteus_hex64_bsr_standard_matrix_assembly_diagnostics_data = {
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
    64,
    64,
    4096,
    1,
    192,
    192,
    36864,
    384,
    36864,
    73728,
    size_t(591360),
};

static const MatrixAssemblyDiagnostics *const linear_elasticity_proteus_hex64_matrix_assembly_variants[1] = {
    &linear_elasticity_proteus_hex64_bsr_standard_matrix_assembly_diagnostics_data,
};

int linear_elasticity_proteus_hex64_matrix_assembly_variant_count() {
    return 1;
}

const MatrixAssemblyDiagnostics *linear_elasticity_proteus_hex64_matrix_assembly_variant(const int variant) {
    return (variant >= 0 && variant < 1) ? linear_elasticity_proteus_hex64_matrix_assembly_variants[variant] : nullptr;
}

void linear_elasticity_proteus_hex64_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    const MatrixAssemblyDiagnostics *const d = linear_elasticity_proteus_hex64_matrix_assembly_variant(variant);
    if (d) {
        MatrixAssemblyDiagnostics_print("linear_elasticity_proteus_hex64", d, nelements);
    }
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_proteus_hex64_matrix_assembly_variant_count() {
    return sfem::codegen::linear_elasticity_proteus_hex64_matrix_assembly_variant_count();
}

extern "C" const sfem_MatrixAssemblyDiagnostics *linear_elasticity_proteus_hex64_matrix_assembly_variant(const int variant) {
    return sfem::codegen::linear_elasticity_proteus_hex64_matrix_assembly_variant(variant);
}

extern "C" void linear_elasticity_proteus_hex64_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    sfem::codegen::linear_elasticity_proteus_hex64_matrix_assembly_print_variant(variant, nelements);
}
