#include "../../matrix_formats.hpp"

namespace sfem {
namespace codegen {

static const MatrixAssemblyDiagnostics linear_elasticity_quad4_bsr_standard_matrix_assembly_diagnostics_data = {
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
    2,
    2,
    2,
    4,
    4,
    16,
    1,
    8,
    8,
    64,
    16,
    64,
    128,
    size_t(1088),
};

static const MatrixAssemblyDiagnostics *const linear_elasticity_quad4_matrix_assembly_variants[1] = {
    &linear_elasticity_quad4_bsr_standard_matrix_assembly_diagnostics_data,
};

int linear_elasticity_quad4_matrix_assembly_variant_count() {
    return 1;
}

const MatrixAssemblyDiagnostics *linear_elasticity_quad4_matrix_assembly_variant(const int variant) {
    return (variant >= 0 && variant < 1) ? linear_elasticity_quad4_matrix_assembly_variants[variant] : nullptr;
}

void linear_elasticity_quad4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    const MatrixAssemblyDiagnostics *const d = linear_elasticity_quad4_matrix_assembly_variant(variant);
    if (d) {
        MatrixAssemblyDiagnostics_print("linear_elasticity_quad4", d, nelements);
    }
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_quad4_matrix_assembly_variant_count() {
    return sfem::codegen::linear_elasticity_quad4_matrix_assembly_variant_count();
}

extern "C" const sfem_MatrixAssemblyDiagnostics *linear_elasticity_quad4_matrix_assembly_variant(const int variant) {
    return sfem::codegen::linear_elasticity_quad4_matrix_assembly_variant(variant);
}

extern "C" void linear_elasticity_quad4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    sfem::codegen::linear_elasticity_quad4_matrix_assembly_print_variant(variant, nelements);
}
