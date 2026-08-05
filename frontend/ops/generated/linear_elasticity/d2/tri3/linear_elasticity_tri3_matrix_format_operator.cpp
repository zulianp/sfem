#include "../../matrix_formats.hpp"

namespace sfem {
namespace codegen {

static const MatrixAssemblyDiagnostics linear_elasticity_tri3_bsr_standard_matrix_assembly_diagnostics_data = {
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
    3,
    3,
    9,
    1,
    6,
    6,
    36,
    12,
    36,
    72,
    size_t(624),
};

static const MatrixAssemblyDiagnostics linear_elasticity_tri3_block_diag_sym_standard_matrix_assembly_diagnostics_data = {
    "block_diag_sym",
    "standard",
    "none",
    "unspecified",
    "block_diag_sym",
    "unspecified",
    "unspecified",
    "unspecified",
    "unspecified",
    "unspecified",
    0,
    1,
    1,
    1,
    6,
    6,
    36,
    1,
    6,
    6,
    36,
    12,
    9,
    72,
    size_t(408),
};

static const MatrixAssemblyDiagnostics *const linear_elasticity_tri3_matrix_assembly_variants[2] = {
    &linear_elasticity_tri3_bsr_standard_matrix_assembly_diagnostics_data,
    &linear_elasticity_tri3_block_diag_sym_standard_matrix_assembly_diagnostics_data,
};

int linear_elasticity_tri3_matrix_assembly_variant_count() {
    return 2;
}

const MatrixAssemblyDiagnostics *linear_elasticity_tri3_matrix_assembly_variant(const int variant) {
    return (variant >= 0 && variant < 2) ? linear_elasticity_tri3_matrix_assembly_variants[variant] : nullptr;
}

void linear_elasticity_tri3_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    const MatrixAssemblyDiagnostics *const d = linear_elasticity_tri3_matrix_assembly_variant(variant);
    if (d) {
        MatrixAssemblyDiagnostics_print("linear_elasticity_tri3", d, nelements);
    }
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_tri3_matrix_assembly_variant_count() {
    return sfem::codegen::linear_elasticity_tri3_matrix_assembly_variant_count();
}

extern "C" const sfem_MatrixAssemblyDiagnostics *linear_elasticity_tri3_matrix_assembly_variant(const int variant) {
    return sfem::codegen::linear_elasticity_tri3_matrix_assembly_variant(variant);
}

extern "C" void linear_elasticity_tri3_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    sfem::codegen::linear_elasticity_tri3_matrix_assembly_print_variant(variant, nelements);
}
