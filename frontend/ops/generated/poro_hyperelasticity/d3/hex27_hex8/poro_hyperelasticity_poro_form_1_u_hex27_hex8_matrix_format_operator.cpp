#include "../../matrix_formats.hpp"

namespace sfem {
namespace codegen {

static const MatrixAssemblyDiagnostics poro_hyperelasticity_poro_form_1_u_hex27_hex8_crs_standard_matrix_assembly_diagnostics_data = {
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
    0,
    1,
    1,
    1,
    81,
    81,
    6561,
    1,
    81,
    81,
    6561,
    162,
    6561,
    13122,
    size_t(105624),
};

static const MatrixAssemblyDiagnostics poro_hyperelasticity_poro_form_1_u_hex27_hex8_bsr_standard_matrix_assembly_diagnostics_data = {
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
    27,
    27,
    729,
    1,
    81,
    81,
    6561,
    162,
    6561,
    13122,
    size_t(105624),
};

static const MatrixAssemblyDiagnostics poro_hyperelasticity_poro_form_1_u_hex27_hex8_dia_standard_matrix_assembly_diagnostics_data = {
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
    1,
    81,
    1,
    81,
    1,
    81,
    81,
    6561,
    162,
    81,
    13122,
    size_t(53784),
};

static const MatrixAssemblyDiagnostics poro_hyperelasticity_poro_form_1_u_hex27_hex8_coo_standard_matrix_assembly_diagnostics_data = {
    "coo",
    "standard",
    "none",
    "standard_block_elements",
    "coo",
    "rowidx_colidx",
    "triplet_element_matrix",
    "emit_triplets",
    "allows_duplicates",
    "deterministic_element_order_external_reduction",
    0,
    0,
    1,
    1,
    1,
    81,
    81,
    6561,
    1,
    81,
    81,
    6561,
    162,
    6561,
    13122,
    size_t(105624),
};

static const MatrixAssemblyDiagnostics poro_hyperelasticity_poro_form_1_u_hex27_hex8_patch_standard_matrix_assembly_diagnostics_data = {
    "patch",
    "standard",
    "none",
    "standard_block_elements",
    "patch",
    "rowptr_colidx",
    "node_block_crs_block_row_major",
    "add_scatter",
    "requires_full_graph",
    "atomic_add",
    0,
    1,
    1,
    1,
    1,
    81,
    81,
    6561,
    1,
    81,
    81,
    6561,
    162,
    6561,
    13122,
    size_t(105624),
};

static const MatrixAssemblyDiagnostics *const poro_hyperelasticity_poro_form_1_u_hex27_hex8_matrix_assembly_variants[5] = {
    &poro_hyperelasticity_poro_form_1_u_hex27_hex8_crs_standard_matrix_assembly_diagnostics_data,
    &poro_hyperelasticity_poro_form_1_u_hex27_hex8_bsr_standard_matrix_assembly_diagnostics_data,
    &poro_hyperelasticity_poro_form_1_u_hex27_hex8_dia_standard_matrix_assembly_diagnostics_data,
    &poro_hyperelasticity_poro_form_1_u_hex27_hex8_coo_standard_matrix_assembly_diagnostics_data,
    &poro_hyperelasticity_poro_form_1_u_hex27_hex8_patch_standard_matrix_assembly_diagnostics_data,
};

int poro_hyperelasticity_poro_form_1_u_hex27_hex8_matrix_assembly_variant_count() {
    return 5;
}

const MatrixAssemblyDiagnostics *poro_hyperelasticity_poro_form_1_u_hex27_hex8_matrix_assembly_variant(const int variant) {
    return (variant >= 0 && variant < 5) ? poro_hyperelasticity_poro_form_1_u_hex27_hex8_matrix_assembly_variants[variant] : nullptr;
}

void poro_hyperelasticity_poro_form_1_u_hex27_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    const MatrixAssemblyDiagnostics *const d = poro_hyperelasticity_poro_form_1_u_hex27_hex8_matrix_assembly_variant(variant);
    if (d) {
        MatrixAssemblyDiagnostics_print("poro_hyperelasticity_poro_form_1_u_hex27_hex8", d, nelements);
    }
}

} // namespace codegen
} // namespace sfem

extern "C" int poro_hyperelasticity_poro_form_1_u_hex27_hex8_matrix_assembly_variant_count() {
    return sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_matrix_assembly_variant_count();
}

extern "C" const sfem_MatrixAssemblyDiagnostics *poro_hyperelasticity_poro_form_1_u_hex27_hex8_matrix_assembly_variant(const int variant) {
    return sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_matrix_assembly_variant(variant);
}

extern "C" void poro_hyperelasticity_poro_form_1_u_hex27_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    sfem::codegen::poro_hyperelasticity_poro_form_1_u_hex27_hex8_matrix_assembly_print_variant(variant, nelements);
}
