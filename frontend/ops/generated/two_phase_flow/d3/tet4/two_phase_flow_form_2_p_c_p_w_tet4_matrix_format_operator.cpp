#include "../../matrix_formats.hpp"

namespace sfem {
namespace codegen {

static const MatrixAssemblyDiagnostics two_phase_flow_form_2_p_c_p_w_tet4_crs_standard_matrix_assembly_diagnostics_data = {
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
    4,
    4,
    16,
    1,
    4,
    4,
    16,
    8,
    16,
    32,
    size_t(288),
};

static const MatrixAssemblyDiagnostics two_phase_flow_form_2_p_c_p_w_tet4_bsr_standard_matrix_assembly_diagnostics_data = {
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
    1,
    4,
    4,
    16,
    1,
    4,
    4,
    16,
    8,
    16,
    32,
    size_t(288),
};

static const MatrixAssemblyDiagnostics two_phase_flow_form_2_p_c_p_w_tet4_dia_standard_matrix_assembly_diagnostics_data = {
    "dia",
    "standard",
    "none",
    "standard_block_elements",
    "dia",
    "diagonal_offsets",
    "diagonal_node_block_row_major",
    "fill_diagonal_values",
    "unsupported_mixed_or_asymmetric_diagonal_structure",
    "not_emitted",
    0,
    1,
    1,
    1,
    1,
    4,
    1,
    4,
    1,
    4,
    4,
    16,
    8,
    4,
    32,
    size_t(192),
};

static const MatrixAssemblyDiagnostics two_phase_flow_form_2_p_c_p_w_tet4_coo_standard_matrix_assembly_diagnostics_data = {
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
    4,
    4,
    16,
    1,
    4,
    4,
    16,
    8,
    16,
    32,
    size_t(288),
};

static const MatrixAssemblyDiagnostics two_phase_flow_form_2_p_c_p_w_tet4_patch_standard_matrix_assembly_diagnostics_data = {
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
    4,
    4,
    16,
    1,
    4,
    4,
    16,
    8,
    16,
    32,
    size_t(288),
};

static const MatrixAssemblyDiagnostics *const two_phase_flow_form_2_p_c_p_w_tet4_matrix_assembly_variants[5] = {
    &two_phase_flow_form_2_p_c_p_w_tet4_crs_standard_matrix_assembly_diagnostics_data,
    &two_phase_flow_form_2_p_c_p_w_tet4_bsr_standard_matrix_assembly_diagnostics_data,
    &two_phase_flow_form_2_p_c_p_w_tet4_dia_standard_matrix_assembly_diagnostics_data,
    &two_phase_flow_form_2_p_c_p_w_tet4_coo_standard_matrix_assembly_diagnostics_data,
    &two_phase_flow_form_2_p_c_p_w_tet4_patch_standard_matrix_assembly_diagnostics_data,
};

int two_phase_flow_form_2_p_c_p_w_tet4_matrix_assembly_variant_count() {
    return 5;
}

const MatrixAssemblyDiagnostics *two_phase_flow_form_2_p_c_p_w_tet4_matrix_assembly_variant(const int variant) {
    return (variant >= 0 && variant < 5) ? two_phase_flow_form_2_p_c_p_w_tet4_matrix_assembly_variants[variant] : nullptr;
}

void two_phase_flow_form_2_p_c_p_w_tet4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    const MatrixAssemblyDiagnostics *const d = two_phase_flow_form_2_p_c_p_w_tet4_matrix_assembly_variant(variant);
    if (d) {
        MatrixAssemblyDiagnostics_print("two_phase_flow_form_2_p_c_p_w_tet4", d, nelements);
    }
}

} // namespace codegen
} // namespace sfem

extern "C" int two_phase_flow_form_2_p_c_p_w_tet4_matrix_assembly_variant_count() {
    return sfem::codegen::two_phase_flow_form_2_p_c_p_w_tet4_matrix_assembly_variant_count();
}

extern "C" const sfem_MatrixAssemblyDiagnostics *two_phase_flow_form_2_p_c_p_w_tet4_matrix_assembly_variant(const int variant) {
    return sfem::codegen::two_phase_flow_form_2_p_c_p_w_tet4_matrix_assembly_variant(variant);
}

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    sfem::codegen::two_phase_flow_form_2_p_c_p_w_tet4_matrix_assembly_print_variant(variant, nelements);
}
