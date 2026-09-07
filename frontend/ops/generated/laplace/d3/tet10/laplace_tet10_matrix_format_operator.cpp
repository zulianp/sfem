#include "../../matrix_formats.hpp"

namespace sfem {
namespace codegen {

static const MatrixAssemblyDiagnostics laplace_tet10_crs_standard_matrix_assembly_diagnostics_data = {
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
    10,
    10,
    100,
    1,
    10,
    10,
    100,
    20,
    100,
    200,
    size_t(1680),
};

static const MatrixAssemblyDiagnostics laplace_tet10_bsr_standard_matrix_assembly_diagnostics_data = {
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
    10,
    10,
    100,
    1,
    10,
    10,
    100,
    20,
    100,
    200,
    size_t(1680),
};

static const MatrixAssemblyDiagnostics *const laplace_tet10_matrix_assembly_variants[2] = {
    &laplace_tet10_crs_standard_matrix_assembly_diagnostics_data,
    &laplace_tet10_bsr_standard_matrix_assembly_diagnostics_data,
};

int laplace_tet10_matrix_assembly_variant_count() {
    return 2;
}

const MatrixAssemblyDiagnostics *laplace_tet10_matrix_assembly_variant(const int variant) {
    return (variant >= 0 && variant < 2) ? laplace_tet10_matrix_assembly_variants[variant] : nullptr;
}

void laplace_tet10_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    const MatrixAssemblyDiagnostics *const d = laplace_tet10_matrix_assembly_variant(variant);
    if (d) {
        MatrixAssemblyDiagnostics_print("laplace_tet10", d, nelements);
    }
}

} // namespace codegen
} // namespace sfem

extern "C" int laplace_tet10_matrix_assembly_variant_count() {
    return sfem::codegen::laplace_tet10_matrix_assembly_variant_count();
}

extern "C" const sfem_MatrixAssemblyDiagnostics *laplace_tet10_matrix_assembly_variant(const int variant) {
    return sfem::codegen::laplace_tet10_matrix_assembly_variant(variant);
}

extern "C" void laplace_tet10_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
    sfem::codegen::laplace_tet10_matrix_assembly_print_variant(variant, nelements);
}
