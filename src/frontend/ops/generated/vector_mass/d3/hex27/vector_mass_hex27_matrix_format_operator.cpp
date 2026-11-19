#include "matrix_formats.hpp"

namespace sfem {
namespace codegen {

static const MatrixAssemblyDiagnostics vector_mass_hex27_bsr_standard_matrix_assembly_diagnostics_data = {
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

static const MatrixAssemblyDiagnostics *const vector_mass_hex27_matrix_assembly_variants[1] = {
  &vector_mass_hex27_bsr_standard_matrix_assembly_diagnostics_data,
};

int vector_mass_hex27_matrix_assembly_variant_count() {
  return 1;
}

const MatrixAssemblyDiagnostics *vector_mass_hex27_matrix_assembly_variant(const int variant) {
  return (variant >= 0 && variant < 1) ? vector_mass_hex27_matrix_assembly_variants[variant] : nullptr;
}

void vector_mass_hex27_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
  const MatrixAssemblyDiagnostics *const d = vector_mass_hex27_matrix_assembly_variant(variant);
  if (d) {
    MatrixAssemblyDiagnostics_print("vector_mass_hex27", d, nelements);
  }
}

} // namespace codegen
} // namespace sfem

extern "C" int vector_mass_hex27_matrix_assembly_variant_count() {
  return sfem::codegen::vector_mass_hex27_matrix_assembly_variant_count();
}

extern "C" const sfem_MatrixAssemblyDiagnostics *vector_mass_hex27_matrix_assembly_variant(const int variant) {
  return sfem::codegen::vector_mass_hex27_matrix_assembly_variant(variant);
}

extern "C" void vector_mass_hex27_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
  sfem::codegen::vector_mass_hex27_matrix_assembly_print_variant(variant, nelements);
}
