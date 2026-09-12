#include "../../matrix_formats.hpp"

namespace sfem {
namespace codegen {

static const MatrixAssemblyDiagnostics mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_bsr_standard_matrix_assembly_diagnostics_data = {
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

static const MatrixAssemblyDiagnostics *const mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_variants[1] = {
  &mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_bsr_standard_matrix_assembly_diagnostics_data,
};

int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_variant_count() {
  return 1;
}

const MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_variant(const int variant) {
  return (variant >= 0 && variant < 1) ? mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_variants[variant] : nullptr;
}

void mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
  const MatrixAssemblyDiagnostics *const d = mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_variant(variant);
  if (d) {
    MatrixAssemblyDiagnostics_print("mooney_rivlin_kelvin_voigt_newmark_viscous_hex8", d, nelements);
  }
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_variant_count() {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_variant_count();
}

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_variant(const int variant) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_variant(variant);
}

extern "C" void mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements) {
  sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_print_variant(variant, nelements);
}
