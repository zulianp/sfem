#include "sfem_GeneratedMooneyRivlinKelvinVoigt_cuda_c_abi.hpp"
#include <cstdio>

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_proteus_quad4_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_quad4_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tri3_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_hex8_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_proteus_hex8_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_proteus_quad4_residual_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_quad4_residual_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tri3_residual_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_hex8_residual_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_proteus_hex8_residual_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet10_residual_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet4_residual_esoa_diagnostics(void);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_jacobian_action_2d_esoa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_total_proteus_quad4_jacobian_action_esoa_diagnostics();
    case smesh::QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_total_quad4_jacobian_action_esoa_diagnostics();
    case smesh::TRI3:
      return cu_mooney_rivlin_kelvin_voigt_total_tri3_jacobian_action_esoa_diagnostics();
    default:
      std::fprintf(stderr, "mooney_rivlin_kelvin_voigt_total_jacobian_action_2d_esoa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_jacobian_action_3d_esoa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::HEX8:
      return cu_mooney_rivlin_kelvin_voigt_total_hex8_jacobian_action_esoa_diagnostics();
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_kelvin_voigt_total_proteus_hex8_jacobian_action_esoa_diagnostics();
    case smesh::TET10:
      return cu_mooney_rivlin_kelvin_voigt_total_tet10_jacobian_action_esoa_diagnostics();
    case smesh::TET4:
      return cu_mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_esoa_diagnostics();
    default:
      std::fprintf(stderr, "mooney_rivlin_kelvin_voigt_total_jacobian_action_3d_esoa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_residual_2d_esoa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_total_proteus_quad4_residual_esoa_diagnostics();
    case smesh::QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_total_quad4_residual_esoa_diagnostics();
    case smesh::TRI3:
      return cu_mooney_rivlin_kelvin_voigt_total_tri3_residual_esoa_diagnostics();
    default:
      std::fprintf(stderr, "mooney_rivlin_kelvin_voigt_total_residual_2d_esoa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_residual_3d_esoa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::HEX8:
      return cu_mooney_rivlin_kelvin_voigt_total_hex8_residual_esoa_diagnostics();
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_kelvin_voigt_total_proteus_hex8_residual_esoa_diagnostics();
    case smesh::TET10:
      return cu_mooney_rivlin_kelvin_voigt_total_tet10_residual_esoa_diagnostics();
    case smesh::TET4:
      return cu_mooney_rivlin_kelvin_voigt_total_tet4_residual_esoa_diagnostics();
    default:
      std::fprintf(stderr, "mooney_rivlin_kelvin_voigt_total_residual_3d_esoa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}
