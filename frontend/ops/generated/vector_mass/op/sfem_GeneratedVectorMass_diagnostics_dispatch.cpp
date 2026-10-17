#include "sfem_GeneratedVectorMass_c_abi.hpp"
#include <cstdio>

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_proteus_quad4_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_quad4_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tri3_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_hex27_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_hex8_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_proteus_hex27_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_proteus_hex8_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet10_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet4_jacobian_action_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_proteus_quad4_residual_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_quad4_residual_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tri3_residual_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_hex27_residual_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_hex8_residual_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_proteus_hex27_residual_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_proteus_hex8_residual_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet10_residual_esoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_tet4_residual_esoa_diagnostics(void);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_jacobian_action_2d_esoa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return vector_mass_proteus_quad4_jacobian_action_esoa_diagnostics();
    case smesh::QUAD4:
      return vector_mass_quad4_jacobian_action_esoa_diagnostics();
    case smesh::TRI3:
      return vector_mass_tri3_jacobian_action_esoa_diagnostics();
    default:
      std::fprintf(stderr, "vector_mass_jacobian_action_2d_esoa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_jacobian_action_3d_esoa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::HEX27:
      return vector_mass_hex27_jacobian_action_esoa_diagnostics();
    case smesh::HEX8:
      return vector_mass_hex8_jacobian_action_esoa_diagnostics();
    case smesh::PROTEUS_HEX27:
      return vector_mass_proteus_hex27_jacobian_action_esoa_diagnostics();
    case smesh::PROTEUS_HEX8:
      return vector_mass_proteus_hex8_jacobian_action_esoa_diagnostics();
    case smesh::TET10:
      return vector_mass_tet10_jacobian_action_esoa_diagnostics();
    case smesh::TET4:
      return vector_mass_tet4_jacobian_action_esoa_diagnostics();
    default:
      std::fprintf(stderr, "vector_mass_jacobian_action_3d_esoa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_residual_2d_esoa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return vector_mass_proteus_quad4_residual_esoa_diagnostics();
    case smesh::QUAD4:
      return vector_mass_quad4_residual_esoa_diagnostics();
    case smesh::TRI3:
      return vector_mass_tri3_residual_esoa_diagnostics();
    default:
      std::fprintf(stderr, "vector_mass_residual_2d_esoa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *vector_mass_residual_3d_esoa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::HEX27:
      return vector_mass_hex27_residual_esoa_diagnostics();
    case smesh::HEX8:
      return vector_mass_hex8_residual_esoa_diagnostics();
    case smesh::PROTEUS_HEX27:
      return vector_mass_proteus_hex27_residual_esoa_diagnostics();
    case smesh::PROTEUS_HEX8:
      return vector_mass_proteus_hex8_residual_esoa_diagnostics();
    case smesh::TET10:
      return vector_mass_tet10_residual_esoa_diagnostics();
    case smesh::TET4:
      return vector_mass_tet4_residual_esoa_diagnostics();
    default:
      std::fprintf(stderr, "vector_mass_residual_3d_esoa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}
