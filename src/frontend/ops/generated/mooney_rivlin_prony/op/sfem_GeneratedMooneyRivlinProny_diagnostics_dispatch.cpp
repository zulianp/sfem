#include "sfem_GeneratedMooneyRivlinProny_c_abi.hpp"
#include <cstdio>

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_proteus_quad4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_quad4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_tri3_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_hex8_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_proteus_hex8_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_tet10_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_tet4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_proteus_quad4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_quad4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_tri3_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_hex8_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_proteus_hex8_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_tet10_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_tet4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_proteus_quad4_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_quad4_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_tri3_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_hex8_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_proteus_hex8_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_tet10_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_tet4_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_proteus_quad4_quadrature_output_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_quad4_quadrature_output_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_tri3_quadrature_output_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_hex8_quadrature_output_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_proteus_hex8_quadrature_output_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_tet10_quadrature_output_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_tet4_quadrature_output_soa_diagnostics(void);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_apply_2d_soa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return mooney_rivlin_prony_proteus_quad4_apply_soa_diagnostics();
    case smesh::QUAD4:
      return mooney_rivlin_prony_quad4_apply_soa_diagnostics();
    case smesh::TRI3:
      return mooney_rivlin_prony_tri3_apply_soa_diagnostics();
    default:
      std::fprintf(stderr, "mooney_rivlin_prony_apply_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_apply_3d_soa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::HEX8:
      return mooney_rivlin_prony_hex8_apply_soa_diagnostics();
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_prony_proteus_hex8_apply_soa_diagnostics();
    case smesh::TET10:
      return mooney_rivlin_prony_tet10_apply_soa_diagnostics();
    case smesh::TET4:
      return mooney_rivlin_prony_tet4_apply_soa_diagnostics();
    default:
      std::fprintf(stderr, "mooney_rivlin_prony_apply_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_gradient_2d_soa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return mooney_rivlin_prony_proteus_quad4_gradient_soa_diagnostics();
    case smesh::QUAD4:
      return mooney_rivlin_prony_quad4_gradient_soa_diagnostics();
    case smesh::TRI3:
      return mooney_rivlin_prony_tri3_gradient_soa_diagnostics();
    default:
      std::fprintf(stderr, "mooney_rivlin_prony_gradient_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_gradient_3d_soa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::HEX8:
      return mooney_rivlin_prony_hex8_gradient_soa_diagnostics();
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_prony_proteus_hex8_gradient_soa_diagnostics();
    case smesh::TET10:
      return mooney_rivlin_prony_tet10_gradient_soa_diagnostics();
    case smesh::TET4:
      return mooney_rivlin_prony_tet4_gradient_soa_diagnostics();
    default:
      std::fprintf(stderr, "mooney_rivlin_prony_gradient_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_objective_2d_soa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return mooney_rivlin_prony_proteus_quad4_objective_soa_diagnostics();
    case smesh::QUAD4:
      return mooney_rivlin_prony_quad4_objective_soa_diagnostics();
    case smesh::TRI3:
      return mooney_rivlin_prony_tri3_objective_soa_diagnostics();
    default:
      std::fprintf(stderr, "mooney_rivlin_prony_objective_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_objective_3d_soa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::HEX8:
      return mooney_rivlin_prony_hex8_objective_soa_diagnostics();
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_prony_proteus_hex8_objective_soa_diagnostics();
    case smesh::TET10:
      return mooney_rivlin_prony_tet10_objective_soa_diagnostics();
    case smesh::TET4:
      return mooney_rivlin_prony_tet4_objective_soa_diagnostics();
    default:
      std::fprintf(stderr, "mooney_rivlin_prony_objective_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_quadrature_output_2d_soa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return mooney_rivlin_prony_proteus_quad4_quadrature_output_soa_diagnostics();
    case smesh::QUAD4:
      return mooney_rivlin_prony_quad4_quadrature_output_soa_diagnostics();
    case smesh::TRI3:
      return mooney_rivlin_prony_tri3_quadrature_output_soa_diagnostics();
    default:
      std::fprintf(stderr, "mooney_rivlin_prony_quadrature_output_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *mooney_rivlin_prony_quadrature_output_3d_soa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::HEX8:
      return mooney_rivlin_prony_hex8_quadrature_output_soa_diagnostics();
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_prony_proteus_hex8_quadrature_output_soa_diagnostics();
    case smesh::TET10:
      return mooney_rivlin_prony_tet10_quadrature_output_soa_diagnostics();
    case smesh::TET4:
      return mooney_rivlin_prony_tet4_quadrature_output_soa_diagnostics();
    default:
      std::fprintf(stderr, "mooney_rivlin_prony_quadrature_output_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}
