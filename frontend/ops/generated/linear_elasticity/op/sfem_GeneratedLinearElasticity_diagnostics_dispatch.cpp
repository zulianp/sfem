#include "sfem_GeneratedLinearElasticity_c_abi.hpp"
#include <cstdio>

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_quad4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_quad4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tri3_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_hex8_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_hex8_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet10_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_quad4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_quad4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tri3_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_hex8_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_hex8_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet10_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_quad4_inexact_apply_compressed_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tri3_inexact_apply_compressed_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_hex8_inexact_apply_compressed_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet10_inexact_apply_compressed_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet4_inexact_apply_compressed_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_quad4_inexact_apply_stored_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tri3_inexact_apply_stored_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_hex8_inexact_apply_stored_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet10_inexact_apply_stored_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet4_inexact_apply_stored_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_quad4_inexact_apply_tangent_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tri3_inexact_apply_tangent_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_hex8_inexact_apply_tangent_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet10_inexact_apply_tangent_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet4_inexact_apply_tangent_a_msoa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_quad4_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_quad4_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tri3_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_hex8_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_proteus_hex8_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet10_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet4_objective_soa_diagnostics(void);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_apply_2d_soa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return linear_elasticity_proteus_quad4_apply_soa_diagnostics();
    case smesh::QUAD4:
      return linear_elasticity_quad4_apply_soa_diagnostics();
    case smesh::TRI3:
      return linear_elasticity_tri3_apply_soa_diagnostics();
    default:
      std::fprintf(stderr, "linear_elasticity_apply_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_apply_3d_soa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::HEX8:
      return linear_elasticity_hex8_apply_soa_diagnostics();
    case smesh::PROTEUS_HEX8:
      return linear_elasticity_proteus_hex8_apply_soa_diagnostics();
    case smesh::TET10:
      return linear_elasticity_tet10_apply_soa_diagnostics();
    case smesh::TET4:
      return linear_elasticity_tet4_apply_soa_diagnostics();
    default:
      std::fprintf(stderr, "linear_elasticity_apply_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_gradient_2d_soa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return linear_elasticity_proteus_quad4_gradient_soa_diagnostics();
    case smesh::QUAD4:
      return linear_elasticity_quad4_gradient_soa_diagnostics();
    case smesh::TRI3:
      return linear_elasticity_tri3_gradient_soa_diagnostics();
    default:
      std::fprintf(stderr, "linear_elasticity_gradient_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_gradient_3d_soa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::HEX8:
      return linear_elasticity_hex8_gradient_soa_diagnostics();
    case smesh::PROTEUS_HEX8:
      return linear_elasticity_proteus_hex8_gradient_soa_diagnostics();
    case smesh::TET10:
      return linear_elasticity_tet10_gradient_soa_diagnostics();
    case smesh::TET4:
      return linear_elasticity_tet4_gradient_soa_diagnostics();
    default:
      std::fprintf(stderr, "linear_elasticity_gradient_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_inexact_apply_compressed_a_2d_msoa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return linear_elasticity_proteus_quad4_inexact_apply_compressed_a_msoa_diagnostics();
    case smesh::TRI3:
      return linear_elasticity_tri3_inexact_apply_compressed_a_msoa_diagnostics();
    default:
      std::fprintf(stderr, "linear_elasticity_inexact_apply_compressed_a_2d_msoa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_inexact_apply_compressed_a_3d_msoa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_HEX8:
      return linear_elasticity_proteus_hex8_inexact_apply_compressed_a_msoa_diagnostics();
    case smesh::TET10:
      return linear_elasticity_tet10_inexact_apply_compressed_a_msoa_diagnostics();
    case smesh::TET4:
      return linear_elasticity_tet4_inexact_apply_compressed_a_msoa_diagnostics();
    default:
      std::fprintf(stderr, "linear_elasticity_inexact_apply_compressed_a_3d_msoa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_inexact_apply_stored_a_2d_msoa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return linear_elasticity_proteus_quad4_inexact_apply_stored_a_msoa_diagnostics();
    case smesh::TRI3:
      return linear_elasticity_tri3_inexact_apply_stored_a_msoa_diagnostics();
    default:
      std::fprintf(stderr, "linear_elasticity_inexact_apply_stored_a_2d_msoa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_inexact_apply_stored_a_3d_msoa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_HEX8:
      return linear_elasticity_proteus_hex8_inexact_apply_stored_a_msoa_diagnostics();
    case smesh::TET10:
      return linear_elasticity_tet10_inexact_apply_stored_a_msoa_diagnostics();
    case smesh::TET4:
      return linear_elasticity_tet4_inexact_apply_stored_a_msoa_diagnostics();
    default:
      std::fprintf(stderr, "linear_elasticity_inexact_apply_stored_a_3d_msoa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_inexact_apply_tangent_a_2d_msoa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return linear_elasticity_proteus_quad4_inexact_apply_tangent_a_msoa_diagnostics();
    case smesh::TRI3:
      return linear_elasticity_tri3_inexact_apply_tangent_a_msoa_diagnostics();
    default:
      std::fprintf(stderr, "linear_elasticity_inexact_apply_tangent_a_2d_msoa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_inexact_apply_tangent_a_3d_msoa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_HEX8:
      return linear_elasticity_proteus_hex8_inexact_apply_tangent_a_msoa_diagnostics();
    case smesh::TET10:
      return linear_elasticity_tet10_inexact_apply_tangent_a_msoa_diagnostics();
    case smesh::TET4:
      return linear_elasticity_tet4_inexact_apply_tangent_a_msoa_diagnostics();
    default:
      std::fprintf(stderr, "linear_elasticity_inexact_apply_tangent_a_3d_msoa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_objective_2d_soa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return linear_elasticity_proteus_quad4_objective_soa_diagnostics();
    case smesh::QUAD4:
      return linear_elasticity_quad4_objective_soa_diagnostics();
    case smesh::TRI3:
      return linear_elasticity_tri3_objective_soa_diagnostics();
    default:
      std::fprintf(stderr, "linear_elasticity_objective_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_objective_3d_soa_diagnostics(
    const smesh::ElemType element_type) {
  switch (element_type) {
    case smesh::HEX8:
      return linear_elasticity_hex8_objective_soa_diagnostics();
    case smesh::PROTEUS_HEX8:
      return linear_elasticity_proteus_hex8_objective_soa_diagnostics();
    case smesh::TET10:
      return linear_elasticity_tet10_objective_soa_diagnostics();
    case smesh::TET4:
      return linear_elasticity_tet4_objective_soa_diagnostics();
    default:
      std::fprintf(stderr, "linear_elasticity_objective_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
      return nullptr;
  }
}
