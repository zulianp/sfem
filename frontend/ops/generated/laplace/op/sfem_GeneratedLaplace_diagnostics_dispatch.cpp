#include "sfem_GeneratedLaplace_c_abi.hpp"
#include <cstdio>

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_quad4_jacobian_action_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_quad4_jacobian_action_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri3_jacobian_action_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri6_jacobian_action_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex27_jacobian_action_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex8_jacobian_action_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex125_jacobian_action_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex27_jacobian_action_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex64_jacobian_action_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex729_jacobian_action_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex8_jacobian_action_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet10_jacobian_action_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet4_jacobian_action_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_quad4_residual_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_quad4_residual_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri3_residual_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri6_residual_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex27_residual_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex8_residual_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex125_residual_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex27_residual_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex64_residual_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex729_residual_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex8_residual_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet10_residual_element_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet4_residual_element_soa_diagnostics(void);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *laplace_jacobian_action_element_2d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_jacobian_action_element_soa_diagnostics();
        case smesh::QUAD4:
            return laplace_quad4_jacobian_action_element_soa_diagnostics();
        case smesh::TRI3:
            return laplace_tri3_jacobian_action_element_soa_diagnostics();
        case smesh::TRI6:
            return laplace_tri6_jacobian_action_element_soa_diagnostics();
        default:
            std::fprintf(stderr, "laplace_jacobian_action_element_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *laplace_jacobian_action_element_3d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_jacobian_action_element_soa_diagnostics();
        case smesh::HEX8:
            return laplace_hex8_jacobian_action_element_soa_diagnostics();
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_jacobian_action_element_soa_diagnostics();
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_jacobian_action_element_soa_diagnostics();
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_jacobian_action_element_soa_diagnostics();
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_jacobian_action_element_soa_diagnostics();
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_jacobian_action_element_soa_diagnostics();
        case smesh::TET10:
            return laplace_tet10_jacobian_action_element_soa_diagnostics();
        case smesh::TET4:
            return laplace_tet4_jacobian_action_element_soa_diagnostics();
        default:
            std::fprintf(stderr, "laplace_jacobian_action_element_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *laplace_residual_element_2d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_residual_element_soa_diagnostics();
        case smesh::QUAD4:
            return laplace_quad4_residual_element_soa_diagnostics();
        case smesh::TRI3:
            return laplace_tri3_residual_element_soa_diagnostics();
        case smesh::TRI6:
            return laplace_tri6_residual_element_soa_diagnostics();
        default:
            std::fprintf(stderr, "laplace_residual_element_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *laplace_residual_element_3d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::HEX27:
            return laplace_hex27_residual_element_soa_diagnostics();
        case smesh::HEX8:
            return laplace_hex8_residual_element_soa_diagnostics();
        case smesh::PROTEUS_HEX125:
            return laplace_proteus_hex125_residual_element_soa_diagnostics();
        case smesh::PROTEUS_HEX27:
            return laplace_proteus_hex27_residual_element_soa_diagnostics();
        case smesh::PROTEUS_HEX64:
            return laplace_proteus_hex64_residual_element_soa_diagnostics();
        case smesh::PROTEUS_HEX729:
            return laplace_proteus_hex729_residual_element_soa_diagnostics();
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_residual_element_soa_diagnostics();
        case smesh::TET10:
            return laplace_tet10_residual_element_soa_diagnostics();
        case smesh::TET4:
            return laplace_tet4_residual_element_soa_diagnostics();
        default:
            std::fprintf(stderr, "laplace_residual_element_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}
