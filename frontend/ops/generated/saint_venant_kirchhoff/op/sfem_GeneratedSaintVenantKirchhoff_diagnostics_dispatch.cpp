#include "sfem_GeneratedSaintVenantKirchhoff_c_abi.hpp"
#include <cstdio>

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_proteus_quad4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_quad4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_tri3_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_tri6_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_hex27_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_hex8_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_proteus_hex27_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_proteus_hex64_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_proteus_hex8_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_tet10_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_tet4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_proteus_quad4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_quad4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_tri3_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_tri6_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_hex27_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_hex8_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_proteus_hex27_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_proteus_hex64_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_proteus_hex8_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_tet10_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_tet4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_proteus_quad4_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_quad4_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_tri3_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_tri6_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_hex27_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_hex8_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_proteus_hex27_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_proteus_hex64_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_proteus_hex8_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_tet10_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_tet4_objective_soa_diagnostics(void);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_apply_2d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return saint_venant_kirchhoff_proteus_quad4_apply_soa_diagnostics();
        case smesh::QUAD4:
            return saint_venant_kirchhoff_quad4_apply_soa_diagnostics();
        case smesh::TRI3:
            return saint_venant_kirchhoff_tri3_apply_soa_diagnostics();
        case smesh::TRI6:
            return saint_venant_kirchhoff_tri6_apply_soa_diagnostics();
        default:
            std::fprintf(stderr, "saint_venant_kirchhoff_apply_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_apply_3d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::HEX27:
            return saint_venant_kirchhoff_hex27_apply_soa_diagnostics();
        case smesh::HEX8:
            return saint_venant_kirchhoff_hex8_apply_soa_diagnostics();
        case smesh::PROTEUS_HEX27:
            return saint_venant_kirchhoff_proteus_hex27_apply_soa_diagnostics();
        case smesh::PROTEUS_HEX64:
            return saint_venant_kirchhoff_proteus_hex64_apply_soa_diagnostics();
        case smesh::PROTEUS_HEX8:
            return saint_venant_kirchhoff_proteus_hex8_apply_soa_diagnostics();
        case smesh::TET10:
            return saint_venant_kirchhoff_tet10_apply_soa_diagnostics();
        case smesh::TET4:
            return saint_venant_kirchhoff_tet4_apply_soa_diagnostics();
        default:
            std::fprintf(stderr, "saint_venant_kirchhoff_apply_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_gradient_2d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return saint_venant_kirchhoff_proteus_quad4_gradient_soa_diagnostics();
        case smesh::QUAD4:
            return saint_venant_kirchhoff_quad4_gradient_soa_diagnostics();
        case smesh::TRI3:
            return saint_venant_kirchhoff_tri3_gradient_soa_diagnostics();
        case smesh::TRI6:
            return saint_venant_kirchhoff_tri6_gradient_soa_diagnostics();
        default:
            std::fprintf(stderr, "saint_venant_kirchhoff_gradient_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_gradient_3d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::HEX27:
            return saint_venant_kirchhoff_hex27_gradient_soa_diagnostics();
        case smesh::HEX8:
            return saint_venant_kirchhoff_hex8_gradient_soa_diagnostics();
        case smesh::PROTEUS_HEX27:
            return saint_venant_kirchhoff_proteus_hex27_gradient_soa_diagnostics();
        case smesh::PROTEUS_HEX64:
            return saint_venant_kirchhoff_proteus_hex64_gradient_soa_diagnostics();
        case smesh::PROTEUS_HEX8:
            return saint_venant_kirchhoff_proteus_hex8_gradient_soa_diagnostics();
        case smesh::TET10:
            return saint_venant_kirchhoff_tet10_gradient_soa_diagnostics();
        case smesh::TET4:
            return saint_venant_kirchhoff_tet4_gradient_soa_diagnostics();
        default:
            std::fprintf(stderr, "saint_venant_kirchhoff_gradient_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_objective_2d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return saint_venant_kirchhoff_proteus_quad4_objective_soa_diagnostics();
        case smesh::QUAD4:
            return saint_venant_kirchhoff_quad4_objective_soa_diagnostics();
        case smesh::TRI3:
            return saint_venant_kirchhoff_tri3_objective_soa_diagnostics();
        case smesh::TRI6:
            return saint_venant_kirchhoff_tri6_objective_soa_diagnostics();
        default:
            std::fprintf(stderr, "saint_venant_kirchhoff_objective_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *saint_venant_kirchhoff_objective_3d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::HEX27:
            return saint_venant_kirchhoff_hex27_objective_soa_diagnostics();
        case smesh::HEX8:
            return saint_venant_kirchhoff_hex8_objective_soa_diagnostics();
        case smesh::PROTEUS_HEX27:
            return saint_venant_kirchhoff_proteus_hex27_objective_soa_diagnostics();
        case smesh::PROTEUS_HEX64:
            return saint_venant_kirchhoff_proteus_hex64_objective_soa_diagnostics();
        case smesh::PROTEUS_HEX8:
            return saint_venant_kirchhoff_proteus_hex8_objective_soa_diagnostics();
        case smesh::TET10:
            return saint_venant_kirchhoff_tet10_objective_soa_diagnostics();
        case smesh::TET4:
            return saint_venant_kirchhoff_tet4_objective_soa_diagnostics();
        default:
            std::fprintf(stderr, "saint_venant_kirchhoff_objective_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}
