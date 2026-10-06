#include "sfem_GeneratedModifiedMooneyRivlin_c_abi.hpp"
#include <cstdio>

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_quad4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_quad4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_tri3_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_tri6_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_hex27_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_hex8_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_hex27_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_hex64_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_hex8_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_tet10_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_tet4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_quad4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_quad4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_tri3_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_tri6_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_hex27_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_hex8_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_hex27_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_hex64_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_hex8_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_tet10_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_tet4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_quad4_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_quad4_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_tri3_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_tri6_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_hex27_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_hex8_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_hex27_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_hex64_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_proteus_hex8_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_tet10_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_tet4_objective_soa_diagnostics(void);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_apply_2d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return modified_mooney_rivlin_proteus_quad4_apply_soa_diagnostics();
        case smesh::QUAD4:
            return modified_mooney_rivlin_quad4_apply_soa_diagnostics();
        case smesh::TRI3:
            return modified_mooney_rivlin_tri3_apply_soa_diagnostics();
        case smesh::TRI6:
            return modified_mooney_rivlin_tri6_apply_soa_diagnostics();
        default:
            std::fprintf(stderr, "modified_mooney_rivlin_apply_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_apply_3d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::HEX27:
            return modified_mooney_rivlin_hex27_apply_soa_diagnostics();
        case smesh::HEX8:
            return modified_mooney_rivlin_hex8_apply_soa_diagnostics();
        case smesh::PROTEUS_HEX27:
            return modified_mooney_rivlin_proteus_hex27_apply_soa_diagnostics();
        case smesh::PROTEUS_HEX64:
            return modified_mooney_rivlin_proteus_hex64_apply_soa_diagnostics();
        case smesh::PROTEUS_HEX8:
            return modified_mooney_rivlin_proteus_hex8_apply_soa_diagnostics();
        case smesh::TET10:
            return modified_mooney_rivlin_tet10_apply_soa_diagnostics();
        case smesh::TET4:
            return modified_mooney_rivlin_tet4_apply_soa_diagnostics();
        default:
            std::fprintf(stderr, "modified_mooney_rivlin_apply_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_gradient_2d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return modified_mooney_rivlin_proteus_quad4_gradient_soa_diagnostics();
        case smesh::QUAD4:
            return modified_mooney_rivlin_quad4_gradient_soa_diagnostics();
        case smesh::TRI3:
            return modified_mooney_rivlin_tri3_gradient_soa_diagnostics();
        case smesh::TRI6:
            return modified_mooney_rivlin_tri6_gradient_soa_diagnostics();
        default:
            std::fprintf(stderr, "modified_mooney_rivlin_gradient_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_gradient_3d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::HEX27:
            return modified_mooney_rivlin_hex27_gradient_soa_diagnostics();
        case smesh::HEX8:
            return modified_mooney_rivlin_hex8_gradient_soa_diagnostics();
        case smesh::PROTEUS_HEX27:
            return modified_mooney_rivlin_proteus_hex27_gradient_soa_diagnostics();
        case smesh::PROTEUS_HEX64:
            return modified_mooney_rivlin_proteus_hex64_gradient_soa_diagnostics();
        case smesh::PROTEUS_HEX8:
            return modified_mooney_rivlin_proteus_hex8_gradient_soa_diagnostics();
        case smesh::TET10:
            return modified_mooney_rivlin_tet10_gradient_soa_diagnostics();
        case smesh::TET4:
            return modified_mooney_rivlin_tet4_gradient_soa_diagnostics();
        default:
            std::fprintf(stderr, "modified_mooney_rivlin_gradient_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_objective_2d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return modified_mooney_rivlin_proteus_quad4_objective_soa_diagnostics();
        case smesh::QUAD4:
            return modified_mooney_rivlin_quad4_objective_soa_diagnostics();
        case smesh::TRI3:
            return modified_mooney_rivlin_tri3_objective_soa_diagnostics();
        case smesh::TRI6:
            return modified_mooney_rivlin_tri6_objective_soa_diagnostics();
        default:
            std::fprintf(stderr, "modified_mooney_rivlin_objective_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_objective_3d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::HEX27:
            return modified_mooney_rivlin_hex27_objective_soa_diagnostics();
        case smesh::HEX8:
            return modified_mooney_rivlin_hex8_objective_soa_diagnostics();
        case smesh::PROTEUS_HEX27:
            return modified_mooney_rivlin_proteus_hex27_objective_soa_diagnostics();
        case smesh::PROTEUS_HEX64:
            return modified_mooney_rivlin_proteus_hex64_objective_soa_diagnostics();
        case smesh::PROTEUS_HEX8:
            return modified_mooney_rivlin_proteus_hex8_objective_soa_diagnostics();
        case smesh::TET10:
            return modified_mooney_rivlin_tet10_objective_soa_diagnostics();
        case smesh::TET4:
            return modified_mooney_rivlin_tet4_objective_soa_diagnostics();
        default:
            std::fprintf(stderr, "modified_mooney_rivlin_objective_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}
