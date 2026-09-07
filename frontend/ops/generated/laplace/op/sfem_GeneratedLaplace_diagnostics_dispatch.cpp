#include "sfem_GeneratedLaplace_c_abi.hpp"
#include <cstdio>

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_quad4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_quad4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri3_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex8_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex8_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet10_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet4_apply_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_quad4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_quad4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri3_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex8_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex8_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet10_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet4_gradient_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_quad4_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_quad4_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri3_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex8_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex8_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet10_objective_soa_diagnostics(void);
extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet4_objective_soa_diagnostics(void);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *laplace_apply_2d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_apply_soa_diagnostics();
        case smesh::QUAD4:
            return laplace_quad4_apply_soa_diagnostics();
        case smesh::TRI3:
            return laplace_tri3_apply_soa_diagnostics();
        default:
            std::fprintf(stderr, "laplace_apply_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *laplace_apply_3d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::HEX8:
            return laplace_hex8_apply_soa_diagnostics();
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_apply_soa_diagnostics();
        case smesh::TET10:
            return laplace_tet10_apply_soa_diagnostics();
        case smesh::TET4:
            return laplace_tet4_apply_soa_diagnostics();
        default:
            std::fprintf(stderr, "laplace_apply_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *laplace_gradient_2d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_gradient_soa_diagnostics();
        case smesh::QUAD4:
            return laplace_quad4_gradient_soa_diagnostics();
        case smesh::TRI3:
            return laplace_tri3_gradient_soa_diagnostics();
        default:
            std::fprintf(stderr, "laplace_gradient_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *laplace_gradient_3d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::HEX8:
            return laplace_hex8_gradient_soa_diagnostics();
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_gradient_soa_diagnostics();
        case smesh::TET10:
            return laplace_tet10_gradient_soa_diagnostics();
        case smesh::TET4:
            return laplace_tet4_gradient_soa_diagnostics();
        default:
            std::fprintf(stderr, "laplace_gradient_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *laplace_objective_2d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::PROTEUS_QUAD4:
            return laplace_proteus_quad4_objective_soa_diagnostics();
        case smesh::QUAD4:
            return laplace_quad4_objective_soa_diagnostics();
        case smesh::TRI3:
            return laplace_tri3_objective_soa_diagnostics();
        default:
            std::fprintf(stderr, "laplace_objective_2d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" const sfem::codegen::KernelDiagnostics *laplace_objective_3d_soa_diagnostics(
        const smesh::ElemType element_type) {
    switch (element_type) {
        case smesh::HEX8:
            return laplace_hex8_objective_soa_diagnostics();
        case smesh::PROTEUS_HEX8:
            return laplace_proteus_hex8_objective_soa_diagnostics();
        case smesh::TET10:
            return laplace_tet10_objective_soa_diagnostics();
        case smesh::TET4:
            return laplace_tet4_objective_soa_diagnostics();
        default:
            std::fprintf(stderr, "laplace_objective_3d_soa_diagnostics does not support element type %d\n", (int)element_type);
            return nullptr;
    }
}
