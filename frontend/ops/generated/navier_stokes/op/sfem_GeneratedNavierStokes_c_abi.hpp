#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_CODEGEN_OP_HAS_SFEM_BASE
#endif
#endif

#ifndef SFEM_CODEGEN_OP_HAS_SFEM_BASE
typedef ptrdiff_t idx_t;
typedef ptrdiff_t element_idx_t;
typedef ptrdiff_t count_t;
typedef double real_t;
typedef double geom_t;
#endif

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#include "../../kernel_diagnostics.hpp"

#include "smesh_mesh.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_p_jacobian_action_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_p_jacobian_action_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int navier_stokes_form_1_p_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[2],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_p_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[2],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_p_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[3],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_p_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[3],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_p_residual_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_p_residual_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_u_jacobian_action_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_u_jacobian_action_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int navier_stokes_form_1_u_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[2],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[2],
        const void *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_u_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[2],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[2],
        const void *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_u_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t f2,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[3],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[3],
        const void *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_u_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t f2,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[3],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[3],
        const void *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_u_residual_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_u_residual_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int navier_stokes_form_2_p_u_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[2],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_p_u_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[2],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_p_u_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[3],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_p_u_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[3],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_p_u_jacobian_action_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_p_u_jacobian_action_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_p_u_residual_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_p_u_residual_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int navier_stokes_form_2_u_p_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[2],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_u_p_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[2],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_u_p_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[3],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_u_p_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[3],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_p_jacobian_action_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_p_jacobian_action_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_p_residual_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_p_residual_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int navier_stokes_form_2_u_u_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2]
);

extern "C" int navier_stokes_form_2_u_u_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2]
);

extern "C" int navier_stokes_form_2_u_u_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3]
);

extern "C" int navier_stokes_form_2_u_u_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3]
);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_u_jacobian_action_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_u_jacobian_action_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_u_residual_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_u_residual_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int navier_stokes_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[2],
        const void *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[2],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[2],
        const void *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[2],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[3],
        const void *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[3],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[3],
        const void *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[3],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_jacobian_action_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_jacobian_action_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int navier_stokes_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[2],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[2],
        const void *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[2],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[2],
        const void *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t f2,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[3],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[3],
        const void *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t f2,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[3],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[3],
        const void *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_residual_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_residual_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);
