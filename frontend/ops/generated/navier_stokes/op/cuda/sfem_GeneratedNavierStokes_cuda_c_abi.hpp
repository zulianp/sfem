#pragma once

#include <cstddef>
#include <cstdint>

#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif

#ifndef SFEM_GENERATED_SCALAR_T
#define SFEM_GENERATED_SCALAR_T
typedef double real_t;
typedef ptrdiff_t idx_t;
typedef ptrdiff_t element_idx_t;
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif
#ifndef RSTR
#define RSTR SFEM_RESTRICT
#endif

#include "../../../cuda/kernel_diagnostics.cuh"

#include "smesh_mesh.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_1_p_jacobian_action_2d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_1_p_jacobian_action_3d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_form_1_p_residual_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[2],
        const void *const RSTR p_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_1_p_residual_2d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_form_1_p_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[2],
        const void *const RSTR p_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
);

extern "C" int cu_navier_stokes_form_1_p_residual_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[3],
        const void *const RSTR p_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_1_p_residual_3d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_form_1_p_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[3],
        const void *const RSTR p_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_1_u_jacobian_action_2d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_1_u_jacobian_action_3d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_form_1_u_residual_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[2],
        const void *const RSTR p_data,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[2],
        const void *const RSTR p_old_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_1_u_residual_2d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_form_1_u_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[2],
        const void *const RSTR p_data,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[2],
        const void *const RSTR p_old_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
);

extern "C" int cu_navier_stokes_form_1_u_residual_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t f2,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[3],
        const void *const RSTR p_data,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[3],
        const void *const RSTR p_old_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_1_u_residual_3d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_form_1_u_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t f2,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[3],
        const void *const RSTR p_data,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[3],
        const void *const RSTR p_old_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
);

extern "C" int cu_navier_stokes_form_2_p_u_jacobian_action_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_2_p_u_jacobian_action_2d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_form_2_p_u_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
);

extern "C" int cu_navier_stokes_form_2_p_u_jacobian_action_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_2_p_u_jacobian_action_3d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_form_2_p_u_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_2_p_u_residual_2d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_2_p_u_residual_3d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_form_2_u_p_jacobian_action_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_2_u_p_jacobian_action_2d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_form_2_u_p_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
);

extern "C" int cu_navier_stokes_form_2_u_p_jacobian_action_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_2_u_p_jacobian_action_3d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_form_2_u_p_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_2_u_p_residual_2d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_2_u_p_residual_3d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_form_2_u_u_jacobian_action_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[2],
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_2_u_u_jacobian_action_2d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_form_2_u_u_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[2],
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const stream
);

extern "C" int cu_navier_stokes_form_2_u_u_jacobian_action_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[3],
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_2_u_u_jacobian_action_3d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_form_2_u_u_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[3],
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_2_u_u_residual_2d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_2_u_u_residual_3d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_jacobian_action_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[2],
        const void *const RSTR p_old_data,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_jacobian_action_2d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[2],
        const void *const RSTR p_old_data,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
);

extern "C" int cu_navier_stokes_jacobian_action_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[3],
        const void *const RSTR p_old_data,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_jacobian_action_3d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[3],
        const void *const RSTR p_old_data,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
);

extern "C" int cu_navier_stokes_residual_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[2],
        const void *const RSTR p_data,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[2],
        const void *const RSTR p_old_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_residual_2d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[2],
        const void *const RSTR p_data,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[2],
        const void *const RSTR p_old_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
);

extern "C" int cu_navier_stokes_residual_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t f2,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[3],
        const void *const RSTR p_data,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[3],
        const void *const RSTR p_old_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_residual_3d_esoa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_navier_stokes_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t f2,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[3],
        const void *const RSTR p_data,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[3],
        const void *const RSTR p_old_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
);
