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
#ifndef RSTR
#define RSTR SFEM_RESTRICT
#endif

#include "../../kernel_diagnostics.hpp"

#include "smesh_mesh.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_jacobian_action_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_jacobian_action_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int two_phase_flow_form_1_p_c_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_1_p_c_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_1_p_c_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_residual_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_residual_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_jacobian_action_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_jacobian_action_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int two_phase_flow_form_1_p_w_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_1_p_w_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_1_p_w_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_residual_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_residual_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_jacobian_action_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_jacobian_action_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_residual_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_residual_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_jacobian_action_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_jacobian_action_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_residual_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_residual_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_jacobian_action_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_jacobian_action_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_residual_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_residual_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_jacobian_action_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_jacobian_action_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_residual_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_residual_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int two_phase_flow_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_c,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
);

extern "C" int two_phase_flow_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_c,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_c,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
);

extern "C" int two_phase_flow_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_c,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_jacobian_action_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_jacobian_action_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int two_phase_flow_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_c,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output
);

extern "C" int two_phase_flow_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_c,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_adjugate4,
        const geom_t *const RSTR g_jacobian_adjugate5,
        const geom_t *const RSTR g_jacobian_adjugate6,
        const geom_t *const RSTR g_jacobian_adjugate7,
        const geom_t *const RSTR g_jacobian_adjugate8,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_c,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" int two_phase_flow_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output
);

extern "C" int two_phase_flow_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_c,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_residual_element_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_residual_element_3d_soa_diagnostics(
        const smesh::ElemType element_type);
