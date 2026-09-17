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
#include "../../matrix_formats.hpp"

#include "smesh_mesh.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_a_msoa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_soa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_a_msoa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_soa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_a_msoa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_soa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_a_msoa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_soa_diagnostics(
    const smesh::ElemType element_type);

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_matrix_assembly_variant(const int variant);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_matrix_assembly_variant_count();

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_objective_2d_a_msoa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        void *const RSTR value,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_objective_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        void *const RSTR value,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_newmark_elastic_objective_2d_soa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_objective_3d_a_msoa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        void *const RSTR value,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_objective_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        void *const RSTR value,
        void *const stream
);

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_newmark_elastic_objective_3d_soa_diagnostics(
    const smesh::ElemType element_type);

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_matrix_assembly_variant(const int variant);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_matrix_assembly_variant_count();

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_matrix_assembly_variant(const int variant);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_matrix_assembly_variant_count();

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_matrix_assembly_variant(const int variant);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_matrix_assembly_variant_count();

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_matrix_assembly_variant(const int variant);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_matrix_assembly_variant_count();

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_matrix_assembly_variant(const int variant);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_matrix_assembly_variant_count();

extern "C" void mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_matrix_assembly_variant(const int variant);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_matrix_assembly_variant_count();

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_hessian_bsr_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_hessian_bsr_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const void *const RSTR u2_old,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values,
        void *const stream
);

extern "C" void mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_variant(const int variant);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_matrix_assembly_variant_count();

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_a_msoa(
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
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        const void *const RSTR direction,
        void *const RSTR output,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_a_msoa(
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
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const void *const RSTR u2_old,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const void *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        const void *const RSTR direction,
        void *const RSTR output,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const void *const RSTR u2_old,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const void *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
);

extern "C" void mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_matrix_assembly_variant(const int variant);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_matrix_assembly_variant_count();

extern "C" void mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_matrix_assembly_variant(const int variant);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_matrix_assembly_variant_count();

extern "C" void mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_matrix_assembly_variant(const int variant);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_matrix_assembly_variant_count();

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_a_msoa(
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
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_a_msoa(
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
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const void *const RSTR u2_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output,
        void *const stream
);

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const void *const RSTR u2_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
);

extern "C" void mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_matrix_assembly_variant(const int variant);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_matrix_assembly_variant_count();

extern "C" void mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_matrix_assembly_variant(const int variant);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_matrix_assembly_variant_count();

extern "C" void mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_matrix_assembly_variant(const int variant);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_matrix_assembly_variant_count();
