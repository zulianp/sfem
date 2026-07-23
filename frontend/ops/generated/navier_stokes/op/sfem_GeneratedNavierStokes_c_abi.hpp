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
#include "../matrix_formats.hpp"


extern "C" void navier_stokes_form_1_p_hex27_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_hex27_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_1_p_hex27_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_p_hex27_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_1_p_hex27_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_hex27_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_hex27_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_1_p_hex27_hex8_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_1_p_hex27_hex8_matrix_assembly_variant_count();

extern "C" int navier_stokes_form_1_p_hex27_hex8_residual_affine_mesh_soa(
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
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_p_hex27_hex8_residual_affine_mesh_soa_float(
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
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_1_p_hex27_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_hex27_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_1_p_hex27_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_p_hex27_hex8_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_1_p_hex27_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_hex27_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_1_p_hex27_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_p_hex27_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_1_p_hex27_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_hex27_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tet10_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tet10_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tet10_tet4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_1_p_tet10_tet4_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_1_p_tet10_tet4_matrix_assembly_variant_count();

extern "C" int navier_stokes_form_1_p_tet10_tet4_residual_affine_mesh_soa(
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
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_p_tet10_tet4_residual_affine_mesh_soa_float(
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
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_1_p_tet10_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tet10_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_1_p_tet10_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_p_tet10_tet4_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_1_p_tet10_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tet10_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tri6_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tri6_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_1_p_tri6_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_p_tri6_tri3_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_1_p_tri6_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tri6_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tri6_tri3_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_1_p_tri6_tri3_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_1_p_tri6_tri3_matrix_assembly_variant_count();

extern "C" int navier_stokes_form_1_p_tri6_tri3_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[2],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_p_tri6_tri3_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[2],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_1_p_tri6_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tri6_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_1_p_tri6_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_p_tri6_tri3_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_1_p_tri6_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tri6_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_1_p_tri6_tri3_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[2],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_p_tri6_tri3_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[2],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_1_p_tri6_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_p_tri6_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_hex27_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_hex27_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_1_u_hex27_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_u_hex27_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_1_u_hex27_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_hex27_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_hex27_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_1_u_hex27_hex8_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_1_u_hex27_hex8_matrix_assembly_variant_count();

extern "C" int navier_stokes_form_1_u_hex27_hex8_residual_affine_mesh_soa(
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
        const double convection_scale,
        const double dt,
        const double f0,
        const double f1,
        const double f2,
        const double nu,
        const double rho,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_u_hex27_hex8_residual_affine_mesh_soa_float(
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
        const float convection_scale,
        const float dt,
        const float f0,
        const float f1,
        const float f2,
        const float nu,
        const float rho,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_1_u_hex27_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_hex27_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_1_u_hex27_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_u_hex27_hex8_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_1_u_hex27_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_hex27_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double f0,
        const double f1,
        const double f2,
        const double nu,
        const double rho,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float f0,
        const float f1,
        const float f2,
        const float nu,
        const float rho,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tet10_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tet10_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_1_u_tet10_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_u_tet10_tet4_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_1_u_tet10_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tet10_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tet10_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tet10_tet4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_1_u_tet10_tet4_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_1_u_tet10_tet4_matrix_assembly_variant_count();

extern "C" int navier_stokes_form_1_u_tet10_tet4_residual_affine_mesh_soa(
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
        const double convection_scale,
        const double dt,
        const double f0,
        const double f1,
        const double f2,
        const double nu,
        const double rho,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_u_tet10_tet4_residual_affine_mesh_soa_float(
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
        const float convection_scale,
        const float dt,
        const float f0,
        const float f1,
        const float f2,
        const float nu,
        const float rho,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_1_u_tet10_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tet10_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_1_u_tet10_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_u_tet10_tet4_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_1_u_tet10_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tet10_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_1_u_tet10_tet4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double f0,
        const double f1,
        const double f2,
        const double nu,
        const double rho,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_u_tet10_tet4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float f0,
        const float f1,
        const float f2,
        const float nu,
        const float rho,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_1_u_tet10_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tet10_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tri6_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tri6_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tri6_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tri6_tri3_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_1_u_tri6_tri3_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_1_u_tri6_tri3_matrix_assembly_variant_count();

extern "C" int navier_stokes_form_1_u_tri6_tri3_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double convection_scale,
        const double dt,
        const double f0,
        const double f1,
        const double nu,
        const double rho,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[2],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_u_tri6_tri3_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float convection_scale,
        const float dt,
        const float f0,
        const float f1,
        const float nu,
        const float rho,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[2],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_1_u_tri6_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tri6_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_1_u_tri6_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_u_tri6_tri3_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_1_u_tri6_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tri6_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double f0,
        const double f1,
        const double nu,
        const double rho,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[2],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float f0,
        const float f1,
        const float nu,
        const float rho,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[2],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_p_u_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_p_u_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_affine_mesh_soa(
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
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_2_p_u_hex27_hex8_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_2_p_u_hex27_hex8_matrix_assembly_variant_count();

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_p_u_hex27_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_p_u_hex27_hex8_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_hex27_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_p_u_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_p_u_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_affine_mesh_soa(
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
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_affine_mesh_soa_float(
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
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tet10_tet4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_2_p_u_tet10_tet4_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_2_p_u_tet10_tet4_matrix_assembly_variant_count();

extern "C" void navier_stokes_form_2_p_u_tet10_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tet10_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_p_u_tet10_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_p_u_tet10_tet4_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_p_u_tet10_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tet10_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tet10_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tet10_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_p_u_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_p_u_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tri6_tri3_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_2_p_u_tri6_tri3_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_2_p_u_tri6_tri3_matrix_assembly_variant_count();

extern "C" void navier_stokes_form_2_p_u_tri6_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tri6_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_p_u_tri6_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_p_u_tri6_tri3_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_p_u_tri6_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tri6_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tri6_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_p_u_tri6_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_soa(
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
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_2_u_p_hex27_hex8_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_2_u_p_hex27_hex8_matrix_assembly_variant_count();

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_hex27_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_affine_mesh_soa(
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
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_affine_mesh_soa_float(
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
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_2_u_p_tet10_tet4_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_2_u_p_tet10_tet4_matrix_assembly_variant_count();

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_u_p_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_u_p_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tri6_tri3_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_2_u_p_tri6_tri3_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_2_u_p_tri6_tri3_matrix_assembly_variant_count();

extern "C" void navier_stokes_form_2_u_p_tri6_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tri6_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_u_p_tri6_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_p_tri6_tri3_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_u_p_tri6_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tri6_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tri6_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_p_tri6_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_u_u_hex27_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_u_u_hex27_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_u_u_hex27_jacobian_action_affine_mesh_soa(
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
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3]
);

extern "C" int navier_stokes_form_2_u_u_hex27_jacobian_action_affine_mesh_soa_float(
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
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3]
);

extern "C" void navier_stokes_form_2_u_u_hex27_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_hex27_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_hex27_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3]
);

extern "C" int navier_stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3]
);

extern "C" void navier_stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_hex27_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_2_u_u_hex27_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_2_u_u_hex27_matrix_assembly_variant_count();

extern "C" void navier_stokes_form_2_u_u_hex27_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_hex27_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_u_u_hex27_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_u_hex27_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_u_u_hex27_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_hex27_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_hex27_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_hex27_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_u_u_tet10_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_u_u_tet10_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_u_u_tet10_jacobian_action_affine_mesh_soa(
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
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3]
);

extern "C" int navier_stokes_form_2_u_u_tet10_jacobian_action_affine_mesh_soa_float(
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
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3]
);

extern "C" void navier_stokes_form_2_u_u_tet10_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tet10_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_u_u_tet10_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_u_tet10_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_u_u_tet10_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tet10_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_u_u_tet10_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3]
);

extern "C" int navier_stokes_form_2_u_u_tet10_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3]
);

extern "C" void navier_stokes_form_2_u_u_tet10_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tet10_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tet10_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_2_u_u_tet10_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_2_u_u_tet10_matrix_assembly_variant_count();

extern "C" void navier_stokes_form_2_u_u_tet10_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tet10_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_u_u_tet10_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_u_tet10_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_u_u_tet10_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tet10_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tet10_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tet10_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_u_u_tri6_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_u_u_tri6_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_form_2_u_u_tri6_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2]
);

extern "C" int navier_stokes_form_2_u_u_tri6_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2]
);

extern "C" void navier_stokes_form_2_u_u_tri6_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tri6_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tri6_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2]
);

extern "C" int navier_stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2]
);

extern "C" void navier_stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tri6_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_form_2_u_u_tri6_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_form_2_u_u_tri6_matrix_assembly_variant_count();

extern "C" void navier_stokes_form_2_u_u_tri6_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tri6_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_form_2_u_u_tri6_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_u_tri6_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_form_2_u_u_tri6_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tri6_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tri6_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_form_2_u_u_tri6_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_hex27_hex8_jacobian_action_affine_mesh_soa(
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
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_hex27_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_hex27_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_hex27_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_hex27_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_hex27_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_hex27_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_hex27_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_hex27_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_hex27_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_hex27_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_hex27_hex8_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_hex27_hex8_matrix_assembly_variant_count();

extern "C" int navier_stokes_hex27_hex8_residual_affine_mesh_soa(
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
        const double convection_scale,
        const double dt,
        const double f0,
        const double f1,
        const double f2,
        const double nu,
        const double rho,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_hex27_hex8_residual_affine_mesh_soa_float(
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
        const float convection_scale,
        const float dt,
        const float f0,
        const float f1,
        const float f2,
        const float nu,
        const float rho,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_hex27_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_hex27_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_hex27_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_hex27_hex8_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_hex27_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_hex27_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_hex27_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double f0,
        const double f1,
        const double f2,
        const double nu,
        const double rho,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_hex27_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float f0,
        const float f1,
        const float f2,
        const float nu,
        const float rho,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_hex27_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_hex27_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_tet10_tet4_jacobian_action_affine_mesh_soa(
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
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_tet10_tet4_jacobian_action_affine_mesh_soa_float(
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
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_tet10_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_tet10_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_tet10_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_tet10_tet4_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_tet10_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_tet10_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_tet10_tet4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_tet10_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_tet10_tet4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_tet10_tet4_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_tet10_tet4_matrix_assembly_variant_count();

extern "C" int navier_stokes_tet10_tet4_residual_affine_mesh_soa(
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
        const double convection_scale,
        const double dt,
        const double f0,
        const double f1,
        const double f2,
        const double nu,
        const double rho,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_tet10_tet4_residual_affine_mesh_soa_float(
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
        const float convection_scale,
        const float dt,
        const float f0,
        const float f1,
        const float f2,
        const float nu,
        const float rho,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_tet10_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_tet10_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_tet10_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_tet10_tet4_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_tet10_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_tet10_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_tet10_tet4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double f0,
        const double f1,
        const double f2,
        const double nu,
        const double rho,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_tet10_tet4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float f0,
        const float f1,
        const float f2,
        const float nu,
        const float rho,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_tet10_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_tet10_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);

extern "C" int navier_stokes_tri6_tri3_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_tri6_tri3_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_tri6_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_tri6_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_tri6_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_tri6_tri3_jacobian_action_element_soa_diagnostics(void);

extern "C" void navier_stokes_tri6_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_tri6_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_tri6_tri3_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double nu,
        const double rho,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float nu,
        const float rho,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_tri6_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_tri6_tri3_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *navier_stokes_tri6_tri3_matrix_assembly_variant(const int variant);

extern "C" int navier_stokes_tri6_tri3_matrix_assembly_variant_count();

extern "C" int navier_stokes_tri6_tri3_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double convection_scale,
        const double dt,
        const double f0,
        const double f1,
        const double nu,
        const double rho,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[2],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_tri6_tri3_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float convection_scale,
        const float dt,
        const float f0,
        const float f1,
        const float nu,
        const float rho,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[2],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_tri6_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_tri6_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double navier_stokes_tri6_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_tri6_tri3_residual_element_soa_diagnostics(void);

extern "C" void navier_stokes_tri6_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_tri6_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int navier_stokes_tri6_tri3_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double convection_scale,
        const double dt,
        const double f0,
        const double f1,
        const double nu,
        const double rho,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[2],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);

extern "C" int navier_stokes_tri6_tri3_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float convection_scale,
        const float dt,
        const float f0,
        const float f1,
        const float nu,
        const float rho,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[2],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);

extern "C" void navier_stokes_tri6_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void navier_stokes_tri6_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);
