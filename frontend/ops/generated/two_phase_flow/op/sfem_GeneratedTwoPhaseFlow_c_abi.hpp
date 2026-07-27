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

extern "C" void two_phase_flow_form_1_p_c_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_hex8_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_hex8_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_hex8_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_hex8_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_hex8_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_jacobian_action_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_proteus_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_proteus_hex8_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_proteus_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_proteus_quad4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_quad4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int two_phase_flow_form_1_p_c_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_residual_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_c_residual_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_c_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_residual_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
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
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_residual_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
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
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_c_residual_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_c_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_c_residual_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" void two_phase_flow_form_1_p_c_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_tet4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_tet4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_tet4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_tet4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tet4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_tet4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_tet4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_tet4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tet4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_tet4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_tet4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_tet4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tet4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_tet4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_tet4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_tet4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tet4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_tet4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_tri3_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_tri3_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_tri3_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_tri3_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tri3_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_tri3_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_tri3_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_tri3_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tri3_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_tri3_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_tri3_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_tri3_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tri3_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_tri3_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_tri3_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_tri3_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tri3_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_c_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_c_tri3_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_c_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_c_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_hex8_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_jacobian_action_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_proteus_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_proteus_hex8_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_proteus_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_proteus_quad4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_quad4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int two_phase_flow_form_1_p_w_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_residual_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_w_residual_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_w_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_residual_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
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
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_residual_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
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
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_w_residual_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_1_p_w_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_1_p_w_residual_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" void two_phase_flow_form_1_p_w_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_tet4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_tet4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_tet4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_tet4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tet4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_tet4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_tet4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_tet4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tet4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_tet4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_tet4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_tet4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tet4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_tet4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_tet4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_tet4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tet4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_tet4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_tri3_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_tri3_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_tri3_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_tri3_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tri3_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_tri3_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_tri3_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_tri3_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tri3_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_tri3_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_tri3_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_tri3_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tri3_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_tri3_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_tri3_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_tri3_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tri3_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_1_p_w_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_1_p_w_tri3_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_1_p_w_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_1_p_w_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_hex8_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
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
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
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
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_quad4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_c_residual_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tet4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tet4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_c_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_hex8_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
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
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
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
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_quad4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_c_p_w_residual_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tet4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tet4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_c_p_w_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_hex8_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
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
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
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
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_quad4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_c_residual_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_tet4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_tet4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_c_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_c_tri3_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_c_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_hex8_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
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
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
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
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_quad4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_form_2_p_w_p_w_residual_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tri3_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_form_2_p_w_p_w_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_form_2_p_w_p_w_tri3_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_form_2_p_w_p_w_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_hex8_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_hex8_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_hex8_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_hex8_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_hex8_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int two_phase_flow_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_jacobian_action_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_jacobian_action_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_jacobian_action_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_jacobian_action_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
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
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_jacobian_action_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
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
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_jacobian_action_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_jacobian_action_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" void two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_proteus_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_proteus_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_proteus_hex8_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_hex8_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_proteus_hex8_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_hex8_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_proteus_hex8_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_hex8_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_proteus_hex8_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_hex8_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_hex8_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_proteus_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_hex8_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_proteus_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_proteus_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_proteus_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_quad4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_proteus_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_proteus_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_quad4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_proteus_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_proteus_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_quad4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_proteus_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_proteus_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_quad4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_proteus_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_proteus_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_proteus_quad4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_proteus_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_quad4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_quad4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_quad4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_quad4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_quad4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_quad4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_quad4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_quad4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_quad4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_quad4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_quad4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_quad4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_quad4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_quad4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_quad4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_quad4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_quad4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int two_phase_flow_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_residual_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_residual_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_residual_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_residual_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
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
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_residual_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
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
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_residual_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);

extern "C" int two_phase_flow_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);

extern "C" int two_phase_flow_residual_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

extern "C" void two_phase_flow_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tet4_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_tet4_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tet4_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_tet4_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tet4_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_tet4_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tet4_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_tet4_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tet4_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_tet4_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tet4_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_tet4_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tet4_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_tet4_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tet4_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_tet4_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tet4_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tet4_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tri3_jacobian_action_element_soa_diagnostics(void);

extern "C" void two_phase_flow_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_tri3_jacobian_p_c_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tri3_jacobian_p_c_p_c_diagnostics(void);

extern "C" void two_phase_flow_tri3_jacobian_p_c_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tri3_jacobian_p_c_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_tri3_jacobian_p_c_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tri3_jacobian_p_c_p_w_diagnostics(void);

extern "C" void two_phase_flow_tri3_jacobian_p_c_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tri3_jacobian_p_c_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_tri3_jacobian_p_w_p_c_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tri3_jacobian_p_w_p_c_diagnostics(void);

extern "C" void two_phase_flow_tri3_jacobian_p_w_p_c_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tri3_jacobian_p_w_p_c_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_tri3_jacobian_p_w_p_w_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tri3_jacobian_p_w_p_w_diagnostics(void);

extern "C" void two_phase_flow_tri3_jacobian_p_w_p_w_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tri3_jacobian_p_w_p_w_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double two_phase_flow_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *two_phase_flow_tri3_residual_element_soa_diagnostics(void);

extern "C" void two_phase_flow_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void two_phase_flow_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);
