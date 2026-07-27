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

extern "C" void laplace_hex27_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex27_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_hex27_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex27_jacobian_action_element_soa_diagnostics(void);

extern "C" void laplace_hex27_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex27_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex27_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex27_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_hex27_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex27_jacobian_u_u_diagnostics(void);

extern "C" void laplace_hex27_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex27_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex27_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex27_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_hex27_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex27_residual_element_soa_diagnostics(void);

extern "C" void laplace_hex27_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex27_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex27_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex27_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void laplace_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_hex8_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex8_jacobian_u_u_diagnostics(void);

extern "C" void laplace_hex8_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex8_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_hex8_residual_element_soa_diagnostics(void);

extern "C" void laplace_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int laplace_jacobian_action_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_2d_affine_mesh_soa_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_2d_affine_mesh_soa_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_2d_affine_mesh_soa_aos_unit(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double *const SFEM_RESTRICT u_direction,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_2d_affine_mesh_soa_aos_unit_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const float *const SFEM_RESTRICT u_direction,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);

extern "C" int laplace_jacobian_action_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);

extern "C" int laplace_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_3d_affine_mesh_soa(
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
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_3d_affine_mesh_soa_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_3d_affine_mesh_soa_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_3d_affine_mesh_soa_aos_unit(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double *const SFEM_RESTRICT u_direction,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_3d_affine_mesh_soa_aos_unit_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const float *const SFEM_RESTRICT u_direction,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_3d_affine_mesh_soa_float(
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
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);

extern "C" int laplace_jacobian_action_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);

extern "C" int laplace_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_packed_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_packed_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_packed_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_packed_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_packed_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
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
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_packed_3d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
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
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_packed_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_packed_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_packed_two_pass_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        double *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_packed_two_pass_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        float *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_packed_two_pass_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        double *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_jacobian_action_packed_two_pass_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_ptr,
        const ptrdiff_t *const SFEM_RESTRICT ghost_reduce_idx,
        const idx_t *const SFEM_RESTRICT ghost_reduce_dest,
        float *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" void laplace_proteus_hex125_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex125_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex125_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex125_jacobian_action_element_soa_diagnostics(void);

extern "C" void laplace_proteus_hex125_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex125_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex125_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex125_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex125_jacobian_u_u_diagnostics(void);

extern "C" void laplace_proteus_hex125_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex125_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex125_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex125_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex125_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex125_residual_element_soa_diagnostics(void);

extern "C" void laplace_proteus_hex125_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex125_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex125_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex125_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex27_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex27_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex27_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex27_jacobian_action_element_soa_diagnostics(void);

extern "C" void laplace_proteus_hex27_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex27_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex27_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex27_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex27_jacobian_u_u_diagnostics(void);

extern "C" void laplace_proteus_hex27_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex27_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex27_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex27_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex27_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex27_residual_element_soa_diagnostics(void);

extern "C" void laplace_proteus_hex27_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex27_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex27_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex27_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex64_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex64_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex64_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex64_jacobian_action_element_soa_diagnostics(void);

extern "C" void laplace_proteus_hex64_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex64_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex64_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex64_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex64_jacobian_u_u_diagnostics(void);

extern "C" void laplace_proteus_hex64_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex64_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex64_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex64_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex64_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex64_residual_element_soa_diagnostics(void);

extern "C" void laplace_proteus_hex64_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex64_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex64_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex64_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex729_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex729_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex729_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex729_jacobian_action_element_soa_diagnostics(void);

extern "C" void laplace_proteus_hex729_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex729_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex729_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex729_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex729_jacobian_u_u_diagnostics(void);

extern "C" void laplace_proteus_hex729_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex729_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex729_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex729_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex729_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex729_residual_element_soa_diagnostics(void);

extern "C" void laplace_proteus_hex729_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex729_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex729_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex729_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex8_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex8_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex8_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex8_jacobian_action_element_soa_diagnostics(void);

extern "C" void laplace_proteus_hex8_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex8_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex8_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex8_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex8_jacobian_u_u_diagnostics(void);

extern "C" void laplace_proteus_hex8_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex8_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex8_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex8_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_hex8_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_hex8_residual_element_soa_diagnostics(void);

extern "C" void laplace_proteus_hex8_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex8_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex8_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_hex8_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void laplace_proteus_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_quad4_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_quad4_jacobian_u_u_diagnostics(void);

extern "C" void laplace_proteus_quad4_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_quad4_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_proteus_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_proteus_quad4_residual_element_soa_diagnostics(void);

extern "C" void laplace_proteus_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_proteus_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_quad4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_quad4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_quad4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_quad4_jacobian_action_element_soa_diagnostics(void);

extern "C" void laplace_quad4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_quad4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_quad4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_quad4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_quad4_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_quad4_jacobian_u_u_diagnostics(void);

extern "C" void laplace_quad4_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_quad4_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_quad4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_quad4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_quad4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_quad4_residual_element_soa_diagnostics(void);

extern "C" void laplace_quad4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_quad4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_quad4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_quad4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" int laplace_residual_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_2d_affine_mesh_soa_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_2d_affine_mesh_soa_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_2d_affine_mesh_soa_aos_unit(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double *const SFEM_RESTRICT u,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_2d_affine_mesh_soa_aos_unit_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const float *const SFEM_RESTRICT u,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);

extern "C" int laplace_residual_2d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);

extern "C" int laplace_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_3d_affine_mesh_soa(
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
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_3d_affine_mesh_soa_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_3d_affine_mesh_soa_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_3d_affine_mesh_soa_aos_unit(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const double *const SFEM_RESTRICT u,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_3d_affine_mesh_soa_aos_unit_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_geom_metric,
        const float *const SFEM_RESTRICT u,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_3d_affine_mesh_soa_float(
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
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        double *const SFEM_RESTRICT output
);

extern "C" int laplace_residual_3d_isoparametric_mesh_aos_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        float *const SFEM_RESTRICT output
);

extern "C" int laplace_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double kappa,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out
);

extern "C" int laplace_residual_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float kappa,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out
);

extern "C" void laplace_tet10_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet10_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_tet10_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet10_jacobian_action_element_soa_diagnostics(void);

extern "C" void laplace_tet10_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet10_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet10_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet10_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_tet10_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet10_jacobian_u_u_diagnostics(void);

extern "C" void laplace_tet10_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet10_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet10_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet10_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_tet10_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet10_residual_element_soa_diagnostics(void);

extern "C" void laplace_tet10_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet10_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet10_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet10_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet4_jacobian_action_element_soa_diagnostics(void);

extern "C" void laplace_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_tet4_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet4_jacobian_u_u_diagnostics(void);

extern "C" void laplace_tet4_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet4_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tet4_residual_element_soa_diagnostics(void);

extern "C" void laplace_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri3_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri3_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_tri3_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri3_jacobian_action_element_soa_diagnostics(void);

extern "C" void laplace_tri3_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri3_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri3_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri3_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_tri3_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri3_jacobian_u_u_diagnostics(void);

extern "C" void laplace_tri3_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri3_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri3_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri3_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_tri3_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri3_residual_element_soa_diagnostics(void);

extern "C" void laplace_tri3_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri3_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri3_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri3_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri6_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri6_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_tri6_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri6_jacobian_action_element_soa_diagnostics(void);

extern "C" void laplace_tri6_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri6_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri6_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri6_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_tri6_jacobian_u_u_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri6_jacobian_u_u_diagnostics(void);

extern "C" void laplace_tri6_jacobian_u_u_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri6_jacobian_u_u_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri6_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri6_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" double laplace_tri6_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_tri6_residual_element_soa_diagnostics(void);

extern "C" void laplace_tri6_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri6_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri6_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);

extern "C" void laplace_tri6_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs);
