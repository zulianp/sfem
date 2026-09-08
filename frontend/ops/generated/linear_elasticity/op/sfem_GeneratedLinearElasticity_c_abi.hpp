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
#include "../matrix_formats.hpp"

#include "smesh_mesh.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" int linear_elasticity_apply_2d_affine_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
);

extern "C" int linear_elasticity_apply_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
);

extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_apply_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int linear_elasticity_apply_3d_affine_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);

extern "C" int linear_elasticity_apply_3d_affine_mesh_soa_aos_unit(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate_aos,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t mu,
        const real_t lmbda,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);

extern "C" int linear_elasticity_apply_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);

extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_apply_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int linear_elasticity_apply_packed_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);

extern "C" int linear_elasticity_apply_packed_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);

extern "C" int linear_elasticity_apply_packed_two_pass_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const RSTR ghost_reduce_ptr,
        const ptrdiff_t *const RSTR ghost_reduce_idx,
        const idx_t *const RSTR ghost_reduce_dest,
        void *const RSTR ghost_buf,
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);

extern "C" int linear_elasticity_apply_packed_two_pass_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const RSTR ghost_reduce_ptr,
        const ptrdiff_t *const RSTR ghost_reduce_idx,
        const idx_t *const RSTR ghost_reduce_dest,
        void *const RSTR ghost_buf,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);

extern "C" int linear_elasticity_gradient_2d_affine_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
);

extern "C" int linear_elasticity_gradient_2d_isoparametric_mesh_soa(
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
        void *const RSTR outy
);

extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_gradient_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int linear_elasticity_gradient_3d_affine_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);

extern "C" int linear_elasticity_gradient_3d_affine_mesh_soa_aos_unit(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate_aos,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t mu,
        const real_t lmbda,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);

extern "C" int linear_elasticity_gradient_3d_isoparametric_mesh_soa(
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
        void *const RSTR outz
);

extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_gradient_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int linear_elasticity_gradient_packed_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);

extern "C" int linear_elasticity_gradient_packed_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
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
        void *const RSTR outz
);

extern "C" int linear_elasticity_gradient_packed_two_pass_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const RSTR ghost_reduce_ptr,
        const ptrdiff_t *const RSTR ghost_reduce_idx,
        const idx_t *const RSTR ghost_reduce_dest,
        void *const RSTR ghost_buf,
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);

extern "C" int linear_elasticity_gradient_packed_two_pass_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
        const ptrdiff_t n_ghost_entries,
        const ptrdiff_t n_ghost_reduce_rows,
        const ptrdiff_t *const RSTR ghost_reduce_ptr,
        const ptrdiff_t *const RSTR ghost_reduce_idx,
        const idx_t *const RSTR ghost_reduce_dest,
        void *const RSTR ghost_buf,
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
        void *const RSTR outz
);

extern "C" int linear_elasticity_hessian_block_diag_sym_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        void *const RSTR values
);

extern "C" int linear_elasticity_hessian_block_diag_sym_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        void *const RSTR values
);

extern "C" int linear_elasticity_hessian_bsr_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);

extern "C" int linear_elasticity_hessian_bsr_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);

extern "C" void linear_elasticity_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *linear_elasticity_hex8_matrix_assembly_variant(const int variant);

extern "C" int linear_elasticity_hex8_matrix_assembly_variant_count();

extern "C" int linear_elasticity_inexact_apply_compressed_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        const compressed_t *const RSTR tangent,
        const scaling_t *const RSTR scaling,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
);

extern "C" int linear_elasticity_inexact_apply_compressed_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        const compressed_t *const RSTR tangent,
        const scaling_t *const RSTR scaling,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);

extern "C" int linear_elasticity_inexact_apply_stored_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        const metric_tensor_t *const RSTR tangent,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
);

extern "C" int linear_elasticity_inexact_apply_stored_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        const metric_tensor_t *const RSTR tangent,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);

extern "C" int linear_elasticity_inexact_apply_tangent_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        metric_tensor_t *const RSTR tangent
);

extern "C" int linear_elasticity_inexact_apply_tangent_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t tangent_element_stride,
        const ptrdiff_t tangent_component_stride,
        metric_tensor_t *const RSTR tangent
);

extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_objective_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_objective_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int linear_elasticity_objective_steps_2d_affine_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);

extern "C" int linear_elasticity_objective_steps_2d_isoparametric_mesh_soa(
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
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);

extern "C" int linear_elasticity_objective_steps_3d_affine_mesh_soa(
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
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);

extern "C" int linear_elasticity_objective_steps_3d_isoparametric_mesh_soa(
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
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);

extern "C" int linear_elasticity_objective_steps_packed_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
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
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);

extern "C" int linear_elasticity_objective_steps_packed_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const RSTR elements,
        const ptrdiff_t *const RSTR owned_nodes_ptr,
        const ptrdiff_t *const RSTR n_shared_nodes,
        const ptrdiff_t *const RSTR ghost_ptr,
        const idx_t *const RSTR ghost_idx,
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
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);

extern "C" void linear_elasticity_proteus_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *linear_elasticity_proteus_hex8_matrix_assembly_variant(const int variant);

extern "C" int linear_elasticity_proteus_hex8_matrix_assembly_variant_count();

extern "C" void linear_elasticity_proteus_quad4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *linear_elasticity_proteus_quad4_matrix_assembly_variant(const int variant);

extern "C" int linear_elasticity_proteus_quad4_matrix_assembly_variant_count();

extern "C" void linear_elasticity_quad4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *linear_elasticity_quad4_matrix_assembly_variant(const int variant);

extern "C" int linear_elasticity_quad4_matrix_assembly_variant_count();

extern "C" void linear_elasticity_tet10_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *linear_elasticity_tet10_matrix_assembly_variant(const int variant);

extern "C" int linear_elasticity_tet10_matrix_assembly_variant_count();

extern "C" void linear_elasticity_tet4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *linear_elasticity_tet4_matrix_assembly_variant(const int variant);

extern "C" int linear_elasticity_tet4_matrix_assembly_variant_count();

extern "C" void linear_elasticity_tri3_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *linear_elasticity_tri3_matrix_assembly_variant(const int variant);

extern "C" int linear_elasticity_tri3_matrix_assembly_variant_count();
