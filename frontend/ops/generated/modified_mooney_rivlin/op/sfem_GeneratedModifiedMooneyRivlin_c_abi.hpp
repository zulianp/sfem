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

#include "smesh_mesh.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" int modified_mooney_rivlin_apply_2d_affine_mesh_soa(
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
);

extern "C" int modified_mooney_rivlin_apply_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
);

extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_apply_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int modified_mooney_rivlin_apply_3d_affine_mesh_soa(
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const void *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
);

extern "C" int modified_mooney_rivlin_apply_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const void *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
);

extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_apply_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int modified_mooney_rivlin_apply_packed_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
);

extern "C" int modified_mooney_rivlin_apply_packed_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
);

extern "C" int modified_mooney_rivlin_apply_packed_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const void *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
);

extern "C" int modified_mooney_rivlin_apply_packed_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const void *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
);

extern "C" int modified_mooney_rivlin_apply_packed_two_pass_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        void *const SFEM_RESTRICT ghost_buf,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
);

extern "C" int modified_mooney_rivlin_apply_packed_two_pass_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        void *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
);

extern "C" int modified_mooney_rivlin_apply_packed_two_pass_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        void *const SFEM_RESTRICT ghost_buf,
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const void *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
);

extern "C" int modified_mooney_rivlin_apply_packed_two_pass_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        void *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const void *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
);

extern "C" int modified_mooney_rivlin_gradient_2d_affine_mesh_soa(
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
);

extern "C" int modified_mooney_rivlin_gradient_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
);

extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_gradient_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int modified_mooney_rivlin_gradient_3d_affine_mesh_soa(
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
);

extern "C" int modified_mooney_rivlin_gradient_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
);

extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_gradient_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int modified_mooney_rivlin_gradient_packed_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
);

extern "C" int modified_mooney_rivlin_gradient_packed_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
);

extern "C" int modified_mooney_rivlin_gradient_packed_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
);

extern "C" int modified_mooney_rivlin_gradient_packed_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
);

extern "C" int modified_mooney_rivlin_gradient_packed_two_pass_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        void *const SFEM_RESTRICT ghost_buf,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
);

extern "C" int modified_mooney_rivlin_gradient_packed_two_pass_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        void *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
);

extern "C" int modified_mooney_rivlin_gradient_packed_two_pass_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        void *const SFEM_RESTRICT ghost_buf,
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
);

extern "C" int modified_mooney_rivlin_gradient_packed_two_pass_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        void *const SFEM_RESTRICT ghost_buf,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
);

extern "C" int modified_mooney_rivlin_hessian_bsr_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        void *const SFEM_RESTRICT values
);

extern "C" int modified_mooney_rivlin_hessian_bsr_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        void *const SFEM_RESTRICT values
);

extern "C" void modified_mooney_rivlin_hex27_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *modified_mooney_rivlin_hex27_matrix_assembly_variant(const int variant);

extern "C" int modified_mooney_rivlin_hex27_matrix_assembly_variant_count();

extern "C" void modified_mooney_rivlin_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *modified_mooney_rivlin_hex8_matrix_assembly_variant(const int variant);

extern "C" int modified_mooney_rivlin_hex8_matrix_assembly_variant_count();

extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_objective_2d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *modified_mooney_rivlin_objective_3d_soa_diagnostics(
        const smesh::ElemType element_type);

extern "C" int modified_mooney_rivlin_objective_steps_2d_affine_mesh_soa(
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const int nsteps,
        const void *const SFEM_RESTRICT steps,
        void *const SFEM_RESTRICT value
);

extern "C" int modified_mooney_rivlin_objective_steps_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const int nsteps,
        const void *const SFEM_RESTRICT steps,
        void *const SFEM_RESTRICT value
);

extern "C" int modified_mooney_rivlin_objective_steps_3d_affine_mesh_soa(
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const void *const SFEM_RESTRICT hz,
        const int nsteps,
        const void *const SFEM_RESTRICT steps,
        void *const SFEM_RESTRICT value
);

extern "C" int modified_mooney_rivlin_objective_steps_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const void *const SFEM_RESTRICT hz,
        const int nsteps,
        const void *const SFEM_RESTRICT steps,
        void *const SFEM_RESTRICT value
);

extern "C" int modified_mooney_rivlin_objective_steps_packed_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const int nsteps,
        const void *const SFEM_RESTRICT steps,
        void *const SFEM_RESTRICT value
);

extern "C" int modified_mooney_rivlin_objective_steps_packed_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const int nsteps,
        const void *const SFEM_RESTRICT steps,
        void *const SFEM_RESTRICT value
);

extern "C" int modified_mooney_rivlin_objective_steps_packed_3d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const void *const SFEM_RESTRICT hz,
        const int nsteps,
        const void *const SFEM_RESTRICT steps,
        void *const SFEM_RESTRICT value
);

extern "C" int modified_mooney_rivlin_objective_steps_packed_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const void *const SFEM_RESTRICT hz,
        const int nsteps,
        const void *const SFEM_RESTRICT steps,
        void *const SFEM_RESTRICT value
);

extern "C" void modified_mooney_rivlin_proteus_hex27_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *modified_mooney_rivlin_proteus_hex27_matrix_assembly_variant(const int variant);

extern "C" int modified_mooney_rivlin_proteus_hex27_matrix_assembly_variant_count();

extern "C" void modified_mooney_rivlin_proteus_hex64_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *modified_mooney_rivlin_proteus_hex64_matrix_assembly_variant(const int variant);

extern "C" int modified_mooney_rivlin_proteus_hex64_matrix_assembly_variant_count();

extern "C" void modified_mooney_rivlin_proteus_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *modified_mooney_rivlin_proteus_hex8_matrix_assembly_variant(const int variant);

extern "C" int modified_mooney_rivlin_proteus_hex8_matrix_assembly_variant_count();

extern "C" void modified_mooney_rivlin_proteus_quad4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *modified_mooney_rivlin_proteus_quad4_matrix_assembly_variant(const int variant);

extern "C" int modified_mooney_rivlin_proteus_quad4_matrix_assembly_variant_count();

extern "C" void modified_mooney_rivlin_quad4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *modified_mooney_rivlin_quad4_matrix_assembly_variant(const int variant);

extern "C" int modified_mooney_rivlin_quad4_matrix_assembly_variant_count();

extern "C" void modified_mooney_rivlin_tet10_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *modified_mooney_rivlin_tet10_matrix_assembly_variant(const int variant);

extern "C" int modified_mooney_rivlin_tet10_matrix_assembly_variant_count();

extern "C" void modified_mooney_rivlin_tet4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *modified_mooney_rivlin_tet4_matrix_assembly_variant(const int variant);

extern "C" int modified_mooney_rivlin_tet4_matrix_assembly_variant_count();

extern "C" void modified_mooney_rivlin_tri3_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *modified_mooney_rivlin_tri3_matrix_assembly_variant(const int variant);

extern "C" int modified_mooney_rivlin_tri3_matrix_assembly_variant_count();

extern "C" void modified_mooney_rivlin_tri6_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *modified_mooney_rivlin_tri6_matrix_assembly_variant(const int variant);

extern "C" int modified_mooney_rivlin_tri6_matrix_assembly_variant_count();
