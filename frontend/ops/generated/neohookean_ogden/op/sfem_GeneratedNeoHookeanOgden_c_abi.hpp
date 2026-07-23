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

extern "C" int neohookean_ogden_apply_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_apply_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_apply_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_apply_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_apply_3d_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_apply_3d_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_apply_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_apply_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_apply_packed_2d_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_apply_packed_2d_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_apply_packed_2d_isoparametric_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_apply_packed_2d_isoparametric_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_apply_packed_3d_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_apply_packed_3d_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_apply_packed_3d_isoparametric_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_apply_packed_3d_isoparametric_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_gradient_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_gradient_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_gradient_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_gradient_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_gradient_3d_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_gradient_3d_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_gradient_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_gradient_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_gradient_packed_2d_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_gradient_packed_2d_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_gradient_packed_2d_isoparametric_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_gradient_packed_2d_isoparametric_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);

extern "C" int neohookean_ogden_gradient_packed_3d_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_gradient_packed_3d_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_gradient_packed_3d_isoparametric_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
);

extern "C" int neohookean_ogden_gradient_packed_3d_isoparametric_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
);

extern "C" void neohookean_ogden_hex27_hex27_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex27_hex27_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex27_hex27_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_hex27_hex27_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_hex27_hex27_apply_soa_diagnostics(void);

extern "C" void neohookean_ogden_hex27_hex27_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex27_hex27_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex27_hex27_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex27_hex27_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex27_hex27_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_hex27_hex27_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_hex27_hex27_gradient_soa_diagnostics(void);

extern "C" void neohookean_ogden_hex27_hex27_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex27_hex27_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex27_hex27_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex27_hex27_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex27_hex27_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_hex27_hex27_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_hex27_hex27_objective_soa_diagnostics(void);

extern "C" void neohookean_ogden_hex27_hex27_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex27_hex27_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_hex8_hex8_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_hex8_hex8_apply_soa_diagnostics(void);

extern "C" void neohookean_ogden_hex8_hex8_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_hex8_hex8_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_hex8_hex8_gradient_soa_diagnostics(void);

extern "C" void neohookean_ogden_hex8_hex8_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_hex8_hex8_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_hex8_hex8_objective_soa_diagnostics(void);

extern "C" void neohookean_ogden_hex8_hex8_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_hex8_hex8_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" int neohookean_ogden_objective_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        double *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        float *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        double *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        float *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_3d_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        double *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_3d_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        float *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        double *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        float *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_2d_affine_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_2d_affine_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_2d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_3d_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_3d_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_3d_isoparametric_mesh_soa_float(
        const smesh::ElemType element_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_packed_2d_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_packed_2d_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_packed_2d_isoparametric_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_packed_2d_isoparametric_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_packed_3d_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_packed_3d_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_packed_3d_isoparametric_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);

extern "C" int neohookean_ogden_objective_steps_packed_3d_isoparametric_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex125_proteus_hex125_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex125_proteus_hex125_apply_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex125_proteus_hex125_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex125_proteus_hex125_gradient_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex125_proteus_hex125_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex125_proteus_hex125_objective_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex125_proteus_hex125_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex27_proteus_hex27_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex64_proteus_hex64_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex64_proteus_hex64_apply_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex64_proteus_hex64_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex64_proteus_hex64_gradient_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex64_proteus_hex64_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex64_proteus_hex64_objective_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex64_proteus_hex64_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex729_proteus_hex729_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex8_proteus_hex8_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex8_proteus_hex8_apply_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex8_proteus_hex8_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex8_proteus_hex8_gradient_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_hex8_proteus_hex8_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_hex8_proteus_hex8_objective_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_hex8_proteus_hex8_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_quad4_proteus_quad4_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_quad4_proteus_quad4_apply_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_quad4_proteus_quad4_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_quad4_proteus_quad4_gradient_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_proteus_quad4_proteus_quad4_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_proteus_quad4_proteus_quad4_objective_soa_diagnostics(void);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_proteus_quad4_proteus_quad4_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_quad4_quad4_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_quad4_quad4_apply_soa_diagnostics(void);

extern "C" void neohookean_ogden_quad4_quad4_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_quad4_quad4_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_quad4_quad4_gradient_soa_diagnostics(void);

extern "C" void neohookean_ogden_quad4_quad4_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_quad4_quad4_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_quad4_quad4_objective_soa_diagnostics(void);

extern "C" void neohookean_ogden_quad4_quad4_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_quad4_quad4_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_tet10_tet10_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tet10_tet10_apply_soa_diagnostics(void);

extern "C" void neohookean_ogden_tet10_tet10_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_tet10_tet10_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tet10_tet10_gradient_soa_diagnostics(void);

extern "C" void neohookean_ogden_tet10_tet10_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_tet10_tet10_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tet10_tet10_objective_soa_diagnostics(void);

extern "C" void neohookean_ogden_tet10_tet10_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet10_tet10_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_tet4_tet4_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tet4_tet4_apply_soa_diagnostics(void);

extern "C" void neohookean_ogden_tet4_tet4_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_tet4_tet4_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tet4_tet4_gradient_soa_diagnostics(void);

extern "C" void neohookean_ogden_tet4_tet4_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_tet4_tet4_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tet4_tet4_objective_soa_diagnostics(void);

extern "C" void neohookean_ogden_tet4_tet4_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tet4_tet4_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_tri3_tri3_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tri3_tri3_apply_soa_diagnostics(void);

extern "C" void neohookean_ogden_tri3_tri3_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_tri3_tri3_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tri3_tri3_gradient_soa_diagnostics(void);

extern "C" void neohookean_ogden_tri3_tri3_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_tri3_tri3_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tri3_tri3_objective_soa_diagnostics(void);

extern "C" void neohookean_ogden_tri3_tri3_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri3_tri3_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_tri6_tri6_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tri6_tri6_apply_soa_diagnostics(void);

extern "C" void neohookean_ogden_tri6_tri6_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_tri6_tri6_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tri6_tri6_gradient_soa_diagnostics(void);

extern "C" void neohookean_ogden_tri6_tri6_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" double neohookean_ogden_tri6_tri6_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes);

extern "C" const sfem::codegen::KernelDiagnostics *neohookean_ogden_tri6_tri6_objective_soa_diagnostics(void);

extern "C" void neohookean_ogden_tri6_tri6_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);

extern "C" void neohookean_ogden_tri6_tri6_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat);
