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

extern "C" int laplace_apply_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_apply_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_apply_2d_soa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int laplace_apply_3d_a_met_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_apply_3d_a_msoa(
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
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_apply_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_apply_3d_soa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int laplace_apply_packed_3d_a_met_msoa(
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
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_apply_packed_3d_a_msoa(
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
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_apply_packed_3d_i_msoa(
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
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_apply_packed_two_pass_3d_a_met_msoa(
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
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_apply_packed_two_pass_3d_a_msoa(
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
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_apply_packed_two_pass_3d_i_msoa(
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
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_gradient_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_gradient_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_gradient_2d_soa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int laplace_gradient_3d_a_met_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_gradient_3d_a_msoa(
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
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_gradient_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_gradient_3d_soa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int laplace_gradient_packed_3d_a_met_msoa(
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
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_gradient_packed_3d_a_msoa(
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
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_gradient_packed_3d_i_msoa(
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
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_gradient_packed_two_pass_3d_a_met_msoa(
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
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_gradient_packed_two_pass_3d_a_msoa(
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
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_gradient_packed_two_pass_3d_i_msoa(
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
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx
);

extern "C" int laplace_hessian_bsr_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);

extern "C" int laplace_hessian_bsr_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);

extern "C" int laplace_hessian_crs_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);

extern "C" int laplace_hessian_crs_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);

extern "C" void laplace_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *laplace_hex8_matrix_assembly_variant(const int variant);

extern "C" int laplace_hex8_matrix_assembly_variant_count();

extern "C" const sfem::codegen::KernelDiagnostics *laplace_objective_2d_soa_diagnostics(
    const smesh::ElemType element_type);

extern "C" const sfem::codegen::KernelDiagnostics *laplace_objective_3d_soa_diagnostics(
    const smesh::ElemType element_type);

extern "C" int laplace_objective_steps_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);

extern "C" int laplace_objective_steps_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);

extern "C" int laplace_objective_steps_3d_a_met_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);

extern "C" int laplace_objective_steps_3d_a_msoa(
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
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);

extern "C" int laplace_objective_steps_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);

extern "C" int laplace_objective_steps_packed_3d_a_met_msoa(
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
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);

extern "C" int laplace_objective_steps_packed_3d_a_msoa(
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
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);

extern "C" int laplace_objective_steps_packed_3d_i_msoa(
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
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
);

extern "C" void laplace_proteus_hex8_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *laplace_proteus_hex8_matrix_assembly_variant(const int variant);

extern "C" int laplace_proteus_hex8_matrix_assembly_variant_count();

extern "C" void laplace_proteus_quad4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *laplace_proteus_quad4_matrix_assembly_variant(const int variant);

extern "C" int laplace_proteus_quad4_matrix_assembly_variant_count();

extern "C" void laplace_quad4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *laplace_quad4_matrix_assembly_variant(const int variant);

extern "C" int laplace_quad4_matrix_assembly_variant_count();

extern "C" void laplace_tet10_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *laplace_tet10_matrix_assembly_variant(const int variant);

extern "C" int laplace_tet10_matrix_assembly_variant_count();

extern "C" void laplace_tet4_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *laplace_tet4_matrix_assembly_variant(const int variant);

extern "C" int laplace_tet4_matrix_assembly_variant_count();

extern "C" void laplace_tri3_matrix_assembly_print_variant(const int variant, const ptrdiff_t nelements);

extern "C" const sfem_MatrixAssemblyDiagnostics *laplace_tri3_matrix_assembly_variant(const int variant);

extern "C" int laplace_tri3_matrix_assembly_variant_count();
