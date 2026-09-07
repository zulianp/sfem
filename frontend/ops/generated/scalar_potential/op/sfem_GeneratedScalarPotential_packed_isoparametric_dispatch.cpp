#include "sfem_GeneratedScalarPotential_c_abi.hpp"
#include <cstdio>

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif
#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

extern "C" int scalar_potential_proteus_quad4_apply_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_quad4_apply_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_quad4_apply_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_quad4_apply_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tri3_apply_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tri3_apply_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_hex8_apply_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_hex8_apply_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_hex8_apply_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_hex8_apply_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tet4_apply_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tet4_apply_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_quad4_apply_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_quad4_apply_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_quad4_apply_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_quad4_apply_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tri3_apply_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tri3_apply_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_hex8_apply_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_hex8_apply_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_hex8_apply_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_hex8_apply_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tet4_apply_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tet4_apply_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_quad4_gradient_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_quad4_gradient_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_quad4_gradient_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_quad4_gradient_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tri3_gradient_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tri3_gradient_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_hex8_gradient_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_hex8_gradient_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_hex8_gradient_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_hex8_gradient_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tet4_gradient_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tet4_gradient_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_quad4_gradient_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_quad4_gradient_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_quad4_gradient_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_quad4_gradient_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tri3_gradient_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tri3_gradient_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_hex8_gradient_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_hex8_gradient_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_hex8_gradient_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_hex8_gradient_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tet4_gradient_packed_two_pass_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_tet4_gradient_packed_two_pass_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx
);
extern "C" int scalar_potential_proteus_quad4_objective_steps_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int scalar_potential_proteus_quad4_objective_steps_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int scalar_potential_quad4_objective_steps_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int scalar_potential_quad4_objective_steps_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int scalar_potential_tri3_objective_steps_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int scalar_potential_tri3_objective_steps_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int scalar_potential_hex8_objective_steps_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int scalar_potential_hex8_objective_steps_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int scalar_potential_proteus_hex8_objective_steps_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int scalar_potential_proteus_hex8_objective_steps_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);
extern "C" int scalar_potential_tet4_objective_steps_packed_isoparametric_mesh_soa(
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
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
);
extern "C" int scalar_potential_tet4_objective_steps_packed_isoparametric_mesh_soa_float(
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
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int scalar_potential_apply_packed_2d_isoparametric_mesh_soa(
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
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_proteus_quad4_apply_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_proteus_quad4_apply_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_quad4_apply_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_quad4_apply_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI3: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_tri3_apply_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_tri3_apply_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "scalar_potential_apply_packed_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int scalar_potential_apply_packed_3d_isoparametric_mesh_soa(
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
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_hex8_apply_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_hex8_apply_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_proteus_hex8_apply_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_proteus_hex8_apply_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::TET4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_tet4_apply_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_tet4_apply_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "scalar_potential_apply_packed_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int scalar_potential_apply_packed_two_pass_2d_isoparametric_mesh_soa(
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
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_proteus_quad4_apply_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_proteus_quad4_apply_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_quad4_apply_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_quad4_apply_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI3: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_tri3_apply_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_tri3_apply_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "scalar_potential_apply_packed_two_pass_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int scalar_potential_apply_packed_two_pass_3d_isoparametric_mesh_soa(
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
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_hex8_apply_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_hex8_apply_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_proteus_hex8_apply_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_proteus_hex8_apply_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::TET4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_tet4_apply_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_tet4_apply_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "scalar_potential_apply_packed_two_pass_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int scalar_potential_gradient_packed_2d_isoparametric_mesh_soa(
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
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_proteus_quad4_gradient_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_proteus_quad4_gradient_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_quad4_gradient_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_quad4_gradient_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI3: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_tri3_gradient_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_tri3_gradient_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "scalar_potential_gradient_packed_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int scalar_potential_gradient_packed_3d_isoparametric_mesh_soa(
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
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_hex8_gradient_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_hex8_gradient_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_proteus_hex8_gradient_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_proteus_hex8_gradient_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::TET4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_tet4_gradient_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_tet4_gradient_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "scalar_potential_gradient_packed_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int scalar_potential_gradient_packed_two_pass_2d_isoparametric_mesh_soa(
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
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_proteus_quad4_gradient_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_proteus_quad4_gradient_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_quad4_gradient_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_quad4_gradient_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI3: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_tri3_gradient_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_tri3_gradient_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "scalar_potential_gradient_packed_two_pass_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int scalar_potential_gradient_packed_two_pass_3d_isoparametric_mesh_soa(
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
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_hex8_gradient_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_hex8_gradient_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_proteus_hex8_gradient_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_proteus_hex8_gradient_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        case smesh::TET4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_tet4_gradient_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_tet4_gradient_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "scalar_potential_gradient_packed_two_pass_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int scalar_potential_objective_steps_packed_2d_isoparametric_mesh_soa(
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
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const int nsteps,
        const void *const SFEM_RESTRICT steps,
        void *const SFEM_RESTRICT value
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_proteus_quad4_objective_steps_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_proteus_quad4_objective_steps_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_quad4_objective_steps_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_quad4_objective_steps_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::TRI3: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_tri3_objective_steps_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_tri3_objective_steps_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "scalar_potential_objective_steps_packed_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int scalar_potential_objective_steps_packed_3d_isoparametric_mesh_soa(
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
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const int nsteps,
        const void *const SFEM_RESTRICT steps,
        void *const SFEM_RESTRICT value
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_hex8_objective_steps_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_hex8_objective_steps_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_proteus_hex8_objective_steps_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_proteus_hex8_objective_steps_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::TET4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return scalar_potential_tet4_objective_steps_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return scalar_potential_tet4_objective_steps_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "scalar_potential_objective_steps_packed_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}
