#include "sfem_GeneratedPoroHyperelasticity_c_abi.hpp"
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

extern "C" int poro_hyperelasticity_solid_tri6_apply_packed_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tri6_apply_packed_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_hex27_apply_packed_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_hex27_apply_packed_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_apply_packed_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_apply_packed_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_tet10_apply_packed_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tet10_apply_packed_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_tri6_apply_packed_two_pass_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tri6_apply_packed_two_pass_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_hex27_apply_packed_two_pass_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_hex27_apply_packed_two_pass_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_apply_packed_two_pass_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_apply_packed_two_pass_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_tet10_apply_packed_two_pass_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tet10_apply_packed_two_pass_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_tri6_gradient_packed_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tri6_gradient_packed_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_hex27_gradient_packed_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_hex27_gradient_packed_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_gradient_packed_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_gradient_packed_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_tet10_gradient_packed_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tet10_gradient_packed_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_tri6_gradient_packed_two_pass_isoparametric_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy
);
extern "C" int poro_hyperelasticity_solid_tri6_gradient_packed_two_pass_isoparametric_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy
);
extern "C" int poro_hyperelasticity_solid_hex27_gradient_packed_two_pass_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_hex27_gradient_packed_two_pass_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_gradient_packed_two_pass_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_gradient_packed_two_pass_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_tet10_gradient_packed_two_pass_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tet10_gradient_packed_two_pass_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_tri6_objective_steps_packed_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tri6_objective_steps_packed_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_hex27_objective_steps_packed_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_hex27_objective_steps_packed_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_objective_steps_packed_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_objective_steps_packed_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_tet10_objective_steps_packed_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tet10_objective_steps_packed_isoparametric_mesh_soa_float(
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

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_apply_packed_2d_isoparametric_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_tri6_apply_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tri6_apply_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_apply_packed_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_apply_packed_3d_isoparametric_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
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
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_hex27_apply_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_hex27_apply_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_proteus_hex27_apply_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_proteus_hex27_apply_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_tet10_apply_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tet10_apply_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_apply_packed_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_apply_packed_two_pass_2d_isoparametric_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_tri6_apply_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tri6_apply_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_apply_packed_two_pass_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_apply_packed_two_pass_3d_isoparametric_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
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
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_hex27_apply_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_hex27_apply_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_proteus_hex27_apply_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_proteus_hex27_apply_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_tet10_apply_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tet10_apply_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_apply_packed_two_pass_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_gradient_packed_2d_isoparametric_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_tri6_gradient_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tri6_gradient_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_gradient_packed_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_gradient_packed_3d_isoparametric_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_hex27_gradient_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_hex27_gradient_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_proteus_hex27_gradient_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_proteus_hex27_gradient_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_tet10_gradient_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tet10_gradient_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_gradient_packed_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_gradient_packed_two_pass_2d_isoparametric_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_tri6_gradient_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tri6_gradient_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_gradient_packed_two_pass_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_gradient_packed_two_pass_3d_isoparametric_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const void *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT outx,
        void *const SFEM_RESTRICT outy,
        void *const SFEM_RESTRICT outz
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_hex27_gradient_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_hex27_gradient_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_proteus_hex27_gradient_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_proteus_hex27_gradient_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_tet10_gradient_packed_two_pass_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tet10_gradient_packed_two_pass_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_gradient_packed_two_pass_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_objective_steps_packed_2d_isoparametric_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const SFEM_RESTRICT ux,
        const void *const SFEM_RESTRICT uy,
        const ptrdiff_t h_stride,
        const void *const SFEM_RESTRICT hx,
        const void *const SFEM_RESTRICT hy,
        const int nsteps,
        const void *const SFEM_RESTRICT steps,
        void *const SFEM_RESTRICT value
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_tri6_objective_steps_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tri6_objective_steps_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_objective_steps_packed_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_objective_steps_packed_3d_isoparametric_mesh_soa(
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
        const real_t lmbda,
        const real_t mu,
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
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_hex27_objective_steps_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_hex27_objective_steps_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_proteus_hex27_objective_steps_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_proteus_hex27_objective_steps_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_tet10_objective_steps_packed_isoparametric_mesh_soa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tet10_objective_steps_packed_isoparametric_mesh_soa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_objective_steps_packed_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}
