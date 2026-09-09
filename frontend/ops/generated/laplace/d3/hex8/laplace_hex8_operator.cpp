#include "../../op/sfem_GeneratedLaplace_c_abi.hpp"

extern "C" int laplace_proteus_hex8_apply_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_apply_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_hessian_bsr_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_proteus_hex8_hessian_bsr_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_proteus_hex8_hessian_crs_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_proteus_hex8_hessian_crs_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_proteus_hex8_objective_steps_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_proteus_hex8_objective_steps_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int laplace_proteus_hex8_apply_a_msoa(
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
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_apply_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_a_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_objective_steps_a_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_proteus_hex8_objective_steps_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int laplace_proteus_hex8_apply_packed_i_msoa(
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
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_apply_packed_i_msoa_float(
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
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_apply_packed_two_pass_i_msoa(
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
        double *const RSTR ghost_buf,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_apply_packed_two_pass_i_msoa_float(
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
        float *const RSTR ghost_buf,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_packed_i_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_packed_i_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_packed_two_pass_i_msoa(
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
        double *const RSTR ghost_buf,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_packed_two_pass_i_msoa_float(
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
        float *const RSTR ghost_buf,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_objective_steps_packed_i_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_proteus_hex8_objective_steps_packed_i_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int laplace_proteus_hex8_apply_packed_a_msoa(
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
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_apply_packed_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_apply_packed_two_pass_a_msoa(
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
        double *const RSTR ghost_buf,
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
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_apply_packed_two_pass_a_msoa_float(
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
        float *const RSTR ghost_buf,
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
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_packed_a_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_packed_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_packed_two_pass_a_msoa(
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
        double *const RSTR ghost_buf,
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_packed_two_pass_a_msoa_float(
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
        float *const RSTR ghost_buf,
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_objective_steps_packed_a_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_proteus_hex8_objective_steps_packed_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" const sfem::codegen::KernelDiagnostics * laplace_proteus_hex8_apply_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * laplace_proteus_hex8_gradient_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * laplace_proteus_hex8_objective_soa_diagnostics(
        void
);

extern "C" int laplace_hex8_apply_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_apply_i_msoa(nelements, nnodes, proteus_elements, points, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_hex8_apply_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_apply_i_msoa_float(nelements, nnodes, proteus_elements, points, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_hex8_gradient_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_gradient_i_msoa(nelements, nnodes, proteus_elements, points, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_hex8_gradient_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_gradient_i_msoa_float(nelements, nnodes, proteus_elements, points, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_hex8_hessian_bsr_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_hessian_bsr_i_msoa(nelements, nnodes, proteus_elements, points, kappa, rowptr, colidx, values);
}

extern "C" int laplace_hex8_hessian_bsr_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_hessian_bsr_i_msoa_float(nelements, nnodes, proteus_elements, points, kappa, rowptr, colidx, values);
}

extern "C" int laplace_hex8_hessian_crs_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_hessian_crs_i_msoa(nelements, nnodes, proteus_elements, points, kappa, rowptr, colidx, values);
}

extern "C" int laplace_hex8_hessian_crs_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_hessian_crs_i_msoa_float(nelements, nnodes, proteus_elements, points, kappa, rowptr, colidx, values);
}

extern "C" int laplace_hex8_objective_steps_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_objective_steps_i_msoa(nelements, nnodes, proteus_elements, points, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int laplace_hex8_objective_steps_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_objective_steps_i_msoa_float(nelements, nnodes, proteus_elements, points, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int laplace_hex8_apply_a_msoa(
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
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_apply_a_msoa(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_hex8_apply_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_apply_a_msoa_float(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_hex8_gradient_a_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_gradient_a_msoa(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_hex8_gradient_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_gradient_a_msoa_float(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_hex8_objective_steps_a_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_objective_steps_a_msoa(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int laplace_hex8_objective_steps_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
) {
    idx_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_objective_steps_a_msoa_float(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int laplace_hex8_apply_packed_i_msoa(
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
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_apply_packed_i_msoa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_hex8_apply_packed_i_msoa_float(
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
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_apply_packed_i_msoa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_hex8_apply_packed_two_pass_i_msoa(
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
        double *const RSTR ghost_buf,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_apply_packed_two_pass_i_msoa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, points, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_hex8_apply_packed_two_pass_i_msoa_float(
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
        float *const RSTR ghost_buf,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_apply_packed_two_pass_i_msoa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, points, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_hex8_gradient_packed_i_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_gradient_packed_i_msoa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_hex8_gradient_packed_i_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_gradient_packed_i_msoa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_hex8_gradient_packed_two_pass_i_msoa(
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
        double *const RSTR ghost_buf,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_gradient_packed_two_pass_i_msoa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, points, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_hex8_gradient_packed_two_pass_i_msoa_float(
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
        float *const RSTR ghost_buf,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_gradient_packed_two_pass_i_msoa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, points, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_hex8_objective_steps_packed_i_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_objective_steps_packed_i_msoa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int laplace_hex8_objective_steps_packed_i_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_objective_steps_packed_i_msoa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, points, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int laplace_hex8_apply_packed_a_msoa(
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
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_apply_packed_a_msoa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_hex8_apply_packed_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_apply_packed_a_msoa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_hex8_apply_packed_two_pass_a_msoa(
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
        double *const RSTR ghost_buf,
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
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_apply_packed_two_pass_a_msoa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_hex8_apply_packed_two_pass_a_msoa_float(
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
        float *const RSTR ghost_buf,
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
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_apply_packed_two_pass_a_msoa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, h_stride, hx, out_stride, outx);
}

extern "C" int laplace_hex8_gradient_packed_a_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_gradient_packed_a_msoa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_hex8_gradient_packed_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_gradient_packed_a_msoa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_hex8_gradient_packed_two_pass_a_msoa(
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
        double *const RSTR ghost_buf,
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_gradient_packed_two_pass_a_msoa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_hex8_gradient_packed_two_pass_a_msoa_float(
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
        float *const RSTR ghost_buf,
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_gradient_packed_two_pass_a_msoa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, n_ghost_entries, n_ghost_reduce_rows, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, ghost_buf, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, ux, out_stride, outx);
}

extern "C" int laplace_hex8_objective_steps_packed_a_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_objective_steps_packed_a_msoa(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" int laplace_hex8_objective_steps_packed_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
) {
    uint16_t *proteus_elements[8] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2],
        elements[4],
        elements[5],
        elements[7],
        elements[6]
    };
    return laplace_proteus_hex8_objective_steps_packed_a_msoa_float(n_packs, n_elements_per_pack, nelements, nnodes, max_nodes_per_pack, proteus_elements, owned_nodes_ptr, n_shared_nodes, ghost_ptr, ghost_idx, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, ux, h_stride, hx, nsteps, steps, value);
}

extern "C" const sfem::codegen::KernelDiagnostics * laplace_hex8_apply_soa_diagnostics(
        void
) {
    return laplace_proteus_hex8_apply_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * laplace_hex8_gradient_soa_diagnostics(
        void
) {
    return laplace_proteus_hex8_gradient_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * laplace_hex8_objective_soa_diagnostics(
        void
) {
    return laplace_proteus_hex8_objective_soa_diagnostics();
}
