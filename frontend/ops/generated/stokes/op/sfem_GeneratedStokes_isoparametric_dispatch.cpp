#include "sfem_GeneratedStokes_c_abi.hpp"
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

extern "C" int stokes_form_1_p_tri6_tri3_residual_isoparametric_mesh_soa(
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
extern "C" int stokes_form_1_p_tri6_tri3_residual_isoparametric_mesh_soa_float(
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
extern "C" int stokes_form_1_p_hex27_hex8_residual_isoparametric_mesh_soa(
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
extern "C" int stokes_form_1_p_hex27_hex8_residual_isoparametric_mesh_soa_float(
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
extern "C" int stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa(
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
extern "C" int stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa_float(
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
extern "C" int stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[2],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);
extern "C" int stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[2],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);
extern "C" int stokes_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int stokes_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int stokes_form_1_u_tet10_tet4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int stokes_form_1_u_tet10_tet4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int stokes_form_2_p_u_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_p_u_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_p_u_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_p_u_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_p_u_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_p_u_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_p_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa(
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
extern "C" int stokes_form_2_p_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(
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
extern "C" int stokes_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa(
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
extern "C" int stokes_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(
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
extern "C" int stokes_form_2_p_u_tet10_tet4_jacobian_action_isoparametric_mesh_soa(
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
extern "C" int stokes_form_2_p_u_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(
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
extern "C" int stokes_form_2_u_p_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_u_p_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa(
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
extern "C" int stokes_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(
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
extern "C" int stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa(
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
extern "C" int stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(
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
extern "C" int stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa(
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
extern "C" int stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(
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
extern "C" int stokes_form_2_u_u_tri6_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_u_u_tri6_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_u_u_hex27_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_u_u_hex27_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_u_u_tet10_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_u_u_tet10_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2]
);
extern "C" int stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2]
);
extern "C" int stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3]
);
extern "C" int stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3]
);
extern "C" int stokes_form_2_u_u_tet10_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3]
);
extern "C" int stokes_form_2_u_u_tet10_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3]
);
extern "C" int stokes_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int stokes_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int stokes_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int stokes_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int stokes_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int stokes_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int stokes_tri6_tri3_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);
extern "C" int stokes_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);
extern "C" int stokes_hex27_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int stokes_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int stokes_tet10_tet4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int stokes_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int stokes_tri6_tri3_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[2],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);
extern "C" int stokes_tri6_tri3_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[2],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);
extern "C" int stokes_hex27_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int stokes_hex27_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int stokes_tet10_tet4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double mu,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int stokes_tet10_tet4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float mu,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_1_p_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[2],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_1_p_tri6_tri3_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, current_stride, (const double *const *)u_data, (const double *)p_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_1_p_tri6_tri3_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, current_stride, (const float *const *)u_data, (const float *)p_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_1_p_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_1_p_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[3],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_1_p_hex27_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, current_stride, (const double *const *)u_data, (const double *)p_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_1_p_hex27_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, current_stride, (const float *const *)u_data, (const float *)p_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, current_stride, (const double *const *)u_data, (const double *)p_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, current_stride, (const float *const *)u_data, (const float *)p_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_1_p_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_1_u_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t mu,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[2],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, current_stride, (const double *const *)u_data, (const double *)p_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, current_stride, (const float *const *)u_data, (const float *)p_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_1_u_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_1_u_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t mu,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[3],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, current_stride, (const double *const *)u_data, (const double *)p_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, current_stride, (const float *const *)u_data, (const float *)p_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_1_u_tet10_tet4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, current_stride, (const double *const *)u_data, (const double *)p_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_1_u_tet10_tet4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, current_stride, (const float *const *)u_data, (const float *)p_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_1_u_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_2_p_u_hessian_coo_triplet_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        void *const SFEM_RESTRICT values
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_p_u_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_p_u_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_2_p_u_hessian_coo_triplet_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_2_p_u_hessian_coo_triplet_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        void *const SFEM_RESTRICT values
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_p_u_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_p_u_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_p_u_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_p_u_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_2_p_u_hessian_coo_triplet_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_2_p_u_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[2],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_p_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_p_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_2_p_u_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_2_p_u_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[3],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_p_u_tet10_tet4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_p_u_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_2_p_u_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_2_u_p_hessian_coo_triplet_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        void *const SFEM_RESTRICT values
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_u_p_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_u_p_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_2_u_p_hessian_coo_triplet_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_2_u_p_hessian_coo_triplet_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        void *const SFEM_RESTRICT values
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_2_u_p_hessian_coo_triplet_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_2_u_p_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[2],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_2_u_p_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_2_u_p_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[3],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_2_u_p_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_2_u_u_hessian_coo_triplet_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        void *const SFEM_RESTRICT values
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_u_u_tri6_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_u_u_tri6_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_2_u_u_hessian_coo_triplet_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_2_u_u_hessian_coo_triplet_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        void *const SFEM_RESTRICT values
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_u_u_hex27_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_u_u_hex27_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_u_u_tet10_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_u_u_tet10_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_2_u_u_hessian_coo_triplet_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_2_u_u_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t mu,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[2],
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2]
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, direction_stride, (const double *const *)u_direction_data, out_stride, (double *const *)u_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_u_u_tri6_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, direction_stride, (const float *const *)u_direction_data, out_stride, (float *const *)u_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_2_u_u_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_form_2_u_u_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t mu,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[3],
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3]
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, direction_stride, (const double *const *)u_direction_data, out_stride, (double *const *)u_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_u_u_hex27_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, direction_stride, (const float *const *)u_direction_data, out_stride, (float *const *)u_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_form_2_u_u_tet10_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, direction_stride, (const double *const *)u_direction_data, out_stride, (double *const *)u_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_form_2_u_u_tet10_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, direction_stride, (const float *const *)u_direction_data, out_stride, (float *const *)u_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_form_2_u_u_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_hessian_coo_triplet_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        void *const SFEM_RESTRICT values
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return stokes_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_hessian_coo_triplet_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_hessian_coo_triplet_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t mu,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        void *const SFEM_RESTRICT values
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return stokes_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return stokes_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_hessian_coo_triplet_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t mu,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[2],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_tri6_tri3_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t mu,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT u_direction_data[3],
        const void *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_hex27_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_tet10_tet4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t mu,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[2],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[2],
        void *const SFEM_RESTRICT p_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::TRI6: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_tri6_tri3_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, current_stride, (const double *const *)u_data, (const double *)p_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_tri6_tri3_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, current_stride, (const float *const *)u_data, (const float *)p_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int stokes_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t mu,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[3],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT u_out[3],
        void *const SFEM_RESTRICT p_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_hex27_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, current_stride, (const double *const *)u_data, (const double *)p_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_hex27_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, current_stride, (const float *const *)u_data, (const float *)p_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return stokes_tet10_tet4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, mu, current_stride, (const double *const *)u_data, (const double *)p_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return stokes_tet10_tet4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, mu, current_stride, (const float *const *)u_data, (const float *)p_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "stokes_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}
