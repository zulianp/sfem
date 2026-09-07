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

extern "C" int poro_hyperelasticity_poro_form_1_p_tri6_tri3_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[2],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_p_tri6_tri3_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[2],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_p_hex27_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_p_hex27_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_p_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_p_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[2],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[2],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_u_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_u_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_u_tet10_tet4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_1_u_tet10_tet4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_proteus_hex27_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_proteus_hex27_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_proteus_hex27_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_proteus_hex27_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_tet10_tet4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_p_u_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_proteus_hex27_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_proteus_hex27_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_proteus_hex27_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_proteus_hex27_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_proteus_hex27_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_proteus_hex27_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
);
extern "C" int poro_hyperelasticity_poro_tri6_tri3_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[2],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[2],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_hex27_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_proteus_hex27_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_proteus_hex27_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_tet10_tet4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_tri6_tri3_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[2],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[2],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[2],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_tri6_tri3_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[2],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[2],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[2],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_hex27_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_hex27_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_tet10_tet4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_poro_tet10_tet4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
);
extern "C" int poro_hyperelasticity_solid_tri6_apply_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tri6_apply_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_hex27_apply_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_hex27_apply_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_apply_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_apply_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_tet10_apply_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tet10_apply_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_tri6_gradient_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tri6_gradient_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_hex27_gradient_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_hex27_gradient_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_gradient_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_gradient_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_tet10_gradient_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tet10_gradient_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_tri6_objective_steps_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tri6_objective_steps_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_hex27_objective_steps_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_hex27_objective_steps_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_objective_steps_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_proteus_hex27_objective_steps_isoparametric_mesh_soa_float(
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
extern "C" int poro_hyperelasticity_solid_tet10_objective_steps_isoparametric_mesh_soa(
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
extern "C" int poro_hyperelasticity_solid_tet10_objective_steps_isoparametric_mesh_soa_float(
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

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_form_1_p_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
        const real_t dt,
        const real_t hydraulic_conductivity,
        const real_t storage,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[2],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[2],
        const void *const SFEM_RESTRICT p_old_data,
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
                    return poro_hyperelasticity_poro_form_1_p_tri6_tri3_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const double *const *)u_data, (const double *)p_data, previous_stride, (const double *const *)u_old_data, (const double *)p_old_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_1_p_tri6_tri3_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const float *const *)u_data, (const float *)p_data, previous_stride, (const float *const *)u_old_data, (const float *)p_old_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_form_1_p_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_form_1_p_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
        const real_t dt,
        const real_t hydraulic_conductivity,
        const real_t storage,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[3],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[3],
        const void *const SFEM_RESTRICT p_old_data,
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
                    return poro_hyperelasticity_poro_form_1_p_hex27_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const double *const *)u_data, (const double *)p_data, previous_stride, (const double *const *)u_old_data, (const double *)p_old_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_1_p_hex27_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const float *const *)u_data, (const float *)p_data, previous_stride, (const float *const *)u_old_data, (const float *)p_old_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_form_1_p_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const double *const *)u_data, (const double *)p_data, previous_stride, (const double *const *)u_old_data, (const double *)p_old_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_1_p_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const float *const *)u_data, (const float *)p_data, previous_stride, (const float *const *)u_old_data, (const float *)p_old_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const double *const *)u_data, (const double *)p_data, previous_stride, (const double *const *)u_old_data, (const double *)p_old_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const float *const *)u_data, (const float *)p_data, previous_stride, (const float *const *)u_old_data, (const float *)p_old_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_form_1_p_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_form_1_u_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
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
                    return poro_hyperelasticity_poro_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, current_stride, (const double *const *)u_data, (const double *)p_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_1_u_tri6_tri3_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, current_stride, (const float *const *)u_data, (const float *)p_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_form_1_u_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_form_1_u_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
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
                    return poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, current_stride, (const double *const *)u_data, (const double *)p_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_1_u_hex27_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, current_stride, (const float *const *)u_data, (const float *)p_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_form_1_u_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, current_stride, (const double *const *)u_data, (const double *)p_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_1_u_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, current_stride, (const float *const *)u_data, (const float *)p_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_form_1_u_tet10_tet4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, current_stride, (const double *const *)u_data, (const double *)p_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_1_u_tet10_tet4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, current_stride, (const float *const *)u_data, (const float *)p_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_form_1_u_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_form_2_p_u_hessian_coo_triplet_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
        const real_t dt,
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
                    return poro_hyperelasticity_poro_form_2_p_u_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_p_u_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_form_2_p_u_hessian_coo_triplet_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_form_2_p_u_hessian_coo_triplet_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
        const real_t dt,
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
                    return poro_hyperelasticity_poro_form_2_p_u_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_p_u_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_form_2_p_u_proteus_hex27_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_p_u_proteus_hex27_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_form_2_p_u_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_p_u_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_form_2_p_u_hessian_coo_triplet_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_form_2_p_u_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
        const real_t dt,
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
                    return poro_hyperelasticity_poro_form_2_p_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_p_u_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_form_2_p_u_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_form_2_p_u_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
        const real_t dt,
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
                    return poro_hyperelasticity_poro_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_p_u_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_form_2_p_u_proteus_hex27_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_p_u_proteus_hex27_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_form_2_p_u_tet10_tet4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_p_u_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_form_2_p_u_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_form_2_u_p_hessian_coo_triplet_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
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
                    return poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_form_2_u_p_hessian_coo_triplet_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_form_2_u_p_hessian_coo_triplet_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
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
                    return poro_hyperelasticity_poro_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_u_p_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_form_2_u_p_proteus_hex27_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_u_p_proteus_hex27_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_form_2_u_p_hessian_coo_triplet_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_form_2_u_p_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
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
                    return poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_form_2_u_p_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_form_2_u_p_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
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
                    return poro_hyperelasticity_poro_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_u_p_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_form_2_u_p_proteus_hex27_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_u_p_proteus_hex27_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_form_2_u_p_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_hessian_coo_triplet_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
        const real_t dt,
        const real_t hydraulic_conductivity,
        const real_t storage,
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
                    return poro_hyperelasticity_poro_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_tri6_tri3_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_hessian_coo_triplet_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_hessian_coo_triplet_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
        const real_t dt,
        const real_t hydraulic_conductivity,
        const real_t storage,
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
                    return poro_hyperelasticity_poro_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_hex27_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_proteus_hex27_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_proteus_hex27_proteus_hex8_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, out_stride, rows, cols, (double *)values);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, out_stride, rows, cols, (float *)values);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_hessian_coo_triplet_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
        const real_t dt,
        const real_t hydraulic_conductivity,
        const real_t storage,
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
                    return poro_hyperelasticity_poro_tri6_tri3_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_tri6_tri3_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
        const real_t dt,
        const real_t hydraulic_conductivity,
        const real_t storage,
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
                    return poro_hyperelasticity_poro_hex27_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_hex27_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_proteus_hex27_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_proteus_hex27_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_tet10_tet4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
        const real_t dt,
        const real_t hydraulic_conductivity,
        const real_t storage,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[2],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[2],
        const void *const SFEM_RESTRICT p_old_data,
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
                    return poro_hyperelasticity_poro_tri6_tri3_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const double *const *)u_data, (const double *)p_data, previous_stride, (const double *const *)u_old_data, (const double *)p_old_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_tri6_tri3_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const float *const *)u_data, (const float *)p_data, previous_stride, (const float *const *)u_old_data, (const float *)p_old_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_poro_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t alpha,
        const real_t dt,
        const real_t hydraulic_conductivity,
        const real_t storage,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT u_data[3],
        const void *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT u_old_data[3],
        const void *const SFEM_RESTRICT p_old_data,
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
                    return poro_hyperelasticity_poro_hex27_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const double *const *)u_data, (const double *)p_data, previous_stride, (const double *const *)u_old_data, (const double *)p_old_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_hex27_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const float *const *)u_data, (const float *)p_data, previous_stride, (const float *const *)u_old_data, (const float *)p_old_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const double *const *)u_data, (const double *)p_data, previous_stride, (const double *const *)u_old_data, (const double *)p_old_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const float *const *)u_data, (const float *)p_data, previous_stride, (const float *const *)u_old_data, (const float *)p_old_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_poro_tet10_tet4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const double *const *)u_data, (const double *)p_data, previous_stride, (const double *const *)u_old_data, (const double *)p_old_data, out_stride, (double *const *)u_out, (double *)p_out);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_poro_tet10_tet4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, (const float *const *)u_data, (const float *)p_data, previous_stride, (const float *const *)u_old_data, (const float *)p_old_data, out_stride, (float *const *)u_out, (float *)p_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_poro_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_apply_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
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
                    return poro_hyperelasticity_solid_tri6_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tri6_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_apply_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_apply_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
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
                    return poro_hyperelasticity_solid_hex27_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_hex27_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_proteus_hex27_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_proteus_hex27_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_tet10_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tet10_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_apply_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_gradient_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
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
                    return poro_hyperelasticity_solid_tri6_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tri6_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_gradient_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_gradient_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
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
                    return poro_hyperelasticity_solid_hex27_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_hex27_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_proteus_hex27_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_proteus_hex27_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_tet10_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tet10_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_gradient_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_objective_steps_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
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
                    return poro_hyperelasticity_solid_tri6_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tri6_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_objective_steps_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int poro_hyperelasticity_solid_objective_steps_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
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
                    return poro_hyperelasticity_solid_hex27_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_hex27_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX27: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_proteus_hex27_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_proteus_hex27_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return poro_hyperelasticity_solid_tet10_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return poro_hyperelasticity_solid_tet10_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "poro_hyperelasticity_solid_objective_steps_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}
