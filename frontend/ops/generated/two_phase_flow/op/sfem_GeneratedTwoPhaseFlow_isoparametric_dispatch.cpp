#include "sfem_GeneratedTwoPhaseFlow_c_abi.hpp"
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

extern "C" int two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_c_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double m,
        const double mu_c,
        const double porosity,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float m,
        const float mu_c,
        const float porosity,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double P_r,
        const double S_res,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float P_r,
        const float S_res,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT p_w_direction,
        const double *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT p_w_direction,
        const float *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_proteus_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_proteus_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_proteus_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_proteus_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
);
extern "C" int two_phase_flow_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double C_ka1,
        const double C_ka2,
        const double C_kw1,
        const double K_0,
        const double K_1,
        const double K_2,
        const double K_3,
        const double K_4,
        const double K_5,
        const double K_6,
        const double K_7,
        const double K_8,
        const double M_c,
        const double P_r,
        const double R,
        const double S_res,
        const double T,
        const double Z,
        const double dt,
        const double kappa_T,
        const double m,
        const double mu_c,
        const double mu_w,
        const double p_wr,
        const double porosity,
        const double rho_w0,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT p_w,
        const double *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT p_w_old,
        const double *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT p_w_out,
        double *const SFEM_RESTRICT p_c_out
);
extern "C" int two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float C_ka1,
        const float C_ka2,
        const float C_kw1,
        const float K_0,
        const float K_1,
        const float K_2,
        const float K_3,
        const float K_4,
        const float K_5,
        const float K_6,
        const float K_7,
        const float K_8,
        const float M_c,
        const float P_r,
        const float R,
        const float S_res,
        const float T,
        const float Z,
        const float dt,
        const float kappa_T,
        const float m,
        const float mu_c,
        const float mu_w,
        const float p_wr,
        const float porosity,
        const float rho_w0,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT p_w,
        const float *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT p_w_old,
        const float *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT p_w_out,
        float *const SFEM_RESTRICT p_c_out
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_c_jacobian_action_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_c_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_c_jacobian_action_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_c_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT previous,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_c_residual_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT p_w_old,
        const void *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, previous_stride, (const double *)p_w_old, (const double *)p_c_old, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, previous_stride, (const float *)p_w_old, (const float *)p_c_old, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, previous_stride, (const double *)p_w_old, (const double *)p_c_old, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, previous_stride, (const float *)p_w_old, (const float *)p_c_old, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_c_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT previous,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_c_residual_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT p_w_old,
        const void *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, previous_stride, (const double *)p_w_old, (const double *)p_c_old, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, previous_stride, (const float *)p_w_old, (const float *)p_c_old, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, previous_stride, (const double *)p_w_old, (const double *)p_c_old, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, previous_stride, (const float *)p_w_old, (const float *)p_c_old, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_c_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_w_jacobian_action_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_w_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_w_jacobian_action_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_w_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT previous,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_w_residual_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT p_w_old,
        const void *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, previous_stride, (const double *)p_w_old, (const double *)p_c_old, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, previous_stride, (const float *)p_w_old, (const float *)p_c_old, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, previous_stride, (const double *)p_w_old, (const double *)p_c_old, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, previous_stride, (const float *)p_w_old, (const float *)p_c_old, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_w_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT previous,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_w_residual_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT p_w_old,
        const void *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, previous_stride, (const double *)p_w_old, (const double *)p_c_old, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, previous_stride, (const float *)p_w_old, (const float *)p_c_old, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, previous_stride, (const double *)p_w_old, (const double *)p_c_old, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, previous_stride, (const float *)p_w_old, (const float *)p_c_old, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_1_p_w_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT direction,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT p_w_direction,
        const void *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT direction,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT p_w_direction,
        const void *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_c_residual_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_c_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_c_residual_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_c_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT direction,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT p_w_direction,
        const void *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT direction,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t m,
        const real_t mu_c,
        const real_t porosity,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT p_w_direction,
        const void *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_w_residual_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_w_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_w_residual_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_c_p_w_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT direction,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT p_w_direction,
        const void *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT direction,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT p_w_direction,
        const void *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_c_residual_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_c_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_c_residual_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_c_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT direction,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT p_w_direction,
        const void *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT direction,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT p_w_direction,
        const void *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_w_residual_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_w_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_w_residual_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_form_2_p_w_p_w_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT direction,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_jacobian_action_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_c,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT p_w_direction,
        const void *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT direction,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_jacobian_action_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_c,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t direction_stride,
        const void *const SFEM_RESTRICT p_w_direction,
        const void *const SFEM_RESTRICT p_c_direction,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_residual_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT previous,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_residual_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_c,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT p_w_old,
        const void *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, previous_stride, (const double *)p_w_old, (const double *)p_c_old, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, previous_stride, (const float *)p_w_old, (const float *)p_c_old, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, previous_stride, (const double *)p_w_old, (const double *)p_c_old, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, previous_stride, (const float *)p_w_old, (const float *)p_c_old, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_residual_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const void *const SFEM_RESTRICT parameters,
        const void *const SFEM_RESTRICT current,
        const void *const SFEM_RESTRICT previous,
        void *const SFEM_RESTRICT output
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_residual_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t P_r,
        const real_t R,
        const real_t S_res,
        const real_t T,
        const real_t Z,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t mu_c,
        const real_t mu_w,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const ptrdiff_t current_stride,
        const void *const SFEM_RESTRICT p_w,
        const void *const SFEM_RESTRICT p_c,
        const ptrdiff_t previous_stride,
        const void *const SFEM_RESTRICT p_w_old,
        const void *const SFEM_RESTRICT p_c_old,
        const ptrdiff_t out_stride,
        void *const SFEM_RESTRICT p_w_out,
        void *const SFEM_RESTRICT p_c_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, previous_stride, (const double *)p_w_old, (const double *)p_c_old, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, previous_stride, (const float *)p_w_old, (const float *)p_c_old, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, previous_stride, (const double *)p_w_old, (const double *)p_c_old, out_stride, (double *)p_w_out, (double *)p_c_out);
                case smesh::SMESH_FLOAT32:
                    return two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, previous_stride, (const float *)p_w_old, (const float *)p_c_old, out_stride, (float *)p_w_out, (float *)p_c_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "two_phase_flow_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}
