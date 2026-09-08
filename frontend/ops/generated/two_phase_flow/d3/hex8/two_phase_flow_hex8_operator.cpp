#include "../../op/sfem_GeneratedTwoPhaseFlow_c_abi.hpp"

extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        double *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        float *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR previous,
        double *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR previous,
        float *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const double *const RSTR p_w_old,
        const double *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const float *const RSTR p_w_old,
        const float *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        double *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        float *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR previous,
        double *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR previous,
        float *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const double *const RSTR p_w_old,
        const double *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const float *const RSTR p_w_old,
        const float *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR direction,
        double *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR direction,
        float *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        double *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        float *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR direction,
        double *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR direction,
        float *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        double *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        float *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR direction,
        double *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR direction,
        float *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        double *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        float *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR direction,
        double *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR direction,
        float *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        double *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        float *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR direction,
        double *const RSTR output
);
extern "C" int two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR direction,
        float *const RSTR output
);
extern "C" int two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR previous,
        double *const RSTR output
);
extern "C" int two_phase_flow_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR previous,
        float *const RSTR output
);
extern "C" int two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const double *const RSTR p_w_old,
        const double *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const float *const RSTR p_w_old,
        const float *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_residual_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const double *const RSTR p_w_old,
        const double *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_residual_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const float *const RSTR p_w_old,
        const float *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_residual_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const double *const RSTR p_w_old,
        const double *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_residual_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const float *const RSTR p_w_old,
        const float *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" int two_phase_flow_proteus_hex8_residual_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const double *const RSTR p_w_old,
        const double *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
);
extern "C" int two_phase_flow_proteus_hex8_residual_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const float *const RSTR p_w_old,
        const float *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_element_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_c_proteus_hex8_residual_element_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_w_proteus_hex8_residual_element_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_element_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_element_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_element_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_element_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_element_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_proteus_hex8_jacobian_action_element_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_proteus_hex8_residual_element_soa_diagnostics(
        void
);

extern "C" int two_phase_flow_form_1_p_c_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        double *const RSTR output
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
    return two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_1_p_c_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        float *const RSTR output
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
    return two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_1_p_c_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_c_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_c_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR previous,
        double *const RSTR output
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
    return two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int two_phase_flow_form_1_p_c_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR previous,
        float *const RSTR output
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
    return two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int two_phase_flow_form_1_p_c_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const double *const RSTR p_w_old,
        const double *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_c_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const float *const RSTR p_w_old,
        const float *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        double *const RSTR output
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
    return two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        float *const RSTR output
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
    return two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR previous,
        double *const RSTR output
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
    return two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR previous,
        float *const RSTR output
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
    return two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const double *const RSTR p_w_old,
        const double *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const float *const RSTR p_w_old,
        const float *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR direction,
        double *const RSTR output
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
    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR direction,
        float *const RSTR output
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
    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        double *const RSTR output
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
    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        float *const RSTR output
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
    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR direction,
        double *const RSTR output
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
    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR direction,
        float *const RSTR output
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
    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        double *const RSTR output
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
    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        float *const RSTR output
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
    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR direction,
        double *const RSTR output
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
    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR direction,
        float *const RSTR output
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
    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        double *const RSTR output
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
    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        float *const RSTR output
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
    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR direction,
        double *const RSTR output
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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR direction,
        float *const RSTR output
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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        double *const RSTR output
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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        float *const RSTR output
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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, output);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR direction,
        double *const RSTR output
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
    return two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_hex8_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR direction,
        float *const RSTR output
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
    return two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

extern "C" int two_phase_flow_hex8_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_hex8_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR previous,
        double *const RSTR output
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
    return two_phase_flow_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int two_phase_flow_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR previous,
        float *const RSTR output
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
    return two_phase_flow_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int two_phase_flow_hex8_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const double *const RSTR p_w_old,
        const double *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_hex8_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const float *const RSTR p_w_old,
        const float *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_c_hex8_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_c_hex8_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_c_hex8_residual_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const double *const RSTR p_w_old,
        const double *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_c_proteus_hex8_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_c_hex8_residual_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const float *const RSTR p_w_old,
        const float *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_c_proteus_hex8_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_residual_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const double *const RSTR p_w_old,
        const double *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_w_proteus_hex8_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_1_p_w_hex8_residual_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const float *const RSTR p_w_old,
        const float *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_1_p_w_proteus_hex8_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_det0, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_hex8_jacobian_action_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const double *const RSTR p_w_direction,
        const double *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_hex8_jacobian_action_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const float *const RSTR p_w_direction,
        const float *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_proteus_hex8_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_hex8_residual_affine_mesh_soa(
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
        const double *const RSTR p_w,
        const double *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const double *const RSTR p_w_old,
        const double *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        double *const RSTR p_w_out,
        double *const RSTR p_c_out
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
    return two_phase_flow_proteus_hex8_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_hex8_residual_affine_mesh_soa_float(
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
        const float *const RSTR p_w,
        const float *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const float *const RSTR p_w_old,
        const float *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        float *const RSTR p_w_out,
        float *const RSTR p_c_out
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
    return two_phase_flow_proteus_hex8_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_c_hex8_jacobian_action_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_element_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_c_hex8_residual_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_c_proteus_hex8_residual_element_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_w_hex8_jacobian_action_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_1_p_w_hex8_residual_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_1_p_w_proteus_hex8_residual_element_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_element_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_c_hex8_residual_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_element_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_w_hex8_residual_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_element_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_element_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_c_hex8_residual_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_element_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_element_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_w_hex8_residual_element_soa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_element_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_hex8_jacobian_action_element_soa_diagnostics(
        void
) {
    return two_phase_flow_proteus_hex8_jacobian_action_element_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_hex8_residual_element_soa_diagnostics(
        void
) {
    return two_phase_flow_proteus_hex8_residual_element_soa_diagnostics();
}
