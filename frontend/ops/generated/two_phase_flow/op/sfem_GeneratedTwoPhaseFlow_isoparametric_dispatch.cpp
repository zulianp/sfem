#include "sfem_GeneratedTwoPhaseFlow_c_abi.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double),
              "the generated kernels select their scalar by width");
static_assert((int)smesh::SMESH_FLOAT32 == (int)sizeof(float),
              "the generated kernels select their scalar by width");

extern "C" int two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_c_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_c_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_c_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const void *const RSTR p_w_old,
    const void *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const void *const RSTR p_w_old,
    const void *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_c_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const void *const RSTR p_w_old,
    const void *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_c_proteus_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const void *const RSTR p_w_old,
    const void *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_w_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_w_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_w_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const void *const RSTR p_w_old,
    const void *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const void *const RSTR p_w_old,
    const void *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_1_p_w_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const void *const RSTR p_w_old,
    const void *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_1_p_w_proteus_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const void *const RSTR p_w_old,
    const void *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_c_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_c_p_w_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_c_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output
);
extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_proteus_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_proteus_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_proteus_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int two_phase_flow_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_proteus_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_proteus_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output
);
extern "C" int two_phase_flow_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output
);
extern "C" int two_phase_flow_proteus_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const void *const RSTR p_w_old,
    const void *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const void *const RSTR p_w_old,
    const void *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output
);
extern "C" int two_phase_flow_proteus_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output
);
extern "C" int two_phase_flow_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const void *const RSTR p_w_old,
    const void *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);
extern "C" int two_phase_flow_proteus_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t previous_stride,
    const void *const RSTR p_w_old,
    const void *const RSTR p_c_old,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_jacobian_action_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    case smesh::QUAD4:
      return two_phase_flow_form_1_p_c_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_c_jacobian_action_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_1_p_c_proteus_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    case smesh::QUAD4:
      return two_phase_flow_form_1_p_c_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_c_jacobian_action_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_jacobian_action_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_1_p_c_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_c_jacobian_action_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_1_p_c_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_1_p_c_proteus_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_c_jacobian_action_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_residual_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_1_p_c_proteus_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    case smesh::QUAD4:
      return two_phase_flow_form_1_p_c_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_c_residual_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_1_p_c_proteus_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
    case smesh::QUAD4:
      return two_phase_flow_form_1_p_c_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_c_residual_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_residual_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_1_p_c_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_1_p_c_proteus_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_c_residual_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_c_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_1_p_c_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_1_p_c_proteus_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_c_residual_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_jacobian_action_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    case smesh::QUAD4:
      return two_phase_flow_form_1_p_w_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_w_jacobian_action_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_1_p_w_proteus_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    case smesh::QUAD4:
      return two_phase_flow_form_1_p_w_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_w_jacobian_action_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_jacobian_action_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_1_p_w_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_w_jacobian_action_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_1_p_w_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_1_p_w_proteus_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_w_jacobian_action_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_residual_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_1_p_w_proteus_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    case smesh::QUAD4:
      return two_phase_flow_form_1_p_w_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_w_residual_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_1_p_w_proteus_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
    case smesh::QUAD4:
      return two_phase_flow_form_1_p_w_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_w_residual_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_residual_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_1_p_w_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_1_p_w_proteus_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_w_residual_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_1_p_w_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_1_p_w_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_1_p_w_proteus_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_1_p_w_residual_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_c_p_c_proteus_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_c_p_c_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_c_jacobian_action_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_c_p_c_proteus_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_c_jacobian_action_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_residual_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_c_p_c_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_c_residual_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_c_p_c_proteus_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_c_p_c_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_c_residual_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_residual_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_c_p_c_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_c_residual_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_c_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_c_p_c_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_c_p_c_proteus_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_c_residual_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_w_jacobian_action_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_c_p_w_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_c_p_w_proteus_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_w_jacobian_action_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_residual_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_c_p_w_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_w_residual_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_c_p_w_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_w_residual_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_residual_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_c_p_w_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_w_residual_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_c_p_w_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_c_p_w_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_c_p_w_proteus_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_c_p_w_residual_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_w_p_c_proteus_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_w_p_c_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_c_jacobian_action_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_w_p_c_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_w_p_c_proteus_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_c_jacobian_action_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_residual_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_w_p_c_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_c_residual_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_w_p_c_proteus_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_w_p_c_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_c_residual_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_residual_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_w_p_c_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_c_residual_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_c_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_w_p_c_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_w_p_c_proteus_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_c_residual_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_w_p_w_proteus_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_w_p_w_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_w_jacobian_action_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_w_jacobian_action_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_residual_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_w_p_w_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_w_residual_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_form_2_p_w_p_w_proteus_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    case smesh::QUAD4:
      return two_phase_flow_form_2_p_w_p_w_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_w_residual_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_residual_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_w_p_w_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_w_residual_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_form_2_p_w_p_w_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_form_2_p_w_p_w_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_form_2_p_w_p_w_residual_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_jacobian_action_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_proteus_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    case smesh::QUAD4:
      return two_phase_flow_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_jacobian_action_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_proteus_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    case smesh::QUAD4:
      return two_phase_flow_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_jacobian_action_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_jacobian_action_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR direction,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_proteus_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, direction, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_jacobian_action_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t direction_stride,
        const void *const RSTR p_w_direction,
        const void *const RSTR p_c_direction,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_proteus_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_jacobian_action_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_residual_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_proteus_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    case smesh::QUAD4:
      return two_phase_flow_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_residual_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return two_phase_flow_proteus_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
    case smesh::QUAD4:
      return two_phase_flow_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_residual_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_residual_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_proteus_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_residual_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
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
        const void *const RSTR p_w,
        const void *const RSTR p_c,
        const ptrdiff_t previous_stride,
        const void *const RSTR p_w_old,
        const void *const RSTR p_c_old,
        const ptrdiff_t out_stride,
        void *const RSTR p_w_out,
        void *const RSTR p_c_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return two_phase_flow_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
    case smesh::PROTEUS_HEX8:
      return two_phase_flow_proteus_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C_ka1, C_ka2, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, kappa_T, m, mu_c, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, previous_stride, p_w_old, p_c_old, out_stride, p_w_out, p_c_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_residual_3d_i_msoa", (int)element_type, (int)real_type);
}
