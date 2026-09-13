#include "../../op/sfem_GeneratedTwoPhaseFlow_c_abi.hpp"

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
extern "C" int two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
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
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_esoa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_esoa_diagnostics(
        void
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
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_maos(scalar_bytes, nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

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
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
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
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_w_quad4_jacobian_action_esoa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_esoa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_c_p_w_quad4_residual_esoa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_esoa_diagnostics();
}
