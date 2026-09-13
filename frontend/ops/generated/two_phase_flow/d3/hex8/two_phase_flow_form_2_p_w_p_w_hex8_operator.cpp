#include "../../op/sfem_GeneratedTwoPhaseFlow_c_abi.hpp"

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
extern "C" int two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_a_msoa(
        const int scalar_bytes,
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
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_esoa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_esoa_diagnostics(
        void
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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_i_maos(scalar_bytes, nelements, nnodes, proteus_elements, points, parameters, current, direction, output);
}

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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" int two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_a_msoa(
        const int scalar_bytes,
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
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_kw1, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, p_w, p_c, direction_stride, p_w_direction, p_c_direction, out_stride, p_w_out, p_c_out);
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_w_hex8_jacobian_action_esoa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_jacobian_action_esoa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * two_phase_flow_form_2_p_w_p_w_hex8_residual_esoa_diagnostics(
        void
) {
    return two_phase_flow_form_2_p_w_p_w_proteus_hex8_residual_esoa_diagnostics();
}
