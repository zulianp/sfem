#include "../../op/sfem_GeneratedMooneyRivlinKelvinVoigt_c_abi.hpp"

extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_proteus_quad4_jacobian_action_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        const void *const RSTR direction,
        void *const RSTR output
);
extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_proteus_quad4_jacobian_action_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_proteus_quad4_residual_i_maos(
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
extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_proteus_quad4_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_proteus_quad4_jacobian_action_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_proteus_quad4_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out
);

extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_quad4_jacobian_action_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        const void *const RSTR direction,
        void *const RSTR output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_residual_merit_proteus_quad4_jacobian_action_i_maos(scalar_bytes, nelements, nnodes, proteus_elements, points, parameters, current, previous, direction, output);
}

extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_quad4_jacobian_action_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_residual_merit_proteus_quad4_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, u0, u1, previous_stride, u0_old, u1_old, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_quad4_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_residual_merit_proteus_quad4_residual_i_maos(scalar_bytes, nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_quad4_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_residual_merit_proteus_quad4_residual_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_quad4_jacobian_action_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_residual_merit_proteus_quad4_jacobian_action_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, u0, u1, previous_stride, u0_old, u1_old, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_residual_merit_quad4_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_residual_merit_proteus_quad4_residual_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out);
}
