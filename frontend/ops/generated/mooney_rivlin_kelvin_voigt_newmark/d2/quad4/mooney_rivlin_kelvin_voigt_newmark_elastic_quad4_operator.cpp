#include "../../op/sfem_GeneratedMooneyRivlinKelvinVoigtNewmark_c_abi.hpp"

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR previous,
        const double *const RSTR direction,
        double *const RSTR output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR previous,
        const float *const RSTR direction,
        float *const RSTR output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const RSTR u0,
        const double *const RSTR u1,
        const ptrdiff_t previous_stride,
        const double *const RSTR u0_old,
        const double *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const double *const RSTR u0_direction,
        const double *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        double *const RSTR u0_out,
        double *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const RSTR u0,
        const float *const RSTR u1,
        const ptrdiff_t previous_stride,
        const float *const RSTR u0_old,
        const float *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const float *const RSTR u0_direction,
        const float *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        float *const RSTR u0_out,
        float *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR previous,
        double *const RSTR output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR previous,
        float *const RSTR output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const RSTR u0,
        const double *const RSTR u1,
        const ptrdiff_t previous_stride,
        const double *const RSTR u0_old,
        const double *const RSTR u1_old,
        const ptrdiff_t out_stride,
        double *const RSTR u0_out,
        double *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const RSTR u0,
        const float *const RSTR u1,
        const ptrdiff_t previous_stride,
        const float *const RSTR u0_old,
        const float *const RSTR u1_old,
        const ptrdiff_t out_stride,
        float *const RSTR u0_out,
        float *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const RSTR u0,
        const double *const RSTR u1,
        const ptrdiff_t previous_stride,
        const double *const RSTR u0_old,
        const double *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const double *const RSTR u0_direction,
        const double *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        double *const RSTR u0_out,
        double *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const RSTR u0,
        const float *const RSTR u1,
        const ptrdiff_t previous_stride,
        const float *const RSTR u0_old,
        const float *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const float *const RSTR u0_direction,
        const float *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        float *const RSTR u0_out,
        float *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const RSTR u0,
        const double *const RSTR u1,
        const ptrdiff_t previous_stride,
        const double *const RSTR u0_old,
        const double *const RSTR u1_old,
        const ptrdiff_t out_stride,
        double *const RSTR u0_out,
        double *const RSTR u1_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const RSTR u0,
        const float *const RSTR u1,
        const ptrdiff_t previous_stride,
        const float *const RSTR u0_old,
        const float *const RSTR u1_old,
        const ptrdiff_t out_stride,
        float *const RSTR u0_out,
        float *const RSTR u1_out
);
extern "C" const sfem::codegen::KernelDiagnostics * mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_diagnostics(
        void
);

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, lmbda, mu, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, lmbda, mu, u_stride, ux, uy, out_stride, outx, outy);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR previous,
        const double *const RSTR direction,
        double *const RSTR output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, previous, direction, output);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR previous,
        const float *const RSTR direction,
        float *const RSTR output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, previous, direction, output);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const RSTR u0,
        const double *const RSTR u1,
        const ptrdiff_t previous_stride,
        const double *const RSTR u0_old,
        const double *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const double *const RSTR u0_direction,
        const double *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        double *const RSTR u0_out,
        double *const RSTR u1_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const RSTR u0,
        const float *const RSTR u1,
        const ptrdiff_t previous_stride,
        const float *const RSTR u0_old,
        const float *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const float *const RSTR u0_direction,
        const float *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        float *const RSTR u0_out,
        float *const RSTR u1_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR previous,
        double *const RSTR output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR previous,
        float *const RSTR output
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, proteus_elements, points, parameters, current, previous, output);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const RSTR u0,
        const double *const RSTR u1,
        const ptrdiff_t previous_stride,
        const double *const RSTR u0_old,
        const double *const RSTR u1_old,
        const ptrdiff_t out_stride,
        double *const RSTR u0_out,
        double *const RSTR u1_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, proteus_elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const RSTR u0,
        const float *const RSTR u1,
        const ptrdiff_t previous_stride,
        const float *const RSTR u0_old,
        const float *const RSTR u1_old,
        const ptrdiff_t out_stride,
        float *const RSTR u0_out,
        float *const RSTR u1_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, proteus_elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const RSTR u0,
        const double *const RSTR u1,
        const ptrdiff_t previous_stride,
        const double *const RSTR u0_old,
        const double *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const double *const RSTR u0_direction,
        const double *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        double *const RSTR u0_out,
        double *const RSTR u1_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const RSTR u0,
        const float *const RSTR u1,
        const ptrdiff_t previous_stride,
        const float *const RSTR u0_old,
        const float *const RSTR u1_old,
        const ptrdiff_t direction_stride,
        const float *const RSTR u0_direction,
        const float *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        float *const RSTR u0_out,
        float *const RSTR u1_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const RSTR u0,
        const double *const RSTR u1,
        const ptrdiff_t previous_stride,
        const double *const RSTR u0_old,
        const double *const RSTR u1_old,
        const ptrdiff_t out_stride,
        double *const RSTR u0_out,
        double *const RSTR u1_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_affine_mesh_soa(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_jacobian_adjugate0,
        const geom_t *const RSTR g_jacobian_adjugate1,
        const geom_t *const RSTR g_jacobian_adjugate2,
        const geom_t *const RSTR g_jacobian_adjugate3,
        const geom_t *const RSTR g_jacobian_determinant0,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const RSTR u0,
        const float *const RSTR u1,
        const ptrdiff_t previous_stride,
        const float *const RSTR u0_old,
        const float *const RSTR u1_old,
        const ptrdiff_t out_stride,
        float *const RSTR u0_out,
        float *const RSTR u1_out
) {
    idx_t *proteus_elements[4] = {
        elements[0],
        elements[1],
        elements[3],
        elements[2]
    };
    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_affine_mesh_soa_float(nelements, nnodes, proteus_elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out);
}

extern "C" const sfem::codegen::KernelDiagnostics * mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_apply_soa_diagnostics(
        void
) {
    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_gradient_soa_diagnostics(
        void
) {
    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_objective_soa_diagnostics(
        void
) {
    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_soa_diagnostics();
}
