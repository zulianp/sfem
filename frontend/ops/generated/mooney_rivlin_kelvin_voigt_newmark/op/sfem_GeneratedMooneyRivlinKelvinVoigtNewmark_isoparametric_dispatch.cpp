#include "sfem_GeneratedMooneyRivlinKelvinVoigtNewmark_c_abi.hpp"
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
);
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
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
);
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t out_stride,
        double *const RSTR outx,
        double *const RSTR outy,
        double *const RSTR outz
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t out_stride,
        float *const RSTR outx,
        float *const RSTR outy,
        float *const RSTR outz
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
);
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const double *const RSTR uz,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const double *const RSTR hy,
        const double *const RSTR hz,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const float *const RSTR uz,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const float *const RSTR hy,
        const float *const RSTR hz,
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
);
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
);
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_aos(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_aos_float(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_aos(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_aos(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_aos_float(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_soa(
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
        const double *const RSTR u2,
        const ptrdiff_t previous_stride,
        const double *const RSTR u0_old,
        const double *const RSTR u1_old,
        const double *const RSTR u2_old,
        const ptrdiff_t direction_stride,
        const double *const RSTR u0_direction,
        const double *const RSTR u1_direction,
        const double *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        double *const RSTR u0_out,
        double *const RSTR u1_out,
        double *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_soa_float(
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
        const float *const RSTR u2,
        const ptrdiff_t previous_stride,
        const float *const RSTR u0_old,
        const float *const RSTR u1_old,
        const float *const RSTR u2_old,
        const ptrdiff_t direction_stride,
        const float *const RSTR u0_direction,
        const float *const RSTR u1_direction,
        const float *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        float *const RSTR u0_out,
        float *const RSTR u1_out,
        float *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_soa(
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
        const double *const RSTR u2,
        const ptrdiff_t previous_stride,
        const double *const RSTR u0_old,
        const double *const RSTR u1_old,
        const double *const RSTR u2_old,
        const ptrdiff_t direction_stride,
        const double *const RSTR u0_direction,
        const double *const RSTR u1_direction,
        const double *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        double *const RSTR u0_out,
        double *const RSTR u1_out,
        double *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(
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
        const float *const RSTR u2,
        const ptrdiff_t previous_stride,
        const float *const RSTR u0_old,
        const float *const RSTR u1_old,
        const float *const RSTR u2_old,
        const ptrdiff_t direction_stride,
        const float *const RSTR u0_direction,
        const float *const RSTR u1_direction,
        const float *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        float *const RSTR u0_out,
        float *const RSTR u1_out,
        float *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa(
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
        const double *const RSTR u2,
        const ptrdiff_t previous_stride,
        const double *const RSTR u0_old,
        const double *const RSTR u1_old,
        const double *const RSTR u2_old,
        const ptrdiff_t direction_stride,
        const double *const RSTR u0_direction,
        const double *const RSTR u1_direction,
        const double *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        double *const RSTR u0_out,
        double *const RSTR u1_out,
        double *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa_float(
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
        const float *const RSTR u2,
        const ptrdiff_t previous_stride,
        const float *const RSTR u0_old,
        const float *const RSTR u1_old,
        const float *const RSTR u2_old,
        const ptrdiff_t direction_stride,
        const float *const RSTR u0_direction,
        const float *const RSTR u1_direction,
        const float *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        float *const RSTR u0_out,
        float *const RSTR u1_out,
        float *const RSTR u2_out
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR previous,
        double *const RSTR output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_aos_float(
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
);
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR previous,
        double *const RSTR output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR previous,
        float *const RSTR output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR previous,
        double *const RSTR output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR previous,
        float *const RSTR output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double *const RSTR parameters,
        const double *const RSTR current,
        const double *const RSTR previous,
        double *const RSTR output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float *const RSTR parameters,
        const float *const RSTR current,
        const float *const RSTR previous,
        float *const RSTR output
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_soa(
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
        const double *const RSTR u2,
        const ptrdiff_t previous_stride,
        const double *const RSTR u0_old,
        const double *const RSTR u1_old,
        const double *const RSTR u2_old,
        const ptrdiff_t out_stride,
        double *const RSTR u0_out,
        double *const RSTR u1_out,
        double *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_soa_float(
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
        const float *const RSTR u2,
        const ptrdiff_t previous_stride,
        const float *const RSTR u0_old,
        const float *const RSTR u1_old,
        const float *const RSTR u2_old,
        const ptrdiff_t out_stride,
        float *const RSTR u0_out,
        float *const RSTR u1_out,
        float *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_soa(
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
        const double *const RSTR u2,
        const ptrdiff_t previous_stride,
        const double *const RSTR u0_old,
        const double *const RSTR u1_old,
        const double *const RSTR u2_old,
        const ptrdiff_t out_stride,
        double *const RSTR u0_out,
        double *const RSTR u1_out,
        double *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_soa_float(
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
        const float *const RSTR u2,
        const ptrdiff_t previous_stride,
        const float *const RSTR u0_old,
        const float *const RSTR u1_old,
        const float *const RSTR u2_old,
        const ptrdiff_t out_stride,
        float *const RSTR u0_out,
        float *const RSTR u1_out,
        float *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa(
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
        const double *const RSTR u2,
        const ptrdiff_t previous_stride,
        const double *const RSTR u0_old,
        const double *const RSTR u1_old,
        const double *const RSTR u2_old,
        const ptrdiff_t out_stride,
        double *const RSTR u0_out,
        double *const RSTR u1_out,
        double *const RSTR u2_out
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa_float(
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
        const float *const RSTR u2,
        const ptrdiff_t previous_stride,
        const float *const RSTR u0_old,
        const float *const RSTR u1_old,
        const float *const RSTR u2_old,
        const ptrdiff_t out_stride,
        float *const RSTR u0_out,
        float *const RSTR u1_out,
        float *const RSTR u2_out
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_steps_isoparametric_mesh_soa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_steps_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
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
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, out_stride, (double *)u0_out, (double *)u1_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, out_stride, (float *)u0_out, (float *)u1_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, out_stride, (double *)u0_out, (double *)u1_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, out_stride, (float *)u0_out, (float *)u1_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_isoparametric_mesh_aos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (const double *)direction, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (const float *)direction, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const void *const RSTR u2_old,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const void *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_isoparametric_mesh_aos(
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
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
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
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::PROTEUS_QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, out_stride, (double *)u0_out, (double *)u1_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, out_stride, (float *)u0_out, (float *)u1_out);
                default:
                    break;
            }
            break;
        }
        case smesh::QUAD4: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, out_stride, (double *)u0_out, (double *)u1_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, out_stride, (float *)u0_out, (float *)u1_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_isoparametric_mesh_aos(
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
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_aos(nelements, nnodes, elements, points, (const double *)parameters, (const double *)current, (const double *)previous, (double *)output);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_aos_float(nelements, nnodes, elements, points, (const float *)parameters, (const float *)current, (const float *)previous, (float *)output);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_isoparametric_mesh_aos does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_isoparametric_mesh_soa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const void *const RSTR u2_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out
) {
    const enum smesh::PrimitiveType resolved_real_type =
            (real_type == smesh::SMESH_DEFAULT)
                    ? smesh::TypeToEnum<real_t>::value()
                    : real_type;
    switch (element_type) {
        case smesh::HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::PROTEUS_HEX8: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        case smesh::TET10: {
            switch (resolved_real_type) {
                case smesh::SMESH_FLOAT64:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
                case smesh::SMESH_FLOAT32:
                    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
    std::fprintf(stderr,
            "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_isoparametric_mesh_soa does not support element type %d with real type %d\n",
            (int)element_type,
            (int)real_type);
    return SFEM_FAILURE;
}
