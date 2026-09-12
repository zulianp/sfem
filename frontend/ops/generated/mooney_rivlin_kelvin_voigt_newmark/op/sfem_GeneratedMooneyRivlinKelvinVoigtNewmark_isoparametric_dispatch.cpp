#include "sfem_GeneratedMooneyRivlinKelvinVoigtNewmark_c_abi.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double),
              "the generated kernels select their scalar by width");
static_assert((int)smesh::SMESH_FLOAT32 == (int)sizeof(float),
              "the generated kernels select their scalar by width");

extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_apply_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_gradient_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_hessian_bsr_i_msoa(
        const int scalar_bytes,
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_hessian_bsr_i_msoa(
        const int scalar_bytes,
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_hessian_bsr_i_msoa(
        const int scalar_bytes,
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_hessian_bsr_i_msoa(
        const int scalar_bytes,
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_objective_steps_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_steps_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_steps_i_msoa(
        const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_hessian_bsr_i_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    double *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_hessian_bsr_i_msoa_float(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    float *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_hessian_bsr_i_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    double *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_hessian_bsr_i_msoa_float(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    float *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_hessian_bsr_i_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    double *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_hessian_bsr_i_msoa_float(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    float *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_hessian_bsr_i_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    double *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_hessian_bsr_i_msoa_float(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    float *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_hessian_bsr_i_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    double *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_hessian_bsr_i_msoa_float(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    float *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_bsr_i_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    double *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_bsr_i_msoa_float(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    float *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_hessian_bsr_i_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    double *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_hessian_bsr_i_msoa_float(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    float *const RSTR values
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_i_maos(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_i_maos(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_i_maos(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_i_maos(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_maos(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa(
    const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_i_maos(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_i_maos(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_i_msoa(
    const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_i_msoa(
    const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_i_maos(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_i_maos(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_maos(
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
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_i_msoa(
    const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_i_msoa(
    const int scalar_bytes,
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
);
extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa(
    const int scalar_bytes,
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
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_i_msoa(
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
    case smesh::PROTEUS_QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
    case smesh::QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_i_msoa(
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
    case smesh::HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
    case smesh::TET10:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_i_msoa(
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
    case smesh::PROTEUS_QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, out_stride, outx, outy);
    case smesh::QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, out_stride, outx, outy);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_i_msoa(
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
    case smesh::HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
    case smesh::TET10:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hessian_bsr_2d_i_msoa(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, rowptr, colidx, values);
    case smesh::QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, rowptr, colidx, values);
    case smesh::TRI3:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_tri3_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, rowptr, colidx, values);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_hessian_bsr_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_hessian_bsr_3d_i_msoa(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, rowptr, colidx, values);
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, rowptr, colidx, values);
    case smesh::TET10:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, rowptr, colidx, values);
    case smesh::TET4:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_tet4_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, rowptr, colidx, values);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_hessian_bsr_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_2d_i_msoa(
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
    case smesh::PROTEUS_QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_steps_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
    case smesh::QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_objective_steps_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_3d_i_msoa(
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
    case smesh::HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_steps_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_steps_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
    case smesh::TET10:
      return mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_steps_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_objective_steps_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hessian_bsr_2d_i_msoa(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_hessian_bsr_i_msoa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_hessian_bsr_i_msoa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::TRI3: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_hessian_bsr_i_msoa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, previous_stride, (const double *)u0_old, (const double *)u1_old, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, previous_stride, (const float *)u0_old, (const float *)u1_old, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "mooney_rivlin_kelvin_voigt_newmark_viscous_hessian_bsr_2d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_hessian_bsr_3d_i_msoa(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_hessian_bsr_i_msoa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_hessian_bsr_i_msoa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_bsr_i_msoa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::TET4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_hessian_bsr_i_msoa(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "mooney_rivlin_kelvin_voigt_newmark_viscous_hessian_bsr_3d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_i_maos(
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
    case smesh::PROTEUS_QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, direction, output);
    case smesh::QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, direction, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_i_msoa(
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
    case smesh::PROTEUS_QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out);
    case smesh::QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_i_maos(
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
    case smesh::HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, direction, output);
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, direction, output);
    case smesh::TET10:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, direction, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_i_msoa(
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
    case smesh::HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
    case smesh::TET10:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_i_maos(
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
      return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    case smesh::QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_i_msoa(
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
    case smesh::PROTEUS_QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out);
    case smesh::QUAD4:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_i_maos(
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
      return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    case smesh::TET10:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_i_msoa(
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
    case smesh::HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
    case smesh::TET10:
      return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_i_msoa", (int)element_type, (int)real_type);
}
