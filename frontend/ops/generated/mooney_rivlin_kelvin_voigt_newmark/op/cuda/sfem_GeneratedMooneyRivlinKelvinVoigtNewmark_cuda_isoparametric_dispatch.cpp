#include "sfem_GeneratedMooneyRivlinKelvinVoigtNewmark_cuda_c_abi.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double),
              "the generated kernels select their scalar by width");
static_assert((int)smesh::SMESH_FLOAT32 == (int)sizeof(float),
              "the generated kernels select their scalar by width");

extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa(
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
        void *const RSTR outy,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_apply_i_msoa(
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
        void *const RSTR outy,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_i_msoa(
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
        void *const RSTR outz,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_i_msoa(
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
        void *const RSTR outz,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_i_msoa(
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
        void *const RSTR outz,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa(
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
        void *const RSTR outy,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_gradient_i_msoa(
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
        void *const RSTR outy,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_i_msoa(
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
        void *const RSTR outz,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_i_msoa(
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
        void *const RSTR outz,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_i_msoa(
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
        void *const RSTR outz,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_i_msoa(
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
        void *const RSTR value,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_objective_i_msoa(
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
        void *const RSTR value,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_i_msoa(
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
        void *const RSTR value,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_i_msoa(
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
        void *const RSTR value,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_i_msoa(
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
        void *const RSTR value,
        void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_hessian_bsr_i_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_hessian_bsr_i_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_hessian_bsr_i_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_hessian_bsr_i_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_hessian_bsr_i_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_bsr_i_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_hessian_bsr_i_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    const void *const RSTR direction,
    void *const RSTR output,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    const void *const RSTR direction,
    void *const RSTR output,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_i_msoa(
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
    void *const RSTR u1_out,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_i_msoa(
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
    void *const RSTR u1_out,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    const void *const RSTR direction,
    void *const RSTR output,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    const void *const RSTR direction,
    void *const RSTR output,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    const void *const RSTR direction,
    void *const RSTR output,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_i_msoa(
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
    void *const RSTR u2_out,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_i_msoa(
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
    void *const RSTR u2_out,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa(
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
    void *const RSTR u2_out,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_i_msoa(
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
    void *const RSTR u1_out,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_i_msoa(
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
    void *const RSTR u1_out,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_i_msoa(
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
    void *const RSTR u2_out,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_i_msoa(
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
    void *const RSTR u2_out,
    void *const stream
);
extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa(
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
    void *const RSTR u2_out,
    void *const stream
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_i_msoa(
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
        void *const RSTR outy,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy, stream);
    case smesh::QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_apply_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_i_msoa(
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
        void *const RSTR outz,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz, stream);
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz, stream);
    case smesh::TET10:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_apply_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_i_msoa(
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
        void *const RSTR outy,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, out_stride, outx, outy, stream);
    case smesh::QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, out_stride, outx, outy, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_i_msoa(
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
        void *const RSTR outz,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz, stream);
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz, stream);
    case smesh::TET10:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_gradient_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_objective_2d_i_msoa(
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
        void *const RSTR value,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_quad4_objective_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, value, stream);
    case smesh::QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_quad4_objective_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, value, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_objective_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_elastic_objective_3d_i_msoa(
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
        void *const RSTR value,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_hex8_objective_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, value, stream);
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_proteus_hex8_objective_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, value, stream);
    case smesh::TET10:
      return cu_mooney_rivlin_kelvin_voigt_newmark_elastic_tet10_objective_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, value, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_elastic_objective_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_hessian_bsr_2d_i_msoa(
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
        void *const RSTR values,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, rowptr, colidx, values, stream);
    case smesh::QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, rowptr, colidx, values, stream);
    case smesh::TRI3:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_tri3_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, rowptr, colidx, values, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_hessian_bsr_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_hessian_bsr_3d_i_msoa(
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
        void *const RSTR values,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, rowptr, colidx, values, stream);
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, rowptr, colidx, values, stream);
    case smesh::TET10:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, rowptr, colidx, values, stream);
    case smesh::TET4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_tet4_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, rowptr, colidx, values, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_hessian_bsr_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_i_maos(
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
        void *const RSTR output,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, direction, output, stream);
    case smesh::QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, direction, output, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_i_msoa(
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
        void *const RSTR u1_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out, stream);
    case smesh::QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_i_maos(
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
        void *const RSTR output,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, direction, output, stream);
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, direction, output, stream);
    case smesh::TET10:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, direction, output, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_i_msoa(
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
        void *const RSTR u2_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out, stream);
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out, stream);
    case smesh::TET10:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_jacobian_action_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output, stream);
    case smesh::QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_i_msoa(
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
        void *const RSTR u1_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out, stream);
    case smesh::QUAD4:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output, stream);
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output, stream);
    case smesh::TET10:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, previous, output, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_i_msoa(
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
        void *const RSTR u2_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out, stream);
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_proteus_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out, stream);
    case smesh::TET10:
      return cu_mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_kelvin_voigt_newmark_viscous_residual_3d_i_msoa", (int)element_type, (int)real_type);
}
