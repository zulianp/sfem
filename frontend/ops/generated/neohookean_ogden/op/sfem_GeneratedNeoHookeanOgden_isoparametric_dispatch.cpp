#include "sfem_GeneratedNeoHookeanOgden_c_abi.hpp"
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

extern "C" int neohookean_ogden_proteus_quad4_apply_i_msoa(
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
extern "C" int neohookean_ogden_proteus_quad4_apply_i_msoa_float(
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
extern "C" int neohookean_ogden_quad4_apply_i_msoa(
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
extern "C" int neohookean_ogden_quad4_apply_i_msoa_float(
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
extern "C" int neohookean_ogden_hex8_apply_i_msoa(
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
extern "C" int neohookean_ogden_hex8_apply_i_msoa_float(
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
extern "C" int neohookean_ogden_proteus_hex8_apply_i_msoa(
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
extern "C" int neohookean_ogden_proteus_hex8_apply_i_msoa_float(
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
extern "C" int neohookean_ogden_tet10_apply_i_msoa(
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
extern "C" int neohookean_ogden_tet10_apply_i_msoa_float(
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
extern "C" int neohookean_ogden_proteus_quad4_gradient_i_msoa(
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
extern "C" int neohookean_ogden_proteus_quad4_gradient_i_msoa_float(
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
extern "C" int neohookean_ogden_quad4_gradient_i_msoa(
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
extern "C" int neohookean_ogden_quad4_gradient_i_msoa_float(
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
extern "C" int neohookean_ogden_hex8_gradient_i_msoa(
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
extern "C" int neohookean_ogden_hex8_gradient_i_msoa_float(
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
extern "C" int neohookean_ogden_proteus_hex8_gradient_i_msoa(
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
extern "C" int neohookean_ogden_proteus_hex8_gradient_i_msoa_float(
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
extern "C" int neohookean_ogden_tet10_gradient_i_msoa(
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
extern "C" int neohookean_ogden_tet10_gradient_i_msoa_float(
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
extern "C" int neohookean_ogden_proteus_quad4_hessian_bsr_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int neohookean_ogden_proteus_quad4_hessian_bsr_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int neohookean_ogden_quad4_hessian_bsr_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int neohookean_ogden_quad4_hessian_bsr_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int neohookean_ogden_tri3_hessian_bsr_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const double *const RSTR uy,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int neohookean_ogden_tri3_hessian_bsr_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const float *const RSTR uy,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int neohookean_ogden_hex8_hessian_bsr_i_msoa(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int neohookean_ogden_hex8_hessian_bsr_i_msoa_float(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int neohookean_ogden_proteus_hex8_hessian_bsr_i_msoa(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int neohookean_ogden_proteus_hex8_hessian_bsr_i_msoa_float(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int neohookean_ogden_tet10_hessian_bsr_i_msoa(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int neohookean_ogden_tet10_hessian_bsr_i_msoa_float(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int neohookean_ogden_tet4_hessian_bsr_i_msoa(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int neohookean_ogden_tet4_hessian_bsr_i_msoa_float(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int neohookean_ogden_proteus_quad4_objective_steps_i_msoa(
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
extern "C" int neohookean_ogden_proteus_quad4_objective_steps_i_msoa_float(
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
extern "C" int neohookean_ogden_quad4_objective_steps_i_msoa(
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
extern "C" int neohookean_ogden_quad4_objective_steps_i_msoa_float(
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
extern "C" int neohookean_ogden_hex8_objective_steps_i_msoa(
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
extern "C" int neohookean_ogden_hex8_objective_steps_i_msoa_float(
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
extern "C" int neohookean_ogden_proteus_hex8_objective_steps_i_msoa(
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
extern "C" int neohookean_ogden_proteus_hex8_objective_steps_i_msoa_float(
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
extern "C" int neohookean_ogden_tet10_objective_steps_i_msoa(
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
extern "C" int neohookean_ogden_tet10_objective_steps_i_msoa_float(
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

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neohookean_ogden_apply_2d_i_msoa(
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
          return neohookean_ogden_proteus_quad4_apply_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_proteus_quad4_apply_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
        default:
          break;
      }
      break;
    }
    case smesh::QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_quad4_apply_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_quad4_apply_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "neohookean_ogden_apply_2d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neohookean_ogden_apply_3d_i_msoa(
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
          return neohookean_ogden_hex8_apply_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_hex8_apply_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_proteus_hex8_apply_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_proteus_hex8_apply_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_tet10_apply_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_tet10_apply_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "neohookean_ogden_apply_3d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neohookean_ogden_gradient_2d_i_msoa(
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
          return neohookean_ogden_proteus_quad4_gradient_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_proteus_quad4_gradient_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
        default:
          break;
      }
      break;
    }
    case smesh::QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_quad4_gradient_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_quad4_gradient_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "neohookean_ogden_gradient_2d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neohookean_ogden_gradient_3d_i_msoa(
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
          return neohookean_ogden_hex8_gradient_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_hex8_gradient_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_proteus_hex8_gradient_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_proteus_hex8_gradient_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_tet10_gradient_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_tet10_gradient_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "neohookean_ogden_gradient_3d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neohookean_ogden_hessian_bsr_2d_i_msoa(
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
    case smesh::PROTEUS_QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_proteus_quad4_hessian_bsr_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_proteus_quad4_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_quad4_hessian_bsr_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_quad4_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::TRI3: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_tri3_hessian_bsr_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_tri3_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "neohookean_ogden_hessian_bsr_2d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neohookean_ogden_hessian_bsr_3d_i_msoa(
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
    case smesh::HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_hex8_hessian_bsr_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_hex8_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_proteus_hex8_hessian_bsr_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_proteus_hex8_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_tet10_hessian_bsr_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_tet10_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::TET4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_tet4_hessian_bsr_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_tet4_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "neohookean_ogden_hessian_bsr_3d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neohookean_ogden_objective_steps_2d_i_msoa(
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
          return neohookean_ogden_proteus_quad4_objective_steps_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_proteus_quad4_objective_steps_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    case smesh::QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_quad4_objective_steps_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_quad4_objective_steps_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "neohookean_ogden_objective_steps_2d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int neohookean_ogden_objective_steps_3d_i_msoa(
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
          return neohookean_ogden_hex8_objective_steps_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_hex8_objective_steps_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_proteus_hex8_objective_steps_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_proteus_hex8_objective_steps_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return neohookean_ogden_tet10_objective_steps_i_msoa(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return neohookean_ogden_tet10_objective_steps_i_msoa_float(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "neohookean_ogden_objective_steps_3d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}
