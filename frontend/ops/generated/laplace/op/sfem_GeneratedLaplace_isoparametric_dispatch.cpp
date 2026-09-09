#include "sfem_GeneratedLaplace_c_abi.hpp"
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

extern "C" int laplace_proteus_quad4_apply_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_quad4_apply_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_quad4_apply_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_quad4_apply_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_hex8_apply_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_hex8_apply_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_apply_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_apply_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_tet10_apply_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_tet10_apply_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_quad4_gradient_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_quad4_gradient_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_quad4_gradient_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_quad4_gradient_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_hex8_gradient_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_hex8_gradient_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_tet10_gradient_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_tet10_gradient_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_quad4_hessian_bsr_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_proteus_quad4_hessian_bsr_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_quad4_hessian_bsr_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_quad4_hessian_bsr_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_tri3_hessian_bsr_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_tri3_hessian_bsr_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_hex8_hessian_bsr_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_hex8_hessian_bsr_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_proteus_hex8_hessian_bsr_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_proteus_hex8_hessian_bsr_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_tet10_hessian_bsr_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_tet10_hessian_bsr_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_tet4_hessian_bsr_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_tet4_hessian_bsr_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_proteus_quad4_hessian_crs_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_proteus_quad4_hessian_crs_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_quad4_hessian_crs_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_quad4_hessian_crs_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_tri3_hessian_crs_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_tri3_hessian_crs_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_hex8_hessian_crs_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_hex8_hessian_crs_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_proteus_hex8_hessian_crs_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_proteus_hex8_hessian_crs_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_tet10_hessian_crs_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_tet10_hessian_crs_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_tet4_hessian_crs_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        double *const RSTR values
);
extern "C" int laplace_tet4_hessian_crs_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        float *const RSTR values
);
extern "C" int laplace_proteus_quad4_objective_steps_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_proteus_quad4_objective_steps_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int laplace_quad4_objective_steps_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_quad4_objective_steps_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int laplace_hex8_objective_steps_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_hex8_objective_steps_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int laplace_proteus_hex8_objective_steps_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_proteus_hex8_objective_steps_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int laplace_tet10_objective_steps_i_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_tet10_objective_steps_i_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_apply_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_proteus_quad4_apply_i_msoa(nelements, nnodes, elements, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_proteus_quad4_apply_i_msoa_float(nelements, nnodes, elements, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    case smesh::QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_quad4_apply_i_msoa(nelements, nnodes, elements, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_quad4_apply_i_msoa_float(nelements, nnodes, elements, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_apply_2d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_apply_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_hex8_apply_i_msoa(nelements, nnodes, elements, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_hex8_apply_i_msoa_float(nelements, nnodes, elements, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_proteus_hex8_apply_i_msoa(nelements, nnodes, elements, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_proteus_hex8_apply_i_msoa_float(nelements, nnodes, elements, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tet10_apply_i_msoa(nelements, nnodes, elements, points, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_tet10_apply_i_msoa_float(nelements, nnodes, elements, points, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_apply_3d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_gradient_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_proteus_quad4_gradient_i_msoa(nelements, nnodes, elements, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_proteus_quad4_gradient_i_msoa_float(nelements, nnodes, elements, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    case smesh::QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_quad4_gradient_i_msoa(nelements, nnodes, elements, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_quad4_gradient_i_msoa_float(nelements, nnodes, elements, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_gradient_2d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_gradient_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_hex8_gradient_i_msoa(nelements, nnodes, elements, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_hex8_gradient_i_msoa_float(nelements, nnodes, elements, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_proteus_hex8_gradient_i_msoa(nelements, nnodes, elements, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_proteus_hex8_gradient_i_msoa_float(nelements, nnodes, elements, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tet10_gradient_i_msoa(nelements, nnodes, elements, points, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_tet10_gradient_i_msoa_float(nelements, nnodes, elements, points, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_gradient_3d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_bsr_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
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
          return laplace_proteus_quad4_hessian_bsr_i_msoa(nelements, nnodes, elements, points, kappa, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return laplace_proteus_quad4_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_quad4_hessian_bsr_i_msoa(nelements, nnodes, elements, points, kappa, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return laplace_quad4_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::TRI3: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tri3_hessian_bsr_i_msoa(nelements, nnodes, elements, points, kappa, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return laplace_tri3_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_hessian_bsr_2d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_bsr_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
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
          return laplace_hex8_hessian_bsr_i_msoa(nelements, nnodes, elements, points, kappa, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return laplace_hex8_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_proteus_hex8_hessian_bsr_i_msoa(nelements, nnodes, elements, points, kappa, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return laplace_proteus_hex8_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tet10_hessian_bsr_i_msoa(nelements, nnodes, elements, points, kappa, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return laplace_tet10_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::TET4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tet4_hessian_bsr_i_msoa(nelements, nnodes, elements, points, kappa, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return laplace_tet4_hessian_bsr_i_msoa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_hessian_bsr_3d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_crs_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
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
          return laplace_proteus_quad4_hessian_crs_i_msoa(nelements, nnodes, elements, points, kappa, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return laplace_proteus_quad4_hessian_crs_i_msoa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_quad4_hessian_crs_i_msoa(nelements, nnodes, elements, points, kappa, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return laplace_quad4_hessian_crs_i_msoa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::TRI3: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tri3_hessian_crs_i_msoa(nelements, nnodes, elements, points, kappa, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return laplace_tri3_hessian_crs_i_msoa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_hessian_crs_2d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_hessian_crs_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
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
          return laplace_hex8_hessian_crs_i_msoa(nelements, nnodes, elements, points, kappa, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return laplace_hex8_hessian_crs_i_msoa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_proteus_hex8_hessian_crs_i_msoa(nelements, nnodes, elements, points, kappa, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return laplace_proteus_hex8_hessian_crs_i_msoa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tet10_hessian_crs_i_msoa(nelements, nnodes, elements, points, kappa, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return laplace_tet10_hessian_crs_i_msoa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    case smesh::TET4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tet4_hessian_crs_i_msoa(nelements, nnodes, elements, points, kappa, rowptr, colidx, (double *)values);
        case smesh::SMESH_FLOAT32:
          return laplace_tet4_hessian_crs_i_msoa_float(nelements, nnodes, elements, points, kappa, rowptr, colidx, (float *)values);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_hessian_crs_3d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_objective_steps_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
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
          return laplace_proteus_quad4_objective_steps_i_msoa(nelements, nnodes, elements, points, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return laplace_proteus_quad4_objective_steps_i_msoa_float(nelements, nnodes, elements, points, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    case smesh::QUAD4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_quad4_objective_steps_i_msoa(nelements, nnodes, elements, points, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return laplace_quad4_objective_steps_i_msoa_float(nelements, nnodes, elements, points, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_objective_steps_2d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_objective_steps_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
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
          return laplace_hex8_objective_steps_i_msoa(nelements, nnodes, elements, points, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return laplace_hex8_objective_steps_i_msoa_float(nelements, nnodes, elements, points, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_proteus_hex8_objective_steps_i_msoa(nelements, nnodes, elements, points, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return laplace_proteus_hex8_objective_steps_i_msoa_float(nelements, nnodes, elements, points, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tet10_objective_steps_i_msoa(nelements, nnodes, elements, points, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return laplace_tet10_objective_steps_i_msoa_float(nelements, nnodes, elements, points, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_objective_steps_3d_i_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}
