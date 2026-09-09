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

extern "C" int laplace_tri3_apply_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_tri3_apply_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_hex8_apply_a_msoa(
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
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_hex8_apply_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_apply_a_msoa(
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
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_apply_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_tet10_apply_a_msoa(
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
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_tet10_apply_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_tet4_apply_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const double kappa,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_tet4_apply_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const float kappa,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_tri3_gradient_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_tri3_gradient_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_hex8_gradient_a_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_hex8_gradient_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_a_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_proteus_hex8_gradient_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_tet10_gradient_a_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_tet10_gradient_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_tet4_gradient_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t out_stride,
        double *const RSTR outx
);
extern "C" int laplace_tet4_gradient_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t out_stride,
        float *const RSTR outx
);
extern "C" int laplace_tri3_objective_steps_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_tri3_objective_steps_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int laplace_hex8_objective_steps_a_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_hex8_objective_steps_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int laplace_proteus_hex8_objective_steps_a_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_proteus_hex8_objective_steps_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int laplace_tet10_objective_steps_a_msoa(
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
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_tet10_objective_steps_a_msoa_float(
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
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);
extern "C" int laplace_tet4_objective_steps_a_msoa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const double kappa,
        const ptrdiff_t u_stride,
        const double *const RSTR ux,
        const ptrdiff_t h_stride,
        const double *const RSTR hx,
        const int nsteps,
        const double *const RSTR steps,
        double *const RSTR value
);
extern "C" int laplace_tet4_objective_steps_a_msoa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const float kappa,
        const ptrdiff_t u_stride,
        const float *const RSTR ux,
        const ptrdiff_t h_stride,
        const float *const RSTR hx,
        const int nsteps,
        const float *const RSTR steps,
        float *const RSTR value
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_apply_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
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
    case smesh::TRI3: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tri3_apply_a_msoa(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_tri3_apply_a_msoa_float(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_apply_2d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_apply_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
          return laplace_hex8_apply_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_hex8_apply_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_proteus_hex8_apply_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_proteus_hex8_apply_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tet10_apply_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_tet10_apply_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_apply_3d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_apply_3d_a_met_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
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
    case smesh::TET4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tet4_apply_a_msoa(nelements, nnodes, elements, g_met0, g_met1, g_met2, g_met3, g_met4, g_met5, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_tet4_apply_a_msoa_float(nelements, nnodes, elements, g_met0, g_met1, g_met2, g_met3, g_met4, g_met5, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_apply_3d_a_met_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_gradient_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
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
    case smesh::TRI3: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tri3_gradient_a_msoa(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_tri3_gradient_a_msoa_float(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_gradient_2d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_gradient_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
          return laplace_hex8_gradient_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_hex8_gradient_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_proteus_hex8_gradient_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_proteus_hex8_gradient_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tet10_gradient_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_tet10_gradient_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_gradient_3d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_gradient_3d_a_met_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
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
    case smesh::TET4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tet4_gradient_a_msoa(nelements, nnodes, elements, g_met0, g_met1, g_met2, g_met3, g_met4, g_met5, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
        case smesh::SMESH_FLOAT32:
          return laplace_tet4_gradient_a_msoa_float(nelements, nnodes, elements, g_met0, g_met1, g_met2, g_met3, g_met4, g_met5, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_gradient_3d_a_met_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_objective_steps_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
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
    case smesh::TRI3: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tri3_objective_steps_a_msoa(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return laplace_tri3_objective_steps_a_msoa_float(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_objective_steps_2d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_objective_steps_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
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
          return laplace_hex8_objective_steps_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return laplace_hex8_objective_steps_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    case smesh::PROTEUS_HEX8: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_proteus_hex8_objective_steps_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return laplace_proteus_hex8_objective_steps_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    case smesh::TET10: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tet10_objective_steps_a_msoa(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return laplace_tet10_objective_steps_a_msoa_float(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_objective_steps_3d_a_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int laplace_objective_steps_3d_a_met_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
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
    case smesh::TET4: {
      switch (resolved_real_type) {
        case smesh::SMESH_FLOAT64:
          return laplace_tet4_objective_steps_a_msoa(nelements, nnodes, elements, g_met0, g_met1, g_met2, g_met3, g_met4, g_met5, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
        case smesh::SMESH_FLOAT32:
          return laplace_tet4_objective_steps_a_msoa_float(nelements, nnodes, elements, g_met0, g_met1, g_met2, g_met3, g_met4, g_met5, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  std::fprintf(stderr,
      "laplace_objective_steps_3d_a_met_msoa does not support element type %d with real type %d\n",
      (int)element_type,
      (int)real_type);
  return SFEM_FAILURE;
}
