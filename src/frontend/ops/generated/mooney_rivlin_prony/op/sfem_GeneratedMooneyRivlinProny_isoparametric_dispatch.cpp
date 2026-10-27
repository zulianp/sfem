#include "sfem_GeneratedMooneyRivlinProny_c_abi.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double),
              "the generated kernels select their scalar by width");
static_assert((int)smesh::SMESH_FLOAT32 == (int)sizeof(float),
              "the generated kernels select their scalar by width");

extern "C" int mooney_rivlin_prony_proteus_quad4_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
extern "C" int mooney_rivlin_prony_quad4_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
extern "C" int mooney_rivlin_prony_hex8_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
extern "C" int mooney_rivlin_prony_proteus_hex8_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
extern "C" int mooney_rivlin_prony_tet10_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
extern "C" int mooney_rivlin_prony_proteus_quad4_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
);
extern "C" int mooney_rivlin_prony_quad4_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy
);
extern "C" int mooney_rivlin_prony_hex8_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);
extern "C" int mooney_rivlin_prony_proteus_hex8_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);
extern "C" int mooney_rivlin_prony_tet10_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz
);
extern "C" int mooney_rivlin_prony_proteus_quad4_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const int values_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);
extern "C" int mooney_rivlin_prony_quad4_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const int values_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);
extern "C" int mooney_rivlin_prony_hex8_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const int values_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);
extern "C" int mooney_rivlin_prony_proteus_hex8_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const int values_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);
extern "C" int mooney_rivlin_prony_tet10_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const int values_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
);
extern "C" int mooney_rivlin_prony_proteus_quad4_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
extern "C" int mooney_rivlin_prony_quad4_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
extern "C" int mooney_rivlin_prony_hex8_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
extern "C" int mooney_rivlin_prony_proteus_hex8_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
extern "C" int mooney_rivlin_prony_tet10_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
extern "C" int mooney_rivlin_prony_proteus_quad4_quadrature_output_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        void *const RSTR iso_stress,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy
);
extern "C" int mooney_rivlin_prony_quad4_quadrature_output_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        void *const RSTR iso_stress,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy
);
extern "C" int mooney_rivlin_prony_hex8_quadrature_output_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        void *const RSTR iso_stress,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz
);
extern "C" int mooney_rivlin_prony_proteus_hex8_quadrature_output_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        void *const RSTR iso_stress,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz
);
extern "C" int mooney_rivlin_prony_tet10_quadrature_output_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        void *const RSTR iso_stress,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_prony_apply_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
      return mooney_rivlin_prony_proteus_quad4_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
    case smesh::QUAD4:
      return mooney_rivlin_prony_quad4_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, h_stride, hx, hy, out_stride, outx, outy);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_apply_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_prony_apply_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
      return mooney_rivlin_prony_hex8_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_prony_proteus_hex8_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
    case smesh::TET10:
      return mooney_rivlin_prony_tet10_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_apply_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_prony_gradient_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
      return mooney_rivlin_prony_proteus_quad4_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, out_stride, outx, outy);
    case smesh::QUAD4:
      return mooney_rivlin_prony_quad4_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, out_stride, outx, outy);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_gradient_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_prony_gradient_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
      return mooney_rivlin_prony_hex8_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_prony_proteus_hex8_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
    case smesh::TET10:
      return mooney_rivlin_prony_tet10_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_gradient_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_prony_hessian_bsr_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const enum smesh::PrimitiveType values_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
  const enum smesh::PrimitiveType resolved_values_type =
      (values_type == smesh::SMESH_DEFAULT)
          ? resolved_real_type
          : values_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return mooney_rivlin_prony_proteus_quad4_hessian_bsr_i_msoa((int)resolved_real_type, (int)resolved_values_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, rowptr, colidx, values);
    case smesh::QUAD4:
      return mooney_rivlin_prony_quad4_hessian_bsr_i_msoa((int)resolved_real_type, (int)resolved_values_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, rowptr, colidx, values);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_hessian_bsr_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_prony_hessian_bsr_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const enum smesh::PrimitiveType values_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
  const enum smesh::PrimitiveType resolved_values_type =
      (values_type == smesh::SMESH_DEFAULT)
          ? resolved_real_type
          : values_type;
  switch (element_type) {
    case smesh::HEX8:
      return mooney_rivlin_prony_hex8_hessian_bsr_i_msoa((int)resolved_real_type, (int)resolved_values_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, rowptr, colidx, values);
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_prony_proteus_hex8_hessian_bsr_i_msoa((int)resolved_real_type, (int)resolved_values_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, rowptr, colidx, values);
    case smesh::TET10:
      return mooney_rivlin_prony_tet10_hessian_bsr_i_msoa((int)resolved_real_type, (int)resolved_values_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, rowptr, colidx, values);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_hessian_bsr_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_prony_objective_steps_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
      return mooney_rivlin_prony_proteus_quad4_objective_steps_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
    case smesh::QUAD4:
      return mooney_rivlin_prony_quad4_objective_steps_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, h_stride, hx, hy, nsteps, steps, value);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_objective_steps_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_prony_objective_steps_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        const real_t K,
        const real_t iso_relaxation_gamma,
        const void *const RSTR iso_history,
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
      return mooney_rivlin_prony_hex8_objective_steps_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_prony_proteus_hex8_objective_steps_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
    case smesh::TET10:
      return mooney_rivlin_prony_tet10_objective_steps_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_objective_steps_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_prony_quadrature_output_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        void *const RSTR iso_stress,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return mooney_rivlin_prony_proteus_quad4_quadrature_output_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, iso_stress, u_stride, ux, uy);
    case smesh::QUAD4:
      return mooney_rivlin_prony_quad4_quadrature_output_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, iso_stress, u_stride, ux, uy);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_quadrature_output_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int mooney_rivlin_prony_quadrature_output_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t C01,
        const real_t C10,
        void *const RSTR iso_stress,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return mooney_rivlin_prony_hex8_quadrature_output_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, iso_stress, u_stride, ux, uy, uz);
    case smesh::PROTEUS_HEX8:
      return mooney_rivlin_prony_proteus_hex8_quadrature_output_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, iso_stress, u_stride, ux, uy, uz);
    case smesh::TET10:
      return mooney_rivlin_prony_tet10_quadrature_output_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, iso_stress, u_stride, ux, uy, uz);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_quadrature_output_3d_i_msoa", (int)element_type, (int)real_type);
}
