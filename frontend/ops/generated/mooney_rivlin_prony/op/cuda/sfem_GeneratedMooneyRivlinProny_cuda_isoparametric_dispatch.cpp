#include "sfem_GeneratedMooneyRivlinProny_cuda_c_abi.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double),
              "the generated kernels select their scalar by width");
static_assert((int)smesh::SMESH_FLOAT32 == (int)sizeof(float),
              "the generated kernels select their scalar by width");

extern "C" int cu_mooney_rivlin_prony_hex8_apply_i_msoa(
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
        void *const RSTR outz,
        void *const stream
);
extern "C" int cu_mooney_rivlin_prony_proteus_hex8_apply_i_msoa(
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
        void *const RSTR outz,
        void *const stream
);
extern "C" int cu_mooney_rivlin_prony_hex8_gradient_i_msoa(
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
        void *const RSTR outz,
        void *const stream
);
extern "C" int cu_mooney_rivlin_prony_proteus_hex8_gradient_i_msoa(
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
        void *const RSTR outz,
        void *const stream
);
extern "C" int cu_mooney_rivlin_prony_hex8_hessian_bsr_i_msoa(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values,
        void *const stream
);
extern "C" int cu_mooney_rivlin_prony_proteus_hex8_hessian_bsr_i_msoa(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values,
        void *const stream
);
extern "C" int cu_mooney_rivlin_prony_hex8_objective_steps_i_msoa(
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
        void *const RSTR value,
        void *const stream
);
extern "C" int cu_mooney_rivlin_prony_proteus_hex8_objective_steps_i_msoa(
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
        void *const RSTR value,
        void *const stream
);
extern "C" int cu_mooney_rivlin_prony_hex8_quadrature_output_i_msoa(
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
        const void *const RSTR uz,
        void *const stream
);
extern "C" int cu_mooney_rivlin_prony_proteus_hex8_quadrature_output_i_msoa(
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
        const void *const RSTR uz,
        void *const stream
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_prony_apply_3d_i_msoa(
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
        void *const RSTR outz,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_mooney_rivlin_prony_hex8_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz, stream);
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_prony_proteus_hex8_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_apply_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_prony_gradient_3d_i_msoa(
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
        void *const RSTR outz,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_mooney_rivlin_prony_hex8_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, out_stride, outx, outy, outz, stream);
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_prony_proteus_hex8_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, out_stride, outx, outy, outz, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_gradient_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_prony_hessian_bsr_3d_i_msoa(
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
      return cu_mooney_rivlin_prony_hex8_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, rowptr, colidx, values, stream);
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_prony_proteus_hex8_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, rowptr, colidx, values, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_hessian_bsr_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_prony_objective_steps_3d_i_msoa(
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
        void *const RSTR value,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_mooney_rivlin_prony_hex8_objective_steps_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value, stream);
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_prony_proteus_hex8_objective_steps_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, K, iso_relaxation_gamma, iso_history, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_objective_steps_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_mooney_rivlin_prony_quadrature_output_3d_i_msoa(
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
        const void *const RSTR uz,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_mooney_rivlin_prony_hex8_quadrature_output_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, iso_stress, u_stride, ux, uy, uz, stream);
    case smesh::PROTEUS_HEX8:
      return cu_mooney_rivlin_prony_proteus_hex8_quadrature_output_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, C01, C10, iso_stress, u_stride, ux, uy, uz, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "mooney_rivlin_prony_quadrature_output_3d_i_msoa", (int)element_type, (int)real_type);
}
