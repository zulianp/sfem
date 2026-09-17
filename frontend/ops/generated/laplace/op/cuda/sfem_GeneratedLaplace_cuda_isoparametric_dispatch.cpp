#include "sfem_GeneratedLaplace_cuda_c_abi.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double),
              "the generated kernels select their scalar by width");
static_assert((int)smesh::SMESH_FLOAT32 == (int)sizeof(float),
              "the generated kernels select their scalar by width");

extern "C" int cu_laplace_proteus_quad4_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
);
extern "C" int cu_laplace_quad4_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
);
extern "C" int cu_laplace_hex8_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
);
extern "C" int cu_laplace_proteus_hex8_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
);
extern "C" int cu_laplace_tet10_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
);
extern "C" int cu_laplace_proteus_quad4_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
);
extern "C" int cu_laplace_quad4_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
);
extern "C" int cu_laplace_hex8_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
);
extern "C" int cu_laplace_proteus_hex8_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
);
extern "C" int cu_laplace_tet10_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
);
extern "C" int cu_laplace_proteus_quad4_objective_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        void *const RSTR value,
        void *const stream
);
extern "C" int cu_laplace_quad4_objective_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        void *const RSTR value,
        void *const stream
);
extern "C" int cu_laplace_hex8_objective_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        void *const RSTR value,
        void *const stream
);
extern "C" int cu_laplace_proteus_hex8_objective_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        void *const RSTR value,
        void *const stream
);
extern "C" int cu_laplace_tet10_objective_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        void *const RSTR value,
        void *const stream
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_laplace_apply_2d_i_msoa(
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
        void *const RSTR outx,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return cu_laplace_proteus_quad4_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, h_stride, hx, out_stride, outx, stream);
    case smesh::QUAD4:
      return cu_laplace_quad4_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, h_stride, hx, out_stride, outx, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "laplace_apply_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_laplace_apply_3d_i_msoa(
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
        void *const RSTR outx,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_laplace_hex8_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, h_stride, hx, out_stride, outx, stream);
    case smesh::PROTEUS_HEX8:
      return cu_laplace_proteus_hex8_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, h_stride, hx, out_stride, outx, stream);
    case smesh::TET10:
      return cu_laplace_tet10_apply_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, h_stride, hx, out_stride, outx, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "laplace_apply_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_laplace_gradient_2d_i_msoa(
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
        void *const RSTR outx,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return cu_laplace_proteus_quad4_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, u_stride, ux, out_stride, outx, stream);
    case smesh::QUAD4:
      return cu_laplace_quad4_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, u_stride, ux, out_stride, outx, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "laplace_gradient_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_laplace_gradient_3d_i_msoa(
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
        void *const RSTR outx,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_laplace_hex8_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, u_stride, ux, out_stride, outx, stream);
    case smesh::PROTEUS_HEX8:
      return cu_laplace_proteus_hex8_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, u_stride, ux, out_stride, outx, stream);
    case smesh::TET10:
      return cu_laplace_tet10_gradient_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, u_stride, ux, out_stride, outx, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "laplace_gradient_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_laplace_objective_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        void *const RSTR value,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return cu_laplace_proteus_quad4_objective_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, u_stride, ux, value, stream);
    case smesh::QUAD4:
      return cu_laplace_quad4_objective_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, u_stride, ux, value, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "laplace_objective_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_laplace_objective_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        void *const RSTR value,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX8:
      return cu_laplace_hex8_objective_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, u_stride, ux, value, stream);
    case smesh::PROTEUS_HEX8:
      return cu_laplace_proteus_hex8_objective_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, u_stride, ux, value, stream);
    case smesh::TET10:
      return cu_laplace_tet10_objective_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, kappa, u_stride, ux, value, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "laplace_objective_3d_i_msoa", (int)element_type, (int)real_type);
}
