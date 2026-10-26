#include "sfem_GeneratedBodyForce_cuda_c_abi.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double),
              "the generated kernels select their scalar by width");
static_assert((int)smesh::SMESH_FLOAT32 == (int)sizeof(float),
              "the generated kernels select their scalar by width");

extern "C" int cu_body_force_proteus_quad4_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const stream
);
extern "C" int cu_body_force_quad4_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const stream
);
extern "C" int cu_body_force_tri3_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const stream
);
extern "C" int cu_body_force_hex8_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const real_t g2,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out,
    void *const stream
);
extern "C" int cu_body_force_proteus_hex8_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const real_t g2,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out,
    void *const stream
);
extern "C" int cu_body_force_tet10_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const real_t g2,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out,
    void *const stream
);
extern "C" int cu_body_force_tet4_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const real_t g2,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out,
    void *const stream
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_body_force_residual_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const real_t g0,
        const real_t g1,
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
      return cu_body_force_proteus_quad4_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_det0, density, g0, g1, out_stride, u0_out, u1_out, stream);
    case smesh::QUAD4:
      return cu_body_force_quad4_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_det0, density, g0, g1, out_stride, u0_out, u1_out, stream);
    case smesh::TRI3:
      return cu_body_force_tri3_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_det0, density, g0, g1, out_stride, u0_out, u1_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "body_force_residual_2d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_body_force_residual_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const real_t g2,
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
      return cu_body_force_hex8_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_det0, density, g0, g1, g2, out_stride, u0_out, u1_out, u2_out, stream);
    case smesh::PROTEUS_HEX8:
      return cu_body_force_proteus_hex8_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_det0, density, g0, g1, g2, out_stride, u0_out, u1_out, u2_out, stream);
    case smesh::TET10:
      return cu_body_force_tet10_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_det0, density, g0, g1, g2, out_stride, u0_out, u1_out, u2_out, stream);
    case smesh::TET4:
      return cu_body_force_tet4_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_det0, density, g0, g1, g2, out_stride, u0_out, u1_out, u2_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "body_force_residual_3d_a_msoa", (int)element_type, (int)real_type);
}
