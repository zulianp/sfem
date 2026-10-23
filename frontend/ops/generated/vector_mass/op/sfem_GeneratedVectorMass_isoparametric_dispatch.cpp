#include "sfem_GeneratedVectorMass_c_abi.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double),
              "the generated kernels select their scalar by width");
static_assert((int)smesh::SMESH_FLOAT32 == (int)sizeof(float),
              "the generated kernels select their scalar by width");

extern "C" int vector_mass_proteus_quad4_hessian_bsr_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values
);
extern "C" int vector_mass_quad4_hessian_bsr_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values
);
extern "C" int vector_mass_hex27_hessian_bsr_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values
);
extern "C" int vector_mass_hex8_hessian_bsr_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values
);
extern "C" int vector_mass_proteus_hex27_hessian_bsr_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values
);
extern "C" int vector_mass_proteus_hex8_hessian_bsr_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values
);
extern "C" int vector_mass_tet10_hessian_bsr_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values
);
extern "C" int vector_mass_proteus_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int vector_mass_quad4_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int vector_mass_proteus_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out
);
extern "C" int vector_mass_quad4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out
);
extern "C" int vector_mass_hex27_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int vector_mass_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int vector_mass_proteus_hex27_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int vector_mass_proteus_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int vector_mass_tet10_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR direction,
    void *const RSTR output
);
extern "C" int vector_mass_hex27_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const void *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
);
extern "C" int vector_mass_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const void *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
);
extern "C" int vector_mass_proteus_hex27_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const void *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
);
extern "C" int vector_mass_proteus_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const void *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
);
extern "C" int vector_mass_tet10_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const void *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
);
extern "C" int vector_mass_proteus_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    void *const RSTR output
);
extern "C" int vector_mass_quad4_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    void *const RSTR output
);
extern "C" int vector_mass_proteus_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out
);
extern "C" int vector_mass_quad4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out
);
extern "C" int vector_mass_hex27_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    void *const RSTR output
);
extern "C" int vector_mass_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    void *const RSTR output
);
extern "C" int vector_mass_proteus_hex27_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    void *const RSTR output
);
extern "C" int vector_mass_proteus_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    void *const RSTR output
);
extern "C" int vector_mass_tet10_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    void *const RSTR output
);
extern "C" int vector_mass_hex27_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
);
extern "C" int vector_mass_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
);
extern "C" int vector_mass_proteus_hex27_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
);
extern "C" int vector_mass_proteus_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
);
extern "C" int vector_mass_tet10_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int vector_mass_hessian_bsr_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
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
      return vector_mass_proteus_quad4_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, rowptr, colidx, values);
    case smesh::QUAD4:
      return vector_mass_quad4_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, rowptr, colidx, values);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "vector_mass_hessian_bsr_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int vector_mass_hessian_bsr_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return vector_mass_hex27_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, rowptr, colidx, values);
    case smesh::HEX8:
      return vector_mass_hex8_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, rowptr, colidx, values);
    case smesh::PROTEUS_HEX27:
      return vector_mass_proteus_hex27_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, rowptr, colidx, values);
    case smesh::PROTEUS_HEX8:
      return vector_mass_proteus_hex8_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, rowptr, colidx, values);
    case smesh::TET10:
      return vector_mass_tet10_hessian_bsr_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, rowptr, colidx, values);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "vector_mass_hessian_bsr_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int vector_mass_jacobian_action_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR direction,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return vector_mass_proteus_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, direction, output);
    case smesh::QUAD4:
      return vector_mass_quad4_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, direction, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "vector_mass_jacobian_action_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int vector_mass_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
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
      return vector_mass_proteus_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out);
    case smesh::QUAD4:
      return vector_mass_quad4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "vector_mass_jacobian_action_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int vector_mass_jacobian_action_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR direction,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return vector_mass_hex27_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, direction, output);
    case smesh::HEX8:
      return vector_mass_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, direction, output);
    case smesh::PROTEUS_HEX27:
      return vector_mass_proteus_hex27_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, direction, output);
    case smesh::PROTEUS_HEX8:
      return vector_mass_proteus_hex8_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, direction, output);
    case smesh::TET10:
      return vector_mass_tet10_jacobian_action_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, direction, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "vector_mass_jacobian_action_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int vector_mass_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
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
    case smesh::HEX27:
      return vector_mass_hex27_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
    case smesh::HEX8:
      return vector_mass_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
    case smesh::PROTEUS_HEX27:
      return vector_mass_proteus_hex27_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
    case smesh::PROTEUS_HEX8:
      return vector_mass_proteus_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
    case smesh::TET10:
      return vector_mass_tet10_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "vector_mass_jacobian_action_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int vector_mass_residual_2d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::PROTEUS_QUAD4:
      return vector_mass_proteus_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, output);
    case smesh::QUAD4:
      return vector_mass_quad4_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "vector_mass_residual_2d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int vector_mass_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
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
      return vector_mass_proteus_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, current_stride, u0, u1, out_stride, u0_out, u1_out);
    case smesh::QUAD4:
      return vector_mass_quad4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, current_stride, u0, u1, out_stride, u0_out, u1_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "vector_mass_residual_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int vector_mass_residual_3d_i_maos(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        void *const RSTR output
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return vector_mass_hex27_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, output);
    case smesh::HEX8:
      return vector_mass_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, output);
    case smesh::PROTEUS_HEX27:
      return vector_mass_proteus_hex27_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, output);
    case smesh::PROTEUS_HEX8:
      return vector_mass_proteus_hex8_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, output);
    case smesh::TET10:
      return vector_mass_tet10_residual_i_maos((int)resolved_real_type, nelements, nnodes, elements, points, parameters, current, output);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "vector_mass_residual_3d_i_maos", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int vector_mass_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
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
    case smesh::HEX27:
      return vector_mass_hex27_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, current_stride, u0, u1, u2, out_stride, u0_out, u1_out, u2_out);
    case smesh::HEX8:
      return vector_mass_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, current_stride, u0, u1, u2, out_stride, u0_out, u1_out, u2_out);
    case smesh::PROTEUS_HEX27:
      return vector_mass_proteus_hex27_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, current_stride, u0, u1, u2, out_stride, u0_out, u1_out, u2_out);
    case smesh::PROTEUS_HEX8:
      return vector_mass_proteus_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, current_stride, u0, u1, u2, out_stride, u0_out, u1_out, u2_out);
    case smesh::TET10:
      return vector_mass_tet10_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, density, current_stride, u0, u1, u2, out_stride, u0_out, u1_out, u2_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "vector_mass_residual_3d_i_msoa", (int)element_type, (int)real_type);
}
