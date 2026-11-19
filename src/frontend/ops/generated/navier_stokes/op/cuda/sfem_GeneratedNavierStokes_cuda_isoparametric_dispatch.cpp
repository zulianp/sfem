#include "sfem_GeneratedNavierStokes_cuda_c_abi.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double),
              "the generated kernels select their scalar by width");
static_assert((int)smesh::SMESH_FLOAT32 == (int)sizeof(float),
              "the generated kernels select their scalar by width");

extern "C" int cu_navier_stokes_form_1_p_tri6_tri3_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[2],
    const void *const RSTR p_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[2],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_1_p_hex27_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[3],
    const void *const RSTR p_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_1_p_proteus_hex27_proteus_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[3],
    const void *const RSTR p_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_1_p_tet10_tet4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[3],
    const void *const RSTR p_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_1_u_tri6_tri3_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t f0,
    const real_t f1,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[2],
    const void *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[2],
    const void *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[2],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_1_u_hex27_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t f0,
    const real_t f1,
    const real_t f2,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[3],
    const void *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const void *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t f0,
    const real_t f1,
    const real_t f2,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[3],
    const void *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const void *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_1_u_tet10_tet4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t f0,
    const real_t f1,
    const real_t f2,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[3],
    const void *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const void *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[2],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[2],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_2_p_u_proteus_hex27_proteus_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[2],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[2],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_2_u_p_proteus_hex27_proteus_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_form_2_u_u_tri6_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[2],
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[2],
    const ptrdiff_t out_stride,
    void *const RSTR u_out[2],
    void *const stream
);
extern "C" int cu_navier_stokes_form_2_u_u_hex27_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const stream
);
extern "C" int cu_navier_stokes_form_2_u_u_proteus_hex27_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const stream
);
extern "C" int cu_navier_stokes_form_2_u_u_tet10_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const stream
);
extern "C" int cu_navier_stokes_tri6_tri3_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[2],
    const void *const RSTR p_old_data,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[2],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[2],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_hex27_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const void *const RSTR p_old_data,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_proteus_hex27_proteus_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const void *const RSTR p_old_data,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_tet10_tet4_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const void *const RSTR p_old_data,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_tri6_tri3_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t f0,
    const real_t f1,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[2],
    const void *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[2],
    const void *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[2],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_hex27_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t f0,
    const real_t f1,
    const real_t f2,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[3],
    const void *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const void *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_proteus_hex27_proteus_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t f0,
    const real_t f1,
    const real_t f2,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[3],
    const void *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const void *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);
extern "C" int cu_navier_stokes_tet10_tet4_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t f0,
    const real_t f1,
    const real_t f2,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[3],
    const void *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const void *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_navier_stokes_form_1_p_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[2],
        const void *const RSTR p_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI6:
      return cu_navier_stokes_form_1_p_tri6_tri3_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, current_stride, u_data, p_data, out_stride, u_out, p_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_1_p_residual_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_navier_stokes_form_1_p_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[3],
        const void *const RSTR p_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return cu_navier_stokes_form_1_p_hex27_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, current_stride, u_data, p_data, out_stride, u_out, p_out, stream);
    case smesh::PROTEUS_HEX27:
      return cu_navier_stokes_form_1_p_proteus_hex27_proteus_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, current_stride, u_data, p_data, out_stride, u_out, p_out, stream);
    case smesh::TET10:
      return cu_navier_stokes_form_1_p_tet10_tet4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, current_stride, u_data, p_data, out_stride, u_out, p_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_1_p_residual_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_navier_stokes_form_1_u_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[2],
        const void *const RSTR p_data,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[2],
        const void *const RSTR p_old_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI6:
      return cu_navier_stokes_form_1_u_tri6_tri3_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, f0, f1, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_1_u_residual_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_navier_stokes_form_1_u_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t f2,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[3],
        const void *const RSTR p_data,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[3],
        const void *const RSTR p_old_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return cu_navier_stokes_form_1_u_hex27_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out, stream);
    case smesh::PROTEUS_HEX27:
      return cu_navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out, stream);
    case smesh::TET10:
      return cu_navier_stokes_form_1_u_tet10_tet4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_1_u_residual_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_navier_stokes_form_2_p_u_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI6:
      return cu_navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_2_p_u_jacobian_action_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_navier_stokes_form_2_p_u_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return cu_navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out, stream);
    case smesh::PROTEUS_HEX27:
      return cu_navier_stokes_form_2_p_u_proteus_hex27_proteus_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out, stream);
    case smesh::TET10:
      return cu_navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_2_p_u_jacobian_action_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_navier_stokes_form_2_u_p_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI6:
      return cu_navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_2_u_p_jacobian_action_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_navier_stokes_form_2_u_p_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return cu_navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out, stream);
    case smesh::PROTEUS_HEX27:
      return cu_navier_stokes_form_2_u_p_proteus_hex27_proteus_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out, stream);
    case smesh::TET10:
      return cu_navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_2_u_p_jacobian_action_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_navier_stokes_form_2_u_u_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[2],
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI6:
      return cu_navier_stokes_form_2_u_u_tri6_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_2_u_u_jacobian_action_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_navier_stokes_form_2_u_u_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[3],
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return cu_navier_stokes_form_2_u_u_hex27_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out, stream);
    case smesh::PROTEUS_HEX27:
      return cu_navier_stokes_form_2_u_u_proteus_hex27_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out, stream);
    case smesh::TET10:
      return cu_navier_stokes_form_2_u_u_tet10_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_2_u_u_jacobian_action_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_navier_stokes_jacobian_action_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[2],
        const void *const RSTR p_old_data,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI6:
      return cu_navier_stokes_tri6_tri3_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, nu, rho, previous_stride, u_old_data, p_old_data, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_jacobian_action_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_navier_stokes_jacobian_action_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[3],
        const void *const RSTR p_old_data,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return cu_navier_stokes_hex27_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, nu, rho, previous_stride, u_old_data, p_old_data, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out, stream);
    case smesh::PROTEUS_HEX27:
      return cu_navier_stokes_proteus_hex27_proteus_hex8_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, nu, rho, previous_stride, u_old_data, p_old_data, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out, stream);
    case smesh::TET10:
      return cu_navier_stokes_tet10_tet4_jacobian_action_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, nu, rho, previous_stride, u_old_data, p_old_data, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_jacobian_action_3d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_navier_stokes_residual_2d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[2],
        const void *const RSTR p_data,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[2],
        const void *const RSTR p_old_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI6:
      return cu_navier_stokes_tri6_tri3_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, f0, f1, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_residual_2d_i_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int cu_navier_stokes_residual_3d_i_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t convection_scale,
        const real_t dt,
        const real_t f0,
        const real_t f1,
        const real_t f2,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[3],
        const void *const RSTR p_data,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[3],
        const void *const RSTR p_old_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return cu_navier_stokes_hex27_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out, stream);
    case smesh::PROTEUS_HEX27:
      return cu_navier_stokes_proteus_hex27_proteus_hex8_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out, stream);
    case smesh::TET10:
      return cu_navier_stokes_tet10_tet4_residual_i_msoa((int)resolved_real_type, nelements, nnodes, elements, points, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out, stream);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_residual_3d_i_msoa", (int)element_type, (int)real_type);
}
