#include "sfem_GeneratedNavierStokes_c_abi.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double),
              "the generated kernels select their scalar by width");
static_assert((int)smesh::SMESH_FLOAT32 == (int)sizeof(float),
              "the generated kernels select their scalar by width");

extern "C" int navier_stokes_form_1_p_tri6_tri3_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[2],
    const void *const RSTR p_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[2],
    void *const RSTR p_out
);
extern "C" int navier_stokes_form_1_p_hex27_hex8_residual_a_msoa(
    const int scalar_bytes,
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
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[3],
    const void *const RSTR p_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out
);
extern "C" int navier_stokes_form_1_p_tet10_tet4_residual_a_msoa(
    const int scalar_bytes,
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
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[3],
    const void *const RSTR p_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out
);
extern "C" int navier_stokes_form_1_u_tri6_tri3_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
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
    void *const RSTR p_out
);
extern "C" int navier_stokes_form_1_u_hex27_hex8_residual_a_msoa(
    const int scalar_bytes,
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
    void *const RSTR p_out
);
extern "C" int navier_stokes_form_1_u_tet10_tet4_residual_a_msoa(
    const int scalar_bytes,
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
    void *const RSTR p_out
);
extern "C" int navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[2],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[2],
    void *const RSTR p_out
);
extern "C" int navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_a_msoa(
    const int scalar_bytes,
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
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out
);
extern "C" int navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_a_msoa(
    const int scalar_bytes,
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
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out
);
extern "C" int navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[2],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[2],
    void *const RSTR p_out
);
extern "C" int navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_a_msoa(
    const int scalar_bytes,
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
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out
);
extern "C" int navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_a_msoa(
    const int scalar_bytes,
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
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out
);
extern "C" int navier_stokes_form_2_u_u_tri6_jacobian_action_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const real_t convection_scale,
    const real_t dt,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[2],
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[2],
    const ptrdiff_t out_stride,
    void *const RSTR u_out[2]
);
extern "C" int navier_stokes_form_2_u_u_hex27_jacobian_action_a_msoa(
    const int scalar_bytes,
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
    const real_t convection_scale,
    const real_t dt,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3]
);
extern "C" int navier_stokes_form_2_u_u_tet10_jacobian_action_a_msoa(
    const int scalar_bytes,
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
    const real_t convection_scale,
    const real_t dt,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[3],
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3]
);
extern "C" int navier_stokes_tri6_tri3_jacobian_action_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
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
    void *const RSTR p_out
);
extern "C" int navier_stokes_hex27_hex8_jacobian_action_a_msoa(
    const int scalar_bytes,
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
    void *const RSTR p_out
);
extern "C" int navier_stokes_tet10_tet4_jacobian_action_a_msoa(
    const int scalar_bytes,
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
    void *const RSTR p_out
);
extern "C" int navier_stokes_tri6_tri3_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
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
    void *const RSTR p_out
);
extern "C" int navier_stokes_hex27_hex8_residual_a_msoa(
    const int scalar_bytes,
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
    void *const RSTR p_out
);
extern "C" int navier_stokes_tet10_tet4_residual_a_msoa(
    const int scalar_bytes,
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
    void *const RSTR p_out
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int navier_stokes_form_1_p_residual_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[2],
        const void *const RSTR p_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI6:
      return navier_stokes_form_1_p_tri6_tri3_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, current_stride, u_data, p_data, out_stride, u_out, p_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_1_p_residual_2d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int navier_stokes_form_1_p_residual_3d_a_msoa(
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
        const ptrdiff_t current_stride,
        const void *const RSTR u_data[3],
        const void *const RSTR p_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return navier_stokes_form_1_p_hex27_hex8_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, current_stride, u_data, p_data, out_stride, u_out, p_out);
    case smesh::TET10:
      return navier_stokes_form_1_p_tet10_tet4_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, current_stride, u_data, p_data, out_stride, u_out, p_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_1_p_residual_3d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int navier_stokes_form_1_u_residual_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
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
        void *const RSTR p_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI6:
      return navier_stokes_form_1_u_tri6_tri3_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, convection_scale, dt, f0, f1, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_1_u_residual_2d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int navier_stokes_form_1_u_residual_3d_a_msoa(
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
        void *const RSTR p_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return navier_stokes_form_1_u_hex27_hex8_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
    case smesh::TET10:
      return navier_stokes_form_1_u_tet10_tet4_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_1_u_residual_3d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int navier_stokes_form_2_p_u_jacobian_action_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI6:
      return navier_stokes_form_2_p_u_tri6_tri3_jacobian_action_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_2_p_u_jacobian_action_2d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int navier_stokes_form_2_p_u_jacobian_action_3d_a_msoa(
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
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
    case smesh::TET10:
      return navier_stokes_form_2_p_u_tet10_tet4_jacobian_action_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_2_p_u_jacobian_action_3d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int navier_stokes_form_2_u_p_jacobian_action_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2],
        void *const RSTR p_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI6:
      return navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_2_u_p_jacobian_action_2d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int navier_stokes_form_2_u_p_jacobian_action_3d_a_msoa(
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
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return navier_stokes_form_2_u_p_hex27_hex8_jacobian_action_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
    case smesh::TET10:
      return navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_2_u_p_jacobian_action_3d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int navier_stokes_form_2_u_u_jacobian_action_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[2],
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[2],
        const ptrdiff_t out_stride,
        void *const RSTR u_out[2]
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI6:
      return navier_stokes_form_2_u_u_tri6_jacobian_action_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_2_u_u_jacobian_action_2d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int navier_stokes_form_2_u_u_jacobian_action_3d_a_msoa(
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
        const real_t convection_scale,
        const real_t dt,
        const real_t nu,
        const real_t rho,
        const ptrdiff_t previous_stride,
        const void *const RSTR u_old_data[3],
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3]
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return navier_stokes_form_2_u_u_hex27_jacobian_action_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out);
    case smesh::TET10:
      return navier_stokes_form_2_u_u_tet10_jacobian_action_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, convection_scale, dt, nu, rho, previous_stride, u_old_data, direction_stride, u_direction_data, out_stride, u_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_form_2_u_u_jacobian_action_3d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int navier_stokes_jacobian_action_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
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
        void *const RSTR p_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI6:
      return navier_stokes_tri6_tri3_jacobian_action_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, convection_scale, dt, nu, rho, previous_stride, u_old_data, p_old_data, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_jacobian_action_2d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int navier_stokes_jacobian_action_3d_a_msoa(
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
        void *const RSTR p_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return navier_stokes_hex27_hex8_jacobian_action_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, convection_scale, dt, nu, rho, previous_stride, u_old_data, p_old_data, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
    case smesh::TET10:
      return navier_stokes_tet10_tet4_jacobian_action_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, convection_scale, dt, nu, rho, previous_stride, u_old_data, p_old_data, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_jacobian_action_3d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int navier_stokes_residual_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
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
        void *const RSTR p_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI6:
      return navier_stokes_tri6_tri3_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, convection_scale, dt, f0, f1, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_residual_2d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int navier_stokes_residual_3d_a_msoa(
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
        void *const RSTR p_out
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::HEX27:
      return navier_stokes_hex27_hex8_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
    case smesh::TET10:
      return navier_stokes_tet10_tet4_residual_a_msoa((int)resolved_real_type, nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "navier_stokes_residual_3d_a_msoa", (int)element_type, (int)real_type);
}
