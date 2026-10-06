#include "sfem_GeneratedTwoPhaseFlow_c_abi.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double),
              "the generated kernels select their scalar by width");
static_assert((int)smesh::SMESH_FLOAT32 == (int)sizeof(float),
              "the generated kernels select their scalar by width");

extern "C" int two_phase_flow_total_tri3_merit_patch_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t n_owned_nodes,
    const count_t *const RSTR n2e_ptr,
    const element_idx_t *const RSTR n2e_idx,
    const uint8_t *const RSTR n2e_local,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR shape,
    const void *const RSTR grad_ref[2],
    const void *const RSTR q_weight,
    const real_t P_r,
    const real_t S_res,
    const real_t dt,
    const real_t kappa_T,
    const real_t m,
    const real_t p_wr,
    const real_t porosity,
    const real_t rho_w0,
    const real_t C_kw1,
    const real_t K_0,
    const real_t K_1,
    const real_t mu_w,
    const real_t K_2,
    const real_t K_3,
    const real_t M_c,
    const real_t R,
    const real_t T,
    const real_t Z,
    const real_t C_ka1,
    const real_t C_ka2,
    const real_t mu_c,
    const int nsteps,
    const void *const RSTR steps,
    const void *const RSTR x,
    const void *const RSTR h,
    const void *const RSTR p,
    const void *const RSTR accumulator,
    void *const RSTR merit
);
extern "C" int two_phase_flow_total_tet4_merit_patch_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t n_owned_nodes,
    const count_t *const RSTR n2e_ptr,
    const element_idx_t *const RSTR n2e_idx,
    const uint8_t *const RSTR n2e_local,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR shape,
    const void *const RSTR grad_ref[3],
    const void *const RSTR q_weight,
    const real_t P_r,
    const real_t S_res,
    const real_t dt,
    const real_t kappa_T,
    const real_t m,
    const real_t p_wr,
    const real_t porosity,
    const real_t rho_w0,
    const real_t C_kw1,
    const real_t K_0,
    const real_t K_1,
    const real_t K_2,
    const real_t mu_w,
    const real_t K_3,
    const real_t K_4,
    const real_t K_5,
    const real_t K_6,
    const real_t K_7,
    const real_t K_8,
    const real_t M_c,
    const real_t R,
    const real_t T,
    const real_t Z,
    const real_t C_ka1,
    const real_t C_ka2,
    const real_t mu_c,
    const int nsteps,
    const void *const RSTR steps,
    const void *const RSTR x,
    const void *const RSTR h,
    const void *const RSTR p,
    const void *const RSTR accumulator,
    void *const RSTR merit
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_total_merit_patch_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t n_owned_nodes,
        const count_t *const RSTR n2e_ptr,
        const element_idx_t *const RSTR n2e_idx,
        const uint8_t *const RSTR n2e_local,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR shape,
        const void *const RSTR grad_ref[2],
        const void *const RSTR q_weight,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t mu_w,
        const real_t K_2,
        const real_t K_3,
        const real_t M_c,
        const real_t R,
        const real_t T,
        const real_t Z,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t mu_c,
        const int nsteps,
        const void *const RSTR steps,
        const void *const RSTR x,
        const void *const RSTR h,
        const void *const RSTR p,
        const void *const RSTR accumulator,
        void *const RSTR merit
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI3:
      return two_phase_flow_total_tri3_merit_patch_a_msoa((int)resolved_real_type, n_owned_nodes, n2e_ptr, n2e_idx, n2e_local, elements, points, shape, grad_ref, q_weight, P_r, S_res, dt, kappa_T, m, p_wr, porosity, rho_w0, C_kw1, K_0, K_1, mu_w, K_2, K_3, M_c, R, T, Z, C_ka1, C_ka2, mu_c, nsteps, steps, x, h, p, accumulator, merit);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_total_merit_patch_2d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int two_phase_flow_total_merit_patch_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t n_owned_nodes,
        const count_t *const RSTR n2e_ptr,
        const element_idx_t *const RSTR n2e_idx,
        const uint8_t *const RSTR n2e_local,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR shape,
        const void *const RSTR grad_ref[3],
        const void *const RSTR q_weight,
        const real_t P_r,
        const real_t S_res,
        const real_t dt,
        const real_t kappa_T,
        const real_t m,
        const real_t p_wr,
        const real_t porosity,
        const real_t rho_w0,
        const real_t C_kw1,
        const real_t K_0,
        const real_t K_1,
        const real_t K_2,
        const real_t mu_w,
        const real_t K_3,
        const real_t K_4,
        const real_t K_5,
        const real_t K_6,
        const real_t K_7,
        const real_t K_8,
        const real_t M_c,
        const real_t R,
        const real_t T,
        const real_t Z,
        const real_t C_ka1,
        const real_t C_ka2,
        const real_t mu_c,
        const int nsteps,
        const void *const RSTR steps,
        const void *const RSTR x,
        const void *const RSTR h,
        const void *const RSTR p,
        const void *const RSTR accumulator,
        void *const RSTR merit
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TET4:
      return two_phase_flow_total_tet4_merit_patch_a_msoa((int)resolved_real_type, n_owned_nodes, n2e_ptr, n2e_idx, n2e_local, elements, points, shape, grad_ref, q_weight, P_r, S_res, dt, kappa_T, m, p_wr, porosity, rho_w0, C_kw1, K_0, K_1, K_2, mu_w, K_3, K_4, K_5, K_6, K_7, K_8, M_c, R, T, Z, C_ka1, C_ka2, mu_c, nsteps, steps, x, h, p, accumulator, merit);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "two_phase_flow_total_merit_patch_3d_a_msoa", (int)element_type, (int)real_type);
}
