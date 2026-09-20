#include "sfem_GeneratedLinearElasticity_c_abi.hpp"

#ifndef SFEM_CODEGEN_PUBLIC_C_ABI
#define SFEM_CODEGEN_PUBLIC_C_ABI
#endif

static_assert((int)smesh::SMESH_FLOAT64 == (int)sizeof(double),
              "the generated kernels select their scalar by width");
static_assert((int)smesh::SMESH_FLOAT32 == (int)sizeof(float),
              "the generated kernels select their scalar by width");

extern "C" int linear_elasticity_total_tri3_merit_patch_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t n_owned_nodes,
    const count_t *const RSTR n2e_ptr,
    const element_idx_t *const RSTR n2e_idx,
    const uint8_t *const RSTR n2e_local,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR grad_ref[2],
    const void *const RSTR q_weight,
    const real_t lmbda,
    const real_t mu,
    const int nsteps,
    const void *const RSTR steps,
    const void *const RSTR x,
    const void *const RSTR h,
    const void *const RSTR accumulator,
    void *const RSTR merit
);
extern "C" int linear_elasticity_total_tet4_merit_patch_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t n_owned_nodes,
    const count_t *const RSTR n2e_ptr,
    const element_idx_t *const RSTR n2e_idx,
    const uint8_t *const RSTR n2e_local,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR grad_ref[3],
    const void *const RSTR q_weight,
    const real_t lmbda,
    const real_t mu,
    const int nsteps,
    const void *const RSTR steps,
    const void *const RSTR x,
    const void *const RSTR h,
    const void *const RSTR accumulator,
    void *const RSTR merit
);

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int linear_elasticity_total_merit_patch_2d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t n_owned_nodes,
        const count_t *const RSTR n2e_ptr,
        const element_idx_t *const RSTR n2e_idx,
        const uint8_t *const RSTR n2e_local,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR grad_ref[2],
        const void *const RSTR q_weight,
        const real_t lmbda,
        const real_t mu,
        const int nsteps,
        const void *const RSTR steps,
        const void *const RSTR x,
        const void *const RSTR h,
        const void *const RSTR accumulator,
        void *const RSTR merit
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TRI3:
      return linear_elasticity_total_tri3_merit_patch_a_msoa((int)resolved_real_type, n_owned_nodes, n2e_ptr, n2e_idx, n2e_local, elements, points, grad_ref, q_weight, lmbda, mu, nsteps, steps, x, h, accumulator, merit);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "linear_elasticity_total_merit_patch_2d_a_msoa", (int)element_type, (int)real_type);
}

SFEM_CODEGEN_PUBLIC_C_ABI extern "C" int linear_elasticity_total_merit_patch_3d_a_msoa(
        const smesh::ElemType element_type,
        const enum smesh::PrimitiveType real_type,
        const ptrdiff_t n_owned_nodes,
        const count_t *const RSTR n2e_ptr,
        const element_idx_t *const RSTR n2e_idx,
        const uint8_t *const RSTR n2e_local,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR grad_ref[3],
        const void *const RSTR q_weight,
        const real_t lmbda,
        const real_t mu,
        const int nsteps,
        const void *const RSTR steps,
        const void *const RSTR x,
        const void *const RSTR h,
        const void *const RSTR accumulator,
        void *const RSTR merit
) {
  const enum smesh::PrimitiveType resolved_real_type =
      (real_type == smesh::SMESH_DEFAULT)
          ? smesh::TypeToEnum<real_t>::value()
          : real_type;
  switch (element_type) {
    case smesh::TET4:
      return linear_elasticity_total_tet4_merit_patch_a_msoa((int)resolved_real_type, n_owned_nodes, n2e_ptr, n2e_idx, n2e_local, elements, points, grad_ref, q_weight, lmbda, mu, nsteps, steps, x, h, accumulator, merit);
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch(
      "linear_elasticity_total_merit_patch_3d_a_msoa", (int)element_type, (int)real_type);
}
