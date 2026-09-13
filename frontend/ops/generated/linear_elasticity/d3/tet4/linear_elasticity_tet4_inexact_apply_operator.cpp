#include "../../op/sfem_GeneratedLinearElasticity_c_abi.hpp"

#include "linear_elasticity_tet4_inexact_apply_inline.hpp"

extern "C" int linear_elasticity_tet4_inexact_apply_tangent_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
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
    const real_t lmbda,
    const real_t mu,
    const ptrdiff_t tangent_component_stride,
    metric_tensor_t *const RSTR tangent
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      return sfem::codegen::linear_elasticity_tet4_inexact_apply_tangent_a_msoa_impl<double, geom_t, metric_tensor_t, 16>(
          nelements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, tangent_component_stride, tangent);
    }
    case (int)sizeof(float): {
      return sfem::codegen::linear_elasticity_tet4_inexact_apply_tangent_a_msoa_impl<float, geom_t, metric_tensor_t, 16>(
          nelements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, tangent_component_stride, tangent);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("linear_elasticity_tet4_inexact_apply_tangent_a_msoa", -1, (int)scalar_bytes);
}

extern "C" int linear_elasticity_tet4_inexact_apply_stored_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
    const ptrdiff_t tangent_component_stride,
    const metric_tensor_t *const RSTR tangent,
    const ptrdiff_t h_stride,
    const void *const RSTR hx,
    const void *const RSTR hy,
    const void *const RSTR hz,
    const ptrdiff_t out_stride,
    void *const RSTR outx,
    void *const RSTR outy,
    void *const RSTR outz
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      return sfem::codegen::linear_elasticity_tet4_inexact_apply_stored_a_msoa_impl<double, metric_tensor_t, 16>(
          nelements, elements, tangent_component_stride, tangent, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
    }
    case (int)sizeof(float): {
      return sfem::codegen::linear_elasticity_tet4_inexact_apply_stored_a_msoa_impl<float, metric_tensor_t, 16>(
          nelements, elements, tangent_component_stride, tangent, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("linear_elasticity_tet4_inexact_apply_stored_a_msoa", -1, (int)scalar_bytes);
}

extern "C" int linear_elasticity_tet4_inexact_apply_stored_packed_two_pass_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t n_packs,
    const ptrdiff_t n_elements_per_pack,
    const ptrdiff_t nelements,
    const ptrdiff_t max_nodes_per_pack,
    uint16_t **const RSTR elements,
    const ptrdiff_t *const RSTR owned_nodes_ptr,
    const ptrdiff_t n_ghost_entries,
    const ptrdiff_t n_ghost_reduce_rows,
    const ptrdiff_t *const RSTR ghost_ptr,
    const idx_t *const RSTR ghost_idx,
    const ptrdiff_t *const RSTR ghost_reduce_ptr,
    const ptrdiff_t *const RSTR ghost_reduce_idx,
    const idx_t *const RSTR ghost_reduce_dest,
    void *const RSTR ghost_buf,
    const ptrdiff_t tangent_component_stride,
    const metric_tensor_t *const RSTR tangent,
    const ptrdiff_t h_stride,
    const void *const RSTR hx,
    const void *const RSTR hy,
    const void *const RSTR hz,
    const ptrdiff_t out_stride,
    void *const RSTR outx,
    void *const RSTR outy,
    void *const RSTR outz
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      return sfem::codegen::linear_elasticity_tet4_inexact_apply_stored_packed_two_pass_a_msoa_impl<double, metric_tensor_t, 16>(
          n_packs, n_elements_per_pack, nelements, max_nodes_per_pack, elements, owned_nodes_ptr, n_ghost_entries, n_ghost_reduce_rows, ghost_ptr, ghost_idx, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (double *)ghost_buf, tangent_component_stride, tangent, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
    }
    case (int)sizeof(float): {
      return sfem::codegen::linear_elasticity_tet4_inexact_apply_stored_packed_two_pass_a_msoa_impl<float, metric_tensor_t, 16>(
          n_packs, n_elements_per_pack, nelements, max_nodes_per_pack, elements, owned_nodes_ptr, n_ghost_entries, n_ghost_reduce_rows, ghost_ptr, ghost_idx, ghost_reduce_ptr, ghost_reduce_idx, ghost_reduce_dest, (float *)ghost_buf, tangent_component_stride, tangent, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("linear_elasticity_tet4_inexact_apply_stored_packed_two_pass_a_msoa", -1, (int)scalar_bytes);
}

extern "C" int linear_elasticity_tet4_inexact_apply_compressed_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
    const ptrdiff_t tangent_component_stride,
    const compressed_t *const RSTR tangent,
    const scaling_t *const RSTR scaling,
    const ptrdiff_t h_stride,
    const void *const RSTR hx,
    const void *const RSTR hy,
    const void *const RSTR hz,
    const ptrdiff_t out_stride,
    void *const RSTR outx,
    void *const RSTR outy,
    void *const RSTR outz
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      return sfem::codegen::linear_elasticity_tet4_inexact_apply_compressed_a_msoa_impl<double, compressed_t, scaling_t>(
          nelements, elements, tangent_component_stride, tangent, scaling, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
    }
    case (int)sizeof(float): {
      return sfem::codegen::linear_elasticity_tet4_inexact_apply_compressed_a_msoa_impl<float, compressed_t, scaling_t>(
          nelements, elements, tangent_component_stride, tangent, scaling, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("linear_elasticity_tet4_inexact_apply_compressed_a_msoa", -1, (int)scalar_bytes);
}
