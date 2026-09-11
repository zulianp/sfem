#include "../../op/sfem_GeneratedNeoHookeanOgden_c_abi.hpp"

#include "neohookean_ogden_hex8_inexact_apply_inline.hpp"

extern "C" int neohookean_ogden_hex8_inexact_apply_tangent_a_msoa(
    const ptrdiff_t nelements,
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
    const double lmbda,
    const double mu,
    const ptrdiff_t u_stride,
    const double *const RSTR ux,
    const double *const RSTR uy,
    const double *const RSTR uz,
    const ptrdiff_t tangent_component_stride,
    metric_tensor_t *const RSTR tangent
) {
  return sfem::codegen::neohookean_ogden_hex8_inexact_apply_tangent_a_msoa_impl<double, geom_t, metric_tensor_t, 16>(
      nelements, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0,
      lmbda, mu,
      u_stride, ux, uy, uz,
      tangent_component_stride, tangent);
}

extern "C" int neohookean_ogden_hex8_inexact_apply_stored_a_msoa(
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
    const ptrdiff_t tangent_component_stride,
    const metric_tensor_t *const RSTR tangent,
    const ptrdiff_t h_stride,
    const double *const RSTR hx,
    const double *const RSTR hy,
    const double *const RSTR hz,
    const ptrdiff_t out_stride,
    double *const RSTR outx,
    double *const RSTR outy,
    double *const RSTR outz
) {
  return sfem::codegen::neohookean_ogden_hex8_inexact_apply_stored_a_msoa_impl<double, metric_tensor_t, 16>(
      nelements, elements,
      tangent_component_stride, tangent,
      h_stride, hx, hy, hz,
      out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_hex8_inexact_apply_stored_packed_two_pass_a_msoa(
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
    double *const RSTR ghost_buf,
    const ptrdiff_t tangent_component_stride,
    const metric_tensor_t *const RSTR tangent,
    const ptrdiff_t h_stride,
    const double *const RSTR hx,
    const double *const RSTR hy,
    const double *const RSTR hz,
    const ptrdiff_t out_stride,
    double *const RSTR outx,
    double *const RSTR outy,
    double *const RSTR outz
) {
  return sfem::codegen::neohookean_ogden_hex8_inexact_apply_stored_packed_two_pass_a_msoa_impl<double, metric_tensor_t, 16>(
      n_packs, n_elements_per_pack, nelements, max_nodes_per_pack,
      elements, owned_nodes_ptr, n_ghost_entries, n_ghost_reduce_rows,
      ghost_ptr, ghost_idx, ghost_reduce_ptr, ghost_reduce_idx,
      ghost_reduce_dest, ghost_buf,
      tangent_component_stride, tangent,
      h_stride, hx, hy, hz,
      out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_hex8_inexact_apply_compressed_a_msoa(
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
    const ptrdiff_t tangent_component_stride,
    const compressed_t *const RSTR tangent,
    const scaling_t *const RSTR scaling,
    const ptrdiff_t h_stride,
    const double *const RSTR hx,
    const double *const RSTR hy,
    const double *const RSTR hz,
    const ptrdiff_t out_stride,
    double *const RSTR outx,
    double *const RSTR outy,
    double *const RSTR outz
) {
  return sfem::codegen::neohookean_ogden_hex8_inexact_apply_compressed_a_msoa_impl<double, compressed_t, scaling_t>(
      nelements, elements,
      tangent_component_stride, tangent, scaling,
      h_stride, hx, hy, hz,
      out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_hex8_inexact_apply_tangent_a_msoa_float(
    const ptrdiff_t nelements,
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
    const float lmbda,
    const float mu,
    const ptrdiff_t u_stride,
    const float *const RSTR ux,
    const float *const RSTR uy,
    const float *const RSTR uz,
    const ptrdiff_t tangent_component_stride,
    metric_tensor_t *const RSTR tangent
) {
  return sfem::codegen::neohookean_ogden_hex8_inexact_apply_tangent_a_msoa_impl<float, geom_t, metric_tensor_t, 16>(
      nelements, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0,
      lmbda, mu,
      u_stride, ux, uy, uz,
      tangent_component_stride, tangent);
}

extern "C" int neohookean_ogden_hex8_inexact_apply_stored_a_msoa_float(
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
    const ptrdiff_t tangent_component_stride,
    const metric_tensor_t *const RSTR tangent,
    const ptrdiff_t h_stride,
    const float *const RSTR hx,
    const float *const RSTR hy,
    const float *const RSTR hz,
    const ptrdiff_t out_stride,
    float *const RSTR outx,
    float *const RSTR outy,
    float *const RSTR outz
) {
  return sfem::codegen::neohookean_ogden_hex8_inexact_apply_stored_a_msoa_impl<float, metric_tensor_t, 16>(
      nelements, elements,
      tangent_component_stride, tangent,
      h_stride, hx, hy, hz,
      out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_hex8_inexact_apply_stored_packed_two_pass_a_msoa_float(
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
    float *const RSTR ghost_buf,
    const ptrdiff_t tangent_component_stride,
    const metric_tensor_t *const RSTR tangent,
    const ptrdiff_t h_stride,
    const float *const RSTR hx,
    const float *const RSTR hy,
    const float *const RSTR hz,
    const ptrdiff_t out_stride,
    float *const RSTR outx,
    float *const RSTR outy,
    float *const RSTR outz
) {
  return sfem::codegen::neohookean_ogden_hex8_inexact_apply_stored_packed_two_pass_a_msoa_impl<float, metric_tensor_t, 16>(
      n_packs, n_elements_per_pack, nelements, max_nodes_per_pack,
      elements, owned_nodes_ptr, n_ghost_entries, n_ghost_reduce_rows,
      ghost_ptr, ghost_idx, ghost_reduce_ptr, ghost_reduce_idx,
      ghost_reduce_dest, ghost_buf,
      tangent_component_stride, tangent,
      h_stride, hx, hy, hz,
      out_stride, outx, outy, outz);
}

extern "C" int neohookean_ogden_hex8_inexact_apply_compressed_a_msoa_float(
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
    const ptrdiff_t tangent_component_stride,
    const compressed_t *const RSTR tangent,
    const scaling_t *const RSTR scaling,
    const ptrdiff_t h_stride,
    const float *const RSTR hx,
    const float *const RSTR hy,
    const float *const RSTR hz,
    const ptrdiff_t out_stride,
    float *const RSTR outx,
    float *const RSTR outy,
    float *const RSTR outz
) {
  return sfem::codegen::neohookean_ogden_hex8_inexact_apply_compressed_a_msoa_impl<float, compressed_t, scaling_t>(
      nelements, elements,
      tangent_component_stride, tangent, scaling,
      h_stride, hx, hy, hz,
      out_stride, outx, outy, outz);
}
