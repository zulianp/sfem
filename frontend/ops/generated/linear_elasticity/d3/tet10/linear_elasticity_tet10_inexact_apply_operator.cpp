#include "../../op/sfem_GeneratedLinearElasticity_c_abi.hpp"

#include "linear_elasticity_tet10_inexact_apply_inline.hpp"

extern "C" int linear_elasticity_tet10_inexact_apply_tangent_a_msoa(
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
    const double lmbda,
    const double mu,
    const ptrdiff_t tangent_component_stride,
    metric_tensor_t *const RSTR tangent
) {
  return sfem::codegen::linear_elasticity_tet10_inexact_apply_tangent_a_msoa_impl<double, geom_t, metric_tensor_t, 16>(
      nelements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0,
      lmbda, mu,
      tangent_component_stride, tangent);
}

extern "C" int linear_elasticity_tet10_inexact_apply_stored_a_msoa(
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
  return sfem::codegen::linear_elasticity_tet10_inexact_apply_stored_a_msoa_impl<double, metric_tensor_t, 16>(
      nelements, elements,
      tangent_component_stride, tangent,
      h_stride, hx, hy, hz,
      out_stride, outx, outy, outz);
}

extern "C" int linear_elasticity_tet10_inexact_apply_compressed_a_msoa(
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
  return sfem::codegen::linear_elasticity_tet10_inexact_apply_compressed_a_msoa_impl<double, compressed_t, scaling_t>(
      nelements, elements,
      tangent_component_stride, tangent, scaling,
      h_stride, hx, hy, hz,
      out_stride, outx, outy, outz);
}

extern "C" int linear_elasticity_tet10_inexact_apply_tangent_a_msoa_float(
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
    const float lmbda,
    const float mu,
    const ptrdiff_t tangent_component_stride,
    metric_tensor_t *const RSTR tangent
) {
  return sfem::codegen::linear_elasticity_tet10_inexact_apply_tangent_a_msoa_impl<float, geom_t, metric_tensor_t, 16>(
      nelements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0,
      lmbda, mu,
      tangent_component_stride, tangent);
}

extern "C" int linear_elasticity_tet10_inexact_apply_stored_a_msoa_float(
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
  return sfem::codegen::linear_elasticity_tet10_inexact_apply_stored_a_msoa_impl<float, metric_tensor_t, 16>(
      nelements, elements,
      tangent_component_stride, tangent,
      h_stride, hx, hy, hz,
      out_stride, outx, outy, outz);
}

extern "C" int linear_elasticity_tet10_inexact_apply_compressed_a_msoa_float(
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
  return sfem::codegen::linear_elasticity_tet10_inexact_apply_compressed_a_msoa_impl<float, compressed_t, scaling_t>(
      nelements, elements,
      tangent_component_stride, tangent, scaling,
      h_stride, hx, hy, hz,
      out_stride, outx, outy, outz);
}
