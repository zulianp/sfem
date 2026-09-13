#include "../../op/sfem_GeneratedNeoHookeanOgden_c_abi.hpp"

#include "neohookean_ogden_quad4_inexact_apply_inline.hpp"

extern "C" int neohookean_ogden_quad4_inexact_apply_tangent_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const real_t lmbda,
    const real_t mu,
    const ptrdiff_t u_stride,
    const void *const RSTR ux,
    const void *const RSTR uy,
    const ptrdiff_t tangent_component_stride,
    metric_tensor_t *const RSTR tangent
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      return sfem::codegen::neohookean_ogden_quad4_inexact_apply_tangent_a_msoa_impl<double, geom_t, metric_tensor_t, 16>(
          nelements, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, tangent_component_stride, tangent);
    }
    case (int)sizeof(float): {
      return sfem::codegen::neohookean_ogden_quad4_inexact_apply_tangent_a_msoa_impl<float, geom_t, metric_tensor_t, 16>(
          nelements, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, tangent_component_stride, tangent);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_quad4_inexact_apply_tangent_a_msoa", -1, (int)scalar_bytes);
}

extern "C" int neohookean_ogden_quad4_inexact_apply_stored_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
    const ptrdiff_t tangent_component_stride,
    const metric_tensor_t *const RSTR tangent,
    const ptrdiff_t h_stride,
    const void *const RSTR hx,
    const void *const RSTR hy,
    const ptrdiff_t out_stride,
    void *const RSTR outx,
    void *const RSTR outy
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      return sfem::codegen::neohookean_ogden_quad4_inexact_apply_stored_a_msoa_impl<double, metric_tensor_t, 16>(
          nelements, elements, tangent_component_stride, tangent, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
    }
    case (int)sizeof(float): {
      return sfem::codegen::neohookean_ogden_quad4_inexact_apply_stored_a_msoa_impl<float, metric_tensor_t, 16>(
          nelements, elements, tangent_component_stride, tangent, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_quad4_inexact_apply_stored_a_msoa", -1, (int)scalar_bytes);
}

extern "C" int neohookean_ogden_quad4_inexact_apply_compressed_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    idx_t **const RSTR elements,
    const ptrdiff_t tangent_component_stride,
    const compressed_t *const RSTR tangent,
    const scaling_t *const RSTR scaling,
    const ptrdiff_t h_stride,
    const void *const RSTR hx,
    const void *const RSTR hy,
    const ptrdiff_t out_stride,
    void *const RSTR outx,
    void *const RSTR outy
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      return sfem::codegen::neohookean_ogden_quad4_inexact_apply_compressed_a_msoa_impl<double, compressed_t, scaling_t>(
          nelements, elements, tangent_component_stride, tangent, scaling, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
    }
    case (int)sizeof(float): {
      return sfem::codegen::neohookean_ogden_quad4_inexact_apply_compressed_a_msoa_impl<float, compressed_t, scaling_t>(
          nelements, elements, tangent_component_stride, tangent, scaling, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_quad4_inexact_apply_compressed_a_msoa", -1, (int)scalar_bytes);
}
