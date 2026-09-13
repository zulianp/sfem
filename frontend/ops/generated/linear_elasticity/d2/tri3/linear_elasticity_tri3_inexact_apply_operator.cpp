#include "../../op/sfem_GeneratedLinearElasticity_c_abi.hpp"
#include "../../../kernel_diagnostics.hpp"

#include "linear_elasticity_tri3_inexact_apply_inline.hpp"

namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_elasticity_tri3_inexact_apply_tangent_a_msoa_diagnostics_data = {
  "linear_elasticity_tri3_inexact_apply_tangent_a_msoa",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  16,
  38,
  10,
  0,
  8,
  0,
  0,
  0,
  0,
  0,
  142,
  20,
  20,
  0,
  0,
  5,
  0,
  1,
  2,
  0,
  0,
  0,
  0,
  0,
  1.0,
  1.0,
  8.0,
  12.0,
  16.0,
  20.0,
  20.0,
  24.0,
  1.0,
  1.0
};

static const KernelDiagnostics linear_elasticity_tri3_inexact_apply_stored_a_msoa_diagnostics_data = {
  "linear_elasticity_tri3_inexact_apply_stored_a_msoa",
  "TRI3",
  2,
  0,
  3,
  16,
  1,
  18,
  36,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  54,
  54,
  54,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  6,
  6,
  6,
  6,
  1.0,
  1.0,
  8.0,
  12.0,
  16.0,
  20.0,
  20.0,
  24.0,
  1.0,
  1.0
};

static const KernelDiagnostics linear_elasticity_tri3_inexact_apply_compressed_a_msoa_diagnostics_data = {
  "linear_elasticity_tri3_inexact_apply_compressed_a_msoa",
  "TRI3",
  2,
  0,
  3,
  16,
  1,
  18,
  36,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  54,
  60,
  60,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  6,
  6,
  6,
  6,
  1.0,
  1.0,
  8.0,
  12.0,
  16.0,
  20.0,
  20.0,
  24.0,
  1.0,
  1.0
};

} // namespace codegen
} // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tri3_inexact_apply_tangent_a_msoa_diagnostics(void) {
  return &sfem::codegen::linear_elasticity_tri3_inexact_apply_tangent_a_msoa_diagnostics_data;
}
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tri3_inexact_apply_stored_a_msoa_diagnostics(void) {
  return &sfem::codegen::linear_elasticity_tri3_inexact_apply_stored_a_msoa_diagnostics_data;
}
extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tri3_inexact_apply_compressed_a_msoa_diagnostics(void) {
  return &sfem::codegen::linear_elasticity_tri3_inexact_apply_compressed_a_msoa_diagnostics_data;
}

extern "C" int linear_elasticity_tri3_inexact_apply_tangent_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const real_t lmbda,
    const real_t mu,
    const ptrdiff_t tangent_component_stride,
    metric_tensor_t *const RSTR tangent
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      return sfem::codegen::linear_elasticity_tri3_inexact_apply_tangent_a_msoa_impl<double, geom_t, metric_tensor_t, 16>(
          nelements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, lmbda, mu, tangent_component_stride, tangent);
    }
    case (int)sizeof(float): {
      return sfem::codegen::linear_elasticity_tri3_inexact_apply_tangent_a_msoa_impl<float, geom_t, metric_tensor_t, 16>(
          nelements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, lmbda, mu, tangent_component_stride, tangent);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("linear_elasticity_tri3_inexact_apply_tangent_a_msoa", -1, (int)scalar_bytes);
}

extern "C" int linear_elasticity_tri3_inexact_apply_stored_a_msoa(
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
      return sfem::codegen::linear_elasticity_tri3_inexact_apply_stored_a_msoa_impl<double, metric_tensor_t, 16>(
          nelements, elements, tangent_component_stride, tangent, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
    }
    case (int)sizeof(float): {
      return sfem::codegen::linear_elasticity_tri3_inexact_apply_stored_a_msoa_impl<float, metric_tensor_t, 16>(
          nelements, elements, tangent_component_stride, tangent, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("linear_elasticity_tri3_inexact_apply_stored_a_msoa", -1, (int)scalar_bytes);
}

extern "C" int linear_elasticity_tri3_inexact_apply_compressed_a_msoa(
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
      return sfem::codegen::linear_elasticity_tri3_inexact_apply_compressed_a_msoa_impl<double, compressed_t, scaling_t>(
          nelements, elements, tangent_component_stride, tangent, scaling, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
    }
    case (int)sizeof(float): {
      return sfem::codegen::linear_elasticity_tri3_inexact_apply_compressed_a_msoa_impl<float, compressed_t, scaling_t>(
          nelements, elements, tangent_component_stride, tangent, scaling, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("linear_elasticity_tri3_inexact_apply_compressed_a_msoa", -1, (int)scalar_bytes);
}
