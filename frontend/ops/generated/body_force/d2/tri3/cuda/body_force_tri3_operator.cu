#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../../cuda/body_force_d2_simplex_local.cuh"
#include "../../../../cuda/geometry_kernels.cuh"
#include "../../../../cuda/kernel_diagnostics.cuh"
#if defined(__has_include)
#if __has_include("smesh_types.hpp")
#include "smesh_types.hpp"
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__host__ __device__ __forceinline__ const s_t *ageom_stream(
    const g_t *const RSTR source,
    s_t *const RSTR,
    std::true_type) {
  return source;
}

template <typename s_t, typename g_t>
__host__ __device__ __forceinline__ const s_t *ageom_stream(
    const g_t *const RSTR source,
    s_t *const RSTR converted,
    std::false_type) {
  converted[0] = s_t(source[0]);
  return converted;
}

} // namespace codegen
} // namespace sfem
namespace sfem {
namespace codegen {

static const KernelDiagnostics body_force_tri3_residual_esoa_diagnostics_data = {
  "body_force_tri3_residual_esoa",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  0,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  5,
  2,
  4,
  81,
  108,
  0,
  3,
  5,
  9,
  1,
  3,
  0,
  0,
  6,
  1,
  1,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_body_force_tri3_residual_esoa_diagnostics(void) {
  return &sfem::codegen::body_force_tri3_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics body_force_tri3_jacobian_action_esoa_diagnostics_data = {
  "body_force_tri3_jacobian_action_esoa",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  2,
  0,
  81,
  108,
  0,
  0,
  5,
  9,
  1,
  0,
  0,
  0,
  6,
  1,
  1,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_body_force_tri3_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::body_force_tri3_jacobian_action_esoa_diagnostics_data;
}

extern "C" int cu_body_force_tri3_residual_esoa(
    const int scalar_bytes,
    const void *const RSTR determinant,
    const real_t density,
    const real_t g0,
    const real_t g1,
    void *const RSTR output[6],
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::body_force_d2_simplex_tri3_residual_block<double, 1, 3>((const double *)determinant, density, g0, g1, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::body_force_d2_simplex_tri3_residual_block<float, 1, 3>((const float *)determinant, density, g0, g1, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("body_force_tri3_residual_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void body_force_tri3_residual_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const s_t density,
    const s_t g0,
    const s_t g1,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out
) {
  static constexpr int NQ = 1;
  static constexpr int NS = 3;
  static constexpr int NC = 2;

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    s_t boutput[NC * NS];

    for (int stream = 0; stream < 6; ++stream) {
      {
        boutput[stream] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[1] = {g_det0 + evb};
    s_t baffine_geometry_data[1];
    const s_t *bageom_streams[1];
    bageom_streams[0] = ageom_stream<s_t, g_t>(
        affine_geometry_sources[0], &baffine_geometry_data[0], std::is_same<g_t, s_t>());

    body_force_d2_simplex_tri3_residual_block_contiguous<s_t, NQ, NS>(bageom_streams[0], density, g0, g1, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        {
          atomicAdd(&(out[element_shape[evb] * out_stride]), boutput[stream]);
        }
      }
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_body_force_tri3_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::body_force_tri3_residual_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, g0, g1, out_stride, (double *)u0_out, (double *)u1_out);
        return sfem::codegen::launch_status("body_force_tri3_residual_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::body_force_tri3_residual_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, g0, g1, out_stride, (float *)u0_out, (float *)u1_out);
        return sfem::codegen::launch_status("body_force_tri3_residual_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("body_force_tri3_residual_a_msoa", -1, (int)scalar_bytes);
}
