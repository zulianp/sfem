#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "linear_kelvin_voigt_total_d3_simplex_local.cuh"
#include "geometry_kernels.cuh"
#include "kernel_diagnostics.cuh"
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

static const KernelDiagnostics linear_kelvin_voigt_total_tet4_residual_esoa_diagnostics_data = {
  "linear_kelvin_voigt_total_tet4_residual_esoa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  43,
  45,
  0,
  0,
  0,
  0,
  0,
  0,
  32,
  16,
  88,
  253,
  366,
  13,
  20,
  10,
  16,
  1,
  5,
  24,
  0,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_linear_kelvin_voigt_total_tet4_residual_esoa_diagnostics(void) {
  return &sfem::codegen::linear_kelvin_voigt_total_tet4_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_kelvin_voigt_total_tet4_jacobian_u0_u0_diagnostics_data = {
  "linear_kelvin_voigt_total_tet4_jacobian_u0_u0",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  6,
  13,
  0,
  0,
  0,
  0,
  0,
  0,
  11,
  3,
  19,
  253,
  366,
  2,
  12,
  10,
  16,
  1,
  5,
  0,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_linear_kelvin_voigt_total_tet4_jacobian_u0_u0_diagnostics(void) {
  return &sfem::codegen::linear_kelvin_voigt_total_tet4_jacobian_u0_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_kelvin_voigt_total_tet4_jacobian_u0_u1_diagnostics_data = {
  "linear_kelvin_voigt_total_tet4_jacobian_u0_u1",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  4,
  10,
  0,
  0,
  0,
  0,
  0,
  0,
  9,
  2,
  14,
  253,
  366,
  1,
  9,
  10,
  16,
  1,
  5,
  0,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_linear_kelvin_voigt_total_tet4_jacobian_u0_u1_diagnostics(void) {
  return &sfem::codegen::linear_kelvin_voigt_total_tet4_jacobian_u0_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_kelvin_voigt_total_tet4_jacobian_u0_u2_diagnostics_data = {
  "linear_kelvin_voigt_total_tet4_jacobian_u0_u2",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  4,
  10,
  0,
  0,
  0,
  0,
  0,
  0,
  9,
  2,
  14,
  253,
  366,
  1,
  9,
  10,
  16,
  1,
  5,
  0,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_linear_kelvin_voigt_total_tet4_jacobian_u0_u2_diagnostics(void) {
  return &sfem::codegen::linear_kelvin_voigt_total_tet4_jacobian_u0_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_kelvin_voigt_total_tet4_jacobian_u1_u0_diagnostics_data = {
  "linear_kelvin_voigt_total_tet4_jacobian_u1_u0",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  4,
  10,
  0,
  0,
  0,
  0,
  0,
  0,
  9,
  2,
  14,
  253,
  366,
  1,
  9,
  10,
  16,
  1,
  5,
  0,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_linear_kelvin_voigt_total_tet4_jacobian_u1_u0_diagnostics(void) {
  return &sfem::codegen::linear_kelvin_voigt_total_tet4_jacobian_u1_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_kelvin_voigt_total_tet4_jacobian_u1_u1_diagnostics_data = {
  "linear_kelvin_voigt_total_tet4_jacobian_u1_u1",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  6,
  13,
  0,
  0,
  0,
  0,
  0,
  0,
  11,
  3,
  19,
  253,
  366,
  2,
  12,
  10,
  16,
  1,
  5,
  0,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_linear_kelvin_voigt_total_tet4_jacobian_u1_u1_diagnostics(void) {
  return &sfem::codegen::linear_kelvin_voigt_total_tet4_jacobian_u1_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_kelvin_voigt_total_tet4_jacobian_u1_u2_diagnostics_data = {
  "linear_kelvin_voigt_total_tet4_jacobian_u1_u2",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  4,
  10,
  0,
  0,
  0,
  0,
  0,
  0,
  9,
  2,
  14,
  253,
  366,
  1,
  9,
  10,
  16,
  1,
  5,
  0,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_linear_kelvin_voigt_total_tet4_jacobian_u1_u2_diagnostics(void) {
  return &sfem::codegen::linear_kelvin_voigt_total_tet4_jacobian_u1_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_kelvin_voigt_total_tet4_jacobian_u2_u0_diagnostics_data = {
  "linear_kelvin_voigt_total_tet4_jacobian_u2_u0",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  4,
  10,
  0,
  0,
  0,
  0,
  0,
  0,
  9,
  2,
  14,
  253,
  366,
  1,
  9,
  10,
  16,
  1,
  5,
  0,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_linear_kelvin_voigt_total_tet4_jacobian_u2_u0_diagnostics(void) {
  return &sfem::codegen::linear_kelvin_voigt_total_tet4_jacobian_u2_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_kelvin_voigt_total_tet4_jacobian_u2_u1_diagnostics_data = {
  "linear_kelvin_voigt_total_tet4_jacobian_u2_u1",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  4,
  10,
  0,
  0,
  0,
  0,
  0,
  0,
  9,
  2,
  14,
  253,
  366,
  1,
  9,
  10,
  16,
  1,
  5,
  0,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_linear_kelvin_voigt_total_tet4_jacobian_u2_u1_diagnostics(void) {
  return &sfem::codegen::linear_kelvin_voigt_total_tet4_jacobian_u2_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_kelvin_voigt_total_tet4_jacobian_u2_u2_diagnostics_data = {
  "linear_kelvin_voigt_total_tet4_jacobian_u2_u2",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  6,
  13,
  0,
  0,
  0,
  0,
  0,
  0,
  11,
  3,
  19,
  253,
  366,
  2,
  12,
  10,
  16,
  1,
  5,
  0,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_linear_kelvin_voigt_total_tet4_jacobian_u2_u2_diagnostics(void) {
  return &sfem::codegen::linear_kelvin_voigt_total_tet4_jacobian_u2_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_kelvin_voigt_total_tet4_jacobian_action_esoa_diagnostics_data = {
  "linear_kelvin_voigt_total_tet4_jacobian_action_esoa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  23,
  43,
  0,
  0,
  0,
  0,
  0,
  0,
  23,
  20,
  66,
  253,
  366,
  17,
  20,
  10,
  16,
  1,
  5,
  0,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_linear_kelvin_voigt_total_tet4_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::linear_kelvin_voigt_total_tet4_jacobian_action_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void linear_kelvin_voigt_total_tet4_residual_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_adj4,
    const g_t *const RSTR g_adj5,
    const g_t *const RSTR g_adj6,
    const g_t *const RSTR g_adj7,
    const g_t *const RSTR g_adj8,
    const g_t *const RSTR g_det0,
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const s_t *const RSTR u2_old,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int NC = 3;

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    s_t bcurrent[NC * NS];
    s_t bprevious[NC * NS];
    s_t boutput[NC * NS];
    const s_t *const current_components[NC] = {u0, u1, u2};
    const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bcurrent[stream] = current_components[field][node * current_stride];
          bprevious[stream] = previous_components[field][node * previous_stride];
        }
      }
    }

    for (int stream = 0; stream < 12; ++stream) {
      {
        boutput[stream] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[10] = {g_adj0 + evb, g_adj1 + evb, g_adj2 + evb, g_adj3 + evb, g_adj4 + evb, g_adj5 + evb, g_adj6 + evb, g_adj7 + evb, g_adj8 + evb, g_det0 + evb};
    s_t baffine_geometry_data[10];
    const s_t *bageom_streams[10];
    for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
      bageom_streams[geometry_stream] = ageom_stream<s_t, g_t>(
          affine_geometry_sources[geometry_stream], &baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
    }
    const s_t *badjugate[9];
    for (int component = 0; component < 9; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    linear_kelvin_voigt_total_d3_simplex_tet4_residual_block_contiguous<s_t, NQ, NS>(bageom_streams[9], badjugate, bcurrent, bprevious, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
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

extern "C" int cu_linear_kelvin_voigt_total_tet4_residual_a_msoa(
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
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
    const ptrdiff_t previous_stride,
    const void *const RSTR u0_old,
    const void *const RSTR u1_old,
    const void *const RSTR u2_old,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::linear_kelvin_voigt_total_tet4_residual_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("linear_kelvin_voigt_total_tet4_residual_a_msoa_impl");
    }
    case (int)sizeof(float): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::linear_kelvin_voigt_total_tet4_residual_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("linear_kelvin_voigt_total_tet4_residual_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("linear_kelvin_voigt_total_tet4_residual_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void linear_kelvin_voigt_total_tet4_jacobian_action_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_adj4,
    const g_t *const RSTR g_adj5,
    const g_t *const RSTR g_adj6,
    const g_t *const RSTR g_adj7,
    const g_t *const RSTR g_adj8,
    const g_t *const RSTR g_det0,
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u0_direction,
    const s_t *const RSTR u1_direction,
    const s_t *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int NC = 3;

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    s_t bdirection[NC * NS];
    s_t boutput[NC * NS];
    const s_t *const direction_components[NC] = {u0_direction, u1_direction, u2_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bdirection[stream] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 12; ++stream) {
      {
        boutput[stream] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[10] = {g_adj0 + evb, g_adj1 + evb, g_adj2 + evb, g_adj3 + evb, g_adj4 + evb, g_adj5 + evb, g_adj6 + evb, g_adj7 + evb, g_adj8 + evb, g_det0 + evb};
    s_t baffine_geometry_data[10];
    const s_t *bageom_streams[10];
    for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
      bageom_streams[geometry_stream] = ageom_stream<s_t, g_t>(
          affine_geometry_sources[geometry_stream], &baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
    }
    const s_t *badjugate[9];
    for (int component = 0; component < 9; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    linear_kelvin_voigt_total_d3_simplex_tet4_jacobian_action_block_contiguous<s_t, NQ, NS>(bageom_streams[9], badjugate, bdirection, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
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

extern "C" int cu_linear_kelvin_voigt_total_tet4_jacobian_action_a_msoa(
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
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const void *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    void *const RSTR u0_out,
    void *const RSTR u1_out,
    void *const RSTR u2_out,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::linear_kelvin_voigt_total_tet4_jacobian_action_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("linear_kelvin_voigt_total_tet4_jacobian_action_a_msoa_impl");
    }
    case (int)sizeof(float): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::linear_kelvin_voigt_total_tet4_jacobian_action_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("linear_kelvin_voigt_total_tet4_jacobian_action_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("linear_kelvin_voigt_total_tet4_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_tet4_hessian_crs_a_msoa_find_cols(
    const idx_t *const RSTR targets,
    const idx_t *const RSTR row,
    const int lenrow,
    idx_t *const RSTR ks) {
#pragma unroll(4)
  for (int d = 0; d < 4; ++d) {
    ks[d] = 0;
  }
  for (int k = 0; k < lenrow; ++k) {
#pragma unroll(4)
    for (int d = 0; d < 4; ++d) {
      ks[d] += row[k] < targets[d];
    }
  }
}

template <typename s_t, typename v_t>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_tet4_hessian_crs_a_msoa_scatter_crs(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    v_t *const RSTR values) {
  static constexpr int NS = 4;
  static constexpr int NC = 3;
  static constexpr int N_COL_STREAMS = 12;
  count_t entries[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    const count_t row_begin = rowptr[ev[i]];
    const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);
    const idx_t *const RSTR cols = &colidx[row_begin];
    linear_kelvin_voigt_total_tet4_hessian_crs_a_msoa_find_cols(ev, cols, lenrow, ks);
    for (int j = 0; j < NS; ++j) {
      entries[i * NS + j] = row_begin + ks[j];
    }
  }
  for (int bi = 0; bi < NC; ++bi) {
    for (int row_shape = 0; row_shape < NS; ++row_shape) {
      const s_t *const RSTR row = &element_matrix[(bi * NS + row_shape) * N_COL_STREAMS];
      for (int bj = 0; bj < NC; ++bj) {
        for (int col_shape = 0; col_shape < NS; ++col_shape) {
          v_t *const block = &values[(ptrdiff_t)entries[row_shape * NS + col_shape] * NC * NC];
          atomicAdd(&(block[bi * NC + bj]), v_t(row[bj * NS + col_shape]));
        }
      }
    }
  }
}

template <typename s_t, typename v_t>
__global__ void linear_kelvin_voigt_total_tet4_hessian_crs_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
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
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    v_t *const RSTR values
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 1;
  static constexpr int NS = 4;

  for (ptrdiff_t element = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; element < nelements; element += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[NS];
    s_t element_matrix[144];
    s_t badjugate_data[ND * ND][NQ];
    s_t bdeterminant[NQ];

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t node = elements[shape][element];
      ev[shape] = node;
    }


    badjugate_data[0][0] = s_t(g_adj0[element]);
    badjugate_data[1][0] = s_t(g_adj1[element]);
    badjugate_data[2][0] = s_t(g_adj2[element]);
    badjugate_data[3][0] = s_t(g_adj3[element]);
    badjugate_data[4][0] = s_t(g_adj4[element]);
    badjugate_data[5][0] = s_t(g_adj5[element]);
    badjugate_data[6][0] = s_t(g_adj6[element]);
    badjugate_data[7][0] = s_t(g_adj7[element]);
    badjugate_data[8][0] = s_t(g_adj8[element]);
    bdeterminant[0] = s_t(g_det0[element]);
    const s_t *const badjugate[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    linear_kelvin_voigt_total_d3_simplex_tet4_hessian_block<s_t, NQ, NS>(0, bdeterminant, badjugate, eta_b, eta_s, lmbda, mu, u_dt_shift, element_matrix);

    linear_kelvin_voigt_total_tet4_hessian_crs_a_msoa_scatter_crs(ev, element_matrix, rowptr, colidx, values);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_linear_kelvin_voigt_total_tet4_hessian_bsr_a_msoa(
    const int scalar_bytes,
    const int values_bytes,
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
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      switch (values_bytes) {
        case (int)sizeof(double): {
          if (nelements <= 0) return SFEM_SUCCESS;
          const int block_size = 256;
          const int grid_size = (int)((nelements + block_size - 1) / block_size);
          sfem::codegen::linear_kelvin_voigt_total_tet4_hessian_crs_a_msoa_impl<double, double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, rowptr, colidx, (double *)values);
          return sfem::codegen::launch_status("linear_kelvin_voigt_total_tet4_hessian_crs_a_msoa_impl");
        }
        case (int)sizeof(float): {
          if (nelements <= 0) return SFEM_SUCCESS;
          const int block_size = 256;
          const int grid_size = (int)((nelements + block_size - 1) / block_size);
          sfem::codegen::linear_kelvin_voigt_total_tet4_hessian_crs_a_msoa_impl<double, float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, rowptr, colidx, (float *)values);
          return sfem::codegen::launch_status("linear_kelvin_voigt_total_tet4_hessian_crs_a_msoa_impl");
        }
        default:
          break;
      }
      break;
    }
    case (int)sizeof(float): {
      switch (values_bytes) {
        case (int)sizeof(float): {
          if (nelements <= 0) return SFEM_SUCCESS;
          const int block_size = 256;
          const int grid_size = (int)((nelements + block_size - 1) / block_size);
          sfem::codegen::linear_kelvin_voigt_total_tet4_hessian_crs_a_msoa_impl<float, float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, rowptr, colidx, (float *)values);
          return sfem::codegen::launch_status("linear_kelvin_voigt_total_tet4_hessian_crs_a_msoa_impl");
        }
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_values_dispatch("linear_kelvin_voigt_total_tet4_hessian_bsr_a_msoa", (int)scalar_bytes, (int)values_bytes);
}
