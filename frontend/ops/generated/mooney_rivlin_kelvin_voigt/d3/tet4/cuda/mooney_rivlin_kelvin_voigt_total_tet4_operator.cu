#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../../cuda/mooney_rivlin_kelvin_voigt_total_d3_simplex_local.cuh"
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
    const int,
    const g_t *const RSTR source,
    s_t *const RSTR,
    std::true_type) {
  return source;
}

template <typename s_t, typename g_t>
__host__ __device__ __forceinline__ const s_t *ageom_stream(
    const int,
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

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet4_residual_esoa_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet4_residual_esoa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  206,
  239,
  1,
  0,
  9,
  0,
  0,
  0,
  32,
  84,
  462,
  253,
  366,
  81,
  62,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet4_residual_esoa_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u0_u0_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u0_u0",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  186,
  230,
  1,
  0,
  10,
  0,
  0,
  0,
  29,
  99,
  434,
  253,
  366,
  98,
  59,
  10,
  16,
  1,
  5,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u0_u0_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u0_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u0_u1_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u0_u1",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  211,
  268,
  1,
  0,
  1,
  0,
  0,
  0,
  29,
  115,
  488,
  253,
  366,
  114,
  77,
  10,
  16,
  1,
  5,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u0_u1_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u0_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u0_u2_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u0_u2",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  211,
  266,
  1,
  0,
  1,
  0,
  0,
  0,
  29,
  113,
  486,
  253,
  366,
  112,
  77,
  10,
  16,
  1,
  5,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u0_u2_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u0_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u1_u0_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u1_u0",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  211,
  268,
  1,
  0,
  1,
  0,
  0,
  0,
  29,
  115,
  488,
  253,
  366,
  114,
  77,
  10,
  16,
  1,
  5,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u1_u0_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u1_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u1_u1_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u1_u1",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  186,
  230,
  1,
  0,
  10,
  0,
  0,
  0,
  29,
  98,
  434,
  253,
  366,
  97,
  58,
  10,
  16,
  1,
  5,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u1_u1_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u1_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u1_u2_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u1_u2",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  211,
  266,
  1,
  0,
  1,
  0,
  0,
  0,
  29,
  114,
  486,
  253,
  366,
  113,
  77,
  10,
  16,
  1,
  5,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u1_u2_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u1_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u2_u0_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u2_u0",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  211,
  266,
  1,
  0,
  1,
  0,
  0,
  0,
  29,
  114,
  486,
  253,
  366,
  113,
  76,
  10,
  16,
  1,
  5,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u2_u0_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u2_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u2_u1_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u2_u1",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  211,
  266,
  1,
  0,
  1,
  0,
  0,
  0,
  29,
  112,
  486,
  253,
  366,
  111,
  76,
  10,
  16,
  1,
  5,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u2_u1_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u2_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u2_u2_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u2_u2",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  185,
  227,
  1,
  0,
  10,
  0,
  0,
  0,
  29,
  97,
  430,
  253,
  366,
  96,
  57,
  10,
  16,
  1,
  5,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u2_u2_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_jacobian_u2_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_esoa_diagnostics_data = {
  "mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_esoa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  973,
  1170,
  1,
  0,
  19,
  0,
  0,
  0,
  41,
  464,
  2170,
  253,
  366,
  461,
  267,
  10,
  16,
  1,
  5,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_esoa_diagnostics_data;
}

extern "C" int cu_mooney_rivlin_kelvin_voigt_total_tet4_residual_esoa(
    const int scalar_bytes,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[9],
    const void *const RSTR current[12],
    const void *const RSTR previous[12],
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    void *const RSTR output[12],
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_d3_simplex_tet4_residual_block<double, 1, 4>((const double *)determinant, (const double *const *)adjugate, (const double *const *)current, (const double *const *)previous, eta_b, eta_s, lmbda, mu, u_dt_shift, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_d3_simplex_tet4_residual_block<float, 1, 4>((const float *)determinant, (const float *const *)adjugate, (const float *const *)current, (const float *const *)previous, eta_b, eta_s, lmbda, mu, u_dt_shift, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tet4_residual_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void mooney_rivlin_kelvin_voigt_total_tet4_residual_a_msoa_impl(
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
          ne, affine_geometry_sources[geometry_stream], &baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
    }
    const s_t *badjugate[9];
    for (int component = 0; component < 9; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    mooney_rivlin_kelvin_voigt_total_d3_simplex_tet4_residual_block_contiguous<s_t, NQ, NS>(bageom_streams[9], badjugate, bcurrent, bprevious, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

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

extern "C" int cu_mooney_rivlin_kelvin_voigt_total_tet4_residual_a_msoa(
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
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_residual_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("mooney_rivlin_kelvin_voigt_total_tet4_residual_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_residual_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("mooney_rivlin_kelvin_voigt_total_tet4_residual_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tet4_residual_a_msoa", -1, (int)scalar_bytes);
}

extern "C" int cu_mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_esoa(
    const int scalar_bytes,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[9],
    const void *const RSTR current[12],
    const void *const RSTR previous[12],
    const void *const RSTR direction[12],
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    void *const RSTR output[12],
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_d3_simplex_tet4_jacobian_action_block<double, 1, 4>((const double *)determinant, (const double *const *)adjugate, (const double *const *)current, (const double *const *)previous, (const double *const *)direction, eta_b, eta_s, lmbda, mu, u_dt_shift, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_d3_simplex_tet4_jacobian_action_block<float, 1, 4>((const float *)determinant, (const float *const *)adjugate, (const float *const *)current, (const float *const *)previous, (const float *const *)direction, eta_b, eta_s, lmbda, mu, u_dt_shift, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_a_msoa_impl(
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
    s_t bcurrent[NC * NS];
    s_t bprevious[NC * NS];
    s_t bdirection[NC * NS];
    s_t boutput[NC * NS];
    const s_t *const current_components[NC] = {u0, u1, u2};
    const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};
    const s_t *const direction_components[NC] = {u0_direction, u1_direction, u2_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bcurrent[stream] = current_components[field][node * current_stride];
          bprevious[stream] = previous_components[field][node * previous_stride];
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
          ne, affine_geometry_sources[geometry_stream], &baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
    }
    const s_t *badjugate[9];
    for (int component = 0; component < 9; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    mooney_rivlin_kelvin_voigt_total_d3_simplex_tet4_jacobian_action_block_contiguous<s_t, NQ, NS>(bageom_streams[9], badjugate, bcurrent, bprevious, bdirection, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

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

extern "C" int cu_mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_a_msoa(
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
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tet4_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_tet4_hessian_crs_a_msoa_find_cols(
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

template <typename s_t>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_tet4_hessian_crs_a_msoa_scatter_crs(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values) {
  static constexpr int NS = 4;
  static constexpr int NC = 3;
  static constexpr int N_COL_STREAMS = 12;
  count_t entries[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    const count_t row_begin = rowptr[ev[i]];
    const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);
    const idx_t *const RSTR cols = &colidx[row_begin];
    mooney_rivlin_kelvin_voigt_total_tet4_hessian_crs_a_msoa_find_cols(ev, cols, lenrow, ks);
    for (int j = 0; j < NS; ++j) {
      entries[i * NS + j] = row_begin + ks[j];
    }
  }
  for (int bi = 0; bi < NC; ++bi) {
    for (int row_shape = 0; row_shape < NS; ++row_shape) {
      const s_t *const RSTR row = &element_matrix[(bi * NS + row_shape) * N_COL_STREAMS];
      for (int bj = 0; bj < NC; ++bj) {
        for (int col_shape = 0; col_shape < NS; ++col_shape) {
          s_t *const block = &values[entries[row_shape * NS + col_shape] * NC * NC];
          atomicAdd(&(block[bi * NC + bj]), row[bj * NS + col_shape]);
        }
      }
    }
  }
}

template <typename s_t>
__global__ void mooney_rivlin_kelvin_voigt_total_tet4_hessian_crs_a_msoa_impl(
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
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const s_t *const RSTR u2_old,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int NC = 3;
  static constexpr int N_STREAMS = NC * NS;

  for (ptrdiff_t element = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; element < nelements; element += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[NS];
    s_t element_matrix[144];
    s_t badjugate_data[ND * ND][NQ];
    s_t bdeterminant[NQ];
    s_t bcurrent[N_STREAMS];
    s_t bprevious[N_STREAMS];

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t node = elements[shape][element];
      ev[shape] = node;
      bcurrent[shape * NC + 0] = u0[node * current_stride];
      bcurrent[shape * NC + 1] = u1[node * current_stride];
      bcurrent[shape * NC + 2] = u2[node * current_stride];
      bprevious[shape * NC + 0] = u0_old[node * previous_stride];
      bprevious[shape * NC + 1] = u1_old[node * previous_stride];
      bprevious[shape * NC + 2] = u2_old[node * previous_stride];
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

    mooney_rivlin_kelvin_voigt_total_d3_simplex_tet4_hessian_block<s_t, NQ, NS>(1, bdeterminant, badjugate, bcurrent, bprevious, eta_b, eta_s, lmbda, mu, u_dt_shift, element_matrix);

    mooney_rivlin_kelvin_voigt_total_tet4_hessian_crs_a_msoa_scatter_crs(ev, element_matrix, rowptr, colidx, values);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_mooney_rivlin_kelvin_voigt_total_tet4_hessian_bsr_a_msoa(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      const int block_size = 256;
      const int grid_size = (int)((nelements + block_size - 1) / block_size);
      sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_hessian_crs_a_msoa_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, rowptr, colidx, (double *)values);
      return sfem::codegen::launch_status("mooney_rivlin_kelvin_voigt_total_tet4_hessian_crs_a_msoa_impl");
    }
    case (int)sizeof(float): {
      const int block_size = 256;
      const int grid_size = (int)((nelements + block_size - 1) / block_size);
      sfem::codegen::mooney_rivlin_kelvin_voigt_total_tet4_hessian_crs_a_msoa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, rowptr, colidx, (float *)values);
      return sfem::codegen::launch_status("mooney_rivlin_kelvin_voigt_total_tet4_hessian_crs_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_total_tet4_hessian_bsr_a_msoa", -1, (int)scalar_bytes);
}
