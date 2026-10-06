#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../../cuda/two_phase_flow_form_2_p_c_p_c_d3_simplex_local.cuh"
#include "../../../../cuda/geometry_kernels.cuh"
#include "../../../../cuda/kernel_diagnostics.cuh"
#include "../../../../reference/cuda/quad_tet_q11.hpp"
#include "../../../../reference/cuda/tet10_q11.hpp"
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

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_tet10_residual_esoa_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_tet10_residual_esoa",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  37,
  56,
  9,
  1,
  11,
  2,
  0,
  0,
  44,
  17,
  228,
  3839,
  6270,
  15,
  35,
  10,
  440,
  11,
  26,
  40,
  0,
  20,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_c_tet10_residual_esoa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tet10_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_w_p_w_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_w_p_w",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  29,
  51,
  9,
  1,
  5,
  1,
  0,
  0,
  32,
  17,
  189,
  3839,
  6270,
  16,
  34,
  10,
  440,
  11,
  19,
  20,
  20,
  20,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_w_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_w_p_w_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_w_p_c_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_w_p_c",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  18,
  30,
  8,
  0,
  6,
  1,
  0,
  0,
  29,
  12,
  138,
  3839,
  6270,
  11,
  27,
  10,
  440,
  11,
  19,
  20,
  20,
  20,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_w_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_w_p_c_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_c_p_w_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_c_p_w",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  16,
  30,
  10,
  0,
  5,
  0,
  0,
  0,
  31,
  10,
  131,
  3839,
  6270,
  9,
  25,
  10,
  440,
  11,
  21,
  20,
  20,
  20,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_c_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_c_p_w_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_c_p_c_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_c_p_c",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  30,
  49,
  10,
  0,
  6,
  0,
  0,
  0,
  34,
  16,
  165,
  3839,
  6270,
  15,
  35,
  10,
  440,
  11,
  21,
  20,
  20,
  20,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_c_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tet10_jacobian_p_c_p_c_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_esoa_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_esoa",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  61,
  109,
  14,
  1,
  9,
  1,
  0,
  0,
  50,
  41,
  323,
  3839,
  6270,
  39,
  36,
  10,
  440,
  11,
  26,
  20,
  20,
  20,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_esoa_diagnostics_data;
}

extern "C" int cu_two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_esoa(
    const int scalar_bytes,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[9],
    const void *const RSTR current[20],
    const void *const RSTR direction[20],
    const real_t C_ka1,
    const real_t C_ka2,
    const real_t K_0,
    const real_t K_1,
    const real_t K_2,
    const real_t K_3,
    const real_t K_4,
    const real_t K_5,
    const real_t K_6,
    const real_t K_7,
    const real_t K_8,
    const real_t M_c,
    const real_t P_r,
    const real_t R,
    const real_t S_res,
    const real_t T,
    const real_t Z,
    const real_t dt,
    const real_t m,
    const real_t mu_c,
    const real_t porosity,
    void *const RSTR output[20],
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::two_phase_flow_form_2_p_c_p_c_d3_simplex_jacobian_action_block<double, 11, 10>(geometry_stride, (const double *)determinant, (const double *const *)adjugate, sfem::codegen::ref_tet10_q11<double>::shape(), sfem::codegen::ref_tet10_q11<double>::grad_ref_x(), sfem::codegen::ref_tet10_q11<double>::grad_ref_y(), sfem::codegen::ref_tet10_q11<double>::grad_ref_z(), sfem::codegen::quad_tet_q11<double>::q_weight(), (const double *const *)current, (const double *const *)direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::two_phase_flow_form_2_p_c_p_c_d3_simplex_jacobian_action_block<float, 11, 10>(geometry_stride, (const float *)determinant, (const float *const *)adjugate, sfem::codegen::ref_tet10_q11<float>::shape(), sfem::codegen::ref_tet10_q11<float>::grad_ref_x(), sfem::codegen::ref_tet10_q11<float>::grad_ref_y(), sfem::codegen::ref_tet10_q11<float>::grad_ref_z(), sfem::codegen::quad_tet_q11<float>::q_weight(), (const float *const *)current, (const float *const *)direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_a_msoa_impl(
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
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t P_r,
    const s_t R,
    const s_t S_res,
    const s_t T,
    const s_t Z,
    const s_t dt,
    const s_t m,
    const s_t mu_c,
    const s_t porosity,
    const ptrdiff_t current_stride,
    const s_t *const RSTR p_w,
    const s_t *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR p_w_direction,
    const s_t *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR p_w_out,
    s_t *const RSTR p_c_out
) {
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  static constexpr int NC = 2;
  const s_t *const affine_shape = sfem::codegen::ref_tet10_q11<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    s_t bcurrent[NC * NS];
    s_t bdirection[NC * NS];
    s_t boutput[NC * NS];
    const s_t *const current_components[NC] = {p_w, p_c};
    const s_t *const direction_components[NC] = {p_w_direction, p_c_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bcurrent[stream] = current_components[field][node * current_stride];
          bdirection[stream] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 20; ++stream) {
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

    two_phase_flow_form_2_p_c_p_c_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS>(0, bageom_streams[9], badjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, bcurrent, bdirection, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, boutput);

    s_t *const output_components[NC] = {p_w_out, p_c_out};
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

extern "C" int cu_two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_a_msoa(
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
    const real_t C_ka1,
    const real_t C_ka2,
    const real_t K_0,
    const real_t K_1,
    const real_t K_2,
    const real_t K_3,
    const real_t K_4,
    const real_t K_5,
    const real_t K_6,
    const real_t K_7,
    const real_t K_8,
    const real_t M_c,
    const real_t P_r,
    const real_t R,
    const real_t S_res,
    const real_t T,
    const real_t Z,
    const real_t dt,
    const real_t m,
    const real_t mu_c,
    const real_t porosity,
    const ptrdiff_t current_stride,
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
        return sfem::codegen::launch_status("two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
        return sfem::codegen::launch_status("two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
__global__ void two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t P_r,
    const s_t R,
    const s_t S_res,
    const s_t T,
    const s_t Z,
    const s_t dt,
    const s_t m,
    const s_t mu_c,
    const s_t porosity,
    const ptrdiff_t current_stride,
    const s_t *const RSTR p_w,
    const s_t *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR p_w_direction,
    const s_t *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR p_w_out,
    s_t *const RSTR p_c_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  static constexpr int NC = 2;
  const s_t *const isoparametric_shape = sfem::codegen::ref_tet10_q11<s_t>::shape();
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    s_t bcoordinates[3 * NS];
    s_t badjugate_data[9][NQ];
    s_t bdeterminant[NQ];
    s_t bcurrent[NC * NS];
    s_t bdirection[NC * NS];
    s_t boutput[NC * NS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int d = 0; d < ND; ++d) {
        {
          const idx_t node = element_shape[evb];
          bcoordinates[shape * ND + d] = coordinate_components[d][node];
        }
      }
    }
    const s_t *const current_components[NC] = {p_w, p_c};
    const s_t *const direction_components[NC] = {p_w_direction, p_c_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bcurrent[stream] = current_components[field][node * current_stride];
          bdirection[stream] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 20; ++stream) {
      {
        boutput[stream] = s_t(0);
      }
    }

    s_t *badjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    for (int q = 0; q < NQ; ++q) {
      const s_t cell_grad_ref0_0 = isoparametric_grad_ref_x[q * NS + 0];
      const s_t cell_grad_ref0_1 = isoparametric_grad_ref_x[q * NS + 1];
      const s_t cell_grad_ref0_2 = isoparametric_grad_ref_x[q * NS + 2];
      const s_t cell_grad_ref0_3 = isoparametric_grad_ref_x[q * NS + 3];
      const s_t cell_grad_ref0_4 = isoparametric_grad_ref_x[q * NS + 4];
      const s_t cell_grad_ref0_5 = isoparametric_grad_ref_x[q * NS + 5];
      const s_t cell_grad_ref0_6 = isoparametric_grad_ref_x[q * NS + 6];
      const s_t cell_grad_ref0_7 = isoparametric_grad_ref_x[q * NS + 7];
      const s_t cell_grad_ref0_8 = isoparametric_grad_ref_x[q * NS + 8];
      const s_t cell_grad_ref0_9 = isoparametric_grad_ref_x[q * NS + 9];
      const s_t cell_grad_ref1_0 = isoparametric_grad_ref_y[q * NS + 0];
      const s_t cell_grad_ref1_1 = isoparametric_grad_ref_y[q * NS + 1];
      const s_t cell_grad_ref1_2 = isoparametric_grad_ref_y[q * NS + 2];
      const s_t cell_grad_ref1_3 = isoparametric_grad_ref_y[q * NS + 3];
      const s_t cell_grad_ref1_4 = isoparametric_grad_ref_y[q * NS + 4];
      const s_t cell_grad_ref1_5 = isoparametric_grad_ref_y[q * NS + 5];
      const s_t cell_grad_ref1_6 = isoparametric_grad_ref_y[q * NS + 6];
      const s_t cell_grad_ref1_7 = isoparametric_grad_ref_y[q * NS + 7];
      const s_t cell_grad_ref1_8 = isoparametric_grad_ref_y[q * NS + 8];
      const s_t cell_grad_ref1_9 = isoparametric_grad_ref_y[q * NS + 9];
      const s_t cell_grad_ref2_0 = isoparametric_grad_ref_z[q * NS + 0];
      const s_t cell_grad_ref2_1 = isoparametric_grad_ref_z[q * NS + 1];
      const s_t cell_grad_ref2_2 = isoparametric_grad_ref_z[q * NS + 2];
      const s_t cell_grad_ref2_3 = isoparametric_grad_ref_z[q * NS + 3];
      const s_t cell_grad_ref2_4 = isoparametric_grad_ref_z[q * NS + 4];
      const s_t cell_grad_ref2_5 = isoparametric_grad_ref_z[q * NS + 5];
      const s_t cell_grad_ref2_6 = isoparametric_grad_ref_z[q * NS + 6];
      const s_t cell_grad_ref2_7 = isoparametric_grad_ref_z[q * NS + 7];
      const s_t cell_grad_ref2_8 = isoparametric_grad_ref_z[q * NS + 8];
      const s_t cell_grad_ref2_9 = isoparametric_grad_ref_z[q * NS + 9];
      {
        const s_t J00 = bcoordinates[0] * cell_grad_ref0_0 + bcoordinates[3] * cell_grad_ref0_1 + bcoordinates[6] * cell_grad_ref0_2 + bcoordinates[9] * cell_grad_ref0_3 + bcoordinates[12] * cell_grad_ref0_4 + bcoordinates[15] * cell_grad_ref0_5 + bcoordinates[18] * cell_grad_ref0_6 + bcoordinates[21] * cell_grad_ref0_7 + bcoordinates[24] * cell_grad_ref0_8 + bcoordinates[27] * cell_grad_ref0_9;
        const s_t J01 = bcoordinates[0] * cell_grad_ref1_0 + bcoordinates[3] * cell_grad_ref1_1 + bcoordinates[6] * cell_grad_ref1_2 + bcoordinates[9] * cell_grad_ref1_3 + bcoordinates[12] * cell_grad_ref1_4 + bcoordinates[15] * cell_grad_ref1_5 + bcoordinates[18] * cell_grad_ref1_6 + bcoordinates[21] * cell_grad_ref1_7 + bcoordinates[24] * cell_grad_ref1_8 + bcoordinates[27] * cell_grad_ref1_9;
        const s_t J02 = bcoordinates[0] * cell_grad_ref2_0 + bcoordinates[3] * cell_grad_ref2_1 + bcoordinates[6] * cell_grad_ref2_2 + bcoordinates[9] * cell_grad_ref2_3 + bcoordinates[12] * cell_grad_ref2_4 + bcoordinates[15] * cell_grad_ref2_5 + bcoordinates[18] * cell_grad_ref2_6 + bcoordinates[21] * cell_grad_ref2_7 + bcoordinates[24] * cell_grad_ref2_8 + bcoordinates[27] * cell_grad_ref2_9;
        const s_t J10 = bcoordinates[1] * cell_grad_ref0_0 + bcoordinates[4] * cell_grad_ref0_1 + bcoordinates[7] * cell_grad_ref0_2 + bcoordinates[10] * cell_grad_ref0_3 + bcoordinates[13] * cell_grad_ref0_4 + bcoordinates[16] * cell_grad_ref0_5 + bcoordinates[19] * cell_grad_ref0_6 + bcoordinates[22] * cell_grad_ref0_7 + bcoordinates[25] * cell_grad_ref0_8 + bcoordinates[28] * cell_grad_ref0_9;
        const s_t J11 = bcoordinates[1] * cell_grad_ref1_0 + bcoordinates[4] * cell_grad_ref1_1 + bcoordinates[7] * cell_grad_ref1_2 + bcoordinates[10] * cell_grad_ref1_3 + bcoordinates[13] * cell_grad_ref1_4 + bcoordinates[16] * cell_grad_ref1_5 + bcoordinates[19] * cell_grad_ref1_6 + bcoordinates[22] * cell_grad_ref1_7 + bcoordinates[25] * cell_grad_ref1_8 + bcoordinates[28] * cell_grad_ref1_9;
        const s_t J12 = bcoordinates[1] * cell_grad_ref2_0 + bcoordinates[4] * cell_grad_ref2_1 + bcoordinates[7] * cell_grad_ref2_2 + bcoordinates[10] * cell_grad_ref2_3 + bcoordinates[13] * cell_grad_ref2_4 + bcoordinates[16] * cell_grad_ref2_5 + bcoordinates[19] * cell_grad_ref2_6 + bcoordinates[22] * cell_grad_ref2_7 + bcoordinates[25] * cell_grad_ref2_8 + bcoordinates[28] * cell_grad_ref2_9;
        const s_t J20 = bcoordinates[2] * cell_grad_ref0_0 + bcoordinates[5] * cell_grad_ref0_1 + bcoordinates[8] * cell_grad_ref0_2 + bcoordinates[11] * cell_grad_ref0_3 + bcoordinates[14] * cell_grad_ref0_4 + bcoordinates[17] * cell_grad_ref0_5 + bcoordinates[20] * cell_grad_ref0_6 + bcoordinates[23] * cell_grad_ref0_7 + bcoordinates[26] * cell_grad_ref0_8 + bcoordinates[29] * cell_grad_ref0_9;
        const s_t J21 = bcoordinates[2] * cell_grad_ref1_0 + bcoordinates[5] * cell_grad_ref1_1 + bcoordinates[8] * cell_grad_ref1_2 + bcoordinates[11] * cell_grad_ref1_3 + bcoordinates[14] * cell_grad_ref1_4 + bcoordinates[17] * cell_grad_ref1_5 + bcoordinates[20] * cell_grad_ref1_6 + bcoordinates[23] * cell_grad_ref1_7 + bcoordinates[26] * cell_grad_ref1_8 + bcoordinates[29] * cell_grad_ref1_9;
        const s_t J22 = bcoordinates[2] * cell_grad_ref2_0 + bcoordinates[5] * cell_grad_ref2_1 + bcoordinates[8] * cell_grad_ref2_2 + bcoordinates[11] * cell_grad_ref2_3 + bcoordinates[14] * cell_grad_ref2_4 + bcoordinates[17] * cell_grad_ref2_5 + bcoordinates[20] * cell_grad_ref2_6 + bcoordinates[23] * cell_grad_ref2_7 + bcoordinates[26] * cell_grad_ref2_8 + bcoordinates[29] * cell_grad_ref2_9;
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badjugate_streams, bdeterminant, q);
      }
    }

    const s_t *const badjugate[9] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    two_phase_flow_form_2_p_c_p_c_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS>(1, bdeterminant, badjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, bcurrent, bdirection, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, boutput);

    s_t *const output_components[NC] = {p_w_out, p_c_out};
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

extern "C" int cu_two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t C_ka1,
    const real_t C_ka2,
    const real_t K_0,
    const real_t K_1,
    const real_t K_2,
    const real_t K_3,
    const real_t K_4,
    const real_t K_5,
    const real_t K_6,
    const real_t K_7,
    const real_t K_8,
    const real_t M_c,
    const real_t P_r,
    const real_t R,
    const real_t S_res,
    const real_t T,
    const real_t Z,
    const real_t dt,
    const real_t m,
    const real_t mu_c,
    const real_t porosity,
    const ptrdiff_t current_stride,
    const void *const RSTR p_w,
    const void *const RSTR p_c,
    const ptrdiff_t direction_stride,
    const void *const RSTR p_w_direction,
    const void *const RSTR p_c_direction,
    const ptrdiff_t out_stride,
    void *const RSTR p_w_out,
    void *const RSTR p_c_out,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_i_msoa_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
        return sfem::codegen::launch_status("two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_i_msoa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
        return sfem::codegen::launch_status("two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int cu_two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR direction,
    void *const RSTR output,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return cu_two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], ((const double *)parameters)[1], ((const double *)parameters)[3], ((const double *)parameters)[4], ((const double *)parameters)[5], ((const double *)parameters)[6], ((const double *)parameters)[7], ((const double *)parameters)[8], ((const double *)parameters)[9], ((const double *)parameters)[10], ((const double *)parameters)[11], ((const double *)parameters)[12], ((const double *)parameters)[13], ((const double *)parameters)[14], ((const double *)parameters)[15], ((const double *)parameters)[16], ((const double *)parameters)[17], ((const double *)parameters)[18], ((const double *)parameters)[20], ((const double *)parameters)[21], ((const double *)parameters)[24], 2, (const double *)current + 0, (const double *)current + 1, 2, (const double *)direction + 0, (const double *)direction + 1, 2, (double *)output + 0, (double *)output + 1, stream);
    }
    case (int)sizeof(float): {
        return cu_two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], ((const float *)parameters)[1], ((const float *)parameters)[3], ((const float *)parameters)[4], ((const float *)parameters)[5], ((const float *)parameters)[6], ((const float *)parameters)[7], ((const float *)parameters)[8], ((const float *)parameters)[9], ((const float *)parameters)[10], ((const float *)parameters)[11], ((const float *)parameters)[12], ((const float *)parameters)[13], ((const float *)parameters)[14], ((const float *)parameters)[15], ((const float *)parameters)[16], ((const float *)parameters)[17], ((const float *)parameters)[18], ((const float *)parameters)[20], ((const float *)parameters)[21], ((const float *)parameters)[24], 2, (const float *)current + 0, (const float *)current + 1, 2, (const float *)direction + 0, (const float *)direction + 1, 2, (float *)output + 0, (float *)output + 1, stream);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_form_2_p_c_p_c_tet10_jacobian_action_i_maos", -1, (int)scalar_bytes);
}
