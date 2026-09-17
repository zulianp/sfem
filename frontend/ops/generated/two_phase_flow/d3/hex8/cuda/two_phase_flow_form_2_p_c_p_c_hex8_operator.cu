#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../../cuda/two_phase_flow_form_2_p_c_p_c_d3_tensor_product_local.cuh"
#include "../../../../cuda/geometry_kernels.cuh"
#include "../../../../cuda/kernel_diagnostics.cuh"
#include "../../../../reference/cuda/line_p1_q4.hpp"
#include "../../../../reference/cuda/quad_line_q4.hpp"
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

template <typename s_t, typename g_t, int VS>
__host__ __device__ __forceinline__ const s_t *ageom_stream(
    const int,
    const g_t *const RSTR source,
    s_t *const RSTR,
    std::true_type) {
  return source;
}

template <typename s_t, typename g_t, int VS>
__host__ __device__ __forceinline__ const s_t *ageom_stream(
    const int ne,
    const g_t *const RSTR source,
    s_t *const RSTR converted,
    std::false_type) {
  {
    converted[0] = s_t(source[0]);
  }
  return converted;
}

} // namespace codegen
} // namespace sfem
namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_hex8_residual_esoa_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_hex8_residual_esoa",
  "HEX8",
  3,
  64,
  8,
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
  12160,
  18624,
  15,
  35,
  10,
  16,
  4,
  26,
  32,
  0,
  16,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_c_hex8_residual_esoa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_hex8_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_w_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_w",
  "HEX8",
  3,
  64,
  8,
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
  12160,
  18624,
  16,
  34,
  10,
  16,
  4,
  19,
  16,
  16,
  16,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_w_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_c_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_c",
  "HEX8",
  3,
  64,
  8,
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
  12160,
  18624,
  11,
  27,
  10,
  16,
  4,
  19,
  16,
  16,
  16,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_w_p_c_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_w_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_w",
  "HEX8",
  3,
  64,
  8,
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
  12160,
  18624,
  9,
  25,
  10,
  16,
  4,
  21,
  16,
  16,
  16,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_w_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_c_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_c",
  "HEX8",
  3,
  64,
  8,
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
  12160,
  18624,
  15,
  35,
  10,
  16,
  4,
  21,
  16,
  16,
  16,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_hex8_jacobian_p_c_p_c_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_esoa_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_esoa",
  "HEX8",
  3,
  64,
  8,
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
  12160,
  18624,
  39,
  36,
  10,
  16,
  4,
  26,
  16,
  16,
  16,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_esoa_diagnostics_data;
}

extern "C" int cu_two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[9],
    const void *const RSTR current[16],
    const void *const RSTR direction[16],
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
    void *const RSTR output[16],
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::two_phase_flow_form_2_p_c_p_c_d3_tensor_product_jacobian_action_block<double, 64, 8, 16>(ne, geometry_stride, (const double *)determinant, (const double *const *)adjugate, sfem::codegen::ref_line_p1_q4<double>::shape_1d(), sfem::codegen::ref_line_p1_q4<double>::grad_1d(), sfem::codegen::quad_line_q4<double>::q_weight_1d(), (const double *const *)current, (const double *const *)direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::two_phase_flow_form_2_p_c_p_c_d3_tensor_product_jacobian_action_block<float, 64, 8, 16>(ne, geometry_stride, (const float *)determinant, (const float *const *)adjugate, sfem::codegen::ref_line_p1_q4<float>::shape_1d(), sfem::codegen::ref_line_p1_q4<float>::grad_1d(), sfem::codegen::quad_line_q4<float>::q_weight_1d(), (const float *const *)current, (const float *const *)direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_a_msoa_impl(
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
  static constexpr int NQ = 64;
  static constexpr int NS = 8;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p1_q4<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::ref_line_p1_q4<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q4<s_t>::q_weight_1d();
  const idx_t *const RSTR field_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    s_t bcurrent[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {p_w, p_c};
    const s_t *const direction_components[NC] = {p_w_direction, p_c_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = field_elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bcurrent[stream][0] = current_components[field][node * current_stride];
          bdirection[stream][0] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 16; ++stream) {
      {
        boutput[stream][0] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[10] = {g_adj0 + evb, g_adj1 + evb, g_adj2 + evb, g_adj3 + evb, g_adj4 + evb, g_adj5 + evb, g_adj6 + evb, g_adj7 + evb, g_adj8 + evb, g_det0 + evb};
    s_t baffine_geometry_data[10][VS];
    const s_t *bageom_streams[10];
    for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
      bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
          ne, affine_geometry_sources[geometry_stream], baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
    }
    const s_t *badjugate[9];
    for (int component = 0; component < 9; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    two_phase_flow_form_2_p_c_p_c_d3_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[9], badjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, bcurrent, bdirection, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, boutput);

    s_t *const output_components[NC] = {p_w_out, p_c_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = field_elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_a_msoa(
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
        sfem::codegen::two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
        return sfem::codegen::launch_status("two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
        return sfem::codegen::launch_status("two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
__global__ void two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_msoa_impl(
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
  static constexpr int NQ = 64;
  static constexpr int NS = 8;
  static constexpr int NC = 2;
  static constexpr int VS = 16;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q4<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q4<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q4<s_t>::q_weight_1d();
  const idx_t *const RSTR field_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};
  const idx_t *const RSTR coordinate_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    s_t bcoordinates[3 * NS][VS];
    s_t badjugate_data[9][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bcurrent[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = coordinate_elements[shape];
      for (int d = 0; d < ND; ++d) {
        {
          const idx_t node = element_shape[evb];
          bcoordinates[shape * ND + d][0] = coordinate_components[d][node];
        }
      }
    }
    const s_t *const current_components[NC] = {p_w, p_c};
    const s_t *const direction_components[NC] = {p_w_direction, p_c_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = field_elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bcurrent[stream][0] = current_components[field][node * current_stride];
          bdirection[stream][0] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 16; ++stream) {
      {
        boutput[stream][0] = s_t(0);
      }
    }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
        coordinate_grad_ref + NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);

    const s_t *const badjugate[9] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    two_phase_flow_form_2_p_c_p_c_d3_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, badjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, bcurrent, bdirection, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, boutput);

    s_t *const output_components[NC] = {p_w_out, p_c_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = field_elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_msoa(
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
        sfem::codegen::two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_msoa_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
        return sfem::codegen::launch_status("two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_msoa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
        return sfem::codegen::launch_status("two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int cu_two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_maos(
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
        return cu_two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], ((const double *)parameters)[1], ((const double *)parameters)[3], ((const double *)parameters)[4], ((const double *)parameters)[5], ((const double *)parameters)[6], ((const double *)parameters)[7], ((const double *)parameters)[8], ((const double *)parameters)[9], ((const double *)parameters)[10], ((const double *)parameters)[11], ((const double *)parameters)[12], ((const double *)parameters)[13], ((const double *)parameters)[14], ((const double *)parameters)[15], ((const double *)parameters)[16], ((const double *)parameters)[17], ((const double *)parameters)[18], ((const double *)parameters)[20], ((const double *)parameters)[21], ((const double *)parameters)[24], 2, (const double *)current + 0, (const double *)current + 1, 2, (const double *)direction + 0, (const double *)direction + 1, 2, (double *)output + 0, (double *)output + 1, stream);
    }
    case (int)sizeof(float): {
        return cu_two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], ((const float *)parameters)[1], ((const float *)parameters)[3], ((const float *)parameters)[4], ((const float *)parameters)[5], ((const float *)parameters)[6], ((const float *)parameters)[7], ((const float *)parameters)[8], ((const float *)parameters)[9], ((const float *)parameters)[10], ((const float *)parameters)[11], ((const float *)parameters)[12], ((const float *)parameters)[13], ((const float *)parameters)[14], ((const float *)parameters)[15], ((const float *)parameters)[16], ((const float *)parameters)[17], ((const float *)parameters)[18], ((const float *)parameters)[20], ((const float *)parameters)[21], ((const float *)parameters)[24], 2, (const float *)current + 0, (const float *)current + 1, 2, (const float *)direction + 0, (const float *)direction + 1, 2, (float *)output + 0, (float *)output + 1, stream);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_form_2_p_c_p_c_hex8_jacobian_action_i_maos", -1, (int)scalar_bytes);
}
