#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../../cuda/two_phase_flow_form_2_p_c_p_w_d2_tensor_product_local.cuh"
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

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_esoa_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_esoa",
  "PROTEUS_QUAD4",
  2,
  16,
  4,
  16,
  4,
  27,
  44,
  9,
  1,
  11,
  2,
  0,
  0,
  35,
  17,
  206,
  1584,
  2144,
  15,
  28,
  5,
  16,
  4,
  21,
  16,
  0,
  8,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_esoa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_w_proteus_quad4_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_w_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_w",
  "PROTEUS_QUAD4",
  2,
  16,
  4,
  16,
  4,
  19,
  38,
  9,
  1,
  5,
  1,
  0,
  0,
  24,
  17,
  166,
  1584,
  2144,
  16,
  27,
  5,
  16,
  4,
  14,
  8,
  8,
  8,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_w_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_c_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_c",
  "PROTEUS_QUAD4",
  2,
  16,
  4,
  16,
  4,
  13,
  24,
  8,
  0,
  6,
  1,
  0,
  0,
  22,
  12,
  127,
  1584,
  2144,
  11,
  20,
  5,
  16,
  4,
  14,
  8,
  8,
  8,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_w_p_c_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_w_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_w",
  "PROTEUS_QUAD4",
  2,
  16,
  4,
  16,
  4,
  11,
  24,
  10,
  0,
  5,
  0,
  0,
  0,
  24,
  10,
  120,
  1584,
  2144,
  9,
  21,
  5,
  16,
  4,
  16,
  8,
  8,
  8,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_w_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_c_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_c",
  "PROTEUS_QUAD4",
  2,
  16,
  4,
  16,
  4,
  20,
  36,
  10,
  0,
  6,
  0,
  0,
  0,
  26,
  16,
  142,
  1584,
  2144,
  15,
  28,
  5,
  16,
  4,
  16,
  8,
  8,
  8,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_p_c_p_c_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_esoa_diagnostics_data = {
  "two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_esoa",
  "PROTEUS_QUAD4",
  2,
  16,
  4,
  16,
  4,
  41,
  83,
  14,
  1,
  9,
  1,
  0,
  0,
  39,
  41,
  277,
  1584,
  2144,
  39,
  29,
  5,
  16,
  4,
  21,
  8,
  8,
  8,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_esoa_diagnostics_data;
}

extern "C" int cu_two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[4],
    const void *const RSTR current[8],
    const void *const RSTR direction[8],
    const real_t C_ka1,
    const real_t C_ka2,
    const real_t K_0,
    const real_t K_1,
    const real_t K_2,
    const real_t K_3,
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
    void *const RSTR output[8],
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::two_phase_flow_form_2_p_c_p_w_d2_tensor_product_jacobian_action_block<double, 16, 4, 1>(ne, geometry_stride, (const double *)determinant, (const double *const *)adjugate, sfem::codegen::ref_line_p1_q4<double>::shape_1d(), sfem::codegen::ref_line_p1_q4<double>::grad_1d(), sfem::codegen::quad_line_q4<double>::q_weight_1d(), (const double *const *)current, (const double *const *)direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::two_phase_flow_form_2_p_c_p_w_d2_tensor_product_jacobian_action_block<float, 16, 4, 1>(ne, geometry_stride, (const float *)determinant, (const float *const *)adjugate, sfem::codegen::ref_line_p1_q4<float>::shape_1d(), sfem::codegen::ref_line_p1_q4<float>::grad_1d(), sfem::codegen::quad_line_q4<float>::q_weight_1d(), (const float *const *)current, (const float *const *)direction, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_det0,
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
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
  static constexpr int NQ = 16;
  static constexpr int NS = 4;
  static constexpr int NC = 2;
  static constexpr int VS = 1;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p1_q4<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::ref_line_p1_q4<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q4<s_t>::q_weight_1d();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    s_t bcurrent[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {p_w, p_c};
    const s_t *const direction_components[NC] = {p_w_direction, p_c_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bcurrent[stream][0] = current_components[field][node * current_stride];
          bdirection[stream][0] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 8; ++stream) {
      {
        boutput[stream][0] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[5] = {g_adj0 + evb, g_adj1 + evb, g_adj2 + evb, g_adj3 + evb, g_det0 + evb};
    s_t baffine_geometry_data[5][VS];
    const s_t *bageom_streams[5];
    for (int geometry_stream = 0; geometry_stream < 5; ++geometry_stream) {
      bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
          ne, affine_geometry_sources[geometry_stream], baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
    }
    const s_t *badjugate[4];
    for (int component = 0; component < 4; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    two_phase_flow_form_2_p_c_p_w_d2_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[4], badjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, bcurrent, bdirection, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, boutput);

    s_t *const output_components[NC] = {p_w_out, p_c_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
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

extern "C" int cu_two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const real_t C_ka1,
    const real_t C_ka2,
    const real_t K_0,
    const real_t K_1,
    const real_t K_2,
    const real_t K_3,
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
        sfem::codegen::two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
        return sfem::codegen::launch_status("two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
        return sfem::codegen::launch_status("two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
__global__ void two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_msoa_impl(
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
  static constexpr int ND = 2;
  static constexpr int NQ = 16;
  static constexpr int NS = 4;
  static constexpr int NC = 2;
  static constexpr int VS = 1;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q4<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q4<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q4<s_t>::q_weight_1d();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    s_t bcoordinates[2 * NS][VS];
    s_t badjugate_data[4][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bcurrent[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1]};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
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
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bcurrent[stream][0] = current_components[field][node * current_stride];
          bdirection[stream][0] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 8; ++stream) {
      {
        boutput[stream][0] = s_t(0);
      }
    }

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 2>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
        coordinate_grad_ref + NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);

    const s_t *const badjugate[4] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};

    two_phase_flow_form_2_p_c_p_w_d2_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, badjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, bcurrent, bdirection, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, boutput);

    s_t *const output_components[NC] = {p_w_out, p_c_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
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

extern "C" int cu_two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_msoa(
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
        sfem::codegen::two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_msoa_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
        return sfem::codegen::launch_status("two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_msoa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, C_ka1, C_ka2, K_0, K_1, K_2, K_3, M_c, P_r, R, S_res, T, Z, dt, m, mu_c, porosity, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
        return sfem::codegen::launch_status("two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int cu_two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_maos(
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
        return cu_two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], ((const double *)parameters)[1], ((const double *)parameters)[3], ((const double *)parameters)[4], ((const double *)parameters)[5], ((const double *)parameters)[6], ((const double *)parameters)[7], ((const double *)parameters)[8], ((const double *)parameters)[9], ((const double *)parameters)[10], ((const double *)parameters)[11], ((const double *)parameters)[12], ((const double *)parameters)[13], ((const double *)parameters)[15], ((const double *)parameters)[16], ((const double *)parameters)[19], 2, (const double *)current + 0, (const double *)current + 1, 2, (const double *)direction + 0, (const double *)direction + 1, 2, (double *)output + 0, (double *)output + 1, stream);
    }
    case (int)sizeof(float): {
        return cu_two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], ((const float *)parameters)[1], ((const float *)parameters)[3], ((const float *)parameters)[4], ((const float *)parameters)[5], ((const float *)parameters)[6], ((const float *)parameters)[7], ((const float *)parameters)[8], ((const float *)parameters)[9], ((const float *)parameters)[10], ((const float *)parameters)[11], ((const float *)parameters)[12], ((const float *)parameters)[13], ((const float *)parameters)[15], ((const float *)parameters)[16], ((const float *)parameters)[19], 2, (const float *)current + 0, (const float *)current + 1, 2, (const float *)direction + 0, (const float *)direction + 1, 2, (float *)output + 0, (float *)output + 1, stream);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_form_2_p_c_p_w_proteus_quad4_jacobian_action_i_maos", -1, (int)scalar_bytes);
}
