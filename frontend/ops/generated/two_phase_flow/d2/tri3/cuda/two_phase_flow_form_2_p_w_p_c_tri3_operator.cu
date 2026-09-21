#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../../cuda/two_phase_flow_form_2_p_w_p_c_d2_simplex_local.cuh"
#include "../../../../cuda/geometry_kernels.cuh"
#include "../../../../cuda/kernel_diagnostics.cuh"
#include "../../../../reference/cuda/quad_tri_q6.hpp"
#include "../../../../reference/cuda/tri3_q6.hpp"
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

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_c_tri3_residual_esoa_diagnostics_data = {
  "two_phase_flow_form_2_p_w_p_c_tri3_residual_esoa",
  "TRI3",
  2,
  6,
  3,
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
  486,
  648,
  15,
  28,
  5,
  54,
  6,
  21,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_w_p_c_tri3_residual_esoa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_w_p_c_tri3_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_w_diagnostics_data = {
  "two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_w",
  "TRI3",
  2,
  6,
  3,
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
  486,
  648,
  16,
  27,
  5,
  54,
  6,
  14,
  6,
  6,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_w_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_c_diagnostics_data = {
  "two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_c",
  "TRI3",
  2,
  6,
  3,
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
  486,
  648,
  11,
  20,
  5,
  54,
  6,
  14,
  6,
  6,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_w_p_c_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_w_diagnostics_data = {
  "two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_w",
  "TRI3",
  2,
  6,
  3,
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
  486,
  648,
  9,
  21,
  5,
  54,
  6,
  16,
  6,
  6,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_w_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_w_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_c_diagnostics_data = {
  "two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_c",
  "TRI3",
  2,
  6,
  3,
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
  486,
  648,
  15,
  28,
  5,
  54,
  6,
  16,
  6,
  6,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_c_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_w_p_c_tri3_jacobian_p_c_p_c_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_esoa_diagnostics_data = {
  "two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_esoa",
  "TRI3",
  2,
  6,
  3,
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
  486,
  648,
  39,
  29,
  5,
  54,
  6,
  21,
  6,
  6,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_esoa_diagnostics_data;
}

extern "C" int cu_two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[4],
    const void *const RSTR current[6],
    const void *const RSTR direction[6],
    const real_t C_kw1,
    const real_t K_0,
    const real_t K_1,
    const real_t K_2,
    const real_t K_3,
    const real_t P_r,
    const real_t S_res,
    const real_t dt,
    const real_t kappa_T,
    const real_t m,
    const real_t mu_w,
    const real_t p_wr,
    const real_t porosity,
    const real_t rho_w0,
    void *const RSTR output[6],
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::two_phase_flow_form_2_p_w_p_c_d2_simplex_jacobian_action_block<double, 6, 3, 1>(ne, geometry_stride, (const double *)determinant, (const double *const *)adjugate, sfem::codegen::ref_tri3_q6<double>::shape(), sfem::codegen::ref_tri3_q6<double>::grad_ref_x(), sfem::codegen::ref_tri3_q6<double>::grad_ref_y(), sfem::codegen::quad_tri_q6<double>::q_weight(), (const double *const *)current, (const double *const *)direction, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::two_phase_flow_form_2_p_w_p_c_d2_simplex_jacobian_action_block<float, 6, 3, 1>(ne, geometry_stride, (const float *)determinant, (const float *const *)adjugate, sfem::codegen::ref_tri3_q6<float>::shape(), sfem::codegen::ref_tri3_q6<float>::grad_ref_x(), sfem::codegen::ref_tri3_q6<float>::grad_ref_y(), sfem::codegen::quad_tri_q6<float>::q_weight(), (const float *const *)current, (const float *const *)direction, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_det0,
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t P_r,
    const s_t S_res,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
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
  static constexpr int NQ = 6;
  static constexpr int NS = 3;
  static constexpr int NC = 2;
  static constexpr int VS = 1;
  const s_t *const affine_shape = sfem::codegen::ref_tri3_q6<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::ref_tri3_q6<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::ref_tri3_q6<s_t>::grad_ref_y();
  const s_t *const affine_q_weight = sfem::codegen::quad_tri_q6<s_t>::q_weight();

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

    for (int stream = 0; stream < 6; ++stream) {
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

    two_phase_flow_form_2_p_w_p_c_d2_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[4], badjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_q_weight, bcurrent, bdirection, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, boutput);

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

extern "C" int cu_two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const real_t C_kw1,
    const real_t K_0,
    const real_t K_1,
    const real_t K_2,
    const real_t K_3,
    const real_t P_r,
    const real_t S_res,
    const real_t dt,
    const real_t kappa_T,
    const real_t m,
    const real_t mu_w,
    const real_t p_wr,
    const real_t porosity,
    const real_t rho_w0,
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
        sfem::codegen::two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const double *)p_w, (const double *)p_c, direction_stride, (const double *)p_w_direction, (const double *)p_c_direction, out_stride, (double *)p_w_out, (double *)p_c_out);
        return sfem::codegen::launch_status("two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, C_kw1, K_0, K_1, K_2, K_3, P_r, S_res, dt, kappa_T, m, mu_w, p_wr, porosity, rho_w0, current_stride, (const float *)p_w, (const float *)p_c, direction_stride, (const float *)p_w_direction, (const float *)p_c_direction, out_stride, (float *)p_w_out, (float *)p_c_out);
        return sfem::codegen::launch_status("two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("two_phase_flow_form_2_p_w_p_c_tri3_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}
