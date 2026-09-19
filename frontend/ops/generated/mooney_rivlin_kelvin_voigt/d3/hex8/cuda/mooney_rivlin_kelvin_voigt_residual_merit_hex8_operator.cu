#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../../cuda/mooney_rivlin_kelvin_voigt_residual_merit_d3_tensor_product_local.cuh"
#include "../../../../cuda/geometry_kernels.cuh"
#include "../../../../cuda/kernel_diagnostics.cuh"
#include "../../../../reference/cuda/line_p1_q2.hpp"
#include "../../../../reference/cuda/quad_line_q2.hpp"
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
extern "C" int cu_mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[9],
    const void *const RSTR current[24],
    const void *const RSTR previous[24],
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    void *const RSTR output[24],
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_d3_tensor_product_residual_block<double, 8, 8, 1>(ne, geometry_stride, (const double *)determinant, (const double *const *)adjugate, sfem::codegen::ref_line_p1_q2<double>::shape_1d(), sfem::codegen::ref_line_p1_q2<double>::grad_1d(), sfem::codegen::quad_line_q2<double>::q_weight_1d(), (const double *const *)current, (const double *const *)previous, eta_b, eta_s, lmbda, mu, u_dt_shift, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_d3_tensor_product_residual_block<float, 8, 8, 1>(ne, geometry_stride, (const float *)determinant, (const float *const *)adjugate, sfem::codegen::ref_line_p1_q2<float>::shape_1d(), sfem::codegen::ref_line_p1_q2<float>::grad_1d(), sfem::codegen::quad_line_q2<float>::q_weight_1d(), (const float *const *)current, (const float *const *)previous, eta_b, eta_s, lmbda, mu, u_dt_shift, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_a_msoa_impl(
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
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int NC = 3;
  static constexpr int VS = 1;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();
  const idx_t *const RSTR field_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {u0, u1, u2};
    const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = field_elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bcurrent[stream][0] = current_components[field][node * current_stride];
          bprevious[stream][0] = previous_components[field][node * previous_stride];
        }
      }
    }

    for (int stream = 0; stream < 24; ++stream) {
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

    mooney_rivlin_kelvin_voigt_residual_merit_d3_tensor_product_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[9], badjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, bcurrent, bprevious, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
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

extern "C" int cu_mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_a_msoa(
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
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
__global__ void mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
  static constexpr int ND = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int NC = 3;
  static constexpr int VS = 1;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();
  const idx_t *const RSTR field_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};
  const idx_t *const RSTR coordinate_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    s_t bcoordinates[3 * NS][VS];
    s_t badjugate_data[9][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
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
    const s_t *const current_components[NC] = {u0, u1, u2};
    const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = field_elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bcurrent[stream][0] = current_components[field][node * current_stride];
          bprevious[stream][0] = previous_components[field][node * previous_stride];
        }
      }
    }

    for (int stream = 0; stream < 24; ++stream) {
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

    mooney_rivlin_kelvin_voigt_residual_merit_d3_tensor_product_residual_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, badjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, bcurrent, bprevious, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
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

extern "C" int cu_mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_i_msoa_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_i_msoa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int cu_mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    void *const RSTR output,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return cu_mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], ((const double *)parameters)[1], ((const double *)parameters)[2], ((const double *)parameters)[3], ((const double *)parameters)[4], 3, (const double *)current + 0, (const double *)current + 1, (const double *)current + 2, 3, (const double *)previous + 0, (const double *)previous + 1, (const double *)previous + 2, 3, (double *)output + 0, (double *)output + 1, (double *)output + 2, stream);
    }
    case (int)sizeof(float): {
        return cu_mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], ((const float *)parameters)[1], ((const float *)parameters)[2], ((const float *)parameters)[3], ((const float *)parameters)[4], 3, (const float *)current + 0, (const float *)current + 1, (const float *)current + 2, 3, (const float *)previous + 0, (const float *)previous + 1, (const float *)previous + 2, 3, (float *)output + 0, (float *)output + 1, (float *)output + 2, stream);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_residual_merit_hex8_residual_i_maos", -1, (int)scalar_bytes);
}

extern "C" int cu_mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR adjugate[9],
    const void *const RSTR current[24],
    const void *const RSTR previous[24],
    const void *const RSTR direction[24],
    const real_t eta_b,
    const real_t eta_s,
    const real_t lmbda,
    const real_t mu,
    const real_t u_dt_shift,
    void *const RSTR output[24],
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_d3_tensor_product_jacobian_action_block<double, 8, 8, 1>(ne, geometry_stride, (const double *)determinant, (const double *const *)adjugate, sfem::codegen::ref_line_p1_q2<double>::shape_1d(), sfem::codegen::ref_line_p1_q2<double>::grad_1d(), sfem::codegen::quad_line_q2<double>::q_weight_1d(), (const double *const *)current, (const double *const *)previous, (const double *const *)direction, eta_b, eta_s, lmbda, mu, u_dt_shift, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_d3_tensor_product_jacobian_action_block<float, 8, 8, 1>(ne, geometry_stride, (const float *)determinant, (const float *const *)adjugate, sfem::codegen::ref_line_p1_q2<float>::shape_1d(), sfem::codegen::ref_line_p1_q2<float>::grad_1d(), sfem::codegen::quad_line_q2<float>::q_weight_1d(), (const float *const *)current, (const float *const *)previous, (const float *const *)direction, eta_b, eta_s, lmbda, mu, u_dt_shift, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_a_msoa_impl(
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
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int NC = 3;
  static constexpr int VS = 1;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();
  const idx_t *const RSTR field_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {u0, u1, u2};
    const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};
    const s_t *const direction_components[NC] = {u0_direction, u1_direction, u2_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = field_elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bcurrent[stream][0] = current_components[field][node * current_stride];
          bprevious[stream][0] = previous_components[field][node * previous_stride];
          bdirection[stream][0] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 24; ++stream) {
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

    mooney_rivlin_kelvin_voigt_residual_merit_d3_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[9], badjugate, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, bcurrent, bprevious, bdirection, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
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

extern "C" int cu_mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_a_msoa(
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
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
__global__ void mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
  static constexpr int ND = 3;
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int NC = 3;
  static constexpr int VS = 1;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();
  const idx_t *const RSTR field_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};
  const idx_t *const RSTR coordinate_elements[8] = {elements[0], elements[1], elements[3], elements[2], elements[4], elements[5], elements[7], elements[6]};

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    s_t bcoordinates[3 * NS][VS];
    s_t badjugate_data[9][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
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
    const s_t *const current_components[NC] = {u0, u1, u2};
    const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};
    const s_t *const direction_components[NC] = {u0_direction, u1_direction, u2_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = field_elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bcurrent[stream][0] = current_components[field][node * current_stride];
          bprevious[stream][0] = previous_components[field][node * previous_stride];
          bdirection[stream][0] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 24; ++stream) {
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

    mooney_rivlin_kelvin_voigt_residual_merit_d3_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, badjugate, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, bcurrent, bprevious, bdirection, eta_b, eta_s, lmbda, mu, u_dt_shift, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
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

extern "C" int cu_mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
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
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_i_msoa_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, previous_stride, (const double *)u0_old, (const double *)u1_old, (const double *)u2_old, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_i_msoa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, previous_stride, (const float *)u0_old, (const float *)u1_old, (const float *)u2_old, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int cu_mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    const void *const RSTR previous,
    const void *const RSTR direction,
    void *const RSTR output,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return cu_mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], ((const double *)parameters)[1], ((const double *)parameters)[2], ((const double *)parameters)[3], ((const double *)parameters)[4], 3, (const double *)current + 0, (const double *)current + 1, (const double *)current + 2, 3, (const double *)previous + 0, (const double *)previous + 1, (const double *)previous + 2, 3, (const double *)direction + 0, (const double *)direction + 1, (const double *)direction + 2, 3, (double *)output + 0, (double *)output + 1, (double *)output + 2, stream);
    }
    case (int)sizeof(float): {
        return cu_mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], ((const float *)parameters)[1], ((const float *)parameters)[2], ((const float *)parameters)[3], ((const float *)parameters)[4], 3, (const float *)current + 0, (const float *)current + 1, (const float *)current + 2, 3, (const float *)previous + 0, (const float *)previous + 1, (const float *)previous + 2, 3, (const float *)direction + 0, (const float *)direction + 1, (const float *)direction + 2, 3, (float *)output + 0, (float *)output + 1, (float *)output + 2, stream);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("mooney_rivlin_kelvin_voigt_residual_merit_hex8_jacobian_action_i_maos", -1, (int)scalar_bytes);
}
