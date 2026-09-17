#include <type_traits>
#include "../../cuda/navier_stokes_form_1_u_d3_tensor_product_mixed_local.cuh"
#include "../../../../reference/cuda/line_p1_q4.hpp"
#include "../../../../reference/cuda/line_p2_q4.hpp"
#include "../../../../reference/cuda/quad_line_q4.hpp"
#include "../../../../cuda/kernel_math.cuh"
#include "../../../../cuda/geometry_kernels.cuh"
#include "../../../../cuda/kernel_diagnostics.cuh"

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT
#endif
#ifndef RSTR
#define RSTR SFEM_RESTRICT
#endif
#ifndef SFEM_GENERATED_SCALAR_T
#define SFEM_GENERATED_SCALAR_T
typedef double real_t;
typedef ptrdiff_t idx_t;
typedef ptrdiff_t element_idx_t;
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

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

static const KernelDiagnostics navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_esoa_diagnostics_data = {
  "navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_esoa",
  "PROTEUS_HEX27",
  3,
  64,
  27,
  16,
  4,
  32,
  45,
  1,
  0,
  0,
  0,
  0,
  0,
  36,
  11,
  85,
  26992,
  36960,
  7,
  21,
  10,
  40,
  4,
  7,
  178,
  0,
  89,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_esoa_diagnostics(void) {
  return &sfem::codegen::navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_1_u_proteus_hex27_proteus_hex8_jacobian_action_esoa_diagnostics_data = {
  "navier_stokes_form_1_u_proteus_hex27_proteus_hex8_jacobian_action_esoa",
  "PROTEUS_HEX27",
  3,
  64,
  27,
  16,
  4,
  29,
  56,
  1,
  0,
  0,
  0,
  0,
  0,
  33,
  21,
  93,
  26992,
  36960,
  17,
  26,
  10,
  40,
  4,
  0,
  0,
  0,
  89,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_1_u_proteus_hex27_proteus_hex8_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::navier_stokes_form_1_u_proteus_hex27_proteus_hex8_jacobian_action_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_affine_mesh_mixed_impl(
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
    const s_t convection_scale,
    const s_t dt,
    const s_t f0,
    const s_t f1,
    const s_t f2,
    const s_t nu,
    const s_t rho,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u_data[3],
    const s_t *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u_old_data[3],
    const s_t *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    s_t *const RSTR u_out[3],
    s_t *const RSTR p_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 64;
  static constexpr int CELL_NS = 27;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 89;
  static constexpr int VS = 16;
  const s_t *const field_shape_1d[NC] = {sfem::codegen::ref_line_p2_q4<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q4<s_t>::shape_1d()};
  const s_t *const field_grad_1d[NC] = {sfem::codegen::ref_line_p2_q4<s_t>::grad_1d(), sfem::codegen::ref_line_p1_q4<s_t>::grad_1d()};
  const idx_t *const RSTR field_3_elements[8] = {elements[0], elements[2], elements[6], elements[8], elements[18], elements[20], elements[24], elements[26]};

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    s_t bcurrent[N_FIELD_STREAMS][VS];
    s_t bprevious[N_FIELD_STREAMS][VS];
    s_t boutput[N_FIELD_STREAMS][VS];

    for (int local_shape = 0; local_shape < 27; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 0 + local_shape;
      {
        const idx_t node = element_shape[evb];
        bcurrent[stream][0] = u_data[0][node * current_stride];
        bprevious[stream][0] = u_old_data[0][node * previous_stride];
      }
    }
    for (int local_shape = 0; local_shape < 27; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 27 + local_shape;
      {
        const idx_t node = element_shape[evb];
        bcurrent[stream][0] = u_data[1][node * current_stride];
        bprevious[stream][0] = u_old_data[1][node * previous_stride];
      }
    }
    for (int local_shape = 0; local_shape < 27; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 54 + local_shape;
      {
        const idx_t node = element_shape[evb];
        bcurrent[stream][0] = u_data[2][node * current_stride];
        bprevious[stream][0] = u_old_data[2][node * previous_stride];
      }
    }
    for (int local_shape = 0; local_shape < 8; ++local_shape) {
      const idx_t *const RSTR element_shape = field_3_elements[local_shape];
      const int stream = 81 + local_shape;
      {
        const idx_t node = element_shape[evb];
        bcurrent[stream][0] = p_data[node * current_stride];
        bprevious[stream][0] = p_old_data[node * previous_stride];
      }
    }

    for (int stream = 0; stream < 89; ++stream) {
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
    const s_t *badjugate[ND * ND];
    for (int component = 0; component < ND * ND; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    navier_stokes_form_1_u_d3_tensor_product_mixed_residual_block_contiguous<s_t, NQ, CELL_NS, VS>(ne, 0, bageom_streams[9], badjugate, field_shape_1d, field_grad_1d, sfem::codegen::quad_line_q4<s_t>::q_weight_1d(), bcurrent, bprevious, convection_scale, dt, f0, f1, f2, nu, rho, boutput);

    {
      s_t *const RSTR out = u_out[0];
      for (int local_shape = 0; local_shape < 27; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 0 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
    {
      s_t *const RSTR out = u_out[1];
      for (int local_shape = 0; local_shape < 27; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 27 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
    {
      s_t *const RSTR out = u_out[2];
      for (int local_shape = 0; local_shape < 27; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 54 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
    {
      s_t *const RSTR out = p_out;
      for (int local_shape = 0; local_shape < 8; ++local_shape) {
        const idx_t *const RSTR element_shape = field_3_elements[local_shape];
        const int stream = 81 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

extern "C" int cu_navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_a_msoa(
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
    const real_t convection_scale,
    const real_t dt,
    const real_t f0,
    const real_t f1,
    const real_t f2,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[3],
    const void *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const void *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_affine_mesh_mixed_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, (const double *const *)u_data, (const double *)p_data, previous_stride, (const double *const *)u_old_data, (const double *)p_old_data, out_stride, (double *const *)u_out, (double *)p_out);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_affine_mesh_mixed_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, (const float *const *)u_data, (const float *)p_data, previous_stride, (const float *const *)u_old_data, (const float *)p_old_data, out_stride, (float *const *)u_out, (float *)p_out);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
__global__ void navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_mixed_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t convection_scale,
    const s_t dt,
    const s_t f0,
    const s_t f1,
    const s_t f2,
    const s_t nu,
    const s_t rho,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u_data[3],
    const s_t *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u_old_data[3],
    const s_t *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    s_t *const RSTR u_out[3],
    s_t *const RSTR p_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 64;
  static constexpr int CELL_NS = 27;
  static constexpr int NS = CELL_NS;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 89;
  static constexpr int VS = 16;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p2_q4<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p2_q4<s_t>::grad_1d();
  const idx_t *const RSTR field_3_elements[8] = {elements[0], elements[2], elements[6], elements[8], elements[18], elements[20], elements[24], elements[26]};
  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    s_t bcoordinates[ND * CELL_NS][VS];
    s_t badjugate_data[ND * ND][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bcurrent[N_FIELD_STREAMS][VS];
    s_t bprevious[N_FIELD_STREAMS][VS];
    s_t boutput[N_FIELD_STREAMS][VS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int d = 0; d < ND; ++d) {
        {
          const idx_t node = element_shape[evb];
          bcoordinates[shape * ND + d][0] = coordinate_components[d][node];
        }
      }
    }

    for (int local_shape = 0; local_shape < 27; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 0 + local_shape;
      {
        const idx_t node = element_shape[evb];
        bcurrent[stream][0] = u_data[0][node * current_stride];
        bprevious[stream][0] = u_old_data[0][node * previous_stride];
      }
    }
    for (int local_shape = 0; local_shape < 27; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 27 + local_shape;
      {
        const idx_t node = element_shape[evb];
        bcurrent[stream][0] = u_data[1][node * current_stride];
        bprevious[stream][0] = u_old_data[1][node * previous_stride];
      }
    }
    for (int local_shape = 0; local_shape < 27; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 54 + local_shape;
      {
        const idx_t node = element_shape[evb];
        bcurrent[stream][0] = u_data[2][node * current_stride];
        bprevious[stream][0] = u_old_data[2][node * previous_stride];
      }
    }
    for (int local_shape = 0; local_shape < 8; ++local_shape) {
      const idx_t *const RSTR element_shape = field_3_elements[local_shape];
      const int stream = 81 + local_shape;
      {
        const idx_t node = element_shape[evb];
        bcurrent[stream][0] = p_data[node * current_stride];
        bprevious[stream][0] = p_old_data[node * previous_stride];
      }
    }

    for (int stream = 0; stream < 89; ++stream) {
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

    const s_t *const field_shape_1d[NC] = {sfem::codegen::ref_line_p2_q4<s_t>::shape_1d(), sfem::codegen::ref_line_p1_q4<s_t>::shape_1d()};
    const s_t *const field_grad_1d[NC] = {sfem::codegen::ref_line_p2_q4<s_t>::grad_1d(), sfem::codegen::ref_line_p1_q4<s_t>::grad_1d()};
    const s_t *const badjugate[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    navier_stokes_form_1_u_d3_tensor_product_mixed_residual_block_contiguous<s_t, NQ, CELL_NS, VS>(ne, VS, bdeterminant, badjugate, field_shape_1d, field_grad_1d, sfem::codegen::quad_line_q4<s_t>::q_weight_1d(), bcurrent, bprevious, convection_scale, dt, f0, f1, f2, nu, rho, boutput);

    {
      s_t *const RSTR out = u_out[0];
      for (int local_shape = 0; local_shape < 27; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 0 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
    {
      s_t *const RSTR out = u_out[1];
      for (int local_shape = 0; local_shape < 27; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 27 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
    {
      s_t *const RSTR out = u_out[2];
      for (int local_shape = 0; local_shape < 27; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 54 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
    {
      s_t *const RSTR out = p_out;
      for (int local_shape = 0; local_shape < 8; ++local_shape) {
        const idx_t *const RSTR element_shape = field_3_elements[local_shape];
        const int stream = 81 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

extern "C" int cu_navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t convection_scale,
    const real_t dt,
    const real_t f0,
    const real_t f1,
    const real_t f2,
    const real_t nu,
    const real_t rho,
    const ptrdiff_t current_stride,
    const void *const RSTR u_data[3],
    const void *const RSTR p_data,
    const ptrdiff_t previous_stride,
    const void *const RSTR u_old_data[3],
    const void *const RSTR p_old_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[3],
    void *const RSTR p_out,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_mixed_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, (const double *const *)u_data, (const double *)p_data, previous_stride, (const double *const *)u_old_data, (const double *)p_old_data, out_stride, (double *const *)u_out, (double *)p_out);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_isoparametric_mesh_mixed_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, convection_scale, dt, f0, f1, f2, nu, rho, current_stride, (const float *const *)u_data, (const float *)p_data, previous_stride, (const float *const *)u_old_data, (const float *)p_old_data, out_stride, (float *const *)u_out, (float *)p_out);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("navier_stokes_form_1_u_proteus_hex27_proteus_hex8_residual_i_msoa", -1, (int)scalar_bytes);
}
