#include <type_traits>
#include "../../cuda/navier_stokes_form_2_u_p_d2_simplex_mixed_local.cuh"
#include "../../../../reference/cuda/quad_tri_q6.hpp"
#include "../../../../reference/cuda/tri3_q6.hpp"
#include "../../../../reference/cuda/tri6_q6.hpp"
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

static const KernelDiagnostics navier_stokes_form_2_u_p_tri6_tri3_residual_esoa_diagnostics_data = {
  "navier_stokes_form_2_u_p_tri6_tri3_residual_esoa",
  "TRI6",
  2,
  6,
  6,
  16,
  4,
  16,
  26,
  1,
  0,
  0,
  0,
  0,
  0,
  22,
  8,
  50,
  1062,
  1368,
  5,
  16,
  5,
  162,
  6,
  0,
  0,
  0,
  15,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_2_u_p_tri6_tri3_residual_esoa_diagnostics(void) {
  return &sfem::codegen::navier_stokes_form_2_u_p_tri6_tri3_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_esoa_diagnostics_data = {
  "navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_esoa",
  "TRI6",
  2,
  6,
  6,
  16,
  4,
  13,
  28,
  1,
  0,
  0,
  0,
  0,
  0,
  20,
  11,
  49,
  1062,
  1368,
  8,
  16,
  5,
  162,
  6,
  0,
  0,
  15,
  15,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_mixed_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_det0,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u_direction_data[2],
    const s_t *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    s_t *const RSTR u_out[2],
    s_t *const RSTR p_out
) {
  static constexpr int ND = 2;
  static constexpr int NQ = 6;
  static constexpr int CELL_NS = 6;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 15;
  static constexpr int VS = 16;
  const s_t *const field_shape[NC] = {sfem::codegen::ref_tri6_q6<s_t>::shape(), sfem::codegen::ref_tri3_q6<s_t>::shape()};
  const s_t *const fgref[NC * ND] = {sfem::codegen::ref_tri6_q6<s_t>::grad_ref_x(), sfem::codegen::ref_tri6_q6<s_t>::grad_ref_y(), sfem::codegen::ref_tri3_q6<s_t>::grad_ref_x(), sfem::codegen::ref_tri3_q6<s_t>::grad_ref_y()};

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    s_t bdirection[N_FIELD_STREAMS][VS];
    s_t boutput[N_FIELD_STREAMS][VS];

    for (int local_shape = 0; local_shape < 6; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 0 + local_shape;
      {
        const idx_t node = element_shape[evb];
        bdirection[stream][0] = u_direction_data[0][node * direction_stride];
      }
    }
    for (int local_shape = 0; local_shape < 6; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 6 + local_shape;
      {
        const idx_t node = element_shape[evb];
        bdirection[stream][0] = u_direction_data[1][node * direction_stride];
      }
    }
    for (int local_shape = 0; local_shape < 3; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 12 + local_shape;
      {
        const idx_t node = element_shape[evb];
        bdirection[stream][0] = p_direction_data[node * direction_stride];
      }
    }

    for (int stream = 0; stream < 15; ++stream) {
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
    const s_t *badjugate[ND * ND];
    for (int component = 0; component < ND * ND; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    navier_stokes_form_2_u_p_d2_simplex_mixed_jacobian_action_block_contiguous<s_t, NQ, CELL_NS, VS>(ne, 0, bageom_streams[4], badjugate, field_shape, fgref, sfem::codegen::quad_tri_q6<s_t>::q_weight(), bdirection, boutput);

    {
      s_t *const RSTR out = u_out[0];
      for (int local_shape = 0; local_shape < 6; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 0 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
    {
      s_t *const RSTR out = u_out[1];
      for (int local_shape = 0; local_shape < 6; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 6 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
    {
      s_t *const RSTR out = p_out;
      for (int local_shape = 0; local_shape < 3; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 12 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

extern "C" int cu_navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_det0,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[2],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[2],
    void *const RSTR p_out,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_mixed_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
        return sfem::codegen::launch_status("navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_mixed_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_mixed_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
        return sfem::codegen::launch_status("navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_affine_mesh_mixed_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
__global__ void navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_mixed_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u_direction_data[2],
    const s_t *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    s_t *const RSTR u_out[2],
    s_t *const RSTR p_out
) {
  static constexpr int ND = 2;
  static constexpr int NQ = 6;
  static constexpr int CELL_NS = 6;
  static constexpr int NS = CELL_NS;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 15;
  static constexpr int VS = 16;
  const s_t *const isoparametric_cell_grad_ref_0 = sfem::codegen::ref_tri6_q6<s_t>::grad_ref_x();
  const s_t *const isoparametric_cell_grad_ref_1 = sfem::codegen::ref_tri6_q6<s_t>::grad_ref_y();
  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    s_t bcoordinates[ND * CELL_NS][VS];
    s_t badjugate_data[ND * ND][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bdirection[N_FIELD_STREAMS][VS];
    s_t boutput[N_FIELD_STREAMS][VS];

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

    for (int local_shape = 0; local_shape < 6; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 0 + local_shape;
      {
        const idx_t node = element_shape[evb];
        bdirection[stream][0] = u_direction_data[0][node * direction_stride];
      }
    }
    for (int local_shape = 0; local_shape < 6; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 6 + local_shape;
      {
        const idx_t node = element_shape[evb];
        bdirection[stream][0] = u_direction_data[1][node * direction_stride];
      }
    }
    for (int local_shape = 0; local_shape < 3; ++local_shape) {
      const idx_t *const RSTR element_shape = elements[local_shape];
      const int stream = 12 + local_shape;
      {
        const idx_t node = element_shape[evb];
        bdirection[stream][0] = p_direction_data[node * direction_stride];
      }
    }

    for (int stream = 0; stream < 15; ++stream) {
      {
        boutput[stream][0] = s_t(0);
      }
    }

    s_t *badjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};
    for (int q = 0; q < NQ; ++q) {
      {
        const s_t J00 = bcoordinates[0][0] * isoparametric_cell_grad_ref_0[q * CELL_NS] + bcoordinates[2][0] * isoparametric_cell_grad_ref_0[q * CELL_NS + 1] + bcoordinates[4][0] * isoparametric_cell_grad_ref_0[q * CELL_NS + 2] + bcoordinates[6][0] * isoparametric_cell_grad_ref_0[q * CELL_NS + 3] + bcoordinates[8][0] * isoparametric_cell_grad_ref_0[q * CELL_NS + 4] + bcoordinates[10][0] * isoparametric_cell_grad_ref_0[q * CELL_NS + 5];
        const s_t J01 = bcoordinates[0][0] * isoparametric_cell_grad_ref_1[q * CELL_NS] + bcoordinates[2][0] * isoparametric_cell_grad_ref_1[q * CELL_NS + 1] + bcoordinates[4][0] * isoparametric_cell_grad_ref_1[q * CELL_NS + 2] + bcoordinates[6][0] * isoparametric_cell_grad_ref_1[q * CELL_NS + 3] + bcoordinates[8][0] * isoparametric_cell_grad_ref_1[q * CELL_NS + 4] + bcoordinates[10][0] * isoparametric_cell_grad_ref_1[q * CELL_NS + 5];
        const s_t J10 = bcoordinates[1][0] * isoparametric_cell_grad_ref_0[q * CELL_NS] + bcoordinates[3][0] * isoparametric_cell_grad_ref_0[q * CELL_NS + 1] + bcoordinates[5][0] * isoparametric_cell_grad_ref_0[q * CELL_NS + 2] + bcoordinates[7][0] * isoparametric_cell_grad_ref_0[q * CELL_NS + 3] + bcoordinates[9][0] * isoparametric_cell_grad_ref_0[q * CELL_NS + 4] + bcoordinates[11][0] * isoparametric_cell_grad_ref_0[q * CELL_NS + 5];
        const s_t J11 = bcoordinates[1][0] * isoparametric_cell_grad_ref_1[q * CELL_NS] + bcoordinates[3][0] * isoparametric_cell_grad_ref_1[q * CELL_NS + 1] + bcoordinates[5][0] * isoparametric_cell_grad_ref_1[q * CELL_NS + 2] + bcoordinates[7][0] * isoparametric_cell_grad_ref_1[q * CELL_NS + 3] + bcoordinates[9][0] * isoparametric_cell_grad_ref_1[q * CELL_NS + 4] + bcoordinates[11][0] * isoparametric_cell_grad_ref_1[q * CELL_NS + 5];
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badjugate_streams, bdeterminant, q * VS);
      }
    }

    const s_t *const field_shape[NC] = {sfem::codegen::ref_tri6_q6<s_t>::shape(), sfem::codegen::ref_tri3_q6<s_t>::shape()};
    const s_t *const fgref[NC * ND] = {sfem::codegen::ref_tri6_q6<s_t>::grad_ref_x(), sfem::codegen::ref_tri6_q6<s_t>::grad_ref_y(), sfem::codegen::ref_tri3_q6<s_t>::grad_ref_x(), sfem::codegen::ref_tri3_q6<s_t>::grad_ref_y()};
    const s_t *const badjugate[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3]};

    navier_stokes_form_2_u_p_d2_simplex_mixed_jacobian_action_block_contiguous<s_t, NQ, CELL_NS, VS>(ne, VS, bdeterminant, badjugate, field_shape, fgref, sfem::codegen::quad_tri_q6<s_t>::q_weight(), bdirection, boutput);

    {
      s_t *const RSTR out = u_out[0];
      for (int local_shape = 0; local_shape < 6; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 0 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
    {
      s_t *const RSTR out = u_out[1];
      for (int local_shape = 0; local_shape < 6; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 6 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
    {
      s_t *const RSTR out = p_out;
      for (int local_shape = 0; local_shape < 3; ++local_shape) {
        const idx_t *const RSTR element_shape = elements[local_shape];
        const int stream = 12 + local_shape;
        for (int scatter = 0; scatter < ne; ++scatter) {
          atomicAdd(&(out[element_shape[evb + scatter] * out_stride]), boutput[stream][scatter]);
        }
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

extern "C" int cu_navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const ptrdiff_t direction_stride,
    const void *const RSTR u_direction_data[2],
    const void *const RSTR p_direction_data,
    const ptrdiff_t out_stride,
    void *const RSTR u_out[2],
    void *const RSTR p_out,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_mixed_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, direction_stride, (const double *const *)u_direction_data, (const double *)p_direction_data, out_stride, (double *const *)u_out, (double *)p_out);
        return sfem::codegen::launch_status("navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_mixed_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_mixed_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, direction_stride, (const float *const *)u_direction_data, (const float *)p_direction_data, out_stride, (float *const *)u_out, (float *)p_out);
        return sfem::codegen::launch_status("navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_isoparametric_mesh_mixed_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("navier_stokes_form_2_u_p_tri6_tri3_jacobian_action_i_msoa", -1, (int)scalar_bytes);
}
