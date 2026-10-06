#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../../cuda/body_force_d3_simplex_local.cuh"
#include "../../../../cuda/geometry_kernels.cuh"
#include "../../../../cuda/kernel_diagnostics.cuh"
#include "../../../../reference/cuda/quad_tet_q4.hpp"
#include "../../../../reference/cuda/tet10_q4.hpp"
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

static const KernelDiagnostics body_force_tet10_residual_esoa_diagnostics_data = {
  "body_force_tet10_residual_esoa",
  "TET10",
  3,
  4,
  10,
  16,
  2,
  0,
  6,
  0,
  0,
  0,
  0,
  0,
  0,
  7,
  3,
  6,
  1876,
  2760,
  0,
  3,
  10,
  160,
  4,
  4,
  0,
  0,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_body_force_tet10_residual_esoa_diagnostics(void) {
  return &sfem::codegen::body_force_tet10_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics body_force_tet10_jacobian_action_esoa_diagnostics_data = {
  "body_force_tet10_jacobian_action_esoa",
  "TET10",
  3,
  4,
  10,
  16,
  2,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  3,
  0,
  1876,
  2760,
  0,
  0,
  10,
  160,
  4,
  0,
  0,
  0,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_body_force_tet10_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::body_force_tet10_jacobian_action_esoa_diagnostics_data;
}

extern "C" int cu_body_force_tet10_residual_esoa(
    const int scalar_bytes,
    const int ne,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const real_t g2,
    void *const RSTR output[30],
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::body_force_d3_simplex_residual_block<double, 4, 10, 1>(ne, geometry_stride, (const double *)determinant, sfem::codegen::ref_tet10_q4<double>::shape(), sfem::codegen::quad_tet_q4<double>::q_weight(), density, g0, g1, g2, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::body_force_d3_simplex_residual_block<float, 4, 10, 1>(ne, geometry_stride, (const float *)determinant, sfem::codegen::ref_tet10_q4<float>::shape(), sfem::codegen::quad_tet_q4<float>::q_weight(), density, g0, g1, g2, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("body_force_tet10_residual_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void body_force_tet10_residual_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const s_t density,
    const s_t g0,
    const s_t g1,
    const s_t g2,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int NQ = 4;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int VS = 1;
  const s_t *const affine_shape = sfem::codegen::ref_tet10_q4<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q4<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    s_t boutput[NC * NS][VS];

    for (int stream = 0; stream < 30; ++stream) {
      {
        boutput[stream][0] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[1] = {g_det0 + evb};
    s_t baffine_geometry_data[1][VS];
    const s_t *bageom_streams[1];
    bageom_streams[0] = ageom_stream<s_t, g_t, VS>(
        ne, affine_geometry_sources[0], baffine_geometry_data[0], std::is_same<g_t, s_t>());

    body_force_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[0], affine_shape, affine_q_weight, density, g0, g1, g2, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
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

extern "C" int cu_body_force_tet10_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const real_t g2,
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
        sfem::codegen::body_force_tet10_residual_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, g0, g1, g2, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("body_force_tet10_residual_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::body_force_tet10_residual_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, g0, g1, g2, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("body_force_tet10_residual_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("body_force_tet10_residual_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
__global__ void body_force_tet10_residual_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t density,
    const s_t g0,
    const s_t g1,
    const s_t g2,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 4;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int VS = 1;
  const s_t *const isoparametric_shape = sfem::codegen::ref_tet10_q4<s_t>::shape();
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tet_q4<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    s_t bcoordinates[3 * NS][VS];
    s_t badjugate_data[9][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t boutput[NC * NS][VS];

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

    for (int stream = 0; stream < 30; ++stream) {
      {
        boutput[stream][0] = s_t(0);
      }
    }

    s_t *badjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    for (int q = 0; q < NQ; ++q) {
      {
        const s_t J00 = bcoordinates[0][0] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[3][0] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[6][0] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[9][0] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[12][0] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[15][0] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[18][0] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[21][0] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[24][0] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[27][0] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J01 = bcoordinates[0][0] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[3][0] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[6][0] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[9][0] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[12][0] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[15][0] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[18][0] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[21][0] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[24][0] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[27][0] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J02 = bcoordinates[0][0] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[3][0] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[6][0] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[9][0] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[12][0] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[15][0] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[18][0] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[21][0] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[24][0] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[27][0] * isoparametric_grad_ref_z[q * NS + 9];
        const s_t J10 = bcoordinates[1][0] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[4][0] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[7][0] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[10][0] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[13][0] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[16][0] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[19][0] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[22][0] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[25][0] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[28][0] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J11 = bcoordinates[1][0] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[4][0] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[7][0] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[10][0] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[13][0] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[16][0] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[19][0] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[22][0] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[25][0] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[28][0] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J12 = bcoordinates[1][0] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[4][0] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[7][0] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[10][0] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[13][0] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[16][0] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[19][0] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[22][0] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[25][0] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[28][0] * isoparametric_grad_ref_z[q * NS + 9];
        const s_t J20 = bcoordinates[2][0] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[5][0] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[8][0] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[11][0] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[14][0] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[17][0] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[20][0] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[23][0] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[26][0] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[29][0] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J21 = bcoordinates[2][0] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[5][0] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[8][0] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[11][0] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[14][0] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[17][0] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[20][0] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[23][0] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[26][0] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[29][0] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J22 = bcoordinates[2][0] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[5][0] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[8][0] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[11][0] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[14][0] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[17][0] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[20][0] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[23][0] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[26][0] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[29][0] * isoparametric_grad_ref_z[q * NS + 9];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badjugate_streams, bdeterminant, q * VS);
      }
    }


    body_force_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, isoparametric_shape, isoparametric_q_weight, density, g0, g1, g2, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
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

extern "C" int cu_body_force_tet10_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const real_t g0,
    const real_t g1,
    const real_t g2,
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
        sfem::codegen::body_force_tet10_residual_i_msoa_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, density, g0, g1, g2, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("body_force_tet10_residual_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::body_force_tet10_residual_i_msoa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, density, g0, g1, g2, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("body_force_tet10_residual_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("body_force_tet10_residual_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int cu_body_force_tet10_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    void *const RSTR output,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return cu_body_force_tet10_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], ((const double *)parameters)[1], ((const double *)parameters)[2], ((const double *)parameters)[3], 3, (double *)output + 0, (double *)output + 1, (double *)output + 2, stream);
    }
    case (int)sizeof(float): {
        return cu_body_force_tet10_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], ((const float *)parameters)[1], ((const float *)parameters)[2], ((const float *)parameters)[3], 3, (float *)output + 0, (float *)output + 1, (float *)output + 2, stream);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("body_force_tet10_residual_i_maos", -1, (int)scalar_bytes);
}
