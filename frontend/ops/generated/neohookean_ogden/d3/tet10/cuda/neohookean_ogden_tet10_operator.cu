#include <type_traits>
#include <cuda_runtime.h>
#include "../../cuda/neohookean_ogden_d3_simplex_local.cuh"
#include "../../../../reference/cuda/quad_tet_q4.hpp"
#include "../../../../reference/cuda/tet10_q4.hpp"
#include "../../../../reference/cuda/quad_tet_q11.hpp"
#include "../../../../reference/cuda/tet10_q11.hpp"
#include "../../../../cuda/geometry_kernels.cuh"
#include "../../../../cuda/kernel_diagnostics.cuh"
#include <cstdint>
#include <cstdlib>

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

static const KernelDiagnostics neohookean_ogden_tet10_objective_soa_diagnostics_data = {
  "neohookean_ogden_tet10_objective_soa",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  19,
  17,
  0,
  0,
  10,
  0,
  1,
  0,
  6,
  5,
  66,
  2618,
  5049,
  4,
  12,
  10,
  330,
  11,
  2,
  30,
  0,
  1,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_neohookean_ogden_tet10_objective_soa_diagnostics(void) {
  return &sfem::codegen::neohookean_ogden_tet10_objective_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void neohookean_ogden_tet10_objective_steps_a_msoa_impl(
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
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const s_t *const RSTR hz,
        const int nsteps,
        const s_t *const RSTR steps,
        s_t *const RSTR value
) {
  static constexpr int NC = 3;
  static constexpr int NQ = 4;
  static constexpr int NS = 10;
  const s_t *const affine_grad_ref_x = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q4<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    idx_t ev[VS * NS];
    s_t bu_data[NS * NC][VS];
    s_t bh_data[NS * NC][VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node * VS];
      {
        ev_node[0] = element_shape[0];
      }
    }

    const s_t *const u_components[NC] = {ux, uy, uz};
    const s_t *const h_components[NC] = {hx, hy, hz};
    const s_t *bu_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bu_streams[stream] = bu_data[stream];
    }
    const s_t *bh_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bh_streams[stream] = bh_data[stream];
    }

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        {
          const idx_t node = ev_shape[0];
          bu_data[shape * NC + d][0] = u_components[d][node * u_stride];
          bh_data[shape * NC + d][0] = h_components[d][node * h_stride];
        }
      }
    }
    s_t badj0_data[VS];
    const s_t *const badj0 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj0 + evb, badj0_data, std::is_same<g_t, s_t>());
    s_t badj1_data[VS];
    const s_t *const badj1 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj1 + evb, badj1_data, std::is_same<g_t, s_t>());
    s_t badj2_data[VS];
    const s_t *const badj2 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj2 + evb, badj2_data, std::is_same<g_t, s_t>());
    s_t badj3_data[VS];
    const s_t *const badj3 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj3 + evb, badj3_data, std::is_same<g_t, s_t>());
    s_t badj4_data[VS];
    const s_t *const badj4 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj4 + evb, badj4_data, std::is_same<g_t, s_t>());
    s_t badj5_data[VS];
    const s_t *const badj5 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj5 + evb, badj5_data, std::is_same<g_t, s_t>());
    s_t badj6_data[VS];
    const s_t *const badj6 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj6 + evb, badj6_data, std::is_same<g_t, s_t>());
    s_t badj7_data[VS];
    const s_t *const badj7 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj7 + evb, badj7_data, std::is_same<g_t, s_t>());
    s_t badj8_data[VS];
    const s_t *const badj8 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj8 + evb, badj8_data, std::is_same<g_t, s_t>());
    s_t bdet0_data[VS];
    const s_t *const bdet0 = ageom_stream<s_t, g_t, VS>(
        ne, g_det0 + evb, bdet0_data, std::is_same<g_t, s_t>());

    for (int step = 0; step < nsteps; ++step) {
      {
        value[(ptrdiff_t)step * nelements + evb + 0] = s_t(0);
      }
    }

    neohookean_ogden_d3_simplex_objective_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, lmbda, mu, bu_streams, bh_streams, nsteps, steps, nelements, &value[evb]);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_neohookean_ogden_tet10_objective_steps_a_msoa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet10_objective_steps_a_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
        return sfem::codegen::launch_status("neohookean_ogden_tet10_objective_steps_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet10_objective_steps_a_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
        return sfem::codegen::launch_status("neohookean_ogden_tet10_objective_steps_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_tet10_objective_steps_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void neohookean_ogden_tet10_objective_steps_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const s_t *const RSTR hz,
        const int nsteps,
        const s_t *const RSTR steps,
        s_t *const RSTR value
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const g_t *const RSTR z = points[2];
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    idx_t ev[VS * NS];
    s_t bu_data[NS * NC][VS];
    s_t bh_data[NS * NC][VS];
    s_t bcoordinate_data[NS * ND][VS];
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t badj4[NQ * VS];
    s_t badj5[NQ * VS];
    s_t badj6[NQ * VS];
    s_t badj7[NQ * VS];
    s_t badj8[NQ * VS];
    s_t bdet0[NQ * VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node * VS];
      {
        ev_node[0] = element_shape[0];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y, z};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < ND; ++d) {
        {
          bcoordinate_data[shape * ND + d][0] = coordinate_components[d][ev_shape[0]];
        }
      }
    }

    const s_t *const u_components[NC] = {ux, uy, uz};
    const s_t *const h_components[NC] = {hx, hy, hz};
    const s_t *bu_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bu_streams[stream] = bu_data[stream];
    }
    const s_t *bh_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bh_streams[stream] = bh_data[stream];
    }

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        {
          const idx_t node = ev_shape[0];
          bu_data[shape * NC + d][0] = u_components[d][node * u_stride];
          bh_data[shape * NC + d][0] = h_components[d][node * h_stride];
        }
      }
    }

    for (int q = 0; q < NQ; ++q) {
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
      s_t J00_values[VS];
      s_t J01_values[VS];
      s_t J02_values[VS];
      s_t J10_values[VS];
      s_t J11_values[VS];
      s_t J12_values[VS];
      s_t J20_values[VS];
      s_t J21_values[VS];
      s_t J22_values[VS];
      {
        J00_values[0] = s_t(0);
        J01_values[0] = s_t(0);
        J02_values[0] = s_t(0);
        J10_values[0] = s_t(0);
        J11_values[0] = s_t(0);
        J12_values[0] = s_t(0);
        J20_values[0] = s_t(0);
        J21_values[0] = s_t(0);
        J22_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = isoparametric_grad_ref_x[q * NS + shape];
        const s_t g1 = isoparametric_grad_ref_y[q * NS + shape];
        const s_t g2 = isoparametric_grad_ref_z[q * NS + shape];
        {
          J00_values[0] += bcoordinate_data[3 * shape][0] * g0;
          J01_values[0] += bcoordinate_data[3 * shape][0] * g1;
          J02_values[0] += bcoordinate_data[3 * shape][0] * g2;
          J10_values[0] += bcoordinate_data[3 * shape + 1][0] * g0;
          J11_values[0] += bcoordinate_data[3 * shape + 1][0] * g1;
          J12_values[0] += bcoordinate_data[3 * shape + 1][0] * g2;
          J20_values[0] += bcoordinate_data[3 * shape + 2][0] * g0;
          J21_values[0] += bcoordinate_data[3 * shape + 2][0] * g1;
          J22_values[0] += bcoordinate_data[3 * shape + 2][0] * g2;
        }
      }
      {
        const s_t J00 = J00_values[0];
        const s_t J01 = J01_values[0];
        const s_t J02 = J02_values[0];
        const s_t J10 = J10_values[0];
        const s_t J11 = J11_values[0];
        const s_t J12 = J12_values[0];
        const s_t J20 = J20_values[0];
        const s_t J21 = J21_values[0];
        const s_t J22 = J22_values[0];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, q * VS + 0);
      }
    }

    for (int step = 0; step < nsteps; ++step) {
      {
        value[(ptrdiff_t)step * nelements + evb + 0] = s_t(0);
      }
    }

    neohookean_ogden_d3_simplex_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, lmbda, mu, bu_streams, bh_streams, nsteps, steps, nelements, &value[evb]);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_neohookean_ogden_tet10_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet10_objective_steps_i_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
        return sfem::codegen::launch_status("neohookean_ogden_tet10_objective_steps_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet10_objective_steps_i_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
        return sfem::codegen::launch_status("neohookean_ogden_tet10_objective_steps_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_tet10_objective_steps_i_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_tet10_gradient_soa_diagnostics_data = {
  "neohookean_ogden_tet10_gradient_soa",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  35,
  54,
  1,
  0,
  0,
  0,
  1,
  0,
  6,
  31,
  117,
  5159,
  7590,
  22,
  16,
  10,
  330,
  11,
  2,
  30,
  0,
  30,
  30,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_neohookean_ogden_tet10_gradient_soa_diagnostics(void) {
  return &sfem::codegen::neohookean_ogden_tet10_gradient_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void neohookean_ogden_tet10_gradient_a_msoa_impl(
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
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy,
        s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int NQ = 4;
  static constexpr int NS = 10;
  const s_t *const affine_grad_ref_x = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q4<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    idx_t ev[VS * NS];
    s_t bu_data[NS * NC][VS];
    s_t bout_data[NS * NC][VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node * VS];
      {
        ev_node[0] = element_shape[0];
      }
    }
    const s_t *const u_components[NC] = {ux, uy, uz};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        {
          const idx_t node = ev_shape[0];
          bu_data[shape * NC + d][0] = u_components[d][node * u_stride];
        }
      }
    }
    for (int stream = 0; stream < NS * NC; ++stream) {
      {
        bout_data[stream][0] = s_t(0);
      }
    }

    const s_t *bu_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bu_streams[stream] = bu_data[stream];
    }
    s_t *bout_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bout_streams[stream] = bout_data[stream];
    }
    s_t badj0_data[VS];
    const s_t *const badj0 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj0 + evb, badj0_data, std::is_same<g_t, s_t>());
    s_t badj1_data[VS];
    const s_t *const badj1 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj1 + evb, badj1_data, std::is_same<g_t, s_t>());
    s_t badj2_data[VS];
    const s_t *const badj2 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj2 + evb, badj2_data, std::is_same<g_t, s_t>());
    s_t badj3_data[VS];
    const s_t *const badj3 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj3 + evb, badj3_data, std::is_same<g_t, s_t>());
    s_t badj4_data[VS];
    const s_t *const badj4 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj4 + evb, badj4_data, std::is_same<g_t, s_t>());
    s_t badj5_data[VS];
    const s_t *const badj5 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj5 + evb, badj5_data, std::is_same<g_t, s_t>());
    s_t badj6_data[VS];
    const s_t *const badj6 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj6 + evb, badj6_data, std::is_same<g_t, s_t>());
    s_t badj7_data[VS];
    const s_t *const badj7 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj7 + evb, badj7_data, std::is_same<g_t, s_t>());
    s_t badj8_data[VS];
    const s_t *const badj8 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj8 + evb, badj8_data, std::is_same<g_t, s_t>());
    s_t bdet0_data[VS];
    const s_t *const bdet0 = ageom_stream<s_t, g_t, VS>(
        ne, g_det0 + evb, bdet0_data, std::is_same<g_t, s_t>());

    neohookean_ogden_d3_simplex_gradient_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, lmbda, mu, bu_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy, outz};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        atomicAdd(&(out_components[d][ev_shape[0] * out_stride]), bout_data[shape * NC + d][0]);
      }
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_neohookean_ogden_tet10_gradient_a_msoa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet10_gradient_a_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        return sfem::codegen::launch_status("neohookean_ogden_tet10_gradient_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet10_gradient_a_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        return sfem::codegen::launch_status("neohookean_ogden_tet10_gradient_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_tet10_gradient_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void neohookean_ogden_tet10_gradient_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy,
        s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const g_t *const RSTR z = points[2];
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    idx_t ev[VS * NS];
    s_t bu_data[NS * NC][VS];
    s_t bout_data[NS * NC][VS];
    s_t bcoordinate_data[NS * ND][VS];
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t badj4[NQ * VS];
    s_t badj5[NQ * VS];
    s_t badj6[NQ * VS];
    s_t badj7[NQ * VS];
    s_t badj8[NQ * VS];
    s_t bdet0[NQ * VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node * VS];
      {
        ev_node[0] = element_shape[0];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y, z};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < ND; ++d) {
        {
          bcoordinate_data[shape * ND + d][0] = coordinate_components[d][ev_shape[0]];
        }
      }
    }
    const s_t *const u_components[NC] = {ux, uy, uz};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        {
          const idx_t node = ev_shape[0];
          bu_data[shape * NC + d][0] = u_components[d][node * u_stride];
        }
      }
    }
    for (int stream = 0; stream < NS * NC; ++stream) {
      {
        bout_data[stream][0] = s_t(0);
      }
    }

    const s_t *bu_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bu_streams[stream] = bu_data[stream];
    }
    s_t *bout_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bout_streams[stream] = bout_data[stream];
    }

    for (int q = 0; q < NQ; ++q) {
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
      s_t J00_values[VS];
      s_t J01_values[VS];
      s_t J02_values[VS];
      s_t J10_values[VS];
      s_t J11_values[VS];
      s_t J12_values[VS];
      s_t J20_values[VS];
      s_t J21_values[VS];
      s_t J22_values[VS];
      {
        J00_values[0] = s_t(0);
        J01_values[0] = s_t(0);
        J02_values[0] = s_t(0);
        J10_values[0] = s_t(0);
        J11_values[0] = s_t(0);
        J12_values[0] = s_t(0);
        J20_values[0] = s_t(0);
        J21_values[0] = s_t(0);
        J22_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = isoparametric_grad_ref_x[q * NS + shape];
        const s_t g1 = isoparametric_grad_ref_y[q * NS + shape];
        const s_t g2 = isoparametric_grad_ref_z[q * NS + shape];
        {
          J00_values[0] += bcoordinate_data[3 * shape][0] * g0;
          J01_values[0] += bcoordinate_data[3 * shape][0] * g1;
          J02_values[0] += bcoordinate_data[3 * shape][0] * g2;
          J10_values[0] += bcoordinate_data[3 * shape + 1][0] * g0;
          J11_values[0] += bcoordinate_data[3 * shape + 1][0] * g1;
          J12_values[0] += bcoordinate_data[3 * shape + 1][0] * g2;
          J20_values[0] += bcoordinate_data[3 * shape + 2][0] * g0;
          J21_values[0] += bcoordinate_data[3 * shape + 2][0] * g1;
          J22_values[0] += bcoordinate_data[3 * shape + 2][0] * g2;
        }
      }
      {
        const s_t J00 = J00_values[0];
        const s_t J01 = J01_values[0];
        const s_t J02 = J02_values[0];
        const s_t J10 = J10_values[0];
        const s_t J11 = J11_values[0];
        const s_t J12 = J12_values[0];
        const s_t J20 = J20_values[0];
        const s_t J21 = J21_values[0];
        const s_t J22 = J22_values[0];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, q * VS + 0);
      }
    }

    neohookean_ogden_d3_simplex_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, lmbda, mu, bu_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy, outz};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        atomicAdd(&(out_components[d][ev_shape[0] * out_stride]), bout_data[shape * NC + d][0]);
      }
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_neohookean_ogden_tet10_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet10_gradient_i_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        return sfem::codegen::launch_status("neohookean_ogden_tet10_gradient_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet10_gradient_i_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        return sfem::codegen::launch_status("neohookean_ogden_tet10_gradient_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_tet10_gradient_i_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_tet10_apply_soa_diagnostics_data = {
  "neohookean_ogden_tet10_apply_soa",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  190,
  288,
  1,
  0,
  10,
  0,
  1,
  0,
  6,
  129,
  516,
  5159,
  7590,
  120,
  69,
  10,
  330,
  11,
  2,
  30,
  30,
  30,
  30,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_neohookean_ogden_tet10_apply_soa_diagnostics(void) {
  return &sfem::codegen::neohookean_ogden_tet10_apply_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void neohookean_ogden_tet10_apply_a_msoa_impl(
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
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const s_t *const RSTR hz,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy,
        s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int NQ = 4;
  static constexpr int NS = 10;
  const s_t *const affine_grad_ref_x = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::ref_tet10_q4<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q4<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    idx_t ev[VS * NS];
    s_t bu_data[NS * NC][VS];
    s_t bh_data[NS * NC][VS];
    s_t bout_data[NS * NC][VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node * VS];
      {
        ev_node[0] = element_shape[0];
      }
    }
    const s_t *const u_components[NC] = {ux, uy, uz};
    const s_t *const h_components[NC] = {hx, hy, hz};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        {
          const idx_t node = ev_shape[0];
          bu_data[shape * NC + d][0] = u_components[d][node * u_stride];
          bh_data[shape * NC + d][0] = h_components[d][node * h_stride];
        }
      }
    }
    for (int stream = 0; stream < NS * NC; ++stream) {
      {
        bout_data[stream][0] = s_t(0);
      }
    }

    const s_t *bu_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bu_streams[stream] = bu_data[stream];
    }
    const s_t *bh_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bh_streams[stream] = bh_data[stream];
    }
    s_t *bout_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bout_streams[stream] = bout_data[stream];
    }
    s_t badj0_data[VS];
    const s_t *const badj0 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj0 + evb, badj0_data, std::is_same<g_t, s_t>());
    s_t badj1_data[VS];
    const s_t *const badj1 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj1 + evb, badj1_data, std::is_same<g_t, s_t>());
    s_t badj2_data[VS];
    const s_t *const badj2 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj2 + evb, badj2_data, std::is_same<g_t, s_t>());
    s_t badj3_data[VS];
    const s_t *const badj3 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj3 + evb, badj3_data, std::is_same<g_t, s_t>());
    s_t badj4_data[VS];
    const s_t *const badj4 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj4 + evb, badj4_data, std::is_same<g_t, s_t>());
    s_t badj5_data[VS];
    const s_t *const badj5 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj5 + evb, badj5_data, std::is_same<g_t, s_t>());
    s_t badj6_data[VS];
    const s_t *const badj6 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj6 + evb, badj6_data, std::is_same<g_t, s_t>());
    s_t badj7_data[VS];
    const s_t *const badj7 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj7 + evb, badj7_data, std::is_same<g_t, s_t>());
    s_t badj8_data[VS];
    const s_t *const badj8 = ageom_stream<s_t, g_t, VS>(
        ne, g_adj8 + evb, badj8_data, std::is_same<g_t, s_t>());
    s_t bdet0_data[VS];
    const s_t *const bdet0 = ageom_stream<s_t, g_t, VS>(
        ne, g_det0 + evb, bdet0_data, std::is_same<g_t, s_t>());

    neohookean_ogden_d3_simplex_apply_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, lmbda, mu, bu_streams, bh_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy, outz};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        atomicAdd(&(out_components[d][ev_shape[0] * out_stride]), bout_data[shape * NC + d][0]);
      }
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_neohookean_ogden_tet10_apply_a_msoa(
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
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet10_apply_a_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        return sfem::codegen::launch_status("neohookean_ogden_tet10_apply_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet10_apply_a_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        return sfem::codegen::launch_status("neohookean_ogden_tet10_apply_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_tet10_apply_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void neohookean_ogden_tet10_apply_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const s_t *const RSTR hz,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy,
        s_t *const RSTR outz
) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const g_t *const RSTR z = points[2];
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    idx_t ev[VS * NS];
    s_t bu_data[NS * NC][VS];
    s_t bh_data[NS * NC][VS];
    s_t bout_data[NS * NC][VS];
    s_t bcoordinate_data[NS * ND][VS];
    s_t badj0[NQ * VS];
    s_t badj1[NQ * VS];
    s_t badj2[NQ * VS];
    s_t badj3[NQ * VS];
    s_t badj4[NQ * VS];
    s_t badj5[NQ * VS];
    s_t badj6[NQ * VS];
    s_t badj7[NQ * VS];
    s_t badj8[NQ * VS];
    s_t bdet0[NQ * VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node * VS];
      {
        ev_node[0] = element_shape[0];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y, z};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < ND; ++d) {
        {
          bcoordinate_data[shape * ND + d][0] = coordinate_components[d][ev_shape[0]];
        }
      }
    }
    const s_t *const u_components[NC] = {ux, uy, uz};
    const s_t *const h_components[NC] = {hx, hy, hz};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        {
          const idx_t node = ev_shape[0];
          bu_data[shape * NC + d][0] = u_components[d][node * u_stride];
          bh_data[shape * NC + d][0] = h_components[d][node * h_stride];
        }
      }
    }
    for (int stream = 0; stream < NS * NC; ++stream) {
      {
        bout_data[stream][0] = s_t(0);
      }
    }

    const s_t *bu_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bu_streams[stream] = bu_data[stream];
    }
    const s_t *bh_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bh_streams[stream] = bh_data[stream];
    }
    s_t *bout_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bout_streams[stream] = bout_data[stream];
    }

    for (int q = 0; q < NQ; ++q) {
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
      s_t J00_values[VS];
      s_t J01_values[VS];
      s_t J02_values[VS];
      s_t J10_values[VS];
      s_t J11_values[VS];
      s_t J12_values[VS];
      s_t J20_values[VS];
      s_t J21_values[VS];
      s_t J22_values[VS];
      {
        J00_values[0] = s_t(0);
        J01_values[0] = s_t(0);
        J02_values[0] = s_t(0);
        J10_values[0] = s_t(0);
        J11_values[0] = s_t(0);
        J12_values[0] = s_t(0);
        J20_values[0] = s_t(0);
        J21_values[0] = s_t(0);
        J22_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = isoparametric_grad_ref_x[q * NS + shape];
        const s_t g1 = isoparametric_grad_ref_y[q * NS + shape];
        const s_t g2 = isoparametric_grad_ref_z[q * NS + shape];
        {
          J00_values[0] += bcoordinate_data[3 * shape][0] * g0;
          J01_values[0] += bcoordinate_data[3 * shape][0] * g1;
          J02_values[0] += bcoordinate_data[3 * shape][0] * g2;
          J10_values[0] += bcoordinate_data[3 * shape + 1][0] * g0;
          J11_values[0] += bcoordinate_data[3 * shape + 1][0] * g1;
          J12_values[0] += bcoordinate_data[3 * shape + 1][0] * g2;
          J20_values[0] += bcoordinate_data[3 * shape + 2][0] * g0;
          J21_values[0] += bcoordinate_data[3 * shape + 2][0] * g1;
          J22_values[0] += bcoordinate_data[3 * shape + 2][0] * g2;
        }
      }
      {
        const s_t J00 = J00_values[0];
        const s_t J01 = J01_values[0];
        const s_t J02 = J02_values[0];
        const s_t J10 = J10_values[0];
        const s_t J11 = J11_values[0];
        const s_t J12 = J12_values[0];
        const s_t J20 = J20_values[0];
        const s_t J21 = J21_values[0];
        const s_t J22 = J22_values[0];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badj_streams, bdet0, q * VS + 0);
      }
    }

    neohookean_ogden_d3_simplex_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, lmbda, mu, bu_streams, bh_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy, outz};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        atomicAdd(&(out_components[d][ev_shape[0] * out_stride]), bout_data[shape * NC + d][0]);
      }
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_neohookean_ogden_tet10_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet10_apply_i_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        return sfem::codegen::launch_status("neohookean_ogden_tet10_apply_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet10_apply_i_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        return sfem::codegen::launch_status("neohookean_ogden_tet10_apply_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_tet10_apply_i_msoa", -1, (int)scalar_bytes);
}
