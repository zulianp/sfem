#include <type_traits>
#include <cuda_runtime.h>
#include "../../cuda/linear_elasticity_d3_simplex_local.cuh"
#include "../../../../reference/cuda/quad_tet_q1.hpp"
#include "../../../../reference/cuda/tet4_q1.hpp"
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

static const KernelDiagnostics linear_elasticity_tet4_objective_soa_diagnostics_data = {
  "linear_elasticity_tet4_objective_soa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  11,
  6,
  0,
  0,
  7,
  0,
  0,
  0,
  6,
  1,
  24,
  130,
  243,
  0,
  11,
  10,
  12,
  1,
  2,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_linear_elasticity_tet4_objective_soa_diagnostics(void) {
  return &sfem::codegen::linear_elasticity_tet4_objective_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void linear_elasticity_tet4_objective_steps_a_msoa_impl(
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
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q1<s_t>::q_weight();

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

    linear_elasticity_d3_simplex_tet4_objective_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bh_streams, nsteps, steps, nelements, &value[evb]);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_linear_elasticity_tet4_objective_steps_a_msoa(
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
        sfem::codegen::linear_elasticity_tet4_objective_steps_a_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
        return sfem::codegen::launch_status("linear_elasticity_tet4_objective_steps_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::linear_elasticity_tet4_objective_steps_a_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
        return sfem::codegen::launch_status("linear_elasticity_tet4_objective_steps_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("linear_elasticity_tet4_objective_steps_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_elasticity_tet4_gradient_soa_diagnostics_data = {
  "linear_elasticity_tet4_gradient_soa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  8,
  8,
  0,
  0,
  0,
  0,
  0,
  0,
  6,
  14,
  16,
  253,
  366,
  5,
  8,
  10,
  12,
  1,
  2,
  12,
  0,
  12,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_linear_elasticity_tet4_gradient_soa_diagnostics(void) {
  return &sfem::codegen::linear_elasticity_tet4_gradient_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void linear_elasticity_tet4_gradient_a_msoa_impl(
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
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q1<s_t>::q_weight();

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

    linear_elasticity_d3_simplex_tet4_gradient_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bu_streams, bout_streams);

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

extern "C" int cu_linear_elasticity_tet4_gradient_a_msoa(
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
        sfem::codegen::linear_elasticity_tet4_gradient_a_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        return sfem::codegen::launch_status("linear_elasticity_tet4_gradient_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::linear_elasticity_tet4_gradient_a_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        return sfem::codegen::launch_status("linear_elasticity_tet4_gradient_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("linear_elasticity_tet4_gradient_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void linear_elasticity_tet4_gradient_a_msoa_aos_unit_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_adj_aos,
        const g_t *const RSTR g_det0,
        const s_t mu,
        const s_t lmbda,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const s_t *const RSTR uz,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy,
        s_t *const RSTR outz
) {

  for (ptrdiff_t element = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; element < nelements; element += (ptrdiff_t)blockDim.x * gridDim.x) {
    const idx_t ev0 = elements[0][element];
    const idx_t ev1 = elements[1][element];
    const idx_t ev2 = elements[2][element];
    const idx_t ev3 = elements[3][element];

    const s_t ux0 = ux[ev0 * u_stride];
    const s_t ux1 = ux[ev1 * u_stride];
    const s_t ux2 = ux[ev2 * u_stride];
    const s_t ux3 = ux[ev3 * u_stride];
    const s_t uy0 = uy[ev0 * u_stride];
    const s_t uy1 = uy[ev1 * u_stride];
    const s_t uy2 = uy[ev2 * u_stride];
    const s_t uy3 = uy[ev3 * u_stride];
    const s_t uz0 = uz[ev0 * u_stride];
    const s_t uz1 = uz[ev1 * u_stride];
    const s_t uz2 = uz[ev2 * u_stride];
    const s_t uz3 = uz[ev3 * u_stride];

    const g_t *const RSTR adjugate = g_adj_aos + element * 9;
    const s_t a0 = s_t(adjugate[0]);
    const s_t a1 = s_t(adjugate[1]);
    const s_t a2 = s_t(adjugate[2]);
    const s_t a3 = s_t(adjugate[3]);
    const s_t a4 = s_t(adjugate[4]);
    const s_t a5 = s_t(adjugate[5]);
    const s_t a6 = s_t(adjugate[6]);
    const s_t a7 = s_t(adjugate[7]);
    const s_t a8 = s_t(adjugate[8]);
    const s_t inv_det = s_t(1) / s_t(g_det0[element]);

    const s_t x1 = ux0 - ux1;
    const s_t x2 = ux0 - ux2;
    const s_t x3 = ux0 - ux3;
    const s_t x4 = uy0 - uy1;
    const s_t x5 = uy0 - uy2;
    const s_t x6 = uy0 - uy3;
    const s_t x7 = uz0 - uz1;
    const s_t x8 = uz0 - uz2;
    const s_t x9 = uz0 - uz3;

    s_t p0 = inv_det * (-a0 * x1 - a3 * x2 - a6 * x3);
    s_t p1 = inv_det * (-a1 * x1 - a4 * x2 - a7 * x3);
    s_t p2 = inv_det * (-a2 * x1 - a5 * x2 - a8 * x3);
    s_t p3 = inv_det * (-a0 * x4 - a3 * x5 - a6 * x6);
    s_t p4 = inv_det * (-a1 * x4 - a4 * x5 - a7 * x6);
    s_t p5 = inv_det * (-a2 * x4 - a5 * x5 - a8 * x6);
    s_t p6 = inv_det * (-a0 * x7 - a3 * x8 - a6 * x9);
    s_t p7 = inv_det * (-a1 * x7 - a4 * x8 - a7 * x9);
    s_t p8 = inv_det * (-a2 * x7 - a5 * x8 - a8 * x9);

    const s_t m0 = (s_t(1) / s_t(6)) * mu;
    const s_t m1 = m0 * (p1 + p3);
    const s_t m2 = m0 * (p2 + p6);
    const s_t m3 = s_t(2) * mu;
    const s_t m4 = lmbda * (p0 + p4 + p8);
    const s_t m5 = (s_t(1) / s_t(6)) * p0 * m3 + (s_t(1) / s_t(6)) * m4;
    const s_t m6 = m0 * (p5 + p7);
    const s_t m7 = (s_t(1) / s_t(6)) * p4 * m3 + (s_t(1) / s_t(6)) * m4;
    const s_t m8 = (s_t(1) / s_t(6)) * p8 * m3 + (s_t(1) / s_t(6)) * m4;

    const s_t q0 = a0 * m5 + a1 * m1 + a2 * m2;
    const s_t q1 = a3 * m5 + a4 * m1 + a5 * m2;
    const s_t q2 = a6 * m5 + a7 * m1 + a8 * m2;
    const s_t q3 = a0 * m1 + a1 * m7 + a2 * m6;
    const s_t q4 = a3 * m1 + a4 * m7 + a5 * m6;
    const s_t q5 = a6 * m1 + a7 * m7 + a8 * m6;
    const s_t q6 = a0 * m2 + a1 * m6 + a2 * m8;
    const s_t q7 = a3 * m2 + a4 * m6 + a5 * m8;
    const s_t q8 = a6 * m2 + a7 * m6 + a8 * m8;

    atomicAdd(&(outx[ev0 * out_stride]), -q0 - q1 - q2);
    atomicAdd(&(outx[ev1 * out_stride]), q0);
    atomicAdd(&(outx[ev2 * out_stride]), q1);
    atomicAdd(&(outx[ev3 * out_stride]), q2);
    atomicAdd(&(outy[ev0 * out_stride]), -q3 - q4 - q5);
    atomicAdd(&(outy[ev1 * out_stride]), q3);
    atomicAdd(&(outy[ev2 * out_stride]), q4);
    atomicAdd(&(outy[ev3 * out_stride]), q5);
    atomicAdd(&(outz[ev0 * out_stride]), -q6 - q7 - q8);
    atomicAdd(&(outz[ev1 * out_stride]), q6);
    atomicAdd(&(outz[ev2 * out_stride]), q7);
    atomicAdd(&(outz[ev3 * out_stride]), q8);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_linear_elasticity_tet4_gradient_a_msoa_aos_unit(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj_aos,
        const geom_t *const RSTR g_det0,
        const real_t mu,
        const real_t lmbda,
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
        sfem::codegen::linear_elasticity_tet4_gradient_a_msoa_aos_unit_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj_aos, g_det0, mu, lmbda, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        return sfem::codegen::launch_status("linear_elasticity_tet4_gradient_a_msoa_aos_unit_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::linear_elasticity_tet4_gradient_a_msoa_aos_unit_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj_aos, g_det0, mu, lmbda, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        return sfem::codegen::launch_status("linear_elasticity_tet4_gradient_a_msoa_aos_unit_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("linear_elasticity_tet4_gradient_a_msoa_aos_unit", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_elasticity_tet4_apply_soa_diagnostics_data = {
  "linear_elasticity_tet4_apply_soa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  8,
  8,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  14,
  16,
  253,
  366,
  5,
  8,
  10,
  12,
  1,
  2,
  0,
  12,
  12,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_linear_elasticity_tet4_apply_soa_diagnostics(void) {
  return &sfem::codegen::linear_elasticity_tet4_apply_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void linear_elasticity_tet4_apply_a_msoa_impl(
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
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q1<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    idx_t ev[VS * NS];
    s_t bh_data[NS * NC][VS];
    s_t bout_data[NS * NC][VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node * VS];
      {
        ev_node[0] = element_shape[0];
      }
    }
    const s_t *const h_components[NC] = {hx, hy, hz};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        {
          const idx_t node = ev_shape[0];
          bh_data[shape * NC + d][0] = h_components[d][node * h_stride];
        }
      }
    }
    for (int stream = 0; stream < NS * NC; ++stream) {
      {
        bout_data[stream][0] = s_t(0);
      }
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

    linear_elasticity_d3_simplex_tet4_apply_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_q_weight, lmbda, mu, bh_streams, bout_streams);

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

extern "C" int cu_linear_elasticity_tet4_apply_a_msoa(
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
        sfem::codegen::linear_elasticity_tet4_apply_a_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        return sfem::codegen::launch_status("linear_elasticity_tet4_apply_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::linear_elasticity_tet4_apply_a_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        return sfem::codegen::launch_status("linear_elasticity_tet4_apply_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("linear_elasticity_tet4_apply_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void linear_elasticity_tet4_apply_a_msoa_aos_unit_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_adj_aos,
        const g_t *const RSTR g_det0,
        const s_t mu,
        const s_t lmbda,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const s_t *const RSTR hz,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy,
        s_t *const RSTR outz
) {

  for (ptrdiff_t element = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; element < nelements; element += (ptrdiff_t)blockDim.x * gridDim.x) {
    const idx_t ev0 = elements[0][element];
    const idx_t ev1 = elements[1][element];
    const idx_t ev2 = elements[2][element];
    const idx_t ev3 = elements[3][element];

    const s_t ux0 = hx[ev0 * h_stride];
    const s_t ux1 = hx[ev1 * h_stride];
    const s_t ux2 = hx[ev2 * h_stride];
    const s_t ux3 = hx[ev3 * h_stride];
    const s_t uy0 = hy[ev0 * h_stride];
    const s_t uy1 = hy[ev1 * h_stride];
    const s_t uy2 = hy[ev2 * h_stride];
    const s_t uy3 = hy[ev3 * h_stride];
    const s_t uz0 = hz[ev0 * h_stride];
    const s_t uz1 = hz[ev1 * h_stride];
    const s_t uz2 = hz[ev2 * h_stride];
    const s_t uz3 = hz[ev3 * h_stride];

    const g_t *const RSTR adjugate = g_adj_aos + element * 9;
    const s_t a0 = s_t(adjugate[0]);
    const s_t a1 = s_t(adjugate[1]);
    const s_t a2 = s_t(adjugate[2]);
    const s_t a3 = s_t(adjugate[3]);
    const s_t a4 = s_t(adjugate[4]);
    const s_t a5 = s_t(adjugate[5]);
    const s_t a6 = s_t(adjugate[6]);
    const s_t a7 = s_t(adjugate[7]);
    const s_t a8 = s_t(adjugate[8]);
    const s_t inv_det = s_t(1) / s_t(g_det0[element]);

    const s_t x1 = ux0 - ux1;
    const s_t x2 = ux0 - ux2;
    const s_t x3 = ux0 - ux3;
    const s_t x4 = uy0 - uy1;
    const s_t x5 = uy0 - uy2;
    const s_t x6 = uy0 - uy3;
    const s_t x7 = uz0 - uz1;
    const s_t x8 = uz0 - uz2;
    const s_t x9 = uz0 - uz3;

    s_t p0 = inv_det * (-a0 * x1 - a3 * x2 - a6 * x3);
    s_t p1 = inv_det * (-a1 * x1 - a4 * x2 - a7 * x3);
    s_t p2 = inv_det * (-a2 * x1 - a5 * x2 - a8 * x3);
    s_t p3 = inv_det * (-a0 * x4 - a3 * x5 - a6 * x6);
    s_t p4 = inv_det * (-a1 * x4 - a4 * x5 - a7 * x6);
    s_t p5 = inv_det * (-a2 * x4 - a5 * x5 - a8 * x6);
    s_t p6 = inv_det * (-a0 * x7 - a3 * x8 - a6 * x9);
    s_t p7 = inv_det * (-a1 * x7 - a4 * x8 - a7 * x9);
    s_t p8 = inv_det * (-a2 * x7 - a5 * x8 - a8 * x9);

    const s_t m0 = (s_t(1) / s_t(6)) * mu;
    const s_t m1 = m0 * (p1 + p3);
    const s_t m2 = m0 * (p2 + p6);
    const s_t m3 = s_t(2) * mu;
    const s_t m4 = lmbda * (p0 + p4 + p8);
    const s_t m5 = (s_t(1) / s_t(6)) * p0 * m3 + (s_t(1) / s_t(6)) * m4;
    const s_t m6 = m0 * (p5 + p7);
    const s_t m7 = (s_t(1) / s_t(6)) * p4 * m3 + (s_t(1) / s_t(6)) * m4;
    const s_t m8 = (s_t(1) / s_t(6)) * p8 * m3 + (s_t(1) / s_t(6)) * m4;

    const s_t q0 = a0 * m5 + a1 * m1 + a2 * m2;
    const s_t q1 = a3 * m5 + a4 * m1 + a5 * m2;
    const s_t q2 = a6 * m5 + a7 * m1 + a8 * m2;
    const s_t q3 = a0 * m1 + a1 * m7 + a2 * m6;
    const s_t q4 = a3 * m1 + a4 * m7 + a5 * m6;
    const s_t q5 = a6 * m1 + a7 * m7 + a8 * m6;
    const s_t q6 = a0 * m2 + a1 * m6 + a2 * m8;
    const s_t q7 = a3 * m2 + a4 * m6 + a5 * m8;
    const s_t q8 = a6 * m2 + a7 * m6 + a8 * m8;

    atomicAdd(&(outx[ev0 * out_stride]), -q0 - q1 - q2);
    atomicAdd(&(outx[ev1 * out_stride]), q0);
    atomicAdd(&(outx[ev2 * out_stride]), q1);
    atomicAdd(&(outx[ev3 * out_stride]), q2);
    atomicAdd(&(outy[ev0 * out_stride]), -q3 - q4 - q5);
    atomicAdd(&(outy[ev1 * out_stride]), q3);
    atomicAdd(&(outy[ev2 * out_stride]), q4);
    atomicAdd(&(outy[ev3 * out_stride]), q5);
    atomicAdd(&(outz[ev0 * out_stride]), -q6 - q7 - q8);
    atomicAdd(&(outz[ev1 * out_stride]), q6);
    atomicAdd(&(outz[ev2 * out_stride]), q7);
    atomicAdd(&(outz[ev3 * out_stride]), q8);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_linear_elasticity_tet4_apply_a_msoa_aos_unit(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj_aos,
        const geom_t *const RSTR g_det0,
        const real_t mu,
        const real_t lmbda,
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
        sfem::codegen::linear_elasticity_tet4_apply_a_msoa_aos_unit_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj_aos, g_det0, mu, lmbda, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        return sfem::codegen::launch_status("linear_elasticity_tet4_apply_a_msoa_aos_unit_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::linear_elasticity_tet4_apply_a_msoa_aos_unit_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj_aos, g_det0, mu, lmbda, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        return sfem::codegen::launch_status("linear_elasticity_tet4_apply_a_msoa_aos_unit_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("linear_elasticity_tet4_apply_a_msoa_aos_unit", -1, (int)scalar_bytes);
}
