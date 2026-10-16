#include <type_traits>
#include <cuda_runtime.h>
#include "neohookean_ogden_d3_simplex_local.cuh"
#include "neohookean_ogden_d3_simplex_hessian.cuh"
#include "geometry_kernels.cuh"
#include "kernel_diagnostics.cuh"
#include <cstdint>
#include <cstdlib>

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

static const KernelDiagnostics neohookean_ogden_tet4_objective_soa_diagnostics_data = {
  "neohookean_ogden_tet4_objective_soa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
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
  130,
  243,
  4,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_neohookean_ogden_tet4_objective_soa_diagnostics(void) {
  return &sfem::codegen::neohookean_ogden_tet4_objective_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void neohookean_ogden_tet4_objective_steps_a_msoa_impl(
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
  static constexpr int NS = 4;

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[NS];
    s_t bu_data[NS * NC];
    s_t bh_data[NS * NC];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node];
      {
        ev_node[0] = element_shape[0];
      }
    }

    const s_t *const u_components[NC] = {ux, uy, uz};
    const s_t *const h_components[NC] = {hx, hy, hz};
    const s_t *bu_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bu_streams[stream] = &bu_data[stream];
    }
    const s_t *bh_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bh_streams[stream] = &bh_data[stream];
    }

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape];
      for (int d = 0; d < NC; ++d) {
        {
          const idx_t node = ev_shape[0];
          bu_data[shape * NC + d] = u_components[d][node * u_stride];
          bh_data[shape * NC + d] = h_components[d][node * h_stride];
        }
      }
    }
    s_t badj0_data;
    const s_t *const badj0 = ageom_stream<s_t, g_t>(
        g_adj0 + evb, &badj0_data, std::is_same<g_t, s_t>());
    s_t badj1_data;
    const s_t *const badj1 = ageom_stream<s_t, g_t>(
        g_adj1 + evb, &badj1_data, std::is_same<g_t, s_t>());
    s_t badj2_data;
    const s_t *const badj2 = ageom_stream<s_t, g_t>(
        g_adj2 + evb, &badj2_data, std::is_same<g_t, s_t>());
    s_t badj3_data;
    const s_t *const badj3 = ageom_stream<s_t, g_t>(
        g_adj3 + evb, &badj3_data, std::is_same<g_t, s_t>());
    s_t badj4_data;
    const s_t *const badj4 = ageom_stream<s_t, g_t>(
        g_adj4 + evb, &badj4_data, std::is_same<g_t, s_t>());
    s_t badj5_data;
    const s_t *const badj5 = ageom_stream<s_t, g_t>(
        g_adj5 + evb, &badj5_data, std::is_same<g_t, s_t>());
    s_t badj6_data;
    const s_t *const badj6 = ageom_stream<s_t, g_t>(
        g_adj6 + evb, &badj6_data, std::is_same<g_t, s_t>());
    s_t badj7_data;
    const s_t *const badj7 = ageom_stream<s_t, g_t>(
        g_adj7 + evb, &badj7_data, std::is_same<g_t, s_t>());
    s_t badj8_data;
    const s_t *const badj8 = ageom_stream<s_t, g_t>(
        g_adj8 + evb, &badj8_data, std::is_same<g_t, s_t>());
    s_t bdet0_data;
    const s_t *const bdet0 = ageom_stream<s_t, g_t>(
        g_det0 + evb, &bdet0_data, std::is_same<g_t, s_t>());

    for (int step = 0; step < nsteps; ++step) {
      {
        value[(ptrdiff_t)step * nelements + evb + 0] = s_t(0);
      }
    }

    neohookean_ogden_d3_simplex_tet4_objective_block<s_t, NS>(badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bh_streams, nsteps, steps, nelements, &value[evb]);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_neohookean_ogden_tet4_objective_steps_a_msoa(
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
        sfem::codegen::neohookean_ogden_tet4_objective_steps_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
        return sfem::codegen::launch_status("neohookean_ogden_tet4_objective_steps_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet4_objective_steps_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
        return sfem::codegen::launch_status("neohookean_ogden_tet4_objective_steps_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_tet4_objective_steps_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_tet4_gradient_soa_diagnostics_data = {
  "neohookean_ogden_tet4_gradient_soa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
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
  253,
  366,
  22,
  16,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_neohookean_ogden_tet4_gradient_soa_diagnostics(void) {
  return &sfem::codegen::neohookean_ogden_tet4_gradient_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void neohookean_ogden_tet4_gradient_a_msoa_impl(
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
  static constexpr int NS = 4;

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[NS];
    s_t bu_data[NS * NC];
    s_t bout_data[NS * NC];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node];
      {
        ev_node[0] = element_shape[0];
      }
    }
    const s_t *const u_components[NC] = {ux, uy, uz};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape];
      for (int d = 0; d < NC; ++d) {
        {
          const idx_t node = ev_shape[0];
          bu_data[shape * NC + d] = u_components[d][node * u_stride];
        }
      }
    }
    for (int stream = 0; stream < NS * NC; ++stream) {
      {
        bout_data[stream] = s_t(0);
      }
    }

    const s_t *bu_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bu_streams[stream] = &bu_data[stream];
    }
    s_t *bout_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bout_streams[stream] = &bout_data[stream];
    }
    s_t badj0_data;
    const s_t *const badj0 = ageom_stream<s_t, g_t>(
        g_adj0 + evb, &badj0_data, std::is_same<g_t, s_t>());
    s_t badj1_data;
    const s_t *const badj1 = ageom_stream<s_t, g_t>(
        g_adj1 + evb, &badj1_data, std::is_same<g_t, s_t>());
    s_t badj2_data;
    const s_t *const badj2 = ageom_stream<s_t, g_t>(
        g_adj2 + evb, &badj2_data, std::is_same<g_t, s_t>());
    s_t badj3_data;
    const s_t *const badj3 = ageom_stream<s_t, g_t>(
        g_adj3 + evb, &badj3_data, std::is_same<g_t, s_t>());
    s_t badj4_data;
    const s_t *const badj4 = ageom_stream<s_t, g_t>(
        g_adj4 + evb, &badj4_data, std::is_same<g_t, s_t>());
    s_t badj5_data;
    const s_t *const badj5 = ageom_stream<s_t, g_t>(
        g_adj5 + evb, &badj5_data, std::is_same<g_t, s_t>());
    s_t badj6_data;
    const s_t *const badj6 = ageom_stream<s_t, g_t>(
        g_adj6 + evb, &badj6_data, std::is_same<g_t, s_t>());
    s_t badj7_data;
    const s_t *const badj7 = ageom_stream<s_t, g_t>(
        g_adj7 + evb, &badj7_data, std::is_same<g_t, s_t>());
    s_t badj8_data;
    const s_t *const badj8 = ageom_stream<s_t, g_t>(
        g_adj8 + evb, &badj8_data, std::is_same<g_t, s_t>());
    s_t bdet0_data;
    const s_t *const bdet0 = ageom_stream<s_t, g_t>(
        g_det0 + evb, &bdet0_data, std::is_same<g_t, s_t>());

    neohookean_ogden_d3_simplex_tet4_gradient_block<s_t, NS>(badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy, outz};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape];
      for (int d = 0; d < NC; ++d) {
        atomicAdd(&(out_components[d][ev_shape[0] * out_stride]), bout_data[shape * NC + d]);
      }
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_neohookean_ogden_tet4_gradient_a_msoa(
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
        sfem::codegen::neohookean_ogden_tet4_gradient_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        return sfem::codegen::launch_status("neohookean_ogden_tet4_gradient_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet4_gradient_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        return sfem::codegen::launch_status("neohookean_ogden_tet4_gradient_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_tet4_gradient_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_tet4_apply_soa_diagnostics_data = {
  "neohookean_ogden_tet4_apply_soa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
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
  253,
  366,
  120,
  69,
  10,
  12,
  1,
  2,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_neohookean_ogden_tet4_apply_soa_diagnostics(void) {
  return &sfem::codegen::neohookean_ogden_tet4_apply_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void neohookean_ogden_tet4_apply_a_msoa_impl(
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
  static constexpr int NS = 4;

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[NS];
    s_t bu_data[NS * NC];
    s_t bh_data[NS * NC];
    s_t bout_data[NS * NC];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node];
      {
        ev_node[0] = element_shape[0];
      }
    }
    const s_t *const u_components[NC] = {ux, uy, uz};
    const s_t *const h_components[NC] = {hx, hy, hz};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape];
      for (int d = 0; d < NC; ++d) {
        {
          const idx_t node = ev_shape[0];
          bu_data[shape * NC + d] = u_components[d][node * u_stride];
          bh_data[shape * NC + d] = h_components[d][node * h_stride];
        }
      }
    }
    for (int stream = 0; stream < NS * NC; ++stream) {
      {
        bout_data[stream] = s_t(0);
      }
    }

    const s_t *bu_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bu_streams[stream] = &bu_data[stream];
    }
    const s_t *bh_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bh_streams[stream] = &bh_data[stream];
    }
    s_t *bout_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bout_streams[stream] = &bout_data[stream];
    }
    s_t badj0_data;
    const s_t *const badj0 = ageom_stream<s_t, g_t>(
        g_adj0 + evb, &badj0_data, std::is_same<g_t, s_t>());
    s_t badj1_data;
    const s_t *const badj1 = ageom_stream<s_t, g_t>(
        g_adj1 + evb, &badj1_data, std::is_same<g_t, s_t>());
    s_t badj2_data;
    const s_t *const badj2 = ageom_stream<s_t, g_t>(
        g_adj2 + evb, &badj2_data, std::is_same<g_t, s_t>());
    s_t badj3_data;
    const s_t *const badj3 = ageom_stream<s_t, g_t>(
        g_adj3 + evb, &badj3_data, std::is_same<g_t, s_t>());
    s_t badj4_data;
    const s_t *const badj4 = ageom_stream<s_t, g_t>(
        g_adj4 + evb, &badj4_data, std::is_same<g_t, s_t>());
    s_t badj5_data;
    const s_t *const badj5 = ageom_stream<s_t, g_t>(
        g_adj5 + evb, &badj5_data, std::is_same<g_t, s_t>());
    s_t badj6_data;
    const s_t *const badj6 = ageom_stream<s_t, g_t>(
        g_adj6 + evb, &badj6_data, std::is_same<g_t, s_t>());
    s_t badj7_data;
    const s_t *const badj7 = ageom_stream<s_t, g_t>(
        g_adj7 + evb, &badj7_data, std::is_same<g_t, s_t>());
    s_t badj8_data;
    const s_t *const badj8 = ageom_stream<s_t, g_t>(
        g_adj8 + evb, &badj8_data, std::is_same<g_t, s_t>());
    s_t bdet0_data;
    const s_t *const bdet0 = ageom_stream<s_t, g_t>(
        g_det0 + evb, &bdet0_data, std::is_same<g_t, s_t>());

    neohookean_ogden_d3_simplex_tet4_apply_block<s_t, NS>(badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_streams, bh_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy, outz};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape];
      for (int d = 0; d < NC; ++d) {
        atomicAdd(&(out_components[d][ev_shape[0] * out_stride]), bout_data[shape * NC + d]);
      }
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_neohookean_ogden_tet4_apply_a_msoa(
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
        sfem::codegen::neohookean_ogden_tet4_apply_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        return sfem::codegen::launch_status("neohookean_ogden_tet4_apply_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet4_apply_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        return sfem::codegen::launch_status("neohookean_ogden_tet4_apply_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_tet4_apply_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

__host__ __device__ __forceinline__ void neohookean_ogden_tet4_hessian_a_msoa_find_cols(
    const idx_t *const RSTR targets,
    const idx_t *const RSTR row,
    const int lenrow,
    idx_t *const RSTR ks) {
#pragma unroll(4)
  for (int d = 0; d < 4; ++d) {
    ks[d] = 0;
  }
  for (int k = 0; k < lenrow; ++k) {
#pragma unroll(4)
    for (int d = 0; d < 4; ++d) {
      ks[d] += row[k] < targets[d];
    }
  }
}

template <typename s_t>
__host__ __device__ __forceinline__ void neohookean_ogden_tet4_hessian_a_msoa_scatter_bsr(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values) {
  static constexpr int NC = 3;
  static constexpr int NS = 4;
  count_t entries[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    const idx_t dof_i = ev[i];
    const count_t row_begin = rowptr[dof_i];
    const int lenrow = (int)(rowptr[dof_i + 1] - row_begin);
    const idx_t *const RSTR cols = &colidx[row_begin];
    neohookean_ogden_tet4_hessian_a_msoa_find_cols(ev, cols, lenrow, ks);
    for (int j = 0; j < NS; ++j) {
      entries[i * NS + j] = row_begin + ks[j];
    }
  }
  for (int i = 0; i < NS; ++i) {
    for (int j = 0; j < NS; ++j) {
      s_t *const block = &values[entries[i * NS + j] * NC * NC];
      for (int bi = 0; bi < NC; ++bi) {
        const int row = bi * NS + i;
        for (int bj = 0; bj < NC; ++bj) {
          const int col = bj * NS + j;
          atomicAdd(&(block[bi * NC + bj]), element_matrix[row * (NC * NS) + col]);
        }
      }
    }
  }
}

template <typename s_t, typename g_t, int FORMAT>
__global__ void neohookean_ogden_tet4_hessian_a_msoa_assemble_impl(
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
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values,
    const int *const RSTR,
    const ptrdiff_t,
    const ptrdiff_t,
    const idx_t *const RSTR,
    const idx_t *const RSTR,
    idx_t *const RSTR,
    idx_t *const RSTR) {
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int NDOFS = NC * NS;
  const s_t *const u_components[NC] = {ux, uy, uz};

  static_assert(FORMAT == 1,
                "this kernel has no scatter for the requested matrix format");
  for (ptrdiff_t element = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; element < nelements; element += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[NS];
    s_t element_matrix[NDOFS * NDOFS];
    s_t bu_data[NS * NC];
    s_t badj0[NQ];
    s_t badj1[NQ];
    s_t badj2[NQ];
    s_t badj3[NQ];
    s_t badj4[NQ];
    s_t badj5[NQ];
    s_t badj6[NQ];
    s_t badj7[NQ];
    s_t badj8[NQ];
    s_t bdet0[NQ];

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t node = elements[shape][element];
      ev[shape] = node;
      for (int d = 0; d < ND; ++d) {
        bu_data[shape * NC + d] = u_components[d][node * u_stride];
      }
    }


    badj0[0] = s_t(g_adj0[element]);
    badj1[0] = s_t(g_adj1[element]);
    badj2[0] = s_t(g_adj2[element]);
    badj3[0] = s_t(g_adj3[element]);
    badj4[0] = s_t(g_adj4[element]);
    badj5[0] = s_t(g_adj5[element]);
    badj6[0] = s_t(g_adj6[element]);
    badj7[0] = s_t(g_adj7[element]);
    badj8[0] = s_t(g_adj8[element]);
    bdet0[0] = s_t(g_det0[element]);

    neohookean_ogden_d3_simplex_tet4_direct_hessian_element_matrix<s_t, NS>(badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, lmbda, mu, bu_data, element_matrix);

    if constexpr (FORMAT == 1) {
      neohookean_ogden_tet4_hessian_a_msoa_scatter_bsr(ev, element_matrix, rowptr, colidx, values);
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_neohookean_ogden_tet4_hessian_bsr_a_msoa(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet4_hessian_a_msoa_assemble_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, rowptr, colidx, (double *)values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
        return sfem::codegen::launch_status("neohookean_ogden_tet4_hessian_a_msoa_assemble_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tet4_hessian_a_msoa_assemble_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, rowptr, colidx, (float *)values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
        return sfem::codegen::launch_status("neohookean_ogden_tet4_hessian_a_msoa_assemble_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_tet4_hessian_bsr_a_msoa", -1, (int)scalar_bytes);
}
