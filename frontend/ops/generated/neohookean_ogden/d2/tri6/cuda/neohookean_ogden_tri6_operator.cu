#include <type_traits>
#include <cuda_runtime.h>
#include "neohookean_ogden_d2_simplex_local.cuh"
#include "neohookean_ogden_d2_simplex_hessian.cuh"
#include "quad_tri_q3.cuh"
#include "tri6_q3.cuh"
#include "quad_tri_q6.cuh"
#include "tri6_q6.cuh"
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

static const KernelDiagnostics neohookean_ogden_tri6_objective_soa_diagnostics_data = {
  "neohookean_ogden_tri6_objective_soa",
  "TRI6",
  2,
  6,
  6,
  16,
  4,
  9,
  7,
  0,
  0,
  5,
  0,
  1,
  0,
  2,
  4,
  41,
  408,
  714,
  3,
  7,
  5,
  72,
  6,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_neohookean_ogden_tri6_objective_soa_diagnostics(void) {
  return &sfem::codegen::neohookean_ogden_tri6_objective_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void neohookean_ogden_tri6_objective_steps_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const int nsteps,
        const s_t *const RSTR steps,
        s_t *const RSTR value
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NQ = 6;
  static constexpr int NS = 6;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tri6_q6<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tri6_q6<s_t>::grad_ref_y();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tri_q6<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[NS];
    s_t bu_data[NS * NC];
    s_t bh_data[NS * NC];
    s_t bcoordinate_data[NS * ND];
    s_t badj0[NQ];
    s_t badj1[NQ];
    s_t badj2[NQ];
    s_t badj3[NQ];
    s_t bdet0[NQ];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node];
      {
        ev_node[0] = element_shape[0];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape];
      for (int d = 0; d < ND; ++d) {
        {
          bcoordinate_data[shape * ND + d] = coordinate_components[d][ev_shape[0]];
        }
      }
    }

    const s_t *const u_components[NC] = {ux, uy};
    const s_t *const h_components[NC] = {hx, hy};
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

    for (int q = 0; q < NQ; ++q) {
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      s_t J00_values;
      s_t J01_values;
      s_t J10_values;
      s_t J11_values;
      {
        J00_values = s_t(0);
        J01_values = s_t(0);
        J10_values = s_t(0);
        J11_values = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = isoparametric_grad_ref_x[q * NS + shape];
        const s_t g1 = isoparametric_grad_ref_y[q * NS + shape];
        {
          J00_values += bcoordinate_data[2 * shape] * g0;
          J01_values += bcoordinate_data[2 * shape] * g1;
          J10_values += bcoordinate_data[2 * shape + 1] * g0;
          J11_values += bcoordinate_data[2 * shape + 1] * g1;
        }
      }
      {
        const s_t J00 = J00_values;
        const s_t J01 = J01_values;
        const s_t J10 = J10_values;
        const s_t J11 = J11_values;
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, q);
      }
    }

    for (int step = 0; step < nsteps; ++step) {
      {
        value[(ptrdiff_t)step * nelements + evb + 0] = s_t(0);
      }
    }

    neohookean_ogden_d2_simplex_objective_block<s_t, NQ, NS>(1, badj0, badj1, badj2, badj3, bdet0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_q_weight, lmbda, mu, bu_streams, bh_streams, nsteps, steps, nelements, &value[evb]);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_neohookean_ogden_tri6_objective_steps_i_msoa(
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
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tri6_objective_steps_i_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, nsteps, (const double *)steps, (double *)value);
        return sfem::codegen::launch_status("neohookean_ogden_tri6_objective_steps_i_msoa_impl");
    }
    case (int)sizeof(float): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tri6_objective_steps_i_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, nsteps, (const float *)steps, (float *)value);
        return sfem::codegen::launch_status("neohookean_ogden_tri6_objective_steps_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_tri6_objective_steps_i_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_tri6_gradient_soa_diagnostics_data = {
  "neohookean_ogden_tri6_gradient_soa",
  "TRI6",
  2,
  6,
  6,
  16,
  4,
  11,
  16,
  1,
  0,
  0,
  0,
  1,
  0,
  2,
  13,
  55,
  774,
  1080,
  9,
  10,
  5,
  72,
  6,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_neohookean_ogden_tri6_gradient_soa_diagnostics(void) {
  return &sfem::codegen::neohookean_ogden_tri6_gradient_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void neohookean_ogden_tri6_gradient_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NQ = 6;
  static constexpr int NS = 6;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tri6_q6<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tri6_q6<s_t>::grad_ref_y();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tri_q6<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[NS];
    s_t bu_data[NS * NC];
    s_t bout_data[NS * NC];
    s_t bcoordinate_data[NS * ND];
    s_t badj0[NQ];
    s_t badj1[NQ];
    s_t badj2[NQ];
    s_t badj3[NQ];
    s_t bdet0[NQ];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node];
      {
        ev_node[0] = element_shape[0];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape];
      for (int d = 0; d < ND; ++d) {
        {
          bcoordinate_data[shape * ND + d] = coordinate_components[d][ev_shape[0]];
        }
      }
    }
    const s_t *const u_components[NC] = {ux, uy};

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

    for (int q = 0; q < NQ; ++q) {
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      s_t J00_values;
      s_t J01_values;
      s_t J10_values;
      s_t J11_values;
      {
        J00_values = s_t(0);
        J01_values = s_t(0);
        J10_values = s_t(0);
        J11_values = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = isoparametric_grad_ref_x[q * NS + shape];
        const s_t g1 = isoparametric_grad_ref_y[q * NS + shape];
        {
          J00_values += bcoordinate_data[2 * shape] * g0;
          J01_values += bcoordinate_data[2 * shape] * g1;
          J10_values += bcoordinate_data[2 * shape + 1] * g0;
          J11_values += bcoordinate_data[2 * shape + 1] * g1;
        }
      }
      {
        const s_t J00 = J00_values;
        const s_t J01 = J01_values;
        const s_t J10 = J10_values;
        const s_t J11 = J11_values;
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, q);
      }
    }

    neohookean_ogden_d2_simplex_gradient_block<s_t, NQ, NS>(1, badj0, badj1, badj2, badj3, bdet0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_q_weight, lmbda, mu, bu_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy};

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

extern "C" int cu_neohookean_ogden_tri6_gradient_i_msoa(
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
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tri6_gradient_i_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, out_stride, (double *)outx, (double *)outy);
        return sfem::codegen::launch_status("neohookean_ogden_tri6_gradient_i_msoa_impl");
    }
    case (int)sizeof(float): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tri6_gradient_i_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, out_stride, (float *)outx, (float *)outy);
        return sfem::codegen::launch_status("neohookean_ogden_tri6_gradient_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_tri6_gradient_i_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics neohookean_ogden_tri6_apply_soa_diagnostics_data = {
  "neohookean_ogden_tri6_apply_soa",
  "TRI6",
  2,
  6,
  6,
  16,
  4,
  36,
  57,
  1,
  0,
  6,
  0,
  1,
  0,
  2,
  34,
  127,
  774,
  1080,
  30,
  20,
  5,
  72,
  6,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_neohookean_ogden_tri6_apply_soa_diagnostics(void) {
  return &sfem::codegen::neohookean_ogden_tri6_apply_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void neohookean_ogden_tri6_apply_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t lmbda,
        const s_t mu,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const s_t *const RSTR uy,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const s_t *const RSTR hy,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx,
        s_t *const RSTR outy
) {
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NQ = 6;
  static constexpr int NS = 6;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tri6_q6<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tri6_q6<s_t>::grad_ref_y();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tri_q6<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[NS];
    s_t bu_data[NS * NC];
    s_t bh_data[NS * NC];
    s_t bout_data[NS * NC];
    s_t bcoordinate_data[NS * ND];
    s_t badj0[NQ];
    s_t badj1[NQ];
    s_t badj2[NQ];
    s_t badj3[NQ];
    s_t bdet0[NQ];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node];
      {
        ev_node[0] = element_shape[0];
      }
    }
    const g_t *const coordinate_components[ND] = {x, y};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape];
      for (int d = 0; d < ND; ++d) {
        {
          bcoordinate_data[shape * ND + d] = coordinate_components[d][ev_shape[0]];
        }
      }
    }
    const s_t *const u_components[NC] = {ux, uy};
    const s_t *const h_components[NC] = {hx, hy};

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

    for (int q = 0; q < NQ; ++q) {
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      s_t J00_values;
      s_t J01_values;
      s_t J10_values;
      s_t J11_values;
      {
        J00_values = s_t(0);
        J01_values = s_t(0);
        J10_values = s_t(0);
        J11_values = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = isoparametric_grad_ref_x[q * NS + shape];
        const s_t g1 = isoparametric_grad_ref_y[q * NS + shape];
        {
          J00_values += bcoordinate_data[2 * shape] * g0;
          J01_values += bcoordinate_data[2 * shape] * g1;
          J10_values += bcoordinate_data[2 * shape + 1] * g0;
          J11_values += bcoordinate_data[2 * shape + 1] * g1;
        }
      }
      {
        const s_t J00 = J00_values;
        const s_t J01 = J01_values;
        const s_t J10 = J10_values;
        const s_t J11 = J11_values;
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, q);
      }
    }

    neohookean_ogden_d2_simplex_apply_block<s_t, NQ, NS>(1, badj0, badj1, badj2, badj3, bdet0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_q_weight, lmbda, mu, bu_streams, bh_streams, bout_streams);

    s_t *const out_components[NC] = {outx, outy};

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

extern "C" int cu_neohookean_ogden_tri6_apply_i_msoa(
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
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tri6_apply_i_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, h_stride, (const double *)hx, (const double *)hy, out_stride, (double *)outx, (double *)outy);
        return sfem::codegen::launch_status("neohookean_ogden_tri6_apply_i_msoa_impl");
    }
    case (int)sizeof(float): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tri6_apply_i_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, h_stride, (const float *)hx, (const float *)hy, out_stride, (float *)outx, (float *)outy);
        return sfem::codegen::launch_status("neohookean_ogden_tri6_apply_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_tri6_apply_i_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

__host__ __device__ __forceinline__ void neohookean_ogden_tri6_hessian_i_msoa_find_cols(
    const idx_t *const RSTR targets,
    const idx_t *const RSTR row,
    const int lenrow,
    idx_t *const RSTR ks) {
#pragma unroll(6)
  for (int d = 0; d < 6; ++d) {
    ks[d] = 0;
  }
  for (int k = 0; k < lenrow; ++k) {
#pragma unroll(6)
    for (int d = 0; d < 6; ++d) {
      ks[d] += row[k] < targets[d];
    }
  }
}

template <typename s_t>
__host__ __device__ __forceinline__ void neohookean_ogden_tri6_hessian_i_msoa_scatter_bsr(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values) {
  static constexpr int NC = 2;
  static constexpr int NS = 6;
  count_t entries[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    const idx_t dof_i = ev[i];
    const count_t row_begin = rowptr[dof_i];
    const int lenrow = (int)(rowptr[dof_i + 1] - row_begin);
    const idx_t *const RSTR cols = &colidx[row_begin];
    neohookean_ogden_tri6_hessian_i_msoa_find_cols(ev, cols, lenrow, ks);
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
__global__ void neohookean_ogden_tri6_hessian_i_msoa_assemble_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const *const RSTR points,
    const s_t lmbda,
    const s_t mu,
    const ptrdiff_t u_stride,
    const s_t *const RSTR ux,
    const s_t *const RSTR uy,
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
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NQ = 6;
  static constexpr int NS = 6;
  static constexpr int NDOFS = NC * NS;
  const s_t *const u_components[NC] = {ux, uy};
  const g_t *const RSTR x = points[0];
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tri6_q6<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tri6_q6<s_t>::grad_ref_y();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tri_q6<s_t>::q_weight();

  static_assert(FORMAT == 1,
                "this kernel has no scatter for the requested matrix format");
  for (ptrdiff_t element = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; element < nelements; element += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[NS];
    s_t element_matrix[NDOFS * NDOFS];
    s_t bcoordinate_data[NS * ND];
    s_t bu_data[NS * NC];
    s_t badj0[NQ];
    s_t badj1[NQ];
    s_t badj2[NQ];
    s_t badj3[NQ];
    s_t bdet0[NQ];

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t node = elements[shape][element];
      ev[shape] = node;
      for (int d = 0; d < ND; ++d) {
        bcoordinate_data[shape * ND + d] = s_t(points[d][node]);
        bu_data[shape * NC + d] = u_components[d][node * u_stride];
      }
    }


    for (int q = 0; q < NQ; ++q) {
      s_t *badj_streams[ND * ND] = {badj0, badj1, badj2, badj3};
      s_t J00_values;
      s_t J01_values;
      s_t J10_values;
      s_t J11_values;
      {
        J00_values = s_t(0);
        J01_values = s_t(0);
        J10_values = s_t(0);
        J11_values = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t g0 = isoparametric_grad_ref_x[q * NS + shape];
        const s_t g1 = isoparametric_grad_ref_y[q * NS + shape];
        {
          J00_values += bcoordinate_data[2 * shape] * g0;
          J01_values += bcoordinate_data[2 * shape] * g1;
          J10_values += bcoordinate_data[2 * shape + 1] * g0;
          J11_values += bcoordinate_data[2 * shape + 1] * g1;
        }
      }
      {
        const s_t J00 = J00_values;
        const s_t J01 = J01_values;
        const s_t J10 = J10_values;
        const s_t J11 = J11_values;
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, badj_streams, bdet0, q);
      }
    }

    neohookean_ogden_d2_simplex_direct_hessian_reference_element_matrix<s_t, NQ, NS>(badj0, badj1, badj2, badj3, bdet0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_q_weight, lmbda, mu, bu_data, element_matrix);

    if constexpr (FORMAT == 1) {
      neohookean_ogden_tri6_hessian_i_msoa_scatter_bsr(ev, element_matrix, rowptr, colidx, values);
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_neohookean_ogden_tri6_hessian_bsr_i_msoa(
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
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tri6_hessian_i_msoa_assemble_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const double *)ux, (const double *)uy, rowptr, colidx, (double *)values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
        return sfem::codegen::launch_status("neohookean_ogden_tri6_hessian_i_msoa_assemble_impl");
    }
    case (int)sizeof(float): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::neohookean_ogden_tri6_hessian_i_msoa_assemble_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, lmbda, mu, u_stride, (const float *)ux, (const float *)uy, rowptr, colidx, (float *)values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
        return sfem::codegen::launch_status("neohookean_ogden_tri6_hessian_i_msoa_assemble_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("neohookean_ogden_tri6_hessian_bsr_i_msoa", -1, (int)scalar_bytes);
}
