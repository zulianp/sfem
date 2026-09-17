#include <type_traits>
#include <cuda_runtime.h>
#include "../../cuda/laplace_d2_simplex_local.cuh"
#include "../../../../reference/cuda/quad_tri_q1.hpp"
#include "../../../../reference/cuda/tri3_q1.hpp"
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

static const KernelDiagnostics laplace_tri3_objective_soa_diagnostics_data = {
  "laplace_tri3_objective_soa",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  1,
  2,
  0,
  0,
  2,
  0,
  0,
  0,
  2,
  1,
  5,
  32,
  59,
  0,
  3,
  5,
  6,
  1,
  2,
  6,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_laplace_tri3_objective_soa_diagnostics(void) {
  return &sfem::codegen::laplace_tri3_objective_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void laplace_tri3_objective_a_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_met0,
        const g_t *const RSTR g_met1,
        const g_t *const RSTR g_met2,
        const s_t kappa,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        s_t *const RSTR value
) {
  static constexpr int NC = 1;
  static constexpr int NQ = 1;
  static constexpr int NS = 3;
  const s_t *const affine_q_weight = sfem::codegen::quad_tri_q1<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    const int ne = 1;
    idx_t ev[VS * NS];
    s_t bu_data[NS * NC][VS];
    s_t bvalue[VS];

    for (int element_node = 0; element_node < NS; ++element_node) {
      const idx_t *const RSTR element_shape = elements[element_node] + evb;
      idx_t *const RSTR ev_node = &ev[element_node * VS];
      {
        ev_node[0] = element_shape[0];
      }
    }
    const s_t *const u_components[NC] = {ux};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR ev_shape = &ev[shape * VS];
      for (int d = 0; d < NC; ++d) {
        {
          const idx_t node = ev_shape[0];
          bu_data[shape * NC + d][0] = u_components[d][node * u_stride];
        }
      }
    }
    {
      bvalue[0] = s_t(0);
    }

    const s_t *bu_streams[NS * NC];
    for (int stream = 0; stream < NS * NC; ++stream) {
      bu_streams[stream] = bu_data[stream];
    }
    s_t bgeom_metric0_data[VS];
    const s_t *const bgeom_metric0 = ageom_stream<s_t, g_t, VS>(
        ne, g_met0 + evb, bgeom_metric0_data, std::is_same<g_t, s_t>());
    s_t bgeom_metric1_data[VS];
    const s_t *const bgeom_metric1 = ageom_stream<s_t, g_t, VS>(
        ne, g_met1 + evb, bgeom_metric1_data, std::is_same<g_t, s_t>());
    s_t bgeom_metric2_data[VS];
    const s_t *const bgeom_metric2 = ageom_stream<s_t, g_t, VS>(
        ne, g_met2 + evb, bgeom_metric2_data, std::is_same<g_t, s_t>());

    laplace_d2_simplex_tri3_metric_objective_block<s_t, NQ, NS, VS>(ne, 0, bgeom_metric0, bgeom_metric1, bgeom_metric2, affine_q_weight, kappa, bu_streams, bvalue);

    {
      value[evb + 0] += bvalue[0];
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_laplace_tri3_objective_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        void *const RSTR value,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tri3_objective_a_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, u_stride, (const double *)ux, (double *)value);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tri3_objective_a_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, u_stride, (const float *)ux, (float *)value);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("laplace_tri3_objective_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_tri3_gradient_soa_diagnostics_data = {
  "laplace_tri3_gradient_soa",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  0,
  2,
  0,
  0,
  0,
  0,
  0,
  0,
  2,
  2,
  2,
  13,
  84,
  0,
  2,
  5,
  6,
  1,
  2,
  6,
  0,
  3,
  3,
  3,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_laplace_tri3_gradient_soa_diagnostics(void) {
  return &sfem::codegen::laplace_tri3_gradient_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void laplace_tri3_gradient_a_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_met0,
        const g_t *const RSTR g_met1,
        const g_t *const RSTR g_met2,
        const s_t kappa,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx
) {

  for (ptrdiff_t element = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; element < nelements; element += (ptrdiff_t)blockDim.x * gridDim.x) {
    const idx_t ev0 = elements[0][element];
    const idx_t ev1 = elements[1][element];
    const idx_t ev2 = elements[2][element];
    const s_t u0 = ux[ev0 * u_stride];
    const s_t u1 = ux[ev1 * u_stride];
    const s_t u2 = ux[ev2 * u_stride];
    const s_t fff0 = kappa * s_t(g_met0[element]);
    const s_t fff1 = kappa * s_t(g_met1[element]);
    const s_t fff2 = kappa * s_t(g_met2[element]);
    const s_t t0 = -u0 + u1;
    const s_t t1 = -u0 + u2;
    const s_t t2 = fff0*t0 + fff1*t1;
    const s_t t3 = fff1*t0 + fff2*t1;
    const s_t e0 = -t2 - t3;
    atomicAdd(&(outx[ev0 * out_stride]), e0);
    const s_t e1 = t2;
    atomicAdd(&(outx[ev1 * out_stride]), e1);
    const s_t e2 = t3;
    atomicAdd(&(outx[ev2 * out_stride]), e2);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_laplace_tri3_gradient_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tri3_gradient_a_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tri3_gradient_a_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("laplace_tri3_gradient_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_tri3_apply_soa_diagnostics_data = {
  "laplace_tri3_apply_soa",
  "TRI3",
  2,
  1,
  3,
  16,
  1,
  0,
  2,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  2,
  2,
  13,
  84,
  0,
  2,
  5,
  6,
  1,
  2,
  0,
  6,
  3,
  3,
  3,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_laplace_tri3_apply_soa_diagnostics(void) {
  return &sfem::codegen::laplace_tri3_apply_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void laplace_tri3_apply_a_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_met0,
        const g_t *const RSTR g_met1,
        const g_t *const RSTR g_met2,
        const s_t kappa,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const ptrdiff_t out_stride,
        s_t *const RSTR outx
) {

  for (ptrdiff_t element = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; element < nelements; element += (ptrdiff_t)blockDim.x * gridDim.x) {
    const idx_t ev0 = elements[0][element];
    const idx_t ev1 = elements[1][element];
    const idx_t ev2 = elements[2][element];
    const s_t u0 = hx[ev0 * h_stride];
    const s_t u1 = hx[ev1 * h_stride];
    const s_t u2 = hx[ev2 * h_stride];
    const s_t fff0 = kappa * s_t(g_met0[element]);
    const s_t fff1 = kappa * s_t(g_met1[element]);
    const s_t fff2 = kappa * s_t(g_met2[element]);
    const s_t t0 = -u0 + u1;
    const s_t t1 = -u0 + u2;
    const s_t t2 = fff0*t0 + fff1*t1;
    const s_t t3 = fff1*t0 + fff2*t1;
    const s_t e0 = -t2 - t3;
    atomicAdd(&(outx[ev0 * out_stride]), e0);
    const s_t e1 = t2;
    atomicAdd(&(outx[ev1 * out_stride]), e1);
    const s_t e2 = t3;
    atomicAdd(&(outx[ev2 * out_stride]), e2);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_laplace_tri3_apply_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tri3_apply_a_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tri3_apply_a_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("laplace_tri3_apply_a_msoa", -1, (int)scalar_bytes);
}
