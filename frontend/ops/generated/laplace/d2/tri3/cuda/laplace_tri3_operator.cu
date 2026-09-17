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
__global__ void laplace_tri3_objective_steps_a_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_met0,
        const g_t *const RSTR g_met1,
        const g_t *const RSTR g_met2,
        const s_t kappa,
        const ptrdiff_t u_stride,
        const s_t *const RSTR ux,
        const ptrdiff_t h_stride,
        const s_t *const RSTR hx,
        const int nsteps,
        const s_t *const RSTR steps,
        s_t *const RSTR value
) {

  for (ptrdiff_t element = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; element < nelements; element += (ptrdiff_t)blockDim.x * gridDim.x) {
    const idx_t ev0 = elements[0][element];
    const idx_t ev1 = elements[1][element];
    const idx_t ev2 = elements[2][element];
    const s_t x0 = ux[ev0 * u_stride];
    const s_t x1 = ux[ev1 * u_stride];
    const s_t x2 = ux[ev2 * u_stride];
    const s_t h0 = hx[ev0 * h_stride];
    const s_t h1 = hx[ev1 * h_stride];
    const s_t h2 = hx[ev2 * h_stride];
    const s_t fff0 = kappa * s_t(g_met0[element]);
    const s_t fff1 = kappa * s_t(g_met1[element]);
    const s_t fff2 = kappa * s_t(g_met2[element]);
    for (int step = 0; step < nsteps; ++step) {
      const s_t alpha = steps[step];
      const s_t u0 = x0 + alpha * h0;
      const s_t u1 = x1 + alpha * h1;
      const s_t u2 = x2 + alpha * h2;
      const s_t t0 = -u0 + u1;
      const s_t t1 = -u0 + u2;
      value[(ptrdiff_t)step * nelements + element] = ((s_t(1) / s_t(2)))*t0*(fff0*t0 + fff1*t1) + ((s_t(1) / s_t(2)))*t1*(fff1*t0 + fff2*t1);
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_laplace_tri3_objective_steps_a_msoa(
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
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tri3_objective_steps_a_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tri3_objective_steps_a_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_met0, g_met1, g_met2, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("laplace_tri3_objective_steps_a_msoa", -1, (int)scalar_bytes);
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
