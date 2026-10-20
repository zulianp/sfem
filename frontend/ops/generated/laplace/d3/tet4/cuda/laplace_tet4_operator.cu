#include <type_traits>
#include <cuda_runtime.h>
#include "laplace_d3_simplex_local.cuh"
#include "laplace_d3_simplex_hessian.cuh"
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

static const KernelDiagnostics laplace_tet4_objective_soa_diagnostics_data = {
  "laplace_tet4_objective_soa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  2,
  2,
  0,
  0,
  3,
  0,
  0,
  0,
  3,
  1,
  7,
  82,
  195,
  0,
  4,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_laplace_tet4_objective_soa_diagnostics(void) {
  return &sfem::codegen::laplace_tet4_objective_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void laplace_tet4_objective_steps_a_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_met0,
        const g_t *const RSTR g_met1,
        const g_t *const RSTR g_met2,
        const g_t *const RSTR g_met3,
        const g_t *const RSTR g_met4,
        const g_t *const RSTR g_met5,
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
    const idx_t ev3 = elements[3][element];
    const s_t x0 = ux[ev0 * u_stride];
    const s_t x1 = ux[ev1 * u_stride];
    const s_t x2 = ux[ev2 * u_stride];
    const s_t x3 = ux[ev3 * u_stride];
    const s_t h0 = hx[ev0 * h_stride];
    const s_t h1 = hx[ev1 * h_stride];
    const s_t h2 = hx[ev2 * h_stride];
    const s_t h3 = hx[ev3 * h_stride];
    const s_t fff0 = kappa * s_t(g_met0[element]);
    const s_t fff1 = kappa * s_t(g_met1[element]);
    const s_t fff2 = kappa * s_t(g_met2[element]);
    const s_t fff3 = kappa * s_t(g_met3[element]);
    const s_t fff4 = kappa * s_t(g_met4[element]);
    const s_t fff5 = kappa * s_t(g_met5[element]);
    for (int step = 0; step < nsteps; ++step) {
      const s_t alpha = steps[step];
      const s_t u0 = x0 + alpha * h0;
      const s_t u1 = x1 + alpha * h1;
      const s_t u2 = x2 + alpha * h2;
      const s_t u3 = x3 + alpha * h3;
      const s_t t0 = -u0 + u1;
      const s_t t1 = -u0 + u2;
      const s_t t2 = -u0 + u3;
      value[(ptrdiff_t)step * nelements + element] = ((s_t(1) / s_t(2)))*t0*(fff0*t0 + fff1*t1 + fff2*t2) + ((s_t(1) / s_t(2)))*t1*(fff1*t0 + fff3*t1 + fff4*t2) + ((s_t(1) / s_t(2)))*t2*(fff2*t0 + fff4*t1 + fff5*t2);
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_laplace_tet4_objective_steps_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
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
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tet4_objective_steps_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_met0, g_met1, g_met2, g_met3, g_met4, g_met5, kappa, u_stride, (const double *)ux, h_stride, (const double *)hx, nsteps, (const double *)steps, (double *)value);
        return sfem::codegen::launch_status("laplace_tet4_objective_steps_a_msoa_impl");
    }
    case (int)sizeof(float): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tet4_objective_steps_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_met0, g_met1, g_met2, g_met3, g_met4, g_met5, kappa, u_stride, (const float *)ux, h_stride, (const float *)hx, nsteps, (const float *)steps, (float *)value);
        return sfem::codegen::launch_status("laplace_tet4_objective_steps_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("laplace_tet4_objective_steps_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_tet4_gradient_soa_diagnostics_data = {
  "laplace_tet4_gradient_soa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  0,
  3,
  0,
  0,
  0,
  0,
  0,
  0,
  3,
  3,
  3,
  27,
  270,
  0,
  2,
  10,
  12,
  1,
  2,
  12,
  0,
  4,
  4,
  4,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_laplace_tet4_gradient_soa_diagnostics(void) {
  return &sfem::codegen::laplace_tet4_gradient_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void laplace_tet4_gradient_a_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_met0,
        const g_t *const RSTR g_met1,
        const g_t *const RSTR g_met2,
        const g_t *const RSTR g_met3,
        const g_t *const RSTR g_met4,
        const g_t *const RSTR g_met5,
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
    const idx_t ev3 = elements[3][element];
    const s_t u0 = ux[ev0 * u_stride];
    const s_t u1 = ux[ev1 * u_stride];
    const s_t u2 = ux[ev2 * u_stride];
    const s_t u3 = ux[ev3 * u_stride];
    const s_t fff0 = kappa * s_t(g_met0[element]);
    const s_t fff1 = kappa * s_t(g_met1[element]);
    const s_t fff2 = kappa * s_t(g_met2[element]);
    const s_t fff3 = kappa * s_t(g_met3[element]);
    const s_t fff4 = kappa * s_t(g_met4[element]);
    const s_t fff5 = kappa * s_t(g_met5[element]);
    const s_t t0 = -u0 + u1;
    const s_t t1 = -u0 + u2;
    const s_t t2 = -u0 + u3;
    const s_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
    const s_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
    const s_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
    const s_t e0 = -t3 - t4 - t5;
    atomicAdd(&(outx[ev0 * out_stride]), e0);
    const s_t e1 = t3;
    atomicAdd(&(outx[ev1 * out_stride]), e1);
    const s_t e2 = t4;
    atomicAdd(&(outx[ev2 * out_stride]), e2);
    const s_t e3 = t5;
    atomicAdd(&(outx[ev3 * out_stride]), e3);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_laplace_tet4_gradient_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const real_t kappa,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tet4_gradient_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_met0, g_met1, g_met2, g_met3, g_met4, g_met5, kappa, u_stride, (const double *)ux, out_stride, (double *)outx);
        return sfem::codegen::launch_status("laplace_tet4_gradient_a_msoa_impl");
    }
    case (int)sizeof(float): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tet4_gradient_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_met0, g_met1, g_met2, g_met3, g_met4, g_met5, kappa, u_stride, (const float *)ux, out_stride, (float *)outx);
        return sfem::codegen::launch_status("laplace_tet4_gradient_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("laplace_tet4_gradient_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics laplace_tet4_apply_soa_diagnostics_data = {
  "laplace_tet4_apply_soa",
  "TET4",
  3,
  1,
  4,
  16,
  1,
  0,
  3,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  3,
  3,
  27,
  270,
  0,
  2,
  10,
  12,
  1,
  2,
  0,
  12,
  4,
  4,
  4,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_laplace_tet4_apply_soa_diagnostics(void) {
  return &sfem::codegen::laplace_tet4_apply_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void laplace_tet4_apply_a_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const RSTR g_met0,
        const g_t *const RSTR g_met1,
        const g_t *const RSTR g_met2,
        const g_t *const RSTR g_met3,
        const g_t *const RSTR g_met4,
        const g_t *const RSTR g_met5,
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
    const idx_t ev3 = elements[3][element];
    const s_t u0 = hx[ev0 * h_stride];
    const s_t u1 = hx[ev1 * h_stride];
    const s_t u2 = hx[ev2 * h_stride];
    const s_t u3 = hx[ev3 * h_stride];
    const s_t fff0 = kappa * s_t(g_met0[element]);
    const s_t fff1 = kappa * s_t(g_met1[element]);
    const s_t fff2 = kappa * s_t(g_met2[element]);
    const s_t fff3 = kappa * s_t(g_met3[element]);
    const s_t fff4 = kappa * s_t(g_met4[element]);
    const s_t fff5 = kappa * s_t(g_met5[element]);
    const s_t t0 = -u0 + u1;
    const s_t t1 = -u0 + u2;
    const s_t t2 = -u0 + u3;
    const s_t t3 = fff0*t0 + fff1*t1 + fff2*t2;
    const s_t t4 = fff1*t0 + fff3*t1 + fff4*t2;
    const s_t t5 = fff2*t0 + fff4*t1 + fff5*t2;
    const s_t e0 = -t3 - t4 - t5;
    atomicAdd(&(outx[ev0 * out_stride]), e0);
    const s_t e1 = t3;
    atomicAdd(&(outx[ev1 * out_stride]), e1);
    const s_t e2 = t4;
    atomicAdd(&(outx[ev2 * out_stride]), e2);
    const s_t e3 = t5;
    atomicAdd(&(outx[ev3 * out_stride]), e3);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_laplace_tet4_apply_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_met0,
        const geom_t *const RSTR g_met1,
        const geom_t *const RSTR g_met2,
        const geom_t *const RSTR g_met3,
        const geom_t *const RSTR g_met4,
        const geom_t *const RSTR g_met5,
        const real_t kappa,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tet4_apply_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_met0, g_met1, g_met2, g_met3, g_met4, g_met5, kappa, h_stride, (const double *)hx, out_stride, (double *)outx);
        return sfem::codegen::launch_status("laplace_tet4_apply_a_msoa_impl");
    }
    case (int)sizeof(float): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tet4_apply_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_met0, g_met1, g_met2, g_met3, g_met4, g_met5, kappa, h_stride, (const float *)hx, out_stride, (float *)outx);
        return sfem::codegen::launch_status("laplace_tet4_apply_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("laplace_tet4_apply_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

__host__ __device__ __forceinline__ void laplace_tet4_hessian_a_msoa_find_cols(
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
__host__ __device__ __forceinline__ void laplace_tet4_hessian_a_msoa_scatter_bsr(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values) {
  static constexpr int NC = 1;
  static constexpr int NS = 4;
  count_t entries[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    const idx_t dof_i = ev[i];
    const count_t row_begin = rowptr[dof_i];
    const int lenrow = (int)(rowptr[dof_i + 1] - row_begin);
    const idx_t *const RSTR cols = &colidx[row_begin];
    laplace_tet4_hessian_a_msoa_find_cols(ev, cols, lenrow, ks);
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
__global__ void laplace_tet4_hessian_a_msoa_assemble_impl(
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
    const s_t kappa,
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
  static constexpr int NC = 1;
  static constexpr int NQ = 1;
  static constexpr int NS = 4;
  static constexpr int NDOFS = NC * NS;

  static_assert(FORMAT == 1,
                "this kernel has no scatter for the requested matrix format");
  for (ptrdiff_t element = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; element < nelements; element += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[NS];
    s_t element_matrix[NDOFS * NDOFS];
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

    laplace_d3_simplex_tet4_direct_hessian_element_matrix<s_t, NS>(badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, kappa, element_matrix);

    if constexpr (FORMAT == 1) {
      laplace_tet4_hessian_a_msoa_scatter_bsr(ev, element_matrix, rowptr, colidx, values);
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_laplace_tet4_hessian_bsr_a_msoa(
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
        const real_t kappa,
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
        sfem::codegen::laplace_tet4_hessian_a_msoa_assemble_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, rowptr, colidx, (double *)values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
        return sfem::codegen::launch_status("laplace_tet4_hessian_a_msoa_assemble_impl");
    }
    case (int)sizeof(float): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::laplace_tet4_hessian_a_msoa_assemble_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, kappa, rowptr, colidx, (float *)values, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr);
        return sfem::codegen::launch_status("laplace_tet4_hessian_a_msoa_assemble_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("laplace_tet4_hessian_bsr_a_msoa", -1, (int)scalar_bytes);
}
