#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "vector_mass_d3_simplex_local.cuh"
#include "geometry_kernels.cuh"
#include "kernel_diagnostics.cuh"
#include "quad_tet_q11.cuh"
#include "tet10_q11.cuh"
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

static const KernelDiagnostics vector_mass_tet10_residual_esoa_diagnostics_data = {
  "vector_mass_tet10_residual_esoa",
  "TET10",
  3,
  11,
  10,
  16,
  4,
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
  5159,
  7590,
  0,
  3,
  10,
  440,
  11,
  1,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet10_residual_esoa_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u0_u0_diagnostics_data = {
  "vector_mass_tet10_jacobian_u0_u0",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  2,
  0,
  0,
  0,
  0,
  0,
  0,
  3,
  1,
  2,
  5159,
  7590,
  0,
  3,
  10,
  440,
  11,
  1,
  0,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet10_jacobian_u0_u0_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u0_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u0_u1_diagnostics_data = {
  "vector_mass_tet10_jacobian_u0_u1",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  0,
  5159,
  7590,
  0,
  0,
  10,
  440,
  11,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet10_jacobian_u0_u1_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u0_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u0_u2_diagnostics_data = {
  "vector_mass_tet10_jacobian_u0_u2",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  0,
  5159,
  7590,
  0,
  0,
  10,
  440,
  11,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet10_jacobian_u0_u2_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u0_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u1_u0_diagnostics_data = {
  "vector_mass_tet10_jacobian_u1_u0",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  0,
  5159,
  7590,
  0,
  0,
  10,
  440,
  11,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet10_jacobian_u1_u0_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u1_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u1_u1_diagnostics_data = {
  "vector_mass_tet10_jacobian_u1_u1",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  2,
  0,
  0,
  0,
  0,
  0,
  0,
  3,
  1,
  2,
  5159,
  7590,
  0,
  3,
  10,
  440,
  11,
  1,
  0,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet10_jacobian_u1_u1_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u1_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u1_u2_diagnostics_data = {
  "vector_mass_tet10_jacobian_u1_u2",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  0,
  5159,
  7590,
  0,
  0,
  10,
  440,
  11,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet10_jacobian_u1_u2_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u1_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u2_u0_diagnostics_data = {
  "vector_mass_tet10_jacobian_u2_u0",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  0,
  5159,
  7590,
  0,
  0,
  10,
  440,
  11,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet10_jacobian_u2_u0_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u2_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u2_u1_diagnostics_data = {
  "vector_mass_tet10_jacobian_u2_u1",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  0,
  5159,
  7590,
  0,
  0,
  10,
  440,
  11,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet10_jacobian_u2_u1_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u2_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_u2_u2_diagnostics_data = {
  "vector_mass_tet10_jacobian_u2_u2",
  "TET10",
  3,
  11,
  10,
  16,
  4,
  0,
  2,
  0,
  0,
  0,
  0,
  0,
  0,
  3,
  1,
  2,
  5159,
  7590,
  0,
  3,
  10,
  440,
  11,
  1,
  0,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet10_jacobian_u2_u2_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_u2_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet10_jacobian_action_esoa_diagnostics_data = {
  "vector_mass_tet10_jacobian_action_esoa",
  "TET10",
  3,
  11,
  10,
  16,
  4,
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
  5159,
  7590,
  0,
  3,
  10,
  440,
  11,
  1,
  0,
  30,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet10_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet10_jacobian_action_esoa_diagnostics_data;
}

extern "C" int cu_vector_mass_tet10_residual_esoa(
    const int scalar_bytes,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR current[30],
    const real_t density,
    void *const RSTR output[30],
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::vector_mass_d3_simplex_residual_block<double, 11, 10>(geometry_stride, (const double *)determinant, sfem::codegen::ref_tet10_q11<double>::shape(), sfem::codegen::quad_tet_q11<double>::q_weight(), (const double *const *)current, density, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::vector_mass_d3_simplex_residual_block<float, 11, 10>(geometry_stride, (const float *)determinant, sfem::codegen::ref_tet10_q11<float>::shape(), sfem::codegen::quad_tet_q11<float>::q_weight(), (const float *const *)current, density, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_residual_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void vector_mass_tet10_residual_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const s_t density,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  const s_t *const affine_shape = sfem::codegen::ref_tet10_q11<s_t>::shape();
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    s_t bcurrent[NC * NS];
    s_t boutput[NC * NS];
    const s_t *const current_components[NC] = {u0, u1, u2};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bcurrent[stream] = current_components[field][node * current_stride];
        }
      }
    }

    for (int stream = 0; stream < 30; ++stream) {
      {
        boutput[stream] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[1] = {g_det0 + evb};
    s_t baffine_geometry_data[1];
    const s_t *bageom_streams[1];
    bageom_streams[0] = ageom_stream<s_t, g_t>(
        affine_geometry_sources[0], &baffine_geometry_data[0], std::is_same<g_t, s_t>());

    vector_mass_d3_simplex_residual_block_contiguous<s_t, NQ, NS>(0, bageom_streams[0], affine_shape, affine_q_weight, bcurrent, density, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        {
          atomicAdd(&(out[element_shape[evb] * out_stride]), boutput[stream]);
        }
      }
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_vector_mass_tet10_residual_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
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
        sfem::codegen::vector_mass_tet10_residual_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("vector_mass_tet10_residual_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::vector_mass_tet10_residual_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("vector_mass_tet10_residual_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_residual_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
__global__ void vector_mass_tet10_residual_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t density,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  const s_t *const isoparametric_shape = sfem::codegen::ref_tet10_q11<s_t>::shape();
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    s_t bcoordinates[3 * NS];
    s_t badjugate_data[9][NQ];
    s_t bdeterminant[NQ];
    s_t bcurrent[NC * NS];
    s_t boutput[NC * NS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int d = 0; d < ND; ++d) {
        {
          const idx_t node = element_shape[evb];
          bcoordinates[shape * ND + d] = coordinate_components[d][node];
        }
      }
    }
    const s_t *const current_components[NC] = {u0, u1, u2};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bcurrent[stream] = current_components[field][node * current_stride];
        }
      }
    }

    for (int stream = 0; stream < 30; ++stream) {
      {
        boutput[stream] = s_t(0);
      }
    }

    s_t *badjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    for (int q = 0; q < NQ; ++q) {
      const s_t cell_grad_ref0_0 = isoparametric_grad_ref_x[q * NS + 0];
      const s_t cell_grad_ref0_1 = isoparametric_grad_ref_x[q * NS + 1];
      const s_t cell_grad_ref0_2 = isoparametric_grad_ref_x[q * NS + 2];
      const s_t cell_grad_ref0_3 = isoparametric_grad_ref_x[q * NS + 3];
      const s_t cell_grad_ref0_4 = isoparametric_grad_ref_x[q * NS + 4];
      const s_t cell_grad_ref0_5 = isoparametric_grad_ref_x[q * NS + 5];
      const s_t cell_grad_ref0_6 = isoparametric_grad_ref_x[q * NS + 6];
      const s_t cell_grad_ref0_7 = isoparametric_grad_ref_x[q * NS + 7];
      const s_t cell_grad_ref0_8 = isoparametric_grad_ref_x[q * NS + 8];
      const s_t cell_grad_ref0_9 = isoparametric_grad_ref_x[q * NS + 9];
      const s_t cell_grad_ref1_0 = isoparametric_grad_ref_y[q * NS + 0];
      const s_t cell_grad_ref1_1 = isoparametric_grad_ref_y[q * NS + 1];
      const s_t cell_grad_ref1_2 = isoparametric_grad_ref_y[q * NS + 2];
      const s_t cell_grad_ref1_3 = isoparametric_grad_ref_y[q * NS + 3];
      const s_t cell_grad_ref1_4 = isoparametric_grad_ref_y[q * NS + 4];
      const s_t cell_grad_ref1_5 = isoparametric_grad_ref_y[q * NS + 5];
      const s_t cell_grad_ref1_6 = isoparametric_grad_ref_y[q * NS + 6];
      const s_t cell_grad_ref1_7 = isoparametric_grad_ref_y[q * NS + 7];
      const s_t cell_grad_ref1_8 = isoparametric_grad_ref_y[q * NS + 8];
      const s_t cell_grad_ref1_9 = isoparametric_grad_ref_y[q * NS + 9];
      const s_t cell_grad_ref2_0 = isoparametric_grad_ref_z[q * NS + 0];
      const s_t cell_grad_ref2_1 = isoparametric_grad_ref_z[q * NS + 1];
      const s_t cell_grad_ref2_2 = isoparametric_grad_ref_z[q * NS + 2];
      const s_t cell_grad_ref2_3 = isoparametric_grad_ref_z[q * NS + 3];
      const s_t cell_grad_ref2_4 = isoparametric_grad_ref_z[q * NS + 4];
      const s_t cell_grad_ref2_5 = isoparametric_grad_ref_z[q * NS + 5];
      const s_t cell_grad_ref2_6 = isoparametric_grad_ref_z[q * NS + 6];
      const s_t cell_grad_ref2_7 = isoparametric_grad_ref_z[q * NS + 7];
      const s_t cell_grad_ref2_8 = isoparametric_grad_ref_z[q * NS + 8];
      const s_t cell_grad_ref2_9 = isoparametric_grad_ref_z[q * NS + 9];
      {
        const s_t J00 = bcoordinates[0] * cell_grad_ref0_0 + bcoordinates[3] * cell_grad_ref0_1 + bcoordinates[6] * cell_grad_ref0_2 + bcoordinates[9] * cell_grad_ref0_3 + bcoordinates[12] * cell_grad_ref0_4 + bcoordinates[15] * cell_grad_ref0_5 + bcoordinates[18] * cell_grad_ref0_6 + bcoordinates[21] * cell_grad_ref0_7 + bcoordinates[24] * cell_grad_ref0_8 + bcoordinates[27] * cell_grad_ref0_9;
        const s_t J01 = bcoordinates[0] * cell_grad_ref1_0 + bcoordinates[3] * cell_grad_ref1_1 + bcoordinates[6] * cell_grad_ref1_2 + bcoordinates[9] * cell_grad_ref1_3 + bcoordinates[12] * cell_grad_ref1_4 + bcoordinates[15] * cell_grad_ref1_5 + bcoordinates[18] * cell_grad_ref1_6 + bcoordinates[21] * cell_grad_ref1_7 + bcoordinates[24] * cell_grad_ref1_8 + bcoordinates[27] * cell_grad_ref1_9;
        const s_t J02 = bcoordinates[0] * cell_grad_ref2_0 + bcoordinates[3] * cell_grad_ref2_1 + bcoordinates[6] * cell_grad_ref2_2 + bcoordinates[9] * cell_grad_ref2_3 + bcoordinates[12] * cell_grad_ref2_4 + bcoordinates[15] * cell_grad_ref2_5 + bcoordinates[18] * cell_grad_ref2_6 + bcoordinates[21] * cell_grad_ref2_7 + bcoordinates[24] * cell_grad_ref2_8 + bcoordinates[27] * cell_grad_ref2_9;
        const s_t J10 = bcoordinates[1] * cell_grad_ref0_0 + bcoordinates[4] * cell_grad_ref0_1 + bcoordinates[7] * cell_grad_ref0_2 + bcoordinates[10] * cell_grad_ref0_3 + bcoordinates[13] * cell_grad_ref0_4 + bcoordinates[16] * cell_grad_ref0_5 + bcoordinates[19] * cell_grad_ref0_6 + bcoordinates[22] * cell_grad_ref0_7 + bcoordinates[25] * cell_grad_ref0_8 + bcoordinates[28] * cell_grad_ref0_9;
        const s_t J11 = bcoordinates[1] * cell_grad_ref1_0 + bcoordinates[4] * cell_grad_ref1_1 + bcoordinates[7] * cell_grad_ref1_2 + bcoordinates[10] * cell_grad_ref1_3 + bcoordinates[13] * cell_grad_ref1_4 + bcoordinates[16] * cell_grad_ref1_5 + bcoordinates[19] * cell_grad_ref1_6 + bcoordinates[22] * cell_grad_ref1_7 + bcoordinates[25] * cell_grad_ref1_8 + bcoordinates[28] * cell_grad_ref1_9;
        const s_t J12 = bcoordinates[1] * cell_grad_ref2_0 + bcoordinates[4] * cell_grad_ref2_1 + bcoordinates[7] * cell_grad_ref2_2 + bcoordinates[10] * cell_grad_ref2_3 + bcoordinates[13] * cell_grad_ref2_4 + bcoordinates[16] * cell_grad_ref2_5 + bcoordinates[19] * cell_grad_ref2_6 + bcoordinates[22] * cell_grad_ref2_7 + bcoordinates[25] * cell_grad_ref2_8 + bcoordinates[28] * cell_grad_ref2_9;
        const s_t J20 = bcoordinates[2] * cell_grad_ref0_0 + bcoordinates[5] * cell_grad_ref0_1 + bcoordinates[8] * cell_grad_ref0_2 + bcoordinates[11] * cell_grad_ref0_3 + bcoordinates[14] * cell_grad_ref0_4 + bcoordinates[17] * cell_grad_ref0_5 + bcoordinates[20] * cell_grad_ref0_6 + bcoordinates[23] * cell_grad_ref0_7 + bcoordinates[26] * cell_grad_ref0_8 + bcoordinates[29] * cell_grad_ref0_9;
        const s_t J21 = bcoordinates[2] * cell_grad_ref1_0 + bcoordinates[5] * cell_grad_ref1_1 + bcoordinates[8] * cell_grad_ref1_2 + bcoordinates[11] * cell_grad_ref1_3 + bcoordinates[14] * cell_grad_ref1_4 + bcoordinates[17] * cell_grad_ref1_5 + bcoordinates[20] * cell_grad_ref1_6 + bcoordinates[23] * cell_grad_ref1_7 + bcoordinates[26] * cell_grad_ref1_8 + bcoordinates[29] * cell_grad_ref1_9;
        const s_t J22 = bcoordinates[2] * cell_grad_ref2_0 + bcoordinates[5] * cell_grad_ref2_1 + bcoordinates[8] * cell_grad_ref2_2 + bcoordinates[11] * cell_grad_ref2_3 + bcoordinates[14] * cell_grad_ref2_4 + bcoordinates[17] * cell_grad_ref2_5 + bcoordinates[20] * cell_grad_ref2_6 + bcoordinates[23] * cell_grad_ref2_7 + bcoordinates[26] * cell_grad_ref2_8 + bcoordinates[29] * cell_grad_ref2_9;
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badjugate_streams, bdeterminant, q);
      }
    }


    vector_mass_d3_simplex_residual_block_contiguous<s_t, NQ, NS>(1, bdeterminant, isoparametric_shape, isoparametric_q_weight, bcurrent, density, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        {
          atomicAdd(&(out[element_shape[evb] * out_stride]), boutput[stream]);
        }
      }
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_vector_mass_tet10_residual_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t current_stride,
    const void *const RSTR u0,
    const void *const RSTR u1,
    const void *const RSTR u2,
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
        sfem::codegen::vector_mass_tet10_residual_i_msoa_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, density, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("vector_mass_tet10_residual_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::vector_mass_tet10_residual_i_msoa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, density, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("vector_mass_tet10_residual_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_residual_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int cu_vector_mass_tet10_residual_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR current,
    void *const RSTR output,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return cu_vector_mass_tet10_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], 3, (const double *)current + 0, (const double *)current + 1, (const double *)current + 2, 3, (double *)output + 0, (double *)output + 1, (double *)output + 2, stream);
    }
    case (int)sizeof(float): {
        return cu_vector_mass_tet10_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], 3, (const float *)current + 0, (const float *)current + 1, (const float *)current + 2, 3, (float *)output + 0, (float *)output + 1, (float *)output + 2, stream);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_residual_i_maos", -1, (int)scalar_bytes);
}

extern "C" int cu_vector_mass_tet10_jacobian_action_esoa(
    const int scalar_bytes,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR direction[30],
    const real_t density,
    void *const RSTR output[30],
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::vector_mass_d3_simplex_jacobian_action_block<double, 11, 10>(geometry_stride, (const double *)determinant, sfem::codegen::ref_tet10_q11<double>::shape(), sfem::codegen::quad_tet_q11<double>::q_weight(), (const double *const *)direction, density, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::vector_mass_d3_simplex_jacobian_action_block<float, 11, 10>(geometry_stride, (const float *)determinant, sfem::codegen::ref_tet10_q11<float>::shape(), sfem::codegen::quad_tet_q11<float>::q_weight(), (const float *const *)direction, density, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_jacobian_action_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void vector_mass_tet10_jacobian_action_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_det0,
    const s_t density,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u0_direction,
    const s_t *const RSTR u1_direction,
    const s_t *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  const s_t *const affine_shape = sfem::codegen::ref_tet10_q11<s_t>::shape();
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    s_t bdirection[NC * NS];
    s_t boutput[NC * NS];
    const s_t *const direction_components[NC] = {u0_direction, u1_direction, u2_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bdirection[stream] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 30; ++stream) {
      {
        boutput[stream] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[1] = {g_det0 + evb};
    s_t baffine_geometry_data[1];
    const s_t *bageom_streams[1];
    bageom_streams[0] = ageom_stream<s_t, g_t>(
        affine_geometry_sources[0], &baffine_geometry_data[0], std::is_same<g_t, s_t>());

    vector_mass_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS>(0, bageom_streams[0], affine_shape, affine_q_weight, bdirection, density, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        {
          atomicAdd(&(out[element_shape[evb] * out_stride]), boutput[stream]);
        }
      }
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_vector_mass_tet10_jacobian_action_a_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const void *const RSTR u2_direction,
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
        sfem::codegen::vector_mass_tet10_jacobian_action_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("vector_mass_tet10_jacobian_action_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::vector_mass_tet10_jacobian_action_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("vector_mass_tet10_jacobian_action_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
__global__ void vector_mass_tet10_jacobian_action_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t density,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u0_direction,
    const s_t *const RSTR u1_direction,
    const s_t *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  const s_t *const isoparametric_shape = sfem::codegen::ref_tet10_q11<s_t>::shape();
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

  for (ptrdiff_t evb = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; evb < nelements; evb += (ptrdiff_t)blockDim.x * gridDim.x) {
    s_t bcoordinates[3 * NS];
    s_t badjugate_data[9][NQ];
    s_t bdeterminant[NQ];
    s_t bdirection[NC * NS];
    s_t boutput[NC * NS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int d = 0; d < ND; ++d) {
        {
          const idx_t node = element_shape[evb];
          bcoordinates[shape * ND + d] = coordinate_components[d][node];
        }
      }
    }
    const s_t *const direction_components[NC] = {u0_direction, u1_direction, u2_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        {
          const idx_t node = element_shape[evb];
          bdirection[stream] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 30; ++stream) {
      {
        boutput[stream] = s_t(0);
      }
    }

    s_t *badjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    for (int q = 0; q < NQ; ++q) {
      const s_t cell_grad_ref0_0 = isoparametric_grad_ref_x[q * NS + 0];
      const s_t cell_grad_ref0_1 = isoparametric_grad_ref_x[q * NS + 1];
      const s_t cell_grad_ref0_2 = isoparametric_grad_ref_x[q * NS + 2];
      const s_t cell_grad_ref0_3 = isoparametric_grad_ref_x[q * NS + 3];
      const s_t cell_grad_ref0_4 = isoparametric_grad_ref_x[q * NS + 4];
      const s_t cell_grad_ref0_5 = isoparametric_grad_ref_x[q * NS + 5];
      const s_t cell_grad_ref0_6 = isoparametric_grad_ref_x[q * NS + 6];
      const s_t cell_grad_ref0_7 = isoparametric_grad_ref_x[q * NS + 7];
      const s_t cell_grad_ref0_8 = isoparametric_grad_ref_x[q * NS + 8];
      const s_t cell_grad_ref0_9 = isoparametric_grad_ref_x[q * NS + 9];
      const s_t cell_grad_ref1_0 = isoparametric_grad_ref_y[q * NS + 0];
      const s_t cell_grad_ref1_1 = isoparametric_grad_ref_y[q * NS + 1];
      const s_t cell_grad_ref1_2 = isoparametric_grad_ref_y[q * NS + 2];
      const s_t cell_grad_ref1_3 = isoparametric_grad_ref_y[q * NS + 3];
      const s_t cell_grad_ref1_4 = isoparametric_grad_ref_y[q * NS + 4];
      const s_t cell_grad_ref1_5 = isoparametric_grad_ref_y[q * NS + 5];
      const s_t cell_grad_ref1_6 = isoparametric_grad_ref_y[q * NS + 6];
      const s_t cell_grad_ref1_7 = isoparametric_grad_ref_y[q * NS + 7];
      const s_t cell_grad_ref1_8 = isoparametric_grad_ref_y[q * NS + 8];
      const s_t cell_grad_ref1_9 = isoparametric_grad_ref_y[q * NS + 9];
      const s_t cell_grad_ref2_0 = isoparametric_grad_ref_z[q * NS + 0];
      const s_t cell_grad_ref2_1 = isoparametric_grad_ref_z[q * NS + 1];
      const s_t cell_grad_ref2_2 = isoparametric_grad_ref_z[q * NS + 2];
      const s_t cell_grad_ref2_3 = isoparametric_grad_ref_z[q * NS + 3];
      const s_t cell_grad_ref2_4 = isoparametric_grad_ref_z[q * NS + 4];
      const s_t cell_grad_ref2_5 = isoparametric_grad_ref_z[q * NS + 5];
      const s_t cell_grad_ref2_6 = isoparametric_grad_ref_z[q * NS + 6];
      const s_t cell_grad_ref2_7 = isoparametric_grad_ref_z[q * NS + 7];
      const s_t cell_grad_ref2_8 = isoparametric_grad_ref_z[q * NS + 8];
      const s_t cell_grad_ref2_9 = isoparametric_grad_ref_z[q * NS + 9];
      {
        const s_t J00 = bcoordinates[0] * cell_grad_ref0_0 + bcoordinates[3] * cell_grad_ref0_1 + bcoordinates[6] * cell_grad_ref0_2 + bcoordinates[9] * cell_grad_ref0_3 + bcoordinates[12] * cell_grad_ref0_4 + bcoordinates[15] * cell_grad_ref0_5 + bcoordinates[18] * cell_grad_ref0_6 + bcoordinates[21] * cell_grad_ref0_7 + bcoordinates[24] * cell_grad_ref0_8 + bcoordinates[27] * cell_grad_ref0_9;
        const s_t J01 = bcoordinates[0] * cell_grad_ref1_0 + bcoordinates[3] * cell_grad_ref1_1 + bcoordinates[6] * cell_grad_ref1_2 + bcoordinates[9] * cell_grad_ref1_3 + bcoordinates[12] * cell_grad_ref1_4 + bcoordinates[15] * cell_grad_ref1_5 + bcoordinates[18] * cell_grad_ref1_6 + bcoordinates[21] * cell_grad_ref1_7 + bcoordinates[24] * cell_grad_ref1_8 + bcoordinates[27] * cell_grad_ref1_9;
        const s_t J02 = bcoordinates[0] * cell_grad_ref2_0 + bcoordinates[3] * cell_grad_ref2_1 + bcoordinates[6] * cell_grad_ref2_2 + bcoordinates[9] * cell_grad_ref2_3 + bcoordinates[12] * cell_grad_ref2_4 + bcoordinates[15] * cell_grad_ref2_5 + bcoordinates[18] * cell_grad_ref2_6 + bcoordinates[21] * cell_grad_ref2_7 + bcoordinates[24] * cell_grad_ref2_8 + bcoordinates[27] * cell_grad_ref2_9;
        const s_t J10 = bcoordinates[1] * cell_grad_ref0_0 + bcoordinates[4] * cell_grad_ref0_1 + bcoordinates[7] * cell_grad_ref0_2 + bcoordinates[10] * cell_grad_ref0_3 + bcoordinates[13] * cell_grad_ref0_4 + bcoordinates[16] * cell_grad_ref0_5 + bcoordinates[19] * cell_grad_ref0_6 + bcoordinates[22] * cell_grad_ref0_7 + bcoordinates[25] * cell_grad_ref0_8 + bcoordinates[28] * cell_grad_ref0_9;
        const s_t J11 = bcoordinates[1] * cell_grad_ref1_0 + bcoordinates[4] * cell_grad_ref1_1 + bcoordinates[7] * cell_grad_ref1_2 + bcoordinates[10] * cell_grad_ref1_3 + bcoordinates[13] * cell_grad_ref1_4 + bcoordinates[16] * cell_grad_ref1_5 + bcoordinates[19] * cell_grad_ref1_6 + bcoordinates[22] * cell_grad_ref1_7 + bcoordinates[25] * cell_grad_ref1_8 + bcoordinates[28] * cell_grad_ref1_9;
        const s_t J12 = bcoordinates[1] * cell_grad_ref2_0 + bcoordinates[4] * cell_grad_ref2_1 + bcoordinates[7] * cell_grad_ref2_2 + bcoordinates[10] * cell_grad_ref2_3 + bcoordinates[13] * cell_grad_ref2_4 + bcoordinates[16] * cell_grad_ref2_5 + bcoordinates[19] * cell_grad_ref2_6 + bcoordinates[22] * cell_grad_ref2_7 + bcoordinates[25] * cell_grad_ref2_8 + bcoordinates[28] * cell_grad_ref2_9;
        const s_t J20 = bcoordinates[2] * cell_grad_ref0_0 + bcoordinates[5] * cell_grad_ref0_1 + bcoordinates[8] * cell_grad_ref0_2 + bcoordinates[11] * cell_grad_ref0_3 + bcoordinates[14] * cell_grad_ref0_4 + bcoordinates[17] * cell_grad_ref0_5 + bcoordinates[20] * cell_grad_ref0_6 + bcoordinates[23] * cell_grad_ref0_7 + bcoordinates[26] * cell_grad_ref0_8 + bcoordinates[29] * cell_grad_ref0_9;
        const s_t J21 = bcoordinates[2] * cell_grad_ref1_0 + bcoordinates[5] * cell_grad_ref1_1 + bcoordinates[8] * cell_grad_ref1_2 + bcoordinates[11] * cell_grad_ref1_3 + bcoordinates[14] * cell_grad_ref1_4 + bcoordinates[17] * cell_grad_ref1_5 + bcoordinates[20] * cell_grad_ref1_6 + bcoordinates[23] * cell_grad_ref1_7 + bcoordinates[26] * cell_grad_ref1_8 + bcoordinates[29] * cell_grad_ref1_9;
        const s_t J22 = bcoordinates[2] * cell_grad_ref2_0 + bcoordinates[5] * cell_grad_ref2_1 + bcoordinates[8] * cell_grad_ref2_2 + bcoordinates[11] * cell_grad_ref2_3 + bcoordinates[14] * cell_grad_ref2_4 + bcoordinates[17] * cell_grad_ref2_5 + bcoordinates[20] * cell_grad_ref2_6 + bcoordinates[23] * cell_grad_ref2_7 + bcoordinates[26] * cell_grad_ref2_8 + bcoordinates[29] * cell_grad_ref2_9;
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badjugate_streams, bdeterminant, q);
      }
    }


    vector_mass_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS>(1, bdeterminant, isoparametric_shape, isoparametric_q_weight, bdirection, density, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        {
          atomicAdd(&(out[element_shape[evb] * out_stride]), boutput[stream]);
        }
      }
    }
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_vector_mass_tet10_jacobian_action_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const ptrdiff_t direction_stride,
    const void *const RSTR u0_direction,
    const void *const RSTR u1_direction,
    const void *const RSTR u2_direction,
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
        sfem::codegen::vector_mass_tet10_jacobian_action_i_msoa_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, density, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("vector_mass_tet10_jacobian_action_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::vector_mass_tet10_jacobian_action_i_msoa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, density, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("vector_mass_tet10_jacobian_action_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_jacobian_action_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int cu_vector_mass_tet10_jacobian_action_i_maos(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const void *const RSTR parameters,
    const void *const RSTR direction,
    void *const RSTR output,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        return cu_vector_mass_tet10_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], 3, (const double *)direction + 0, (const double *)direction + 1, (const double *)direction + 2, 3, (double *)output + 0, (double *)output + 1, (double *)output + 2, stream);
    }
    case (int)sizeof(float): {
        return cu_vector_mass_tet10_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], 3, (const float *)direction + 0, (const float *)direction + 1, (const float *)direction + 2, 3, (float *)output + 0, (float *)output + 1, (float *)output + 2, stream);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_jacobian_action_i_maos", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

__host__ __device__ __forceinline__ void vector_mass_tet10_hessian_crs_i_msoa_find_cols(
    const idx_t *const RSTR targets,
    const idx_t *const RSTR row,
    const int lenrow,
    idx_t *const RSTR ks) {
#pragma unroll(10)
  for (int d = 0; d < 10; ++d) {
    ks[d] = 0;
  }
  for (int k = 0; k < lenrow; ++k) {
#pragma unroll(10)
    for (int d = 0; d < 10; ++d) {
      ks[d] += row[k] < targets[d];
    }
  }
}

template <typename s_t>
__host__ __device__ __forceinline__ void vector_mass_tet10_hessian_crs_i_msoa_scatter_crs(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values) {
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int N_COL_STREAMS = 30;
  count_t entries[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    const count_t row_begin = rowptr[ev[i]];
    const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);
    const idx_t *const RSTR cols = &colidx[row_begin];
    vector_mass_tet10_hessian_crs_i_msoa_find_cols(ev, cols, lenrow, ks);
    for (int j = 0; j < NS; ++j) {
      entries[i * NS + j] = row_begin + ks[j];
    }
  }
  for (int bi = 0; bi < NC; ++bi) {
    for (int row_shape = 0; row_shape < NS; ++row_shape) {
      const s_t *const RSTR row = &element_matrix[(bi * NS + row_shape) * N_COL_STREAMS];
      for (int bj = 0; bj < NC; ++bj) {
        for (int col_shape = 0; col_shape < NS; ++col_shape) {
          s_t *const block = &values[entries[row_shape * NS + col_shape] * NC * NC];
          atomicAdd(&(block[bi * NC + bj]), row[bj * NS + col_shape]);
        }
      }
    }
  }
}

template <typename s_t>
__global__ void vector_mass_tet10_hessian_crs_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t density,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  const s_t *const isoparametric_shape = sfem::codegen::ref_tet10_q11<s_t>::shape();
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::ref_tet10_q11<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tet_q11<s_t>::q_weight();

  for (ptrdiff_t element = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; element < nelements; element += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[NS];
    s_t element_matrix[900];
    s_t bcoordinates[ND * NS];
    s_t badjugate_data[ND * ND][NQ];
    s_t bdeterminant[NQ];
    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t node = elements[shape][element];
      const idx_t coordinate_node = elements[shape][element];
      ev[shape] = node;
      for (int d = 0; d < ND; ++d) {
        bcoordinates[shape * ND + d] = s_t(coordinate_components[d][coordinate_node]);
      }
    }

    s_t *badjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    for (int q = 0; q < NQ; ++q) {
      const s_t cell_grad_ref0_0 = isoparametric_grad_ref_x[q * NS + 0];
      const s_t cell_grad_ref0_1 = isoparametric_grad_ref_x[q * NS + 1];
      const s_t cell_grad_ref0_2 = isoparametric_grad_ref_x[q * NS + 2];
      const s_t cell_grad_ref0_3 = isoparametric_grad_ref_x[q * NS + 3];
      const s_t cell_grad_ref0_4 = isoparametric_grad_ref_x[q * NS + 4];
      const s_t cell_grad_ref0_5 = isoparametric_grad_ref_x[q * NS + 5];
      const s_t cell_grad_ref0_6 = isoparametric_grad_ref_x[q * NS + 6];
      const s_t cell_grad_ref0_7 = isoparametric_grad_ref_x[q * NS + 7];
      const s_t cell_grad_ref0_8 = isoparametric_grad_ref_x[q * NS + 8];
      const s_t cell_grad_ref0_9 = isoparametric_grad_ref_x[q * NS + 9];
      const s_t cell_grad_ref1_0 = isoparametric_grad_ref_y[q * NS + 0];
      const s_t cell_grad_ref1_1 = isoparametric_grad_ref_y[q * NS + 1];
      const s_t cell_grad_ref1_2 = isoparametric_grad_ref_y[q * NS + 2];
      const s_t cell_grad_ref1_3 = isoparametric_grad_ref_y[q * NS + 3];
      const s_t cell_grad_ref1_4 = isoparametric_grad_ref_y[q * NS + 4];
      const s_t cell_grad_ref1_5 = isoparametric_grad_ref_y[q * NS + 5];
      const s_t cell_grad_ref1_6 = isoparametric_grad_ref_y[q * NS + 6];
      const s_t cell_grad_ref1_7 = isoparametric_grad_ref_y[q * NS + 7];
      const s_t cell_grad_ref1_8 = isoparametric_grad_ref_y[q * NS + 8];
      const s_t cell_grad_ref1_9 = isoparametric_grad_ref_y[q * NS + 9];
      const s_t cell_grad_ref2_0 = isoparametric_grad_ref_z[q * NS + 0];
      const s_t cell_grad_ref2_1 = isoparametric_grad_ref_z[q * NS + 1];
      const s_t cell_grad_ref2_2 = isoparametric_grad_ref_z[q * NS + 2];
      const s_t cell_grad_ref2_3 = isoparametric_grad_ref_z[q * NS + 3];
      const s_t cell_grad_ref2_4 = isoparametric_grad_ref_z[q * NS + 4];
      const s_t cell_grad_ref2_5 = isoparametric_grad_ref_z[q * NS + 5];
      const s_t cell_grad_ref2_6 = isoparametric_grad_ref_z[q * NS + 6];
      const s_t cell_grad_ref2_7 = isoparametric_grad_ref_z[q * NS + 7];
      const s_t cell_grad_ref2_8 = isoparametric_grad_ref_z[q * NS + 8];
      const s_t cell_grad_ref2_9 = isoparametric_grad_ref_z[q * NS + 9];
      const s_t J00 = bcoordinates[0] * cell_grad_ref0_0 + bcoordinates[3] * cell_grad_ref0_1 + bcoordinates[6] * cell_grad_ref0_2 + bcoordinates[9] * cell_grad_ref0_3 + bcoordinates[12] * cell_grad_ref0_4 + bcoordinates[15] * cell_grad_ref0_5 + bcoordinates[18] * cell_grad_ref0_6 + bcoordinates[21] * cell_grad_ref0_7 + bcoordinates[24] * cell_grad_ref0_8 + bcoordinates[27] * cell_grad_ref0_9;
      const s_t J01 = bcoordinates[0] * cell_grad_ref1_0 + bcoordinates[3] * cell_grad_ref1_1 + bcoordinates[6] * cell_grad_ref1_2 + bcoordinates[9] * cell_grad_ref1_3 + bcoordinates[12] * cell_grad_ref1_4 + bcoordinates[15] * cell_grad_ref1_5 + bcoordinates[18] * cell_grad_ref1_6 + bcoordinates[21] * cell_grad_ref1_7 + bcoordinates[24] * cell_grad_ref1_8 + bcoordinates[27] * cell_grad_ref1_9;
      const s_t J02 = bcoordinates[0] * cell_grad_ref2_0 + bcoordinates[3] * cell_grad_ref2_1 + bcoordinates[6] * cell_grad_ref2_2 + bcoordinates[9] * cell_grad_ref2_3 + bcoordinates[12] * cell_grad_ref2_4 + bcoordinates[15] * cell_grad_ref2_5 + bcoordinates[18] * cell_grad_ref2_6 + bcoordinates[21] * cell_grad_ref2_7 + bcoordinates[24] * cell_grad_ref2_8 + bcoordinates[27] * cell_grad_ref2_9;
      const s_t J10 = bcoordinates[1] * cell_grad_ref0_0 + bcoordinates[4] * cell_grad_ref0_1 + bcoordinates[7] * cell_grad_ref0_2 + bcoordinates[10] * cell_grad_ref0_3 + bcoordinates[13] * cell_grad_ref0_4 + bcoordinates[16] * cell_grad_ref0_5 + bcoordinates[19] * cell_grad_ref0_6 + bcoordinates[22] * cell_grad_ref0_7 + bcoordinates[25] * cell_grad_ref0_8 + bcoordinates[28] * cell_grad_ref0_9;
      const s_t J11 = bcoordinates[1] * cell_grad_ref1_0 + bcoordinates[4] * cell_grad_ref1_1 + bcoordinates[7] * cell_grad_ref1_2 + bcoordinates[10] * cell_grad_ref1_3 + bcoordinates[13] * cell_grad_ref1_4 + bcoordinates[16] * cell_grad_ref1_5 + bcoordinates[19] * cell_grad_ref1_6 + bcoordinates[22] * cell_grad_ref1_7 + bcoordinates[25] * cell_grad_ref1_8 + bcoordinates[28] * cell_grad_ref1_9;
      const s_t J12 = bcoordinates[1] * cell_grad_ref2_0 + bcoordinates[4] * cell_grad_ref2_1 + bcoordinates[7] * cell_grad_ref2_2 + bcoordinates[10] * cell_grad_ref2_3 + bcoordinates[13] * cell_grad_ref2_4 + bcoordinates[16] * cell_grad_ref2_5 + bcoordinates[19] * cell_grad_ref2_6 + bcoordinates[22] * cell_grad_ref2_7 + bcoordinates[25] * cell_grad_ref2_8 + bcoordinates[28] * cell_grad_ref2_9;
      const s_t J20 = bcoordinates[2] * cell_grad_ref0_0 + bcoordinates[5] * cell_grad_ref0_1 + bcoordinates[8] * cell_grad_ref0_2 + bcoordinates[11] * cell_grad_ref0_3 + bcoordinates[14] * cell_grad_ref0_4 + bcoordinates[17] * cell_grad_ref0_5 + bcoordinates[20] * cell_grad_ref0_6 + bcoordinates[23] * cell_grad_ref0_7 + bcoordinates[26] * cell_grad_ref0_8 + bcoordinates[29] * cell_grad_ref0_9;
      const s_t J21 = bcoordinates[2] * cell_grad_ref1_0 + bcoordinates[5] * cell_grad_ref1_1 + bcoordinates[8] * cell_grad_ref1_2 + bcoordinates[11] * cell_grad_ref1_3 + bcoordinates[14] * cell_grad_ref1_4 + bcoordinates[17] * cell_grad_ref1_5 + bcoordinates[20] * cell_grad_ref1_6 + bcoordinates[23] * cell_grad_ref1_7 + bcoordinates[26] * cell_grad_ref1_8 + bcoordinates[29] * cell_grad_ref1_9;
      const s_t J22 = bcoordinates[2] * cell_grad_ref2_0 + bcoordinates[5] * cell_grad_ref2_1 + bcoordinates[8] * cell_grad_ref2_2 + bcoordinates[11] * cell_grad_ref2_3 + bcoordinates[14] * cell_grad_ref2_4 + bcoordinates[17] * cell_grad_ref2_5 + bcoordinates[20] * cell_grad_ref2_6 + bcoordinates[23] * cell_grad_ref2_7 + bcoordinates[26] * cell_grad_ref2_8 + bcoordinates[29] * cell_grad_ref2_9;
      geometry_jacobian_adjugate_and_determinant_3<s_t>(
          J00, J01, J02, J10, J11, J12, J20, J21, J22,
          badjugate_streams, bdeterminant, q);
    }

    vector_mass_d3_simplex_hessian_block<s_t, NQ, NS>(1, bdeterminant, isoparametric_shape, isoparametric_q_weight, density, element_matrix);

    vector_mass_tet10_hessian_crs_i_msoa_scatter_crs(ev, element_matrix, rowptr, colidx, values);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_vector_mass_tet10_hessian_bsr_i_msoa(
    const int scalar_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const real_t density,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      const int block_size = 256;
      const int grid_size = (int)((nelements + block_size - 1) / block_size);
      sfem::codegen::vector_mass_tet10_hessian_crs_i_msoa_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, density, rowptr, colidx, (double *)values);
      return sfem::codegen::launch_status("vector_mass_tet10_hessian_crs_i_msoa_impl");
    }
    case (int)sizeof(float): {
      const int block_size = 256;
      const int grid_size = (int)((nelements + block_size - 1) / block_size);
      sfem::codegen::vector_mass_tet10_hessian_crs_i_msoa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, density, rowptr, colidx, (float *)values);
      return sfem::codegen::launch_status("vector_mass_tet10_hessian_crs_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet10_hessian_bsr_i_msoa", -1, (int)scalar_bytes);
}
